//! The liquid solver on the graphics card (wgpu compute shaders in `gpu_liquid.wgsl`).
//!
//! Same method as the CPU path, step for step, so the two agree closely; the GPU is many times
//! faster, which buys resolution (finer cells, more particles), and resolution is what makes
//! water look and behave like water. Runs repeat exactly on the same GPU (fixed-point sums);
//! a different GPU or the CPU gives very close but not bit-identical numbers.

use crate::grid::{Dims, SOLID};
use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};

const SHADER: &str = include_str!("gpu_liquid.wgsl");
const BINS: u64 = 48;
const ROW: u32 = 65535;
const KERNELS: [&str; 21] = [
    "measure", "pack",
    "clear", "p2g", "normalize", "extrapolate", "copy_back", "enforce", "gravity", "viscous", "viscous_back", "cells", "sor", "gradient", "mark_touch", "g2p", "advect", "clear_bins", "bin", "separate", "copy_pos",
];

/// What one tick asks of the GPU.
pub struct Job<'a> {
    pub d: Dims,
    pub h: f32,
    pub sdt: f32,
    pub subs: usize,
    pub gravity: f32,
    /// viscosity * dt / h², split into `visc_steps` stable steps
    pub visc: f32,
    pub visc_steps: usize,
    pub sweeps: usize,
    pub solid: &'a [u8],
    pub particles: &'a mut Vec<[f32; 6]>,
    pub affine: &'a mut Vec<[f32; 9]>,
    pub pressure: &'a mut Vec<f32>,
    /// Which particle data the caller holds (see `fresh_generation`); unchanged = already on the card.
    pub generation: u64,
}

static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A new number for particle data that changed on the CPU (so the GPU copy is out of date).
pub fn fresh_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

struct Sized {
    d: Dims,
    cap: usize,
    bufs: Vec<wgpu::Buffer>,
    group: wgpu::BindGroup,
    /// kept from tick to tick: the CPU reads results through it
    staging: wgpu::Buffer,
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipes: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    sized: Option<Sized>,
    /// generation and particle count of what is on the card
    resident: (u64, usize),
}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| Gpu::new().ok().map(Mutex::new)).as_ref()
}

/// Is there a graphics card the liquid solver can use?
pub fn available() -> bool {
    gpu().is_some()
}

impl Gpu {
    fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| e.to_string())?;
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 17 {
            return Err("graphics card allows too few storage buffers".into());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("simtara liquid"), required_limits: limits, ..Default::default() })).map_err(|e| e.to_string())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("simtara liquid"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let mut entries = vec![wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }];
        for b in 1..=17 {
            entries.push(wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("simtara liquid"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara liquid"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let mut pipes = std::collections::HashMap::new();
        for k in KERNELS {
            pipes.entry(k).or_insert_with(|| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(k), layout: Some(&pl), module: &module, entry_point: Some(k), compilation_options: Default::default(), cache: None }));
        }
        // two parameter blocks (red and black pressure sweeps), 256 bytes apart
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara liquid params"), size: 512, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Ok(Gpu { device, queue, layout, pipes, params, sized: None, resident: (0, 0) })
    }

    fn ensure(&mut self, d: Dims, particles: usize) {
        let cap = particles.max(1024).next_power_of_two();
        if self.sized.as_ref().is_some_and(|s| s.d == d && s.cap >= particles) {
            return;
        }
        let nc = d.cells() as u64;
        let nf = (crate::grid::len(d.ud()) + crate::grid::len(d.vd()) + crate::grid::len(d.wd())) as u64;
        let p = cap as u64;
        // sizes in bytes, in binding order 1..=17 (acc has one extra slot: the largest divergence)
        let sizes = [p * 16, p * 16, p * 48, nf * 4, nf * 8 + 16, nf * 4, nf * 4, nc * 4, nc * 4, nc * 4, nc * 4, nc * 16, nc * 4, nc * BINS * 4, p * 16, nf * 4, p * 60];
        let bufs: Vec<wgpu::Buffer> = sizes
            .iter()
            .enumerate()
            .map(|(i, s)| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(&format!("simtara liquid {}", i + 1)), size: (*s).max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }))
            .collect();
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: wgpu::BufferSize::new(64) }) }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simtara liquid"), layout: &self.layout, entries: &entries });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara liquid read"), size: p * 60 + nc * 4 + 16, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        self.sized = Some(Sized { d, cap, bufs, group, staging });
        self.resident = (0, 0);
    }
}


fn params_block(d: Dims, np: u32, j: &Job, colour: u32, omega: f32) -> Vec<u8> {
    let nu = crate::grid::len(d.ud()) as u32;
    let nv = crate::grid::len(d.vd()) as u32;
    let nw = crate::grid::len(d.wd()) as u32;
    let u = |x: u32| x.to_le_bytes();
    let f = |x: f32| x.to_le_bytes();
    let mut out = Vec::with_capacity(64);
    for b in [u(d.nx as u32), u(d.ny as u32), u(d.nz as u32), u(np), f(j.sdt), f(j.h), f(j.gravity), f(j.visc), u(nu), u(nv), u(nw), u(d.cells() as u32), f(omega), u(colour), u(0), u(0)] {
        out.extend_from_slice(&b);
    }
    out
}

fn groups(n: usize) -> (u32, u32) {
    let g = (n as u32).div_ceil(64).max(1);
    if g <= ROW { (g, 1) } else { (ROW, g.div_ceil(ROW)) }
}

/// Run one tick of the liquid on the GPU. Particles stay on the card between ticks (uploaded
/// only when the CPU changed them), are stepped `subs` times and read back packed. Returns the
/// largest divergence left. The face velocities stay on the card.
pub fn step(j: Job) -> Result<f32, String> {
    let m = gpu().ok_or("no graphics card for the liquid solver")?;
    let mut g = m.lock().map_err(|_| "liquid solver lock poisoned")?;
    let d = j.d;
    let np = j.particles.len();
    g.ensure(d, np);
    let s = g.sized.as_ref().unwrap();
    let nf = crate::grid::len(d.ud()) + crate::grid::len(d.vd()) + crate::grid::len(d.wd());
    let nc = d.cells();
    // optimal over-relaxation for a grid this size
    let n = d.nx.max(d.ny).max(d.nz) as f32;
    let omega = 2.0 / (1.0 + (std::f32::consts::PI / n).sin());
    let mut pb = params_block(d, np as u32, &j, 0, omega);
    pb.resize(256, 0);
    pb.extend(params_block(d, np as u32, &j, 1, omega));
    g.queue.write_buffer(&g.params, 0, &pb);
    let q = &g.queue;
    let on_card = g.resident == (j.generation, np);
    if !on_card {
        let pos: Vec<[f32; 4]> = j.particles.par_iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let vel: Vec<[f32; 4]> = j.particles.par_iter().map(|p| [p[3], p[4], p[5], 0.0]).collect();
        let aff: Vec<[f32; 12]> = j.affine.par_iter().map(|c| [c[0], c[1], c[2], 0.0, c[3], c[4], c[5], 0.0, c[6], c[7], c[8], 0.0]).collect();
        q.write_buffer(&s.bufs[0], 0, bytemuck::cast_slice(&pos));
        q.write_buffer(&s.bufs[1], 0, bytemuck::cast_slice(&vel));
        q.write_buffer(&s.bufs[2], 0, bytemuck::cast_slice(&aff));
        let solid: Vec<u32> = j.solid.par_iter().map(|f| (*f == SOLID) as u32).collect();
        q.write_buffer(&s.bufs[8], 0, bytemuck::cast_slice(&solid));
        if j.pressure.len() == nc {
            q.write_buffer(&s.bufs[10], 0, bytemuck::cast_slice(j.pressure));
        } else {
            q.write_buffer(&s.bufs[10], 0, &vec![0u8; nc * 4]);
        }
    }

    // a second bind group whose params point at the "black" block
    let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &g.params, offset: 256, size: wgpu::BufferSize::new(64) }) }];
    for (i, b) in s.bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
    }
    let black = g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simtara liquid black"), layout: &g.layout, entries: &entries });

    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("simtara liquid") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("simtara liquid"), timestamp_writes: None });
        let mut run = |name: &str, count: usize, group: &wgpu::BindGroup| {
            pass.set_pipeline(&g.pipes[name]);
            pass.set_bind_group(0, group, &[]);
            let (x, y) = groups(count);
            pass.dispatch_workgroups(x, y, 1);
        };
        let red = &s.group;
        let all = nf.max(nc);
        for _ in 0..j.subs {
            run("clear", all, red);
            run("p2g", np, red);
            run("normalize", nf, red);
            for _ in 0..2 {
                run("extrapolate", nf, red);
                run("copy_back", nf, red);
            }
            run("enforce", nf, red);
            run("gravity", nf, red);
            for _ in 0..j.visc_steps {
                run("viscous", nf, red);
                run("viscous_back", nf, red);
            }
            run("cells", nc, red);
            for _ in 0..j.sweeps {
                run("sor", nc, red);
                run("sor", nc, &black);
            }
            run("gradient", nf, red);
            run("measure", nc, red);
            run("mark_touch", nf, red);
            for _ in 0..3 {
                run("extrapolate", nf, red);
                run("copy_back", nf, red);
            }
            run("enforce", nf, red);
            run("g2p", np, red);
            run("advect", np, red);
            run("clear_bins", nc, red);
            run("bin", np, red);
            run("separate", np, red);
            run("copy_pos", np, red);
        }
        run("pack", np, red);
    }
    let timing = std::env::var_os("SIMTARA_GPU_TIMING").is_some();
    let t_up = std::time::Instant::now();
    g.queue.submit([enc.finish()]);
    if timing {
        let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
        eprintln!("gpu compute {:.1} ms ({} subs)", t_up.elapsed().as_secs_f64() * 1000.0, j.subs);
    }
    let t_rd = std::time::Instant::now();

    // one copy of only what the CPU needs (packed particles, pressure, one number) into a
    // staging buffer that lives as long as the solver, then one wait
    let (pbytes, cbytes) = (np as u64 * 60, nc as u64 * 4);
    let mut enc = g.device.create_command_encoder(&Default::default());
    if pbytes > 0 {
        enc.copy_buffer_to_buffer(&s.bufs[16], 0, &s.staging, 0, pbytes);
    }
    enc.copy_buffer_to_buffer(&s.bufs[10], 0, &s.staging, pbytes, cbytes);
    enc.copy_buffer_to_buffer(&s.bufs[4], nf as u64 * 8, &s.staging, pbytes + cbytes, 4);
    g.queue.submit([enc.finish()]);
    let slice = s.staging.slice(..pbytes + cbytes + 4);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
    let divergence;
    {
        let view = slice.get_mapped_range().expect("mapped");
        let all: &[f32] = bytemuck::cast_slice(&view);
        let (packed, rest) = all.split_at(np * 15);
        let (pres, stat) = rest.split_at(nc);
        j.particles.par_iter_mut().zip(j.affine.par_iter_mut()).zip(packed.par_chunks_exact(15)).for_each(|((p, c), x)| {
            *p = [x[0], x[1], x[2], x[3], x[4], x[5]];
            *c = [x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]];
        });
        j.pressure.clear();
        j.pressure.extend_from_slice(pres);
        divergence = stat[0];
    }
    s.staging.unmap();
    if timing {
        eprintln!("gpu read back {:.1} ms", t_rd.elapsed().as_secs_f64() * 1000.0);
    }
    // what the caller now holds is exactly what is on the card
    let mut g = g;
    g.resident = (j.generation, np);
    Ok(divergence)
}