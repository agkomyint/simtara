//! The gas solver on the graphics card (wgpu compute shaders in `gpu_smoke.wgsl`).
//!
//! The same steps as `Fluid::smoke_substep`, kernel for kernel: MacCormack advection, gas
//! leaks and amounts kept exact, Smagorinsky turbulence, diffusion, buoyancy from the ideal gas
//! law, vorticity confinement, viscosity, and a pressure solve weighted by 1 / rho (red-black
//! over-relaxation instead of the CPU's conjugate gradients, so the two agree closely but not
//! bit for bit). The fluid's state is read back after every tick, so saving, metrics and the
//! CPU path always see the same numbers; it is uploaded again only when the CPU changed it.
//! Every kernel only gathers and sums go through a fixed tree, so runs repeat exactly.

use crate::grid::{Dims, SOLID};
use crate::{gas, Fluid, Mac, Resolved, Source, AMBIENT_K};
use std::sync::{Mutex, OnceLock};

const SHADER: &str = include_str!("gpu_smoke.wgsl");
const KERNELS: [&str; 29] = [
    "save_vel", "save_scal", "adv_vel_a", "adv_vel_b", "adv_scal_a", "adv_scal_b", "sum_a", "sum_b", "rescale", "clear_flow", "src_gas", "src_heat", "src_vel", "nut", "diffuse_scal", "copy_scal", "cool", "buoy", "vort", "vforce", "vapply", "diffuse_vel", "copy_vel", "enforce", "div_target", "sub_mean", "prep", "sor", "gradient",
];
/// Parameter blocks (one per value of `sel`), 256 bytes apart.
const BLOCKS: u64 = 48;
/// At most this many gases on the card.
pub const MAX_GASES: usize = 8;
const ROW: u32 = 65535;

struct Sized {
    d: Dims,
    ng: usize,
    src_cap: usize,
    bufs: Vec<wgpu::Buffer>,
    group: wgpu::BindGroup,
    staging: wgpu::Buffer,
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipes: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    measure: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    sized: Option<Sized>,
    /// generation of the fluid state on the card
    resident: u64,
}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| Gpu::new().ok().map(Mutex::new)).as_ref()
}

/// Is there a graphics card the gas solver can use?
pub fn available() -> bool {
    gpu().is_some()
}

impl Gpu {
    fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| e.to_string())?;
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 12 {
            return Err("graphics card allows too few storage buffers".into());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("simtara gas"), required_limits: limits, ..Default::default() })).map_err(|e| e.to_string())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("simtara gas"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let mut entries = vec![wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(256) }, count: None }];
        for b in 1..=12 {
            entries.push(wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("simtara gas"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara gas"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let make = |k: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(k), layout: Some(&pl), module: &module, entry_point: Some(k), compilation_options: Default::default(), cache: None });
        let mut pipes = std::collections::HashMap::new();
        for k in KERNELS {
            pipes.insert(k, make(k));
        }
        let measure = make("measure");
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara gas params"), size: BLOCKS * 256, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Ok(Gpu { device, queue, layout, pipes, measure, params, sized: None, resident: 0 })
    }

    fn ensure(&mut self, d: Dims, ng: usize, nsrc: usize) {
        if self.sized.as_ref().is_some_and(|s| s.d == d && s.ng == ng && s.src_cap >= nsrc) {
            return;
        }
        let nc = d.cells() as u64;
        let nf = (crate::grid::len(d.ud()) + crate::grid::len(d.vd()) + crate::grid::len(d.wd())) as u64;
        let ns = nc * (2 + ng as u64);
        let nwg = nc.div_ceil(256);
        let src_cap = nsrc.max(64).next_power_of_two();
        let sizes = [nf * 4, nf * 4, nf * 4, ns * 4, ns * 4, ns * 4, nc * 4, nc * 24, nc * 32, src_cap as u64 * 48, (nwg + 48) * 4, 16];
        let bufs: Vec<wgpu::Buffer> = sizes
            .iter()
            .enumerate()
            .map(|(i, s)| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(&format!("simtara gas {}", i + 1)), size: (*s).max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }))
            .collect();
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: wgpu::BufferSize::new(256) }) }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simtara gas"), layout: &self.layout, entries: &entries });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara gas read"), size: (nf + ns + nc + 4) * 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        self.sized = Some(Sized { d, ng, src_cap, bufs, group, staging });
        self.resident = 0;
    }
}

fn groups(n: usize) -> (u32, u32) {
    let g = (n as u32).div_ceil(64).max(1);
    if g <= ROW { (g, 1) } else { (ROW, g.div_ceil(ROW)) }
}

/// One tick of the gas on the GPU: `subs` steps of `sdt` seconds.
pub(crate) fn step(f: &mut Fluid, r: &Resolved, sources: &[Source], subs: usize, sdt: f32) -> Result<(), String> {
    let ng = f.gases.len();
    if ng > MAX_GASES || f.fractions.len() != ng {
        return Err(format!("at most {MAX_GASES} gases on the graphics card"));
    }
    let m = gpu().ok_or("no graphics card for the gas solver")?;
    let mut g = m.lock().map_err(|_| "gas solver lock poisoned")?;
    let d = f.dims();
    let h = f.cell;
    let nc = d.cells();
    let (nu, nv, nw) = (crate::grid::len(d.ud()), crate::grid::len(d.vd()), crate::grid::len(d.wd()));
    let nf = nu + nv + nw;
    let ns = nc * (2 + ng);
    let nwg = nc.div_ceil(256);
    let records: Vec<[f32; 12]> = sources
        .iter()
        .flat_map(|s| {
            s.cells.iter().map(move |&c| {
                let v = s.vel.unwrap_or([0.0; 3]);
                [f32::from_bits(c), s.smoke, s.heat, s.gas.map_or(-1.0, |x| x as f32), s.amount, s.flow, v[0], v[1], v[2], if s.vel.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0]
            })
        })
        .collect();
    g.ensure(d, ng, records.len());
    let on_card = f.gpu_generation != 0 && g.resident == f.gpu_generation;
    let s = g.sized.as_ref().unwrap();
    let q = &g.queue;
    if !on_card {
        let mac = f.mac.get_or_insert_with(|| Mac::new(d));
        let mut v: Vec<f32> = Vec::with_capacity(nf);
        v.extend_from_slice(&mac.u);
        v.extend_from_slice(&mac.v);
        v.extend_from_slice(&mac.w);
        q.write_buffer(&s.bufs[0], 0, bytemuck::cast_slice(&v));
        let mut sc: Vec<f32> = Vec::with_capacity(ns);
        sc.extend_from_slice(&f.smoke);
        sc.extend_from_slice(&f.temp);
        for x in &f.fractions {
            sc.extend_from_slice(x);
        }
        q.write_buffer(&s.bufs[3], 0, bytemuck::cast_slice(&sc));
        let solid: Vec<u32> = f.solid.iter().map(|x| (*x == SOLID) as u32).collect();
        q.write_buffer(&s.bufs[6], 0, bytemuck::cast_slice(&solid));
        let mut cells = vec![0.0f32; nc * 6];
        if f.pressure.len() == nc {
            cells[5 * nc..].copy_from_slice(&f.pressure);
        }
        q.write_buffer(&s.bufs[7], 0, bytemuck::cast_slice(&cells));
    }
    if !records.is_empty() {
        q.write_buffer(&s.bufs[9], 0, bytemuck::cast_slice(&records));
    }
    q.write_buffer(&s.bufs[11], 0, &[0u8; 16]);

    // diffusion in small enough steps to stay stable, sized from the last tick's turbulence
    let max_d = f.gases.iter().map(|k| gas::GASES[*k].diffusion).fold(0.0f32, f32::max);
    let k0 = gas::ALPHA_AIR.max(max_d).max(r.viscosity);
    let nut_est = f.eddy_max * 1.25;
    let k_max = (r.viscosity + nut_est).max(k0 + 2.0 * nut_est);
    let n_diff = ((k_max * sdt / (h * h) / 0.15).ceil() as usize).clamp(1, 400);
    let kfac = sdt / n_diff as f32 / (h * h);
    let nu_cap = ((0.16 / kfac - k0) / 2.0).max(0.0);
    let fluid_cells = f.solid.iter().filter(|x| **x != SOLID).count().max(1) as f32;
    let closed = !f.open_top && !f.open_side;
    let flags = (f.open_top as u32) | ((f.open_side as u32) << 1) | ((r.wind.is_some() as u32) << 2) | ((f.no_slip as u32) << 3) | ((closed as u32) << 4);
    let n = d.nx.max(d.ny).max(d.nz) as f32;
    let omega = 2.0 / (1.0 + (std::f32::consts::PI / n).sin());
    let rho0 = gas::rho_air(AMBIENT_K);
    let mut dif = [0.0f32; 8];
    let mut mol = [0.0f32; 8];
    for (i, k) in f.gases.iter().enumerate() {
        dif[i] = gas::GASES[*k].diffusion;
        mol[i] = gas::GASES[*k].molar_mass;
    }
    let mut blocks = vec![0u8; (BLOCKS * 256) as usize];
    for sel in 0..BLOCKS as u32 {
        let mut b: Vec<u8> = Vec::with_capacity(256);
        for x in [d.nx as u32, d.ny as u32, d.nz as u32, nc as u32, nu as u32, nv as u32, nw as u32, ng as u32, sel, records.len() as u32, flags, nwg as u32] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in [sdt / h, sdt, h, r.gravity, r.buoyancy, r.weight, r.vorticity, (-r.cooling * sdt).exp(), (-r.fade * sdt).exp(), r.viscosity, r.turbulence, r.expansion, omega, r.wind.unwrap_or(0.0), f.lid, rho0, AMBIENT_K, nu_cap, fluid_cells, kfac] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in dif.iter().chain(mol.iter()) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let at = sel as usize * 256;
        blocks[at..at + b.len()].copy_from_slice(&b);
    }
    q.write_buffer(&g.params, 0, &blocks);

    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("simtara gas") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("simtara gas"), timestamp_writes: None });
        let group = &s.group;
        let pipes = &g.pipes;
        let run = |pass: &mut wgpu::ComputePass, name: &str, count: usize, sel: u32| {
            pass.set_pipeline(&pipes[name]);
            pass.set_bind_group(0, group, &[sel * 256]);
            let (x, y) = groups(count);
            pass.dispatch_workgroups(x, y, 1);
        };
        // `sum_a` works in groups of 256, `sum_b` and the emitter loops are one group
        let sum = |pass: &mut wgpu::ComputePass, sel: u32| {
            pass.set_pipeline(&pipes["sum_a"]);
            pass.set_bind_group(0, group, &[sel * 256]);
            pass.dispatch_workgroups(nwg as u32, 1, 1);
            pass.set_pipeline(&pipes["sum_b"]);
            pass.dispatch_workgroups(1, 1, 1);
        };
        let one = |pass: &mut wgpu::ComputePass, name: &str| {
            pass.set_pipeline(&pipes[name]);
            pass.set_bind_group(0, group, &[0]);
            pass.dispatch_workgroups(1, 1, 1);
        };
        let sweeps = r.iterations * 3;
        for _ in 0..subs {
            run(&mut pass, "save_vel", nf, 0);
            run(&mut pass, "save_scal", ns, 0);
            for gg in 0..ng as u32 {
                sum(&mut pass, gg);
            }
            run(&mut pass, "adv_vel_a", nf, 0);
            run(&mut pass, "adv_vel_b", nf, 0);
            run(&mut pass, "adv_scal_a", ns, 0);
            run(&mut pass, "adv_scal_b", ns, 0);
            for gg in 0..ng as u32 {
                sum(&mut pass, 16 + gg);
            }
            if ng > 0 {
                run(&mut pass, "rescale", nc, 0);
            }
            run(&mut pass, "clear_flow", nc, 0);
            if !records.is_empty() {
                one(&mut pass, "src_gas");
                one(&mut pass, "src_heat");
            }
            run(&mut pass, "nut", nc, 0);
            for _ in 0..n_diff {
                for field in 0..(2 + ng) as u32 {
                    run(&mut pass, "diffuse_scal", nc, field);
                    run(&mut pass, "copy_scal", nc, field);
                }
            }
            run(&mut pass, "cool", nc, 0);
            run(&mut pass, "buoy", nf, 0);
            if r.vorticity > 0.0 {
                run(&mut pass, "vort", nc, 0);
                run(&mut pass, "vforce", nc, 0);
                run(&mut pass, "vapply", nf, 0);
            }
            for _ in 0..n_diff {
                run(&mut pass, "diffuse_vel", nf, 0);
                run(&mut pass, "copy_vel", nf, 0);
            }
            if !records.is_empty() {
                one(&mut pass, "src_vel");
            }
            run(&mut pass, "enforce", nf, 0);
            run(&mut pass, "div_target", nc, 0);
            if closed {
                sum(&mut pass, 32);
                run(&mut pass, "sub_mean", nc, 32);
            }
            run(&mut pass, "prep", nc, 0);
            for _ in 0..sweeps {
                run(&mut pass, "sor", nc, 0);
                run(&mut pass, "sor", nc, 1);
            }
            run(&mut pass, "gradient", nf, 0);
        }        pass.set_pipeline(&g.measure);
        pass.set_bind_group(0, group, &[0]);
        let (x, y) = groups(nf.max(nc));
        pass.dispatch_workgroups(x, y, 1);
    }
    let timing = std::env::var_os("SIMTARA_GPU_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let (bv, bs, bp) = (nf as u64 * 4, ns as u64 * 4, nc as u64 * 4);
    enc.copy_buffer_to_buffer(&s.bufs[0], 0, &s.staging, 0, bv);
    enc.copy_buffer_to_buffer(&s.bufs[3], 0, &s.staging, bv, bs);
    enc.copy_buffer_to_buffer(&s.bufs[7], 5 * bp, &s.staging, bv + bs, bp);
    enc.copy_buffer_to_buffer(&s.bufs[11], 0, &s.staging, bv + bs + bp, 16);
    g.queue.submit([enc.finish()]);
    let slice = s.staging.slice(..bv + bs + bp + 16);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
    {
        let view = slice.get_mapped_range().expect("mapped");
        let all: &[f32] = bytemuck::cast_slice(&view);
        let (vel, rest) = all.split_at(nf);
        let (sc, rest) = rest.split_at(ns);
        let (pres, stats) = rest.split_at(nc);
        let mac = f.mac.get_or_insert_with(|| Mac::new(d));
        mac.u.copy_from_slice(&vel[..nu]);
        mac.v.copy_from_slice(&vel[nu..nu + nv]);
        mac.w.copy_from_slice(&vel[nu + nv..]);
        f.smoke.copy_from_slice(&sc[..nc]);
        f.temp.copy_from_slice(&sc[nc..2 * nc]);
        for (gi, x) in f.fractions.iter_mut().enumerate() {
            x.copy_from_slice(&sc[(2 + gi) * nc..(3 + gi) * nc]);
        }
        f.pressure.clear();
        f.pressure.extend_from_slice(pres);
        f.divergence = stats[0];
        f.eddy_max = stats[2];
    }
    s.staging.unmap();
    if timing {
        eprintln!("gpu gas {:.1} ms ({} subs, {} diffusion steps)", t0.elapsed().as_secs_f64() * 1000.0, subs, n_diff);
    }
    // what the fluid now holds is exactly what is on the card
    let gen = crate::gpu::fresh_generation();
    f.gpu_generation = gen;
    let mut g = g;
    g.resident = gen;
    Ok(())
}

