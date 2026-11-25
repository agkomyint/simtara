//! GPU compute backend for SimTara (wgpu + WGSL).
//!
//! Adding `GpuModule` to a simulation registers `Backend::Gpu` and installs a GPU
//! implementation of the particle force rule. The CPU implementation stays in place and
//! takes over automatically if no GPU is found or the user switches back to `Backend::Cpu`.
//! Neither the core nor the particle module had to know about wgpu.
//!
//! Per tick: the CPU builds the spatial grid (O(n)), uploads state, the GPU computes every
//! particle's acceleration (the O(n * neighbours) part), and the result is read back.

use bytemuck::{Pod, Zeroable};
use simtara_core::{Backend, Builder, Context, Module, Stage, System};
use simtara_particles::{
    apply_acceleration, build_grid, params_valid, Boundary, ParticleParams, Particles, FORCE_SCALE,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("forces.wgsl");

/// Shared handle to the GPU device. Create once, reuse across universe resets.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend_api: String,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simtara"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device request failed: {e}"))?;
        Ok(GpuContext { device, queue, adapter_name: info.name, backend_api: format!("{:?}", info.backend) })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    n: u32,
    gw: u32,
    gh: u32,
    kinds: u32,
    wrap: u32,
    width: f32,
    height: f32,
    rmax: f32,
    beta: f32,
    gscale: f32,
    coulomb: f32,
    _pad: f32,
}

struct Buffers {
    capacity: usize,
    cells_len: usize,
    params: wgpu::Buffer,
    pos: wgpu::Buffer,
    info: wgpu::Buffer,
    cells: wgpu::Buffer,
    start: wgpu::Buffer,
    order: wgpu::Buffer,
    attraction: wgpu::Buffer,
    acc: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

struct Kernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bufs: Option<Buffers>,
}

impl Kernel {
    fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_forces"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_forces"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Kernel { ctx, pipeline, bufs: None }
    }

    fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn ensure(&mut self, n: usize, cells_len: usize, attraction_len: usize) {
        if let Some(b) = &self.bufs {
            if b.capacity >= n && b.cells_len == cells_len {
                return;
            }
        }
        let cap = n.next_power_of_two().max(1024);
        let dev = &self.ctx.device;
        let params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&GpuParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos = self.storage("pos", cap as u64 * 8);
        let info = self.storage("info", cap as u64 * 16);
        let cells = self.storage("cells", cap as u64 * 4);
        let start = self.storage("start", (cells_len as u64 + 1) * 4);
        let order = self.storage("order", cap as u64 * 4);
        let attraction = self.storage("attraction", attraction_len as u64 * 4);
        let acc = self.storage("acc", cap as u64 * 8);
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: cap as u64 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: info.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: order.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: attraction.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: acc.as_entire_binding() },
            ],
        });
        self.bufs = Some(Buffers {
            capacity: cap,
            cells_len,
            params,
            pos,
            info,
            cells,
            start,
            order,
            attraction,
            acc,
            staging,
            bind,
        });
    }

    /// Per-particle acceleration, computed on the GPU. Same contract as `accelerations_cpu`.
    fn accelerations(&mut self, p: &Particles, prm: &ParticleParams) -> Vec<[f32; 2]> {
        let n = p.len();
        let grid = build_grid(p, prm);
        let ncells = (grid.gw * grid.gh) as usize;
        self.ensure(n, ncells, prm.attraction.len());
        let b = self.bufs.as_ref().unwrap();
        let q = &self.ctx.queue;

        let rmax = prm.rmax.max(1e-3);
        let params = GpuParams {
            n: n as u32,
            gw: grid.gw,
            gh: grid.gh,
            kinds: prm.kinds as u32,
            wrap: (prm.boundary == Boundary::Wrap) as u32,
            width: prm.width,
            height: prm.height,
            rmax,
            beta: prm.repel_core.clamp(0.01, 0.95),
            gscale: prm.gravity * FORCE_SCALE * rmax,
            coulomb: prm.coulomb,
            _pad: 0.0,
        };
        let pos: Vec<[f32; 2]> = (0..n).map(|i| [p.x[i], p.y[i]]).collect();
        let info: Vec<[f32; 4]> = (0..n).map(|i| [p.kind[i] as f32, p.charge[i], p.mass[i], 0.0]).collect();
        q.write_buffer(&b.params, 0, bytemuck::bytes_of(&params));
        q.write_buffer(&b.pos, 0, bytemuck::cast_slice(&pos));
        q.write_buffer(&b.info, 0, bytemuck::cast_slice(&info));
        q.write_buffer(&b.cells, 0, bytemuck::cast_slice(&grid.cells));
        q.write_buffer(&b.start, 0, bytemuck::cast_slice(&grid.start));
        q.write_buffer(&b.order, 0, bytemuck::cast_slice(&grid.order));
        q.write_buffer(&b.attraction, 0, bytemuck::cast_slice(&prm.attraction));

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &b.bind, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        let bytes = n as u64 * 8;
        enc.copy_buffer_to_buffer(&b.acc, 0, &b.staging, 0, bytes);
        q.submit(Some(enc.finish()));

        let slice = b.staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("GPU readback channel").expect("GPU readback failed");
        let out: Vec<[f32; 2]> = bytemuck::cast_slice(&slice.get_mapped_range().expect("GPU mapped range")).to_vec();
        b.staging.unmap();
        out
    }
}

struct GpuForceSystem {
    kernel: Kernel,
}

impl System for GpuForceSystem {
    fn name(&self) -> &str {
        "particles.forces.gpu"
    }
    fn run(&mut self, ctx: &mut Context) {
        if ctx.compute.active() != Backend::Gpu {
            return; // the CPU implementation of this rule is handling it
        }
        let Some(mut p) = ctx.world.take::<Particles>() else { return };
        if let Some(prm) = ctx.world.get::<ParticleParams>() {
            if params_valid(&p, prm) {
                let acc = self.kernel.accelerations(&p, prm);
                apply_acceleration(&mut p, &acc, ctx.dt as f32);
            }
        }
        ctx.world.put(p);
    }
}

/// Plug-in that adds the GPU backend. Falls back to CPU silently if no GPU is available
/// (check `available()` to find out).
pub struct GpuModule {
    ctx: Option<Arc<GpuContext>>,
    pub error: Option<String>,
}

impl GpuModule {
    pub fn new() -> Self {
        match GpuContext::new() {
            Ok(c) => GpuModule { ctx: Some(Arc::new(c)), error: None },
            Err(e) => GpuModule { ctx: None, error: Some(e) },
        }
    }
    pub fn available(&self) -> bool {
        self.ctx.is_some()
    }
    /// e.g. "NVIDIA GeForce RTX 3050 Ti Laptop GPU (Vulkan)"
    pub fn description(&self) -> String {
        match &self.ctx {
            Some(c) => format!("{} ({})", c.adapter_name, c.backend_api),
            None => format!("unavailable: {}", self.error.as_deref().unwrap_or("unknown")),
        }
    }
}

impl Default for GpuModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GpuModule {
    fn name(&self) -> &str {
        "gpu"
    }
    fn build(&self, b: &mut Builder) {
        let Some(ctx) = &self.ctx else { return };
        b.compute.register(Backend::Gpu);
        // PreUpdate puts the GPU force pass ahead of the CPU one and of integrate; only one of the
        // two does work on a given tick (both gate on `Compute::active`).
        b.add_system(Stage::PreUpdate, GpuForceSystem { kernel: Kernel::new(ctx.clone()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simtara_core::Config;
    use simtara_particles::{accelerations_cpu, simulation, ParticleConfig};

    fn gpu() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_reference() {
        let Some(ctx) = gpu() else { return };
        for b in [Boundary::Bounce, Boundary::Wrap] {
            let mut pc = ParticleConfig { count: 3000, ..Default::default() };
            pc.params.boundary = b;
            pc.params.coulomb = 0.3;
            let mut sim = simulation(Config::default(), &pc);
            sim.advance(30); // get away from the uniform start
            let p = sim.world().get::<Particles>().unwrap();
            let prm = sim.world().get::<ParticleParams>().unwrap();
            let cpu = accelerations_cpu(p, prm);
            let gpu = Kernel::new(ctx.clone()).accelerations(p, prm);
            assert_eq!(cpu.len(), gpu.len());
            let scale = cpu.iter().flatten().fold(1.0f32, |m, v| m.max(v.abs()));
            let worst = cpu.iter().zip(&gpu).flat_map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()]).fold(0.0f32, f32::max);
            assert!(worst / scale < 1e-3, "{b:?}: worst error {worst} vs scale {scale}");
        }
    }

    #[test]
    fn module_switches_backend_and_falls_back() {
        let module = GpuModule::new();
        let has_gpu = module.available();
        let pc = ParticleConfig { count: 2000, ..Default::default() };
        let mut sim = simulation(Config::default(), &pc);
        sim.add_module(module);
        sim.compute_mut().set_requested(Backend::Gpu);
        sim.advance(20); // must run on GPU if present, CPU otherwise — either way, no panic
        let p = sim.world().get::<Particles>().unwrap();
        assert!(p.x.iter().all(|x| x.is_finite()));
        assert_eq!(sim.compute_mut().is_available(Backend::Gpu), has_gpu);
    }
}
