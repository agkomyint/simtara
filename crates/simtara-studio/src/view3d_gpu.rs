//! GPU drawing for the 3D view (wgpu), the same picture as the CPU painter in `view3d.rs`.
//!
//! The scene is drawn into an offscreen texture with a depth buffer and 4x multisampling, then
//! shown in egui as an image. Spheres are camera-facing quads shaded per pixel as real balls
//! (with correct depth); cubes, pyramids and `.obj` models are instanced meshes. One draw call per
//! shape, so hundreds of thousands of things stay fast. Camera maths is the same as the CPU path:
//! `view3d` fills in [`Uniforms`] from its `View`.

use crate::view3d::{Item, Mesh, Volume};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use std::collections::HashMap;

const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;
/// Bytes per instance: pos (3 x f32), size (f32), colour (4 x u8), heading (f32).
const INST: u64 = 24;
/// Bytes per mesh vertex: position (3 x f32), face normal (3 x f32).
const VERT: u64 = 24;

/// Everything the shaders need about the camera, light and ground. Filled in by `view3d`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Uniforms {
    pub eye: [f32; 3],
    pub r: [f32; 3],
    pub u: [f32; 3],
    pub f: [f32; 3],
    /// Screen-space scale: focal length divided by half the width / half the height.
    pub sx: f32,
    pub sy: f32,
    pub near: f32,
    pub far: f32,
    pub ortho: bool,
    pub ambient: f32,
    pub strength: f32,
    /// World radius of a thing of size 1.
    pub unit: f32,
    pub light: [f32; 3],
    /// Ground rectangle (x0, y0, x1, y1) at height `ground_z`.
    pub ground: [f32; 4],
    pub ground_z: f32,
    pub ground_on: bool,
    pub grid_on: bool,
    /// Picture size (filled in by the renderer, in pixels).
    pub viewport: [f32; 2],
    /// Smoke box: size in metres and cells, when there is one.
    pub volume: Option<([f32; 3], [u32; 3])>,
    /// Radius of a water particle when water is drawn as a surface, world units.
    pub water_radius: f32,
    /// Draw the fluid's box as a tiled pool with a sky behind (liquid scenes).
    pub pool: bool,
    /// How far between the last two steps this frame is (0 = the older, 1 = the newest).
    pub blend: f32,
    /// Objects in the liquid, drawn exactly: (ball?, centre, half sizes), world units. Up to 4.
    pub objects: [(bool, [f32; 3], [f32; 3]); 4],
    pub n_objects: u32,
    /// Which object is being steered (drawn with a glow).
    pub steered: Option<usize>,
}

impl Uniforms {
    fn bytes(&self) -> Vec<u8> {
        let v4 = |a: [f32; 3], w: f32| [a[0], a[1], a[2], w];
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let (vs, vd) = self.volume.unwrap_or(([1.0; 3], [1; 3]));
        let cell = (0..3).map(|a| vs[a] / vd[a] as f32).fold(f32::MAX, f32::min);
        let ob = |i: usize| {
            let (ball, c, h) = self.objects[i];
            [[c[0], c[1], c[2], if ball { 0.0 } else { 1.0 }], [h[0], h[1], h[2], if Some(i) == self.steered { 1.0 } else { 0.0 }]]
        };
        let all: [[f32; 4]; 21] = [
            v4(self.eye, 0.0),
            v4(self.r, 0.0),
            v4(self.u, 0.0),
            v4(self.f, 0.0),
            [self.sx, self.sy, self.near, self.far],
            [b(self.ortho), self.ambient, self.strength, self.unit],
            v4(self.light, 0.0),
            self.ground,
            [self.ground_z, b(self.ground_on), b(self.grid_on), 10.0],
            [self.viewport[0], self.viewport[1], crate::view3d::SMOKE_PER_METRE, crate::view3d::GLOW_PER_METRE],
            [vs[0], vs[1], vs[2], cell],
            [vd[0] as f32, vd[1] as f32, vd[2] as f32, b(self.volume.is_some())],
            [self.water_radius, b(self.pool), self.blend, self.n_objects as f32],
            ob(0)[0], ob(0)[1], ob(1)[0], ob(1)[1], ob(2)[0], ob(2)[1], ob(3)[0], ob(3)[1],
        ];
        all.iter().flatten().flat_map(|x| x.to_le_bytes()).collect()
    }
}

/// Shapes drawn as shaded balls rather than meshes (same rule as the CPU painter).
pub fn is_ball(shape: &str, models: &HashMap<String, Mesh>) -> bool {
    matches!(shape, "" | "sphere" | "ball" | "point" | "dot") || !models.contains_key(shape)
}

fn push_instance(out: &mut Vec<u8>, it: &Item) {
    for x in [it.pos[0], it.pos[1], it.pos[2], it.size] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&it.color.to_array());
    out.extend_from_slice(&it.heading.to_le_bytes());
}

/// A vertex buffer that grows when needed.
#[derive(Default)]
struct GrowBuf {
    buf: Option<wgpu::Buffer>,
    cap: u64,
}

impl GrowBuf {
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let need = (data.len() as u64).max(INST);
        if self.buf.is_none() || self.cap < need {
            let cap = need.next_power_of_two();
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara 3d instances"), size: cap, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            self.cap = cap;
        }
        if !data.is_empty() {
            queue.write_buffer(self.buf.as_ref().unwrap(), 0, data);
        }
    }
}

struct GpuMesh {
    /// Cheap fingerprint, so a reloaded model with the same name is uploaded again.
    key: (usize, usize, u32),
    verts: wgpu::Buffer,
    count: u32,
    inst: GrowBuf,
}

fn fingerprint(m: &Mesh) -> (usize, usize, u32) {
    (m.v.len(), m.t.len(), m.v.first().map_or(0, |p| p[0].to_bits() ^ p[1].to_bits().rotate_left(11) ^ p[2].to_bits().rotate_left(22)))
}

/// Flat-shaded triangles: every corner carries its face normal, like the CPU painter.
fn mesh_bytes(m: &Mesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.t.len() * 3 * VERT as usize);
    for t in &m.t {
        let (a, b, c) = (m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]);
        let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        let n = [n[0] / l, n[1] / l, n[2] / l];
        for p in [a, b, c] {
            for x in p.iter().chain(n.iter()) {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    out
}

struct Target {
    size: [u32; 2],
    /// the scene drawn without water, which the water looks through (refraction)
    scene: wgpu::TextureView,
    key: u64,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
}

/// The GPU renderer. Needs only a device and queue, so it can be tested without a window.
pub struct Gpu3d {
    device: wgpu::Device,
    queue: wgpu::Queue,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    ground: wgpu::RenderPipeline,
    balls: wgpu::RenderPipeline,
    meshes_pipe: wgpu::RenderPipeline,
    vol_pipe: wgpu::RenderPipeline,
    vol_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The smoke box on the card: texture, its bind group, and its size in cells.
    vol: Option<(wgpu::Texture, wgpu::BindGroup, [u32; 3])>,
    sky_pipe: wgpu::RenderPipeline,
    solids_pipe: wgpu::RenderPipeline,
    pool_pipe: wgpu::RenderPipeline,
    /// caustics on the pool floor (a 2D texture; same layout as the smoke volume's group)
    caustics: (wgpu::BindGroup, [u32; 2]),
    /// generation of the fluid pictures last uploaded
    uploaded: u64,
    target: Option<Target>,
    ball_inst: GrowBuf,
    mist: wgpu::RenderPipeline,
    mist_inst: GrowBuf,
    water: crate::view3d_water::Water,
    terrain: crate::view3d_terrain::Terrain3d,
    meshes: HashMap<String, GpuMesh>,
}

impl Gpu3d {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Gpu3d {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("simtara 3d"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara 3d uniforms"), size: 21 * 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simtara 3d"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simtara 3d"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }] });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara 3d"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: INST,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 20, shader_location: 3 },
            ],
        };
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: VERT,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 5 },
            ],
        };
        let pipeline = |label: &str, vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout>], blend: Option<wgpu::BlendState>, depth_write: bool, compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // the ground is drawn first and never hides a thing (the CPU painter does the same)
        let ground = pipeline("simtara 3d ground", "vs_ground", "fs_ground", &[], Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false, wgpu::CompareFunction::Always);
        let balls = pipeline("simtara 3d balls", "vs_ball", "fs_ball", &[Some(instance_layout.clone())], None, true, wgpu::CompareFunction::Less);
        // spray: soft see-through puffs, several per droplet, that never hide what is behind them
        let mist = pipeline("simtara 3d spray", "vs_mist", "fs_mist", &[Some(instance_layout.clone())], Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false, wgpu::CompareFunction::Less);
        let meshes_pipe = pipeline("simtara 3d meshes", "vs_mesh", "fs_mesh", &[Some(vertex_layout), Some(instance_layout)], None, true, wgpu::CompareFunction::Less);
        // smoke: a full-screen pass that marches rays through a 3D texture, blended over the scene
        let vol_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simtara 3d volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let vol_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara 3d volume"), bind_group_layouts: &[Some(&layout), Some(&vol_layout)], immediate_size: 0 });
        let vol_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("simtara 3d volume"),
            layout: Some(&vol_pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_volume"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("simtara 3d volume"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let head = &SHADER[..SHADER.find("struct GroundOut").unwrap()];
        let water = crate::view3d_water::Water::new(&device, &layout, COLOR, DEPTH, SAMPLES, head);
        let terrain = crate::view3d_terrain::Terrain3d::new(&device, &layout, COLOR, DEPTH, SAMPLES, head);
        let full = |label: &str, fs: &str, layout: &wgpu::PipelineLayout, depth_write: bool, compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(if fs == "fs_pool" { "vs_pool" } else { "vs_full" }), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(depth_write), depth_compare: Some(compare), stencil: Default::default(), bias: Default::default() }),
                multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sky_pipe = full("simtara 3d sky", "fs_sky", &pl, false, wgpu::CompareFunction::Always);
        let solids_pipe = full("simtara 3d solids", "fs_solids", &vol_pl, true, wgpu::CompareFunction::Less);
        let pool_pipe = full("simtara 3d pool", "fs_pool", &vol_pl, true, wgpu::CompareFunction::Less);
        let caustics = Self::caustics_group(&device, &queue, &vol_layout, &sampler, 1, 1, &[85]);
        Gpu3d { device, queue, uniforms, bind, ground, balls, meshes_pipe, vol_pipe, vol_layout, sampler, vol: None, sky_pipe, solids_pipe, pool_pipe, caustics, uploaded: 0, target: None, ball_inst: GrowBuf::default(), mist, mist_inst: GrowBuf::default(), water, terrain, meshes: HashMap::new() }
    }

    fn ensure_target(&mut self, size: [u32; 2]) -> bool {
        if self.target.as_ref().is_some_and(|t| t.size == size) {
            return false;
        }
        let tex = |label: &str, format, samples, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = tex("simtara 3d msaa", COLOR, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let depth = tex("simtara 3d depth", DEPTH, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let resolve = tex("simtara 3d picture", COLOR, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let resolve_view = resolve.create_view(&Default::default());
        let scene = tex("simtara 3d scene", COLOR, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let key = self.target.as_ref().map_or(1, |t| t.key + 1);
        self.target = Some(Target { size, scene, key, msaa, depth, resolve, resolve_view });
        true
    }

    /// Draw one picture. Returns true when the output texture was (re)created, so a caller that
    /// shows it elsewhere (egui) must point at the new one.
    pub fn render(&mut self, size: [u32; 2], u: &Uniforms, items: &[Item], models: &HashMap<String, Mesh>, volume: Option<&Volume>) -> bool {
        let size = [size[0].clamp(1, 8192), size[1].clamp(1, 8192)];
        let recreated = self.ensure_target(size);
        let mut u = *u;
        u.viewport = [size[0] as f32, size[1] as f32];
        u.volume = volume.map(|v| (v.size, v.dims));
        u.water_radius = volume.map_or(0.0, |v| v.water_radius);
        u.pool = volume.is_some_and(|v| v.water_radius > 0.0);
        u.blend = volume.map_or(1.0, |v| v.blend.clamp(0.0, 1.0));
        u.n_objects = 0;
        u.steered = volume.and_then(|v| v.steered);
        if let Some(v) = volume {
            for (i, (shape, c, h)) in v.objects.iter().take(4).enumerate() {
                u.objects[i] = (shape == "ball", *c, *h);
                u.n_objects = i as u32 + 1;
            }
        }
        self.queue.write_buffer(&self.uniforms, 0, &u.bytes());
        // the fluid's pictures go to the card only when they changed (once per step, not per frame)
        let fresh = volume.is_some_and(|v| v.generation == 0 || v.generation != self.uploaded);
        let terr = volume.and_then(|v| v.terrain.as_ref());
        if let Some(t) = terr {
            // a flood: ground and water from height grids, no box of cells
            self.terrain.prepare(&self.device, &self.queue, t, fresh);
            if fresh {
                self.uploaded = volume.unwrap().generation;
            }
        } else if let (Some(v), true) = (volume, fresh) {
            self.upload_volume(v);
            self.uploaded = v.generation;
        }

        // group things by how they are drawn
        let mut ball_bytes = Vec::new();
        let mut mist_bytes = Vec::new();
        let mut by_mesh: HashMap<&str, Vec<u8>> = HashMap::new();
        for it in items {
            if it.shape == "water" {
                // liquid is drawn from its field (view3d_water), never as particles
                continue;
            } else if it.shape == "spray" && u.water_radius > 0.0 {
                // four puffs per droplet (the shader scatters them by instance number)
                for _ in 0..4 {
                    push_instance(&mut mist_bytes, it);
                }
            } else if is_ball(&it.shape, models) {
                push_instance(&mut ball_bytes, it);
            } else {
                push_instance(by_mesh.entry(it.shape.as_str()).or_default(), it);
            }
        }
        let balls = (ball_bytes.len() as u64 / INST) as u32;
        self.ball_inst.write(&self.device, &self.queue, &ball_bytes);
        let mut draws: Vec<(String, u32)> = Vec::new();
        for (name, bytes) in &by_mesh {
            let m = &models[*name];
            let key = fingerprint(m);
            if self.meshes.get(*name).is_none_or(|g| g.key != key) {
                let data = mesh_bytes(m);
                let verts = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara 3d mesh"), size: (data.len() as u64).max(VERT), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                self.queue.write_buffer(&verts, 0, &data);
                self.meshes.insert(name.to_string(), GpuMesh { key, verts, count: (m.t.len() * 3) as u32, inst: GrowBuf::default() });
            }
            let g = self.meshes.get_mut(*name).unwrap();
            g.inst.write(&self.device, &self.queue, bytes);
            draws.push((name.to_string(), (bytes.len() as u64 / INST) as u32));
        }

        let mists = (mist_bytes.len() as u64 / INST) as u32;
        self.mist_inst.write(&self.device, &self.queue, &mist_bytes);
        let pool = u.pool && volume.is_some();
        if let Some(c) = volume.and_then(|v| v.caustics.as_ref()).filter(|_| pool && fresh) {
            self.caustics = Self::caustics_group(&self.device, &self.queue, &self.vol_layout, &self.sampler, c.0, c.1, &c.2);
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("simtara 3d") });
        let t = self.target.as_ref().unwrap();
        let liquid = volume.and_then(|v| v.liquid.as_ref().map(|l| (l, v.dims)));
        if let Some((l, dims)) = liquid {
            self.water.prepare(&self.device, &self.queue, l, dims, &t.scene, t.key, fresh);
        }
        let bg = 14.0 / 255.0;
        fn colour_att<'a>(msaa: &'a wgpu::TextureView, load: wgpu::LoadOp<wgpu::Color>, store: wgpu::StoreOp, resolve: &'a wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'a> {
            wgpu::RenderPassColorAttachment { view: msaa, depth_slice: None, resolve_target: Some(resolve), ops: wgpu::Operations { load, store } }
        }
        // with water: first the scene without water (kept, so the water can look through it),
        // then the water and spray on top; without water: everything in one pass
        let two = liquid.is_some();
        {
            let clear = wgpu::LoadOp::Clear(wgpu::Color { r: bg, g: bg, b: bg, a: 1.0 });
            let keep = if two { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("simtara 3d scene"),
                color_attachments: &[Some(colour_att(&t.msaa, clear, keep, if two { &t.scene } else { &t.resolve_view }))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: keep }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind, &[]);
            if let (Some(t), Some(g)) = (terr, self.terrain.group()) {
                pass.set_pipeline(&self.sky_pipe);
                pass.draw(0..3, 0..1);
                pass.set_pipeline(&self.terrain.ground);
                pass.set_bind_group(1, g, &[]);
                pass.draw(0..crate::view3d_terrain::Terrain3d::vertices(t), 0..1);
            } else if pool {
                pass.set_pipeline(&self.sky_pipe);
                pass.draw(0..3, 0..1);
                pass.set_pipeline(&self.pool_pipe);
                pass.set_bind_group(1, &self.caustics.0, &[]);
                pass.draw(0..30, 0..1);
            } else if u.ground_on || u.grid_on {
                pass.set_pipeline(&self.ground);
                pass.draw(0..6, 0..1);
            }
            if balls > 0 {
                pass.set_pipeline(&self.balls);
                pass.set_vertex_buffer(0, self.ball_inst.buf.as_ref().unwrap().slice(..));
                pass.draw(0..6, 0..balls);
            }
            pass.set_pipeline(&self.meshes_pipe);
            for (name, n) in &draws {
                let g = &self.meshes[name];
                pass.set_vertex_buffer(0, g.verts.slice(..));
                pass.set_vertex_buffer(1, g.inst.buf.as_ref().unwrap().slice(..));
                pass.draw(0..g.count, 0..*n);
            }
            if let (Some(t), Some(g)) = (terr, self.terrain.group()) {
                pass.set_pipeline(&self.terrain.water);
                pass.set_bind_group(1, g, &[]);
                pass.draw(0..crate::view3d_terrain::Terrain3d::vertices(t), 0..1);
            } else if let (Some(_), Some((_, vg, _))) = (volume, &self.vol) {
                if pool {
                    // solids with depth, so the water can be in front of or behind them
                    pass.set_pipeline(&self.solids_pipe);
                } else {
                    pass.set_pipeline(&self.vol_pipe);
                }
                pass.set_bind_group(1, vg, &[]);
                pass.draw(0..3, 0..1);
            }
            if !two && mists > 0 {
                pass.set_pipeline(&self.mist);
                pass.set_vertex_buffer(0, self.mist_inst.buf.as_ref().unwrap().slice(..));
                pass.draw(0..6, 0..mists);
            }
        }
        if two {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("simtara 3d water"),
                color_attachments: &[Some(colour_att(&t.msaa, wgpu::LoadOp::Load, wgpu::StoreOp::Discard, &t.resolve_view))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &t.depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind, &[]);
            if let Some(g) = self.water.surface_group() {
                pass.set_pipeline(&self.water.composite);
                pass.set_bind_group(1, g, &[]);
                pass.draw(0..3, 0..1);
            }
            if mists > 0 {
                pass.set_pipeline(&self.mist);
                pass.set_vertex_buffer(0, self.mist_inst.buf.as_ref().unwrap().slice(..));
                pass.draw(0..6, 0..mists);
            }
            if let (Some(_), Some((_, vg, _)), false) = (volume, &self.vol, pool) {
                pass.set_pipeline(&self.vol_pipe);
                pass.set_bind_group(1, vg, &[]);
                pass.draw(0..3, 0..1);
            }
        }        self.queue.submit([enc.finish()]);
        recreated
    }

    fn caustics_group(device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, w: u32, h: u32, data: &[u8]) -> (wgpu::BindGroup, [u32; 2]) {
        // stored as a 2D slice of a 3D texture so it fits the volume's bind group layout
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("simtara 3d caustics"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rgba: Vec<u8> = data.iter().flat_map(|v| [*v, *v, *v, 255]).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simtara 3d caustics"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) }],
        });
        (bg, [w, h])
    }

    fn upload_volume(&mut self, v: &Volume) {
        if self.vol.as_ref().is_none_or(|(_, _, d)| *d != v.dims) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("simtara 3d smoke"),
                size: wgpu::Extent3d { width: v.dims[0], height: v.dims[1], depth_or_array_layers: v.dims[2] },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simtara 3d smoke"),
                layout: &self.vol_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) }],
            });
            self.vol = Some((tex, bg, v.dims));
        }
        let (tex, _, d) = self.vol.as_ref().unwrap();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &v.data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(d[0] * 4), rows_per_image: Some(d[1]) },
            wgpu::Extent3d { width: d[0], height: d[1], depth_or_array_layers: d[2] },
        );
    }

    /// The finished picture as RGBA bytes, row by row (for saving pictures and for tests).
    pub fn read_back(&self, size: [u32; 2]) -> Vec<u8> {
        let row = (size[0] * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * size[1]) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: self.texture().unwrap(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("gpu");
        let data = buf.slice(..).get_mapped_range().expect("mapped").to_vec();
        (0..size[1]).flat_map(|y| data[(y * row) as usize..(y * row + size[0] * 4) as usize].to_vec()).collect()
    }

    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.target.as_ref().map(|t| &t.resolve)
    }

    fn view(&self) -> Option<&wgpu::TextureView> {
        self.target.as_ref().map(|t| &t.resolve_view)
    }
}

/// The GPU renderer inside egui: owns the egui texture that shows the picture.
pub struct GpuView {
    rs: egui_wgpu::RenderState,
    gpu: Gpu3d,
    tex: Option<egui::TextureId>,
}

impl GpuView {
    pub fn new(rs: egui_wgpu::RenderState) -> GpuView {
        let gpu = Gpu3d::new(rs.device.clone(), rs.queue.clone());
        GpuView { rs, gpu, tex: None }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paint(&mut self, painter: &egui::Painter, rect: egui::Rect, pixels_per_point: f32, u: &Uniforms, items: &[Item], models: &HashMap<String, Mesh>, volume: Option<&Volume>) {
        let size = [(rect.width() * pixels_per_point).round().max(1.0) as u32, (rect.height() * pixels_per_point).round().max(1.0) as u32];
        let recreated = self.gpu.render(size, u, items, models, volume);
        let Some(view) = self.gpu.view() else { return };
        let mut renderer = self.rs.renderer.write();
        let id = match self.tex {
            None => renderer.register_native_texture(&self.rs.device, view, wgpu::FilterMode::Linear),
            Some(id) => {
                if recreated {
                    renderer.update_egui_texture_from_wgpu_texture(&self.rs.device, view, wgpu::FilterMode::Linear, id);
                }
                id
            }
        };
        self.tex = Some(id);
        drop(renderer);
        painter.image(id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    }
}

impl Drop for GpuView {
    fn drop(&mut self) {
        if let Some(id) = self.tex.take() {
            self.rs.renderer.write().free_texture(&id);
        }
    }
}

const SHADER: &str = r#"
struct Uni {
    eye: vec4<f32>, r: vec4<f32>, u: vec4<f32>, f: vec4<f32>,
    params: vec4<f32>,   // sx, sy, near, far
    mode: vec4<f32>,     // ortho, ambient, strength, unit
    light: vec4<f32>,
    ground: vec4<f32>,   // x0, y0, x1, y1
    ground2: vec4<f32>,  // z, ground on, grid on, grid steps
    vp: vec4<f32>,       // picture width, height (pixels), smoke per metre, glow per metre
    vsize: vec4<f32>,    // smoke box size (metres), cell size
    vdims: vec4<f32>,    // cells along x, y, z, volume on
    water: vec4<f32>,    // water particle radius, pool, blend, number of objects
    objs: array<vec4<f32>, 8>, // per object: centre + kind (0 ball, 1 box), half sizes
};
@group(0) @binding(0) var<uniform> U: Uni;

// the same camera as the CPU painter: x right, y up, z = distance in front of the eye
fn project(p: vec3<f32>) -> vec4<f32> {
    let rel = p - U.eye.xyz;
    let x = dot(rel, U.r.xyz);
    let y = dot(rel, U.u.xyz);
    let z = dot(rel, U.f.xyz);
    let near = U.params.z;
    let far = U.params.w;
    if (U.mode.x > 0.5) {
        return vec4<f32>(x * U.params.x, y * U.params.y, clamp((z - near) / (far - near), 0.0, 1.0), 1.0);
    }
    return vec4<f32>(x * U.params.x, y * U.params.y, far * (z - near) / (far - near), z);
}

fn lit(n: vec3<f32>) -> f32 {
    return U.mode.y + U.mode.z * max(dot(n, U.light.xyz), 0.0);
}

fn shade(c: vec3<f32>, k: f32) -> vec3<f32> {
    return min(c * clamp(k, 0.0, 1.4), vec3<f32>(1.0));
}

fn corner(i: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    return c[i];
}

// a clear daytime sky (horizon pale, zenith deeper blue) with the sun, for water to reflect
fn sky(d: vec3<f32>) -> vec3<f32> {
    let up = clamp(d.z, -1.0, 1.0);
    var c = mix(vec3<f32>(0.78, 0.86, 0.94), vec3<f32>(0.30, 0.50, 0.82), pow(max(up, 0.0), 0.6));
    if (up < 0.0) {
        c = mix(vec3<f32>(0.78, 0.86, 0.94), vec3<f32>(0.32, 0.34, 0.36), min(-up * 4.0, 1.0));
    }
    let s = max(dot(d, U.light.xyz), 0.0);
    return c + vec3<f32>(1.0, 0.95, 0.85) * (pow(s, 800.0) * 5.0 + pow(s, 16.0) * 0.12) * U.mode.z;
}

struct GroundOut { @builtin(position) pos: vec4<f32>, @location(0) t: vec2<f32> };

@vertex fn vs_ground(@builtin(vertex_index) i: u32) -> GroundOut {
    let t = corner(i) * 0.5 + vec2<f32>(0.5);
    let g = U.ground;
    var o: GroundOut;
    o.pos = project(vec3<f32>(mix(g.x, g.z, t.x), mix(g.y, g.w, t.y), U.ground2.x));
    o.t = t;
    return o;
}

@fragment fn fs_ground(in: GroundOut) -> @location(0) vec4<f32> {
    let g = in.t * U.ground2.w;
    let d = abs(fract(g - vec2<f32>(0.5)) - vec2<f32>(0.5)) / max(fwidth(g), vec2<f32>(1e-6));
    let line = (1.0 - min(min(d.x, d.y), 1.0)) * U.ground2.z;
    let a = line * 22.0 / 255.0;
    // premultiplied: ground colour (if on), with white grid lines blended over it
    var col = vec4<f32>(0.0);
    if (U.ground2.y > 0.5) {
        col = vec4<f32>(22.0 / 255.0, 26.0 / 255.0, 30.0 / 255.0, 1.0);
    }
    return vec4<f32>(col.rgb * (1.0 - a) + vec3<f32>(a), col.a + a * (1.0 - col.a));
}

struct Inst {
    @location(0) pos: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) heading: f32,
};

struct BallOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) q: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) @interpolate(flat) centre: vec3<f32>,
    @location(3) @interpolate(flat) rad: f32,
};

@vertex fn vs_ball(@builtin(vertex_index) i: u32, inst: Inst) -> BallOut {
    let q = corner(i);
    let rad = U.mode.w * inst.size;
    var o: BallOut;
    o.pos = project(inst.pos + (U.r.xyz * q.x + U.u.xyz * q.y) * rad);
    o.q = q;
    o.color = inst.color.rgb;
    o.centre = inst.pos;
    o.rad = rad;
    return o;
}

struct BallFrag { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 };

// ---- spray: each droplet becomes a few soft puffs scattered around it (fine mist) ----
struct MistOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) q: vec2<f32>,
    @location(1) shade: f32,
};

fn hash(n: u32) -> f32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) / 4294967295.0;
}

@vertex fn vs_mist(@builtin(vertex_index) i: u32, @builtin(instance_index) n: u32, inst: Inst) -> MistOut {
    let q = corner(i);
    let r = U.water.x;
    // a stable random offset and size for each of the puffs of a droplet
    let off = vec3<f32>(hash(n * 3u) - 0.5, hash(n * 3u + 1u) - 0.5, hash(n * 3u + 2u) - 0.5) * r * 3.0;
    let size = r * (0.8 + 1.2 * hash(n * 7u + 5u)) * inst.size;
    var o: MistOut;
    o.pos = project(inst.pos + off + (U.r.xyz * q.x + U.u.xyz * q.y) * size);
    o.q = q;
    o.shade = U.mode.y + U.mode.z * 0.6;
    return o;
}

@fragment fn fs_mist(in: MistOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.q, in.q);
    if (d2 > 1.0) {
        discard;
    }
    let a = 0.16 * exp(-3.0 * d2);
    let c = vec3<f32>(0.92, 0.95, 1.0) * min(in.shade, 1.2);
    return vec4<f32>(c * a, a);
}

@fragment fn fs_ball(in: BallOut) -> BallFrag {
    let d2 = dot(in.q, in.q);
    if (d2 > 1.0) {
        discard;
    }
    // the point on the ball's surface seen at this pixel, and its normal (towards the camera)
    let n = U.r.xyz * in.q.x + U.u.xyz * in.q.y - U.f.xyz * sqrt(1.0 - d2);
    let c = project(in.centre + n * in.rad);
    var o: BallFrag;
    o.color = vec4<f32>(shade(in.color, lit(n)), 1.0);
    o.depth = clamp(c.z / c.w, 0.0, 1.0);
    return o;
}

struct MeshOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex fn vs_mesh(@location(4) p: vec3<f32>, @location(5) n: vec3<f32>, inst: Inst) -> MeshOut {
    let s = U.mode.w * inst.size * 2.0;
    let h = radians(inst.heading);
    let ch = cos(h);
    let sh = sin(h);
    let rp = vec3<f32>(p.x * ch - p.y * sh, p.x * sh + p.y * ch, p.z);
    let rn = vec3<f32>(n.x * ch - n.y * sh, n.x * sh + n.y * ch, n.z);
    let w = inst.pos + rp * s;
    var o: MeshOut;
    o.pos = project(w);
    o.world = w;
    o.n = rn;
    o.color = inst.color.rgb;
    return o;
}

@fragment fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    var n = normalize(in.n);
    // light both sides of a face the same, so models with mixed winding still look right
    if (dot(n, U.eye.xyz - in.world) < 0.0) {
        n = -n;
    }
    return vec4<f32>(shade(in.color, lit(n)), 1.0);
}

// ---- smoke, fire and solids: rays marched through the 3D texture (same as view3d::march) ----
@group(1) @binding(0) var vol_tex: texture_3d<f32>;
@group(1) @binding(1) var vol_smp: sampler;

struct FullOut { @builtin(position) pos: vec4<f32> };

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: FullOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    return o;
}

fn vol(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(vol_tex, vol_smp, p / U.vsize.xyz, 0.0);
}

fn fire_colour(g: f32) -> vec3<f32> {
    let h = clamp(g, 0.0, 1.0);
    return vec3<f32>(min(h * 2.2, 1.0), clamp(h * 1.6 - 0.35, 0.0, 1.0) * 0.85, clamp(h * 2.0 - 1.4, 0.0, 1.0) * 0.6);
}

@fragment fn fs_volume(in: FullOut) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.pos.x / U.vp.x * 2.0 - 1.0, 1.0 - in.pos.y / U.vp.y * 2.0);
    let dx = ndc.x / U.params.x;
    let dy = ndc.y / U.params.y;
    var o = U.eye.xyz;
    var d = U.f.xyz;
    if (U.mode.x > 0.5) {
        o = U.eye.xyz + U.r.xyz * dx + U.u.xyz * dy;
    } else {
        d = normalize(U.f.xyz + U.r.xyz * dx + U.u.xyz * dy);
    }
    let size = U.vsize.xyz;
    let inv = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let ta = (vec3<f32>(0.0) - o) * inv;
    let tb = (size - o) * inv;
    let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
    let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    if (t1 <= t0 || U.vdims.w < 0.5) {
        discard;
    }
    let cell = U.vsize.w;
    let step = cell * 0.75;
    let l = U.light.xyz;
    var col = vec3<f32>(0.0);
    var trans = 1.0;
    var t = t0 + step * 0.5;
    for (var n = 0; n < 1024; n = n + 1) {
        if (t >= t1 || trans < 0.01) {
            break;
        }
        let p = o + d * t;
        let s = vol(p);
        if (s.b > 0.5) {
            // a solid surface: normal from how the solid channel changes
            let e = cell;
            let g = vec3<f32>(vol(p - vec3<f32>(e, 0.0, 0.0)).b - vol(p + vec3<f32>(e, 0.0, 0.0)).b,
                              vol(p - vec3<f32>(0.0, e, 0.0)).b - vol(p + vec3<f32>(0.0, e, 0.0)).b,
                              vol(p - vec3<f32>(0.0, 0.0, e)).b - vol(p + vec3<f32>(0.0, 0.0, e)).b);
            let nn = normalize(g + vec3<f32>(0.0, 0.0, 1e-6));
            let k = min(U.mode.y + U.mode.z * max(dot(nn, l), 0.0), 1.4);
            col = col + trans * vec3<f32>(0.55, 0.57, 0.6) * k;
            trans = 0.0;
            break;
        }
        let dens = s.r * U.vp.z;
        if (dens > 1e-4 || s.g > 0.02) {
            let alpha = 1.0 - exp(-dens * step);
            var shadow = 0.0;
            for (var k = 1; k <= 4; k = k + 1) {
                shadow = shadow + vol(p + l * cell * 1.5 * f32(k)).r;
            }
            let lit = U.mode.y + U.mode.z * exp(-shadow * U.vp.z * cell * 1.5);
            let glow = s.g * s.g * step * U.vp.w;
            col = col + trans * (alpha * vec3<f32>(0.82, 0.84, 0.88) * lit + glow * fire_colour(s.g));
            trans = trans * (1.0 - alpha);
        }
        t = t + step;
    }
    return vec4<f32>(min(col, vec3<f32>(1.0)), 1.0 - trans);
}

fn view_ray(fc: vec2<f32>) -> array<vec3<f32>, 2> {
    let ndc = vec2<f32>(fc.x / U.vp.x * 2.0 - 1.0, 1.0 - fc.y / U.vp.y * 2.0);
    let dx = ndc.x / U.params.x;
    let dy = ndc.y / U.params.y;
    if (U.mode.x > 0.5) {
        return array<vec3<f32>, 2>(U.eye.xyz + U.r.xyz * dx + U.u.xyz * dy, U.f.xyz);
    }
    return array<vec3<f32>, 2>(U.eye.xyz, normalize(U.f.xyz + U.r.xyz * dx + U.u.xyz * dy));
}

@fragment fn fs_sky(in: FullOut) -> @location(0) vec4<f32> {
    let ray = view_ray(in.pos.xy);
    return vec4<f32>(sky(ray[1]), 1.0);
}

// solids of a liquid scene, drawn with their depth so water in front of them hides them
struct SolidOut { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };

// where a ray first meets object i (a big number if it does not), and the normal there
fn hit_object(o: vec3<f32>, d: vec3<f32>, i: u32) -> vec4<f32> {
    let c = U.objs[2u * i].xyz;
    let h = U.objs[2u * i + 1u].xyz;
    if (U.objs[2u * i].w < 0.5) {
        let oc = o - c;
        let b = dot(oc, d);
        let disc = b * b - (dot(oc, oc) - h.x * h.x);
        if (disc < 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 1e9); }
        let t = -b - sqrt(disc);
        if (t < 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 1e9); }
        return vec4<f32>(normalize(o + d * t - c), t);
    }
    let inv = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let ta = (c - h - o) * inv;
    let tb = (c + h - o) * inv;
    let tmin = min(ta, tb);
    let tmax = max(ta, tb);
    let t0 = max(max(tmin.x, tmin.y), tmin.z);
    let t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 < max(t0, 0.0)) { return vec4<f32>(0.0, 0.0, 0.0, 1e9); }
    var n = vec3<f32>(-sign(d.x), 0.0, 0.0);
    if (t0 == tmin.y) { n = vec3<f32>(0.0, -sign(d.y), 0.0); }
    if (t0 == tmin.z) { n = vec3<f32>(0.0, 0.0, -sign(d.z)); }
    return vec4<f32>(n, t0);
}

@fragment fn fs_solids(in: FullOut) -> SolidOut {
    let ray = view_ray(in.pos.xy);
    let o = ray[0];
    let d = ray[1];
    // the nearest object, drawn with its exact shape (a warm yellow, easy to follow in the water)
    var best = vec4<f32>(0.0, 0.0, 0.0, 1e9);
    var which = 0u;
    let count = u32(U.water.w);
    for (var i = 0u; i < count; i = i + 1u) {
        let hit = hit_object(o, d, i);
        if (hit.w < best.w) { best = hit; which = i; }
    }
    if (best.w < 1e8) {
        let p = o + d * best.w;
        let k = min(U.mode.y + U.mode.z * max(dot(best.xyz, U.light.xyz), 0.0), 1.4);
        let spec = pow(max(dot(reflect(d, best.xyz), U.light.xyz), 0.0), 40.0) * 0.4 * U.mode.z;
        var out: SolidOut;
        var palette = array<vec3<f32>, 4>(vec3<f32>(0.95, 0.72, 0.22), vec3<f32>(0.90, 0.30, 0.25), vec3<f32>(0.30, 0.55, 0.95), vec3<f32>(0.40, 0.80, 0.35));
        let rim = pow(1.0 - abs(dot(best.xyz, d)), 2.0) * U.objs[2u * which + 1u].w;
        out.colour = vec4<f32>(min(palette[which] * k + vec3<f32>(spec) + vec3<f32>(1.0, 1.0, 0.8) * rim * 0.9, vec3<f32>(1.0)), 1.0);
        let c = project(p);
        out.depth = clamp(c.z / c.w, 0.0, 1.0);
        return out;
    }
    let size = U.vsize.xyz;
    let inv = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let ta = (vec3<f32>(0.0) - o) * inv;
    let tb = (size - o) * inv;
    let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
    let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    if (t1 <= t0) {
        discard;
    }
    let cell = U.vsize.w;
    let step = cell * 0.9;
    var t = t0 + step * 0.5;
    for (var n = 0; n < 1024; n = n + 1) {
        if (t >= t1) {
            break;
        }
        var p = o + d * t;
        if (vol(p).b > 0.5) {
            // pin the surface of the solid down between the last two steps
            var lo = t - step;
            var hi = t;
            for (var k = 0; k < 5; k = k + 1) {
                let m = 0.5 * (lo + hi);
                if (vol(o + d * m).b > 0.5) { hi = m; } else { lo = m; }
            }
            p = o + d * hi;
            let e = cell;
            let g = vec3<f32>(vol(p - vec3<f32>(e, 0.0, 0.0)).b - vol(p + vec3<f32>(e, 0.0, 0.0)).b,
                              vol(p - vec3<f32>(0.0, e, 0.0)).b - vol(p + vec3<f32>(0.0, e, 0.0)).b,
                              vol(p - vec3<f32>(0.0, 0.0, e)).b - vol(p + vec3<f32>(0.0, 0.0, e)).b);
            let nn = normalize(g + vec3<f32>(0.0, 0.0, 1e-6));
            let k = min(U.mode.y + U.mode.z * max(dot(nn, U.light.xyz), 0.0), 1.4);
            var out: SolidOut;
            out.colour = vec4<f32>(vec3<f32>(0.62, 0.6, 0.56) * k, 1.0);
            let c = project(p);
            out.depth = clamp(c.z / c.w, 0.0, 1.0);
            return out;
        }
        t = t + step;
    }
    discard;
}

// ---- the tank as a tiled pool: floor and the walls seen from inside, with caustics ----
// (the caustics are bound like the smoke volume: a 3D texture one slice deep)

struct PoolOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) n: vec3<f32>,
};

@vertex fn vs_pool(@builtin(vertex_index) i: u32) -> PoolOut {
    let face = i / 6u;
    let c = corner(i % 6u) * 0.5 + vec2<f32>(0.5);
    let s = U.vsize.xyz;
    var p = vec3<f32>(0.0);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (face == 0u) { p = vec3<f32>(c.x * s.x, c.y * s.y, 0.0); n = vec3<f32>(0.0, 0.0, 1.0); }
    if (face == 1u) { p = vec3<f32>(0.0, c.x * s.y, c.y * s.z); n = vec3<f32>(1.0, 0.0, 0.0); }
    if (face == 2u) { p = vec3<f32>(s.x, c.x * s.y, c.y * s.z); n = vec3<f32>(-1.0, 0.0, 0.0); }
    if (face == 3u) { p = vec3<f32>(c.x * s.x, 0.0, c.y * s.z); n = vec3<f32>(0.0, 1.0, 0.0); }
    if (face == 4u) { p = vec3<f32>(c.x * s.x, s.y, c.y * s.z); n = vec3<f32>(0.0, -1.0, 0.0); }
    var o: PoolOut;
    o.pos = project(p);
    o.world = p;
    o.n = n;
    return o;
}

@fragment fn fs_pool(in: PoolOut) -> @location(0) vec4<f32> {
    let p = in.world;
    let n = in.n;
    // walls on the camera's side would hide the water: only the far walls and the floor
    if (dot(n, U.eye.xyz - p) < 0.0) {
        discard;
    }
    let s = U.vsize.xyz;
    // tiles about 1/8 of the pool's short side, with thin grout
    let tile = min(s.x, s.y) / 8.0;
    var uv = p.xy;
    if (abs(n.x) > 0.5) { uv = p.yz; } else if (abs(n.y) > 0.5) { uv = p.xz; }
    let g = abs(fract(uv / tile) - vec2<f32>(0.5));
    let line = 1.0 - smoothstep(0.475, 0.495, max(g.x, g.y));
    let base = mix(vec3<f32>(0.60, 0.66, 0.68), vec3<f32>(0.84, 0.89, 0.90), line);
    // soft shadow in the corners (ambient occlusion), as in WebGL Water
    let edge = min(min(min(p.x, s.x - p.x), min(p.y, s.y - p.y)), p.z);
    let ao = 1.0 - 0.35 * exp(-edge / (tile * 1.5));
    var light = U.mode.y + U.mode.z * max(dot(n, U.light.xyz), 0.0);
    if (n.z > 0.5) {
        // the floor gets the caustics: sunlight focused by the water's surface
        light = U.mode.y + U.mode.z * max(dot(n, U.light.xyz), 0.0) * textureSampleLevel(vol_tex, vol_smp, vec3<f32>(p.xy / s.xy, 0.5), 0.0).r * 3.0;
    }
    return vec4<f32>(min(base * light * ao, vec3<f32>(1.0)), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn read(gpu: &Gpu3d, size: [u32; 2]) -> Vec<u8> {
        gpu.read_back(size)
    }

    fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [img[i], img[i + 1], img[i + 2], img[i + 3]]
    }

    /// Camera looking down -y at the origin from 10 units away.
    fn camera(ortho: bool) -> Uniforms {
        Uniforms {
            eye: [0.0, 10.0, 0.0],
            r: [-1.0, 0.0, 0.0],
            u: [0.0, 0.0, 1.0],
            f: [0.0, -1.0, 0.0],
            sx: if ortho { 0.2 } else { 2.0 },
            sy: if ortho { 0.2 } else { 2.0 },
            near: 0.01,
            far: 30.0,
            ortho,
            ambient: 1.0,
            strength: 0.0,
            unit: 1.0,
            light: [0.0, 0.0, 1.0],
            ground: [-1.0, -1.0, 1.0, 1.0],
            ground_z: -5.0,
            ground_on: false,
            grid_on: false,
            viewport: [0.0; 2],
            volume: None,
            water_radius: 0.0,
            pool: false,
            blend: 1.0,
            objects: [(true, [0.0; 3], [0.0; 3]); 4],
            n_objects: 0,
            steered: None,
        }
    }

    fn item(pos: [f32; 3], color: egui::Color32, shape: &str) -> Item {
        Item { id: 1, pos, color, size: 1.0, heading: 0.0, shape: shape.into() }
    }

    #[test]
    fn draws_balls_and_meshes_with_depth_in_both_cameras() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter, skipping");
            return;
        };
        let mut gpu = Gpu3d::new(device, queue);
        let mut models = HashMap::new();
        models.insert("cube".to_string(), crate::view3d::builtin("cube").unwrap());
        let (red, blue) = (egui::Color32::from_rgb(200, 0, 0), egui::Color32::from_rgb(0, 0, 200));
        let size = [64, 64];
        for ortho in [false, true] {
            // a red ball in front (nearer the eye) of a blue cube, both on the centre line
            let items = [item([0.0, 0.0, 0.0], blue, "cube"), item([0.0, 3.0, 0.0], red, "sphere")];
            assert!(gpu.render(size, &camera(ortho), &items, &models, None) || ortho, "first picture creates the texture");
            let img = read(&gpu, size);
            assert_eq!(px(&img, 64, 1, 1), [14, 14, 14, 255], "background in the corner");
            let mid = px(&img, 64, 32, 32);
            assert!(mid[0] > 150 && mid[2] < 30, "the nearer red ball hides the cube (ortho={ortho}): {mid:?}");

            // the cube alone shows blue in the middle; an unknown model falls back to a ball
            gpu.render(size, &camera(ortho), &[item([0.0, 0.0, 0.0], blue, "cube")], &models, None);
            let mid = px(&read(&gpu, size), 64, 32, 32);
            assert!(mid[2] > 150 && mid[0] < 30, "cube drawn (ortho={ortho}): {mid:?}");
            gpu.render(size, &camera(ortho), &[item([0.0, 0.0, 0.0], red, "no-such-model")], &models, None);
            let mid = px(&read(&gpu, size), 64, 32, 32);
            assert!(mid[0] > 150, "unknown model drawn as a ball (ortho={ortho}): {mid:?}");
        }
        // a new size makes a new texture
        assert!(gpu.render([32, 16], &camera(false), &[], &models, None));
        assert!(!gpu.render([32, 16], &camera(false), &[], &models, None));
    }

    #[test]
    fn things_land_where_the_cpu_painter_puts_them() {
        let Some((device, queue)) = device() else { return };
        let mut gpu = Gpu3d::new(device, queue);
        let red = egui::Color32::from_rgb(200, 0, 0);
        let size = [64, 64];
        // CPU painter: screen x = centre.x + x / z * focal, screen y = centre.y - y / z * focal,
        // with focal = sx * half width. Here x = -2 (r points to -x), z = 10, so x = 32 - 12.8.
        gpu.render(size, &camera(false), &[item([2.0, 0.0, 0.0], red, "sphere")], &HashMap::new(), None);
        let img = read(&gpu, size);
        assert!(px(&img, 64, 19, 32)[0] > 150, "left of centre");
        assert_eq!(px(&img, 64, 45, 32), [14, 14, 14, 255], "nothing on the right");
        // up in the world is up on the screen
        gpu.render(size, &camera(false), &[item([0.0, 0.0, 2.0], red, "sphere")], &HashMap::new(), None);
        let img = read(&gpu, size);
        assert!(px(&img, 64, 32, 19)[0] > 150, "above centre");
        assert_eq!(px(&img, 64, 32, 45), [14, 14, 14, 255], "nothing below");
    }

    #[test]
    fn smoke_and_solids_look_the_same_as_the_cpu_painter() {
        let Some((device, queue)) = device() else { return };
        let mut gpu = Gpu3d::new(device, queue);
        // a 2 m box of thin smoke (density 0.1) with a solid cube in its middle third
        let n = 6u32;
        let mut data = vec![0u8; (n * n * n * 4) as usize];
        for c in 0..(n * n * n) as usize {
            data[c * 4] = 26;
        }
        let vol = Volume { dims: [n; 3], size: [2.0; 3], data, water_radius: 0.0, caustics: None, liquid: None, generation: 0, blend: 1.0, objects: Vec::new(), steered: None, terrain: None };
        let mut u = camera(false);
        u.eye = [1.0, 10.0, 1.0];
        let size = [64, 64];
        gpu.render(size, &u, &[], &HashMap::new(), Some(&vol));
        let mid = px(&read(&gpu, size), 64, 32, 32);
        let light = simtara_things::Light { ambient: 1.0, strength: 0.0, ..Default::default() };
        let cpu = crate::view3d::march(&vol, u.eye, u.f, [0.0, 0.0, 1.0], &light, 2.0 / n as f32);
        // background is 14; smoke over it: colour + (1 - alpha) * background
        let expect = |a: usize| ((cpu[a] + (1.0 - cpu[3]) * 14.0 / 255.0) * 255.0) as i32;
        for a in 0..3 {
            assert!((mid[a] as i32 - expect(a)).abs() <= 6, "channel {a}: gpu {mid:?}, cpu {cpu:?}");
        }
        assert!(cpu[3] > 0.7 && cpu[3] < 0.95, "2 m of thin smoke is mostly opaque: {cpu:?}");

        // a solid block inside stops the ray: lit grey, fully covering
        let mut vol = vol;
        for k in 2..4u32 {
            for j in 2..4u32 {
                for i in 2..4u32 {
                    vol.data[((i + n * (j + n * k)) * 4 + 2) as usize] = 255;
                }
            }
        }
        gpu.render(size, &u, &[], &HashMap::new(), Some(&vol));
        let mid = px(&read(&gpu, size), 64, 32, 32);
        let cpu = crate::view3d::march(&vol, u.eye, u.f, [0.0, 0.0, 1.0], &light, 2.0 / n as f32);
        assert!((cpu[3] - 1.0).abs() < 1e-4, "{cpu:?}");
        for a in 0..3 {
            assert!((mid[a] as i32 - (cpu[a] * 255.0) as i32).abs() <= 8, "solid channel {a}: gpu {mid:?}, cpu {cpu:?}");
        }
        // and outside the box there is nothing
        assert_eq!(px(&read(&gpu, size), 64, 1, 1), [14, 14, 14, 255]);
    }

    #[test]
    fn water_is_one_smooth_liquid_surface_with_sky_around_it() {
        let Some((device, queue)) = device() else { return };
        let mut gpu = Gpu3d::new(device, queue);
        // a box of 20 x 20 x 20 cells (1 m), full of water in its middle
        let n = 20u32;
        let mut field = vec![0.0f32; (n * n * n) as usize];
        for k in 2..18 {
            for j in 2..18 {
                for i in 2..18 {
                    field[(i + n * (j + n * k)) as usize] = 1.0;
                }
            }
        }
        let vol = Volume { dims: [n; 3], size: [1.0; 3], data: vec![0; (n * n * n * 4) as usize], water_radius: 0.03, caustics: None, liquid: Some(field), generation: 0, blend: 1.0, objects: Vec::new(), steered: None, terrain: None };
        let mut u = camera(false);
        u.eye = [0.5, 4.0, 0.5];
        u.ambient = 0.5;
        u.strength = 0.5;
        u.light = [0.0, 0.6, 0.8];
        let size = [64, 64];
        gpu.render(size, &u, &[], &HashMap::new(), Some(&vol));
        let img = read(&gpu, size);
        let mid = px(&img, 64, 32, 32);
        assert!(mid[2] > mid[0] + 20 && mid[1] > mid[0] + 10, "water is blue-green: {mid:?}");
        // smooth: neighbouring pixels on the flat face barely differ
        let next = px(&img, 64, 34, 33);
        assert!((0..3).all(|a| (mid[a] as i32 - next[a] as i32).abs() < 10), "one smooth surface: {mid:?} vs {next:?}");
        let sky = px(&img, 64, 1, 1);
        assert!(sky[0] > 100 && sky[2] > 150 && (sky[0] as i32 - mid[0] as i32).abs() > 40, "sky around the water: {sky:?} vs water {mid:?}");
    }
    #[test]
    fn grid_without_ground_is_faint_lines_over_the_background() {
        let Some((device, queue)) = device() else { return };
        let mut gpu = Gpu3d::new(device, queue);
        let mut u = camera(false);
        // look straight down at a ground below the camera
        u.eye = [0.0, 0.0, 10.0];
        u.r = [1.0, 0.0, 0.0];
        u.u = [0.0, 1.0, 0.0];
        u.f = [0.0, 0.0, -1.0];
        u.ground = [-2.0, -2.0, 2.0, 2.0];
        u.ground_z = 0.0;
        u.ground_on = true;
        let size = [64, 64];
        gpu.render(size, &u, &[], &HashMap::new(), None);
        let img = read(&gpu, size);
        let p = px(&img, 64, 34, 34);
        assert!((20..=30).contains(&p[0]) && p[2] >= 28, "ground colour: {p:?}");
        u.ground_on = false;
        u.grid_on = true;
        gpu.render(size, &u, &[], &HashMap::new(), None);
        let img = read(&gpu, size);
        let brightest = img.chunks(4).map(|c| c[0]).max().unwrap();
        assert!(brightest > 14 && brightest < 60, "faint grid lines: {brightest}");
    }
}
