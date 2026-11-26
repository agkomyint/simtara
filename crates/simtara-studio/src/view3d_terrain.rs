//! Ground and flood water, drawn on the GPU from the flood's height grids.
//!
//! One grid of triangles through the cell centres is drawn twice. First the ground: shaded from
//! its slope, coloured by height (sea floor, beach sand, grass, rock) with buildings as solid
//! blocks with roofs. Then the water on top: its surface is ground + depth, and where a cell is
//! dry the surface dips just under the ground, so the shoreline is cut cleanly by the terrain.
//! The water reflects the sky (Fresnel), lets the ground show through where it is shallow
//! (Beer-Lambert), is blue at sea and muddy brown over land, and turns white where it moves
//! faster than its own waves (Froude number above 1: a breaking front, a bore, rapids).
//!
//! The last two steps are kept and blended, so the water moves smoothly between steps.

use eframe::egui_wgpu::wgpu;

pub struct Terrain3d {
    layout: wgpu::BindGroupLayout,
    pub ground: wgpu::RenderPipeline,
    pub water: wgpu::RenderPipeline,
    /// the cut-away sides: earth below the ground, water between the ground and the surface
    pub skirt_ground: wgpu::RenderPipeline,
    pub skirt_water: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    tex: Option<([wgpu::Texture; 2], [u32; 2], usize)>,
    group: Option<wgpu::BindGroup>,
}

impl Terrain3d {
    pub fn new(device: &wgpu::Device, camera: &wgpu::BindGroupLayout, colour: wgpu::TextureFormat, depth: wgpu::TextureFormat, samples: u32, shader_head: &str) -> Terrain3d {
        let src = format!("{shader_head}\n{TERRAIN_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("simtara terrain"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simtara terrain"),
            entries: &[
                tex(0),
                tex(1),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara terrain"), bind_group_layouts: &[Some(camera), Some(&layout)], immediate_size: 0 });
        let pipe = |label: &str, vs: &str, fs: &str, blend: Option<wgpu::BlendState>, write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: depth, depth_write_enabled: Some(write), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
                multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: colour, blend, write_mask: wgpu::ColorWrites::ALL })] }),
                multiview_mask: None,
                cache: None,
            })
        };
        let ground = pipe("simtara terrain ground", "vs_ground_t", "fs_ground_t", None, true);
        let water = pipe("simtara terrain water", "vs_water_t", "fs_water_t", Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false);
        let skirt_ground = pipe("simtara terrain sides", "vs_skirt_g", "fs_skirt_g", None, true);
        let skirt_water = pipe("simtara terrain water sides", "vs_skirt_w", "fs_skirt_w", Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false);
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("simtara terrain params"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Terrain3d { layout, ground, water, skirt_ground, skirt_water, params, tex: None, group: None }
    }

    /// Upload a new step (`fresh`), or only the drawing parameters.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: &crate::view3d::Terrain, fresh: bool) {
        let dims = [t.nx as u32, t.ny as u32];
        let new_size = self.tex.as_ref().is_none_or(|x| x.1 != dims);
        if new_size {
            let make = || {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("simtara terrain"),
                    size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            self.tex = Some(([make(), make()], dims, 0));
            self.group = None;
        }
        if fresh || new_size {
            let data: Vec<f32> = (0..t.nx * t.ny).flat_map(|c| [t.bed[c], t.depth[c], t.speed[c], t.building[c] as f32]).collect();
            let (texs, _, newest) = self.tex.as_mut().unwrap();
            let slots: Vec<usize> = if new_size { vec![0, 1] } else { vec![1 - *newest] };
            for s in slots {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &texs[s], mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    bytemuck::cast_slice(&data),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[0] * 16), rows_per_image: Some(dims[1]) },
                    wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: 1 },
                );
            }
            if !new_size {
                *newest = 1 - *newest;
            }
            self.group = None;
        }
        let p: [f32; 8] = [t.nx as f32, t.ny as f32, t.cell, t.vertical, t.sea_z, t.wet, 0.0, 0.0];
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));
        if self.group.is_none() {
            let (texs, _, newest) = self.tex.as_ref().unwrap();
            let (old, new) = (texs[1 - *newest].create_view(&Default::default()), texs[*newest].create_view(&Default::default()));
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simtara terrain"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&old) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&new) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
                ],
            }));
        }
    }

    pub fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.as_ref()
    }

    /// Vertices for the four cut-away sides.
    pub fn side_vertices(t: &crate::view3d::Terrain) -> u32 {
        ((2 * t.nx.saturating_sub(1) + 2 * t.ny.saturating_sub(1)) * 6) as u32
    }

    /// Vertices for one draw of the grid.
    pub fn vertices(t: &crate::view3d::Terrain) -> u32 {
        (t.nx.saturating_sub(1) * t.ny.saturating_sub(1) * 6) as u32
    }
}

const TERRAIN_WGSL: &str = r#"
@group(1) @binding(0) var t_old: texture_2d<f32>;
@group(1) @binding(1) var t_new: texture_2d<f32>;
struct TP { n: vec4<f32>, s: vec4<f32> };   // nx, ny, cell, vertical; sea z (picture units), wet depth (m)
@group(1) @binding(2) var<uniform> T: TP;

// ground height (picture units), depth (m), speed (m/s), building of a cell, blended between steps
fn cell_at(i: i32, j: i32) -> vec4<f32> {
    let c = vec2<i32>(clamp(i, 0, i32(T.n.x) - 1), clamp(j, 0, i32(T.n.y) - 1));
    let a = textureLoad(t_old, c, 0);
    let b = textureLoad(t_new, c, 0);
    var m = mix(a, b, U.water.z);
    m.w = b.w;
    return m;
}

fn grid_vertex(vi: u32) -> vec2<i32> {
    let q = vi / 6u;
    let k = vi % 6u;
    let w = u32(T.n.x) - 1u;
    var off = array<vec2<i32>, 6>(vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(1, 1), vec2<i32>(0, 0), vec2<i32>(1, 1), vec2<i32>(0, 1));
    return vec2<i32>(i32(q % w), i32(q / w)) + off[k];
}

fn place(ij: vec2<i32>, z: f32) -> vec3<f32> {
    return vec3<f32>((f32(ij.x) + 0.5) * T.n.z, (f32(ij.y) + 0.5) * T.n.z, z);
}

fn hash2(p: vec2<i32>) -> f32 {
    var x = u32(p.x) * 73856093u ^ u32(p.y) * 19349663u;
    x = (x ^ (x >> 13u)) * 1274126177u;
    return f32(x & 0xffffu) / 65535.0;
}

struct GroundT {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) building: f32,
    @location(3) @interpolate(flat) id: vec2<i32>,
};

@vertex fn vs_ground_t(@builtin(vertex_index) vi: u32) -> GroundT {
    let ij = grid_vertex(vi);
    let c = cell_at(ij.x, ij.y);
    let dx = (cell_at(ij.x + 1, ij.y).x - cell_at(ij.x - 1, ij.y).x) / (2.0 * T.n.z);
    let dy = (cell_at(ij.x, ij.y + 1).x - cell_at(ij.x, ij.y - 1).x) / (2.0 * T.n.z);
    var o: GroundT;
    o.world = place(ij, c.x);
    o.pos = project(o.world);
    o.n = normalize(vec3<f32>(-dx, -dy, 1.0));
    o.building = c.w;
    o.id = ij;
    return o;
}

@fragment fn fs_ground_t(in: GroundT) -> @location(0) vec4<f32> {
    var n = normalize(in.n);
    // metres above the sea
    let e = (in.world.z - T.s.x) / max(T.n.w, 1e-6);
    var col: vec3<f32>;
    if (in.building > 0.02) {
        // buildings: crisp faces, light walls, roofs of a few colours
        var fn_ = normalize(cross(dpdx(in.world), dpdy(in.world)));
        if (fn_.z < 0.0) { fn_ = -fn_; }
        n = fn_;
        if (n.z > 0.7) {
            let h = hash2(in.id / vec2<i32>(2, 2));
            col = select(select(vec3<f32>(0.62, 0.30, 0.24), vec3<f32>(0.46, 0.47, 0.50), h > 0.45), vec3<f32>(0.72, 0.66, 0.56), h > 0.8);
        } else {
            col = vec3<f32>(0.86, 0.84, 0.80);
        }
    } else {
        let sand = vec3<f32>(0.84, 0.77, 0.58);
        let seabed = vec3<f32>(0.52, 0.50, 0.40);
        let grass = mix(vec3<f32>(0.36, 0.55, 0.26), vec3<f32>(0.24, 0.40, 0.20), clamp(e / 60.0, 0.0, 1.0));
        let rock = vec3<f32>(0.47, 0.44, 0.40);
        col = mix(seabed, sand, smoothstep(-6.0, -0.5, e));
        col = mix(col, grass, smoothstep(1.0, 3.0, e));
        col = mix(col, rock, smoothstep(0.82, 0.62, n.z) * smoothstep(-0.5, 1.0, e));
        col = mix(col, vec3<f32>(0.92, 0.93, 0.95), smoothstep(300.0, 360.0, e));
    }
    return vec4<f32>(shade(col, lit(n)), 1.0);
}

struct WaterT {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) depth: f32,
    @location(3) speed: f32,
    @location(4) land: f32,
    @location(5) rise: f32,
};

fn surface(i: i32, j: i32, own: f32) -> f32 {
    let c = cell_at(i, j);
    if (c.y > T.s.y) { return c.x + c.y * T.n.w; }
    return own;
}

@vertex fn vs_water_t(@builtin(vertex_index) vi: u32) -> WaterT {
    let ij = grid_vertex(vi);
    let c = cell_at(ij.x, ij.y);
    let wet = c.y > T.s.y;
    var z = c.x - 0.05 * T.n.z;
    if (wet) { z = c.x + c.y * T.n.w; }
    let s0 = select(c.x, z, wet);
    let dx = (surface(ij.x + 1, ij.y, s0) - surface(ij.x - 1, ij.y, s0)) / (2.0 * T.n.z);
    let dy = (surface(ij.x, ij.y + 1, s0) - surface(ij.x, ij.y - 1, s0)) / (2.0 * T.n.z);
    var o: WaterT;
    o.world = place(ij, z);
    o.pos = project(o.world);
    o.n = normalize(vec3<f32>(-dx, -dy, 1.0));
    o.depth = select(-T.s.y, c.y, wet);
    o.speed = c.z;
    // over land that is normally dry (above the sea): flood water, brown with mud
    o.land = smoothstep(0.3, 1.5, (c.x - T.s.x) / max(T.n.w, 1e-6));
    // how far the surface stands above (or below) the still sea, metres
    o.rise = (z - T.s.x) / max(T.n.w, 1e-6);
    return o;
}

@fragment fn fs_water_t(in: WaterT) -> @location(0) vec4<f32> {
    if (in.depth < T.s.y) {
        discard;
    }
    let n = normalize(in.n);
    var v = U.f.xyz;
    if (U.mode.x < 0.5) { v = normalize(in.world - U.eye.xyz); }
    let cosv = max(dot(-v, n), 0.0);
    let fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
    let r = reflect(v, n);
    let refl = sky(r);
    let d = max(in.depth, 0.0);
    let sea = mix(vec3<f32>(0.10, 0.45, 0.50), vec3<f32>(0.02, 0.12, 0.24), 1.0 - exp(-d / 6.0));
    let mud = mix(vec3<f32>(0.55, 0.45, 0.30), vec3<f32>(0.36, 0.28, 0.17), 1.0 - exp(-d / 1.5));
    // a wave out at sea is too gentle to see by its slope: show its height, as tsunami maps do
    // (crests turn pale turquoise, troughs deep blue)
    let up = clamp(in.rise / 3.0, -1.0, 1.0) * (1.0 - in.land);
    var seac = mix(sea, vec3<f32>(0.45, 0.85, 0.85), max(up, 0.0));
    seac = mix(seac, vec3<f32>(0.01, 0.04, 0.12), max(-up, 0.0) * 0.8);
    var body = mix(seac, mud, in.land) * (U.mode.y + U.mode.z * 0.7);
    // white water: faster than its own waves (Froude > 1), or very fast
    let froude = in.speed / sqrt(9.81 * max(d, 1e-3));
    let foam = clamp((froude - 1.4) * 0.8, 0.0, 1.0) * clamp((d - 0.1) / 0.4, 0.0, 1.0) * 0.6 + clamp((in.speed - 6.0) / 8.0, 0.0, 0.4);
    let f = clamp(foam, 0.0, 0.7);
    var col = mix(body, refl, fres);
    col = col + vec3<f32>(1.0, 0.96, 0.88) * pow(max(dot(r, U.light.xyz), 0.0), 250.0) * U.mode.z * 1.5;
    col = mix(col, vec3<f32>(0.95, 0.97, 1.0) * (U.mode.y + U.mode.z * max(n.z, 0.0)), f);
    // shallow water lets the ground show through; muddy water hides it sooner
    let a = clamp(max(1.0 - exp(-d / mix(2.5, 0.4, in.land)), f) + fres * 0.5, 0.25, 1.0);
    return vec4<f32>(min(col, vec3<f32>(1.0)) * a, a);
}
// ---- the cut-away sides of the block, like a map seen as a diorama ----
struct SideT {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) n: vec3<f32>,
    @location(2) depth: f32,
    @location(3) below: f32,
};

// which edge cell and its neighbour along the edge, and the outward normal
fn side_cells(q: u32) -> array<vec3<i32>, 3> {
    let nx = i32(T.n.x);
    let ny = i32(T.n.y);
    var k = i32(q);
    if (k < nx - 1) { return array<vec3<i32>, 3>(vec3<i32>(k, 0, 0), vec3<i32>(k + 1, 0, 0), vec3<i32>(0, -1, 0)); }
    k = k - (nx - 1);
    if (k < nx - 1) { return array<vec3<i32>, 3>(vec3<i32>(k, ny - 1, 0), vec3<i32>(k + 1, ny - 1, 0), vec3<i32>(0, 1, 0)); }
    k = k - (nx - 1);
    if (k < ny - 1) { return array<vec3<i32>, 3>(vec3<i32>(0, k, 0), vec3<i32>(0, k + 1, 0), vec3<i32>(-1, 0, 0)); }
    k = k - (ny - 1);
    return array<vec3<i32>, 3>(vec3<i32>(nx - 1, k, 0), vec3<i32>(nx - 1, k + 1, 0), vec3<i32>(1, 0, 0));
}

fn side_vertex(vi: u32, water: bool) -> SideT {
    let s = side_cells(vi / 6u);
    let k = vi % 6u;
    var along = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
    var up = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    var ij = s[0].xy;
    if (along[k] == 1u) { ij = s[1].xy; }
    let c = cell_at(ij.x, ij.y);
    // the block's base sits a little below the lowest ground
    let base = -2.0 * T.n.z;
    var lo = base;
    var hi = c.x;
    if (water) {
        lo = c.x;
        hi = select(c.x, c.x + c.y * T.n.w, c.y > T.s.y);
    }
    let z = select(lo, hi, up[k] == 1u);
    var o: SideT;
    // the sides run along the edge cells' centres, where the ground and water surfaces end
    o.world = place(ij, z);
    o.pos = project(o.world);
    o.n = vec3<f32>(f32(s[2].x), f32(s[2].y), 0.0);
    o.depth = (hi - z) / max(T.n.w, 1e-6);
    o.below = (c.x - z) / max(T.n.w, 1e-6);
    return o;
}

@vertex fn vs_skirt_g(@builtin(vertex_index) vi: u32) -> SideT { return side_vertex(vi, false); }
@vertex fn vs_skirt_w(@builtin(vertex_index) vi: u32) -> SideT { return side_vertex(vi, true); }

@fragment fn fs_skirt_g(in: SideT) -> @location(0) vec4<f32> {
    // soil near the top, then layers of rock
    let d = in.below;
    let band = 0.5 + 0.5 * sin(in.world.z * 0.9);
    var col = mix(vec3<f32>(0.42, 0.32, 0.22), vec3<f32>(0.50, 0.46, 0.42), smoothstep(1.0, 6.0, d));
    col = col * (0.88 + 0.12 * band);
    return vec4<f32>(shade(col, lit(in.n) * 0.85), 1.0);
}

@fragment fn fs_skirt_w(in: SideT) -> @location(0) vec4<f32> {
    // the water seen from the side: lighter near the surface, darker with depth
    let d = max(in.depth, 0.0);
    let col = mix(vec3<f32>(0.12, 0.42, 0.52), vec3<f32>(0.01, 0.07, 0.16), 1.0 - exp(-d / 12.0)) * (U.mode.y + U.mode.z * 0.6);
    let a = 0.88;
    return vec4<f32>(col * a, a);
}
"#;
