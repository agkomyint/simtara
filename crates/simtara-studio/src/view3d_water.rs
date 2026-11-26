//! Water drawn as one continuous liquid surface, never as particles.
//!
//! The simulation's particles only build a smooth "how full of liquid" field
//! (`simtara_fields::liquid::surface_field`, a quadratic B-spline spread over the grid). Here, on
//! the GPU, each pixel's ray is followed into that field until it crosses 0.5: that is the water
//! surface, found exactly (bisection), with its normal from the field's gradient. So there are no
//! spheres or circles, even very close up. This is how ray-marched liquid surfaces work in
//! Evan Wallace's WebGL Water, in volume renderers and in level-set tools.
//!
//! At the surface:
//! - Fresnel (Schlick, water F0 = 0.02): grazing views reflect the sky and sun.
//! - Refraction (Snell, n = 1.33): the ray bends into the water; it is followed through the water
//!   to measure how much water it crosses, and the scene behind (drawn first, without water) is
//!   looked up where the bent ray comes out.
//! - Absorption (Beer-Lambert): red is absorbed first, then green, so deep water is blue-green and
//!   shallow water nearly clear.
//!
//! Spray (droplets flying alone) is drawn as soft mist by the normal renderer.

use eframe::egui_wgpu::wgpu;

/// Convert to IEEE half precision (the texture format every GPU can filter).
fn half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((man >> 13) as u16)
}

/// The last two steps' fields: frames between steps blend from one to the next, so water moves
/// smoothly at the window's frame rate even when the simulation steps less often.
struct Field {
    dims: [u32; 3],
    tex: [wgpu::Texture; 2],
    view: [wgpu::TextureView; 2],
    /// which of the two holds the newest step
    newest: usize,
}

pub struct Water {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub composite: wgpu::RenderPipeline,
    field: Option<Field>,
    group: Option<(wgpu::BindGroup, u64, [u32; 3], usize)>,
}

impl Water {
    /// `camera` is the renderer's bind group layout (group 0).
    pub fn new(device: &wgpu::Device, camera: &wgpu::BindGroupLayout, colour: wgpu::TextureFormat, depth: wgpu::TextureFormat, samples: u32, shader_head: &str) -> Water {
        let src = format!("{shader_head}\n{WATER_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("simtara water"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let tex = |binding, dim| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simtara water"),
            entries: &[
                tex(0, wgpu::TextureViewDimension::D3),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                tex(2, wgpu::TextureViewDimension::D2),
                tex(3, wgpu::TextureViewDimension::D3),
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("simtara water"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simtara water"), bind_group_layouts: &[Some(camera), Some(&layout)], immediate_size: 0 });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("simtara water surface"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_full_w"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: depth, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Less), stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_liquid"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: colour, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        });
        Water { layout, sampler, composite, field: None, group: None }
    }

    /// Upload this frame's liquid field (one value per cell, x fastest) and point the surface pass
    /// at the picture of the scene drawn without water (`scene`, identified by `scene_key`).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, values: &[f32], dims: [u32; 3], scene: &wgpu::TextureView, scene_key: u64, fresh: bool) {
        let make = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let new_size = self.field.as_ref().is_none_or(|f| f.dims != dims);
        if new_size {
            let tex = [make("simtara liquid field a"), make("simtara liquid field b")];
            let view = [tex[0].create_view(&Default::default()), tex[1].create_view(&Default::default())];
            self.field = Some(Field { dims, tex, view, newest: 0 });
            self.group = None;
        }
        if fresh || new_size {
            let f = self.field.as_mut().unwrap();
            let data: Vec<u8> = values.iter().flat_map(|v| half(*v).to_le_bytes()).collect();
            // the new step goes into the older slot; on a new size both start the same
            let slots: &[usize] = if new_size { &[0, 1] } else { &[1 - f.newest] };
            for &slot in slots {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &f.tex[slot], mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &data,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[0] * 2), rows_per_image: Some(dims[1]) },
                    wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
                );
            }
            if !new_size {
                f.newest = 1 - f.newest;
            }
        }
        let f = self.field.as_ref().unwrap();
        if self.group.as_ref().is_none_or(|g| g.1 != scene_key || g.2 != dims || g.3 != f.newest) {
            let (older, newer) = (&f.view[1 - f.newest], &f.view[f.newest]);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simtara water"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(older) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(scene) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(newer) },
                ],
            });
            self.group = Some((bg, scene_key, dims, f.newest));
        }
    }
    pub fn surface_group(&self) -> Option<&wgpu::BindGroup> {
        self.group.as_ref().map(|g| &g.0)
    }
}

/// Appended to the renderer's shader head (which defines `U`, `project` and `sky`).
const WATER_WGSL: &str = r#"
@group(1) @binding(0) var liq_tex: texture_3d<f32>;
@group(1) @binding(1) var liq_smp: sampler;
@group(1) @binding(2) var scene_tex: texture_2d<f32>;
@group(1) @binding(3) var liq_new: texture_3d<f32>;

struct FullW { @builtin(position) pos: vec4<f32> };

@vertex fn vs_full_w(@builtin(vertex_index) i: u32) -> FullW {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: FullW;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    return o;
}

// how full of liquid the point is (1 = full, the surface is at 0.5)
fn liquid(p: vec3<f32>) -> f32 {
    // blend from the older step to the newest as time passes between steps
    let uvw = p / U.vsize.xyz;
    return mix(textureSampleLevel(liq_tex, liq_smp, uvw, 0.0).r, textureSampleLevel(liq_new, liq_smp, uvw, 0.0).r, U.water.z);
}

// where a ray enters and leaves the box of the simulation
fn box_span(o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
    let ta = (vec3<f32>(0.0) - o) * inv;
    let tb = (U.vsize.xyz - o) * inv;
    let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
    let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    return vec2<f32>(t0, t1);
}

// distance along a ray to the nearest object (a big number if none): the picture of the scene
// is only looked up through the water where no object stands in front of the water there
fn object_distance(o: vec3<f32>, d: vec3<f32>) -> f32 {
    var best = 1e9;
    let count = u32(U.water.w);
    for (var i = 0u; i < count; i = i + 1u) {
        let c = U.objs[2u * i].xyz;
        let h = U.objs[2u * i + 1u].xyz;
        if (U.objs[2u * i].w < 0.5) {
            let oc = o - c;
            let b = dot(oc, d);
            let disc = b * b - (dot(oc, oc) - h.x * h.x);
            if (disc >= 0.0) {
                let t = -b - sqrt(disc);
                if (t > 0.0) { best = min(best, t); }
            }
        } else {
            let inv = 1.0 / select(d, vec3<f32>(1e-9), abs(d) < vec3<f32>(1e-9));
            let ta = (c - h - o) * inv;
            let tb = (c + h - o) * inv;
            let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
            let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
            if (t1 >= max(t0, 0.0)) { best = min(best, max(t0, 0.0)); }
        }
    }
    return best;
}

struct WaterOut { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };

@fragment fn fs_liquid(in: FullW) -> WaterOut {
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
    let span = box_span(o, d);
    if (span.y <= span.x) {
        discard;
    }
    let cell = U.vsize.w;
    let step = cell * 0.4;
    // march to the first point that is inside the water, then pin the surface down by bisection
    // far from any water the field is ~0: stride a whole cell there, slow down near the surface
    var t = span.x;
    var prev = span.x;
    var hit = false;
    for (var n = 0; n < 768; n = n + 1) {
        if (t > span.y) {
            break;
        }
        let f = liquid(o + d * t);
        if (f >= 0.5) {
            hit = true;
            break;
        }
        prev = t;
        t = t + select(step, cell * 1.0, f < 0.02);
    }
    if (!hit) {
        discard;
    }
    var lo = prev;
    var hi = t;
    for (var k = 0; k < 6; k = k + 1) {
        let mid = 0.5 * (lo + hi);
        if (liquid(o + d * mid) >= 0.5) { hi = mid; } else { lo = mid; }
    }
    let p = o + d * hi;
    // the surface normal points out of the water: down the field's gradient
    let e = cell * 0.6;
    let g = vec3<f32>(liquid(p + vec3<f32>(e, 0.0, 0.0)) - liquid(p - vec3<f32>(e, 0.0, 0.0)),
                      liquid(p + vec3<f32>(0.0, e, 0.0)) - liquid(p - vec3<f32>(0.0, e, 0.0)),
                      liquid(p + vec3<f32>(0.0, 0.0, e)) - liquid(p - vec3<f32>(0.0, 0.0, e)));
    var n = -normalize(g + vec3<f32>(0.0, 0.0, -1e-7));
    let v = -d;
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    // Fresnel: reflection of the sky and sun at grazing angles
    let cos_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - cos_v, 5.0);
    let reflected = sky(reflect(d, n));
    // refraction: follow the bent ray through the water to see how much water it crosses
    let bent = refract(d, n, 1.0 / 1.33);
    var thick = 0.0;
    var q = p + bent * step;
    let span2 = box_span(q, bent);
    var s = 0.0;
    let tstep = cell * 0.9;
    for (var k = 0; k < 96; k = k + 1) {
        if (s > span2.y || liquid(q + bent * s) < 0.5) {
            break;
        }
        s = s + tstep;
    }
    thick = s + step;
    let out = p + bent * thick;
    let cq = project(out);
    var uv = vec2<f32>(cq.x / cq.w * 0.5 + 0.5, 0.5 - cq.y / cq.w * 0.5);
    // if an object stands between the eye and the water at that spot, it is not behind the water:
    // look straight through instead
    let look = select(normalize(out - U.eye.xyz), U.f.xyz, U.mode.x > 0.5);
    let look_from = select(U.eye.xyz, out - U.f.xyz * dot(out - U.eye.xyz, U.f.xyz), U.mode.x > 0.5);
    if (object_distance(look_from, look) < length(p - look_from)) {
        uv = in.pos.xy / U.vp.xy;
    }
    let behind = textureSampleLevel(scene_tex, liq_smp, clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0).rgb;
    // Beer-Lambert absorption, and light scattered back out of the water body
    let absorb = vec3<f32>(2.0, 0.75, 0.5);
    let through = exp(-absorb * thick);
    let body = vec3<f32>(0.03, 0.26, 0.34) * (U.mode.y + U.mode.z * max(dot(n, U.light.xyz), 0.0));
    let inside = behind * through + body * (vec3<f32>(1.0) - through);
    var w: WaterOut;
    w.colour = vec4<f32>(min(mix(inside, reflected, fresnel), vec3<f32>(1.0)), 1.0);
    let c = project(p);
    w.depth = clamp(c.z / c.w, 0.0, 1.0);
    return w;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn half_floats_round_trip_the_values_the_field_uses() {
        for (x, bits) in [(0.0f32, 0u16), (1.0, 0x3c00), (0.5, 0x3800), (2.0, 0x4000)] {
            assert_eq!(super::half(x), bits, "{x}");
        }
    }
}
