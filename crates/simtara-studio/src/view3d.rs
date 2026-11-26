//! A small 3D picture of a universe: orbit camera, one light, shapes and `.obj` models.
//!
//! Drawn on the GPU by `view3d_gpu` when there is a wgpu device, otherwise on the CPU through
//! egui, back to front (fine for thousands of things). Both share the camera maths here, and the
//! mouse picking. World coordinates are z-up.

use eframe::egui::{self, Color32, Pos2};
use simtara_things::{Camera, Light};
use std::collections::HashMap;

#[derive(Clone, Default, Debug)]
pub struct Mesh {
    pub v: Vec<[f32; 3]>,
    pub t: Vec<[u32; 3]>,
}

impl Mesh {
    /// Centre the model on its middle, with its longest side 1 long.
    fn normalise(&mut self) {
        if self.v.is_empty() {
            return;
        }
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &self.v {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let side = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max).max(1e-9);
        for p in &mut self.v {
            // centred in x and y, standing on z = 0 so a model sits on the ground at its position
            p[0] = (p[0] - (lo[0] + hi[0]) / 2.0) / side;
            p[1] = (p[1] - (lo[1] + hi[1]) / 2.0) / side;
            p[2] = (p[2] - (lo[2] + hi[2]) / 2.0) / side;
        }
    }
}

/// Read a Wavefront `.obj` (vertices and faces; faces with more than 3 corners are split).
/// Models are z-up like the universe; a line `# simtara up=y` marks a y-up model.
pub fn parse_obj(text: &str) -> Result<Mesh, String> {
    let y_up = text.lines().take(20).any(|l| l.contains("simtara") && l.contains("up=y"));
    let mut m = Mesh::default();
    for (n, line) in text.lines().enumerate() {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("v") => {
                let c: Vec<f32> = w.take(3).map(|s| s.parse::<f32>().map_err(|_| format!("line {}: bad number '{s}'", n + 1))).collect::<Result<_, _>>()?;
                if c.len() < 3 {
                    return Err(format!("line {}: a vertex needs x y z", n + 1));
                }
                m.v.push(if y_up { [c[0], -c[2], c[1]] } else { [c[0], c[1], c[2]] });
            }
            Some("f") => {
                let mut idx = Vec::new();
                for s in w {
                    let i: i64 = s.split('/').next().unwrap_or("").parse().map_err(|_| format!("line {}: bad face '{s}'", n + 1))?;
                    let i = if i < 0 { m.v.len() as i64 + i } else { i - 1 };
                    if i < 0 || i as usize >= m.v.len() {
                        return Err(format!("line {}: face uses a vertex that does not exist", n + 1));
                    }
                    idx.push(i as u32);
                }
                for k in 1..idx.len().saturating_sub(1) {
                    m.t.push([idx[0], idx[k], idx[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if m.t.is_empty() {
        return Err("no faces found (is this an .obj file?)".into());
    }
    m.normalise();
    Ok(m)
}

pub fn builtin(name: &str) -> Option<Mesh> {
    let mut m = match name {
        "cube" | "box" => Mesh {
            v: vec![[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]],
            t: vec![[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        },
        "pyramid" | "cone" => Mesh {
            v: vec![[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [0., 0., 1.]],
            t: vec![[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        },
        _ => return None,
    };
    m.normalise();
    Some(m)
}

pub struct Item {
    pub id: u64,
    pub pos: [f32; 3],
    pub color: Color32,
    /// 1 = normal size; the picture makes it bigger or smaller.
    pub size: f32,
    /// Degrees around the up axis.
    pub heading: f32,
    pub shape: String,
}

pub struct Out {
    pub hovered: Option<u64>,
    pub selected_at: Option<Pos2>,
}

/// A box of smoke, heat and solids to draw see-through (from `simtara_fields::Fluid`).
pub struct Volume {
    /// Cells along x, y, z.
    pub dims: [u32; 3],
    /// Size of the box in world units (metres), from the origin.
    pub size: [f32; 3],
    /// RGBA per cell, x fastest: r = smoke, g = heat, b = solid.
    pub data: Vec<u8>,
    /// Water particles (shape `water`) are drawn as one surface with spheres of this radius (GPU).
    pub water_radius: f32,
    /// Light focused on the floor by the water surface: width, height, brightness/3 (`Fluid::caustics`).
    pub caustics: Option<(u32, u32, Vec<u8>)>,
    /// How full of liquid each cell is (`Fluid::liquid_field`): drawn as one surface on the GPU.
    pub liquid: Option<Vec<f32>>,
    /// Changes when the contents change (0 = always upload).
    pub generation: u64,
    /// For smooth motion: how far this frame is from the previous step to this one (0..1).
    pub blend: f32,
    /// Objects moving in the liquid: shape, centre and half sizes (metres).
    pub objects: Vec<(String, [f32; 3], [f32; 3])>,
    /// The object the keys steer (drawn with a glow).
    pub steered: Option<usize>,
}

impl Volume {
    fn texel(&self, i: i64, j: i64, k: i64) -> [f32; 3] {
        let d = self.dims.map(|x| x as i64);
        let (i, j, k) = (i.clamp(0, d[0] - 1), j.clamp(0, d[1] - 1), k.clamp(0, d[2] - 1));
        let c = ((i + d[0] * (j + d[1] * k)) * 4) as usize;
        [self.data[c] as f32 / 255.0, self.data[c + 1] as f32 / 255.0, self.data[c + 2] as f32 / 255.0]
    }

    /// Trilinear sample at a world position (outside = clamped to the edge).
    pub fn sample(&self, p: [f32; 3]) -> [f32; 3] {
        let g = [p[0] / self.size[0] * self.dims[0] as f32 - 0.5, p[1] / self.size[1] * self.dims[1] as f32 - 0.5, p[2] / self.size[2] * self.dims[2] as f32 - 0.5];
        let b = g.map(f32::floor);
        let t = [g[0] - b[0], g[1] - b[1], g[2] - b[2]];
        let mut out = [0.0f32; 3];
        for c in 0..8 {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let w: f32 = (0..3).map(|a| if o[a] == 1 { t[a] } else { 1.0 - t[a] }).product();
            let v = self.texel(b[0] as i64 + o[0] as i64, b[1] as i64 + o[1] as i64, b[2] as i64 + o[2] as i64);
            for a in 0..3 {
                out[a] += w * v[a];
            }
        }
        out
    }
}

/// How smoke and fire look: shared by the CPU and GPU volume painters.
pub const SMOKE_PER_METRE: f32 = 9.0;
/// How brightly hot gas glows, per metre of it at full heat.
pub const GLOW_PER_METRE: f32 = 6.0;

/// Colour of glowing hot gas for heat 0..1 (0..600 K above the air): dark red, orange, yellow.
pub fn fire_colour(g: f32) -> [f32; 3] {
    let g = g.clamp(0.0, 1.0);
    [(g * 2.2).min(1.0), (g * 1.6 - 0.35).clamp(0.0, 1.0) * 0.85, (g * 2.0 - 1.4).clamp(0.0, 1.0) * 0.6]
}

/// An angle in degrees, brought into -180..180.
pub fn wrap_degrees(a: f64) -> f64 {
    let w = (a + 180.0).rem_euclid(360.0) - 180.0;
    if w == -180.0 { 180.0 } else { w }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

struct View {
    eye: [f32; 3],
    r: [f32; 3],
    u: [f32; 3],
    f: [f32; 3],
    focal: f32,
    ortho: bool,
    centre: Pos2,
}

impl View {
    /// Screen position, depth in front of the camera.
    fn project(&self, p: [f32; 3]) -> Option<(Pos2, f32)> {
        let rel = sub(p, self.eye);
        let (x, y, z) = (dot(rel, self.r), dot(rel, self.u), dot(rel, self.f));
        if self.ortho {
            return Some((Pos2::new(self.centre.x + x * self.focal, self.centre.y - y * self.focal), z));
        }
        if z < 0.05 {
            return None;
        }
        Some((Pos2::new(self.centre.x + x / z * self.focal, self.centre.y - y / z * self.focal), z))
    }
}

fn shade(c: Color32, k: f32) -> Color32 {
    let k = k.clamp(0.0, 1.4);
    let f = |v: u8| ((v as f32 * k).round().min(255.0)) as u8;
    Color32::from_rgb(f(c.r()), f(c.g()), f(c.b()))
}

fn vertex(pos: Pos2, color: Color32) -> egui::epaint::Vertex {
    egui::epaint::Vertex { pos, uv: egui::epaint::WHITE_UV, color }
}

/// Draw the 3D view. With `gpu` the picture is drawn by the graphics card (`view3d_gpu`), otherwise
/// by the CPU painter below. Camera, hovering and selection are the same either way.
#[allow(clippy::too_many_arguments)]
pub fn draw(ui: &egui::Ui, rect: egui::Rect, resp: &egui::Response, items: &[Item], cam: &mut Camera, light: &Light, models: &HashMap<String, Mesh>, selected: Option<u64>, gpu: Option<&mut crate::view3d_gpu::GpuView>, volume: Option<&Volume>) -> Out {
    let painter = ui.painter_at(rect);
    if gpu.is_none() {
        painter.rect_filled(rect, 4.0, Color32::from_gray(14));
    }
    if resp.dragged() {
        let d = resp.drag_delta();
        // turning wraps around, so dragging can go round and round
        cam.yaw = wrap_degrees(cam.yaw - d.x as f64 * 0.4);
        cam.pitch = (cam.pitch + d.y as f64 * 0.4).clamp(-89.0, 89.0);
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            cam.distance = (cam.distance * (1.0 - scroll as f64 * 0.0015)).clamp(0.05, 50.0);
        }
    }

    let Some(sc) = scene(rect, items, cam, light, volume) else {
        if gpu.is_some() {
            painter.rect_filled(rect, 4.0, Color32::from_gray(14));
        }
        return Out { hovered: None, selected_at: None };
    };
    let Scene { view, uniforms: u_, unit, focal, ortho, lo, hi, l, z0, ground: [x0, y0, x1, y1] } = sc;
    let (r, u) = (view.r, view.u);
    let (ly, lp) = ((light.yaw as f32).to_radians(), (light.pitch as f32).to_radians());
    let lit = |n: [f32; 3]| light.ambient as f32 + light.strength as f32 * dot(n, l).max(0.0);
    let _ = (lo, hi);

    if let Some(gpu) = gpu {
        gpu.paint(&painter, rect, ui.ctx().pixels_per_point(), &u_, items, models, volume);
        let (hovered, selected_at) = pick(&view, rect, resp, items, unit, focal, ortho, selected);
        axes(&painter, rect, r, u);
        return Out { hovered, selected_at };
    }

    if light.ground || light.grid {
        if light.ground {
            let corners: Vec<Pos2> = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].iter().filter_map(|c| view.project([c[0], c[1], z0]).map(|p| p.0)).collect();
            if corners.len() == 4 {
                painter.add(egui::Shape::convex_polygon(corners, Color32::from_rgb(22, 26, 30), egui::Stroke::NONE));
            }
        }
        if light.grid {
            let stroke = egui::Stroke::new(0.6, Color32::from_white_alpha(22));
            let steps = 10;
            for i in 0..=steps {
                let a = i as f32 / steps as f32;
                let (x, y) = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a);
                if let (Some(p), Some(q)) = (view.project([x, y0, z0]), view.project([x, y1, z0])) {
                    painter.line_segment([p.0, q.0], stroke);
                }
                if let (Some(p), Some(q)) = (view.project([x0, y, z0]), view.project([x1, y, z0])) {
                    painter.line_segment([p.0, q.0], stroke);
                }
            }
        }
    }

    // project every thing, then draw far to near
    struct Seen {
        i: usize,
        at: Pos2,
        depth: f32,
        px: f32,
    }
    let mut seen: Vec<Seen> = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        let Some((at, depth)) = view.project(it.pos) else { continue };
        let world_r = unit * it.size;
        let px = if ortho { world_r * focal } else { world_r / depth * focal };
        if !rect.expand(px + 4.0).contains(at) {
            continue;
        }
        seen.push(Seen { i, at, depth, px });
    }
    seen.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

    let big = seen.len() > 20_000;
    let mut tri_budget = 350_000usize;
    let mut mesh = egui::Mesh::default();
    let mut hovered: Option<(f32, f32, u64)> = None;
    let hover = resp.hover_pos();
    let mut selected_at = None;

    for s in &seen {
        let it = &items[s.i];
        if Some(it.id) == selected {
            selected_at = Some(s.at);
        }
        if let Some(hp) = hover {
            let d = s.at.distance(hp);
            if d < s.px.max(3.0) + 4.0 && hovered.is_none_or(|(bd, bz, _)| d < bd - 0.5 || (d < bd + 0.5 && s.depth < bz)) {
                hovered = Some((d, s.depth, it.id));
            }
        }
        let mesh_of: Option<&Mesh> = match it.shape.as_str() {
            "" | "sphere" | "ball" | "point" | "dot" => None,
            name => models.get(name),
        };
        match mesh_of {
            Some(m) if !big && s.px > 3.0 && tri_budget > m.t.len() => {
                tri_budget -= m.t.len();
                let (sh, ch) = (it.heading.to_radians().sin(), it.heading.to_radians().cos());
                let size = unit * it.size * 2.0;
                let world: Vec<[f32; 3]> = m.v.iter().map(|p| [it.pos[0] + (p[0] * ch - p[1] * sh) * size, it.pos[1] + (p[0] * sh + p[1] * ch) * size, it.pos[2] + p[2] * size]).collect();
                let proj: Vec<Option<(Pos2, f32)>> = world.iter().map(|p| view.project(*p)).collect();
                let mut tris: Vec<(f32, [usize; 3], f32)> = Vec::new();
                for t in &m.t {
                    let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                    if proj[a].is_none() || proj[b].is_none() || proj[c].is_none() {
                        continue;
                    }
                    let mut n = norm(cross(sub(world[b], world[a]), sub(world[c], world[a])));
                    // light both sides of a face the same, so models with mixed winding still look right
                    if dot(n, sub(view.eye, world[a])) < 0.0 {
                        n = [-n[0], -n[1], -n[2]];
                    }
                    let depth = (proj[a].unwrap().1 + proj[b].unwrap().1 + proj[c].unwrap().1) / 3.0;
                    tris.push((depth, [a, b, c], lit(n)));
                }
                tris.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
                for (_, [a, b, c], k) in tris {
                    let col = shade(it.color, k);
                    let base = mesh.vertices.len() as u32;
                    mesh.vertices.push(vertex(proj[a].unwrap().0, col));
                    mesh.vertices.push(vertex(proj[b].unwrap().0, col));
                    mesh.vertices.push(vertex(proj[c].unwrap().0, col));
                    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
                }
            }
            _ => {
                let px = s.px.max(0.8);
                if big || px < 2.0 {
                    mesh.add_colored_rect(egui::Rect::from_center_size(s.at, egui::vec2(px * 2.0, px * 2.0)), shade(it.color, lit([0.0, 0.0, 1.0])));
                } else {
                    // a ball: lighter in the middle (towards the light), darker at the rim
                    let seg = if px > 12.0 { 20 } else { 12 };
                    let up_light = lp.sin() * 0.45;
                    let hi_off = egui::vec2(px * 0.25 * (-ly.cos()).signum() * lp.cos(), -px * 0.3 * (1.0 + up_light));
                    let base = mesh.vertices.len() as u32;
                    mesh.vertices.push(vertex(s.at + hi_off, shade(it.color, lit([0.0, 0.0, 1.0]) * 1.25)));
                    for k in 0..seg {
                        let a = k as f32 / seg as f32 * std::f32::consts::TAU;
                        mesh.vertices.push(vertex(s.at + egui::vec2(a.cos(), a.sin()) * px, shade(it.color, lit([0.0, 0.0, 1.0]) * 0.55)));
                    }
                    for k in 0..seg as u32 {
                        mesh.indices.extend_from_slice(&[base, base + 1 + k, base + 1 + (k + 1) % seg as u32]);
                    }
                }
            }
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    if let Some(vol) = volume {
        let img = cpu_volume(&view, rect, vol, light, ortho, focal);
        let id = egui::Id::unique("simtara cpu volume");
        let tex = match ui.ctx().data(|d| d.get_temp::<egui::TextureHandle>(id)) {
            Some(mut t) => {
                t.set(img, egui::TextureOptions::LINEAR);
                t
            }
            None => {
                let t = ui.ctx().load_texture("simtara cpu volume", img, egui::TextureOptions::LINEAR);
                ui.ctx().data_mut(|d| d.insert_temp(id, t.clone()));
                t
            }
        };
        painter.image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
    }
    axes(&painter, rect, r, u);
    Out { hovered: hovered.map(|h| h.2), selected_at }
}

/// Ray-march the volume on the CPU at a low resolution (the GPU path does the same per pixel).
fn cpu_volume(view: &View, rect: egui::Rect, vol: &Volume, light: &Light, ortho: bool, focal: f32) -> egui::ColorImage {
    use rayon::prelude::*;
    let scale = (170.0 / rect.width().max(1.0)).min(1.0);
    let (w, h) = (((rect.width() * scale) as usize).max(1), ((rect.height() * scale) as usize).max(1));
    let (ly, lp) = ((light.yaw as f32).to_radians(), (light.pitch as f32).to_radians());
    let l = [lp.cos() * ly.cos(), lp.cos() * ly.sin(), lp.sin()];
    let cell = (0..3).map(|a| vol.size[a] / vol.dims[a] as f32).fold(f32::MAX, f32::min);
    let pixels: Vec<Color32> = (0..w * h)
        .into_par_iter()
        .map(|px| {
            let (x, y) = ((px % w) as f32 + 0.5, (px / w) as f32 + 0.5);
            // screen offset from the centre in points, as in View::project
            let (dx, dy) = ((x / scale - rect.width() / 2.0) / focal, -(y / scale - rect.height() / 2.0) / focal);
            let (o, d) = if ortho {
                ([view.eye[0] + view.r[0] * dx + view.u[0] * dy, view.eye[1] + view.r[1] * dx + view.u[1] * dy, view.eye[2] + view.r[2] * dx + view.u[2] * dy], view.f)
            } else {
                (view.eye, norm([view.f[0] + view.r[0] * dx + view.u[0] * dy, view.f[1] + view.r[1] * dx + view.u[1] * dy, view.f[2] + view.r[2] * dx + view.u[2] * dy]))
            };
            let rgba = march(vol, o, d, l, light, cell);
            Color32::from_rgba_premultiplied((rgba[0] * 255.0) as u8, (rgba[1] * 255.0) as u8, (rgba[2] * 255.0) as u8, (rgba[3] * 255.0) as u8)
        })
        .collect();
    egui::ColorImage { size: [w, h], pixels, source_size: egui::vec2(w as f32, h as f32) }
}

/// One ray through the volume: premultiplied colour and coverage.
pub(crate) fn march(vol: &Volume, o: [f32; 3], d: [f32; 3], l: [f32; 3], light: &Light, cell: f32) -> [f32; 4] {
    let (mut t0, mut t1) = (0.0f32, f32::MAX);
    for a in 0..3 {
        let inv = 1.0 / if d[a].abs() < 1e-9 { 1e-9 } else { d[a] };
        let (mut ta, mut tb) = ((0.0 - o[a]) * inv, (vol.size[a] - o[a]) * inv);
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
    }
    if t1 <= t0 {
        return [0.0; 4];
    }
    // the same steps and light samples as the GPU shader, so both pictures match
    let step = cell * 0.75;
    let (mut col, mut trans) = ([0.0f32; 3], 1.0f32);
    let mut t = t0 + step * 0.5;
    let at = |t: f32| [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    while t < t1 && trans > 0.01 {
        let p = at(t);
        let s = vol.sample(p);
        if s[2] > 0.5 {
            let e = cell;
            let g = |a: usize| {
                let (mut q1, mut q0) = (p, p);
                q1[a] += e;
                q0[a] -= e;
                vol.sample(q0)[2] - vol.sample(q1)[2]
            };
            let n = norm([g(0), g(1), g(2)]);
            let k = (light.ambient as f32 + light.strength as f32 * dot(n, l).max(0.0)).min(1.4);
            for a in 0..3 {
                col[a] += trans * [0.55, 0.57, 0.6][a] * k;
            }
            trans = 0.0;
            break;
        }
        let dens = s[0] * SMOKE_PER_METRE;
        if dens > 1e-4 || s[1] > 0.02 {
            let alpha = 1.0 - (-dens * step).exp();
            // light reaching this point through the smoke towards the light
            let mut shade = 0.0;
            for k in 1..=4 {
                let r = cell * 1.5 * k as f32;
                shade += vol.sample([p[0] + l[0] * r, p[1] + l[1] * r, p[2] + l[2] * r])[0];
            }
            let lit = light.ambient as f32 + light.strength as f32 * (-shade * SMOKE_PER_METRE * cell * 1.5).exp();
            let fire = fire_colour(s[1]);
            let glow = s[1] * s[1] * step * GLOW_PER_METRE;
            for a in 0..3 {
                col[a] += trans * (alpha * [0.82, 0.84, 0.88][a] * lit + glow * fire[a]);
            }
            trans *= 1.0 - alpha;
        }
        t += step;
    }
    let a = 1.0 - trans;
    [col[0].min(1.0), col[1].min(1.0), col[2].min(1.0), a]
}

/// Everything the camera and light give for one picture: shared by the CPU painter, the GPU
/// renderer, and pictures made without a window (`render_offscreen`).
pub struct Scene {
    view: View,
    pub uniforms: crate::view3d_gpu::Uniforms,
    unit: f32,
    focal: f32,
    ortho: bool,
    lo: [f32; 3],
    hi: [f32; 3],
    l: [f32; 3],
    z0: f32,
    ground: [f32; 4],
}

/// Work out the camera for a picture of `rect` (None when there is nothing to show).
pub fn scene(rect: egui::Rect, items: &[Item], cam: &Camera, light: &Light, volume: Option<&Volume>) -> Option<Scene> {    // the scene: its box, middle and size
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for it in items {
        for k in 0..3 {
            lo[k] = lo[k].min(it.pos[k]);
            hi[k] = hi[k].max(it.pos[k]);
        }
    }
    if let Some(v) = volume {
        for k in 0..3 {
            lo[k] = lo[k].min(0.0);
            hi[k] = hi[k].max(v.size[k]);
        }
    }
    if items.is_empty() && volume.is_none() {
        return None;
    }
    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let extent = span.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let dims = span.iter().filter(|s| **s > extent * 1e-3).count().max(1) as f32;
    let n = items.len() as f32;
    let spacing = extent / n.powf(1.0 / dims).max(1.0);
    let unit = if items.is_empty() { extent * 0.01 } else { (spacing * 0.45).max(extent * 0.004) };
    let mut centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    if cam.follow != 0 {
        if let Some(it) = items.iter().find(|i| i.id == cam.follow) {
            centre = it.pos;
        }
    }
    let radius = (span[0] * span[0] + span[1] * span[1] + span[2] * span[2]).sqrt().max(extent) / 2.0;

    let (yaw, pitch) = ((cam.yaw as f32).to_radians(), (cam.pitch as f32).to_radians());
    let dir = [pitch.cos() * yaw.cos(), pitch.cos() * yaw.sin(), pitch.sin()];
    let half = rect.width().min(rect.height()) * 0.5;
    let ortho = cam.fov < 1.0;
    let (dist, focal) = if ortho {
        (radius * 3.0, half / (radius * cam.distance as f32).max(1e-6))
    } else {
        let t = (cam.fov as f32 / 2.0).to_radians().tan();
        (radius / (cam.fov as f32 / 2.0).to_radians().sin() * cam.distance as f32 * 0.95, half / t)
    };
    let eye = [centre[0] + dir[0] * dist, centre[1] + dir[1] * dist, centre[2] + dir[2] * dist];
    let f = norm(sub(centre, eye));
    let r = norm(cross(f, [0.0, 0.0, 1.0]));
    let u = cross(r, f);
    let view = View { eye, r, u, f, focal, ortho, centre: rect.center() };

    // light
    let (ly, lp) = ((light.yaw as f32).to_radians(), (light.pitch as f32).to_radians());
    let l = [lp.cos() * ly.cos(), lp.cos() * ly.sin(), lp.sin()];

    // ground and grid, at the lowest thing
    let z0 = lo[2] - unit;
    let pad = unit * 2.0;
    let (x0, x1, y0, y1) = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad);

        let reach = radius * 4.0 + unit * 4.0;
        let u_ = crate::view3d_gpu::Uniforms {
            eye,
            r,
            u,
            f,
            sx: focal / (rect.width() / 2.0).max(1.0),
            sy: focal / (rect.height() / 2.0).max(1.0),
            near: (dist - reach).max(radius.max(unit) * 1e-4),
            far: dist + reach,
            ortho,
            ambient: light.ambient as f32,
            strength: light.strength as f32,
            unit,
            light: l,
            ground: [x0, y0, x1, y1],
            ground_z: z0,
            ground_on: light.ground,
            grid_on: light.grid,
            viewport: [0.0; 2],
            volume: volume.map(|v| (v.size, v.dims)),
            water_radius: volume.map_or(0.0, |v| v.water_radius),
            pool: volume.is_some_and(|v| v.water_radius > 0.0),
            blend: volume.map_or(1.0, |v| v.blend),
            objects: [(true, [0.0; 3], [0.0; 3]); 4],
            n_objects: 0,
            steered: None,
        };
    Some(Scene { view, uniforms: u_, unit, focal, ortho, lo, hi, l, z0, ground: [x0, y0, x1, y1] })
}
/// Little axes in the corner: red x, green y, blue z.
fn axes(painter: &egui::Painter, rect: egui::Rect, r: [f32; 3], u: [f32; 3]) {
    let o = rect.left_bottom() + egui::vec2(38.0, -38.0);
    for (axis, col, name) in [([1.0f32, 0.0, 0.0], Color32::from_rgb(230, 80, 80), "x"), ([0.0, 1.0, 0.0], Color32::from_rgb(90, 200, 90), "y"), ([0.0, 0.0, 1.0], Color32::from_rgb(90, 140, 240), "z")] {
        let tip = o + egui::vec2(dot(axis, r), -dot(axis, u)) * 24.0;
        painter.line_segment([o, tip], egui::Stroke::new(1.6, col));
        painter.text(tip + (tip - o) * 0.25, egui::Align2::CENTER_CENTER, name, egui::FontId::monospace(10.0), col);
    }
}

/// Which thing is under the mouse, and where the selected one is on screen (GPU path; the CPU
/// painter does the same while it draws).
#[allow(clippy::too_many_arguments)]
fn pick(view: &View, rect: egui::Rect, resp: &egui::Response, items: &[Item], unit: f32, focal: f32, ortho: bool, selected: Option<u64>) -> (Option<u64>, Option<Pos2>) {
    let hover = resp.hover_pos();
    let mut hovered: Option<(f32, f32, u64)> = None;
    let mut selected_at = None;
    for it in items {
        let Some((at, depth)) = view.project(it.pos) else { continue };
        if Some(it.id) == selected {
            selected_at = Some(at);
        }
        let Some(hp) = hover else { continue };
        let world_r = unit * it.size;
        let px = if ortho { world_r * focal } else { world_r / depth * focal };
        if !rect.expand(px + 4.0).contains(at) {
            continue;
        }
        let d = at.distance(hp);
        // nearest to the camera wins among things under the mouse
        if d < px.max(3.0) + 4.0 && hovered.is_none_or(|(bd, bz, _)| depth < bz || (depth == bz && d < bd)) {
            hovered = Some((d, depth, it.id));
        }
    }
    (hovered.map(|h| h.2), selected_at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_obj_is_read_centred_and_sized_to_one() {
        let m = parse_obj("v 0 0 0\nv 4 0 0\nv 0 2 0\nv 0 0 1\nf 1 2 3\nf 1/1/1 2/2/2 4/3/3 3\n").unwrap();
        assert_eq!(m.t.len(), 3, "a quad is split in two");
        let max = m.v.iter().flat_map(|p| p.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
        assert!((max - 0.5).abs() < 1e-6, "longest side is 1, centred: {max}");
    }

    #[test]
    fn a_broken_obj_says_what_is_wrong() {
        assert!(parse_obj("v 1 2\nf 1 2 3").unwrap_err().contains("line 1"));
        assert!(parse_obj("v 0 0 0\nf 1 2 3").unwrap_err().contains("does not exist"));
        assert!(parse_obj("hello").is_err());
    }

    #[test]
    fn turning_wraps_around() {
        assert_eq!(wrap_degrees(190.0), -170.0);
        assert_eq!(wrap_degrees(-725.0), -5.0);
        assert_eq!(wrap_degrees(180.0), 180.0);
        assert_eq!(wrap_degrees(-180.0), 180.0);
        assert_eq!(wrap_degrees(35.0), 35.0);
    }

    #[test]
    fn built_in_shapes_exist() {
        assert!(builtin("cube").is_some() && builtin("pyramid").is_some() && builtin("nope").is_none());
    }
}
