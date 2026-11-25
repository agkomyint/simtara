//! A small 3D picture of a universe: orbit camera, one light, shapes and `.obj` models.
//!
//! It is drawn on the CPU through egui (no GPU 3D pipeline yet), back to front, which is plenty for
//! thousands of things and models with a few thousand triangles. World coordinates are z-up.

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

#[allow(clippy::too_many_arguments)]
pub fn draw(ui: &egui::Ui, rect: egui::Rect, resp: &egui::Response, items: &[Item], cam: &mut Camera, light: &Light, models: &HashMap<String, Mesh>, selected: Option<u64>) -> Out {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_gray(14));
    if resp.dragged() {
        let d = resp.drag_delta();
        cam.yaw -= d.x as f64 * 0.4;
        cam.pitch = (cam.pitch + d.y as f64 * 0.4).clamp(-89.0, 89.0);
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            cam.distance = (cam.distance * (1.0 - scroll as f64 * 0.0015)).clamp(0.05, 50.0);
        }
    }

    // the scene: its box, middle and size
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for it in items {
        for k in 0..3 {
            lo[k] = lo[k].min(it.pos[k]);
            hi[k] = hi[k].max(it.pos[k]);
        }
    }
    if items.is_empty() {
        return Out { hovered: None, selected_at: None };
    }
    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let extent = span.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let dims = span.iter().filter(|s| **s > extent * 1e-3).count().max(1) as f32;
    let n = items.len() as f32;
    let spacing = extent / n.powf(1.0 / dims).max(1.0);
    let unit = (spacing * 0.45).max(extent * 0.004);
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
    let lit = |n: [f32; 3]| light.ambient as f32 + light.strength as f32 * dot(n, l).max(0.0);

    // ground and grid, at the lowest thing
    let z0 = lo[2] - unit;
    if light.ground || light.grid {
        let pad = unit * 2.0;
        let (x0, x1, y0, y1) = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad);
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

    // little axes in the corner: red x, green y, blue z
    let o = rect.left_bottom() + egui::vec2(38.0, -38.0);
    for (axis, col, name) in [([1.0f32, 0.0, 0.0], Color32::from_rgb(230, 80, 80), "x"), ([0.0, 1.0, 0.0], Color32::from_rgb(90, 200, 90), "y"), ([0.0, 0.0, 1.0], Color32::from_rgb(90, 140, 240), "z")] {
        let tip = o + egui::vec2(dot(axis, r), -dot(axis, u)) * 24.0;
        painter.line_segment([o, tip], egui::Stroke::new(1.6, col));
        painter.text(tip + (tip - o) * 0.25, egui::Align2::CENTER_CENTER, name, egui::FontId::monospace(10.0), col);
    }
    Out { hovered: hovered.map(|h| h.2), selected_at }
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
    fn built_in_shapes_exist() {
        assert!(builtin("cube").is_some() && builtin("pyramid").is_some() && builtin("nope").is_none());
    }
}
