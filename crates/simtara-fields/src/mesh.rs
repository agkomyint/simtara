//! A real triangle mesh of a liquid's surface, built from its particles, for export
//! (Blender, Maya, Houdini, ParaView ...) and for drawing.
//!
//! 1. A smooth distance-to-the-surface field from the particles, after Zhu & Bridson (2005,
//!    "Animating sand as a fluid"): at each point, the weighted average position and radius of the
//!    particles nearby; the distance is |point - average| - radius. It is smoother than spheres
//!    around each particle and keeps flat surfaces flat.
//! 2. Marching tetrahedra on a grid finer than the simulation's (each cube split into six
//!    tetrahedra): where the field changes sign, a triangle. Corners shared by cubes are shared
//!    by triangles, so the mesh is closed and watertight.
//! 3. Taubin smoothing (shrink-free), as surface tools such as splashsurf do, to remove the
//!    small bumps of single particles.
//!
//! Positions are in metres, z up. Deterministic.

use crate::grid::Dims;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// Distance field from particles (grid units) sampled on nodes `step` apart (grid units).
/// Positive outside the liquid, negative inside.
fn field(particles: &[[f32; 6]], d: Dims, step: f32) -> (Vec<f32>, [usize; 3]) {
    // particles binned by simulation cell (counting sort: same order every run)
    let cells = d.cells();
    let cell_of = |p: &[f32; 6]| d.c((p[0].max(0.0) as usize).min(d.nx - 1), (p[1].max(0.0) as usize).min(d.ny - 1), (p[2].max(0.0) as usize).min(d.nz - 1));
    let mut start = vec![0u32; cells + 1];
    for p in particles {
        start[cell_of(p) + 1] += 1;
    }
    for c in 0..cells {
        start[c + 1] += start[c];
    }
    let mut fill = start.clone();
    let mut order = vec![0u32; particles.len()];
    for (i, p) in particles.iter().enumerate() {
        let c = cell_of(p);
        order[fill[c] as usize] = i as u32;
        fill[c] += 1;
    }
    // kernel radius: one cell (two particle spacings); particle radius a little over half a spacing
    let reach = 1.0f32;
    let radius = 0.32f32;
    // one extra node outside the box on every side, always outside the liquid, so water against
    // the tank walls is closed off there too (the mesh is watertight)
    let inner = [(d.nx as f32 / step).round() as usize + 1, (d.ny as f32 / step).round() as usize + 1, (d.nz as f32 / step).round() as usize + 1];
    let n = [inner[0] + 2, inner[1] + 2, inner[2] + 2];
    let far = reach;
    let phi: Vec<f32> = (0..n[0] * n[1] * n[2])
        .into_par_iter()
        .map(|g| {
            let (i, j, k) = (g % n[0], (g / n[0]) % n[1], g / (n[0] * n[1]));
            if i == 0 || j == 0 || k == 0 || i == n[0] - 1 || j == n[1] - 1 || k == n[2] - 1 {
                return far;
            }
            let x = [(i as f32 - 1.0) * step, (j as f32 - 1.0) * step, (k as f32 - 1.0) * step];
            let (ci, cj, ck) = (x[0] as isize, x[1] as isize, x[2] as isize);
            let (mut wsum, mut avg) = (0.0f32, [0.0f32; 3]);
            for kk in ck - 1..=ck + 1 {
                for jj in cj - 1..=cj + 1 {
                    for ii in ci - 1..=ci + 1 {
                        if ii < 0 || jj < 0 || kk < 0 || ii >= d.nx as isize || jj >= d.ny as isize || kk >= d.nz as isize {
                            continue;
                        }
                        let c = d.c(ii as usize, jj as usize, kk as usize);
                        for &o in &order[start[c] as usize..start[c + 1] as usize] {
                            let p = &particles[o as usize];
                            let dd = (p[0] - x[0]).powi(2) + (p[1] - x[1]).powi(2) + (p[2] - x[2]).powi(2);
                            let s = dd / (reach * reach);
                            if s >= 1.0 {
                                continue;
                            }
                            let w = (1.0 - s).powi(3);
                            wsum += w;
                            avg[0] += w * p[0];
                            avg[1] += w * p[1];
                            avg[2] += w * p[2];
                        }
                    }
                }
            }
            if wsum <= 1e-9 {
                return far;
            }
            let a = [avg[0] / wsum, avg[1] / wsum, avg[2] / wsum];
            ((x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2) + (x[2] - a[2]).powi(2)).sqrt() - radius
        })
        .collect();
    (phi, n)
}

/// Cube corners and the six tetrahedra around the diagonal 0-6.
const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS: [[usize; 4]; 6] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

/// The liquid surface as a closed triangle mesh. `detail` = grid nodes per simulation cell
/// (2 is a good default: twice as fine as the simulation). `cell` = cell size in metres.
pub fn surface(particles: &[[f32; 6]], d: Dims, cell: f32, detail: usize) -> Mesh {
    if particles.is_empty() {
        return Mesh::default();
    }
    let step = 1.0 / detail.max(1) as f32;
    let (phi, n) = field(particles, d, step);
    let node = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
    // triangles per z-slab of cubes, with vertices named by the grid edge they sit on
    type Tri = [(u32, u32); 3];
    let slabs: Vec<Vec<(Tri, [[f32; 3]; 3])>> = (0..n[2] - 1)
        .into_par_iter()
        .map(|k| {
            let mut out = Vec::new();
            for j in 0..n[1] - 1 {
                for i in 0..n[0] - 1 {
                    let ids: Vec<usize> = CORNERS.iter().map(|c| node(i + c[0], j + c[1], k + c[2])).collect();
                    let vals: Vec<f32> = ids.iter().map(|&g| phi[g]).collect();
                    if vals.iter().all(|v| *v > 0.0) || vals.iter().all(|v| *v <= 0.0) {
                        continue;
                    }
                    let pos = |c: usize| [((i + CORNERS[c][0]) as f32 - 1.0) * step, ((j + CORNERS[c][1]) as f32 - 1.0) * step, ((k + CORNERS[c][2]) as f32 - 1.0) * step];
                    for t in TETS {
                        let inside: Vec<usize> = t.iter().copied().filter(|&c| vals[c] <= 0.0).collect();
                        let outside: Vec<usize> = t.iter().copied().filter(|&c| vals[c] > 0.0).collect();
                        if inside.is_empty() || outside.is_empty() {
                            continue;
                        }
                        let cut = |a: usize, b: usize| -> ((u32, u32), [f32; 3]) {
                            let (va, vb) = (vals[a], vals[b]);
                            let s = (va / (va - vb)).clamp(0.0, 1.0);
                            let (pa, pb) = (pos(a), pos(b));
                            let key = if ids[a] < ids[b] { (ids[a] as u32, ids[b] as u32) } else { (ids[b] as u32, ids[a] as u32) };
                            (key, [pa[0] + s * (pb[0] - pa[0]), pa[1] + s * (pb[1] - pa[1]), pa[2] + s * (pb[2] - pa[2])])
                        };
                        let mut tris: Vec<[((u32, u32), [f32; 3]); 3]> = Vec::new();
                        match (inside.len(), outside.len()) {
                            (1, 3) => tris.push([cut(inside[0], outside[0]), cut(inside[0], outside[1]), cut(inside[0], outside[2])]),
                            (3, 1) => tris.push([cut(outside[0], inside[0]), cut(outside[0], inside[1]), cut(outside[0], inside[2])]),
                            _ => {
                                let (a, b, c, e) = (cut(inside[0], outside[0]), cut(inside[0], outside[1]), cut(inside[1], outside[1]), cut(inside[1], outside[0]));
                                tris.push([a, b, c]);
                                tris.push([a, c, e]);
                            }
                        }
                        // face outwards: towards where the field grows
                        let grad = {
                            let mut g = [0.0f32; 3];
                            let centre: [f32; 3] = {
                                let mut c = [0.0f32; 3];
                                for &v in &t {
                                    let p = pos(v);
                                    for a in 0..3 {
                                        c[a] += p[a] / 4.0;
                                    }
                                }
                                c
                            };
                            for &v in &t {
                                let p = pos(v);
                                for a in 0..3 {
                                    g[a] += vals[v] * (p[a] - centre[a]);
                                }
                            }
                            g
                        };
                        for mut tri in tris {
                            let (p0, p1, p2) = (tri[0].1, tri[1].1, tri[2].1);
                            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                            let nrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                            if nrm[0] * grad[0] + nrm[1] * grad[1] + nrm[2] * grad[2] < 0.0 {
                                tri.swap(1, 2);
                            }
                            out.push(([tri[0].0, tri[1].0, tri[2].0], [tri[0].1, tri[1].1, tri[2].1]));
                        }
                    }
                }
            }
            out
        })
        .collect();
    // weld: one vertex per grid edge
    let mut index: HashMap<(u32, u32), u32> = HashMap::new();
    let mut mesh = Mesh::default();
    for slab in slabs {
        for (keys, ps) in slab {
            let mut t = [0u32; 3];
            for v in 0..3 {
                t[v] = *index.entry(keys[v]).or_insert_with(|| {
                    mesh.positions.push(ps[v]);
                    (mesh.positions.len() - 1) as u32
                });
            }
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                mesh.triangles.push(t);
            }
        }
    }
    taubin(&mut mesh, 6);
    for p in mesh.positions.iter_mut() {
        for a in p.iter_mut() {
            *a *= cell;
        }
    }
    mesh.normals = normals(&mesh);
    mesh
}

/// Taubin smoothing: alternate a shrinking and an inflating Laplacian step, so bumps go away
/// but the liquid keeps its size.
fn taubin(mesh: &mut Mesh, rounds: usize) {
    let n = mesh.positions.len();
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for t in &mesh.triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            nbrs[a as usize].push(b);
            nbrs[b as usize].push(a);
        }
    }
    for l in nbrs.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    for _ in 0..rounds {
        for factor in [0.5f32, -0.53] {
            let old = mesh.positions.clone();
            mesh.positions.par_iter_mut().enumerate().for_each(|(i, p)| {
                let l = &nbrs[i];
                if l.is_empty() {
                    return;
                }
                let mut c = [0.0f32; 3];
                for &o in l {
                    for a in 0..3 {
                        c[a] += old[o as usize][a];
                    }
                }
                for a in 0..3 {
                    p[a] += factor * (c[a] / l.len() as f32 - old[i][a]);
                }
            });
        }
    }
}

fn normals(mesh: &Mesh) -> Vec<[f32; 3]> {
    let mut nrm = vec![[0.0f32; 3]; mesh.positions.len()];
    for t in &mesh.triangles {
        let (p0, p1, p2) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        for &v in t {
            for a in 0..3 {
                nrm[v as usize][a] += c[a];
            }
        }
    }
    for v in nrm.iter_mut() {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        for a in v.iter_mut() {
            *a /= l;
        }
    }
    nrm
}

impl Mesh {
    /// Volume enclosed (divergence theorem), cubic metres.
    pub fn volume(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let (a, b, c) = (self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
                (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64) - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64) + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64)) / 6.0
            })
            .sum()
    }

    /// Wavefront OBJ with normals. Z is up (in Blender's importer choose "Up: Z", "Forward: Y").
    pub fn to_obj(&self, name: &str) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(self.positions.len() * 60 + self.triangles.len() * 30);
        let _ = writeln!(s, "# SimTara liquid surface, metres, z up\no {name}");
        for p in &self.positions {
            let _ = writeln!(s, "v {:.5} {:.5} {:.5}", p[0], p[1], p[2]);
        }
        for n in &self.normals {
            let _ = writeln!(s, "vn {:.4} {:.4} {:.4}", n[0], n[1], n[2]);
        }
        for t in &self.triangles {
            let _ = writeln!(s, "f {a}//{a} {b}//{b} {c}//{c}", a = t[0] + 1, b = t[1] + 1, c = t[2] + 1);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball(d: Dims, centre: [f32; 3], r: f32) -> Vec<[f32; 6]> {
        // 8 particles per cell, jittered the same way every time, inside a ball (grid units)
        let mut out = Vec::new();
        let mut seed = 9u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 0.25 - 0.125
        };
        for k in 0..d.nz * 2 {
            for j in 0..d.ny * 2 {
                for i in 0..d.nx * 2 {
                    let p = [(i as f32 + 0.5) * 0.5 + rnd(), (j as f32 + 0.5) * 0.5 + rnd(), (k as f32 + 0.5) * 0.5 + rnd()];
                    if (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2) + (p[2] - centre[2]).powi(2) <= r * r {
                        out.push([p[0], p[1], p[2], 0.0, 0.0, 0.0]);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn a_ball_of_liquid_becomes_a_closed_round_surface_of_the_right_size() {
        let d = Dims { nx: 24, ny: 24, nz: 24 };
        let (c, r) = ([12.0f32, 12.0, 12.0], 7.0f32);
        let mesh = surface(&ball(d, c, r), d, 0.01, 2);
        assert!(mesh.triangles.len() > 1000, "{} triangles", mesh.triangles.len());
        // closed: every edge is shared by exactly two triangles, facing opposite ways
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &mesh.triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a.min(b), a.max(b))).or_default() += if a < b { 1 } else { -1 };
            }
        }
        assert!(edges.values().all(|v| *v == 0), "watertight and consistently oriented");
        // round and the right size: ball of 7 cells = 0.07 m
        let want = 4.0 / 3.0 * std::f64::consts::PI * 0.07f64.powi(3);
        let got = mesh.volume();
        assert!((got - want).abs() < 0.08 * want, "volume {got:.3e} m³ vs ball {want:.3e} m³");
        let centre = [0.12f32, 0.12, 0.12];
        let radii: Vec<f32> = mesh.positions.iter().map(|p| ((p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2) + (p[2] - centre[2]).powi(2)).sqrt()).collect();
        let (lo, hi) = radii.iter().fold((f32::MAX, f32::MIN), |(a, b), r| (a.min(*r), b.max(*r)));
        assert!(lo > 0.065 && hi < 0.075, "surface within half a cell of the ball: {lo:.4}..{hi:.4} m");
        // normals point outwards
        let out = mesh.positions.iter().zip(&mesh.normals).filter(|(p, n)| (p[0] - centre[0]) * n[0] + (p[1] - centre[1]) * n[1] + (p[2] - centre[2]) * n[2] > 0.0).count();
        assert!(out as f32 > 0.99 * mesh.positions.len() as f32);
        let obj = mesh.to_obj("water");
        assert!(obj.lines().filter(|l| l.starts_with("f ")).count() == mesh.triangles.len());
        assert!(surface(&[], d, 0.01, 2).triangles.is_empty());
    }

    #[test]
    fn water_filling_a_tank_is_closed_at_the_walls_and_holds_its_volume() {
        let d = Dims { nx: 10, ny: 8, nz: 10 };
        let mut pv = Vec::new();
        for k in 0..10 {
            for j in 0..16 {
                for i in 0..20 {
                    pv.push([(i as f32 + 0.5) * 0.5, (j as f32 + 0.5) * 0.5, (k as f32 + 0.5) * 0.5, 0.0, 0.0, 0.0]);
                }
            }
        }
        // 10 x 8 x 5 cells of 1 cm = 400 cm³
        let mesh = surface(&pv, d, 0.01, 2);
        let v = mesh.volume();
        assert!((v - 4.0e-4).abs() < 0.08 * 4.0e-4, "volume {v:.3e} m³ vs 4.0e-4 m³");
    }

    #[test]
    fn still_water_has_a_flat_top_at_its_real_height() {
        let d = Dims { nx: 16, ny: 16, nz: 16 };
        let mut pv = Vec::new();
        for k in 0..12 {
            for j in 0..32 {
                for i in 0..32 {
                    pv.push([(i as f32 + 0.5) * 0.5, (j as f32 + 0.5) * 0.5, (k as f32 + 0.5) * 0.5, 0.0, 0.0, 0.0]);
                }
            }
        }
        // water 6 cells deep (0.06 m with 1 cm cells)
        let mesh = surface(&pv, d, 0.01, 2);
        let top: Vec<f32> = mesh.positions.iter().zip(&mesh.normals).filter(|(p, n)| n[2] > 0.9 && p[0] > 0.04 && p[0] < 0.12 && p[1] > 0.04 && p[1] < 0.12).map(|(p, _)| p[2]).collect();
        assert!(!top.is_empty());
        let mean = top.iter().sum::<f32>() / top.len() as f32;
        let spread = top.iter().fold(0.0f32, |m, z| m.max((z - mean).abs()));
        assert!((mean - 0.06).abs() < 0.004, "top at {mean:.4} m, water is 0.06 m deep");
        assert!(spread < 0.0015, "flat: within {spread:.4} m");
    }
}
