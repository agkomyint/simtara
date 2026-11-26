//! The staggered (MAC) grid both fluids share: velocities live on cell faces, everything else
//! in cell centres. Positions are in *grid units* (cell i spans [i, i+1)); velocities in m/s.
//!
//! Everything here is deterministic: parallel loops only write their own slab, and the pressure
//! solver is red-black Gauss-Seidel (each colour reads only the other colour), so a run repeats
//! exactly however many cores there are.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const FLUID: u8 = 0;
pub const SOLID: u8 = 1;
/// Empty space with zero pressure (above a water surface, or outside an open side).
pub const AIR: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dims {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Dims {
    pub fn cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    pub fn c(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }
    pub fn cd(&self) -> [usize; 3] {
        [self.nx, self.ny, self.nz]
    }
    pub fn ud(&self) -> [usize; 3] {
        [self.nx + 1, self.ny, self.nz]
    }
    pub fn vd(&self) -> [usize; 3] {
        [self.nx, self.ny + 1, self.nz]
    }
    pub fn wd(&self) -> [usize; 3] {
        [self.nx, self.ny, self.nz + 1]
    }
}

pub fn idx(d: [usize; 3], i: usize, j: usize, k: usize) -> usize {
    i + d[0] * (j + d[1] * k)
}

pub fn len(d: [usize; 3]) -> usize {
    d[0] * d[1] * d[2]
}

/// Where element (0,0,0) of each array sits, in grid units.
pub const OFF_C: [f32; 3] = [0.5, 0.5, 0.5];
pub const OFF_U: [f32; 3] = [0.0, 0.5, 0.5];
pub const OFF_V: [f32; 3] = [0.5, 0.0, 0.5];
pub const OFF_W: [f32; 3] = [0.5, 0.5, 0.0];

#[inline]
fn corner(x: f32, n: usize) -> (usize, f32) {
    let x = x.clamp(0.0, (n - 1) as f32);
    let i = (x.floor() as usize).min(n.saturating_sub(2));
    (i, x - i as f32)
}

/// Trilinear sample of `a` (dims `d`) at index-space position `p`, clamped to the array.
pub fn sample(a: &[f32], d: [usize; 3], p: [f32; 3]) -> f32 {
    let (i, tx) = corner(p[0], d[0]);
    let (j, ty) = corner(p[1], d[1]);
    let (k, tz) = corner(p[2], d[2]);
    let (i1, j1, k1) = ((i + 1).min(d[0] - 1), (j + 1).min(d[1] - 1), (k + 1).min(d[2] - 1));
    let g = |i, j, k| a[idx(d, i, j, k)];
    let x00 = g(i, j, k) * (1.0 - tx) + g(i1, j, k) * tx;
    let x10 = g(i, j1, k) * (1.0 - tx) + g(i1, j1, k) * tx;
    let x01 = g(i, j, k1) * (1.0 - tx) + g(i1, j, k1) * tx;
    let x11 = g(i, j1, k1) * (1.0 - tx) + g(i1, j1, k1) * tx;
    let y0 = x00 * (1.0 - ty) + x10 * ty;
    let y1 = x01 * (1.0 - ty) + x11 * ty;
    y0 * (1.0 - tz) + y1 * tz
}

/// The smallest and largest of the 8 values `sample` blends (used to keep MacCormack bounded).
fn sample_range(a: &[f32], d: [usize; 3], p: [f32; 3]) -> (f32, f32) {
    let (i, _) = corner(p[0], d[0]);
    let (j, _) = corner(p[1], d[1]);
    let (k, _) = corner(p[2], d[2]);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for kk in [k, (k + 1).min(d[2] - 1)] {
        for jj in [j, (j + 1).min(d[1] - 1)] {
            for ii in [i, (i + 1).min(d[0] - 1)] {
                let v = a[idx(d, ii, jj, kk)];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    (lo, hi)
}

fn at(p: [f32; 3], off: [f32; 3]) -> [f32; 3] {
    [p[0] - off[0], p[1] - off[1], p[2] - off[2]]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mac {
    pub d: Dims,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
}

impl Mac {
    pub fn new(d: Dims) -> Mac {
        Mac { d, u: vec![0.0; len(d.ud())], v: vec![0.0; len(d.vd())], w: vec![0.0; len(d.wd())] }
    }

    /// Velocity (m/s) at a grid-space position.
    pub fn vel(&self, p: [f32; 3]) -> [f32; 3] {
        [sample(&self.u, self.d.ud(), at(p, OFF_U)), sample(&self.v, self.d.vd(), at(p, OFF_V)), sample(&self.w, self.d.wd(), at(p, OFF_W))]
    }

    /// Where the fluid now at `p` was `dt` seconds ago (second-order backtrace), grid units.
    pub fn back(&self, p: [f32; 3], s: f32) -> [f32; 3] {
        let v = self.vel(p);
        let mid = [p[0] - 0.5 * s * v[0], p[1] - 0.5 * s * v[1], p[2] - 0.5 * s * v[2]];
        let v = self.vel(mid);
        [p[0] - s * v[0], p[1] - s * v[1], p[2] - s * v[2]]
    }

    /// Velocity at a cell centre.
    pub fn centre(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        let d = self.d;
        [
            0.5 * (self.u[idx(d.ud(), i, j, k)] + self.u[idx(d.ud(), i + 1, j, k)]),
            0.5 * (self.v[idx(d.vd(), i, j, k)] + self.v[idx(d.vd(), i, j + 1, k)]),
            0.5 * (self.w[idx(d.wd(), i, j, k)] + self.w[idx(d.wd(), i, j, k + 1)]),
        ]
    }

    pub fn max_speed(&self) -> f32 {
        let m = |a: &[f32]| a.par_iter().fold(|| 0.0f32, |m, x| m.max(x.abs())).reduce(|| 0.0, f32::max);
        m(&self.u).max(m(&self.v)).max(m(&self.w))
    }
}

/// Semi-Lagrangian advection of one array (`off` = where its elements sit) through `mac`.
/// `s` = dt / cell size, so velocity times `s` is a distance in cells.
pub fn advect(src: &[f32], d: [usize; 3], off: [f32; 3], mac: &Mac, s: f32) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    dst.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..d[1] {
            for i in 0..d[0] {
                let p = [i as f32 + off[0], j as f32 + off[1], k as f32 + off[2]];
                slab[i + d[0] * j] = sample(src, d, at(mac.back(p, s), off));
            }
        }
    });
    dst
}

/// MacCormack advection of cell-centred quantities: sharper than semi-Lagrangian (smoke keeps
/// its curls), clamped to the values it came from so it never overshoots. The paths are traced
/// once and shared by every quantity (smoke and heat move together).
pub fn advect_sharp_many(fields: &mut [&mut Vec<f32>], d: Dims, mac: &Mac, s: f32) {
    let cd = d.cd();
    let n = d.cells();
    let centre = |c: usize| [(c % cd[0]) as f32 + 0.5, ((c / cd[0]) % cd[1]) as f32 + 0.5, (c / (cd[0] * cd[1])) as f32 + 0.5];
    // where each cell's air came from, and where it goes (both in index space of the cells)
    let back: Vec<[f32; 3]> = (0..n).into_par_iter().map(|c| at(mac.back(centre(c), s), OFF_C)).collect();
    let fwd: Vec<[f32; 3]> = (0..n).into_par_iter().map(|c| at(mac.back(centre(c), -s), OFF_C)).collect();
    for src in fields.iter_mut() {
        let a: Vec<f32> = back.par_iter().map(|p| sample(src, cd, *p)).collect();
        let out: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|c| {
                let b = sample(&a, cd, fwd[c]);
                let v = a[c] + 0.5 * (src[c] - b);
                let (lo, hi) = sample_range(src, cd, back[c]);
                if v < lo || v > hi { a[c] } else { v }
            })
            .collect();
        **src = out;
    }
}

/// MacCormack advection of an array that sits at `off` (a face velocity or cell values): far
/// less smearing than plain semi-Lagrangian, so a wake sheds vortices at the right Reynolds
/// number and the air keeps its momentum. Clamped to the values it came from (no overshoot).
pub fn advect_sharp_at(src: &[f32], dims: [usize; 3], off: [f32; 3], mac: &Mac, s: f32) -> Vec<f32> {
    let n = src.len();
    let pos = |e: usize| [(e % dims[0]) as f32 + off[0], ((e / dims[0]) % dims[1]) as f32 + off[1], (e / (dims[0] * dims[1])) as f32 + off[2]];
    let back: Vec<[f32; 3]> = (0..n).into_par_iter().map(|e| at(mac.back(pos(e), s), off)).collect();
    let a: Vec<f32> = back.par_iter().map(|p| sample(src, dims, *p)).collect();
    (0..n)
        .into_par_iter()
        .map(|e| {
            let b = sample(&a, dims, at(mac.back(pos(e), -s), off));
            let v = a[e] + 0.5 * (src[e] - b);
            let (lo, hi) = sample_range(src, dims, back[e]);
            if v < lo || v > hi { a[e] } else { v }
        })
        .collect()
}

/// [`advect_sharp_many`] for one quantity.
pub fn advect_sharp(src: &[f32], d: Dims, mac: &Mac, s: f32) -> Vec<f32> {
    let mut v = src.to_vec();
    advect_sharp_many(&mut [&mut v], d, mac, s);
    v
}

/// How the sides of the box behave.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sides {
    /// Air can leave through the top (zero pressure there).
    pub open_top: bool,
    /// Air can leave through the far x side.
    pub open_xmax: bool,
    /// Air blows in through the x = 0 side at this speed (m/s); None = a wall.
    pub inflow: Option<f32>,
    /// The walls of the box hold the air next to them still (no-slip); false = it slides.
    #[serde(default)]
    pub no_slip: bool,
    /// The top wall slides along x at this speed, m/s (the lid-driven cavity benchmark).
    #[serde(default)]
    pub lid: f32,
}

/// Walls, solids and the inflow: set the face velocities that are fixed.
pub fn enforce(mac: &mut Mac, flags: &[u8], sides: Sides) {
    let d = mac.d;
    let solid = |i: usize, j: usize, k: usize| flags[d.c(i, j, k)] == SOLID;
    let (ud, vd, wd) = (d.ud(), d.vd(), d.wd());
    mac.u.par_chunks_mut(ud[0] * ud[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..d.ny {
            for i in 0..=d.nx {
                let f = &mut slab[i + ud[0] * j];
                if i == 0 {
                    *f = if solid(0, j, k) { 0.0 } else { sides.inflow.unwrap_or(0.0) };
                } else if i == d.nx {
                    if !sides.open_xmax || solid(i - 1, j, k) {
                        *f = 0.0;
                    }
                } else if solid(i - 1, j, k) || solid(i, j, k) {
                    *f = 0.0;
                }
            }
        }
    });
    mac.v.par_chunks_mut(vd[0] * vd[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..=d.ny {
            for i in 0..d.nx {
                if j == 0 || j == d.ny || solid(i, j - 1, k) || solid(i, j, k) {
                    slab[i + vd[0] * j] = 0.0;
                }
            }
        }
    });
    mac.w.par_chunks_mut(wd[0] * wd[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..d.ny {
            for i in 0..d.nx {
                let f = &mut slab[i + wd[0] * j];
                if k == 0 {
                    *f = 0.0;
                } else if k == d.nz {
                    if !sides.open_top || solid(i, j, k - 1) {
                        *f = 0.0;
                    }
                } else if solid(i, j, k - 1) || solid(i, j, k) {
                    *f = 0.0;
                }
            }
        }
    });
}

/// Faces next to (or inside) a moving object take its velocity: it is a wall that moves, so the
/// liquid in front of it is pushed and the liquid behind follows. `which` = 1 + object index
/// per cell (0 = none).
pub fn apply_objects(mac: &mut Mac, which: &[u8], vel: &[[f32; 3]]) {
    if vel.is_empty() {
        return;
    }
    let d = mac.d;
    let obj = |i: usize, j: usize, k: usize| which[d.c(i, j, k)];
    for (axis, dims) in [(0usize, d.ud()), (1, d.vd()), (2, d.wd())] {
        let arr = match axis {
            0 => &mut mac.u,
            1 => &mut mac.v,
            _ => &mut mac.w,
        };
        arr.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(k, slab)| {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let (a, b) = match axis {
                        0 => ((i > 0).then(|| obj(i - 1, j, k)), (i < d.nx).then(|| obj(i, j, k))),
                        1 => ((j > 0).then(|| obj(i, j - 1, k)), (j < d.ny).then(|| obj(i, j, k))),
                        _ => ((k > 0).then(|| obj(i, j, k - 1)), (k < d.nz).then(|| obj(i, j, k))),
                    };
                    if let Some(o) = a.filter(|x| *x > 0).or(b.filter(|x| *x > 0)) {
                        slab[i + dims[0] * j] = vel[o as usize - 1][axis];
                    }
                }
            }
        });
    }
}

/// Make the flow divergence-free in every FLUID cell: air neither piles up nor disappears.
/// SOLID neighbours are walls, AIR cells and open sides hold zero pressure. `p` is the
/// pressure (scaled, warm-started from the last step). Returns the largest divergence left.
/// `target`: how much each cell should expand (+) or shrink (-), in the same units as the
/// divergence (used to hold a liquid's volume); None = exactly incompressible.
pub fn project(mac: &mut Mac, flags: &[u8], sides: Sides, p: &mut Vec<f32>, iterations: usize, target: Option<&[f32]>) -> f32 {
    project_with(mac, flags, sides, p, iterations, target, None)
}

/// [`project`] with a free surface placed between cell centres from `fill` (how full each cell
/// is, 1 = full of liquid): the ghost fluid method (Gibou et al. 2002). Without it the surface
/// sits at the air cell's centre, half a cell too high, and pressures and waves come out wrong.
pub fn project_with(mac: &mut Mac, flags: &[u8], sides: Sides, p: &mut Vec<f32>, iterations: usize, target: Option<&[f32]>, fill: Option<&[f32]>) -> f32 {
    project_full(mac, flags, sides, p, iterations, target, fill, None)
}

/// Face weights for a fluid whose density changes from place to place (hot air, helium, CO2):
/// `rho0 / rho` on every face (u, v, w arrays). A light gas is pushed further by the same
/// pressure difference than a heavy one, as Newton says (a = -grad p / rho).
pub type Beta = [Vec<f32>; 3];

/// [`project_with`] with face weights `beta` (None = all 1, one density everywhere).
#[allow(clippy::too_many_arguments)]
pub fn project_full(mac: &mut Mac, flags: &[u8], sides: Sides, p: &mut Vec<f32>, iterations: usize, target: Option<&[f32]>, fill: Option<&[f32]>, beta: Option<&Beta>) -> f32 {
    let d = mac.d;
    let cd = d.cd();
    let n = d.cells();
    if p.len() != n {
        *p = vec![0.0; n];
    }
    // divergence, and for each cell: how many neighbours count, and which are fluid
    let mut div = vec![0.0f32; n];
    let mut diag = vec![0.0f32; n];
    // fraction of the way from a liquid cell's centre to its air neighbour's centre where the surface is
    let theta = |c: usize, nc: usize| -> f32 {
        match fill {
            Some(f) => {
                let (a, b) = (f[c].min(1.0), f[nc].min(1.0));
                if a > 0.5 && a - b > 1e-4 { ((a - 0.5) / (a - b)).clamp(0.2, 1.0) } else { 1.0 }
            }
            None => 1.0,
        }
    };
    // weight of each of a cell's six faces (-x, +x, -y, +y, -z, +z) towards fluid neighbours
    let mut coef = vec![[0.0f32; 6]; n];
    let (bu, bv, bw) = match beta {
        Some(b) => (Some(&b[0]), Some(&b[1]), Some(&b[2])),
        None => (None, None, None),
    };
    let wt = |b: Option<&Vec<f32>>, f: usize| b.map_or(1.0, |a| a[f]);
    div.par_chunks_mut(cd[0] * cd[1]).zip(diag.par_chunks_mut(cd[0] * cd[1])).zip(coef.par_chunks_mut(cd[0] * cd[1])).enumerate().for_each(|(k, ((dv, dg), cf))| {
        for j in 0..d.ny {
            for i in 0..d.nx {
                let c = d.c(i, j, k);
                let s = i + cd[0] * j;
                if flags[c] != FLUID {
                    continue;
                }
                dv[s] = mac.u[idx(d.ud(), i + 1, j, k)] - mac.u[idx(d.ud(), i, j, k)] + mac.v[idx(d.vd(), i, j + 1, k)] - mac.v[idx(d.vd(), i, j, k)] + mac.w[idx(d.wd(), i, j, k + 1)] - mac.w[idx(d.wd(), i, j, k)] - target.map_or(0.0, |t| t[c]);
                let mut count = 0.0f32;
                let mut row = [0.0f32; 6];
                let mut look = |inside: Option<usize>, open: bool, slot: usize, b: f32| match inside {
                    Some(nc) => match flags[nc] {
                        SOLID => {}
                        AIR => count += b / theta(c, nc),
                        _ => {
                            count += b;
                            row[slot] = b;
                        }
                    },
                    None if open => count += b,
                    None => {}
                };
                look((i > 0).then(|| d.c(i - 1, j, k)), false, 0, wt(bu, idx(d.ud(), i, j, k)));
                look((i + 1 < d.nx).then(|| d.c(i + 1, j, k)), sides.open_xmax, 1, wt(bu, idx(d.ud(), i + 1, j, k)));
                look((j > 0).then(|| d.c(i, j - 1, k)), false, 2, wt(bv, idx(d.vd(), i, j, k)));
                look((j + 1 < d.ny).then(|| d.c(i, j + 1, k)), false, 3, wt(bv, idx(d.vd(), i, j + 1, k)));
                look((k > 0).then(|| d.c(i, j, k - 1)), false, 4, wt(bw, idx(d.wd(), i, j, k)));
                look((k + 1 < d.nz).then(|| d.c(i, j, k + 1)), sides.open_top, 5, wt(bw, idx(d.wd(), i, j, k + 1)));
                dg[s] = count;
                cf[s] = row;
            }
        }
    });

    // Conjugate gradients with a diagonal preconditioner, warm-started from the last pressure.
    // It converges far faster than relaxation sweeps, which matters for water: a half-solved
    // pressure lets water compress and gives the wrong hydrostatic pressure and wave speeds.
    // Sums are taken in fixed chunks in a fixed order, so the result repeats exactly.
    // every vector covers all cells; cells outside the solve keep zeros (inv_diag = 0), so each
    // update is one parallel pass over plain arrays
    let inv_diag: Vec<f32> = (0..n).into_par_iter().map(|c| if flags[c] == FLUID && diag[c] > 0.0 { 1.0 / diag[c] } else { 0.0 }).collect();
    let (sx, sy, sz) = (1usize, cd[0], cd[0] * cd[1]);
    let apply = |x: &[f32], out: &mut [f32]| {
        out.par_iter_mut().enumerate().for_each(|(c, o)| {
            if inv_diag[c] == 0.0 {
                *o = 0.0;
                return;
            }
            let m = &coef[c];
            let mut s = diag[c] * x[c];
            if m[0] != 0.0 { s -= m[0] * x[c - sx]; }
            if m[1] != 0.0 { s -= m[1] * x[c + sx]; }
            if m[2] != 0.0 { s -= m[2] * x[c - sy]; }
            if m[3] != 0.0 { s -= m[3] * x[c + sy]; }
            if m[4] != 0.0 { s -= m[4] * x[c - sz]; }
            if m[5] != 0.0 { s -= m[5] * x[c + sz]; }
            *o = s;
        });
    };
    // sums in fixed chunks, added in a fixed order: the result repeats exactly
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        let parts: Vec<f64> = a.par_chunks(8192).zip(b.par_chunks(8192)).map(|(x, y)| x.iter().zip(y).map(|(u, v)| *u as f64 * *v as f64).sum::<f64>()).collect();
        parts.iter().sum()
    };
    let absmax = |a: &[f32]| a.par_iter().fold(|| 0.0f32, |m, x| m.max(x.abs())).reduce(|| 0.0, f32::max);
    p.par_iter_mut().zip(inv_diag.par_iter()).for_each(|(x, d)| if *d == 0.0 { *x = 0.0 });
    let mut ap = vec![0.0f32; n];
    apply(p, &mut ap);
    // residual r = b - A p, with b = -div
    let mut r: Vec<f32> = (0..n).into_par_iter().map(|c| if inv_diag[c] == 0.0 { 0.0 } else { -div[c] - ap[c] }).collect();
    let bnorm = (0..n).into_par_iter().map(|c| if inv_diag[c] == 0.0 { 0.0 } else { div[c].abs() }).reduce(|| 0.0, f32::max).max(1e-12);
    let mut z: Vec<f32> = r.par_iter().zip(inv_diag.par_iter()).map(|(r, d)| r * d).collect();
    let mut dir = z.clone();
    let mut rz = dot(&r, &z);
    let max_iter = (iterations * 4).max(20);
    // liquids need the pressure almost exact (hydrostatics, wave speeds); for a gas, a thousandth
    // of the divergence left is far below what the grid can resolve, and many times cheaper
    let tol = if beta.is_some() { 1e-3 } else { 1e-5 };
    for _ in 0..max_iter {
        if absmax(&r) <= tol * bnorm || rz.abs() < 1e-30 {
            break;
        }
        apply(&dir, &mut ap);
        let dap = dot(&dir, &ap);
        if dap.abs() < 1e-30 {
            break;
        }
        let alpha = (rz / dap) as f32;
        p.par_iter_mut().zip(r.par_iter_mut()).zip(z.par_iter_mut()).zip(dir.par_iter().zip(ap.par_iter()).zip(inv_diag.par_iter())).for_each(|(((p, r), z), ((d, a), i))| {
            *p += alpha * d;
            *r -= alpha * a;
            *z = *r * i;
        });
        let rz_new = dot(&r, &z);
        let bk = (rz_new / rz) as f32;
        rz = rz_new;
        dir.par_iter_mut().zip(z.par_iter()).for_each(|(d, z)| *d = z + bk * *d);
    }    for c in 0..n {
        if flags[c] != FLUID {
            p[c] = 0.0;
        }
    }

    // subtract the pressure gradient from faces next to fluid
    // the jump in pressure across a face; next to the surface the air side holds a ghost value
    let jump = |a: usize, b: usize| -> Option<f32> {
        match (flags[a], flags[b]) {
            (SOLID, _) | (_, SOLID) => None,
            (FLUID, AIR) => Some(-p[a] / theta(a, b)),
            (AIR, FLUID) => Some(p[b] / theta(b, a)),
            _ => Some(p[b] - p[a]),
        }
    };
    let pf = |i: usize, j: usize, k: usize| -> Option<f32> {
        match flags[d.c(i, j, k)] {
            SOLID => None,
            _ => Some(p[d.c(i, j, k)]),
        }
    };
    let fluid = |i: usize, j: usize, k: usize| flags[d.c(i, j, k)] == FLUID;
    let (ud, vd, wd) = (d.ud(), d.vd(), d.wd());
    mac.u.par_chunks_mut(ud[0] * ud[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..d.ny {
            for i in 1..=d.nx {
                let b = wt(bu, idx(ud, i, j, k));
                let (l, r) = if i == d.nx {
                    if !sides.open_xmax {
                        continue;
                    }
                    (pf(i - 1, j, k), Some(0.0))
                } else {
                    if !fluid(i - 1, j, k) && !fluid(i, j, k) {
                        continue;
                    }
                    if let Some(g) = jump(d.c(i - 1, j, k), d.c(i, j, k)) {
                        slab[i + ud[0] * j] -= b * g;
                    }
                    continue;
                };
                if let (Some(l), Some(r)) = (l, r) {
                    slab[i + ud[0] * j] -= b * (r - l);
                }
            }
        }
    });
    mac.v.par_chunks_mut(vd[0] * vd[1]).enumerate().for_each(|(k, slab)| {
        for j in 1..d.ny {
            for i in 0..d.nx {
                if !fluid(i, j - 1, k) && !fluid(i, j, k) {
                    continue;
                }
                if let Some(g) = jump(d.c(i, j - 1, k), d.c(i, j, k)) {
                    slab[i + vd[0] * j] -= wt(bv, idx(vd, i, j, k)) * g;
                }
            }
        }
    });
    mac.w.par_chunks_mut(wd[0] * wd[1]).enumerate().for_each(|(k, slab)| {
        if k == 0 {
            return;
        }
        for j in 0..d.ny {
            for i in 0..d.nx {
                let b = wt(bw, idx(wd, i, j, k));
                let (l, r) = if k == d.nz {
                    if !sides.open_top {
                        continue;
                    }
                    (pf(i, j, k - 1), Some(0.0))
                } else {
                    if !fluid(i, j, k - 1) && !fluid(i, j, k) {
                        continue;
                    }
                    if let Some(g) = jump(d.c(i, j, k - 1), d.c(i, j, k)) {
                        slab[i + wd[0] * j] -= b * g;
                    }
                    continue;
                };
                if let (Some(l), Some(r)) = (l, r) {
                    slab[i + wd[0] * j] -= b * (r - l);
                }
            }
        }
    });
    max_divergence(mac, flags)
}

/// Largest |inflow - outflow| of any fluid cell, in m/s (0 = perfectly incompressible).
pub fn max_divergence(mac: &Mac, flags: &[u8]) -> f32 {
    let d = mac.d;
    (0..d.nz)
        .into_par_iter()
        .map(|k| {
            let mut m = 0.0f32;
            for j in 0..d.ny {
                for i in 0..d.nx {
                    if flags[d.c(i, j, k)] != FLUID {
                        continue;
                    }
                    let dv = mac.u[idx(d.ud(), i + 1, j, k)] - mac.u[idx(d.ud(), i, j, k)] + mac.v[idx(d.vd(), i, j + 1, k)] - mac.v[idx(d.vd(), i, j, k)] + mac.w[idx(d.wd(), i, j, k + 1)] - mac.w[idx(d.wd(), i, j, k)];
                    m = m.max(dv.abs());
                }
            }
            m
        })
        .reduce(|| 0.0, f32::max)
}

/// Fill faces that have no value yet (`known` false) from known neighbours, `layers` times,
/// so particles just above a water surface still find a sensible velocity.
pub fn extrapolate(a: &mut [f32], known: &mut [bool], d: [usize; 3], layers: usize) {
    for _ in 0..layers {
        let mut add: Vec<(usize, f32)> = Vec::new();
        for k in 0..d[2] {
            for j in 0..d[1] {
                for i in 0..d[0] {
                    let c = idx(d, i, j, k);
                    if known[c] {
                        continue;
                    }
                    let (mut s, mut n) = (0.0, 0);
                    let mut take = |ok: bool, nc: usize| {
                        if ok && known[nc] {
                            s += a[nc];
                            n += 1;
                        }
                    };
                    take(i > 0, c.wrapping_sub(1));
                    take(i + 1 < d[0], c + 1);
                    take(j > 0, c.wrapping_sub(d[0]));
                    take(j + 1 < d[1], c + d[0]);
                    take(k > 0, c.wrapping_sub(d[0] * d[1]));
                    take(k + 1 < d[2], c + d[0] * d[1]);
                    if n > 0 {
                        add.push((c, s / n as f32));
                    }
                }
            }
        }
        if add.is_empty() {
            break;
        }
        for (c, v) in add {
            a[c] = v;
            known[c] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilinear_sampling_is_exact_on_a_linear_field() {
        let d = [4, 3, 5];
        let a: Vec<f32> = (0..len(d)).map(|c| {
            let (i, j, k) = (c % 4, (c / 4) % 3, c / 12);
            1.0 + 2.0 * i as f32 - 0.5 * j as f32 + 3.0 * k as f32
        }).collect();
        let v = sample(&a, d, [1.25, 0.5, 2.75]);
        assert!((v - (1.0 + 2.5 - 0.25 + 8.25)).abs() < 1e-5, "{v}");
        assert_eq!(sample(&a, d, [-3.0, -3.0, -3.0]), 1.0, "clamped");
    }

    #[test]
    fn projection_removes_divergence_in_a_closed_box() {
        let d = Dims { nx: 12, ny: 10, nz: 8 };
        let mut mac = Mac::new(d);
        // a lump of air pushed outwards from the middle
        for k in 0..d.nz {
            for j in 0..d.ny {
                for i in 1..d.nx {
                    let x = i as f32 - 6.0;
                    mac.u[idx(d.ud(), i, j, k)] = x.signum() * (-(x * x) / 8.0).exp();
                }
            }
        }
        let flags = vec![FLUID; d.cells()];
        let sides = Sides::default();
        enforce(&mut mac, &flags, sides);
        let before = max_divergence(&mac, &flags);
        let mut p = Vec::new();
        let after = project(&mut mac, &flags, sides, &mut p, 300, None);
        assert!(before > 0.3 && after < 1e-3 * before, "divergence {before} -> {after}");
    }

    #[test]
    fn an_open_top_lets_air_leave() {
        let d = Dims { nx: 6, ny: 6, nz: 6 };
        let mut mac = Mac::new(d);
        // air blowing in from the side, nowhere to go but up
        let flags = vec![FLUID; d.cells()];
        let sides = Sides { open_top: true, open_xmax: false, inflow: Some(1.0), ..Default::default() };
        enforce(&mut mac, &flags, sides);
        let mut p = Vec::new();
        let left = project(&mut mac, &flags, sides, &mut p, 400, None);
        assert!(left < 1e-3, "{left}");
        let up: f32 = (0..d.nx).flat_map(|i| (0..d.ny).map(move |j| (i, j))).map(|(i, j)| mac.w[idx(d.wd(), i, j, d.nz)]).sum();
        assert!((up - d.ny as f32 * d.nz as f32).abs() < 0.05 * up, "what comes in leaves through the top: {up}");
    }

    #[test]
    #[ignore]
    fn bench_parts() {
        let d = Dims { nx: 80, ny: 40, nz: 40 };
        let mut mac = Mac::new(d);
        for (i, u) in mac.u.iter_mut().enumerate() {
            *u = 1.0 + (i % 7) as f32 * 0.1;
        }
        let flags = vec![FLUID; d.cells()];
        let sides = Sides { open_top: false, open_xmax: true, inflow: Some(1.0), ..Default::default() };
        let mut p = Vec::new();
        let t = std::time::Instant::now();
        for _ in 0..5 {
            project(&mut mac, &flags, sides, &mut p, 40, None);
        }
        eprintln!("project x40: {:.1} ms", t.elapsed().as_secs_f64() * 200.0);
        let t = std::time::Instant::now();
        for _ in 0..5 {
            let _ = advect(&mac.u, d.ud(), OFF_U, &mac, 0.3);
        }
        eprintln!("advect one component: {:.1} ms", t.elapsed().as_secs_f64() * 200.0);
        let mut a = vec![0.5f32; d.cells()];
        let mut b = vec![0.5f32; d.cells()];
        let t = std::time::Instant::now();
        for _ in 0..5 {
            advect_sharp_many(&mut [&mut a, &mut b], d, &mac, 0.3);
        }
        eprintln!("maccormack smoke+temp: {:.1} ms", t.elapsed().as_secs_f64() * 200.0);
    }
}