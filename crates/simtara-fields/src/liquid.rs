//! Liquid particles and the grid: APIC transfers, keeping the volume, keeping particles apart.
//!
//! - **APIC** (Jiang et al. 2015, "The Affine Particle-In-Cell Method"): each particle carries its
//!   velocity *and* how the velocity changes around it (an affine matrix). Moving momentum
//!   between particles and grid then neither smears the flow (as PIC does) nor adds noise (as
//!   FLIP does), so splashes stay coherent and swirls keep spinning.
//! - **Volume**: particles carry no volume of their own, so liquid slowly inflates or shrinks.
//!   The pressure solve is asked to push crowded cells apart and pull thinned-out cells in, back
//!   to the rest density (as in Müller's FLIP notes), and particles closer than their rest spacing
//!   are pushed apart. Together they hold the volume to a few percent.
//!
//! Positions are in grid units (cell i spans [i, i+1)), velocities in m/s, the affine terms in
//! m/s per cell. Everything is deterministic: scatters run in a fixed order, gathers in parallel.

use crate::grid::{self, idx, Dims, Mac, FLUID, OFF_C, OFF_U, OFF_V, OFF_W, SOLID};
use rayon::prelude::*;

/// Particles seeded per cell (2 x 2 x 2), so the rest density is 8 per cell.
pub const PER_CELL: f32 = 8.0;
/// Rest spacing between particles in cells (8 per cell).
const SPACING: f32 = 0.5;

const OFFS: [[f32; 3]; 3] = [OFF_U, OFF_V, OFF_W];

/// Trilinear weights of the 8 nodes around `q` and their gradients, with node indices.
#[inline]
fn stencil(q: [f32; 3]) -> ([i64; 3], [[f32; 3]; 3]) {
    let b = [q[0].floor(), q[1].floor(), q[2].floor()];
    let t = [q[0] - b[0], q[1] - b[1], q[2] - b[2]];
    // per axis: weight of the low node, of the high node, and their derivatives (-1, +1)
    ([b[0] as i64, b[1] as i64, b[2] as i64], [[1.0 - t[0], t[0], 0.0], [1.0 - t[1], t[1], 0.0], [1.0 - t[2], t[2], 0.0]])
}

/// Particles -> grid with APIC. Returns the grid and which faces got any weight.
pub fn p2g(pos_vel: &[[f32; 6]], affine: &[[f32; 9]], d: Dims) -> (Mac, [Vec<bool>; 3]) {
    let mut mac = Mac::new(d);
    let mut known: [Vec<bool>; 3] = Default::default();
    let dims_all = [d.ud(), d.vd(), d.wd()];
    let arrays: Vec<(Vec<f32>, Vec<bool>)> = (0..3usize)
        .into_par_iter()
        .map(|axis| {
            let (off, dims) = (OFFS[axis], dims_all[axis]);
            let n = grid::len(dims);
            let (mut sum, mut wsum) = (vec![0.0f32; n], vec![0.0f32; n]);
            for (p, c) in pos_vel.iter().zip(affine) {
                let q = [p[0] - off[0], p[1] - off[1], p[2] - off[2]];
                let (b, w) = stencil(q);
                let ca = [c[axis * 3], c[axis * 3 + 1], c[axis * 3 + 2]];
                for corner in 0..8 {
                    let o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
                    let ii = [b[0] + o[0] as i64, b[1] + o[1] as i64, b[2] + o[2] as i64];
                    if (0..3).any(|a| ii[a] < 0 || ii[a] >= dims[a] as i64) {
                        continue;
                    }
                    let wt = w[0][o[0]] * w[1][o[1]] * w[2][o[2]];
                    let dx = [ii[0] as f32 - q[0], ii[1] as f32 - q[1], ii[2] as f32 - q[2]];
                    let v = p[3 + axis] + ca[0] * dx[0] + ca[1] * dx[1] + ca[2] * dx[2];
                    let cell = idx(dims, ii[0] as usize, ii[1] as usize, ii[2] as usize);
                    sum[cell] += wt * v;
                    wsum[cell] += wt;
                }
            }
            let ok: Vec<bool> = wsum.iter().map(|w| *w > 1e-6).collect();
            for c in 0..n {
                if ok[c] {
                    sum[c] /= wsum[c];
                }
            }
            (sum, ok)
        })
        .collect();
    for (axis, (arr, ok)) in arrays.into_iter().enumerate() {
        match axis {
            0 => mac.u = arr,
            1 => mac.v = arr,
            _ => mac.w = arr,
        }
        known[axis] = ok;
    }
    (mac, known)
}

/// Grid -> particles with APIC: the velocity at each particle and the affine matrix.
pub fn g2p(mac: &Mac, pos_vel: &mut [[f32; 6]], affine: &mut [[f32; 9]]) {
    let d = mac.d;
    let dims_all = [d.ud(), d.vd(), d.wd()];
    pos_vel.par_iter_mut().zip(affine.par_iter_mut()).for_each(|(p, c)| {
        for axis in 0..3 {
            let (off, dims) = (OFFS[axis], dims_all[axis]);
            let arr = match axis {
                0 => &mac.u,
                1 => &mac.v,
                _ => &mac.w,
            };
            let q = [p[0] - off[0], p[1] - off[1], p[2] - off[2]];
            let (b, w) = stencil(q);
            let (mut v, mut g) = (0.0f32, [0.0f32; 3]);
            for corner in 0..8 {
                let o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
                let ii = [(b[0] + o[0] as i64).clamp(0, dims[0] as i64 - 1), (b[1] + o[1] as i64).clamp(0, dims[1] as i64 - 1), (b[2] + o[2] as i64).clamp(0, dims[2] as i64 - 1)];
                let val = arr[idx(dims, ii[0] as usize, ii[1] as usize, ii[2] as usize)];
                let (wx, wy, wz) = (w[0][o[0]], w[1][o[1]], w[2][o[2]]);
                let s = |k: usize| if o[k] == 1 { 1.0 } else { -1.0 };
                v += wx * wy * wz * val;
                g[0] += s(0) * wy * wz * val;
                g[1] += wx * s(1) * wz * val;
                g[2] += wx * wy * s(2) * val;
            }
            p[3 + axis] = v;
            c[axis * 3] = g[0];
            c[axis * 3 + 1] = g[1];
            c[axis * 3 + 2] = g[2];
        }
    });
}

/// Particles per cell, spread trilinearly onto cell centres (8 in the middle of still liquid).
pub fn density(pos_vel: &[[f32; 6]], d: Dims) -> Vec<f32> {
    let cd = d.cd();
    let mut rho = vec![0.0f32; d.cells()];
    for p in pos_vel {
        let q = [p[0] - OFF_C[0], p[1] - OFF_C[1], p[2] - OFF_C[2]];
        let (b, w) = stencil(q);
        for corner in 0..8 {
            let o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            // nodes outside the box reflect back in, so cells by a wall are not counted as thin
            let ii = [(b[0] + o[0] as i64).clamp(0, cd[0] as i64 - 1), (b[1] + o[1] as i64).clamp(0, cd[1] as i64 - 1), (b[2] + o[2] as i64).clamp(0, cd[2] as i64 - 1)];
            rho[idx(cd, ii[0] as usize, ii[1] as usize, ii[2] as usize)] += w[0][o[0]] * w[1][o[1]] * w[2][o[2]];
        }
    }
    rho
}

/// A smooth "how full of liquid" field at the cell centres, for drawing the surface: each
/// particle spreads over the 27 nearest cells with a quadratic B-spline (smoother than the
/// trilinear `density`, so the surface and its normals are smooth). 1 = full, the surface is
/// where it crosses 0.5. Particles only steer it; on screen there are no particles.
pub fn surface_field(pos_vel: &[[f32; 6]], d: Dims) -> Vec<f32> {
    let cd = d.cd();
    let n = d.cells();
    let bspline = |x: f32| -> [f32; 3] {
        // weights of the cells at floor(x+0.5)-1, +0, +1 for a point x (in cell-centre units)
        let f = x - (x + 0.5).floor() + 0.5;
        [0.5 * (1.0 - f) * (1.0 - f), 0.75 - (f - 0.5) * (f - 0.5), 0.5 * f * f]
    };
    let chunk = (pos_vel.len() / rayon::current_num_threads().max(1)).max(4096);
    let parts: Vec<Vec<f32>> = pos_vel
        .par_chunks(chunk)
        .map(|ps| {
            let mut out = vec![0.0f32; n];
            for p in ps {
                let q = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
                let base = [(q[0] + 0.5).floor() as i64 - 1, (q[1] + 0.5).floor() as i64 - 1, (q[2] + 0.5).floor() as i64 - 1];
                let (wx, wy, wz) = (bspline(q[0]), bspline(q[1]), bspline(q[2]));
                for (c, w3) in wz.iter().enumerate() {
                    let k = (base[2] + c as i64).clamp(0, cd[2] as i64 - 1) as usize;
                    for (b, w2) in wy.iter().enumerate() {
                        let j = (base[1] + b as i64).clamp(0, cd[1] as i64 - 1) as usize;
                        for (a, w1) in wx.iter().enumerate() {
                            let i = (base[0] + a as i64).clamp(0, cd[0] as i64 - 1) as usize;
                            out[idx(cd, i, j, k)] += w1 * w2 * w3;
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mut field = vec![0.0f32; n];
    for part in parts {
        for (f, v) in field.iter_mut().zip(part) {
            *f += v;
        }
    }
    for f in field.iter_mut() {
        *f /= PER_CELL;
    }
    field
}

/// How much each liquid cell should expand (+) or shrink (-) this step to get back to the rest
/// density, as a divergence in the solver's units (m/s summed over the faces of a cell).
/// Cells at the surface are only pushed apart, never pulled in (they are partly empty anyway).
pub fn volume_target(rho: &[f32], flags: &[u8], solid: &[u8], d: Dims, h: f32, dt: f32) -> Vec<f32> {
    let rate = h / dt;
    (0..d.cells())
        .into_par_iter()
        .map(|c| {
            if flags[c] != FLUID {
                return 0.0;
            }
            // solid neighbours hold particles out of a share of the cell; count them as liquid
            let (i, j, k) = (c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny));
            let inside = |ok: bool, n: usize| !ok || flags[n] == FLUID || solid[n] == SOLID;
            let interior = inside(i > 0, c.wrapping_sub(1))
                && inside(i + 1 < d.nx, c + 1)
                && inside(j > 0, c.wrapping_sub(d.nx))
                && inside(j + 1 < d.ny, c + d.nx)
                && inside(k > 0, c.wrapping_sub(d.nx * d.ny))
                && inside(k + 1 < d.nz, c + d.nx * d.ny);
            let excess = (rho[c] - PER_CELL) / PER_CELL;
            if excess > 0.0 {
                0.5 * excess.min(1.0) * rate
            } else if interior {
                0.25 * excess.max(-0.5) * rate
            } else {
                0.0
            }
        })
        .collect()
}

/// Push particles closer than the rest spacing apart (a few rounds, each from the same start).
pub fn separate(pos_vel: &mut [[f32; 6]], solid: &[u8], d: Dims, rounds: usize) {
    let n = pos_vel.len();
    if n == 0 {
        return;
    }
    let cells = d.cells();
    let cell_of = |p: &[f32; 6]| d.c((p[0] as usize).min(d.nx - 1), (p[1] as usize).min(d.ny - 1), (p[2] as usize).min(d.nz - 1));
    let min = SPACING * 0.9;
    for _ in 0..rounds {
        // bin particles by cell (counting sort: same order every run)
        let mut start = vec![0u32; cells + 1];
        for p in pos_vel.iter() {
            start[cell_of(p) + 1] += 1;
        }
        for c in 0..cells {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut order = vec![0u32; n];
        for (i, p) in pos_vel.iter().enumerate() {
            let c = cell_of(p);
            order[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        let snapshot: Vec<[f32; 3]> = pos_vel.iter().map(|p| [p[0], p[1], p[2]]).collect();
        pos_vel.par_iter_mut().enumerate().for_each(|(me, p)| {
            let x = snapshot[me];
            let (ci, cj, ck) = ((x[0] as usize).min(d.nx - 1), (x[1] as usize).min(d.ny - 1), (x[2] as usize).min(d.nz - 1));
            let mut push = [0.0f32; 3];
            for k in ck.saturating_sub(1)..(ck + 2).min(d.nz) {
                for j in cj.saturating_sub(1)..(cj + 2).min(d.ny) {
                    for i in ci.saturating_sub(1)..(ci + 2).min(d.nx) {
                        let c = d.c(i, j, k);
                        for &o in &order[start[c] as usize..start[c + 1] as usize] {
                            let o = o as usize;
                            if o == me {
                                continue;
                            }
                            let y = snapshot[o];
                            let dv = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
                            let dd = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                            if dd >= min * min {
                                continue;
                            }
                            let dist = dd.sqrt();
                            // exactly on top of each other: split along a fixed direction by index
                            let dir = if dist > 1e-6 { [dv[0] / dist, dv[1] / dist, dv[2] / dist] } else if me < o { [0.0, 0.0, 1.0] } else { [0.0, 0.0, -1.0] };
                            let s = 0.5 * (min - dist);
                            for a in 0..3 {
                                push[a] += s * dir[a];
                            }
                        }
                    }
                }
            }
            let lim = [d.nx as f32 - 0.01, d.ny as f32 - 0.01, d.nz as f32 - 0.01];
            for a in 0..3 {
                let mut q = [p[0], p[1], p[2]];
                q[a] = (q[a] + push[a].clamp(-0.25, 0.25)).clamp(0.01, lim[a]);
                if !crate::is_solid(solid, d, q) {
                    p[a] = q[a];
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apic_keeps_a_linear_flow_exactly() {
        // a shear flow u = 0.2 * z (in cells) is linear, so APIC must carry it with no error
        let d = Dims { nx: 6, ny: 6, nz: 6 };
        let mut pv: Vec<[f32; 6]> = Vec::new();
        for k in 0..12 {
            for j in 0..12 {
                for i in 0..12 {
                    let p = [0.25 + i as f32 * 0.5, 0.25 + j as f32 * 0.5, 0.25 + k as f32 * 0.5];
                    pv.push([p[0], p[1], p[2], 0.2 * (p[2] - OFF_U[2]), 0.0, 0.0]);
                }
            }
        }
        let mut c = vec![[0.0f32; 9]; pv.len()];
        for c in c.iter_mut() {
            c[2] = 0.2;
        }
        let (mac, _) = p2g(&pv, &c, d);
        let mut back = pv.clone();
        let mut c2 = c.clone();
        g2p(&mac, &mut back, &mut c2);
        for (a, b) in pv.iter().zip(&back) {
            if a[2] > 1.0 && a[2] < 5.0 && a[0] > 1.0 && a[0] < 5.0 {
                assert!((a[3] - b[3]).abs() < 1e-4, "{a:?} -> {b:?}");
            }
        }
        assert!(c2.iter().zip(&pv).filter(|(_, p)| p[2] > 1.0 && p[2] < 5.0 && p[0] > 1.0 && p[0] < 5.0).all(|(c, _)| (c[2] - 0.2).abs() < 1e-4));
    }

    #[test]
    fn crowded_particles_move_apart_and_never_into_solids() {
        let d = Dims { nx: 4, ny: 4, nz: 4 };
        let mut solid = vec![FLUID; d.cells()];
        solid[d.c(2, 1, 1)] = SOLID;
        let mut pv: Vec<[f32; 6]> = (0..20).map(|i| [1.5 + 0.01 * (i % 5) as f32, 1.5 + 0.01 * (i / 5) as f32, 1.5, 0.0, 0.0, 0.0]).collect();
        separate(&mut pv, &solid, d, 30);
        let mut closest = f32::MAX;
        for a in 0..pv.len() {
            for b in a + 1..pv.len() {
                let dd = ((pv[a][0] - pv[b][0]).powi(2) + (pv[a][1] - pv[b][1]).powi(2) + (pv[a][2] - pv[b][2]).powi(2)).sqrt();
                closest = closest.min(dd);
            }
        }
        assert!(closest > 0.3, "spread out: {closest}");
        assert!(pv.iter().all(|p| !crate::is_solid(&solid, d, [p[0], p[1], p[2]])));
    }
}
