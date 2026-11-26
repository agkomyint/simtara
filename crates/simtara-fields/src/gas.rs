//! Real gases: air and the gases mixed into it (CO2, helium, methane, ...), at low speed.
//!
//! The method is the *low-Mach* one used by fire and building-airflow codes such as NIST FDS:
//! below about 100 m/s the pressure hardly changes, so sound waves are left out, but the density
//! changes freely with temperature and with which gas is where (ideal gas law,
//! rho = p M / (R T)). That gives, with no tuning numbers:
//!
//! - buoyancy: hot air and light gases rise, heavy gases (CO2, propane, SF6) sink and pool on the
//!   floor, each by exactly g (rho0 - rho) / rho;
//! - a pressure solve weighted by 1 / rho, so a light gas is pushed further than a heavy one by
//!   the same pressure difference;
//! - expansion: air that is heated takes more room (div u = DT/Dt / T - DM/Dt / M), so a fire
//!   pushes air outwards as well as up;
//! - mixing by molecular diffusion plus turbulence: a Smagorinsky large-eddy model adds the
//!   mixing of the swirls smaller than a cell (nu_t = (Cs h)^2 |S|), also used for heat
//!   (turbulent Prandtl number 0.5) and gases (Schmidt number 0.5), as in FDS.

use crate::grid::{idx, Dims, Mac, Sides, SOLID};
use rayon::prelude::*;

/// Universal gas constant, J/(mol K).
pub const R: f32 = 8.314_462;
/// Pressure of the air around, Pa (sea level).
pub const P0: f32 = 101_325.0;
/// Molar mass of dry air, kg/mol.
pub const M_AIR: f32 = 0.028_965;
/// Kinematic viscosity of air at 20 °C, m²/s.
pub const NU_AIR: f32 = 1.516e-5;
/// How fast heat spreads in still air at 20 °C, m²/s.
pub const ALPHA_AIR: f32 = 2.2e-5;
/// Turbulent Prandtl and Schmidt numbers (how much the swirls mix heat and gas, relative to momentum).
pub const PR_T: f32 = 0.5;
pub const SC_T: f32 = 0.5;

/// A gas: name, molar mass (kg/mol), diffusion coefficient in air at 20 °C (m²/s), and the colour
/// it is drawn with (real gases are mostly invisible: this is a tint so you can see where it goes).
#[derive(Clone, Copy, Debug)]
pub struct GasKind {
    pub name: &'static str,
    pub molar_mass: f32,
    pub diffusion: f32,
    pub colour: [f32; 3],
}

/// Gases that can be mixed into the air (standard tables: CRC Handbook, Cussler).
pub const GASES: [GasKind; 11] = [
    GasKind { name: "co2", molar_mass: 0.044_01, diffusion: 1.6e-5, colour: [0.45, 0.85, 0.40] },
    GasKind { name: "helium", molar_mass: 0.004_003, diffusion: 7.0e-5, colour: [1.0, 0.55, 0.85] },
    GasKind { name: "hydrogen", molar_mass: 0.002_016, diffusion: 7.6e-5, colour: [0.55, 0.85, 1.0] },
    GasKind { name: "methane", molar_mass: 0.016_04, diffusion: 2.2e-5, colour: [0.50, 0.65, 1.0] },
    GasKind { name: "propane", molar_mass: 0.044_10, diffusion: 1.0e-5, colour: [1.0, 0.70, 0.30] },
    GasKind { name: "steam", molar_mass: 0.018_015, diffusion: 2.5e-5, colour: [0.95, 0.95, 0.97] },
    GasKind { name: "sf6", molar_mass: 0.146_06, diffusion: 0.9e-5, colour: [0.70, 0.50, 1.0] },
    GasKind { name: "argon", molar_mass: 0.039_95, diffusion: 1.9e-5, colour: [0.60, 0.60, 1.0] },
    GasKind { name: "chlorine", molar_mass: 0.070_90, diffusion: 1.2e-5, colour: [0.80, 0.95, 0.25] },
    GasKind { name: "nitrogen", molar_mass: 0.028_013, diffusion: 2.0e-5, colour: [0.80, 0.80, 0.85] },
    GasKind { name: "oxygen", molar_mass: 0.031_999, diffusion: 2.0e-5, colour: [0.55, 0.75, 1.0] },
];

/// Find a gas by name (also "carbon dioxide", "he", "h2", "ch4", "water vapour", ...).
pub fn find(name: &str) -> Option<usize> {
    let n = name.trim().to_lowercase();
    let n = match n.as_str() {
        "carbon dioxide" | "co₂" => "co2",
        "he" => "helium",
        "h2" => "hydrogen",
        "ch4" | "natural gas" => "methane",
        "c3h8" | "lpg" => "propane",
        "water vapour" | "water vapor" | "vapour" | "vapor" | "h2o" => "steam",
        "ar" => "argon",
        "cl2" => "chlorine",
        "n2" => "nitrogen",
        "o2" => "oxygen",
        other => other,
    };
    GASES.iter().position(|g| g.name == n)
}

pub fn names() -> String {
    GASES.iter().map(|g| g.name).collect::<Vec<_>>().join(", ")
}

/// Density of air at the ambient temperature, kg/m³.
pub fn rho_air(t_ambient: f32) -> f32 {
    P0 * M_AIR / (R * t_ambient)
}

/// Molar mass of the mixture in cell `c`, kg/mol. The gases are stored as *volume* (mole)
/// fractions `x[g][c]`: ideal gases mix by volume, so mixing two cells' contents (by the flow or
/// by diffusion) changes neither the total volume nor the mass, and a mole fraction is what a
/// gas detector reads (1 = pure, 0.01 = 1 %).
#[inline]
pub fn molar_mass(x: &[Vec<f32>], kinds: &[usize], c: usize) -> f32 {
    let mut rest = 1.0f32;
    let mut m = 0.0f32;
    for (g, &k) in kinds.iter().enumerate() {
        let f = x[g][c].clamp(0.0, 1.0);
        rest -= f;
        m += f * GASES[k].molar_mass;
    }
    m + rest.max(0.0) * M_AIR
}

/// Density of one cell, kg/m³ (temperature in kelvin above `t0`).
#[inline]
pub fn cell_density(temp: f32, x: &[Vec<f32>], kinds: &[usize], c: usize, t0: f32) -> f32 {
    P0 * molar_mass(x, kinds, c) / (R * (t0 + temp).max(50.0))
}

/// Density of every cell, kg/m³, from its temperature (kelvin above `t0`) and gas mix.
pub fn density(temp: &[f32], x: &[Vec<f32>], kinds: &[usize], t0: f32) -> Vec<f32> {
    (0..temp.len()).into_par_iter().map(|c| cell_density(temp[c], x, kinds, c, t0)).collect()
}

/// Volume (mole) fraction of gas `g` in cell `c` (1 = pure, 0.01 = 1 %).
pub fn volume_fraction(x: &[Vec<f32>], _kinds: &[usize], g: usize, c: usize) -> f32 {
    x[g][c].clamp(0.0, 1.0)
}

/// Density on every face (average of the two cells; walls keep the cell's own), and the
/// projection weights rho0 / rho_face.
pub fn faces(rho: &[f32], d: Dims, rho0: f32) -> ([Vec<f32>; 3], crate::grid::Beta) {
    let mut out: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (axis, dims) in [d.ud(), d.vd(), d.wd()].into_iter().enumerate() {
        out[axis] = (0..crate::grid::len(dims))
            .into_par_iter()
            .map(|f| {
                let (i, j, k) = (f % dims[0], (f / dims[0]) % dims[1], f / (dims[0] * dims[1]));
                let (lo, hi) = match axis {
                    0 => ((i > 0).then(|| d.c(i - 1, j, k)), (i < d.nx).then(|| d.c(i, j, k))),
                    1 => ((j > 0).then(|| d.c(i, j - 1, k)), (j < d.ny).then(|| d.c(i, j, k))),
                    _ => ((k > 0).then(|| d.c(i, j, k - 1)), (k < d.nz).then(|| d.c(i, j, k))),
                };
                match (lo, hi) {
                    (Some(a), Some(b)) => 0.5 * (rho[a] + rho[b]),
                    (Some(a), None) | (None, Some(a)) => rho[a],
                    _ => rho0,
                }
            })
            .collect();
    }
    let beta = [out[0].par_iter().map(|r| rho0 / r).collect(), out[1].par_iter().map(|r| rho0 / r).collect(), out[2].par_iter().map(|r| rho0 / r).collect()];
    (out, beta)
}

/// Smagorinsky eddy viscosity per cell, m²/s: (cs h)² |S|, |S| = sqrt(2 S_ij S_ij).
pub fn eddy_viscosity(mac: &Mac, solid: &[u8], cs: f32, h: f32) -> Vec<f32> {
    let d = mac.d;
    let n = d.cells();
    if cs <= 0.0 {
        return vec![0.0; n];
    }
    let vc: Vec<[f32; 3]> = (0..n).into_par_iter().map(|c| mac.centre(c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny))).collect();
    let k2 = (cs * h) * (cs * h);
    (0..n)
        .into_par_iter()
        .map(|c| {
            if solid[c] == SOLID {
                return 0.0;
            }
            let (i, j, k) = (c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny));
            // diagonal parts straight from the faces
            let dudx = (mac.u[idx(d.ud(), i + 1, j, k)] - mac.u[idx(d.ud(), i, j, k)]) / h;
            let dvdy = (mac.v[idx(d.vd(), i, j + 1, k)] - mac.v[idx(d.vd(), i, j, k)]) / h;
            let dwdz = (mac.w[idx(d.wd(), i, j, k + 1)] - mac.w[idx(d.wd(), i, j, k)]) / h;
            // cross parts from the cell-centre velocities around (one-sided at the edges)
            let grad = |comp: usize, axis: usize| -> f32 {
                let (lo, hi, span) = match axis {
                    0 => (d.c(i.saturating_sub(1), j, k), d.c((i + 1).min(d.nx - 1), j, k), ((i + 1).min(d.nx - 1) - i.saturating_sub(1)) as f32),
                    1 => (d.c(i, j.saturating_sub(1), k), d.c(i, (j + 1).min(d.ny - 1), k), ((j + 1).min(d.ny - 1) - j.saturating_sub(1)) as f32),
                    _ => (d.c(i, j, k.saturating_sub(1)), d.c(i, j, (k + 1).min(d.nz - 1)), ((k + 1).min(d.nz - 1) - k.saturating_sub(1)) as f32),
                };
                if span == 0.0 { 0.0 } else { (vc[hi][comp] - vc[lo][comp]) / (span * h) }
            };
            let sxy = 0.5 * (grad(0, 1) + grad(1, 0));
            let sxz = 0.5 * (grad(0, 2) + grad(2, 0));
            let syz = 0.5 * (grad(1, 2) + grad(2, 1));
            let s2 = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz) + 4.0 * (sxy * sxy + sxz * sxz + syz * syz);
            k2 * s2.sqrt()
        })
        .collect()
}

/// Viscosity: each face velocity diffuses with the viscosity `nu0 + nu_t` (m²/s) of the cells
/// it sits between, explicitly, in as many small steps as needed to stay stable. The walls of
/// the box hold the air still (no-slip, a wall half a cell beyond the last face) or let it slide,
/// and the top wall can move at `sides.lid` (lid-driven cavity).
pub fn diffuse_velocity(mac: &mut Mac, flags: &[u8], sides: Sides, nu0: f32, nu_t: &[f32], dt: f32, h: f32) {
    let d = mac.d;
    let nu_max = nu0 + nu_t.par_iter().cloned().reduce(|| 0.0, f32::max);
    let alpha_max = nu_max * dt / (h * h);
    if alpha_max <= 1e-7 {
        return;
    }
    let steps = ((alpha_max / 0.15).ceil() as usize).clamp(1, 400);
    let sdt = dt / steps as f32;
    for _ in 0..steps {
        for axis in 0..3 {
            let dims = [d.ud(), d.vd(), d.wd()][axis];
            let src = match axis {
                0 => mac.u.clone(),
                1 => mac.v.clone(),
                _ => mac.w.clone(),
            };
            let dst = match axis {
                0 => &mut mac.u,
                1 => &mut mac.v,
                _ => &mut mac.w,
            };
            dst.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(k, slab)| {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let p = [i, j, k];
                        // faces on the box's sides along their own axis, and faces touching a
                        // solid, are set by the walls (enforce), not by diffusion
                        if p[axis] == 0 || p[axis] == dims[axis] - 1 {
                            continue;
                        }
                        let (lo, hi) = match axis {
                            0 => (d.c(i - 1, j, k), d.c(i, j, k)),
                            1 => (d.c(i, j - 1, k), d.c(i, j, k)),
                            _ => (d.c(i, j, k - 1), d.c(i, j, k)),
                        };
                        if flags[lo] == SOLID || flags[hi] == SOLID {
                            continue;
                        }
                        let nu = nu0 + 0.5 * (nu_t[lo] + nu_t[hi]);
                        let c = idx(dims, i, j, k);
                        let me = src[c];
                        let mut lap = 0.0f32;
                        for b in 0..3 {
                            let stride = match b {
                                0 => 1,
                                1 => dims[0],
                                _ => dims[0] * dims[1],
                            };
                            for up in [false, true] {
                                let inside = if up { p[b] + 1 < dims[b] } else { p[b] > 0 };
                                let v = if inside {
                                    src[if up { c + stride } else { c - stride }]
                                } else if b == axis {
                                    me
                                } else {
                                    // a wall of the box beside this face: no-slip mirrors the
                                    // velocity about the wall's own speed (a box one cell thick
                                    // is a 2D slice: nothing to stick to in that direction)
                                    let open = (b == 2 && up && sides.open_top) || (b == 0 && up && sides.open_xmax) || d.cd()[b] == 1;
                                    let wall_speed = if axis == 0 && b == 2 && up { sides.lid } else { 0.0 };
                                    if sides.no_slip && !open { 2.0 * wall_speed - me } else { me }
                                };
                                lap += v - me;
                            }
                        }
                        slab[i + dims[0] * j] = me + nu * sdt / (h * h) * lap;
                    }
                }
            });
        }
    }
}

/// Diffuse a cell-centred quantity with diffusivity `k0 + k_t[c]` (m²/s); solids and the box's
/// sides let nothing through.
pub fn diffuse_scalar(a: &mut [f32], flags: &[u8], d: Dims, k0: f32, k_t: &[f32], scale_t: f32, dt: f32, h: f32) {
    let kmax = k0 + scale_t * k_t.par_iter().cloned().reduce(|| 0.0, f32::max);
    let alpha = kmax * dt / (h * h);
    if alpha <= 1e-7 {
        return;
    }
    let steps = ((alpha / 0.15).ceil() as usize).clamp(1, 400);
    let sdt = dt / steps as f32;
    let r = sdt / (h * h);
    for _ in 0..steps {
        let src = a.to_vec();
        a.par_chunks_mut(d.nx * d.ny).enumerate().for_each(|(k, slab)| {
            for j in 0..d.ny {
                for i in 0..d.nx {
                    let c = d.c(i, j, k);
                    if flags[c] == SOLID {
                        continue;
                    }
                    let kc = k0 + scale_t * k_t[c];
                    let mut flux = 0.0f32;
                    let mut nb = |ok: bool, o: usize| {
                        if ok && flags[o] != SOLID {
                            let kf = 0.5 * (kc + k0 + scale_t * k_t[o]);
                            flux += kf * (src[o] - src[c]);
                        }
                    };
                    nb(i > 0, c.wrapping_sub(1));
                    nb(i + 1 < d.nx, c + 1);
                    nb(j > 0, c.wrapping_sub(d.nx));
                    nb(j + 1 < d.ny, c + d.nx);
                    nb(k > 0, c.wrapping_sub(d.nx * d.ny));
                    nb(k + 1 < d.nz, c + d.nx * d.ny);
                    slab[i + d.nx * j] = src[c] + r * flux;
                }
            }
        });
    }
}

/// Centre-line u of the lid-driven cavity at Re = 100, from Ghia, Ghia & Shin (1982): (y, u / U_lid).
pub const GHIA_RE100_U: [(f32, f32); 17] = [
    (0.0000, 0.00000),
    (0.0547, -0.03717),
    (0.0625, -0.04192),
    (0.0703, -0.04775),
    (0.1016, -0.06434),
    (0.1719, -0.10150),
    (0.2813, -0.15662),
    (0.4531, -0.21090),
    (0.5000, -0.20581),
    (0.6172, -0.13641),
    (0.7344, 0.00332),
    (0.8516, 0.23151),
    (0.9531, 0.68717),
    (0.9609, 0.73722),
    (0.9688, 0.78871),
    (0.9766, 0.84123),
    (1.0000, 1.00000),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ideal_gas_law_gives_the_densities_in_the_tables() {
        let t0 = 293.15;
        let air = rho_air(t0);
        assert!((air - 1.204).abs() < 0.005, "air {air}");
        let one = |name: &str| {
            let g = find(name).unwrap();
            let y = vec![vec![1.0f32]];
            density(&[0.0], &y, &[g], t0)[0]
        };
        // CRC handbook at 20 °C, 1 atm
        for (name, want) in [("co2", 1.842), ("helium", 0.1664), ("methane", 0.668), ("propane", 1.83), ("sf6", 6.07)] {
            let got = one(name);
            assert!((got - want).abs() / want < 0.02, "{name}: {got} vs {want}");
        }
        // hot air at 100 °C
        let hot = density(&[80.0], &[], &[], t0)[0];
        assert!((hot - 0.946).abs() < 0.01, "{hot}");
        // half and half by volume: CO2 in air, the mean of the two densities
        let g = find("carbon dioxide").unwrap();
        let half = density(&[0.0], &[vec![0.5]], &[g], t0)[0];
        assert!((half - 0.5 * (air + one("co2"))).abs() < 1e-3, "{half}");
    }
}
