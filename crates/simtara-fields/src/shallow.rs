//! Floods and tsunamis: the shallow-water equations over real ground.
//!
//! Water much wider than it is deep (rivers, floods, a tsunami crossing the sea) moves as a
//! sheet: each column of water has one depth `h` and one velocity `(u, v)`, pushed by the slope of
//! its surface. These are the equations flood and tsunami models use (HEC-RAS 2D, TUFLOW,
//! LISFLOOD, GeoClaw, MOST). The method:
//!
//! - finite volumes with an HLL Riemann solver: water is passed between cells exactly, nothing
//!   is lost or made;
//! - hydrostatic reconstruction (Audusse et al. 2004): a lake over bumpy ground stays perfectly
//!   still, and the edge of the water (wet / dry) moves without negative depths;
//! - second order in space (MUSCL, minmod) and time (Heun), so a wave keeps its height over long
//!   distances;
//! - Manning friction (the roughness of the ground), rain, rivers, gates whose height follows a
//!   control (open a dam), and an earthquake that lifts the sea floor (a tsunami).
//!
//! Time steps follow the wave speed (CFL), so a run repeats exactly for the same input.

use crate::Num;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use simtara_core::{Builder, Context, Stage, System, World};
use simtara_things::{Param, Things};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Edge {
    /// Water bounces back.
    #[default]
    Wall,
    /// Water and waves leave freely.
    Open,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rain {
    /// millimetres per hour
    pub rate: Num,
    pub region: Option<[f64; 4]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct River {
    pub at: [f64; 2],
    pub radius: f64,
    /// cubic metres per second
    pub flow: Num,
    /// speed the water comes in with, m/s
    pub velocity: [Num; 2],
}

/// A dam, levee or gate: the ground in a box is at least `height` (metres above the datum).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gate {
    pub region: [f64; 4],
    pub height: Num,
    /// ground under it without the gate
    pub under: Vec<(usize, f64)>,
}

/// An earthquake: the sea floor (and the sea above it) lifts by `uplift` metres over an ellipse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quake {
    pub at: [f64; 2],
    pub size: [f64; 2],
    pub uplift: Num,
    /// seconds into the run it strikes
    pub when: f64,
    /// or: the first tick this control is 1 or more
    pub trigger: Option<String>,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FloodMetric {
    pub name: String,
    /// depth | level | speed | wet | max_depth | arrival
    pub field: String,
    /// sum | mean | max | min | count
    pub op: String,
    pub region: Option<[f64; 4]>,
}

/// What a window needs to draw the flood (made once per step, on the flood's own thread).
#[derive(Clone, Debug, Default)]
pub struct FloodRender {
    pub generation: u64,
    pub nx: usize,
    pub ny: usize,
    pub cell: f32,
    pub bed: Vec<f32>,
    pub depth: Vec<f32>,
    pub speed: Vec<f32>,
    pub building: Vec<u8>,
    pub sea_level: f32,
}

static RENDER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[derive(Default)]
struct Job(Option<std::thread::JoinHandle<Flood>>);

impl Clone for Job {
    fn clone(&self) -> Job {
        Job(None)
    }
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Job({})", if self.0.is_some() { "running" } else { "idle" })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Flood {
    pub nx: usize,
    pub ny: usize,
    /// metres
    pub cell: f64,
    /// simulated seconds per tick
    pub dt: f64,
    pub gravity: f64,
    /// Manning roughness, s/m^(1/3) (0.03 = grass, 0.015 = concrete, 0.1 = forest)
    pub manning: Num,
    /// ground height (m above the datum) and water depth (m), momentum (m²/s), per cell
    pub bed: Vec<f64>,
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    pub building: Vec<u8>,
    /// deepest water seen, and when each cell first got wet (s; -1 not yet, -2 wet at the start)
    pub max_depth: Vec<f32>,
    pub arrival: Vec<f32>,
    /// west (x = 0), east, south (y = 0), north
    pub edges: [Edge; 4],
    pub rain: Vec<Rain>,
    pub rivers: Vec<River>,
    pub gates: Vec<Gate>,
    pub quakes: Vec<Quake>,
    pub metrics: Vec<FloodMetric>,
    /// sea level (for telling land from sea), metres
    pub sea_level: f64,
    /// depth that counts as wet, metres
    pub wet: f64,
    /// how much the 3D view stretches heights (1 = true; a coast kilometres wide looks flat otherwise)
    #[serde(default)]
    pub vertical: f64,
    pub seconds: f64,
    pub substeps: usize,
    /// water added by rain and rivers, m³
    pub added_m3: f64,
    pub started: bool,
    #[serde(default)]
    pending: Option<BTreeMap<String, Param>>,
    #[serde(skip)]
    job: Job,
    #[serde(skip)]
    changed: bool,
    #[serde(skip)]
    pub want_render: bool,
    #[serde(skip)]
    pub render: Option<std::sync::Arc<FloodRender>>,
}

/// One side of a cell, reconstructed: depth, ground, velocity.
#[derive(Clone, Copy, Debug, Default)]
struct Side {
    h: f64,
    z: f64,
    u: f64,
    v: f64,
}

#[inline]
fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 { 0.0 } else if a.abs() < b.abs() { a } else { b }
}

/// HLL flux across a face (normal velocity u, along-face v), with the along-face momentum
/// carried by the mass flux (upwind).
#[inline]
fn hll(l: Side, r: Side, g: f64) -> [f64; 3] {
    if l.h <= 0.0 && r.h <= 0.0 {
        return [0.0; 3];
    }
    let (cl, cr) = ((g * l.h).sqrt(), (g * r.h).sqrt());
    // dry side: the wave edge moves at u +- 2c (Toro)
    let sl = if l.h <= 0.0 { r.u - 2.0 * cr } else { (l.u - cl).min(r.u - cr) };
    let sr = if r.h <= 0.0 { l.u + 2.0 * cl } else { (l.u + cl).max(r.u + cr) };
    let fl = [l.h * l.u, l.h * l.u * l.u + 0.5 * g * l.h * l.h];
    let fr = [r.h * r.u, r.h * r.u * r.u + 0.5 * g * r.h * r.h];
    let f = if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        let k = 1.0 / (sr - sl);
        [(sr * fl[0] - sl * fr[0] + sl * sr * (r.h - l.h)) * k, (sr * fl[1] - sl * fr[1] + sl * sr * (r.h * r.u - l.h * l.u)) * k]
    };
    let v = if f[0] >= 0.0 { l.v } else { r.v };
    [f[0], f[1], f[0] * v]
}

impl Flood {
    pub fn ready(&self) -> bool {
        self.nx > 0
    }

    pub fn busy(&self) -> bool {
        self.job.0.as_ref().is_some_and(|h| !h.is_finished())
    }

    pub fn create(&mut self, nx: usize, ny: usize, cell: f64, dt: f64) {
        let n = nx * ny;
        let old = std::mem::take(self);
        *self = Flood {
            nx,
            ny,
            cell,
            dt,
            gravity: 9.81,
            manning: Num::N(0.03),
            bed: vec![0.0; n],
            h: vec![0.0; n],
            hu: vec![0.0; n],
            hv: vec![0.0; n],
            building: vec![0; n],
            max_depth: vec![0.0; n],
            arrival: vec![-1.0; n],
            edges: [Edge::Open; 4],
            wet: 0.01,
            vertical: 1.0,
            metrics: old.metrics,
            want_render: old.want_render,
            changed: true,
            ..Flood::default()
        };
    }

    /// Size in metres.
    pub fn extent(&self) -> [f64; 2] {
        [self.nx as f64 * self.cell, self.ny as f64 * self.cell]
    }

    fn centre(&self, c: usize) -> [f64; 2] {
        [((c % self.nx) as f64 + 0.5) * self.cell, ((c / self.nx) as f64 + 0.5) * self.cell]
    }

    fn inside(&self, c: usize, region: Option<[f64; 4]>) -> bool {
        match region {
            None => true,
            Some(b) => {
                let p = self.centre(c);
                p[0] >= b[0].min(b[2]) && p[0] <= b[0].max(b[2]) && p[1] >= b[1].min(b[3]) && p[1] <= b[1].max(b[3])
            }
        }
    }

    /// Thin sheets of water (thinner than this) have their speed tamed (Kurganov & Petrova):
    /// a film a hair thick cannot race along.
    fn eps(&self) -> f64 {
        (1e-4 * self.cell).max(1e-5)
    }

    #[inline]
    fn vel(h: f64, hu: f64, eps: f64) -> f64 {
        if h <= 1e-10 {
            return 0.0;
        }
        let h4 = h * h * h * h;
        hu * h * std::f64::consts::SQRT_2 / (h4 + h4.max(eps * eps * eps * eps)).sqrt()
    }

    /// The state of cell (i, j) with i, j possibly one outside the box (a ghost: a wall mirrors
    /// the water and reverses the speed across it; an open edge copies it). `axis` 0 = x, 1 = y:
    /// returned with u along that axis and v across it.
    #[inline]
    fn state(&self, h: &[f64], hu: &[f64], hv: &[f64], i: isize, j: isize, axis: usize) -> (f64, f64, f64, f64) {
        let (nx, ny) = (self.nx as isize, self.ny as isize);
        let (ci, cj) = (i.clamp(0, nx - 1), j.clamp(0, ny - 1));
        let c = (cj * nx + ci) as usize;
        let eps = self.eps();
        let (mut u, mut v) = (Self::vel(h[c], hu[c], eps), Self::vel(h[c], hv[c], eps));
        if i < 0 && self.edges[0] == Edge::Wall || i >= nx && self.edges[1] == Edge::Wall {
            u = -u;
        }
        if j < 0 && self.edges[2] == Edge::Wall || j >= ny && self.edges[3] == Edge::Wall {
            v = -v;
        }
        let (a, b) = if axis == 0 { (u, v) } else { (v, u) };
        // past an open edge the ground keeps sloping as it did (so a river runs out at its own
        // pace instead of meeting a flat step); behind a wall it is the same ground mirrored
        let mut z = self.bed[c];
        let open_x = (i < 0 && self.edges[0] == Edge::Open) || (i >= nx && self.edges[1] == Edge::Open);
        let open_y = (j < 0 && self.edges[2] == Edge::Open) || (j >= ny && self.edges[3] == Edge::Open);
        if open_x && nx > 1 {
            let inner = (cj * nx + if i < 0 { 1 } else { nx - 2 }) as usize;
            z = 2.0 * z - self.bed[inner];
        }
        if open_y && ny > 1 {
            let inner = ((if j < 0 { 1 } else { ny - 2 }) * nx + ci) as usize;
            z = 2.0 * z - self.bed[inner];
        }
        (h[c], z, a, b)
    }

    /// The two sides (low and high along `axis`) of cell (i, j), second order where the water
    /// around is wet, first order at the edge of the water.
    #[inline]
    fn recon(&self, h: &[f64], hu: &[f64], hv: &[f64], i: isize, j: isize, axis: usize) -> (Side, Side) {
        let (di, dj) = if axis == 0 { (1, 0) } else { (0, 1) };
        let me = self.state(h, hu, hv, i, j, axis);
        let flat = Side { h: me.0, z: me.1, u: me.2, v: me.3 };
        let inside = i >= 0 && j >= 0 && i < self.nx as isize && j < self.ny as isize;
        if !inside {
            return (flat, flat);
        }
        let lo = self.state(h, hu, hv, i - di, j - dj, axis);
        let hi = self.state(h, hu, hv, i + di, j + dj, axis);
        let dry = 1e-6;
        if me.0 <= dry || lo.0 <= dry || hi.0 <= dry {
            return (flat, flat);
        }
        let sh = 0.5 * minmod(me.0 - lo.0, hi.0 - me.0);
        let se = 0.5 * minmod(me.0 + me.1 - lo.0 - lo.1, hi.0 + hi.1 - me.0 - me.1);
        let su = 0.5 * minmod(me.2 - lo.2, hi.2 - me.2);
        let sv = 0.5 * minmod(me.3 - lo.3, hi.3 - me.3);
        let (h0, h1) = (me.0 - sh, me.0 + sh);
        let eta = me.0 + me.1;
        (Side { h: h0, z: eta - se - h0, u: me.2 - su, v: me.3 - sv }, Side { h: h1, z: eta + se - h1, u: me.2 + su, v: me.3 + sv })
    }

    /// The water just outside edge `e` (west, east, south, north) facing side `s`.
    #[inline]
    fn ghost(&self, s: Side, e: usize) -> Side {
        if self.edges[e] == Edge::Wall { Side { u: -s.u, ..s } } else { s }
    }

    /// Flux through the face between `l` (low side's high face) and `r`, hydrostatically
    /// reconstructed, and the two pressure corrections (Audusse et al. 2004).
    #[inline]
    fn face(l: Side, r: Side, g: f64) -> ([f64; 3], f64, f64) {
        let zs = l.z.max(r.z);
        let hl = (l.h + l.z - zs).max(0.0);
        let hr = (r.h + r.z - zs).max(0.0);
        let f = hll(Side { h: hl, ..l }, Side { h: hr, ..r }, g);
        (f, 0.5 * g * (l.h * l.h - hl * hl), 0.5 * g * (r.h * r.h - hr * hr))
    }

    /// d(h, hu, hv)/dt for every cell, for an Euler step of `sdt` seconds. Fluxes are worked out
    /// once per face, so what leaves one cell enters the next exactly; a cell can never send out
    /// more water than it holds in the step (the "draining" limit of Bollermann et al. 2013),
    /// which keeps every depth at zero or above with no water made.
    fn rates(&self, h: &[f64], hu: &[f64], hv: &[f64], sdt: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (nx, ny) = (self.nx, self.ny);
        let g = self.gravity;
        let dx = self.cell;
        let n = nx * ny;
        // the two sides of every cell along x and along y
        let rec: Vec<[(Side, Side); 2]> = (0..n).into_par_iter().map(|c| {
            let (i, j) = ((c % nx) as isize, (c / nx) as isize);
            [self.recon(h, hu, hv, i, j, 0), self.recon(h, hu, hv, i, j, 1)]
        }).collect();
        let side = |i: isize, j: isize, axis: usize| -> (Side, Side) {
            if i >= 0 && j >= 0 && i < nx as isize && j < ny as isize { rec[j as usize * nx + i as usize][axis] } else { self.recon(h, hu, hv, i, j, axis) }
        };
        // face fluxes: (mass, normal momentum, along momentum, pressure fix low side, high side)
        let fx: Vec<[f64; 5]> = (0..(nx + 1) * ny).into_par_iter().map(|e| {
            let (i, j) = ((e % (nx + 1)) as isize, (e / (nx + 1)) as isize);
            // at the box's edge the outside is the inside mirrored (a wall) or copied (open)
            let (l, r) = if i == 0 {
                let r = side(0, j, 0).0;
                (self.ghost(r, 0), r)
            } else if i == nx as isize {
                let l = side(i - 1, j, 0).1;
                (l, self.ghost(l, 1))
            } else {
                (side(i - 1, j, 0).1, side(i, j, 0).0)
            };
            if l.h <= 0.0 && r.h <= 0.0 { return [0.0, 0.5 * g * (l.h * l.h), 0.0, 0.0, 0.0]; }
            let (f, cl, cr) = Self::face(l, r, g);
            [f[0], f[1], f[2], cl, cr]
        }).collect();
        let fy: Vec<[f64; 5]> = (0..nx * (ny + 1)).into_par_iter().map(|e| {
            let (i, j) = ((e % nx) as isize, (e / nx) as isize);
            let (l, r) = if j == 0 {
                let r = side(i, 0, 1).0;
                (self.ghost(r, 2), r)
            } else if j == ny as isize {
                let l = side(i, j - 1, 1).1;
                (l, self.ghost(l, 3))
            } else {
                (side(i, j - 1, 1).1, side(i, j, 1).0)
            };
            if l.h <= 0.0 && r.h <= 0.0 { return [0.0, 0.0, 0.0, 0.0, 0.0]; }
            let (f, cl, cr) = Self::face(l, r, g);
            [f[0], f[1], f[2], cl, cr]
        }).collect();
        // how much of its outflow each cell can afford this step
        let k = sdt / dx;
        let afford: Vec<f64> = (0..n).into_par_iter().map(|c| {
            let (i, j) = (c % nx, c / nx);
            let out = fx[j * (nx + 1) + i + 1][0].max(0.0) + (-fx[j * (nx + 1) + i][0]).max(0.0) + fy[(j + 1) * nx + i][0].max(0.0) + (-fy[j * nx + i][0]).max(0.0);
            let out = out * k;
            if out > h[c] && out > 0.0 { h[c] / out } else { 1.0 }
        }).collect();
        let scale_x = |e: usize| -> f64 {
            let (i, j) = (e % (nx + 1), e / (nx + 1));
            let f = fx[e][0];
            if f > 0.0 && i > 0 { afford[j * nx + i - 1] } else if f < 0.0 && i < nx { afford[j * nx + i] } else { 1.0 }
        };
        let scale_y = |e: usize| -> f64 {
            let (i, j) = (e % nx, e / nx);
            let f = fy[e][0];
            if f > 0.0 && j > 0 { afford[(j - 1) * nx + i] } else if f < 0.0 && j < ny { afford[j * nx + i] } else { 1.0 }
        };
        let mut dh = vec![0.0; n];
        let mut dhu = vec![0.0; n];
        let mut dhv = vec![0.0; n];
        dh.par_iter_mut().zip(dhu.par_iter_mut()).zip(dhv.par_iter_mut()).enumerate().for_each(|(c, ((rh, ru), rv))| {
            let (i, j) = (c % nx, c / nx);
            let (w, e) = (j * (nx + 1) + i, j * (nx + 1) + i + 1);
            let (s, nn) = (j * nx + i, (j + 1) * nx + i);
            let (sw, se, ss, sn) = (scale_x(w), scale_x(e), scale_y(s), scale_y(nn));
            let (fw, fe, fs, fnn) = (&fx[w], &fx[e], &fy[s], &fy[nn]);
            // the slope of the ground inside the cell (second-order source)
            let [(xl, xh), (yl, yh)] = rec[c];
            let scx = -0.5 * g * (xl.h + xh.h) * (xh.z - xl.z);
            let scy = -0.5 * g * (yl.h + yh.h) * (yh.z - yl.z);
            *rh = -((se * fe[0] - sw * fw[0]) + (sn * fnn[0] - ss * fs[0])) / dx;
            *ru = -((se * fe[1] + fe[3] - sw * fw[1] - fw[4] - scx) + (sn * fnn[2] - ss * fs[2])) / dx;
            *rv = -((sn * fnn[1] + fnn[3] - ss * fs[1] - fs[4] - scy) + (se * fe[2] - sw * fw[2])) / dx;
        });
        (dh, dhu, dhv)
    }
    /// Largest (|u| + c) + (|v| + c) anywhere, m/s.
    fn wave_speed(&self) -> f64 {
        let eps = self.eps();
        let g = self.gravity;
        (0..self.h.len())
            .into_par_iter()
            .map(|c| {
                let h = self.h[c];
                if h <= 1e-6 {
                    return 0.0;
                }
                let s = (g * h).sqrt();
                Self::vel(h, self.hu[c], eps).abs() + Self::vel(h, self.hv[c], eps).abs() + 2.0 * s
            })
            .reduce(|| 0.0, f64::max)
    }

    /// One tick, reading controls from `params`.
    pub fn step(&mut self, params: &BTreeMap<String, Param>) {
        if !self.ready() {
            return;
        }
        if !self.started {
            // what is wet at the start is not "flooded"
            for c in 0..self.h.len() {
                if self.h[c] > self.wet {
                    self.arrival[c] = -2.0;
                }
            }
            self.started = true;
        }
        self.apply_gates(params);
        self.apply_quakes(params);
        let manning = self.manning.get(params).max(0.0) as f64;
        let rain: Vec<(f64, Option<[f64; 4]>)> = self.rain.iter().map(|r| (r.rate.get(params).max(0.0) as f64 / 1000.0 / 3600.0, r.region)).collect();
        let rivers: Vec<(Vec<usize>, f64, [f64; 2])> = self
            .rivers
            .iter()
            .map(|r| {
                let cells: Vec<usize> = (0..self.h.len()).filter(|&c| {
                    let p = self.centre(c);
                    (p[0] - r.at[0]).powi(2) + (p[1] - r.at[1]).powi(2) <= r.radius.max(self.cell * 0.5).powi(2)
                }).collect();
                (cells, r.flow.get(params) as f64, [r.velocity[0].get(params) as f64, r.velocity[1].get(params) as f64])
            })
            .collect();
        let mut left = self.dt;
        let mut count = 0usize;
        let eps = self.eps();
        while left > 1e-12 && count < 5000 {
            let a = self.wave_speed();
            let sdt = if a > 0.0 { (0.45 * self.cell / a).min(left) } else { left };
            // a little left over is folded into this step rather than taken as a tiny one
            let sdt = if left - sdt < 0.05 * sdt { left } else { sdt };
            // Heun: two Euler steps, averaged
            let (h0, hu0, hv0) = (self.h.clone(), self.hu.clone(), self.hv.clone());
            let (a1, b1, c1) = self.rates(&h0, &hu0, &hv0, sdt);
            let mut h1: Vec<f64> = h0.par_iter().zip(&a1).map(|(h, d)| (h + sdt * d).max(0.0)).collect();
            let mut hu1: Vec<f64> = hu0.par_iter().zip(&b1).map(|(q, d)| q + sdt * d).collect();
            let mut hv1: Vec<f64> = hv0.par_iter().zip(&c1).map(|(q, d)| q + sdt * d).collect();
            tame(&mut h1, &mut hu1, &mut hv1, eps);
            let (a2, b2, c2) = self.rates(&h1, &hu1, &hv1, sdt);
            self.h = (0..h0.len()).into_par_iter().map(|c| (0.5 * (h0[c] + h1[c] + sdt * a2[c])).max(0.0)).collect();
            self.hu = (0..h0.len()).into_par_iter().map(|c| 0.5 * (hu0[c] + hu1[c] + sdt * b2[c])).collect();
            self.hv = (0..h0.len()).into_par_iter().map(|c| 0.5 * (hv0[c] + hv1[c] + sdt * c2[c])).collect();
            // friction: Manning, implicit so it can never reverse the flow
            if manning > 0.0 {
                let k = self.gravity * manning * manning * sdt;
                let (h, hu, hv) = (&self.h, &mut self.hu, &mut self.hv);
                hu.par_iter_mut().zip(hv.par_iter_mut()).zip(h.par_iter()).for_each(|((qu, qv), h)| {
                    if *h > 1e-6 {
                        let speed = (*qu * *qu + *qv * *qv).sqrt() / h;
                        let f = 1.0 + k * speed / h.powf(4.0 / 3.0);
                        *qu /= f;
                        *qv /= f;
                    }
                });
            }
            // rain and rivers
            for (rate, region) in &rain {
                if *rate > 0.0 {
                    for c in 0..self.h.len() {
                        if self.building[c] == 0 && self.inside(c, *region) {
                            self.h[c] += rate * sdt;
                            self.added_m3 += rate * sdt * self.cell * self.cell;
                        }
                    }
                }
            }
            for (cells, q, vel) in &rivers {
                if cells.is_empty() || *q == 0.0 {
                    continue;
                }
                let dh = q * sdt / (cells.len() as f64 * self.cell * self.cell);
                for &c in cells {
                    let dh = dh.max(-self.h[c]);
                    self.h[c] += dh;
                    if dh > 0.0 {
                        self.hu[c] += dh * vel[0];
                        self.hv[c] += dh * vel[1];
                    }
                    self.added_m3 += dh * self.cell * self.cell;
                }
            }
            let (mut h, mut hu, mut hv) = (std::mem::take(&mut self.h), std::mem::take(&mut self.hu), std::mem::take(&mut self.hv));
            tame(&mut h, &mut hu, &mut hv, eps);
            self.h = h;
            self.hu = hu;
            self.hv = hv;
            left -= sdt;
            self.seconds += sdt;
            count += 1;
            // flood records
            let (wet, t) = (self.wet, self.seconds as f32);
            let h = &self.h;
            self.max_depth.par_iter_mut().zip(self.arrival.par_iter_mut()).zip(h.par_iter()).for_each(|((m, a), h)| {
                *m = m.max(*h as f32);
                if *a == -1.0 && *h > wet {
                    *a = t;
                }
            });
        }
        self.substeps = count;
    }

    fn apply_gates(&mut self, params: &BTreeMap<String, Param>) {
        for g in &self.gates {
            let top = g.height.get(params) as f64;
            for &(c, under) in &g.under {
                let z = under.max(top);
                if z != self.bed[c] {
                    // the water there keeps its surface where it can (a gate rising pushes it up)
                    self.bed[c] = z;
                }
            }
        }
    }

    fn apply_quakes(&mut self, params: &BTreeMap<String, Param>) {
        for qi in 0..self.quakes.len() {
            let q = self.quakes[qi].clone();
            if q.done {
                continue;
            }
            let due = match &q.trigger {
                Some(name) => params.get(name).is_some_and(|p| p.value >= 1.0),
                None => self.seconds + 1e-9 >= q.when,
            };
            if !due {
                continue;
            }
            let lift = q.uplift.get(params) as f64;
            for c in 0..self.bed.len() {
                let p = self.centre(c);
                let r = (((p[0] - q.at[0]) / q.size[0].max(1e-9)).powi(2) + ((p[1] - q.at[1]) / q.size[1].max(1e-9)).powi(2)).sqrt();
                if r < 1.0 {
                    // a smooth dome: the floor lifts, the water column rides on it
                    let s = 0.5 * (1.0 + (std::f64::consts::PI * r).cos());
                    self.bed[c] += lift * s;
                }
            }
            self.quakes[qi].done = true;
        }
    }

    /// A solitary wave (the classic model of a tsunami's leading wave) of height `height`,
    /// centred `at` metres along `dir`, travelling along it, on still water of depth `depth`.
    pub fn add_solitary(&mut self, height: f64, at: f64, dir: [f64; 2], depth: f64) {
        let l = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-12);
        let dir = [dir[0] / l, dir[1] / l];
        let k = (3.0 * height / (4.0 * depth.powi(3))).sqrt();
        let c = (self.gravity * (depth + height)).sqrt();
        for cell in 0..self.h.len() {
            if self.h[cell] <= 1e-6 {
                continue;
            }
            let p = self.centre(cell);
            let s = p[0] * dir[0] + p[1] * dir[1] - at;
            let eta = height / (k * s).cosh().powi(2);
            if eta < 1e-9 * height {
                continue;
            }
            let d = self.h[cell];
            self.h[cell] = d + eta;
            let u = c * eta / (d + eta);
            self.hu[cell] = (d + eta) * u * dir[0];
            self.hv[cell] = (d + eta) * u * dir[1];
        }
    }

    pub fn volume(&self) -> f64 {
        self.h.par_iter().sum::<f64>() * self.cell * self.cell
    }

    fn cell_value(&self, field: &str, c: usize) -> f64 {
        let eps = self.eps();
        match field {
            "depth" => self.h[c],
            "level" => self.h[c] + self.bed[c],
            "speed" => {
                let (u, v) = (Self::vel(self.h[c], self.hu[c], eps), Self::vel(self.h[c], self.hv[c], eps));
                (u * u + v * v).sqrt()
            }
            "wet" => (self.h[c] > self.wet) as u8 as f64,
            "max_depth" => self.max_depth[c] as f64,
            "arrival" => self.arrival[c] as f64,
            "ground" => self.bed[c],
            _ => 0.0,
        }
    }

    pub fn evaluate(&self, m: &FloodMetric) -> f64 {
        if !self.ready() {
            return 0.0;
        }
        let area = self.cell * self.cell;
        let cells: Vec<usize> = (0..self.h.len()).filter(|&c| self.inside(c, m.region)).collect();
        let vals: Vec<f64> = if m.field == "level" || m.field == "speed" {
            // only where there is water
            cells.iter().filter(|&&c| self.h[c] > self.wet).map(|&c| self.cell_value(&m.field, c)).collect()
        } else if m.field == "arrival" {
            cells.iter().map(|&c| self.arrival[c] as f64).filter(|t| *t >= 0.0).collect()
        } else {
            cells.iter().map(|&c| self.cell_value(&m.field, c)).collect()
        };
        match m.op.as_str() {
            // sums are amounts: depth -> cubic metres, wet -> square metres
            "sum" if m.field == "depth" || m.field == "max_depth" => vals.iter().sum::<f64>() * area,
            "sum" if m.field == "wet" => vals.iter().sum::<f64>() * area,
            "sum" => vals.iter().sum(),
            "mean" => if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 },
            "max" => vals.iter().cloned().fold(f64::MIN, f64::max).max(if m.field == "arrival" { -1.0 } else { f64::MIN }),
            "min" => if vals.is_empty() { -1.0 } else { vals.iter().cloned().fold(f64::MAX, f64::min) },
            "count" => vals.iter().filter(|v| **v > if m.field == "arrival" { -1.0 } else { 1e-9 }).count() as f64,
            _ => 0.0,
        }
    }

    pub fn summary(&self) -> Vec<(String, f64)> {
        if !self.ready() {
            return Vec::new();
        }
        let area = self.cell * self.cell;
        let eps = self.eps();
        let wet = self.wet;
        let land: Vec<usize> = (0..self.h.len()).filter(|&c| self.arrival[c] != -2.0).collect();
        let flooded = land.iter().filter(|&&c| self.h[c] > wet).count() as f64 * area;
        let ever = land.iter().filter(|&&c| self.max_depth[c] as f64 > wet).count() as f64 * area;
        let runup = land.iter().filter(|&&c| self.max_depth[c] as f64 > wet && self.building[c] == 0).map(|&c| self.bed[c]).fold(f64::MIN, f64::max);
        let deepest = land.iter().map(|&c| self.h[c]).fold(0.0, f64::max);
        let fastest = (0..self.h.len()).into_par_iter().map(|c| if self.h[c] > wet { (Self::vel(self.h[c], self.hu[c], eps).powi(2) + Self::vel(self.h[c], self.hv[c], eps).powi(2)).sqrt() } else { 0.0 }).reduce(|| 0.0, f64::max);
        let mut out = vec![
            ("flood.seconds".to_string(), self.seconds),
            ("flood.volume_m3".into(), self.volume()),
            ("flood.flooded_land_m2".into(), flooded),
            ("flood.ever_flooded_m2".into(), ever),
            ("flood.deepest_on_land_m".into(), deepest),
            ("flood.fastest_m_s".into(), fastest),
            ("flood.substeps".into(), self.substeps as f64),
        ];
        if runup > f64::MIN {
            out.push(("flood.highest_reached_m".into(), runup - self.sea_level));
        }
        out
    }

    pub fn make_render(&mut self) {
        if !self.want_render || !self.ready() {
            return;
        }
        let eps = self.eps();
        self.render = Some(std::sync::Arc::new(FloodRender {
            generation: RENDER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            nx: self.nx,
            ny: self.ny,
            cell: self.cell as f32,
            bed: self.bed.iter().map(|x| *x as f32).collect(),
            depth: self.h.iter().map(|x| *x as f32).collect(),
            speed: (0..self.h.len()).into_par_iter().map(|c| (Self::vel(self.h[c], self.hu[c], eps).powi(2) + Self::vel(self.h[c], self.hv[c], eps).powi(2)).sqrt() as f32).collect(),
            building: self.building.clone(),
            sea_level: self.sea_level as f32,
        }));
    }

    /// Same pipeline as `Fluid::tick`: take the step computed in the background, start the next.
    pub fn tick(&mut self, params: &BTreeMap<String, Param>, background: bool) {
        if let Some(h) = self.job.0.take() {
            let done = h.join().expect("flood step failed");
            if self.changed {
                let p = self.pending.take().unwrap_or_default();
                self.step(&p);
            } else {
                *self = done;
            }
        } else if let Some(p) = self.pending.take() {
            self.step(&p);
        }
        self.pending = None;
        self.changed = false;
        if !self.ready() {
            return;
        }
        if background {
            let mut next = self.clone();
            let p = params.clone();
            self.pending = Some(p.clone());
            self.job = Job(Some(std::thread::spawn(move || {
                next.step(&p);
                next.pending = None;
                next.make_render();
                next
            })));
        } else {
            self.step(params);
            self.make_render();
        }
    }

    /// Change the ground: `f(x, y, old) -> new` for every cell (metres).
    fn shape_ground(&mut self, f: impl Fn(f64, f64, f64) -> f64) {
        for c in 0..self.bed.len() {
            let p = self.centre(c);
            self.bed[c] = f(p[0], p[1], self.bed[c]);
        }
    }
}

/// Keep thin sheets sensible: no water below zero, no speed in dry cells, the speed of a film
/// recomputed from the tamed velocity.
fn tame(h: &mut [f64], hu: &mut [f64], hv: &mut [f64], eps: f64) {
    h.par_iter_mut().zip(hu.par_iter_mut()).zip(hv.par_iter_mut()).for_each(|((h, qu), qv)| {
        if *h <= 1e-8 {
            *h = h.max(0.0);
            *qu = 0.0;
            *qv = 0.0;
        } else if *h < eps {
            *qu = *h * Flood::vel(*h, *qu, eps);
            *qv = *h * Flood::vel(*h, *qv, eps);
        }
    });
}

/// Smooth random hills (value noise, a few octaves), the same for the same seed.
fn noise(x: f64, y: f64, seed: u64) -> f64 {
    let hash = |i: i64, j: i64| -> f64 {
        let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let one = |x: f64, y: f64| {
        let (i, j) = (x.floor() as i64, y.floor() as i64);
        let (fx, fy) = (x - i as f64, y - j as f64);
        let s = |t: f64| t * t * (3.0 - 2.0 * t);
        let (sx, sy) = (s(fx), s(fy));
        let a = hash(i, j) * (1.0 - sx) + hash(i + 1, j) * sx;
        let b = hash(i, j + 1) * (1.0 - sx) + hash(i + 1, j + 1) * sx;
        a * (1.0 - sy) + b * sy
    };
    let (mut total, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for _ in 0..4 {
        total += amp * one(x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / norm
}

fn dist_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    ((p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)).sqrt()
}

struct FloodSystem;

impl System for FloodSystem {
    fn name(&self) -> &str {
        "fields.flood"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(mut f) = ctx.world.take::<Flood>() else { return };
        if f.ready() {
            let empty = BTreeMap::new();
            let params = ctx.world.get::<Things>().map_or(&empty, |t| &t.params);
            f.tick(params, true);
        }
        ctx.world.put(f);
    }
}

pub(crate) fn install(b: &mut Builder) {
    b.world.insert_persistent("flood", Flood::default());
    register_commands(b.world);
    b.add_system(Stage::Update, FloodSystem);
    b.add_metric_source(|w| {
        let Some(f) = w.get::<Flood>() else { return Vec::new() };
        let mut out = f.summary();
        out.extend(f.metrics.iter().map(|m| (m.name.clone(), f.evaluate(m))));
        out
    });
}

fn flood_mut(w: &mut World) -> Result<&mut Flood, String> {
    let f = w.get_mut::<Flood>().ok_or_else(|| "this universe has no floods".to_string())?;
    if !f.ready() {
        return Err("make the flood first: flood{size = {200, 150}, cell = 10}".into());
    }
    f.changed = true;
    Ok(f)
}

fn nums(a: &Value, key: &str, n: usize) -> Result<Option<Vec<f64>>, String> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(v)) if (n == 0 || v.len() == n) && v.iter().all(Value::is_number) => Ok(Some(v.iter().map(|x| x.as_f64().unwrap()).collect())),
        Some(_) if n == 0 => Err(format!("'{key}' must be a list of numbers")),
        Some(_) => Err(format!("'{key}' must be {n} numbers, e.g. {key} = {{{}}}", vec!["1"; n].join(", "))),
    }
}

fn number(a: &Value, key: &str) -> Result<Option<Num>, String> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(Some(Num::N(n.as_f64().unwrap_or(0.0)))),
        Some(Value::String(s)) => Ok(Some(Num::P(s.clone()))),
        Some(_) => Err(format!("'{key}' must be a number or the name of a control (param)")),
    }
}

fn edge(v: &Value) -> Result<Edge, String> {
    match v.as_str() {
        Some("open") => Ok(Edge::Open),
        Some("wall") => Ok(Edge::Wall),
        _ => Err("an edge is \"open\" (water leaves) or \"wall\" (it bounces back)".into()),
    }
}

fn register_commands(w: &mut World) {
    // flood{size = {nx, ny}, cell = metres, dt = seconds per tick, manning = 0.03, edges = "open" | "wall" | {west=, east=, south=, north=}, sea_level = 0, wet = 0.01}
    w.register_command("flood", |w, a| {
        let f = w.get_mut::<Flood>().ok_or("this universe has no floods")?;
        f.changed = true;
        if let Some(size) = nums(a, "size", 2)? {
            let n = |x: f64| (x.round() as usize).clamp(1, 4096);
            let (nx, ny) = (n(size[0]), n(size[1]));
            if nx * ny > 4_000_000 {
                return Err(format!("{} cells is too many (at most 4 million)", nx * ny));
            }
            let cell = a.get("cell").and_then(Value::as_f64).unwrap_or(10.0);
            let dt = a.get("dt").and_then(Value::as_f64).unwrap_or(1.0);
            if cell <= 0.0 || dt <= 0.0 {
                return Err("'cell' and 'dt' must be above zero".into());
            }
            f.create(nx, ny, cell, dt);
        } else if !f.ready() {
            return Err("the first flood{...} needs a size, e.g. flood{size = {200, 150}, cell = 10}".into());
        }
        if let Some(m) = number(a, "manning")? {
            f.manning = m;
        }
        if let Some(g) = a.get("gravity").and_then(Value::as_f64) {
            f.gravity = g;
        }
        if let Some(s) = a.get("sea_level").and_then(Value::as_f64) {
            f.sea_level = s;
        }
        if let Some(v) = a.get("vertical").and_then(Value::as_f64) {
            f.vertical = v.clamp(0.1, 100.0);
        }
        if let Some(s) = a.get("wet").and_then(Value::as_f64) {
            f.wet = s.max(1e-6);
        }
        match a.get("edges") {
            None | Some(Value::Null) => {}
            Some(Value::String(_)) => f.edges = [edge(&a["edges"])?; 4],
            Some(Value::Object(o)) => {
                for (k, slot) in [("west", 0), ("east", 1), ("south", 2), ("north", 3)] {
                    if let Some(v) = o.get(k) {
                        f.edges[slot] = edge(v)?;
                    }
                }
            }
            Some(_) => return Err("edges = \"open\", \"wall\" or {west = ..., east = ..., south = ..., north = ...}".into()),
        }
        Ok(json!({"cells": f.nx * f.ny, "size_m": f.extent()}))
    });
    // ground{height = h} | {ramp = {x0, y0, h0, x1, y1, h1}} | {hill = {x, y, radius, height}} |
    // {box = {x0, y0, x1, y1}, height = h, building = true} | {noise = amplitude, scale = metres, seed = 1} |
    // {channel = {x0, y0, x1, y1, ...}, width = w, depth = d}; op = "set" | "add" | "max" | "min";
    // within = {x0, y0, x1, y1}: change only that part
    w.register_command("ground", |w, a| {
        let f = flood_mut(w)?;
        let op = a.get("op").and_then(Value::as_str).unwrap_or("").to_string();
        let apply = move |old: f64, new: f64, default: &str| match if op.is_empty() { default } else { op.as_str() } {
            "add" => old + new,
            "max" => old.max(new),
            "min" => old.min(new),
            _ => new,
        };
        let height = a.get("height").and_then(Value::as_f64);
        // only inside this box (metres), when given
        let within = nums(a, "within", 4)?.map(|b| [b[0], b[1], b[2], b[3]]);
        let keep: Vec<f64> = f.bed.clone();
        let cells_out: Vec<usize> = (0..f.bed.len()).filter(|&c| !f.inside(c, within)).collect();
        let result = (|| -> Result<Value, String> {
        if let Some(r) = nums(a, "ramp", 6)? {
            let (ax, ay, bx, by) = (r[0], r[1], r[3], r[4]);
            let (dx, dy) = (bx - ax, by - ay);
            let l2 = (dx * dx + dy * dy).max(1e-12);
            f.shape_ground(|x, y, old| {
                let t = (((x - ax) * dx + (y - ay) * dy) / l2).clamp(0.0, 1.0);
                apply(old, r[2] + t * (r[5] - r[2]), "set")
            });
        } else if let Some(hl) = nums(a, "hill", 4)? {
            f.shape_ground(|x, y, old| {
                let d = ((x - hl[0]).powi(2) + (y - hl[1]).powi(2)).sqrt() / hl[2].max(1e-9);
                let s = if d < 1.0 { 0.5 * (1.0 + (std::f64::consts::PI * d).cos()) } else { 0.0 };
                apply(old, hl[3] * s, "add")
            });
        } else if let Some(b) = nums(a, "box", 4)? {
            let hgt = height.ok_or("a box needs height = metres")?;
            let building = a.get("building").and_then(Value::as_bool).unwrap_or(false);
            let region = Some([b[0], b[1], b[2], b[3]]);
            let mut n = 0;
            for c in 0..f.bed.len() {
                if f.inside(c, region) {
                    f.bed[c] = if building { f.bed[c] + hgt } else { apply(f.bed[c], hgt, "set") };
                    if building {
                        f.building[c] = 1;
                        f.h[c] = 0.0;
                        f.hu[c] = 0.0;
                        f.hv[c] = 0.0;
                    }
                    n += 1;
                }
            }
            return Ok(json!(n));
        } else if let Some(amp) = a.get("noise").and_then(Value::as_f64) {
            let scale = a.get("scale").and_then(Value::as_f64).unwrap_or(500.0).max(1e-6);
            let seed = a.get("seed").and_then(Value::as_u64).unwrap_or(1);
            f.shape_ground(|x, y, old| apply(old, amp * noise(x / scale, y / scale, seed), "add"));
        } else if let Some(pts) = nums(a, "channel", 0)? {
            if pts.len() < 4 || pts.len() % 2 != 0 {
                return Err("channel = {x0, y0, x1, y1, ...}: two or more points".into());
            }
            let width = a.get("width").and_then(Value::as_f64).ok_or("a channel needs width = metres")?;
            let depth = a.get("depth").and_then(Value::as_f64).ok_or("a channel needs depth = metres")?;
            let segs: Vec<([f64; 2], [f64; 2])> = pts.chunks(2).collect::<Vec<_>>().windows(2).map(|w| ([w[0][0], w[0][1]], [w[1][0], w[1][1]])).collect();
            f.shape_ground(|x, y, old| {
                let d = segs.iter().map(|(a, b)| dist_to_segment([x, y], *a, *b)).fold(f64::MAX, f64::min);
                let t = (d / (0.5 * width)).min(1.0);
                old - depth * (1.0 - t * t)
            });
        } else if let Some(hgt) = height {
            f.shape_ground(|_, _, old| apply(old, hgt, "set"));
        } else {
            return Err("ground needs height, ramp, hill, box, noise or channel".into());
        }
        Ok(json!(true))
        })();
        for c in cells_out {
            f.bed[c] = keep[c];
        }
        result
    });
    // water_level{level = metres, box = {x0, y0, x1, y1}?}: fill everything lower than the level (a sea, a lake)
    w.register_command("water_level", |w, a| {
        let f = flood_mut(w)?;
        let level = a.get("level").and_then(Value::as_f64).ok_or("water_level needs level = metres")?;
        let region = nums(a, "box", 4)?.map(|b| [b[0], b[1], b[2], b[3]]);
        let mut n = 0;
        for c in 0..f.h.len() {
            if f.building[c] == 0 && f.inside(c, region) && f.bed[c] < level && f.h[c] < level - f.bed[c] {
                f.h[c] = level - f.bed[c];
                n += 1;
            }
        }
        Ok(json!(n))
    });
    // rain{rate = mm per hour, box = {...}?}
    w.register_command("rain", |w, a| {
        let f = flood_mut(w)?;
        let rate = number(a, "rate")?.ok_or("rain needs rate = millimetres per hour")?;
        f.rain.push(Rain { rate, region: nums(a, "box", 4)?.map(|b| [b[0], b[1], b[2], b[3]]) });
        Ok(json!(f.rain.len()))
    });
    // river{at = {x, y}, radius = metres, flow = m³/s, velocity = {vx, vy}}
    w.register_command("river", |w, a| {
        let f = flood_mut(w)?;
        let at = nums(a, "at", 2)?.ok_or("river needs at = {x, y} (metres)")?;
        let flow = number(a, "flow")?.ok_or("river needs flow = cubic metres per second")?;
        let velocity = match a.get("velocity") {
            Some(Value::Array(v)) if v.len() == 2 => {
                let one = |x: &Value| match x {
                    Value::Number(n) => Ok(Num::N(n.as_f64().unwrap_or(0.0))),
                    Value::String(s) => Ok(Num::P(s.clone())),
                    _ => Err("velocity parts must be numbers or control names".to_string()),
                };
                [one(&v[0])?, one(&v[1])?]
            }
            None | Some(Value::Null) => [Num::N(0.0), Num::N(0.0)],
            Some(_) => return Err("velocity = {vx, vy} in m/s".into()),
        };
        let radius = a.get("radius").and_then(Value::as_f64).unwrap_or(f.cell);
        f.rivers.push(River { at: [at[0], at[1]], radius, flow, velocity });
        Ok(json!(f.rivers.len()))
    });
    // gate{box = {x0, y0, x1, y1}, height = metres or a control}: a dam, levee or gate whose top follows a control
    w.register_command("gate", |w, a| {
        let f = flood_mut(w)?;
        let b = nums(a, "box", 4)?.ok_or("gate needs box = {x0, y0, x1, y1} (metres)")?;
        let height = number(a, "height")?.ok_or("gate needs height = metres (or a control)")?;
        let region = [b[0], b[1], b[2], b[3]];
        let under: Vec<(usize, f64)> = (0..f.bed.len()).filter(|&c| f.inside(c, Some(region))).map(|c| (c, f.bed[c])).collect();
        let n = under.len();
        f.gates.push(Gate { region, height, under });
        Ok(json!(n))
    });
    // wave{height = metres, at = metres along the direction, direction = {dx, dy}, depth = metres}: a solitary wave
    w.register_command("wave", |w, a| {
        let f = flood_mut(w)?;
        let height = a.get("height").and_then(Value::as_f64).ok_or("wave needs height = metres")?;
        let at = a.get("at").and_then(Value::as_f64).ok_or("wave needs at = metres (where its crest is, along its direction)")?;
        let dir = nums(a, "direction", 2)?.unwrap_or(vec![1.0, 0.0]);
        let depth = a.get("depth").and_then(Value::as_f64).ok_or("wave needs depth = metres (the still water it travels on)")?;
        if height <= 0.0 || depth <= 0.0 {
            return Err("height and depth must be above zero".into());
        }
        f.add_solitary(height, at, [dir[0], dir[1]], depth);
        Ok(json!(true))
    });
    // quake{at = {x, y}, size = {a, b}, uplift = metres, when = seconds | trigger = "control"}
    w.register_command("quake", |w, a| {
        let f = flood_mut(w)?;
        let at = nums(a, "at", 2)?.ok_or("quake needs at = {x, y} (metres)")?;
        let size = nums(a, "size", 2)?.ok_or("quake needs size = {a, b}: the half widths of the lifted area, metres")?;
        let uplift = number(a, "uplift")?.ok_or("quake needs uplift = metres (or a control)")?;
        f.quakes.push(Quake { at: [at[0], at[1]], size: [size[0], size[1]], uplift, when: a.get("when").and_then(Value::as_f64).unwrap_or(0.0), trigger: a.get("trigger").and_then(Value::as_str).map(str::to_string), done: false });
        Ok(json!(f.quakes.len()))
    });
    // flood_metric{name, field = depth | level | speed | wet | max_depth | arrival | ground, op = sum | mean | max | min | count, box = {...}?}
    w.register_command("flood_metric", |w, a| {
        let f = w.get_mut::<Flood>().ok_or("this universe has no floods")?;
        f.changed = true;
        let name = a.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("flood_metric needs a name")?.to_string();
        let field = a.get("field").and_then(Value::as_str).unwrap_or("depth").to_string();
        if !["depth", "level", "speed", "wet", "max_depth", "arrival", "ground"].contains(&field.as_str()) {
            return Err(format!("field '{field}': use depth, level, speed, wet, max_depth, arrival or ground"));
        }
        let op = a.get("op").and_then(Value::as_str).unwrap_or("max").to_string();
        if !["sum", "mean", "max", "min", "count"].contains(&op.as_str()) {
            return Err(format!("op '{op}': use sum, mean, max, min or count"));
        }
        let region = nums(a, "box", 4)?.map(|b| [b[0], b[1], b[2], b[3]]);
        f.metrics.retain(|m| m.name != name);
        f.metrics.push(FloodMetric { name, field, op, region });
        Ok(json!(true))
    });
    w.register_command("flood_info", |w, _| {
        let f = w.get::<Flood>().filter(|f| f.ready()).ok_or("make the flood first: flood{size = {200, 150}, cell = 10}")?;
        Ok(json!({"cells": f.nx * f.ny, "cell_m": f.cell, "dt_s": f.dt, "size_m": f.extent(), "seconds": f.seconds, "volume_m3": f.volume(), "substeps": f.substeps, "lowest_ground_m": f.bed.iter().cloned().fold(f64::MAX, f64::min), "highest_ground_m": f.bed.iter().cloned().fold(f64::MIN, f64::max)}))
    });
}

#[cfg(test)]
#[path = "shallow_tests.rs"]
mod tests;
