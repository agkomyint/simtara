//! Particle System v0 — the first module on the SimTara runtime.
//!
//! Rules (each a separate system, run in this order):
//!   1. forces    — pairwise attraction / repulsion (per-kind matrix, optional charge term)
//!   2. integrate — friction, speed limit, move
//!   3. boundary  — bounce off walls or wrap around
//!
//! All tunables live in `ParticleParams`, a persistent resource: edit it while
//! the universe runs and the next tick uses the new values.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simtara_core::{Backend, Builder, Context, Module, Rng, Simulation, Stage, System, World};

pub const MODULE: &str = "particles";

/// Acceleration scale: force (dimensionless, -1..1) -> world units / s^2, per unit of `rmax`.
const FORCE_SCALE: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    Bounce,
    Wrap,
}

/// Live-tunable laws of this universe.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleParams {
    pub width: f32,
    pub height: f32,
    pub kinds: usize,
    /// Row-major kinds x kinds; row = the particle feeling the force, col = the one exerting it.
    /// Empty means "generate from seed using ParticleConfig".
    pub attraction: Vec<f32>,
    /// Global strength of all pair forces (the "gravity" dial).
    pub gravity: f32,
    pub friction: f32,
    /// Interaction radius.
    pub rmax: f32,
    /// Fraction of rmax inside which every pair repels.
    pub repel_core: f32,
    /// Extra charge term: like charges repel, opposite attract (scaled by this).
    pub coulomb: f32,
    pub max_speed: f32,
    pub boundary: Boundary,
}

impl Default for ParticleParams {
    fn default() -> Self {
        ParticleParams {
            width: 1000.0,
            height: 1000.0,
            kinds: 4,
            attraction: Vec::new(),
            gravity: 1.0,
            friction: 6.0,
            rmax: 60.0,
            repel_core: 0.3,
            coulomb: 0.0,
            max_speed: 400.0,
            boundary: Boundary::Bounce,
        }
    }
}

/// How to create the universe (startup only).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleConfig {
    pub count: usize,
    /// Mean attraction between kinds.
    pub base_attraction: f32,
    /// Random +/- variation in attraction strength between kind pairs.
    pub spread: f32,
    pub params: ParticleParams,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        ParticleConfig { count: 5000, base_attraction: 0.0, spread: 0.8, params: ParticleParams::default() }
    }
}

/// Struct-of-arrays particle state (GPU-friendly layout).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Particles {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub mass: Vec<f32>,
    pub charge: Vec<f32>,
    pub kind: Vec<u8>,
}

impl Particles {
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    fn random(count: usize, p: &ParticleParams, rng: &mut Rng) -> Self {
        let mut s = Particles::default();
        for _ in 0..count {
            let k = rng.below(p.kinds.max(1));
            s.x.push(rng.range(0.0, p.width));
            s.y.push(rng.range(0.0, p.height));
            s.vx.push(0.0);
            s.vy.push(0.0);
            s.mass.push(1.0);
            s.charge.push((k % 3) as f32 - 1.0);
            s.kind.push(k as u8);
        }
        s
    }
}

pub struct ParticleModule;

impl Module for ParticleModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        let cfg: ParticleConfig = b.config.module(MODULE);
        let mut rng = Rng::new(b.config.seed);
        let mut params = cfg.params.clone();
        params.kinds = params.kinds.clamp(1, 255);
        if params.attraction.len() != params.kinds * params.kinds {
            params.attraction = (0..params.kinds * params.kinds)
                .map(|_| (cfg.base_attraction + cfg.spread * rng.range(-1.0, 1.0)).clamp(-1.0, 1.0))
                .collect();
        }
        let particles = Particles::random(cfg.count, &params, &mut rng);
        b.world.insert_persistent("particles", particles);
        b.world.insert_persistent("particle_params", params);
        b.add_system(Stage::Update, ForceSystem);
        b.add_system(Stage::Update, IntegrateSystem);
        b.add_system(Stage::Update, BoundarySystem);
    }
}

/// Convenience: a simulation with the particle module installed.
pub fn simulation(mut config: simtara_core::Config, pc: &ParticleConfig) -> Simulation {
    config.set_module(MODULE, pc);
    let mut sim = Simulation::new(config);
    sim.add_module(ParticleModule);
    sim
}

// ---------------------------------------------------------------- systems

struct ForceSystem;
impl System for ForceSystem {
    fn name(&self) -> &str {
        "particles.forces"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(mut p) = ctx.world.take::<Particles>() else { return };
        if let Some(prm) = ctx.world.get::<ParticleParams>() {
            // A GPU implementation would be selected here; today only CPU is supported.
            match ctx.compute.resolve(&[Backend::Cpu]) {
                _ => forces_cpu(&mut p, prm, ctx.dt as f32),
            }
        }
        ctx.world.put(p);
    }
}

fn forces_cpu(p: &mut Particles, prm: &ParticleParams, dt: f32) {
    let n = p.len();
    let kinds = prm.kinds;
    if n == 0 || prm.attraction.len() != kinds * kinds {
        return;
    }
    let rmax = prm.rmax.max(1e-3);
    let (w, h) = (prm.width, prm.height);
    let wrap = prm.boundary == Boundary::Wrap;
    let beta = prm.repel_core.clamp(0.01, 0.95);

    // uniform grid, cell >= rmax so only the 3x3 neighbourhood matters
    let gw = ((w / rmax).floor() as usize).max(1);
    let gh = ((h / rmax).floor() as usize).max(1);
    let (cw, ch) = (w / gw as f32, h / gh as f32);
    let cells: Vec<usize> = (0..n)
        .map(|i| {
            let cx = ((p.x[i] / cw) as isize).clamp(0, gw as isize - 1) as usize;
            let cy = ((p.y[i] / ch) as isize).clamp(0, gh as isize - 1) as usize;
            cy * gw + cx
        })
        .collect();
    let mut start = vec![0u32; gw * gh + 1];
    for &c in &cells {
        start[c + 1] += 1;
    }
    for i in 0..gw * gh {
        start[i + 1] += start[i];
    }
    let mut fill = start[..gw * gh].to_vec();
    let mut order = vec![0u32; n];
    for (i, &c) in cells.iter().enumerate() {
        order[fill[c] as usize] = i as u32;
        fill[c] += 1;
    }

    let (px, py, kind, charge, mass) = (&p.x, &p.y, &p.kind, &p.charge, &p.mass);
    let r2max = rmax * rmax;
    let acc: Vec<[f32; 2]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (xi, yi) = (px[i], py[i]);
            let ki = (kind[i] as usize).min(kinds - 1);
            let (cx, cy) = ((cells[i] % gw) as isize, (cells[i] / gw) as isize);
            let mut seen = [usize::MAX; 9];
            let mut ns = 0;
            let (mut ax, mut ay) = (0.0f32, 0.0f32);
            for oy in -1..=1isize {
                for ox in -1..=1isize {
                    let (mut nx, mut ny) = (cx + ox, cy + oy);
                    if wrap {
                        nx = nx.rem_euclid(gw as isize);
                        ny = ny.rem_euclid(gh as isize);
                    } else if nx < 0 || ny < 0 || nx >= gw as isize || ny >= gh as isize {
                        continue;
                    }
                    let cid = ny as usize * gw + nx as usize;
                    if seen[..ns].contains(&cid) {
                        continue; // small grids wrap onto the same cell twice
                    }
                    seen[ns] = cid;
                    ns += 1;
                    for k in start[cid]..start[cid + 1] {
                        let j = order[k as usize] as usize;
                        if j == i {
                            continue;
                        }
                        let (mut dx, mut dy) = (px[j] - xi, py[j] - yi);
                        if wrap {
                            if dx > w * 0.5 {
                                dx -= w
                            } else if dx < -w * 0.5 {
                                dx += w
                            }
                            if dy > h * 0.5 {
                                dy -= h
                            } else if dy < -h * 0.5 {
                                dy += h
                            }
                        }
                        let r2 = dx * dx + dy * dy;
                        if r2 >= r2max || r2 < 1e-12 {
                            continue;
                        }
                        let r = r2.sqrt();
                        let t = r / rmax;
                        let kj = (kind[j] as usize).min(kinds - 1);
                        let a = prm.attraction[ki * kinds + kj] - prm.coulomb * charge[i] * charge[j];
                        let f = if t < beta {
                            t / beta - 1.0 // universal short-range repulsion
                        } else {
                            a * (1.0 - (1.0 + beta - 2.0 * t).abs() / (1.0 - beta))
                        };
                        let s = f / r;
                        ax += dx * s;
                        ay += dy * s;
                    }
                }
            }
            let g = prm.gravity * FORCE_SCALE * rmax / mass[i].max(1e-6);
            [ax * g, ay * g]
        })
        .collect();

    for i in 0..n {
        p.vx[i] += acc[i][0] * dt;
        p.vy[i] += acc[i][1] * dt;
    }
}

struct IntegrateSystem;
impl System for IntegrateSystem {
    fn name(&self) -> &str {
        "particles.integrate"
    }
    fn run(&mut self, ctx: &mut Context) {
        let dt = ctx.dt as f32;
        let Some(prm) = ctx.world.get::<ParticleParams>().cloned() else { return };
        let Some(p) = ctx.world.get_mut::<Particles>() else { return };
        let damp = (-prm.friction * dt).exp();
        let vmax2 = prm.max_speed * prm.max_speed;
        for i in 0..p.len() {
            let (mut vx, mut vy) = (p.vx[i] * damp, p.vy[i] * damp);
            let v2 = vx * vx + vy * vy;
            if v2 > vmax2 {
                let s = prm.max_speed / v2.sqrt();
                vx *= s;
                vy *= s;
            }
            p.vx[i] = vx;
            p.vy[i] = vy;
            p.x[i] += vx * dt;
            p.y[i] += vy * dt;
        }
    }
}

struct BoundarySystem;
impl System for BoundarySystem {
    fn name(&self) -> &str {
        "particles.boundary"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(prm) = ctx.world.get::<ParticleParams>().cloned() else { return };
        let Some(p) = ctx.world.get_mut::<Particles>() else { return };
        let (w, h) = (prm.width, prm.height);
        for i in 0..p.len() {
            match prm.boundary {
                Boundary::Wrap => {
                    p.x[i] = p.x[i].rem_euclid(w);
                    p.y[i] = p.y[i].rem_euclid(h);
                }
                Boundary::Bounce => {
                    if p.x[i] < 0.0 {
                        p.x[i] = -p.x[i];
                        p.vx[i] = -p.vx[i];
                    } else if p.x[i] > w {
                        p.x[i] = 2.0 * w - p.x[i];
                        p.vx[i] = -p.vx[i];
                    }
                    if p.y[i] < 0.0 {
                        p.y[i] = -p.y[i];
                        p.vy[i] = -p.vy[i];
                    } else if p.y[i] > h {
                        p.y[i] = 2.0 * h - p.y[i];
                        p.vy[i] = -p.vy[i];
                    }
                    p.x[i] = p.x[i].clamp(0.0, w);
                    p.y[i] = p.y[i].clamp(0.0, h);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- observation

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub count: usize,
    pub kinetic_energy: f32,
    /// Average number of neighbours within rmax/2 (sampled). Rising = clustering.
    pub mean_neighbors: f32,
}

pub fn stats(world: &World) -> Option<Stats> {
    let p = world.get::<Particles>()?;
    let prm = world.get::<ParticleParams>()?;
    let n = p.len();
    if n == 0 {
        return Some(Stats { count: 0, kinetic_energy: 0.0, mean_neighbors: 0.0 });
    }
    let ke = (0..n).map(|i| 0.5 * p.mass[i] * (p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i])).sum::<f32>() / n as f32;
    let r2 = (prm.rmax * 0.5).powi(2);
    let step = (n / 400).max(1);
    let sample: Vec<usize> = (0..n).step_by(step).collect();
    let total: usize = sample
        .par_iter()
        .map(|&i| (0..n).filter(|&j| j != i && (p.x[i] - p.x[j]).powi(2) + (p.y[i] - p.y[j]).powi(2) < r2).count())
        .sum();
    Some(Stats { count: n, kinetic_energy: ke, mean_neighbors: total as f32 / sample.len() as f32 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use simtara_core::Config;

    fn small() -> Simulation {
        simulation(Config::default(), &ParticleConfig { count: 500, ..Default::default() })
    }

    #[test]
    fn particles_stay_in_bounds_both_modes() {
        for b in [Boundary::Bounce, Boundary::Wrap] {
            let mut pc = ParticleConfig { count: 500, ..Default::default() };
            pc.params.boundary = b;
            let mut sim = simulation(Config::default(), &pc);
            sim.advance(120);
            let p = sim.world().get::<Particles>().unwrap();
            assert!(p.x.iter().all(|x| (0.0..=1000.0).contains(x) && x.is_finite()));
            assert!(p.y.iter().all(|y| (0.0..=1000.0).contains(y) && y.is_finite()));
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let (mut a, mut b) = (small(), small());
        a.advance(60);
        b.advance(60);
        assert_eq!(a.world().get::<Particles>().unwrap().x, b.world().get::<Particles>().unwrap().x);
    }

    #[test]
    fn snapshot_restore_resumes_identically() {
        let mut a = small();
        a.advance(30);
        let snap = a.snapshot().unwrap();
        a.advance(30);

        let mut b = small();
        b.restore(&snap).unwrap();
        b.advance(30);
        assert_eq!(a.world().get::<Particles>().unwrap().x, b.world().get::<Particles>().unwrap().x);
        assert_eq!(a.world().clock.tick, b.world().clock.tick);
    }

    #[test]
    fn live_param_change_takes_effect() {
        let mut a = small();
        let mut b = small();
        b.world_mut().get_mut::<ParticleParams>().unwrap().gravity = 3.0;
        a.advance(20);
        b.advance(20);
        assert_ne!(a.world().get::<Particles>().unwrap().x, b.world().get::<Particles>().unwrap().x);
    }

    #[test]
    fn attraction_causes_clustering() {
        let mut pc = ParticleConfig { count: 1500, base_attraction: 0.6, spread: 0.1, ..Default::default() };
        pc.params.boundary = Boundary::Wrap;
        let mut sim = simulation(Config::default(), &pc);
        let before = stats(sim.world()).unwrap().mean_neighbors;
        sim.advance(300);
        let after = stats(sim.world()).unwrap().mean_neighbors;
        assert!(after > before * 2.0, "expected clustering: {before} -> {after}");
    }
}
