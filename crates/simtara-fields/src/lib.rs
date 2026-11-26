//! Fluids: real smoke and real water, in SI units, for any SimTara universe.
//!
//! - **Smoke** uses *stable fluids* on a staggered grid: air carries itself, smoke and heat
//!   along (semi-Lagrangian + MacCormack advection), hot air rises (buoyancy, g * dT / T),
//!   heavy smoke sinks, *vorticity confinement* puts back the small swirls a coarse grid loses,
//!   and a pressure solve keeps the air incompressible so it flows around solids.
//! - **Water** uses *FLIP*: many particles carry the water, the grid makes it incompressible
//!   each step, the particles pick up the change. It splashes, flows around obstacles and settles
//!   flat.
//!
//! A script sets it up (`fluid{...}`, `solid{...}`, `emitter{...}`, `water{...}`) and any
//! number in it may name a `param{...}` slider instead, so a control changes the fluid live.
//! Runs repeat exactly for a seed, on one core or many.

pub mod gpu;
pub mod grid;
pub mod liquid;
pub mod mesh;

use grid::{advect, enforce, extrapolate, idx, project, Dims, Mac, Sides, AIR, FLUID, OFF_U, OFF_V, OFF_W, SOLID};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use simtara_core::{Builder, Context, Module, Stage, System, World};
use simtara_things::{Param, Things};
use std::collections::BTreeMap;

pub const MODULE: &str = "fields";
/// Air temperature the heat is measured from, kelvin (20 °C).
pub const AMBIENT_K: f32 = 293.15;

/// A number, or the name of a `param{...}` control whose value is read every tick.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Num {
    N(f64),
    P(String),
    /// Another number times a factor (e.g. a slider in millilitres used as cubic metres).
    Scaled(Box<Num>, f64),
}

impl Num {
    fn get(&self, params: &BTreeMap<String, Param>) -> f32 {
        match self {
            Num::N(x) => *x as f32,
            Num::P(name) => params.get(name).map_or(0.0, |p| p.value as f32),
            Num::Scaled(n, k) => (n.get(params) as f64 * k) as f32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    #[default]
    Smoke,
    Water,
}

/// Settings that may follow a control. Defaults are real air / real water.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Air blowing in through the x = 0 side, m/s (None = that side is a wall).
    pub wind: Option<Num>,
    /// 1 = real buoyancy of hot air (g * dT / T).
    pub buoyancy: Num,
    /// How fast dense smoke sinks, m/s² per unit of density.
    pub weight: Num,
    /// Strength of the swirls put back (vorticity confinement); 0 = none.
    pub vorticity: Num,
    /// How fast hot air cools, per second.
    pub cooling: Num,
    /// How fast smoke thins out, per second.
    pub fade: Num,
    pub gravity: Num,
    /// Unused since liquids moved from FLIP to APIC (kept so older saves still open).
    pub flip: Num,
    /// Pressure solver sweeps per step: more = more incompressible, slower.
    pub iterations: Num,
    /// Kinematic viscosity of the liquid, m²/s (water 1e-6, honey ~4e-3).
    #[serde(default = "zero")]
    pub viscosity: Num,
    /// Density of the liquid, kg/m³ (for its mass; a single liquid flows the same at any density).
    #[serde(default = "water_density")]
    pub density: f32,
    /// Name of the liquid, when a known one was chosen.
    #[serde(default)]
    pub liquid: String,
}

fn yes() -> bool {
    true
}

fn zero() -> Num {
    Num::N(0.0)
}

fn water_density() -> f32 {
    998.0
}

/// Real liquids at room temperature: (name, kinematic viscosity m²/s, density kg/m³).
/// Viscosity = dynamic viscosity / density, from standard tables (honey varies a lot: 2-10 Pa·s).
pub const LIQUIDS: [(&str, f64, f32); 8] = [
    ("water", 1.0e-6, 998.0),
    ("sea water", 1.05e-6, 1025.0),
    ("milk", 2.0e-6, 1030.0),
    ("mercury", 1.15e-7, 13534.0),
    ("olive oil", 8.5e-5, 911.0),
    ("glycerine", 1.12e-3, 1261.0),
    ("honey", 4.2e-3, 1420.0),
    ("lava", 4.0e-2, 2600.0),
];

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            wind: None,
            buoyancy: Num::N(1.0),
            weight: Num::N(0.0),
            vorticity: Num::N(0.25),
            cooling: Num::N(0.15),
            fade: Num::N(0.0),
            gravity: Num::N(9.81),
            flip: Num::N(0.95),
            iterations: Num::N(40.0),
            viscosity: Num::N(0.0),
            density: 998.0,
            liquid: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Resolved {
    wind: Option<f32>,
    buoyancy: f32,
    weight: f32,
    vorticity: f32,
    cooling: f32,
    fade: f32,
    gravity: f32,
    iterations: usize,
    viscosity: f32,
}

impl Settings {
    fn resolve(&self, p: &BTreeMap<String, Param>) -> Resolved {
        Resolved {
            wind: self.wind.as_ref().map(|w| w.get(p)),
            buoyancy: self.buoyancy.get(p),
            weight: self.weight.get(p),
            vorticity: self.vorticity.get(p).max(0.0),
            cooling: self.cooling.get(p).max(0.0),
            fade: self.fade.get(p).max(0.0),
            gravity: self.gravity.get(p),
            iterations: self.iterations.get(p).clamp(1.0, 500.0) as usize,
            viscosity: self.viscosity.get(p).max(0.0),
        }
    }
}

/// A source of smoke, heat, wind or water: a ball at `at` (metres) with `radius` (metres).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Emitter {
    pub at: [f32; 3],
    pub radius: f32,
    /// Smoke density added per second.
    pub smoke: Num,
    /// Temperature above the air, kelvin.
    pub heat: Num,
    /// Velocity of what comes out, m/s.
    pub velocity: [Num; 3],
    /// Liquid flow, cubic metres per second (`water_ml` in a script gives millilitres per second).
    pub water: Num,
    #[serde(default)]
    carry: f32,
}

/// A solid object that moves through a liquid: a ball or a box. It pushes the water (its cells
/// are a moving wall), and a floating one is pushed back (buoyancy, drag). Up to 4 per fluid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    pub name: String,
    /// "ball" or "box"
    pub shape: String,
    /// Centre, metres.
    pub pos: [f32; 3],
    /// Velocity, m/s.
    pub vel: [f32; 3],
    /// Ball: radius in [0]; box: half sizes, metres.
    pub half: [f32; 3],
    /// The velocity it is driven at (numbers or controls); for a floating object, a push.
    pub drive: [Num; 3],
    /// Floats (buoyancy, gravity, drag) instead of moving exactly as driven.
    pub float: bool,
    /// kg/m³ (a floating object sinks when denser than the liquid).
    pub density: f32,
    /// Steered with the arrow keys in Studio (sets the drive controls).
    pub keys: bool,
    /// Top speed when steered, m/s.
    pub speed: f32,
}

impl Object {
    fn volume(&self) -> f32 {
        if self.shape == "ball" { 4.0 / 3.0 * std::f32::consts::PI * self.half[0].powi(3) } else { 8.0 * self.half[0] * self.half[1] * self.half[2] }
    }
    /// Is the point (metres) inside?
    pub fn contains(&self, p: [f32; 3]) -> bool {
        let d = [p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2]];
        if self.shape == "ball" { d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= self.half[0] * self.half[0] } else { (0..3).all(|a| d[a].abs() <= self.half[a]) }
    }
    /// Nearest point on the surface (metres) and the outward normal, for a point inside.
    fn surface(&self, p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let d = [p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2]];
        if self.shape == "ball" {
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
            let n = [d[0] / l, d[1] / l, d[2] / l];
            let r = self.half[0] * 1.001;
            ([self.pos[0] + n[0] * r, self.pos[1] + n[1] * r, self.pos[2] + n[2] * r], n)
        } else {
            let a = (0..3).min_by(|&a, &b| (self.half[a] - d[a].abs()).partial_cmp(&(self.half[b] - d[b].abs())).unwrap()).unwrap();
            let mut q = p;
            let mut n = [0.0; 3];
            let s = if d[a] >= 0.0 { 1.0 } else { -1.0 };
            q[a] = self.pos[a] + s * self.half[a] * 1.001;
            n[a] = s;
            (q, n)
        }
    }
}

/// A measurement over the whole fluid or a box of it (metres).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMetric {
    pub name: String,
    /// smoke | temp | speed | water
    pub field: String,
    /// sum | mean | max | count
    pub op: String,
    pub region: Option<[f32; 6]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Fluid {
    pub kind: Kind,
    pub d: Option<Dims>,
    /// Cell size, metres.
    pub cell: f32,
    /// Simulated seconds per tick.
    pub dt: f32,
    pub settings: Settings,
    pub open_top: bool,
    pub open_side: bool,
    pub mac: Option<Mac>,
    /// SOLID or FLUID per cell (water marks AIR itself each step).
    pub solid: Vec<u8>,
    pub pressure: Vec<f32>,
    pub smoke: Vec<f32>,
    /// Kelvin above the air.
    pub temp: Vec<f32>,
    /// Water particles: position (grid units) and velocity (m/s).
    pub particles: Vec<[f32; 6]>,
    /// APIC: how velocity changes around each particle (du/dx..dw/dz, m/s per cell).
    #[serde(default)]
    pub affine: Vec<[f32; 9]>,
    pub emitters: Vec<Emitter>,
    pub metrics: Vec<FieldMetric>,
    /// Objects moving through the liquid.
    #[serde(default)]
    pub objects: Vec<Object>,
    /// Which copy of the particles the graphics card holds (changes whenever the CPU changes them).
    #[serde(skip)]
    gpu_generation: u64,
    /// Solve liquids on the graphics card when there is one (`fluid{backend = "cpu"}` turns it off).
    #[serde(default = "yes")]
    pub use_gpu: bool,
    pub seconds: f64,
    pub divergence: f32,
    /// Length of the last pressure step, seconds (turns the solver's pressure into pascals).
    #[serde(default)]
    pub last_dt: f32,
    rng: u64,
    /// Controls of the step that is being computed in the background (saved, so a reopened
    /// universe redoes that step and stays on the same path).
    #[serde(default)]
    pending: Option<BTreeMap<String, Param>>,
    #[serde(skip)]
    job: Job,
    /// Set by commands that change the fluid while a step is in the background: that step is
    /// then redone from the changed fluid.
    #[serde(skip)]
    changed: bool,
    /// Set by a window that wants `render` made after each step.
    #[serde(skip)]
    pub want_render: bool,
    /// Direction towards the sun, for the caustics.
    #[serde(skip)]
    pub sun: [f32; 3],
    #[serde(skip)]
    pub render: Option<std::sync::Arc<RenderData>>,
}

/// What a window needs to draw a fluid, made once per step on the fluid's own thread (so the
/// window does not redo it every frame). Not saved; not part of any measurement.
#[derive(Clone, Debug, Default)]
pub struct RenderData {
    /// Changes every step (a window uploads to the GPU only when it changed).
    pub generation: u64,
    /// Smoke, heat and solids per cell (`volume_rgba`).
    pub rgba: Vec<u8>,
    /// How full of liquid each cell is (`liquid_field`), empty for smoke.
    pub liquid: Vec<f32>,
    /// Spray droplets (metres) drawn as mist.
    pub spray: Vec<[f32; 3]>,
    /// Light focused on the floor: width, height, brightness / 3.
    pub caustics: Option<(u32, u32, Vec<u8>)>,
    /// Objects: shape, centre and half sizes (metres), to draw exactly.
    pub objects: Vec<(String, [f32; 3], [f32; 3])>,
}

static RENDER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The next step, running on another thread so a window stays smooth while it computes.
#[derive(Default)]
struct Job(Option<std::thread::JoinHandle<Fluid>>);

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

fn splitmix(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

impl Fluid {
    pub fn ready(&self) -> bool {
        self.d.is_some()
    }

    /// True while the next step is still being computed (a window can skip a tick and stay smooth).
    pub fn busy(&self) -> bool {
        self.job.0.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// One tick of the pipeline: take the step computed in the background (or compute it now),
    /// then start the next one. What is visible is always exactly one step behind the newest
    /// controls, whatever the timing, so runs repeat exactly.
    pub fn tick(&mut self, params: &BTreeMap<String, Param>, background: bool) {
        if let Some(h) = self.job.0.take() {
            let done = h.join().expect("fluid step failed");
            if self.changed {
                // the fluid was changed while that step ran: redo it from the changed fluid
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

    pub fn dims(&self) -> Dims {
        self.d.unwrap_or(Dims { nx: 1, ny: 1, nz: 1 })
    }

    /// Size of the box in metres.
    pub fn extent(&self) -> [f32; 3] {
        let d = self.dims();
        [d.nx as f32 * self.cell, d.ny as f32 * self.cell, d.nz as f32 * self.cell]
    }

    fn sides(&self, r: &Resolved) -> Sides {
        Sides { open_top: self.open_top, open_xmax: self.open_side, inflow: r.wind }
    }

    pub fn create(&mut self, kind: Kind, d: Dims, cell: f32, dt: f32, seed: u64) {
        let n = d.cells();
        let old = std::mem::take(self);
        *self = Fluid {
            kind,
            d: Some(d),
            cell,
            dt,
            settings: Settings::default(),
            open_top: kind == Kind::Smoke,
            open_side: false,
            mac: Some(Mac::new(d)),
            solid: vec![FLUID; n],
            pressure: vec![0.0; n],
            smoke: vec![0.0; n],
            temp: vec![0.0; n],
            particles: Vec::new(),
            affine: Vec::new(),
            emitters: Vec::new(),
            metrics: old.metrics,
            objects: Vec::new(),
            use_gpu: old.use_gpu || old.d.is_none(),
            gpu_generation: 0,
            seconds: 0.0,
            divergence: 0.0,
            last_dt: dt,
            rng: seed ^ 0x51_7c_c1_b7_27_22_0a_95,
            pending: None,
            job: Job::default(),
            changed: true,
            want_render: old.want_render,
            sun: old.sun,
            render: None,
        };
    }

    /// Mark the cells whose centres are inside `inside(x, y, z)` (metres) as solid.
    pub fn add_solid(&mut self, inside: impl Fn([f32; 3]) -> bool) -> usize {
        let d = self.dims();
        let h = self.cell;
        let mut n = 0;
        for k in 0..d.nz {
            for j in 0..d.ny {
                for i in 0..d.nx {
                    if inside([(i as f32 + 0.5) * h, (j as f32 + 0.5) * h, (k as f32 + 0.5) * h]) {
                        let c = d.c(i, j, k);
                        self.solid[c] = SOLID;
                        self.smoke[c] = 0.0;
                        self.temp[c] = 0.0;
                        n += 1;
                    }
                }
            }
        }
        let solid = self.solid.clone();
        self.particles.retain(|p| !is_solid(&solid, d, [p[0], p[1], p[2]]));
        self.affine.clear();
        self.gpu_generation = 0;
        n
    }

    /// Fill a box (metres) with water at rest: 8 particles per cell, jittered.
    pub fn add_water(&mut self, lo: [f32; 3], hi: [f32; 3]) -> usize {
        let d = self.dims();
        let h = self.cell;
        let before = self.particles.len();
        for k in 0..d.nz {
            for j in 0..d.ny {
                for i in 0..d.nx {
                    let centre = [(i as f32 + 0.5) * h, (j as f32 + 0.5) * h, (k as f32 + 0.5) * h];
                    if (0..3).any(|a| centre[a] < lo[a] || centre[a] > hi[a]) || self.solid[d.c(i, j, k)] == SOLID {
                        continue;
                    }
                    for s in 0..8 {
                        let jit = |r: &mut u64| 0.25 + 0.5 * splitmix(r);
                        let p = [i as f32 + 0.5 * (s & 1) as f32 + 0.5 * jit(&mut self.rng), j as f32 + 0.5 * ((s >> 1) & 1) as f32 + 0.5 * jit(&mut self.rng), k as f32 + 0.5 * ((s >> 2) & 1) as f32 + 0.5 * jit(&mut self.rng)];
                        self.particles.push([p[0], p[1], p[2], 0.0, 0.0, 0.0]);
                    }
                }
            }
        }
        self.gpu_generation = 0;
        self.particles.len() - before
    }

    /// Advance one tick, reading controls from `params`.
    pub fn step(&mut self, params: &BTreeMap<String, Param>) {
        if !self.ready() {
            return;
        }
        let r = self.settings.resolve(params);
        match self.kind {
            Kind::Smoke => self.step_smoke(&r, params),
            Kind::Water => self.step_water(&r, params),
        }
        self.seconds += self.dt as f64;
    }

    fn emitter_cells(&self, e: &Emitter) -> Vec<usize> {
        let d = self.dims();
        let h = self.cell;
        let r = e.radius.max(h * 0.5);
        let lo = |a: usize| (((e.at[a] - r) / h).floor().max(0.0)) as usize;
        let hi = |a: usize, n: usize| (((e.at[a] + r) / h).ceil() as usize).min(n);
        let mut out = Vec::new();
        for k in lo(2)..hi(2, d.nz) {
            for j in lo(1)..hi(1, d.ny) {
                for i in lo(0)..hi(0, d.nx) {
                    let p = [(i as f32 + 0.5) * h, (j as f32 + 0.5) * h, (k as f32 + 0.5) * h];
                    let dd = (p[0] - e.at[0]).powi(2) + (p[1] - e.at[1]).powi(2) + (p[2] - e.at[2]).powi(2);
                    if dd <= r * r && self.solid[d.c(i, j, k)] != SOLID {
                        out.push(d.c(i, j, k));
                    }
                }
            }
        }
        out
    }

    fn step_smoke(&mut self, r: &Resolved, params: &BTreeMap<String, Param>) {
        let d = self.dims();
        let h = self.cell;
        let speed = self.mac.as_ref().unwrap().max_speed().max(r.wind.unwrap_or(0.0).abs());
        // semi-Lagrangian steps stay stable at any speed; split only when air crosses many cells
        let subs = ((speed * self.dt / h / 5.0).ceil() as usize).clamp(1, 4);
        let sdt = self.dt / subs as f32;
        for _ in 0..subs {
            // sources
            for e in self.emitters.clone() {
                let (smoke, heat) = (e.smoke.get(params), e.heat.get(params));
                let vel = [e.velocity[0].get(params), e.velocity[1].get(params), e.velocity[2].get(params)];
                let cells = self.emitter_cells(&e);
                let mac = self.mac.as_mut().unwrap();
                for &c in &cells {
                    self.smoke[c] += smoke * sdt;
                    self.temp[c] = self.temp[c].max(heat);
                    if vel != [0.0; 3] {
                        let (i, j, k) = (c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny));
                        mac.u[idx(d.ud(), i, j, k)] = vel[0];
                        mac.u[idx(d.ud(), i + 1, j, k)] = vel[0];
                        mac.v[idx(d.vd(), i, j, k)] = vel[1];
                        mac.v[idx(d.vd(), i, j + 1, k)] = vel[1];
                        mac.w[idx(d.wd(), i, j, k)] = vel[2];
                        mac.w[idx(d.wd(), i, j, k + 1)] = vel[2];
                    }
                }
            }
            // carry everything along
            let k_cells = sdt / h;
            let old = self.mac.take().unwrap();
            let mut mac = Mac { d, u: advect(&old.u, d.ud(), OFF_U, &old, k_cells), v: advect(&old.v, d.vd(), OFF_V, &old, k_cells), w: advect(&old.w, d.wd(), OFF_W, &old, k_cells) };
            grid::advect_sharp_many(&mut [&mut self.smoke, &mut self.temp], d, &old, k_cells);
            // hot air rises, dense smoke sinks
            let lift = r.buoyancy * r.gravity / AMBIENT_K;
            let (temp, smoke) = (&self.temp, &self.smoke);
            let wd = d.wd();
            mac.w.par_chunks_mut(wd[0] * wd[1]).enumerate().for_each(|(k, slab)| {
                if k == 0 || k == d.nz {
                    return;
                }
                for j in 0..d.ny {
                    for i in 0..d.nx {
                        let (a, b) = (d.c(i, j, k - 1), d.c(i, j, k));
                        let t = 0.5 * (temp[a] + temp[b]);
                        let s = 0.5 * (smoke[a] + smoke[b]);
                        slab[i + wd[0] * j] += sdt * (lift * t - r.weight * s);
                    }
                }
            });
            if r.vorticity > 0.0 {
                confine(&mut mac, &self.solid, r.vorticity, h, sdt);
            }
            // cooling, thinning, nothing inside solids
            let (cool, fade) = ((-r.cooling * sdt).exp(), (-r.fade * sdt).exp());
            let solid = &self.solid;
            self.temp.par_iter_mut().zip(self.smoke.par_iter_mut()).zip(solid.par_iter()).for_each(|((t, s), f)| {
                if *f == SOLID {
                    *t = 0.0;
                    *s = 0.0;
                } else {
                    *t *= cool;
                    *s = (*s * fade).max(0.0);
                }
            });
            let sides = self.sides(r);
            enforce(&mut mac, &self.solid, sides);
            self.divergence = project(&mut mac, &self.solid, sides, &mut self.pressure, r.iterations, None);
            self.mac = Some(mac);
        }
    }

    /// True when this fluid's liquid is solved on the graphics card.
    pub fn on_gpu(&self) -> bool {
        self.kind == Kind::Water && self.use_gpu && gpu::available()
    }

    /// Solid cells with the objects in them (`SOLID`), and which object is in each cell
    /// (0 = none, 1 + index), for this moment.
    fn live_solids(&self) -> (Vec<u8>, Vec<u8>) {
        let d = self.dims();
        let h = self.cell;
        let mut solid = self.solid.clone();
        let mut which = vec![0u8; d.cells()];
        for (o, ob) in self.objects.iter().enumerate().take(4) {
            let ext = [ob.half[0], if ob.shape == "ball" { ob.half[0] } else { ob.half[1] }, if ob.shape == "ball" { ob.half[0] } else { ob.half[2] }];
            let lo = |a: usize| (((ob.pos[a] - ext[a]) / h).floor().max(0.0)) as usize;
            let hi = |a: usize, n: usize| (((ob.pos[a] + ext[a]) / h).ceil().max(0.0) as usize).min(n);
            for k in lo(2)..hi(2, d.nz) {
                for j in lo(1)..hi(1, d.ny) {
                    for i in lo(0)..hi(0, d.nx) {
                        if ob.contains([(i as f32 + 0.5) * h, (j as f32 + 0.5) * h, (k as f32 + 0.5) * h]) {
                            let c = d.c(i, j, k);
                            solid[c] = SOLID;
                            which[c] = o as u8 + 1;
                        }
                    }
                }
            }
        }
        (solid, which)
    }

    /// Move the objects over `dt`: driven ones at their drive velocity, floating ones by
    /// gravity, buoyancy (Archimedes: the weight of the liquid they push aside) and drag.
    fn move_objects(&mut self, dt: f32, gravity: f32, params: &BTreeMap<String, Param>) {
        if self.objects.is_empty() {
            return;
        }
        let d = self.dims();
        let h = self.cell;
        let ext = self.extent();
        let rho_l = self.settings.density.max(1.0);
        let per_particle = h.powi(3) / liquid::PER_CELL;
        for ob in self.objects.iter_mut() {
            let drive = [ob.drive[0].get(params), ob.drive[1].get(params), ob.drive[2].get(params)];
            if ob.float {
                // the liquid around it: how deep it is there and how it moves (a ring round the object)
                let r = ob.half[0].max(ob.half[1]);
                let (r0, r1) = (r, r * 1.8);
                let (mut n, mut v) = (0.0f32, [0.0f32; 3]);
                for p in &self.particles {
                    let (x, y) = (p[0] * h - ob.pos[0], p[1] * h - ob.pos[1]);
                    let dd = x * x + y * y;
                    if dd >= r0 * r0 && dd <= r1 * r1 {
                        n += 1.0;
                        v[0] += p[3];
                        v[1] += p[4];
                        v[2] += p[5];
                    }
                }
                let area = std::f32::consts::PI * (r1 * r1 - r0 * r0);
                let level = n * per_particle / area;
                let water_v = if n > 0.0 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0; 3] };
                // how much of it is under the water line
                let height = if ob.shape == "ball" { 2.0 * ob.half[0] } else { 2.0 * ob.half[2] };
                let bottom = ob.pos[2] - height / 2.0;
                let x = ((level - bottom) / height).clamp(0.0, 1.0);
                let sub = if ob.shape == "ball" { x * x * (3.0 - 2.0 * x) } else { x };
                let vol = ob.volume();
                let mass = ob.density.max(1.0) * vol;
                let mut acc = [0.0, 0.0, -gravity + rho_l * vol * sub * gravity / mass];
                // drag towards the water's motion where it is wet, a little air drag elsewhere
                let k = 3.0 * sub + 0.05;
                for a in 0..3 {
                    acc[a] += k * (water_v[a] - ob.vel[a]) + drive[a] * 2.0 * sub.max(0.2);
                }
                for a in 0..3 {
                    ob.vel[a] += acc[a] * dt;
                }
            } else {
                ob.vel = drive;
            }
            // stay inside the tank
            let e = if ob.shape == "ball" { [ob.half[0]; 3] } else { ob.half };
            for a in 0..3 {
                let next = ob.pos[a] + ob.vel[a] * dt;
                let (lo, hi) = (e[a] + h * 0.5, ext[a] - e[a] - h * 0.5);
                if next < lo || next > hi {
                    ob.vel[a] = 0.0;
                }
                ob.pos[a] = next.clamp(lo, hi.max(lo));
            }
        }
        let _ = d;
    }

    /// Particles inside an object (it moved onto them) go to its surface and move with it.
    fn push_out_of_objects(&mut self) {
        if self.objects.is_empty() {
            return;
        }
        let h = self.cell;
        let objects = &self.objects;
        self.particles.par_iter_mut().for_each(|p| {
            let x = [p[0] * h, p[1] * h, p[2] * h];
            for ob in objects {
                if ob.contains(x) {
                    let (q, n) = ob.surface(x);
                    p[0] = q[0] / h;
                    p[1] = q[1] / h;
                    p[2] = q[2] / h;
                    // no faster into it than it moves
                    let rel = (p[3] - ob.vel[0]) * n[0] + (p[4] - ob.vel[1]) * n[1] + (p[5] - ob.vel[2]) * n[2];
                    if rel < 0.0 {
                        p[3] -= rel * n[0];
                        p[4] -= rel * n[1];
                        p[5] -= rel * n[2];
                    }
                }
            }
        });
    }

    fn step_water(&mut self, r: &Resolved, params: &BTreeMap<String, Param>) {
        let d = self.dims();
        let h = self.cell;
        if self.affine.len() != self.particles.len() {
            self.affine = vec![[0.0; 9]; self.particles.len()];
        }
        let vmax = self.particles.par_iter().map(|p| p[3].abs().max(p[4].abs()).max(p[5].abs())).reduce(|| 0.0, f32::max);
        let subs = ((vmax * self.dt / h).ceil() as usize).clamp(1, 6);
        let sdt = self.dt / subs as f32;
        // objects move once per tick; the solid cells (with them) stay fixed for its steps
        self.move_objects(self.dt, r.gravity, params);
        let (live, which) = self.live_solids();
        let obj_vel: Vec<[f32; 3]> = self.objects.iter().map(|o| o.vel).collect();
        if self.on_gpu() {
            let before = self.particles.len();
            for _ in 0..subs {
                self.emit_water(sdt, params);
            }
            if self.particles.len() != before || self.gpu_generation == 0 || self.affine.len() != self.particles.len() {
                self.gpu_generation = gpu::fresh_generation();
            }
            self.affine.resize(self.particles.len(), [0.0; 9]);
            let alpha = r.viscosity * sdt / (h * h);
            let visc_steps = if alpha > 0.0 { ((alpha / 0.15).ceil() as usize).clamp(1, 200) } else { 0 };
            let job = gpu::Job {
                d,
                h,
                sdt,
                subs,
                gravity: r.gravity,
                visc: if visc_steps > 0 { alpha / visc_steps as f32 } else { 0.0 },
                visc_steps,
                sweeps: r.iterations * 3,
                solid: &live,
                which: &which,
                objects: self.objects.iter().take(4).map(|o| gpu::GpuObject::new(o, h)).collect(),
                particles: &mut self.particles,
                affine: &mut self.affine,
                pressure: &mut self.pressure,
                generation: self.gpu_generation,
            };
            match gpu::step(job) {
                Ok(divergence) => {
                    // the grid stays on the card; speeds are measured from the particles
                    self.divergence = divergence;
                    self.mac = None;
                    self.last_dt = sdt;
                    return;
                }
                Err(_) => self.use_gpu = false,
            }
        }
        // the CPU is about to change the particles: the card's copy is out of date
        self.gpu_generation = 0;
        let sides = Sides { open_top: false, open_xmax: false, inflow: None };
        self.push_out_of_objects();
        for _ in 0..subs {
            self.emit_water(sdt, params);
            self.affine.resize(self.particles.len(), [0.0; 9]);
            // particles -> grid (APIC)
            let (mut mac, known) = liquid::p2g(&self.particles, &self.affine, d);
            let mut flags = live.clone();
            for f in flags.iter_mut() {
                if *f != SOLID {
                    *f = AIR;
                }
            }
            for p in &self.particles {
                let c = d.c((p[0] as usize).min(d.nx - 1), (p[1] as usize).min(d.ny - 1), (p[2] as usize).min(d.nz - 1));
                if flags[c] != SOLID {
                    flags[c] = FLUID;
                }
            }
            let [mut ku, mut kv, mut kw] = known;
            extrapolate(&mut mac.u, &mut ku, d.ud(), 2);
            extrapolate(&mut mac.v, &mut kv, d.vd(), 2);
            extrapolate(&mut mac.w, &mut kw, d.wd(), 2);
            enforce(&mut mac, &flags, sides);
            grid::apply_objects(&mut mac, &which, &obj_vel);
            for w in mac.w.iter_mut() {
                *w -= r.gravity * sdt;
            }
            enforce(&mut mac, &flags, sides);
            grid::apply_objects(&mut mac, &which, &obj_vel);
            if r.viscosity > 0.0 {
                viscous(&mut mac, &flags, sides, r.viscosity * sdt / (h * h));
            }
            // incompressible, and back towards the rest density where particles drifted
            let rho = liquid::density(&self.particles, d);
            let target = liquid::volume_target(&rho, &flags, &live, d, h, sdt);
            let fill: Vec<f32> = rho.iter().map(|x| x / liquid::PER_CELL).collect();
            self.divergence = grid::project_with(&mut mac, &flags, sides, &mut self.pressure, r.iterations, Some(&target), Some(&fill));
            self.last_dt = sdt;
            // only faces touching water have a solved velocity; spread it a little outwards
            let touches = |a: &[u8], i: usize, j: usize, k: usize, axis: usize| -> bool {
                let f = |i: usize, j: usize, k: usize| a[d.c(i, j, k)] == FLUID;
                match axis {
                    0 => (i > 0 && f(i - 1, j, k)) || (i < d.nx && f(i, j, k)),
                    1 => (j > 0 && f(i, j - 1, k)) || (j < d.ny && f(i, j, k)),
                    _ => (k > 0 && f(i, j, k - 1)) || (k < d.nz && f(i, j, k)),
                }
            };
            for (axis, (arr, dims)) in [(&mut mac.u, d.ud()), (&mut mac.v, d.vd()), (&mut mac.w, d.wd())].into_iter().enumerate() {
                let mut ok = vec![false; arr.len()];
                for k in 0..dims[2] {
                    for j in 0..dims[1] {
                        for i in 0..dims[0] {
                            ok[idx(dims, i, j, k)] = touches(&flags, i, j, k, axis);
                        }
                    }
                }
                extrapolate(arr, &mut ok, dims, 3);
            }
            enforce(&mut mac, &flags, sides);
            grid::apply_objects(&mut mac, &which, &obj_vel);
            // grid -> particles (APIC), then move them through the new flow
            liquid::g2p(&mac, &mut self.particles, &mut self.affine);
            let k_cells = sdt / h;
            let solid = &live;
            self.particles.par_iter_mut().for_each(|p| {
                let x = [p[0], p[1], p[2]];
                let v0 = mac.vel(x);
                let mid = [x[0] + 0.5 * k_cells * v0[0], x[1] + 0.5 * k_cells * v0[1], x[2] + 0.5 * k_cells * v0[2]];
                let vm = mac.vel(mid);
                let mut pos = x;
                let lim = [d.nx as f32, d.ny as f32, d.nz as f32];
                for a in 0..3 {
                    let mut q = pos;
                    q[a] = (pos[a] + k_cells * vm[a]).clamp(0.01, lim[a] - 0.01);
                    if is_solid(solid, d, q) {
                        p[3 + a] = 0.0;
                    } else {
                        if q[a] <= 0.01 || q[a] >= lim[a] - 0.01 {
                            p[3 + a] = 0.0;
                        }
                        pos = q;
                    }
                }
                p[0] = pos[0];
                p[1] = pos[1];
                p[2] = pos[2];
            });
            self.push_out_of_objects();
            liquid::separate(&mut self.particles, &live, d, 1);
            self.mac = Some(mac);
        }
    }
    fn emit_water(&mut self, sdt: f32, params: &BTreeMap<String, Param>) {
        let per_particle = self.cell.powi(3) / 8.0;
        for ei in 0..self.emitters.len() {
            let e = self.emitters[ei].clone();
            let flow = e.water.get(params).max(0.0);
            if flow <= 0.0 {
                continue;
            }
            let vel = [e.velocity[0].get(params), e.velocity[1].get(params), e.velocity[2].get(params)];
            let want = flow * sdt / per_particle + e.carry;
            let n = want.floor() as usize;
            self.emitters[ei].carry = want - n as f32;
            let d = self.dims();
            for _ in 0..n.min(20_000) {
                // a random point in the ball
                let mut q = [0.0f32; 3];
                loop {
                    for v in q.iter_mut() {
                        *v = 2.0 * splitmix(&mut self.rng) - 1.0;
                    }
                    if q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= 1.0 {
                        break;
                    }
                }
                let g = [(e.at[0] + q[0] * e.radius) / self.cell, (e.at[1] + q[1] * e.radius) / self.cell, (e.at[2] + q[2] * e.radius) / self.cell];
                if (0..3).all(|a| g[a] > 0.0 && g[a] < d.cd()[a] as f32) && !is_solid(&self.solid, d, g) {
                    self.particles.push([g[0], g[1], g[2], vel[0], vel[1], vel[2]]);
                }
            }
        }
    }

    /// Value of one field in one cell (for metrics and pictures).
    fn cell_value(&self, field: &str, c: usize, per_cell: &[f32]) -> f32 {
        let d = self.dims();
        match field {
            "smoke" => self.smoke[c],
            "temp" => self.temp[c],
            "speed" if self.kind == Kind::Water => per_cell.get(c).copied().unwrap_or(0.0),
            "speed" => {
                let Some(mac) = self.mac.as_ref() else { return 0.0 };
                let (i, j, k) = (c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny));
                let v = mac.centre(i, j, k);
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            }
            "water" => per_cell.get(c).copied().unwrap_or(0.0),
            "pressure" => self.pressure_pa(c),
            _ => 0.0,
        }
    }

    fn water_counts(&self) -> Vec<u32> {
        let d = self.dims();
        let mut n = vec![0u32; d.cells()];
        for p in &self.particles {
            n[d.c((p[0] as usize).min(d.nx - 1), (p[1] as usize).min(d.ny - 1), (p[2] as usize).min(d.nz - 1))] += 1;
        }
        n
    }

    pub fn evaluate(&self, m: &FieldMetric) -> f64 {
        if !self.ready() {
            return 0.0;
        }
        let d = self.dims();
        let h = self.cell;
        // liquids: per-cell values from the particles (the grid may be on the graphics card)
        let counts: Vec<f32> = match (m.field.as_str(), self.kind) {
            ("water", _) => self.water_counts().into_iter().map(|n| n as f32).collect(),
            ("speed", Kind::Water) => {
                let mut fastest = vec![0.0f32; d.cells()];
                for p in &self.particles {
                    let c = d.c((p[0] as usize).min(d.nx - 1), (p[1] as usize).min(d.ny - 1), (p[2] as usize).min(d.nz - 1));
                    fastest[c] = fastest[c].max((p[3] * p[3] + p[4] * p[4] + p[5] * p[5]).sqrt());
                }
                fastest
            }
            _ => Vec::new(),
        };
        let (mut sum, mut max, mut n) = (0.0f64, f64::MIN, 0usize);
        for k in 0..d.nz {
            for j in 0..d.ny {
                for i in 0..d.nx {
                    let c = d.c(i, j, k);
                    if self.solid[c] == SOLID {
                        continue;
                    }
                    if let Some(b) = m.region {
                        let p = [(i as f32 + 0.5) * h, (j as f32 + 0.5) * h, (k as f32 + 0.5) * h];
                        if (0..3).any(|a| p[a] < b[a].min(b[a + 3]) || p[a] > b[a].max(b[a + 3])) {
                            continue;
                        }
                    }
                    let v = self.cell_value(&m.field, c, &counts) as f64;
                    sum += v;
                    max = max.max(v);
                    n += (v > 1e-3) as usize;
                }
            }
        }
        // sums are physical amounts: smoke in m³ of full density, water in m³
        let vol = (h as f64).powi(3);
        match (m.op.as_str(), m.field.as_str()) {
            ("count", _) => n as f64,
            ("max", _) if max > f64::MIN => max,
            ("sum", "water") => sum * vol / 8.0,
            ("sum", _) => sum * vol,
            ("mean", _) => {
                let cells = d.cells().max(1) as f64;
                sum / cells
            }
            _ => 0.0,
        }
    }

    /// Pressure in a cell above the air's, pascals (from the last pressure solve).
    pub fn pressure_pa(&self, c: usize) -> f32 {
        let rho = if self.kind == Kind::Water { self.settings.density } else { 1.2 };
        self.pressure.get(c).copied().unwrap_or(0.0) * rho * self.cell / self.last_dt.max(1e-9)
    }

    /// Built-in measurements.
    pub fn summary(&self) -> Vec<(String, f64)> {
        if !self.ready() {
            return Vec::new();
        }
        let h = self.cell as f64;
        let mut out = vec![("fluid.seconds".to_string(), self.seconds), ("fluid.divergence".to_string(), self.divergence as f64)];
        match self.kind {
            Kind::Smoke => {
                let mac = self.mac.as_ref().unwrap();
                out.push(("fluid.fastest_m_s".into(), mac.max_speed() as f64));
                out.push(("smoke.amount_m3".into(), self.smoke.iter().map(|x| *x as f64).sum::<f64>() * h.powi(3)));
                out.push(("smoke.hottest_c".into(), 20.0 + self.temp.iter().cloned().fold(0.0f32, f32::max) as f64));
                // where the smoke is: its centre, metres
                let d = self.dims();
                let total: f64 = self.smoke.iter().map(|x| *x as f64).sum();
                if total > 1e-9 {
                    let (mut cx, mut cz) = (0.0, 0.0);
                    for (c, s) in self.smoke.iter().enumerate() {
                        cx += *s as f64 * ((c % d.nx) as f64 + 0.5) * h;
                        cz += *s as f64 * ((c / (d.nx * d.ny)) as f64 + 0.5) * h;
                    }
                    out.push(("smoke.centre_x_m".into(), cx / total));
                    out.push(("smoke.centre_height_m".into(), cz / total));
                }
            }
            Kind::Water => {
                let n = self.particles.len() as f64;
                out.push(("water.particles".into(), n));
                out.push(("water.volume_m3".into(), n * h.powi(3) / 8.0));
                out.push(("water.mass_kg".into(), n * h.powi(3) / 8.0 * self.settings.density as f64));
                // the space the liquid really fills: cells count as full at the rest density
                let rho = liquid::density(&self.particles, self.dims());
                out.push(("water.filled_m3".into(), rho.iter().zip(&self.solid).filter(|(_, s)| **s != SOLID).map(|(r, _)| (*r / liquid::PER_CELL).min(1.0) as f64).sum::<f64>() * h.powi(3)));
                if n > 0.0 {
                    out.push(("water.mean_height_m".into(), self.particles.iter().map(|p| p[2] as f64).sum::<f64>() / n * h));
                    for ob in &self.objects {
                        out.push((format!("{}.x_m", ob.name), ob.pos[0] as f64));
                        out.push((format!("{}.height_m", ob.name), ob.pos[2] as f64));
                        out.push((format!("{}.speed_m_s", ob.name), (ob.vel[0] * ob.vel[0] + ob.vel[1] * ob.vel[1] + ob.vel[2] * ob.vel[2]).sqrt() as f64));
                    }
                    out.push(("water.centre_x_m".into(), self.particles.iter().map(|p| p[0] as f64).sum::<f64>() / n * h));
                    out.push(("water.front_m".into(), self.particles.iter().map(|p| p[0]).fold(0.0f32, f32::max) as f64 * h));
                    out.push(("fluid.fastest_m_s".into(), self.particles.iter().map(|p| (p[3] * p[3] + p[4] * p[4] + p[5] * p[5]).sqrt()).fold(0.0f32, f32::max) as f64));
                }
            }
        }
        out
    }

    /// Picture data for the 3D view, one RGBA byte quad per cell (x fastest, then y, then z):
    /// r = smoke, g = heat (0..600 K above the air), b = solid, a = unused.
    pub fn volume_rgba(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.solid.len() * 4];
        out.par_chunks_mut(4).enumerate().for_each(|(c, px)| {
            px[0] = (self.smoke.get(c).copied().unwrap_or(0.0).clamp(0.0, 1.0) * 255.0) as u8;
            px[1] = ((self.temp.get(c).copied().unwrap_or(0.0) / 600.0).clamp(0.0, 1.0) * 255.0) as u8;
            px[2] = if self.solid[c] == SOLID { 255 } else { 0 };
        });
        out
    }

    /// Caustics on the floor: sunlight refracted by the water surface and focused into bright
    /// lines, as in Evan Wallace's WebGL Water. The surface height of each column of water is
    /// taken from the particles; light through each piece of surface is bent by Snell's law
    /// (water n = 1.33) and gathered where it hits the floor. 1 = as bright as without water.
    /// Returns (width, height, brightness / 3 as bytes), scale texels per cell.
    pub fn caustics(&self, light: [f32; 3], scale: usize) -> (u32, u32, Vec<u8>) {
        let d = self.dims();
        let scale = scale.max(1);
        let (w, hgt) = (d.nx * scale, d.ny * scale);
        let none = (w as u32, hgt as u32, vec![85u8; w * hgt]);
        if self.kind != Kind::Water || self.particles.is_empty() || light[2] <= 0.05 {
            return none;
        }
        let rho = liquid::density(&self.particles, d);
        // water height of each column, in cells (0 = dry)
        let heights: Vec<f32> = (0..d.nx * d.ny)
            .into_par_iter()
            .map(|c| {
                let (i, j) = (c % d.nx, c / d.nx);
                let mut top = 0.0f32;
                for k in 0..d.nz {
                    let f = (rho[d.c(i, j, k)] / liquid::PER_CELL).min(1.0);
                    if f > 0.5 {
                        top = k as f32 + f.min(1.0);
                    }
                }
                top
            })
            .collect();
        // smooth the column heights a little: particles sit a little unevenly, real water does not
        let mut heights = heights;
        for _ in 0..3 {
            let old = heights.clone();
            heights = (0..d.nx * d.ny)
                .into_par_iter()
                .map(|c| {
                    let (i, j) = ((c % d.nx) as isize, (c / d.nx) as isize);
                    let (mut s, mut n) = (0.0, 0.0);
                    for (di, dj, w) in [(0, 0, 4.0), (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)] {
                        let (a, b) = (i + di, j + dj);
                        if a >= 0 && b >= 0 && a < d.nx as isize && b < d.ny as isize {
                            let v = old[b as usize * d.nx + a as usize];
                            // a dry neighbour does not pull the water's edge down
                            if v > 0.0 || old[c] == 0.0 {
                                s += w * v;
                                n += w;
                            }
                        }
                    }
                    if n > 0.0 { s / n } else { 0.0 }
                })
                .collect();
        }
        let at = |x: f32, y: f32| crate::grid::sample(&heights, [d.nx, d.ny, 1], [x - 0.5, y - 0.5, 0.0]);
        let eta = 1.0 / 1.33f32;
        let l = { let n = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt(); [-light[0] / n, -light[1] / n, -light[2] / n] };
        let sub = 2 * scale;
        let (rw, rh) = (d.nx * sub, d.ny * sub);
        // every ray in parallel: where it started and where it lands on the floor
        let rays: Vec<(usize, Option<(f32, f32)>)> = (0..rw * rh)
            .into_par_iter()
            .map(|r| {
                let (si, sj) = (r % rw, r / rw);
                let (x, y) = ((si as f32 + 0.5) / sub as f32, (sj as f32 + 0.5) / sub as f32);
                let start = ((y * scale as f32) as usize).min(hgt - 1) * w + ((x * scale as f32) as usize).min(w - 1);
                let depth = at(x, y);
                if depth < 0.3 {
                    return (start, Some((x * scale as f32 - 0.5, y * scale as f32 - 0.5)));
                }
                let e = 0.5;
                let n = [-(at(x + e, y) - at(x - e, y)) / (2.0 * e), -(at(x, y + e) - at(x, y - e)) / (2.0 * e), 1.0];
                let nl = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
                let n = [n[0] / nl, n[1] / nl, n[2] / nl];
                // Snell: refract the downward light through the surface
                let c = -(n[0] * l[0] + n[1] * l[1] + n[2] * l[2]);
                let k = 1.0 - eta * eta * (1.0 - c * c);
                if k < 0.0 {
                    return (start, None);
                }
                let t = [eta * l[0] + (eta * c - k.sqrt()) * n[0], eta * l[1] + (eta * c - k.sqrt()) * n[1], eta * l[2] + (eta * c - k.sqrt()) * n[2]];
                // the straight (flat-water) ray lands here; caustics are where bent rays bunch up
                let flat_t = [eta * l[0], eta * l[1], -(1.0 - eta * eta * (1.0 - l[2] * l[2])).sqrt()];
                let hx = x + depth * (t[0] / -t[2] - flat_t[0] / -flat_t[2]);
                let hy = y + depth * (t[1] / -t[2] - flat_t[1] / -flat_t[2]);
                (start, Some((hx * scale as f32 - 0.5, hy * scale as f32 - 0.5)))
            })
            .collect();
        let mut acc = vec![0.0f32; w * hgt];
        let mut base = vec![0.0f32; w * hgt];
        for (start, land) in rays {
            base[start] += 1.0;
            let Some((fx, fy)) = land else { continue };
            let (ix, iy) = (fx.floor(), fy.floor());
            let (ax, ay) = (fx - ix, fy - iy);
            for (dx, dy, wgt) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
                let (px, py) = (ix as i64 + dx, iy as i64 + dy);
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < hgt {
                    acc[py as usize * w + px as usize] += wgt;
                }
            }
        }        let bytes = acc.iter().zip(&base).map(|(a, b)| ((a / b.max(1e-6)) / 3.0 * 255.0).clamp(0.0, 255.0) as u8).collect();
        (w as u32, hgt as u32, bytes)
    }

    /// The liquid's surface as a closed triangle mesh in metres (`detail` nodes per cell, 2 is good).
    pub fn surface_mesh(&self, detail: usize) -> mesh::Mesh {
        if self.kind != Kind::Water || !self.ready() {
            return mesh::Mesh::default();
        }
        mesh::surface(&self.particles, self.dims(), self.cell, detail)
    }

    /// Make `render` for this moment, if a window asked for it.
    pub fn make_render(&mut self) {
        if !self.want_render || !self.ready() {
            return;
        }
        let water = self.kind == Kind::Water;
        let spray = if water { self.water_points().into_iter().filter(|(_, foam)| *foam > 0.5).map(|(p, _)| p).collect() } else { Vec::new() };
        self.render = Some(std::sync::Arc::new(RenderData {
            generation: RENDER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            rgba: self.volume_rgba(),
            liquid: if water {
                // no water inside an object: the field there is spread from the water around it
                let mut field = self.liquid_field();
                if !self.objects.is_empty() {
                    let (_, which) = self.live_solids();
                    for (f, w) in field.iter_mut().zip(which) {
                        if w > 0 {
                            *f = 0.0;
                        }
                    }
                }
                field
            } else {
                Vec::new()
            },
            spray,
            caustics: water.then(|| self.caustics(self.sun, 3)),
            objects: self.objects.iter().map(|o| (o.shape.clone(), o.pos, o.half)).collect(),
        }));
    }

    /// The smooth liquid field for drawing (`liquid::surface_field`), one value per cell.
    pub fn liquid_field(&self) -> Vec<f32> {
        if self.kind != Kind::Water || !self.ready() {
            return Vec::new();
        }
        liquid::surface_field(&self.particles, self.dims())
    }

    /// Water particles in metres, with how much they are spray (0 = in the body of the water,
    /// 1 = a fast droplet flying on its own: where real water turns white).
    pub fn water_points(&self) -> Vec<([f32; 3], f32)> {
        let h = self.cell;
        let d = self.dims();
        let rho = liquid::density(&self.particles, d);
        self.particles
            .par_iter()
            .map(|p| {
                let x = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
                let around = grid::sample(&rho, d.cd(), x) / liquid::PER_CELL;
                let speed = (p[3] * p[3] + p[4] * p[4] + p[5] * p[5]).sqrt();
                // spray = few neighbours and moving fast; sheets and jets stay liquid
                // only droplets with almost nothing around them; sheets and jets are drawn as liquid
                let alone = ((0.35 - around) / 0.15).clamp(0.0, 1.0);
                let fast = (speed / 0.5).clamp(0.0, 1.0);
                ([p[0] * h, p[1] * h, p[2] * h], alone * fast)
            })
            .collect()
    }
}

pub(crate) fn is_solid(solid: &[u8], d: Dims, p: [f32; 3]) -> bool {
    let (i, j, k) = (p[0].max(0.0) as usize, p[1].max(0.0) as usize, p[2].max(0.0) as usize);
    i >= d.nx || j >= d.ny || k >= d.nz || solid[d.c(i, j, k)] == SOLID
}

/// Viscosity: liquid drags on the liquid next to it (and sticks to walls, whose faces are 0).
/// Explicit diffusion of the face velocities that touch liquid, split into small enough steps
/// to stay stable; lpha = viscosity * dt / cell².
fn viscous(mac: &mut Mac, flags: &[u8], sides: Sides, alpha: f32) {
    let d = mac.d;
    let steps = ((alpha / 0.15).ceil() as usize).clamp(1, 200);
    let a = alpha / steps as f32;
    let liquid = |i: usize, j: usize, k: usize| flags[d.c(i, j, k)] == FLUID;
    for _ in 0..steps {
        for (axis, dims) in [(0usize, d.ud()), (1, d.vd()), (2, d.wd())] {
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
                        let (lo, hi) = match axis {
                            0 => (i > 0 && liquid(i - 1, j, k), i < d.nx && liquid(i, j, k)),
                            1 => (j > 0 && liquid(i, j - 1, k), j < d.ny && liquid(i, j, k)),
                            _ => (k > 0 && liquid(i, j, k - 1), k < d.nz && liquid(i, j, k)),
                        };
                        if !lo && !hi {
                            continue;
                        }
                        let c = idx(dims, i, j, k);
                        let me = src[c];
                        // past the edge of the box: along this velocity it slides (same value), across
                        // it the wall holds the liquid still (no-slip), as real walls do
                        let nb = |ok: bool, o: usize, dir: usize| if ok { src[o] } else if dir == axis { me } else { 0.0 };
                        let lap = nb(i > 0, c.wrapping_sub(1), 0) + nb(i + 1 < dims[0], c + 1, 0) + nb(j > 0, c.wrapping_sub(dims[0]), 1) + nb(j + 1 < dims[1], c + dims[0], 1) + nb(k > 0, c.wrapping_sub(dims[0] * dims[1]), 2) + nb(k + 1 < dims[2], c + dims[0] * dims[1], 2) - 6.0 * me;
                        slab[i + dims[0] * j] = me + a * lap;
                    }
                }
            });
        }
        enforce(mac, flags, sides);
    }
}

/// Vorticity confinement: find the swirls and push to keep them spinning.
fn confine(mac: &mut Mac, solid: &[u8], eps: f32, h: f32, dt: f32) {
    let d = mac.d;
    let n = d.cells();
    let vc: Vec<[f32; 3]> = (0..n).into_par_iter().map(|c| mac.centre(c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny))).collect();
    let nb = |c: usize, a: usize, plus: bool| -> usize {
        let (i, j, k) = (c % d.nx, (c / d.nx) % d.ny, c / (d.nx * d.ny));
        let (mut i, mut j, mut k) = (i, j, k);
        match (a, plus) {
            (0, true) => i = (i + 1).min(d.nx - 1),
            (0, false) => i = i.saturating_sub(1),
            (1, true) => j = (j + 1).min(d.ny - 1),
            (1, false) => j = j.saturating_sub(1),
            (_, true) => k = (k + 1).min(d.nz - 1),
            (_, false) => k = k.saturating_sub(1),
        }
        d.c(i, j, k)
    };
    let dv = |c: usize, comp: usize, a: usize| (vc[nb(c, a, true)][comp] - vc[nb(c, a, false)][comp]) / (2.0 * h);
    let omega: Vec<[f32; 3]> = (0..n).into_par_iter().map(|c| [dv(c, 2, 1) - dv(c, 1, 2), dv(c, 0, 2) - dv(c, 2, 0), dv(c, 1, 0) - dv(c, 0, 1)]).collect();
    let mag: Vec<f32> = omega.par_iter().map(|w| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()).collect();
    let force: Vec<[f32; 3]> = (0..n)
        .into_par_iter()
        .map(|c| {
            if solid[c] == SOLID {
                return [0.0; 3];
            }
            let g = [(mag[nb(c, 0, true)] - mag[nb(c, 0, false)]), (mag[nb(c, 1, true)] - mag[nb(c, 1, false)]), (mag[nb(c, 2, true)] - mag[nb(c, 2, false)])];
            let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if l < 1e-9 {
                return [0.0; 3];
            }
            let nn = [g[0] / l, g[1] / l, g[2] / l];
            let w = omega[c];
            [eps * h * (nn[1] * w[2] - nn[2] * w[1]), eps * h * (nn[2] * w[0] - nn[0] * w[2]), eps * h * (nn[0] * w[1] - nn[1] * w[0])]
        })
        .collect();
    for k in 0..d.nz {
        for j in 0..d.ny {
            for i in 1..d.nx {
                mac.u[idx(d.ud(), i, j, k)] += dt * 0.5 * (force[d.c(i - 1, j, k)][0] + force[d.c(i, j, k)][0]);
            }
        }
    }
    for k in 0..d.nz {
        for j in 1..d.ny {
            for i in 0..d.nx {
                mac.v[idx(d.vd(), i, j, k)] += dt * 0.5 * (force[d.c(i, j - 1, k)][1] + force[d.c(i, j, k)][1]);
            }
        }
    }
    for k in 1..d.nz {
        for j in 0..d.ny {
            for i in 0..d.nx {
                mac.w[idx(d.wd(), i, j, k)] += dt * 0.5 * (force[d.c(i, j, k - 1)][2] + force[d.c(i, j, k)][2]);
            }
        }
    }
}

struct FluidSystem;

impl System for FluidSystem {
    fn name(&self) -> &str {
        "fields.fluid"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(mut f) = ctx.world.take::<Fluid>() else { return };
        if f.ready() {
            let empty = BTreeMap::new();
            let params = ctx.world.get::<Things>().map_or(&empty, |t| &t.params);
            f.tick(params, true);
        }
        ctx.world.put(f);
    }
}

pub struct FluidModule;

impl Module for FluidModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        b.world.insert_persistent("fluid", Fluid { rng: b.config.seed, ..Fluid::default() });
        register_commands(b.world);
        b.add_system(Stage::Update, FluidSystem);
        b.add_metric_source(|w| {
            let Some(f) = w.get::<Fluid>() else { return Vec::new() };
            let mut out = f.summary();
            out.extend(f.metrics.iter().map(|m| (m.name.clone(), f.evaluate(m))));
            out
        });
    }
}

fn fluid_mut(w: &mut World) -> Result<&mut Fluid, String> {
    let f = w.get_mut::<Fluid>().ok_or_else(|| "this universe has no fluids".to_string())?;
    if !f.ready() {
        return Err("make the fluid first: fluid{size = {64, 32, 32}, cell = 0.1}".into());
    }
    f.changed = true;
    Ok(f)
}

fn num(a: &Value, key: &str) -> Result<Option<Num>, String> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(Some(Num::N(n.as_f64().unwrap_or(0.0)))),
        Some(Value::String(s)) => Ok(Some(Num::P(s.clone()))),
        Some(_) => Err(format!("'{key}' must be a number or the name of a control (param)")),
    }
}

fn floats(a: &Value, key: &str, n: usize) -> Result<Option<Vec<f32>>, String> {
    match a.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(v)) if v.len() == n && v.iter().all(Value::is_number) => Ok(Some(v.iter().map(|x| x.as_f64().unwrap() as f32).collect())),
        Some(_) => Err(format!("'{key}' must be {n} numbers, e.g. {key} = {{{}}}", vec!["1"; n].join(", "))),
    }
}

fn register_commands(w: &mut World) {
    // fluid{kind = "smoke" | "water", size = {nx, ny, nz}, cell = metres, dt = seconds per tick, ...settings}
    w.register_command("fluid", |w, a| {
        let seed = w.get::<Fluid>().map_or(1, |f| f.rng);
        let f = w.get_mut::<Fluid>().ok_or("this universe has no fluids")?;
        f.changed = true;
        if let Some(size) = floats(a, "size", 3)? {
            let kind = match a.get("kind").and_then(Value::as_str).unwrap_or("smoke") {
                "smoke" | "air" => Kind::Smoke,
                "water" => Kind::Water,
                other => return Err(format!("kind '{other}': use \"smoke\" or \"water\"")),
            };
            let n = |x: f32| (x.round() as usize).clamp(4, 512);
            let d = Dims { nx: n(size[0]), ny: n(size[1]), nz: n(size[2]) };
            if d.cells() > 8_000_000 {
                return Err(format!("{} cells is too many (at most 8 million)", d.cells()));
            }
            let cell = a.get("cell").and_then(Value::as_f64).unwrap_or(0.1) as f32;
            let dt = a.get("dt").and_then(Value::as_f64).unwrap_or(1.0 / 30.0) as f32;
            if cell <= 0.0 || dt <= 0.0 {
                return Err("'cell' and 'dt' must be above zero".into());
            }
            f.create(kind, d, cell, dt, seed);
        } else if !f.ready() {
            return Err("the first fluid{...} needs a size, e.g. fluid{size = {64, 32, 32}, cell = 0.1}".into());
        }
        let s = &mut f.settings;
        if a.get("wind").is_some() {
            // a wind tunnel: in at x = 0, out at the far side, closed above (unless asked)
            s.wind = num(a, "wind")?;
            f.open_side = true;
            f.open_top = false;
        }
        for (key, slot) in [("buoyancy", &mut s.buoyancy), ("weight", &mut s.weight), ("vorticity", &mut s.vorticity), ("cooling", &mut s.cooling), ("fade", &mut s.fade), ("gravity", &mut s.gravity), ("flip", &mut s.flip), ("iterations", &mut s.iterations)] {
            if let Some(v) = num(a, key)? {
                *slot = v;
            }
        }
        if let Some(name) = a.get("liquid").and_then(Value::as_str) {
            let (n, nu, rho) = LIQUIDS.iter().find(|l| l.0 == name.to_lowercase()).ok_or_else(|| format!("liquid '{name}': known liquids are {}", LIQUIDS.iter().map(|l| l.0).collect::<Vec<_>>().join(", ")))?;
            s.viscosity = Num::N(*nu);
            s.density = *rho;
            s.liquid = n.to_string();
        }
        if let Some(v) = num(a, "viscosity")? {
            s.viscosity = v;
        }
        if let Some(rho) = a.get("density").and_then(Value::as_f64) {
            s.density = rho as f32;
        }
        if let Some(b) = a.get("backend").and_then(Value::as_str) {
            f.use_gpu = match b {
                "gpu" => true,
                "cpu" => false,
                other => return Err(format!("backend '{other}': use \"gpu\" or \"cpu\"")),
            };
        }
        if let Some(b) = a.get("open_top").and_then(Value::as_bool) {
            f.open_top = b;
        }
        if let Some(b) = a.get("open_side").and_then(Value::as_bool) {
            f.open_side = b;
        }
        let d = f.dims();
        Ok(json!({"cells": d.cells(), "size_m": f.extent(), "kind": f.kind}))
    });
    // solid{box = {x0, y0, z0, x1, y1, z1}} or solid{ball = {x, y, z, r}}, metres
    w.register_command("solid", |w, a| {
        let f = fluid_mut(w)?;
        let n = if let Some(b) = floats(a, "box", 6)? {
            f.add_solid(move |p| (0..3).all(|k| p[k] >= b[k].min(b[k + 3]) && p[k] <= b[k].max(b[k + 3])))
        } else if let Some(b) = floats(a, "ball", 4)? {
            f.add_solid(move |p| (p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2) <= b[3] * b[3])
        } else if let Some(b) = floats(a, "cup", 6)? {
            // an open cup or glass: x, y, outer radius, wall thickness, bottom, top
            let (x, y, ro, wall, lo, hi) = (b[0], b[1], b[2], b[3], b[4], b[5]);
            f.add_solid(move |p| {
                let rr = ((p[0] - x).powi(2) + (p[1] - y).powi(2)).sqrt();
                p[2] >= lo && p[2] <= hi && rr <= ro && (rr >= ro - wall || p[2] <= lo + wall)
            })
        } else if let Some(b) = floats(a, "cylinder", 5)? {
            // upright: x, y, radius, bottom, top
            f.add_solid(move |p| (p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) <= b[2] * b[2] && p[2] >= b[3] && p[2] <= b[4])
        } else {
            return Err("solid needs box = {x0, y0, z0, x1, y1, z1}, ball = {x, y, z, r}, cylinder = {x, y, r, bottom, top} or cup = {x, y, r, wall, bottom, top} (metres)".into());
        };
        Ok(json!(n))
    });
    // emitter{at = {x, y, z}, radius = 0.2, smoke = 1, heat = 300, velocity = {0, 0, 1}, water = 0.01}
    w.register_command("emitter", |w, a| {
        let f = fluid_mut(w)?;
        let at = floats(a, "at", 3)?.ok_or("emitter needs at = {x, y, z} (metres)")?;
        let vel = match a.get("velocity") {
            None | Some(Value::Null) => [Num::N(0.0), Num::N(0.0), Num::N(0.0)],
            Some(Value::Array(v)) if v.len() == 3 => {
                let one = |x: &Value| match x {
                    Value::Number(n) => Ok(Num::N(n.as_f64().unwrap_or(0.0))),
                    Value::String(s) => Ok(Num::P(s.clone())),
                    _ => Err("velocity parts must be numbers or control names".to_string()),
                };
                [one(&v[0])?, one(&v[1])?, one(&v[2])?]
            }
            Some(_) => return Err("velocity must be {vx, vy, vz} in m/s".into()),
        };
        let e = Emitter {
            at: [at[0], at[1], at[2]],
            radius: a.get("radius").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            smoke: num(a, "smoke")?.unwrap_or(Num::N(0.0)),
            heat: num(a, "heat")?.unwrap_or(Num::N(0.0)),
            velocity: vel,
            water: match (num(a, "water")?, num(a, "water_ml")?) {
                (Some(w), _) => w,
                (None, Some(ml)) => Num::Scaled(Box::new(ml), 1e-6),
                (None, None) => Num::N(0.0),
            },
            carry: 0.0,
        };
        f.emitters.push(e);
        Ok(json!(f.emitters.len()))
    });
    // water{box = {x0, y0, z0, x1, y1, z1}}: fill with still water
    w.register_command("water", |w, a| {
        let f = fluid_mut(w)?;
        if f.kind != Kind::Water {
            return Err("water{...} needs fluid{kind = \"water\", ...}".into());
        }
        let b = floats(a, "box", 6)?.ok_or("water needs box = {x0, y0, z0, x1, y1, z1} (metres)")?;
        Ok(json!(f.add_water([b[0].min(b[3]), b[1].min(b[4]), b[2].min(b[5])], [b[0].max(b[3]), b[1].max(b[4]), b[2].max(b[5])])))
    });
    // object{name, shape = "ball" | "box", at = {x, y, z}, radius = 0.1 | size = {sx, sy, sz},
    //        velocity = {vx, vy, vz} (numbers or controls), float = false, density = 500, keys = true, speed = 1}
    w.register_command("object", |w, a| {
        let f = fluid_mut(w)?;
        if f.kind != Kind::Water {
            return Err("objects move through liquids: use fluid{kind = \"water\", ...}".into());
        }
        if f.objects.len() >= 4 {
            return Err("at most 4 objects per fluid".into());
        }
        let shape = a.get("shape").and_then(Value::as_str).unwrap_or("ball").to_string();
        let half = match shape.as_str() {
            "ball" => {
                let r = a.get("radius").and_then(Value::as_f64).ok_or("a ball needs radius (metres)")? as f32;
                [r, r, r]
            }
            "box" => {
                let s = floats(a, "size", 3)?.ok_or("a box needs size = {sx, sy, sz} (metres)")?;
                [s[0] / 2.0, s[1] / 2.0, s[2] / 2.0]
            }
            other => return Err(format!("shape '{other}': use \"ball\" or \"box\"")),
        };
        let at = floats(a, "at", 3)?.ok_or("object needs at = {x, y, z} (metres)")?;
        let drive = match a.get("velocity") {
            None | Some(Value::Null) => [Num::N(0.0), Num::N(0.0), Num::N(0.0)],
            Some(Value::Array(v)) if v.len() == 3 => {
                let one = |x: &Value| match x {
                    Value::Number(n) => Ok(Num::N(n.as_f64().unwrap_or(0.0))),
                    Value::String(s) => Ok(Num::P(s.clone())),
                    _ => Err("velocity parts must be numbers or control names".to_string()),
                };
                [one(&v[0])?, one(&v[1])?, one(&v[2])?]
            }
            Some(_) => return Err("velocity must be {vx, vy, vz} in m/s".into()),
        };
        let name = a.get("name").and_then(Value::as_str).unwrap_or("object").to_string();
        f.objects.push(Object {
            name,
            shape,
            pos: [at[0], at[1], at[2]],
            vel: [0.0; 3],
            half,
            drive,
            float: a.get("float").and_then(Value::as_bool).unwrap_or(false),
            density: a.get("density").and_then(Value::as_f64).unwrap_or(500.0) as f32,
            keys: a.get("keys").and_then(Value::as_bool).unwrap_or(false),
            speed: a.get("speed").and_then(Value::as_f64).unwrap_or(1.0) as f32,
        });
        let o = f.objects.last().unwrap().clone();
        // the water where it starts is pushed aside: those particles go
        let h = f.cell;
        f.particles.retain(|p| !o.contains([p[0] * h, p[1] * h, p[2] * h]));
        f.affine.clear();
        f.gpu_generation = 0;
        Ok(json!(f.objects.len()))
    });
    // fluid_metric{name, field = "smoke" | "temp" | "speed" | "water", op = "sum" | "mean" | "max" | "count", box = {...}?}
    w.register_command("fluid_metric", |w, a| {
        let f = w.get_mut::<Fluid>().ok_or("this universe has no fluids")?;
        f.changed = true;
        let name = a.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or("fluid_metric needs a name")?.to_string();
        let field = a.get("field").and_then(Value::as_str).unwrap_or("smoke").to_string();
        if !["smoke", "temp", "speed", "water", "pressure"].contains(&field.as_str()) {
            return Err(format!("field '{field}': use smoke, temp, speed, water or pressure"));
        }
        let op = a.get("op").and_then(Value::as_str).unwrap_or("sum").to_string();
        if !["sum", "mean", "max", "count"].contains(&op.as_str()) {
            return Err(format!("op '{op}': use sum, mean, max or count"));
        }
        let region = floats(a, "box", 6)?.map(|b| [b[0], b[1], b[2], b[3], b[4], b[5]]);
        f.metrics.retain(|m| m.name != name);
        f.metrics.push(FieldMetric { name, field, op, region });
        Ok(json!(true))
    });
    w.register_command("fluid_info", |w, _| {
        let f = w.get::<Fluid>().filter(|f| f.ready()).ok_or("make the fluid first: fluid{size = {64, 32, 32}, cell = 0.1}")?;
        Ok(json!({"kind": f.kind, "backend": if f.on_gpu() { "gpu" } else { "cpu" }, "liquid": f.settings.liquid, "density": f.settings.density, "cells": f.dims().cells(), "cell_m": f.cell, "dt_s": f.dt, "size_m": f.extent(), "seconds": f.seconds, "particles": f.particles.len(), "divergence": f.divergence, "emitters": f.emitters.len()}))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use simtara_core::{Config, Simulation};

    fn sim() -> Simulation {
        let mut s = Simulation::new(Config { seed: 3, ..Config::default() });
        s.add_module(simtara_things::ThingsModule);
        s.add_module(FluidModule);
        s
    }

    fn call(s: &mut Simulation, name: &str, a: Value) -> Value {
        s.world_mut().call(name, &a).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    #[test]
    fn hot_smoke_rises_and_wind_carries_it() {
        let mut s = sim();
        call(&mut s, "fluid", json!({"size": [24, 16, 32], "cell": 0.1}));
        call(&mut s, "emitter", json!({"at": [0.8, 0.8, 0.3], "radius": 0.2, "smoke": 4, "heat": 300}));
        s.advance(5);
        let m0 = s.collect_metrics();
        s.advance(40);
        let m1 = s.collect_metrics();
        assert!(m1["smoke.centre_height_m"] > m0["smoke.centre_height_m"] + 0.3, "hot smoke rises: {m0:?} -> {m1:?}");
        assert!(m1["fluid.divergence"] < 0.05, "air stays incompressible: {}", m1["fluid.divergence"]);

        // the same smoke, no heat, with a wind: it goes downstream instead
        let mut s = sim();
        call(&mut s, "fluid", json!({"size": [32, 16, 16], "cell": 0.1, "wind": 1.5}));
        call(&mut s, "emitter", json!({"at": [0.5, 0.8, 0.8], "radius": 0.2, "smoke": 4}));
        s.advance(30);
        let m = s.collect_metrics();
        assert!(m["smoke.centre_x_m"] > 1.0, "wind carries smoke: {m:?}");
    }

    #[test]
    fn heavy_cold_smoke_sinks() {
        let mut s = sim();
        call(&mut s, "fluid", json!({"size": [16, 16, 24], "cell": 0.1, "weight": 3, "open_top": false}));
        call(&mut s, "emitter", json!({"at": [0.8, 0.8, 1.8], "radius": 0.25, "smoke": 4}));
        s.advance(4);
        let h0 = s.collect_metrics()["smoke.centre_height_m"];
        s.advance(40);
        let h1 = s.collect_metrics()["smoke.centre_height_m"];
        assert!(h1 < h0 - 0.2, "{h0} -> {h1}");
    }

    #[test]
    fn a_solid_block_stops_air_and_shows_in_the_picture() {
        let mut s = sim();
        call(&mut s, "fluid", json!({"size": [32, 12, 12], "cell": 0.1, "wind": 2}));
        let n = call(&mut s, "solid", json!({"box": [1.2, 0, 0, 1.8, 1.2, 0.8]}));
        assert!(n.as_u64().unwrap() > 0);
        s.advance(20);
        let f = s.world().get::<Fluid>().unwrap();
        let mac = f.mac.as_ref().unwrap();
        let d = f.dims();
        for k in 0..d.nz {
            for j in 0..d.ny {
                for i in 0..d.nx {
                    if f.solid[d.c(i, j, k)] == SOLID {
                        let v = mac.centre(i, j, k);
                        assert!(v.iter().all(|x| x.abs() < 1e-4), "air inside a solid at {i},{j},{k}: {v:?}");
                    }
                }
            }
        }
        // air speeds up over the top of the block
        let over = mac.centre(15, 6, 10)[0];
        assert!(over > 2.0, "faster over the block: {over}");
        let pic = f.volume_rgba();
        assert_eq!(pic.len(), d.cells() * 4);
        assert_eq!(pic[d.c(15, 6, 2) * 4 + 2], 255, "solid marked in the picture");
    }

    #[test]
    fn still_water_stays_still_and_level() {
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [12, 12, 16], "cell": 0.05, "iterations": 80}));
        let n = call(&mut s, "water", json!({"box": [0, 0, 0, 0.6, 0.6, 0.3]})).as_u64().unwrap();
        assert_eq!(n, 12 * 12 * 6 * 8);
        let h0 = s.collect_metrics()["water.mean_height_m"];
        s.advance(60);
        let m = s.collect_metrics();
        assert_eq!(m["water.particles"], n as f64, "no water lost");
        assert!(m["fluid.fastest_m_s"] < 0.25, "still water stays still: {m:?}");
        assert!((m["water.mean_height_m"] - h0).abs() < 0.03, "level stays: {h0} -> {}", m["water.mean_height_m"]);
    }

    #[test]
    fn a_dam_break_spreads_no_faster_than_the_shallow_water_limit() {
        // water column 0.4 m high and 0.3 m long, released at t = 0
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [64, 6, 20], "cell": 0.025, "dt": 0.01}));
        call(&mut s, "water", json!({"box": [0, 0, 0, 0.3, 0.15, 0.4]}));
        s.advance(31);
        let m = s.collect_metrics();
        let t = m["fluid.seconds"];
        let front = m["water.front_m"];
        let ritter = 0.3 + 2.0 * (9.81f64 * 0.4).sqrt() * t;
        assert!(front > 2.0 * 0.3, "the column collapses and runs: front {front} m after {t} s");
        assert!(front < ritter + 0.05, "no faster than 2 sqrt(g h): {front} vs {ritter}");
        assert!(m["water.mean_height_m"] < 0.2 * 0.9, "it falls: {m:?}");
        // and nothing goes through the walls
        let f = s.world().get::<Fluid>().unwrap();
        let d = f.dims();
        assert!(f.particles.iter().all(|p| p[0] > 0.0 && p[0] < d.nx as f32 && p[1] > 0.0 && p[1] < d.ny as f32 && p[2] > 0.0 && p[2] < d.nz as f32));
    }

    #[test]
    fn controls_drive_the_fluid_and_runs_repeat() {
        let run = |wind: f64| {
            let mut s = sim();
            call(&mut s, "param", json!({"name": "wind", "value": wind, "min": 0, "max": 5}));
            call(&mut s, "fluid", json!({"size": [24, 10, 10], "cell": 0.1, "wind": "wind"}));
            call(&mut s, "emitter", json!({"at": [0.3, 0.5, 0.5], "radius": 0.2, "smoke": "wind"}));
            s.advance(15);
            s.collect_metrics()
        };
        let (a, b, c) = (run(2.0), run(2.0), run(0.5));
        assert_eq!(a, b, "same seed, same numbers");
        assert!(a["smoke.centre_x_m"] > c["smoke.centre_x_m"], "more wind, smoke further");

        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [16, 8, 12], "cell": 0.05}));
        call(&mut s, "emitter", json!({"at": [0.4, 0.2, 0.5], "radius": 0.06, "water": 0.002, "velocity": [1, 0, 0]}));
        s.advance(30);
        let p = s.collect_metrics()["water.particles"];
        assert!(p > 100.0, "a tap fills: {p}");
        let snap = serde_json::to_string(s.world().get::<Fluid>().unwrap()).unwrap();
        let back: Fluid = serde_json::from_str(&snap).unwrap();
        assert_eq!(back.particles, s.world().get::<Fluid>().unwrap().particles, "saves and loads exactly");
    }

    #[test]
    fn background_steps_match_steps_done_in_place_even_across_a_save() {
        let make = || {
            let mut f = Fluid::default();
            f.create(Kind::Smoke, Dims { nx: 16, ny: 10, nz: 12 }, 0.1, 1.0 / 30.0, 5);
            f.emitters.push(Emitter { at: [0.8, 0.5, 0.3], radius: 0.2, smoke: Num::N(3.0), heat: Num::P("heat".into()), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.5)], water: Num::N(0.0), carry: 0.0 });
            f
        };
        let mut params = BTreeMap::new();
        params.insert("heat".to_string(), Param { value: 300.0, min: 0.0, max: 500.0, about: String::new(), owner: String::new() });
        // in place: what you see after n ticks is n steps
        let mut a = make();
        for _ in 0..12 {
            a.tick(&params, false);
        }
        // in the background: always one step behind, so 13 ticks show 12 steps
        let mut b = make();
        for t in 0..13 {
            b.tick(&params, true);
            if t == 6 {
                // save while a step is running, reopen, and carry on
                let text = serde_json::to_string(&b).unwrap();
                b = serde_json::from_str(&text).unwrap();
            }
        }
        assert_eq!(a.smoke, b.smoke);
        assert_eq!(a.mac.as_ref().unwrap().w, b.mac.as_ref().unwrap().w);
        assert_eq!(a.seconds, b.seconds);

        // a change while a step runs (a new emitter) is not lost: that step is redone
        // (it applies from the step after the one you are looking at, in both ways of running)
        let extra = Emitter { at: [0.4, 0.5, 0.6], radius: 0.15, smoke: Num::N(5.0), heat: Num::N(0.0), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.0)], water: Num::N(0.0), carry: 0.0 };
        let mut c = make();
        for t in 0..6 {
            c.tick(&params, false);
            if t == 1 {
                c.emitters.push(extra.clone());
            }
        }
        let mut e = make();
        for t in 0..7 {
            e.tick(&params, true);
            if t == 2 {
                assert!((e.seconds - 2.0 * (1.0f32 / 30.0) as f64).abs() < 1e-9, "two steps visible: {}", e.seconds);
                e.emitters.push(extra.clone());
                e.changed = true;
            }
        }
        assert_eq!(c.smoke, e.smoke);
    }

    #[test]
    fn honey_runs_slower_than_water_and_a_cup_holds_liquid() {
        let front = |liquid: &str| {
            let mut s = sim();
            // a small column (5 cm x 6 cm), where viscosity matters, as on a kitchen table
            call(&mut s, "fluid", json!({"kind": "water", "liquid": liquid, "size": [48, 4, 16], "cell": 0.005, "dt": 0.004}));
            call(&mut s, "water", json!({"box": [0, 0, 0, 0.05, 0.02, 0.06]}));
            s.advance(60);
            let m = s.collect_metrics();
            (m["water.front_m"], m["water.mass_kg"], m["water.volume_m3"])
        };
        let (w, w_mass, w_vol) = front("water");
        let (o, _, _) = front("olive oil");
        let (h, h_mass, h_vol) = front("honey");
        eprintln!("front after 0.24 s: water {w:.3} m, olive oil {o:.3} m, honey {h:.3} m");
        assert!(h < 0.6 * w, "honey {h} m far behind water {w} m");
        assert!(o <= w + 0.02 && o > h, "olive oil in between: water {w}, oil {o}, honey {h}");
        assert!((w_mass / w_vol - 998.0).abs() < 1e-6 && (h_mass / h_vol - 1420.0).abs() < 1e-6, "mass = volume x density");

        // pour water into a glass: it stays in the glass
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "liquid": "water", "size": [20, 20, 24], "cell": 0.01, "dt": 0.01}));
        call(&mut s, "solid", json!({"cup": [0.1, 0.1, 0.06, 0.012, 0, 0.12]}));
        call(&mut s, "emitter", json!({"at": [0.1, 0.1, 0.2], "radius": 0.012, "water": 0.0001, "velocity": [0, 0, -0.5]}));
        call(&mut s, "fluid_metric", json!({"name": "in the glass or falling into it", "field": "water", "op": "sum", "box": [0.04, 0.04, 0, 0.16, 0.16, 0.24]}));
        s.advance(60);
        let m = s.collect_metrics();
        assert!(m["water.volume_m3"] > 4e-5, "poured: {m:?}");
        assert!(m["in the glass or falling into it"] > 0.99 * m["water.volume_m3"], "the glass holds it, nothing leaks out: {m:?}");
        assert!(s.world_mut().call("fluid", &json!({"liquid": "custard"})).unwrap_err().contains("honey"), "unknown liquids are explained");
    }

    #[test]
    fn still_water_presses_with_rho_g_h_and_keeps_its_volume() {
        // a tank of still water 0.3 m deep: the pressure at depth d is rho * g * d (hydrostatics)
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [8, 8, 24], "cell": 0.02, "dt": 0.01, "iterations": 200}));
        call(&mut s, "water", json!({"box": [0, 0, 0, 0.16, 0.16, 0.3]}));
        s.advance(40);
        let f = s.world().get::<Fluid>().unwrap();
        let d = f.dims();
        for k in [2usize, 6, 10] {
            let depth = 0.3 - (k as f32 + 0.5) * 0.02;
            let want = 998.0 * 9.81 * depth;
            let got = f.pressure_pa(d.c(4, 4, k));
            assert!((got - want).abs() < 0.08 * want, "depth {depth:.2} m: {got:.0} Pa, hydrostatics says {want:.0} Pa");
        }
        // and the water keeps its volume over a long, violent run (a dam break that sloshes)
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [40, 8, 20], "cell": 0.025, "dt": 0.01}));
        call(&mut s, "water", json!({"box": [0, 0, 0, 0.35, 0.2, 0.4]}));
        let v0 = s.collect_metrics()["water.filled_m3"];
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for _ in 0..30 {
            s.advance(10);
            let v = s.collect_metrics()["water.filled_m3"];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo > 0.95 * v0 && hi < 1.05 * v0, "volume stays within 5%: {v0:.4} m³, range {lo:.4}..{hi:.4}");
    }

    #[test]
    fn a_tank_sloshes_at_the_period_linear_wave_theory_gives() {
        // tank 0.5 m long, water 0.2 m deep, surface tilted a little: it rocks back and forth.
        // First mode: omega² = g k tanh(k H) with k = pi / L, so T = 2 pi / omega ≈ 0.868 s.
        let mut s = sim();
        call(&mut s, "fluid", json!({"kind": "water", "size": [40, 2, 24], "cell": 0.0125, "dt": 0.005, "iterations": 80}));
        call(&mut s, "water", json!({"box": [0, 0, 0, 0.25, 0.025, 0.215]}));
        call(&mut s, "water", json!({"box": [0.25, 0, 0, 0.5, 0.025, 0.185]}));
        let (l, depth, g) = (0.5f64, 0.2f64, 9.81f64);
        let k = std::f64::consts::PI / l;
        let theory = 2.0 * std::f64::consts::PI / (g * k * (k * depth).tanh()).sqrt();
        let mut xs = Vec::new();
        for _ in 0..500 {
            s.advance(1);
            let m = s.collect_metrics();
            xs.push((m["fluid.seconds"], m["water.centre_x_m"]));
        }
        let mean = xs.iter().map(|x| x.1).sum::<f64>() / xs.len() as f64;
        // times the centre of the water crosses its middle going the same way
        let ups: Vec<f64> = xs.windows(2).filter(|w| w[0].1 < mean && w[1].1 >= mean).map(|w| w[0].0).collect();
        assert!(ups.len() >= 2, "it should rock: {ups:?}");
        let period = (ups[ups.len() - 1] - ups[0]) / (ups.len() - 1) as f64;
        eprintln!("sloshing period {period:.3} s, theory {theory:.3} s");
        assert!((period - theory).abs() < 0.1 * theory, "period {period:.3} s vs theory {theory:.3} s");
    }

    #[test]
    fn the_gpu_solver_agrees_with_the_cpu_one_and_repeats() {
        if !gpu::available() {
            eprintln!("no graphics card: GPU liquid solver not tested");
            return;
        }
        let run = |backend: &str| {
            let mut s = sim();
            call(&mut s, "fluid", json!({"kind": "water", "backend": backend, "size": [40, 8, 20], "cell": 0.025, "dt": 0.01}));
            call(&mut s, "water", json!({"box": [0, 0, 0, 0.35, 0.2, 0.4]}));
            call(&mut s, "solid", json!({"box": [0.6, 0.05, 0, 0.7, 0.15, 0.1]}));
            let info = s.world_mut().call("fluid_info", &json!({})).unwrap();
            s.advance(40);
            (info["backend"].clone(), s.collect_metrics(), s.world().get::<Fluid>().unwrap().particles.clone())
        };
        let (b_cpu, cpu, _) = run("cpu");
        let (b_gpu, gpu1, p1) = run("gpu");
        let (_, gpu2, p2) = run("gpu");
        assert_eq!((b_cpu, b_gpu), (json!("cpu"), json!("gpu")));
        assert_eq!(p1, p2, "the GPU repeats exactly");
        assert_eq!(gpu1, gpu2);
        let close = |k: &str, tol: f64| {
            let (a, b) = (cpu[k], gpu1[k]);
            assert!((a - b).abs() <= tol * a.abs().max(1e-6), "{k}: cpu {a}, gpu {b}");
        };
        close("water.filled_m3", 0.04);
        close("water.mean_height_m", 0.08);
        close("water.centre_x_m", 0.08);
        close("water.front_m", 0.1);
        assert_eq!(cpu["water.particles"], gpu1["water.particles"]);
    }

    #[test]
    #[ignore]
    fn bench_gpu_liquid() {
        if !gpu::available() {
            return;
        }
        let mut f = Fluid::default();
        f.create(Kind::Water, Dims { nx: 128, ny: 48, nz: 64 }, 0.02, 0.02, 1);
        f.add_water([0.0, 0.0, 0.0], [0.7, 0.96, 0.9]);
        let p = BTreeMap::new();
        for sweeps_scale in [1usize, 5] {
            f.settings.iterations = Num::N(60.0 / sweeps_scale as f64 * 1.0);
            let t = std::time::Instant::now();
            for _ in 0..10 {
                f.step(&p);
            }
            eprintln!("{} particles, {} sweeps: {:.1} ms per tick", f.particles.len(), 300 / sweeps_scale, t.elapsed().as_secs_f64() * 100.0);
        }
        let time = |name: &str, run: &dyn Fn()| {
            let t = std::time::Instant::now();
            run();
            eprintln!("{name}: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        };
        time("measurements", &|| drop(f.summary()));
        time("water points (spray)", &|| drop(f.water_points()));
        time("liquid field", &|| drop(f.liquid_field()));
        time("caustics", &|| drop(f.caustics([0.3, 0.3, 0.9], 4)));
        time("volume picture", &|| drop(f.volume_rgba()));
        let t = std::time::Instant::now();
        let _ = f.clone();
        eprintln!("clone for the background step: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    #[test]
    fn objects_push_water_and_float_or_sink_by_their_density() {
        for backend in ["cpu", "gpu"] {
            if backend == "gpu" && !gpu::available() {
                continue;
            }
            // a paddle (box) driven through still water pushes it along
            let mut s = sim();
            call(&mut s, "fluid", json!({"kind": "water", "backend": backend, "size": [40, 8, 16], "cell": 0.025, "dt": 0.01}));
            call(&mut s, "water", json!({"box": [0, 0, 0, 1.0, 0.2, 0.15]}));
            let x0 = s.collect_metrics()["water.centre_x_m"];
            call(&mut s, "object", json!({"name": "paddle", "shape": "box", "at": [0.1, 0.1, 0.1], "size": [0.05, 0.2, 0.2], "velocity": [1.0, 0, 0]}));
            s.advance(40);
            let m = s.collect_metrics();
            assert!(m["paddle.x_m"] > 0.4, "{backend}: the paddle moved: {m:?}");
            assert!(m["water.centre_x_m"] > x0 + 0.05, "{backend}: water pushed ahead of it: {x0} -> {}", m["water.centre_x_m"]);
            let h = 0.025;
            let f = s.world().get::<Fluid>().unwrap();
            let ob = &f.objects[0];
            let inside = f.particles.iter().filter(|p| {
                let q = [p[0] * h - ob.pos[0], p[1] * h - ob.pos[1], p[2] * h - ob.pos[2]];
                q[0].abs() < ob.half[0] * 0.8 && q[1].abs() < ob.half[1] * 0.8 && q[2].abs() < ob.half[2] * 0.8
            });
            assert_eq!(inside.count(), 0, "{backend}: no water inside the paddle");

            // a light ball floats at the water line; a heavy one sinks to the floor
            for (density, floats) in [(400.0, true), (3000.0, false)] {
                let mut s = sim();
                call(&mut s, "fluid", json!({"kind": "water", "backend": backend, "size": [24, 24, 32], "cell": 0.025, "dt": 0.01}));
                call(&mut s, "water", json!({"box": [0, 0, 0, 0.6, 0.6, 0.4]}));
                call(&mut s, "object", json!({"name": "ball", "shape": "ball", "at": [0.3, 0.3, 0.55], "radius": 0.08, "float": true, "density": density}));
                s.advance(150);
                let z = s.collect_metrics()["ball.height_m"];
                if floats {
                    assert!(z > 0.36 && z < 0.5, "{backend}: density {density} floats at the surface (~0.4 m): {z}");
                } else {
                    assert!(z < 0.15, "{backend}: density {density} sinks to the floor: {z}");
                }
            }
        }
        let mut s = sim();
        call(&mut s, "fluid", json!({"size": [8, 8, 8]}));
        assert!(s.world_mut().call("object", &json!({"shape": "ball", "at": [0.4, 0.4, 0.4], "radius": 0.1})).unwrap_err().contains("liquid"));
    }

    #[test]
    fn bad_input_explains_itself() {
        let mut s = sim();
        assert!(s.world_mut().call("solid", &json!({"box": [0, 0, 0, 1, 1, 1]})).unwrap_err().contains("fluid{"));
        assert!(s.world_mut().call("fluid", &json!({})).unwrap_err().contains("size"));
        call(&mut s, "fluid", json!({"size": [8, 8, 8]}));
        assert!(s.world_mut().call("solid", &json!({"box": [0, 1]})).unwrap_err().contains("6 numbers"));
        assert!(s.world_mut().call("water", &json!({"box": [0, 0, 0, 1, 1, 1]})).unwrap_err().contains("water"));
        assert!(s.world_mut().call("fluid_metric", &json!({"name": "x", "field": "mud"})).unwrap_err().contains("mud"));
    }
}
