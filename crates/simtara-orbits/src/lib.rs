//! Orbital gravity: a handful of bodies that pull on each other.
//!
//! Real SI units (metres, kilograms, seconds), so numbers can be checked against
//! the real world. Motion uses the leapfrog (velocity Verlet) method, which keeps
//! orbits stable for a long time. Everything is plain f64 on the CPU, so a run
//! repeats exactly.
//!
//! Plugs in like any module and offers these commands to scripts and tools:
//! `remove_body {name}`, `hide_body {name}`, `show_body {name}`, `set_mass {name, mass}`, `body {name}`, `bodies`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use simtara_core::{Builder, Config, Context, Module, Simulation, Stage, System, World};

pub const MODULE: &str = "orbits";

/// Newton's gravitational constant, m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;
pub const AU: f64 = 1.495_978_707e11;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    /// kilograms
    pub mass: f64,
    /// metres
    pub pos: [f64; 2],
    /// metres per second
    pub vel: [f64; 2],
    /// metres, for drawing only
    pub radius: f64,
    pub color: [u8; 3],
    /// A hidden body still moves but pulls on nothing (as if it were not there). `show_body` brings it back.
    #[serde(default)]
    pub hidden: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OrbitParams {
    pub g: f64,
    /// Simulated seconds that pass in one tick.
    pub seconds_per_tick: f64,
    /// Integration steps per tick (more = more accurate).
    pub substeps: u32,
    /// Metres; avoids infinite force when two bodies touch.
    pub softening: f64,
}

impl Default for OrbitParams {
    fn default() -> Self {
        OrbitParams { g: G, seconds_per_tick: 6.0 * 3600.0, substeps: 6, softening: 0.0 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OrbitConfig {
    pub bodies: Vec<Body>,
    pub params: OrbitParams,
}

impl OrbitConfig {
    /// The Sun, the Earth and the Moon on circular orbits, flat (2D), real masses and distances.
    /// The Sun gets a tiny velocity so the system as a whole does not drift.
    pub fn sun_earth_moon() -> Self {
        let (ms, me, mm) = (1.988_47e30, 5.972_2e24, 7.342e22);
        let (re, rm) = (AU, 3.844e8);
        // The Earth-Moon pair as a whole moves on the circular orbit around the Sun;
        // the Earth and Moon circle their shared centre of mass.
        let ve = (G * ms / re).sqrt();
        let vm = (G * me / rm).sqrt();
        let (fm, fe) = (mm / (me + mm), me / (me + mm));
        let earth_pos = [re - rm * fm, 0.0];
        let moon_pos = [re + rm * fe, 0.0];
        let earth_v = [0.0, ve - vm * fm];
        let moon_v = [0.0, ve + vm * fe];
        let sun_v = [0.0, -(me * earth_v[1] + mm * moon_v[1]) / ms];
        OrbitConfig {
            bodies: vec![
                Body { name: "Sun".into(), mass: ms, pos: [0.0, 0.0], vel: sun_v, radius: 6.957e8, color: [255, 205, 70], hidden: false },
                Body { name: "Earth".into(), mass: me, pos: earth_pos, vel: earth_v, radius: 6.371e6, color: [90, 160, 255], hidden: false },
                Body { name: "Moon".into(), mass: mm, pos: moon_pos, vel: moon_v, radius: 1.737e6, color: [200, 200, 205], hidden: false },
            ],
            params: OrbitParams::default(),
        }
    }
}

/// The live bodies. Saved in snapshots.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Bodies {
    pub list: Vec<Body>,
}

impl Bodies {
    pub fn find(&self, name: &str) -> Option<&Body> {
        self.list.iter().find(|b| b.name == name)
    }
    /// Total energy (kinetic + gravitational), joules.
    pub fn energy(&self, p: &OrbitParams) -> f64 {
        let mut e = 0.0;
        for (i, a) in self.list.iter().enumerate().filter(|(_, a)| !a.hidden) {
            e += 0.5 * a.mass * (a.vel[0] * a.vel[0] + a.vel[1] * a.vel[1]);
            for b in self.list[i + 1..].iter().filter(|b| !b.hidden) {
                let d = dist(a.pos, b.pos).max(1.0);
                e -= p.g * a.mass * b.mass / d;
            }
        }
        e
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn accelerations(list: &[Body], p: &OrbitParams, out: &mut Vec<[f64; 2]>) {
    out.clear();
    out.resize(list.len(), [0.0, 0.0]);
    let eps2 = p.softening * p.softening;
    for i in 0..list.len() {
        for j in (i + 1)..list.len() {
            let d = [list[j].pos[0] - list[i].pos[0], list[j].pos[1] - list[i].pos[1]];
            let r2 = d[0] * d[0] + d[1] * d[1] + eps2;
            if r2 <= 0.0 {
                continue;
            }
            let inv = p.g / (r2 * r2.sqrt());
            let src = |b: &Body| if b.hidden { 0.0 } else { b.mass };
            let (fi, fj) = (src(&list[j]) * inv, src(&list[i]) * inv);
            out[i][0] += d[0] * fi;
            out[i][1] += d[1] * fi;
            out[j][0] -= d[0] * fj;
            out[j][1] -= d[1] * fj;
        }
    }
}

/// Advance the bodies by `p.seconds_per_tick`.
pub fn step(bodies: &mut Bodies, p: &OrbitParams) {
    let n = p.substeps.max(1);
    let h = p.seconds_per_tick / n as f64;
    let mut acc = Vec::new();
    accelerations(&bodies.list, p, &mut acc);
    for _ in 0..n {
        for (b, a) in bodies.list.iter_mut().zip(&acc) {
            b.vel[0] += 0.5 * h * a[0];
            b.vel[1] += 0.5 * h * a[1];
            b.pos[0] += h * b.vel[0];
            b.pos[1] += h * b.vel[1];
        }
        accelerations(&bodies.list, p, &mut acc);
        for (b, a) in bodies.list.iter_mut().zip(&acc) {
            b.vel[0] += 0.5 * h * a[0];
            b.vel[1] += 0.5 * h * a[1];
        }
    }
}

struct OrbitSystem;
impl System for OrbitSystem {
    fn name(&self) -> &str {
        "orbits.gravity"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(mut b) = ctx.world.take::<Bodies>() else { return };
        if let Some(p) = ctx.world.get::<OrbitParams>() {
            step(&mut b, p);
        }
        ctx.world.put(b);
    }
}

fn name_arg(args: &Value) -> Result<&str, String> {
    args.as_str().or_else(|| args.get("name").and_then(Value::as_str)).ok_or_else(|| "needs a body name, e.g. {name = \"Moon\"}".to_string())
}

fn register_commands(w: &mut World) {
    w.register_command("remove_body", |w, a| {
        let name = name_arg(a)?.to_string();
        let b = w.get_mut::<Bodies>().ok_or("no orbits in this universe")?;
        let before = b.list.len();
        b.list.retain(|x| x.name != name);
        Ok(json!(b.list.len() < before))
    });
    for (cmd, hide) in [("hide_body", true), ("show_body", false)] {
        w.register_command(cmd, move |w, a| {
            let name = name_arg(a)?.to_string();
            let b = w.get_mut::<Bodies>().ok_or("no orbits in this universe")?;
            match b.list.iter_mut().find(|x| x.name == name) {
                Some(x) => {
                    let changed = x.hidden != hide;
                    x.hidden = hide;
                    Ok(json!(changed))
                }
                None => Ok(json!(false)),
            }
        });
    }
    w.register_command("set_mass", |w, a| {
        let name = name_arg(a)?.to_string();
        let mass = a
            .get("mass")
            .and_then(Value::as_f64)
            .filter(|m| *m >= 0.0 && m.is_finite())
            .ok_or("needs a mass in kg, e.g. {name = \"Moon\", mass = 1e22}")?;
        let b = w.get_mut::<Bodies>().ok_or("no orbits in this universe")?;
        match b.list.iter_mut().find(|x| x.name == name) {
            Some(x) => {
                x.mass = mass;
                Ok(json!(true))
            }
            None => Ok(json!(false)),
        }
    });
    w.register_command("body", |w, a| {
        let name = name_arg(a)?;
        let b = w.get::<Bodies>().ok_or("no orbits in this universe")?;
        Ok(match b.find(name) {
            Some(x) => json!({"name": x.name, "mass": x.mass, "x": x.pos[0], "y": x.pos[1], "vx": x.vel[0], "vy": x.vel[1]}),
            None => Value::Null,
        })
    });
    w.register_command("bodies", |w, _| {
        let b = w.get::<Bodies>().ok_or("no orbits in this universe")?;
        Ok(json!(b.list.iter().map(|x| x.name.clone()).collect::<Vec<_>>()))
    });
}

pub struct OrbitModule;

impl Module for OrbitModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        let cfg: OrbitConfig = b.config.module(MODULE);
        b.world.insert_persistent("bodies", Bodies { list: cfg.bodies.clone() });
        b.world.insert_persistent("orbit_params", cfg.params.clone());
        b.add_system(Stage::Update, OrbitSystem);
        register_commands(b.world);
        b.add_metric("orbits.bodies", |w| w.get::<Bodies>().map_or(0.0, |b| b.list.iter().filter(|x| !x.hidden).count() as f64));
        b.add_metric("orbits.energy", |w| match (w.get::<Bodies>(), w.get::<OrbitParams>()) {
            (Some(b), Some(p)) => b.energy(p),
            _ => 0.0,
        });
        // Per-body measurements. Names come from the config; a removed body reads 0.
        if let Some(first) = cfg.bodies.first() {
            let center = first.name.clone();
            for body in cfg.bodies.iter().skip(1) {
                let (n, c) = (body.name.clone(), center.clone());
                b.add_metric(&format!("orbits.distance_km.{n}"), move |w| {
                    let Some(list) = w.get::<Bodies>() else { return 0.0 };
                    match (list.find(&n), list.find(&c)) {
                        (Some(a), Some(s)) => dist(a.pos, s.pos) / 1000.0,
                        _ => 0.0,
                    }
                });
                let n = body.name.clone();
                b.add_metric(&format!("orbits.speed_kms.{n}"), move |w| {
                    w.get::<Bodies>().and_then(|b| b.find(&n).map(|x| (x.vel[0].powi(2) + x.vel[1].powi(2)).sqrt() / 1000.0)).unwrap_or(0.0)
                });
            }
        }
    }
}

/// Convenience: a simulation with the orbits module installed.
pub fn simulation(mut config: Config, oc: &OrbitConfig) -> Simulation {
    config.set_module(MODULE, oc);
    let mut sim = Simulation::new(config);
    sim.add_module(OrbitModule);
    sim
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 365.25 days at 6 hours per tick.
    const YEAR: u64 = 1461;

    #[test]
    fn earth_stays_on_its_orbit_for_a_year() {
        let mut sim = simulation(Config::default(), &OrbitConfig::sun_earth_moon());
        let d0 = sim.collect_metrics()["orbits.distance_km.Earth"];
        sim.advance(YEAR);
        let d1 = sim.collect_metrics()["orbits.distance_km.Earth"];
        assert!(((d1 - d0) / d0).abs() < 0.02, "distance changed {d0} -> {d1}");
        let b = sim.world().get::<Bodies>().unwrap();
        let (e, s) = (b.find("Earth").unwrap(), b.find("Sun").unwrap());
        let ang = (e.pos[1] - s.pos[1]).atan2(e.pos[0] - s.pos[0]);
        assert!(ang.abs() < 0.1, "angle after a year: {ang}");
    }

    #[test]
    fn moon_goes_around_in_about_27_days() {
        let mut sim = simulation(Config::default(), &OrbitConfig::sun_earth_moon());
        let rel = |sim: &Simulation| {
            let b = sim.world().get::<Bodies>().unwrap();
            let (e, m) = (b.find("Earth").unwrap(), b.find("Moon").unwrap());
            (m.pos[1] - e.pos[1]).atan2(m.pos[0] - e.pos[0])
        };
        let a0 = rel(&sim);
        sim.advance(4 * 27); // 27 days = 108 ticks of 6 h
        let mut d = (rel(&sim) - a0).abs();
        if d > std::f64::consts::PI {
            d = 2.0 * std::f64::consts::PI - d;
        }
        assert!(d < 0.35, "moon angle off by {d}");
    }

    #[test]
    fn energy_is_nearly_conserved() {
        let mut sim = simulation(Config::default(), &OrbitConfig::sun_earth_moon());
        let e0 = sim.collect_metrics()["orbits.energy"];
        sim.advance(YEAR * 2);
        let e1 = sim.collect_metrics()["orbits.energy"];
        assert!(((e1 - e0) / e0).abs() < 1e-4, "energy drifted {e0} -> {e1}");
    }

    #[test]
    fn commands_work_and_removal_is_safe_to_repeat() {
        let mut sim = simulation(Config::default(), &OrbitConfig::sun_earth_moon());
        assert_eq!(sim.world_mut().call("remove_body", &json!({"name": "Moon"})).unwrap(), json!(true));
        assert_eq!(sim.world_mut().call("remove_body", &json!("Moon")).unwrap(), json!(false));
        assert_eq!(sim.collect_metrics()["orbits.bodies"], 2.0);
        assert_eq!(sim.collect_metrics()["orbits.distance_km.Moon"], 0.0);
        assert!(sim.world_mut().call("nope", &json!(null)).is_err());
        sim.advance(10);
    }

    #[test]
    fn save_and_restore_continue_identically() {
        let mut a = simulation(Config::default(), &OrbitConfig::sun_earth_moon());
        a.advance(100);
        let snap = a.snapshot().unwrap();
        let mut b = Simulation::new(snap.config.clone());
        b.add_module(OrbitModule);
        b.restore(&snap).unwrap();
        a.advance(100);
        b.advance(100);
        assert_eq!(a.snapshot().unwrap().state, b.snapshot().unwrap().state);
    }
}
