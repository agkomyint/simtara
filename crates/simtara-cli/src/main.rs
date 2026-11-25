//! `simtara` CLI.
//!
//!   simtara run  [opts]   headless: advance N ticks, print stats, optionally save/render
//!   simtara view [opts]   live window: Play/Pause, Reset, Gravity, Count, Speed
//!
//! opts: --particles N  --kinds K  --seed S  --steps N  --spread X  --base X
//!       --load FILE  --save FILE  --png FILE

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use simtara_core::{Config, RunState, Simulation};
use simtara_particles::{simulation, stats, Boundary, ParticleConfig, ParticleParams, Particles};
use std::collections::HashMap;

const WIN: usize = 900;

fn parse(args: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(k) = args[i].strip_prefix("--") {
            m.insert(k.to_string(), args.get(i + 1).cloned().unwrap_or_default());
            i += 1;
        }
        i += 1;
    }
    m
}

fn get<T: std::str::FromStr>(o: &HashMap<String, String>, k: &str, d: T) -> T {
    o.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn build(o: &HashMap<String, String>) -> Result<Simulation, String> {
    if let Some(f) = o.get("load") {
        let snap = Simulation::read_snapshot(f)?;
        let pc: ParticleConfig = snap.config.module(simtara_particles::MODULE);
        let mut sim = simulation(snap.config.clone(), &pc);
        sim.restore(&snap)?;
        return Ok(sim);
    }
    let mut pc = ParticleConfig::default();
    pc.count = get(o, "particles", 5000);
    pc.params.kinds = get(o, "kinds", 4);
    pc.spread = get(o, "spread", 0.8);
    pc.base_attraction = get(o, "base", 0.0);
    let cfg = Config { seed: get(o, "seed", 1), ..Config::default() };
    Ok(simulation(cfg, &pc))
}

fn report(sim: &Simulation) {
    if let Some(s) = stats(sim.world()) {
        println!(
            "tick {:>6}  t={:>7.2}s  particles={}  KE/particle={:>8.2}  neighbors(r<rmax/2)={:>6.2}",
            sim.world().clock.tick,
            sim.world().clock.time,
            s.count,
            s.kinetic_energy,
            s.mean_neighbors
        );
    }
}

fn hue(k: u8, kinds: usize) -> u32 {
    let h = k as f32 / kinds.max(1) as f32 * 6.0;
    let x = 1.0 - ((h % 2.0) - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    let c = |v: f32| (60.0 + 195.0 * v) as u32;
    (c(r) << 16) | (c(g) << 8) | c(b)
}

fn render(sim: &Simulation, size: usize, buf: &mut [u32]) {
    buf.fill(0x07080c);
    let (Some(p), Some(prm)) = (sim.world().get::<Particles>(), sim.world().get::<ParticleParams>()) else { return };
    let (sx, sy) = (size as f32 / prm.width, size as f32 / prm.height);
    for i in 0..p.len() {
        let c = hue(p.kind[i], prm.kinds);
        let (x, y) = ((p.x[i] * sx) as isize, (p.y[i] * sy) as isize);
        for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (px, py) = (x + ox, y + oy);
            if px >= 0 && py >= 0 && (px as usize) < size && (py as usize) < size {
                buf[py as usize * size + px as usize] = c;
            }
        }
    }
}

fn save_png(path: &str, size: usize, buf: &[u32]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size as u32, size as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    let data: Vec<u8> = buf.iter().flat_map(|c| [(c >> 16) as u8, (c >> 8) as u8, *c as u8]).collect();
    w.write_image_data(&data).map_err(|e| e.to_string())
}

fn run(o: &HashMap<String, String>) -> Result<(), String> {
    let mut sim = build(o)?;
    let steps: u64 = get(o, "steps", 600);
    let every = (steps / 6).max(1);
    report(&sim);
    for _ in 0..steps / every {
        sim.advance(every);
        report(&sim);
    }
    if let Some(f) = o.get("save") {
        sim.save_to(f)?;
        println!("saved snapshot -> {f}");
    }
    if let Some(f) = o.get("png") {
        let mut buf = vec![0u32; 1000 * 1000];
        render(&sim, 1000, &mut buf);
        save_png(f, 1000, &buf)?;
        println!("rendered -> {f}");
    }
    Ok(())
}

/// Apply a new particle count / current live params and rebuild the universe.
fn reconfigure(sim: &mut Simulation, count: usize, keep_live_params: bool) {
    let mut pc: ParticleConfig = sim.config().module(simtara_particles::MODULE);
    pc.count = count;
    if keep_live_params {
        if let Some(p) = sim.world().get::<ParticleParams>() {
            pc.params = p.clone();
        }
    }
    sim.config_mut().set_module(simtara_particles::MODULE, &pc);
    sim.reset();
}

fn view(o: &HashMap<String, String>) -> Result<(), String> {
    let mut sim = build(o)?;
    let initial: ParticleConfig = sim.config().module(simtara_particles::MODULE);
    let mut window = Window::new("SimTara — Universe 001", WIN, WIN, WindowOptions::default()).map_err(|e| e.to_string())?;
    window.set_target_fps(60);
    let mut buf = vec![0u32; WIN * WIN];
    sim.play();
    println!("Space play/pause | R reset | Up/Down gravity | PgUp/PgDn particle count | Left/Right speed | B bounce/wrap | S save | L load | Esc quit");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let press = |k| window.is_key_pressed(k, KeyRepeat::No);
        let hold = |k| window.is_key_pressed(k, KeyRepeat::Yes);
        let count = sim.world().get::<Particles>().map(|p| p.len()).unwrap_or(0);

        if press(Key::Space) {
            sim.toggle();
        }
        if press(Key::R) {
            let was = sim.run_state();
            reconfigure(&mut sim, initial.count, false);
            if was == RunState::Running {
                sim.play();
            }
        }
        if hold(Key::Up) || hold(Key::Down) {
            let f = if window.is_key_down(Key::Up) { 1.05 } else { 1.0 / 1.05 };
            if let Some(p) = sim.world_mut().get_mut::<ParticleParams>() {
                p.gravity = (p.gravity * f).clamp(0.0, 20.0);
            }
        }
        if hold(Key::Right) {
            sim.set_speed(sim.speed() * 1.15);
        }
        if hold(Key::Left) {
            sim.set_speed(sim.speed() / 1.15);
        }
        if press(Key::PageUp) {
            reconfigure(&mut sim, (count * 3 / 2).clamp(1000, 100_000), true);
        }
        if press(Key::PageDown) {
            reconfigure(&mut sim, (count * 2 / 3).clamp(1000, 100_000), true);
        }
        if press(Key::B) {
            if let Some(p) = sim.world_mut().get_mut::<ParticleParams>() {
                p.boundary = if p.boundary == Boundary::Bounce { Boundary::Wrap } else { Boundary::Bounce };
            }
        }
        if press(Key::S) {
            match sim.save_to("universe.snap.json") {
                Ok(()) => println!("saved universe.snap.json"),
                Err(e) => eprintln!("save failed: {e}"),
            }
        }
        if press(Key::L) {
            match Simulation::read_snapshot("universe.snap.json").and_then(|s| sim.restore(&s)) {
                Ok(()) => println!("loaded universe.snap.json"),
                Err(e) => eprintln!("load failed: {e}"),
            }
        }

        sim.run_frame();
        render(&sim, WIN, &mut buf);
        let (g, b) = sim.world().get::<ParticleParams>().map(|p| (p.gravity, p.boundary)).unwrap_or((0.0, Boundary::Bounce));
        window.set_title(&format!(
            "SimTara — {} | tick {} | {} particles | gravity {:.2} | speed {:.2}x | {:?}",
            if sim.run_state() == RunState::Running { "▶" } else { "⏸" },
            sim.world().clock.tick,
            count,
            g,
            sim.speed(),
            b
        ));
        window.update_with_buffer(&buf, WIN, WIN).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("view");
    let opts = parse(args.get(1..).unwrap_or(&[]));
    let result = match cmd {
        "run" => run(&opts),
        "view" => view(&opts),
        _ => {
            println!("usage: simtara <run|view> [--particles N] [--kinds K] [--seed S] [--steps N] [--spread X] [--base X] [--load F] [--save F] [--png F]");
            return;
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
