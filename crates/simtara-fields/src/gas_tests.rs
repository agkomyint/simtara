//! Checks of the gas solver against measurements and exact results.

use crate::grid::{idx, Dims, SOLID};
use crate::*;
use std::collections::BTreeMap;

fn gas_box(d: Dims, cell: f32, dt: f32) -> Fluid {
    let mut f = Fluid::default();
    f.create(Kind::Smoke, d, cell, dt, 7);
    f.use_gpu = false;
    f.settings.vorticity = Num::N(0.0);
    f.settings.cooling = Num::N(0.0);
    f.open_top = false;
    f
}

fn run(f: &mut Fluid, ticks: usize) {
    let p = BTreeMap::new();
    for _ in 0..ticks {
        f.step(&p);
    }
}

/// Lid-driven cavity at Re = 100 against Ghia, Ghia & Shin (1982), the standard test of a
/// viscous flow solver: a square box whose lid slides; the air inside turns into one big swirl.
#[test]
fn lid_driven_cavity_matches_ghia_at_re_100() {
    let n = 40;
    let mut f = gas_box(Dims { nx: n, ny: 1, nz: n }, 1.0 / n as f32, 0.01);
    f.settings.viscosity = Num::N(0.01); // U L / nu = 1 * 1 / 0.01 = 100
    f.settings.turbulence = Num::N(0.0);
    f.settings.gravity = Num::N(0.0);
    f.settings.iterations = Num::N(200.0);
    f.no_slip = true;
    f.lid = 1.0;
    run(&mut f, 1500);
    let mac = f.mac.as_ref().unwrap();
    let d = f.dims();
    // u on the vertical centre line (faces at x = 0.5), heights at cell centres
    let line: Vec<(f32, f32)> = (0..d.nz).map(|k| ((k as f32 + 0.5) / n as f32, mac.u[idx(d.ud(), n / 2, 0, k)])).collect();
    let at = |y: f32| {
        let i = line.iter().position(|p| p.0 >= y).unwrap_or(line.len() - 1).max(1);
        let (a, b) = (line[i - 1], line[i]);
        let t = ((y - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
        a.1 + t * (b.1 - a.1)
    };
    let mut worst = 0.0f32;
    for (y, want) in gas::GHIA_RE100_U.iter().filter(|p| p.0 > 0.03 && p.0 < 0.97) {
        let got = at(*y);
        worst = worst.max((got - want).abs());
    }
    let min_u = line.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    eprintln!("cavity: worst |u - Ghia| = {worst:.4}, min u = {min_u:.4} (Ghia -0.2109)");
    assert!(worst < 0.05, "centre-line velocity within 0.05 U of Ghia et al.: {worst}");
    assert!((min_u + 0.211).abs() < 0.03, "strongest return flow {min_u} vs -0.211");
}

/// Flow past a cylinder at Re = 100 sheds a von Karman vortex street at a Strouhal number of
/// about 0.165 (Williamson 1996; a little higher with walls this close).
#[test]
fn a_cylinder_sheds_vortices_at_the_right_strouhal_number() {
    let (dia, u0) = (0.1f32, 1.0f32);
    let h = dia / 10.0;
    let d = Dims { nx: 240, ny: 80, nz: 1 };
    let mut f = gas_box(d, h, 0.01);
    // on the graphics card when there is one (what Studio runs), else the CPU
    f.use_gpu = gpu_smoke::available();
    f.settings.wind = Some(Num::N(u0 as f64));
    f.open_side = true;
    f.settings.viscosity = Num::N((u0 * dia / 100.0) as f64);
    f.settings.turbulence = Num::N(0.0);
    f.settings.gravity = Num::N(0.0);
    f.settings.iterations = Num::N(60.0);
    let (cx, cy) = (0.5f32, 0.4 + 0.5 * h);
    f.add_solid(|p| (p[0] - cx).powi(2) + (p[1] - cy).powi(2) <= (dia / 2.0).powi(2));
    run(&mut f, 400);
    // the side-to-side velocity two diameters behind it swings back and forth
    let probe = |f: &Fluid| {
        let mac = f.mac.as_ref().unwrap();
        let (i, j) = (((cx + 2.0 * dia) / h) as usize, (cy / h) as usize);
        mac.v[idx(f.dims().vd(), i, j, 0)]
    };
    let (mut last, mut crossings, mut first, mut last_t) = (probe(&f), 0usize, None, 0.0f64);
    let mut amp = 0.0f32;
    for _ in 0..500 {
        run(&mut f, 1);
        let v = probe(&f);
        amp = amp.max(v.abs());
        if last < 0.0 && v >= 0.0 {
            let t = f.seconds;
            if first.is_none() {
                first = Some(t);
            } else {
                crossings += 1;
            }
            last_t = t;
        }
        last = v;
    }
    assert!(amp > 0.1 * u0, "it sheds: side velocity swings by {amp} m/s");
    let period = (last_t - first.unwrap()) / crossings.max(1) as f64;
    let st = dia as f64 / (period * u0 as f64);
    eprintln!("cylinder: {crossings} periods, St = {st:.3}");
    assert!(crossings >= 3 && (0.15..0.21).contains(&st), "Strouhal number {st} (expected about 0.165-0.18)");
}

/// Heated air expands: the volume that leaves through the open top in the step it is heated is
/// the volume it gained (ideal gas: V grows by T1 / T0).
#[test]
fn heated_air_expands_by_the_ideal_gas_law() {
    let d = Dims { nx: 12, ny: 12, nz: 12 };
    let h = 0.1;
    let dt = 0.01;
    let mut f = gas_box(d, h, dt);
    f.open_top = true;
    f.settings.buoyancy = Num::N(0.0);
    f.settings.turbulence = Num::N(0.0);
    f.settings.iterations = Num::N(300.0);
    let p = BTreeMap::new();
    let heat = 300.0;
    f.emitters.push(Emitter { at: [0.6, 0.6, 0.4], radius: 0.25, smoke: Num::N(0.0), heat: Num::N(heat), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.0)], water: Num::N(0.0), carry: 0.0, gas: None, amount: Num::N(1.0), flow: Num::N(0.0) });
    let cells = f.emitter_cells(&f.emitters[0].clone()).len() as f64;
    f.step(&p);
    let mac = f.mac.as_ref().unwrap();
    let out: f64 = (0..d.nx).flat_map(|i| (0..d.ny).map(move |j| (i, j))).map(|(i, j)| mac.w[idx(d.wd(), i, j, d.nz)] as f64).sum::<f64>() * (h * h) as f64 * dt as f64;
    let want = cells * (h as f64).powi(3) * ((AMBIENT_K as f64 + heat as f64) / AMBIENT_K as f64).ln();
    eprintln!("expansion: {out:.5} m³ out, {want:.5} m³ expected");
    assert!((out - want).abs() < 0.03 * want, "{out} vs {want}");
}

/// CO2 is 1.5 times as heavy as air and sinks; helium is 7 times lighter and rises. A layer of
/// CO2 on the floor stays put (no made-up currents from the density jump).
#[test]
fn heavy_gas_sinks_light_gas_rises_and_a_still_layer_stays_still() {
    let d = Dims { nx: 16, ny: 16, nz: 24 };
    let run_gas = |name: &str| {
        let mut f = gas_box(d, 0.05, 1.0 / 30.0);
        let slot = f.gas_slot(gas::find(name).unwrap());
        f.add_gas(slot, 1.0, None, |p| (p[0] - 0.4).powi(2) + (p[1] - 0.4).powi(2) + (p[2] - 0.6).powi(2) <= 0.15f32.powi(2));
        let kg0 = f.gas_kg(slot, None);
        let h0 = f.summary().into_iter().find(|m| m.0.ends_with("centre_height_m")).unwrap().1;
        run(&mut f, 30);
        let h1 = f.summary().into_iter().find(|m| m.0.ends_with("centre_height_m")).unwrap().1;
        let kg1 = f.gas_kg(slot, None);
        (h0, h1, kg0, kg1)
    };
    let (a, b, kg0, kg1) = run_gas("co2");
    eprintln!("co2: {a:.3} -> {b:.3} m, {kg0:.5} -> {kg1:.5} kg");
    assert!(b < a - 0.15, "CO2 sinks: {a} -> {b}");
    assert!((kg1 - kg0).abs() < 0.01 * kg0, "CO2 is kept: {kg0} -> {kg1} kg");
    let (a, b, _, _) = run_gas("helium");
    eprintln!("helium: {a:.3} -> {b:.3} m");
    assert!(b > a + 0.15, "helium rises: {a} -> {b}");

    // a still layer of CO2 on the floor
    let mut f = gas_box(d, 0.05, 1.0 / 30.0);
    let slot = f.gas_slot(gas::find("co2").unwrap());
    f.add_gas(slot, 1.0, None, |p| p[2] < 0.3);
    run(&mut f, 60);
    let fastest = f.mac.as_ref().unwrap().max_speed();
    assert!(fastest < 0.01, "a resting heavy layer stays at rest: {fastest} m/s");
}

/// A leak of CO2 at 1 litre a second: the gas in the room grows by its flow times its density.
#[test]
fn a_gas_leak_adds_the_mass_it_lets_out() {
    let d = Dims { nx: 20, ny: 20, nz: 16 };
    let mut f = gas_box(d, 0.1, 0.05);
    f.open_top = true;
    let slot = f.gas_slot(gas::find("co2").unwrap());
    f.emitters.push(Emitter { at: [1.0, 1.0, 0.25], radius: 0.12, smoke: Num::N(0.0), heat: Num::N(0.0), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.0)], water: Num::N(0.0), carry: 0.0, gas: Some(slot), amount: Num::N(1.0), flow: Num::N(1e-3) });
    run(&mut f, 100);
    let kg = f.gas_kg(slot, None);
    let rho_co2 = gas::P0 * gas::GASES[gas::find("co2").unwrap()].molar_mass / (gas::R * AMBIENT_K);
    let want = 1e-3 * f.seconds * rho_co2 as f64;
    eprintln!("leak: {kg:.5} kg in the room, {want:.5} kg let out");
    assert!((kg - want).abs() < 0.03 * want, "{kg} vs {want}");
    // and it lies on the floor
    let m = f.summary();
    let hgt = m.iter().find(|m| m.0 == "co2.centre_height_m").unwrap().1;
    assert!(hgt < 0.4, "the CO2 stays low: centre {hgt} m");
    assert!(f.solid.iter().all(|s| *s != SOLID));
}

/// The graphics card solves the same gas: a hot plume with a CO2 leak beside it, both ways.
#[test]
fn the_gpu_gas_solver_agrees_with_the_cpu_and_repeats() {
    if !gpu_smoke::available() {
        eprintln!("no graphics card: skipped");
        return;
    }
    let make = |gpu: bool| {
        let mut f = gas_box(Dims { nx: 20, ny: 16, nz: 28 }, 0.05, 1.0 / 30.0);
        f.use_gpu = gpu;
        f.open_top = true;
        f.settings.vorticity = Num::N(0.2);
        f.settings.cooling = Num::N(0.1);
        let co2 = f.gas_slot(gas::find("co2").unwrap());
        f.emitters.push(Emitter { at: [0.5, 0.4, 0.15], radius: 0.1, smoke: Num::N(3.0), heat: Num::N(300.0), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.3)], water: Num::N(0.0), carry: 0.0, gas: None, amount: Num::N(1.0), flow: Num::N(0.0) });
        f.emitters.push(Emitter { at: [0.7, 0.4, 0.1], radius: 0.08, smoke: Num::N(0.0), heat: Num::N(0.0), velocity: [Num::N(0.0), Num::N(0.0), Num::N(0.0)], water: Num::N(0.0), carry: 0.0, gas: Some(co2), amount: Num::N(1.0), flow: Num::N(2e-4) });
        f.add_solid(|p| p[2] > 0.8 && p[2] < 0.9 && p[0] < 0.6);
        run(&mut f, 45);
        f
    };
    let (c, g1, g2) = (make(false), make(true), make(true));
    let pick = |f: &Fluid, k: &str| f.summary().into_iter().find(|m| m.0 == k).map(|m| m.1).unwrap();
    for k in ["smoke.centre_height_m", "smoke.centre_x_m", "co2.kg", "co2.centre_height_m", "smoke.amount_m3"] {
        let (a, b) = (pick(&c, k), pick(&g1, k));
        eprintln!("{k}: cpu {a:.5} gpu {b:.5}");
        assert!((a - b).abs() <= 0.08 * a.abs().max(1e-3), "{k}: cpu {a} vs gpu {b}");
    }
    assert!(g1.divergence < 0.05, "the GPU keeps the gas incompressible: {}", g1.divergence);
    assert_eq!(g1.smoke, g2.smoke, "the GPU repeats exactly");
    assert_eq!(g1.mac.as_ref().unwrap().w, g2.mac.as_ref().unwrap().w);
}
#[test]
#[ignore]
fn bench_gas() {
    for gpu in [true, false] {
        let mut f = gas_box(Dims { nx: 80, ny: 40, nz: 40 }, 0.1, 1.0 / 30.0);
        f.use_gpu = gpu;
        f.open_top = true;
        f.settings.vorticity = Num::N(0.3);
        if std::env::var_os("NOEXP").is_some() { f.settings.expansion = Num::N(0.0); }
        f.emitters.push(Emitter { at: [3.4, 2.0, 0.45], radius: 0.25, smoke: Num::N(3.0), heat: Num::N(250.0), velocity: [Num::N(0.0), Num::N(0.0), Num::N(1.0)], water: Num::N(0.0), carry: 0.0, gas: None, amount: Num::N(1.0), flow: Num::N(0.0) });
        run(&mut f, 3);
        let n = if gpu { 20 } else { 3 };
        let t = std::time::Instant::now();
        run(&mut f, n);
        let step = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let t = std::time::Instant::now();
        let c = f.clone();
        let clone = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let _ = c.summary();
        let summ = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("gpu {gpu}: step {step:.1} ms, clone {clone:.1} ms, summary {summ:.1} ms");
    }
}