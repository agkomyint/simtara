//! Checks of the flood solver against exact solutions and laboratory measurements.

use super::*;

fn basin(nx: usize, ny: usize, cell: f64, dt: f64) -> Flood {
    let mut f = Flood::default();
    f.create(nx, ny, cell, dt);
    f.edges = [Edge::Wall; 4];
    f.manning = Num::N(0.0);
    f
}

fn run(f: &mut Flood, ticks: usize) {
    let p = BTreeMap::new();
    for _ in 0..ticks {
        f.step(&p);
    }
}

/// A lake over hills and islands, partly dry, must stay perfectly still (no made-up currents
/// from the slopes of the ground): the "well-balanced" property every flood model needs.
#[test]
fn a_lake_over_bumpy_ground_stays_still() {
    let mut f = basin(40, 40, 1.0, 1.0);
    f.shape_ground(|x, y, _| 0.3 * (x * 0.4).sin() * (y * 0.3).cos() + 0.4 * (-((x - 20.0).powi(2) + (y - 20.0).powi(2)) / 20.0).exp() + 0.002 * x);
    for c in 0..f.h.len() {
        f.h[c] = (0.35 - f.bed[c]).max(0.0);
    }
    assert!(f.h.iter().any(|h| *h == 0.0), "an island sticks out");
    run(&mut f, 30);
    let fastest = f.hu.iter().chain(&f.hv).fold(0.0f64, |m, x| m.max(x.abs()));
    let level = (0..f.h.len()).filter(|&c| f.h[c] > 0.0).map(|c| (f.h[c] + f.bed[c] - 0.35).abs()).fold(0.0, f64::max);
    assert!(fastest < 1e-12 && level < 1e-12, "still: momentum {fastest:e}, level off by {level:e}");
}

/// Exact solution of the dam break on a wet bed (Stoker 1957).
fn stoker(x: f64, x0: f64, t: f64, hl: f64, hr: f64, g: f64) -> f64 {
    let cl = (g * hl).sqrt();
    // the middle depth: rarefaction speed = shock speed (bisection)
    let f = |hm: f64| 2.0 * (cl - (g * hm).sqrt()) - (hm - hr) * (g * (hm + hr) / (2.0 * hm * hr)).sqrt();
    let (mut lo, mut hi) = (hr, hl);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 { lo = mid } else { hi = mid }
    }
    let hm = 0.5 * (lo + hi);
    let cm = (g * hm).sqrt();
    let um = 2.0 * (cl - cm);
    let s = hm * um / (hm - hr);
    let xi = (x - x0) / t;
    if xi < -cl {
        hl
    } else if xi < um - cm {
        (2.0 * cl - xi).powi(2) / (9.0 * g)
    } else if xi < s {
        hm
    } else {
        hr
    }
}

#[test]
fn a_dam_break_matches_the_exact_solutions_of_stoker_and_ritter() {
    let (n, dx, x0, t) = (800usize, 0.125f64, 50.0f64, 5.0f64);
    for hr in [0.1f64, 0.0] {
        let mut f = basin(n, 1, dx, t);
        for i in 0..n {
            f.h[i] = if (i as f64 + 0.5) * dx < x0 { 1.0 } else { hr };
        }
        let v0 = f.volume();
        run(&mut f, 1);
        let g = f.gravity;
        let err: f64 = (0..n).map(|i| (f.h[i] - stoker((i as f64 + 0.5) * dx, x0, t, 1.0, hr.max(1e-12), g)).abs()).sum::<f64>() / n as f64;
        eprintln!("dam break onto {hr} m: mean depth error {err:.4} m");
        assert!(err < 0.005, "mean depth error {err} m (h_l = 1 m)");
        assert!((f.volume() - v0).abs() < 1e-9 * v0, "no water made or lost");
        if hr == 0.0 {
            // Ritter: the front runs at 2 sqrt(g h)
            let front = (0..n).rev().find(|&i| f.h[i] > 1e-3).unwrap() as f64 * dx;
            let exact = x0 + 2.0 * (g * 1.0f64).sqrt() * t;
            eprintln!("front at {front:.1} m, exact {exact:.1} m");
            assert!(front > x0 + 0.85 * (exact - x0) && front < exact + 0.5, "front {front} vs {exact}");
        }
    }
}

/// A solitary wave running up a plane beach of slope 1:19.85 (Synolakis 1987, the standard
/// tsunami run-up benchmark, NOAA benchmark 1): H/d = 0.0185 climbs R/d = 2.831 sqrt(cot b) (H/d)^(5/4).
#[test]
fn a_solitary_wave_runs_up_a_beach_as_high_as_synolakis_measured() {
    let (d, hh, cot) = (1.0f64, 0.0185f64, 19.85f64);
    let dx = 0.05;
    let toe = 60.0;
    let n = ((toe + cot * d + 4.0) / dx) as usize;
    let mut f = basin(n, 1, dx, 0.5);
    f.edges[0] = Edge::Open;
    f.wet = 1e-4;
    f.shape_ground(|x, _, _| if x < toe { -d } else { -d + (x - toe) / cot });
    for c in 0..n {
        f.h[c] = (-f.bed[c]).max(0.0);
    }
    let gamma = (3.0 * hh / (4.0 * d)).sqrt();
    let l = (20.0f64).sqrt().acosh() / gamma;
    f.add_solitary(hh, toe - 0.5 * l, [1.0, 0.0], d);
    let mut highest = f64::MIN;
    let p = BTreeMap::new();
    for _ in 0..80 {
        f.step(&p);
        for c in 0..n {
            if f.h[c] > 1e-4 * d {
                highest = highest.max(f.bed[c] + f.h[c]);
            }
        }
    }
    let want = 2.831 * cot.sqrt() * hh.powf(1.25) * d;
    eprintln!("run-up {highest:.4} m, Synolakis {want:.4} m");
    assert!((highest - want).abs() < 0.12 * want, "run-up {highest} vs {want}");
}

/// Rain on a closed basin: every drop is kept.
#[test]
fn rain_fills_a_basin_by_exactly_what_falls() {
    let mut f = basin(30, 30, 2.0, 60.0);
    f.shape_ground(|x, y, _| 0.5 * (x / 7.0).sin() + 0.3 * (y / 5.0).cos());
    f.rain.push(Rain { rate: Num::N(100.0), region: None });
    run(&mut f, 60);
    // 100 mm an hour for an hour
    let want = 0.1 * f.extent()[0] * f.extent()[1];
    let got = f.volume();
    eprintln!("rain: {got:.3} m³, {want:.3} m³ fell");
    assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
}

/// A river down a uniform slope settles at the normal depth Manning's formula gives:
/// h = (n q / sqrt(S))^(3/5) for a wide channel.
#[test]
fn a_river_on_a_slope_flows_at_mannings_normal_depth() {
    let (n_cells, dx, slope, q, n) = (200usize, 10.0f64, 0.001f64, 1.0f64, 0.03f64);
    let mut f = basin(n_cells, 1, dx, 10.0);
    f.edges[1] = Edge::Open;
    f.manning = Num::N(n);
    f.shape_ground(|x, _, _| 2.0 - slope * x);
    f.rivers.push(River { at: [5.0, 5.0], radius: 4.0, flow: Num::N(q * dx), velocity: [Num::N(0.0), Num::N(0.0)] });
    run(&mut f, 500);
    let want = (n * q / slope.sqrt()).powf(0.6);
    let got = f.h[n_cells / 2];
    eprintln!("normal depth: {got:.4} m, Manning {want:.4} m");
    assert!((got - want).abs() < 0.03 * want, "{got} vs {want}");
}

/// A dam whose crest follows a control: lowered, the reservoir empties into the valley.
#[test]
fn lowering_a_gate_lets_the_reservoir_out() {
    let mut f = basin(60, 20, 5.0, 2.0);
    f.edges[1] = Edge::Open;
    f.shape_ground(|x, _, _| 10.0 - 0.02 * x);
    for c in 0..f.h.len() {
        if f.centre(c)[0] < 100.0 {
            f.h[c] = 19.0 - f.bed[c];
        }
    }
    let region = [100.0, 0.0, 110.0, 100.0];
    let under = (0..f.bed.len()).filter(|&c| f.inside(c, Some(region))).map(|c| (c, f.bed[c])).collect();
    f.gates.push(Gate { region, height: Num::P("dam".into()), under });
    let mut p = BTreeMap::new();
    p.insert("dam".to_string(), Param { value: 20.0, min: 0.0, max: 30.0, about: String::new(), owner: String::new() });
    for _ in 0..30 {
        f.step(&p);
    }
    let downstream = |f: &Flood| (0..f.h.len()).filter(|&c| f.centre(c)[0] > 150.0).map(|c| f.h[c]).sum::<f64>();
    assert!(downstream(&f) < 1e-9, "the dam holds");
    p.insert("dam".to_string(), Param { value: 0.0, min: 0.0, max: 30.0, about: String::new(), owner: String::new() });
    for _ in 0..30 {
        f.step(&p);
    }
    assert!(downstream(&f) > 10.0, "the water rushes out: {}", downstream(&f));
}


/// Wind blowing along a closed lake piles the water up at the far end until the slope of the
/// surface holds it back: g h d(eta)/dx = tau / rho (the "wind set-up" of storm surges).
#[test]
fn wind_piles_water_up_by_its_stress() {
    let (n, dx, depth, u10) = (100usize, 100.0f64, 5.0f64, 20.0f64);
    let mut f = basin(n, 1, dx, 60.0);
    f.manning = Num::N(0.05);
    f.wind_speed = Num::N(u10);
    f.wind_dir = Num::N(0.0);
    for c in 0..n {
        f.h[c] = depth;
    }
    run(&mut f, 1440);
    let tau = RHO_AIR * wind_drag(u10) * u10 * u10;
    let want: f64 = (0..n - 1).map(|c| tau * dx / (f.density * f.gravity * 0.5 * (f.h[c] + f.h[c + 1]))).sum();
    let got = f.h[n - 1] - f.h[0];
    eprintln!("wind set-up: {got:.4} m, expected {want:.4} m (stress {tau:.3} Pa)");
    assert!((got - want).abs() < 0.03 * want, "{got} vs {want}");
    assert!(f.summary().iter().any(|m| m.0 == "wind.stress_pa" && (m.1 - tau).abs() < 1e-9));
}