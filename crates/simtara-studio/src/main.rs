//! SimTara Studio: launcher (new + saved universes) and a live universe with a control panel,
//! saving, and experiments.

use eframe::egui;
use simtara_core::{Backend, Config, Simulation, Snapshot};
use simtara_gpu::GpuModule;
use simtara_particles::{simulation, Boundary, ParticleConfig, ParticleParams, Particles};
use simtara_orbits::Bodies;
use simtara_store::{Experiment, Store, UniverseMeta, TEMPLATE_ORBITS, TEMPLATE_PARTICLES};
use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver};

/// Shown in the script box of a new orbits universe. Does nothing until the comment is removed.
const DEFAULT_SCRIPT: &str = "-- Runs once per tick. One tick = 6 hours, so 1461 ticks = one year.\n-- Try: remove the Moon after a year (delete the two dashes in front of 'if').\nfunction on_tick(t)\n  -- if t == 1461 then remove_body(\"Moon\") end\nend\n";

/// Starting text for the changed run of an orbits experiment.
const EXPERIMENT_SCRIPT: &str = "-- The changed run uses this script instead of the universe's own.
-- t is the universe's tick, so remember when this run began.
-- hide_body: the Moon keeps moving but pulls on nothing. show_body brings it back.
function on_tick(t)
  if not began then
    began = t
    hide_body(\"Moon\")
  end
  -- if t - began == 730 then show_body(\"Moon\") end
end
";

const VIEW: usize = 900;

struct Launch {
    template: &'static str,
    count: usize,
    kinds: usize,
    seed: u64,
    spread: f32,
    base: f32,
    use_gpu: bool,
}

enum JobMsg {
    Progress(Vec<simtara_store::Series>),
    Done(Result<(Experiment, Snapshot), String>),
}
type ExperimentJob = Receiver<JobMsg>;

struct Running {
    sim: Simulation,
    initial: ParticleConfig,
    seed: u64,
    gpu: GpuModule,
    tex: Option<egui::TextureHandle>,
    ms: f32,
    history: Vec<[f64; 2]>,
    // saving
    saved_as: Option<String>,
    name_input: String,
    status: String,
    // experiments
    experiments: Vec<Experiment>,
    selected: Option<usize>,
    job: Option<ExperimentJob>,
    live: Option<Experiment>,
    exp_param: usize,
    exp_alt: f64,
    exp_ticks: u64,
    exp_metric: String,
    exp_script: String,
    exp_name: String,
    // which template this universe came from
    template: &'static str,
    base_config: Config,
    // orbits view
    follow: String,
    zoom_km_per_px: f64,
    trails: Vec<(String, VecDeque<[f64; 2]>)>,
    trail_tick: u64,
    script_text: String,
}

enum Screen {
    Launcher,
    Universe(Box<Running>),
}

struct Studio {
    launch: Launch,
    screen: Screen,
    gpu_desc: String,
    gpu_ok: bool,
    store: Store,
    saved: Vec<UniverseMeta>,
    confirm_delete: Option<String>,
    notice: String,
}

fn hue(k: u8, kinds: usize) -> [u8; 3] {
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
    let c = |v: f32| (60.0 + 195.0 * v) as u8;
    [c(r), c(g), c(b)]
}

fn render(sim: &Simulation) -> egui::ColorImage {
    let mut img = egui::ColorImage::filled([VIEW, VIEW], egui::Color32::from_rgb(7, 8, 12));
    let (Some(p), Some(prm)) = (sim.world().get::<Particles>(), sim.world().get::<ParticleParams>()) else { return img };
    let (sx, sy) = (VIEW as f32 / prm.width, VIEW as f32 / prm.height);
    for i in 0..p.len() {
        let [r, g, b] = hue(p.kind[i], prm.kinds);
        let (x, y) = ((p.x[i] * sx) as isize, (p.y[i] * sy) as isize);
        for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (px, py) = (x + ox, y + oy);
            if px >= 0 && py >= 0 && (px as usize) < VIEW && (py as usize) < VIEW {
                img.pixels[py as usize * VIEW + px as usize] = egui::Color32::from_rgb(r, g, b);
            }
        }
    }
    img
}

/// "just now", "5 min ago", "3 h ago", "2 d ago"
fn age(unix: u64) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let s = now.saturating_sub(unix);
    match s {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", s / 60),
        3600..=86399 => format!("{} h ago", s / 3600),
        _ => format!("{} d ago", s / 86400),
    }
}

impl Running {
    fn fresh(sim: Simulation, initial: ParticleConfig, seed: u64, gpu: GpuModule, name: String, template: &'static str, base_config: Config) -> Running {
        let orbits = template == TEMPLATE_ORBITS;
        let script_text = simtara_script::source(sim.world());
        Running {
            exp_metric: simtara_store::default_metric(template).to_string(),
            exp_script: EXPERIMENT_SCRIPT.to_string(),
            exp_name: "Remove the Moon".to_string(),
            template,
            base_config,
            follow: if orbits { "Sun".into() } else { String::new() },
            zoom_km_per_px: 4.0e5,
            trails: Vec::new(),
            trail_tick: 0,
            script_text,
            sim,
            initial,
            seed,
            gpu,
            tex: None,
            ms: 0.0,
            history: Vec::new(),
            saved_as: None,
            name_input: name,
            status: String::new(),
            experiments: Vec::new(),
            selected: None,
            job: None,
            live: None,
            exp_param: 0,
            exp_alt: 0.0,
            exp_ticks: if orbits { 1461 } else { 2000 },
        }
    }

    fn start(l: &Launch, gpu: GpuModule, name: String) -> Running {
        if l.template == TEMPLATE_ORBITS {
            let config = simtara_store::new_orbits_config(l.seed, DEFAULT_SCRIPT);
            let mut sim = simtara_store::build_simulation(&config);
            sim.set_speed(3.0);
            sim.play();
            return Running::fresh(sim, ParticleConfig::default(), l.seed, gpu, name, TEMPLATE_ORBITS, config);
        }
        let mut pc = ParticleConfig::default();
        pc.count = l.count;
        pc.params.kinds = l.kinds;
        pc.spread = l.spread;
        pc.base_attraction = l.base;
        let mut sim = simulation(Config { seed: l.seed, ..Config::default() }, &pc);
        sim.add_module(gpu.clone());
        sim.compute_mut().set_requested(if l.use_gpu { Backend::Gpu } else { Backend::Cpu });
        sim.play();
        Running::fresh(sim, pc, l.seed, gpu, name, TEMPLATE_PARTICLES, Config::default())
    }

    /// Reopen a saved universe at the tick it was saved.
    fn open(store: &Store, name: &str, gpu: GpuModule, use_gpu: bool) -> Result<Running, String> {
        let (meta, snap) = store.load(name)?;
        let template = simtara_store::template_of(&snap.config);
        let mut sim = simtara_store::build_simulation(&snap.config);
        if template == TEMPLATE_PARTICLES {
            sim.add_module(gpu.clone());
        }
        sim.restore(&snap)?;
        sim.compute_mut().set_requested(if use_gpu && gpu.available() && template == TEMPLATE_PARTICLES { Backend::Gpu } else { Backend::Cpu });
        if template == TEMPLATE_ORBITS {
            sim.set_speed(3.0);
        }
        sim.play();
        let mut run = Running::fresh(sim, simtara_store::particle_config(&snap), snap.config.seed, gpu, meta.name.clone(), template, snap.config.clone());
        run.saved_as = Some(meta.name.clone());
        run.experiments = store.list_experiments(&meta.name);
        run.selected = run.experiments.len().checked_sub(1);
        run.status = format!("opened '{}' at tick {}", meta.name, meta.tick);
        Ok(run)
    }

    fn save(&mut self, store: &Store) {
        match store.save(&self.name_input, &self.sim, self.template) {
            Ok(meta) => {
                self.status = format!("saved '{}' at tick {}", meta.name, meta.tick);
                self.name_input = meta.name.clone();
                self.saved_as = Some(meta.name);
            }
            Err(e) => self.status = format!("could not save: {e}"),
        }
    }

    /// Back to tick 0 of the universe as first created (keeping the script as edited).
    fn rebuild_orbits(&mut self) {
        let mut cfg = self.base_config.clone();
        cfg.set_module(simtara_script::MODULE, &simtara_script::ScriptConfig { source: self.script_text.clone() });
        let speed = self.sim.speed();
        let mut sim = simtara_store::build_simulation(&cfg);
        sim.set_speed(speed);
        sim.play();
        self.sim = sim;
        self.trails.clear();
        self.history.clear();
    }

    /// Keep a short path behind each body, measured from the body being followed.
    fn update_trails(&mut self) {
        let tick = self.sim.world().clock.tick;
        let Some(b) = self.sim.world().get::<Bodies>() else { return };
        if tick == self.trail_tick && !self.trails.is_empty() {
            return;
        }
        let origin = b.find(&self.follow).map_or([0.0, 0.0], |f| f.pos);
        for body in &b.list {
            if !self.trails.iter().any(|t| t.0 == body.name) {
                self.trails.push((body.name.clone(), VecDeque::new()));
            }
            if let Some(t) = self.trails.iter_mut().find(|t| t.0 == body.name) {
                t.1.push_back([body.pos[0] - origin[0], body.pos[1] - origin[1]]);
                if t.1.len() > 900 {
                    t.1.pop_front();
                }
            }
        }
        self.trails.retain(|t| b.find(&t.0).is_some());
        self.trail_tick = tick;
    }

    fn rebuild(&mut self, count: Option<usize>, new_seed: Option<u64>) {
        if self.template == TEMPLATE_ORBITS {
            return self.rebuild_orbits();
        }
        let requested = self.sim.compute_requested();
        let speed = self.sim.speed();
        let mut pc = self.initial.clone();
        if let Some(c) = count {
            pc.count = c;
        }
        if let Some(p) = self.sim.world().get::<ParticleParams>() {
            pc.params = ParticleParams { attraction: if new_seed.is_some() { Vec::new() } else { p.attraction.clone() }, ..p.clone() };
        }
        if let Some(s) = new_seed {
            self.seed = s;
        }
        let mut sim = simulation(Config { seed: self.seed, ..Config::default() }, &pc);
        sim.add_module(self.gpu.clone());
        sim.compute_mut().set_requested(requested);
        sim.set_speed(speed);
        sim.play();
        self.sim = sim;
        self.initial = pc;
        self.history.clear();
    }

    /// Save the universe, then run "original vs changed" on a background thread.
    fn run_experiment(&mut self, store: &Store) {
        let Some(name) = self.saved_as.clone() else { return };
        self.save(store);
        let field = simtara_store::PARAMETERS[self.exp_param].0;
        let result = (|| -> Result<_, String> {
            let id = store.next_experiment_id(&name)?;
            let mut spec = if self.template == TEMPLATE_ORBITS {
                simtara_store::new_script_spec(id, &self.exp_name, &self.exp_script, &self.exp_metric, self.exp_ticks)
            } else {
                simtara_store::new_spec(id, &self.sim, field, self.exp_alt, self.exp_ticks)?
            };
            spec.metric = self.exp_metric.clone();
            let start = self.sim.snapshot()?;
            Ok((spec, start))
        })();
        match result {
            Ok((spec, start)) => {
                let spec_for_live = spec.clone();
                let (tx, rx) = channel();
                std::thread::spawn(move || {
                    let txp = tx.clone();
                    let out = simtara_store::run_experiment_live(&start, &spec, &mut |s| {
                        let _ = txp.send(JobMsg::Progress(s.to_vec()));
                    })
                    .map(|result| (Experiment { spec, result }, start));
                    let _ = tx.send(JobMsg::Done(out));
                });
                self.live = Some(Experiment {
                    spec: spec_for_live,
                    result: simtara_store::ExperimentResult { schema_version: 0, series: Vec::new(), record: simtara_store::RunRecord { engine_version: String::new(), seed: 0, backend: "cpu".into(), start_tick: 0, millis: 0 } },
                });
                self.selected = None;
                self.job = Some(rx);
                self.status = "experiment running...".into();
            }
            Err(e) => self.status = format!("could not start experiment: {e}"),
        }
    }

    /// Collect a finished experiment (if any) and write it to disk.
    fn poll_experiment(&mut self, store: &Store) {
        let Some(rx) = &self.job else { return };
        let mut finished = None;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                JobMsg::Progress(series) => {
                    if let Some(l) = self.live.as_mut() {
                        l.result.series = series;
                    }
                }
                JobMsg::Done(out) => {
                    finished = Some(out);
                    break;
                }
            }
        }
        let Some(out) = finished else { return };
        self.job = None;
        self.live = None;
        match out {
            Ok((exp, start)) => {
                let name = self.saved_as.clone().unwrap_or_default();
                match store.save_experiment(&name, &exp, &start) {
                    Ok(()) => self.status = format!("experiment finished in {} ms and saved", exp.result.record.millis),
                    Err(e) => self.status = format!("experiment finished but could not be saved: {e}"),
                }
                self.experiments.push(exp);
                self.selected = Some(self.experiments.len() - 1);
            }
            Err(e) => self.status = format!("experiment failed: {e}"),
        }
    }
}

/// Two-line chart: original vs changed.
fn draw_chart(ui: &mut egui::Ui, exp: &Experiment) {
    let colors = [egui::Color32::from_rgb(120, 200, 255), egui::Color32::from_rgb(255, 170, 70)];
    let label = simtara_store::PARAMETERS.iter().find(|p| p.0 == exp.spec.parameter).map_or(exp.spec.parameter.as_str(), |p| p.1);
    ui.horizontal(|ui| {
        for (s, c) in exp.result.series.iter().zip(colors) {
            if exp.spec.alt_script.is_some() {
                ui.colored_label(c, format!("● {}", if s.label == "changed" { exp.spec.name.as_str() } else { "original" }));
            } else {
                ui.colored_label(c, format!("● {} ({label} = {:.2})", s.label, s.value));
            }
        }
    });
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 190.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 3.0, egui::Color32::from_gray(18));
    let all = exp.result.series.iter().flat_map(|s| s.points.iter());
    let (mut x0, mut x1, mut lo, mut hi) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in all {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        lo = lo.min(p[1]);
        hi = hi.max(p[1]);
    }
    if x1 > x0 && hi >= lo {
        if hi - lo < 1e-9 {
            hi = lo + 1.0;
        }
        let plot = rect.shrink2(egui::vec2(56.0, 12.0));
        for (k, (s, c)) in exp.result.series.iter().zip(colors).enumerate() {
            let width = if k == 0 { 4.0 } else { 1.5 };
            let pts: Vec<egui::Pos2> = s
                .points
                .iter()
                .map(|p| egui::pos2(plot.left() + plot.width() * ((p[0] - x0) / (x1 - x0)) as f32, plot.bottom() - plot.height() * ((p[1] - lo) / (hi - lo)) as f32))
                .collect();
            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(width, c)));
        }
        let font = egui::FontId::proportional(11.0);
        let grey = egui::Color32::GRAY;
        ui.painter().text(egui::pos2(rect.left() + 4.0, plot.top()), egui::Align2::LEFT_TOP, fmt_num(hi), font.clone(), grey);
        ui.painter().text(egui::pos2(rect.left() + 4.0, plot.bottom()), egui::Align2::LEFT_BOTTOM, fmt_num(lo), font.clone(), grey);
        ui.painter().text(egui::pos2(plot.left(), rect.bottom() - 1.0), egui::Align2::LEFT_BOTTOM, format!("tick {x0:.0}"), font.clone(), grey);
        ui.painter().text(egui::pos2(plot.right(), rect.bottom() - 1.0), egui::Align2::RIGHT_BOTTOM, format!("tick {x1:.0}"), font, grey);
    }
    // The two runs are often nearly identical next to the size of the whole value, so show their difference on its own scale.
    if let [orig, chg] = &exp.result.series[..] {
        let diff: Vec<[f64; 2]> = orig.points.iter().zip(&chg.points).map(|(o, c)| [o[0], c[1] - o[1]]).collect();
        let maxd = diff.iter().fold(0.0_f64, |m, d| m.max(d[1].abs()));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 56.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 3.0, egui::Color32::from_gray(18));
        let plot = rect.shrink2(egui::vec2(56.0, 6.0));
        let mid = plot.center().y;
        ui.painter().line_segment([egui::pos2(plot.left(), mid), egui::pos2(plot.right(), mid)], egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
        if diff.len() > 1 && maxd > 0.0 {
            let n = (diff.len() - 1) as f32;
            let pts: Vec<egui::Pos2> = diff.iter().enumerate().map(|(i, d)| egui::pos2(plot.left() + plot.width() * i as f32 / n, mid - (plot.height() / 2.0) * (d[1] / maxd) as f32)).collect();
            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 120, 120))));
        }
        let grey = egui::Color32::GRAY;
        ui.painter().text(egui::pos2(rect.left() + 4.0, rect.top() + 4.0), egui::Align2::LEFT_TOP, format!("+{}", fmt_num(maxd)), egui::FontId::proportional(11.0), grey);
        if maxd == 0.0 {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "no difference between the two runs: does the script actually do anything in this run?", egui::FontId::proportional(12.0), egui::Color32::LIGHT_RED);
        }
        ui.weak("difference: changed minus original");
    }
    let r = &exp.result.record;
    ui.weak(format!("metric {} · engine v{} · seed {} · {} · {} ms", exp.spec.metric, r.engine_version, r.seed, r.backend, r.millis));
}

/// Readable numbers from tiny to astronomical.
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 1e6 || v.abs() < 0.01 {
        format!("{v:.6e}")
    } else if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// Left-panel controls for the orbits template: view and script.
fn orbit_controls(ui: &mut egui::Ui, run: &mut Running) {
    ui.label("View (scroll over the picture to zoom)");
    let names: Vec<String> = run.sim.world().get::<Bodies>().map(|b| b.list.iter().map(|x| x.name.clone()).collect()).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("follow");
        egui::ComboBox::from_id_salt("follow").selected_text(run.follow.clone()).show_ui(ui, |ui| {
            for n in &names {
                if ui.selectable_value(&mut run.follow, n.clone(), n).changed() {
                    run.trails.clear();
                }
            }
        });
    });
    ui.add(egui::Slider::new(&mut run.zoom_km_per_px, 50.0..=2.0e6).logarithmic(true).text("km per pixel"));
    ui.horizontal(|ui| {
        if ui.button("Sun view").clicked() {
            run.follow = "Sun".into();
            run.zoom_km_per_px = 4.0e5;
            run.trails.clear();
        }
        if ui.button("Earth and Moon").clicked() {
            run.follow = "Earth".into();
            run.zoom_km_per_px = 1500.0;
            run.trails.clear();
        }
    });
    ui.separator();
    ui.strong("Script (Lua)");
    ui.weak("on_tick(t) runs every tick. Commands: hide_body(name), show_body(name), remove_body(name), set_mass(name, kg), body(name), log(...)");
    ui.add(egui::TextEdit::multiline(&mut run.script_text).code_editor().desired_rows(8).desired_width(f32::INFINITY));
    let applied = simtara_script::source(run.sim.world());
    ui.horizontal(|ui| {
        if ui.button("Apply script").clicked() {
            if let Err(e) = simtara_script::set_source(run.sim.world_mut(), &run.script_text) {
                run.status = e;
            }
        }
        if run.script_text != applied && ui.small_button("undo edits").clicked() {
            run.script_text = applied.clone();
        }
        if run.script_text != applied {
            ui.weak("not applied yet");
        }
    });
    let (err, log) = simtara_script::status(run.sim.world());
    if let Some(e) = err {
        ui.colored_label(egui::Color32::LIGHT_RED, e);
    }
    for line in log.iter().rev().take(5) {
        ui.weak(line);
    }
}

/// The orbits picture: bodies, their paths, and a scale you can zoom.
fn draw_orbits(ui: &mut egui::Ui, run: &mut Running) {
    let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            run.zoom_km_per_px = (run.zoom_km_per_px * (-scroll as f64 * 0.003).exp()).clamp(50.0, 2.0e6);
        }
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(7, 8, 12));
    let scale = 1.0 / (run.zoom_km_per_px * 1000.0); // pixels per metre
    let c = rect.center();
    let Some(b) = run.sim.world().get::<Bodies>() else { return };
    let origin = b.find(&run.follow).map_or([0.0, 0.0], |f| f.pos);
    let to_screen = |rel: [f64; 2]| egui::pos2(c.x + (rel[0] * scale) as f32, c.y - (rel[1] * scale) as f32);
    let color = |rgb: [u8; 3]| egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    for (name, trail) in &run.trails {
        let col = b.find(name).map_or(egui::Color32::GRAY, |x| color(x.color)).gamma_multiply(0.45);
        let pts: Vec<egui::Pos2> = trail.iter().map(|p| to_screen(*p)).collect();
        if pts.len() > 1 {
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, col)));
        }
    }
    for body in &b.list {
        let p = to_screen([body.pos[0] - origin[0], body.pos[1] - origin[1]]);
        let r = ((body.radius * scale) as f32).clamp(3.0, 400.0);
        if body.hidden {
            painter.circle_stroke(p, r, egui::Stroke::new(1.0, color(body.color).gamma_multiply(0.5)));
        } else {
            painter.circle_filled(p, r, color(body.color));
        }
        painter.text(p + egui::vec2(r + 4.0, -r - 2.0), egui::Align2::LEFT_BOTTOM, if body.hidden { format!("{} (hidden)", body.name) } else { body.name.clone() }, egui::FontId::proportional(13.0), egui::Color32::from_gray(190));
    }
    painter.text(
        rect.left_bottom() + egui::vec2(10.0, -10.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{:.0} km per pixel · following {}", run.zoom_km_per_px, run.follow),
        egui::FontId::proportional(12.0),
        egui::Color32::from_gray(120),
    );
}

impl Studio {
    fn refresh(&mut self) {
        self.saved = self.store.list();
    }

    /// Save the open universe if it has been saved before (used when leaving it).
    fn autosave(&mut self) {
        if let Screen::Universe(run) = &mut self.screen {
            if run.saved_as.is_some() {
                run.save(&self.store);
            }
        }
    }
}

impl eframe::App for Studio {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        match &mut self.screen {
            Screen::Launcher => {
                let mut go = false;
                let mut open: Option<String> = None;
                let mut delete: Option<String> = None;
                let accent = egui::Color32::from_rgb(90, 150, 255);
                egui::Panel::left("saved_universes").min_size(320.0).show(ui, |ui| {
                    ui.add_space(14.0);
                    ui.label(egui::RichText::new("SimTara").size(28.0).strong());
                    ui.weak("Build a universe. Change one thing. See what happens.");
                    ui.add_space(14.0);
                    ui.label(egui::RichText::new("My universes").size(16.0).strong());
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if self.saved.is_empty() {
                            ui.weak("Nothing saved yet. Launch a universe, then press Save.");
                        }
                        for m in &self.saved {
                            egui::Frame::group(ui.style()).inner_margin(10.0).show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(egui::RichText::new(&m.name).strong().size(15.0));
                                ui.weak(format!("{} · tick {} · {}", m.summary, m.tick, age(m.updated)));
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    if ui.add(egui::Button::new("Open").fill(accent.gamma_multiply(0.5))).clicked() {
                                        open = Some(m.name.clone());
                                    }
                                    if self.confirm_delete.as_deref() == Some(m.name.as_str()) {
                                        ui.colored_label(egui::Color32::LIGHT_RED, "Delete?");
                                        if ui.small_button("Yes").clicked() {
                                            delete = Some(m.name.clone());
                                        }
                                        if ui.small_button("No").clicked() {
                                            self.confirm_delete = None;
                                        }
                                    } else if ui.small_button("Delete").clicked() {
                                        self.confirm_delete = Some(m.name.clone());
                                    }
                                });
                            });
                            ui.add_space(4.0);
                        }
                        if !self.notice.is_empty() {
                            ui.colored_label(egui::Color32::LIGHT_RED, &self.notice);
                        }
                        ui.add_space(8.0);
                        ui.weak(format!("Saved in {}", self.store.root().display()));
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add_space(14.0);
                        ui.label(egui::RichText::new("New universe").size(22.0).strong());
                        ui.weak("1 · Pick a template");
                        ui.add_space(8.0);
                        let cards = [
                            (TEMPLATE_PARTICLES, "Particle life", "Thousands of coloured particles that attract and repel. Watch life-like patterns appear. Runs on CPU or GPU."),
                            (TEMPLATE_ORBITS, "Sun, Earth & Moon", "Real gravity with real masses and distances. Script events in Lua, like making the Moon vanish."),
                        ];
                        ui.columns(2, |cols| {
                            for (i, (id, title, text)) in cards.iter().enumerate() {
                                let selected = self.launch.template == *id;
                                let stroke = if selected { egui::Stroke::new(2.0, accent) } else { egui::Stroke::new(1.0, egui::Color32::from_gray(70)) };
                                let r = egui::Frame::new().stroke(stroke).corner_radius(8.0).inner_margin(14.0).fill(if selected { accent.gamma_multiply(0.12) } else { egui::Color32::TRANSPARENT }).show(&mut cols[i], |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.set_min_height(90.0);
                                    ui.label(egui::RichText::new(*title).strong().size(16.0));
                                    ui.add_space(4.0);
                                    ui.label(*text);
                                });
                                let click = cols[i].interact(r.response.rect, egui::Id::new(("card", i)), egui::Sense::click());
                                if click.clicked() {
                                    self.launch.template = id;
                                }
                                if click.hovered() {
                                    cols[i].ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                            }
                        });
                        ui.add_space(18.0);
                        ui.weak("2 · Starting conditions");
                        ui.add_space(8.0);
                        let l = &mut self.launch;
                        egui::Frame::group(ui.style()).inner_margin(14.0).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            if l.template == TEMPLATE_ORBITS {
                                ui.label("Sun, Earth and Moon start in their real orbits. One tick is 6 hours, so 1461 ticks is one year.");
                                ui.add_space(4.0);
                                ui.weak("Once inside, write a Lua script and run an experiment to compare two versions.");
                            } else {
                                egui::Grid::new("launch").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                                    ui.label("Particles");
                                    ui.add(egui::Slider::new(&mut l.count, 500..=100_000).logarithmic(true));
                                    ui.end_row();
                                    ui.label("Kinds (colours)");
                                    ui.add(egui::Slider::new(&mut l.kinds, 2..=8));
                                    ui.end_row();
                                    ui.label("Seed");
                                    ui.add(egui::DragValue::new(&mut l.seed));
                                    ui.end_row();
                                    ui.label("Spread");
                                    ui.add(egui::Slider::new(&mut l.spread, 0.0..=2.0));
                                    ui.end_row();
                                    ui.label("Base attraction");
                                    ui.add(egui::Slider::new(&mut l.base, -1.0..=1.0));
                                    ui.end_row();
                                    ui.label("Compute");
                                    ui.horizontal(|ui| {
                                        ui.add_enabled(self.gpu_ok, egui::Checkbox::new(&mut l.use_gpu, "GPU"));
                                        ui.weak(&self.gpu_desc);
                                    });
                                    ui.end_row();
                                });
                            }
                        });
                        ui.add_space(20.0);
                        if ui.add(egui::Button::new(egui::RichText::new("▶  Launch universe").size(17.0).strong()).fill(accent.gamma_multiply(0.6)).min_size(egui::vec2(240.0, 44.0))).clicked() {
                            go = true;
                        }
                    });
                });
                if let Some(name) = delete {
                    self.notice = match self.store.delete(&name) {
                        Ok(()) => String::new(),
                        Err(e) => format!("could not delete: {e}"),
                    };
                    self.confirm_delete = None;
                    self.refresh();
                }
                if let Some(name) = open {
                    match Running::open(&self.store, &name, GpuModule::new(), self.launch.use_gpu) {
                        Ok(run) => {
                            self.notice.clear();
                            self.screen = Screen::Universe(Box::new(run));
                        }
                        Err(e) => self.notice = format!("could not open '{name}': {e}"),
                    }
                } else if go {
                    self.notice.clear();
                    let name = self.store.suggest_name();
                    self.screen = Screen::Universe(Box::new(Running::start(&self.launch, GpuModule::new(), name)));
                }
            }
            Screen::Universe(run) => {
                let mut back = false;
                let mut do_save = false;
                let mut do_experiment = false;
                let mut branch: Option<(String, String, String)> = None;
                run.poll_experiment(&self.store);

                egui::Panel::left("controls").min_size(300.0).show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Controls");
                        ui.horizontal(|ui| {
                            ui.label("name");
                            ui.add(egui::TextEdit::singleline(&mut run.name_input).desired_width(130.0));
                            if ui.button("💾 Save").clicked() {
                                do_save = true;
                            }
                        });
                        if !run.status.is_empty() {
                            ui.weak(&run.status);
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            let playing = run.sim.run_state() == simtara_core::RunState::Running;
                            if ui.button(if playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                                run.sim.toggle();
                            }
                            if ui.button("↺ Reset").clicked() {
                                run.rebuild(None, None);
                            }
                            if run.template == TEMPLATE_PARTICLES && ui.button("🎲 New seed").clicked() {
                                let s = run.seed.wrapping_add(1);
                                run.rebuild(None, Some(s));
                            }
                        });
                        let mut speed = run.sim.speed() as f32;
                        if ui.add(egui::Slider::new(&mut speed, 0.1..=10.0).logarithmic(true).text("speed")).changed() {
                            run.sim.set_speed(speed as f64);
                        }
                        ui.separator();
                        if run.template == TEMPLATE_ORBITS {
                            orbit_controls(ui, run);
                        } else {
                        ui.label("Laws (live)");
                        if let Some(p) = run.sim.world_mut().get_mut::<ParticleParams>() {
                            ui.add(egui::Slider::new(&mut p.gravity, 0.0..=20.0).text("gravity"));
                            ui.add(egui::Slider::new(&mut p.friction, 0.0..=10.0).text("friction"));
                            ui.add(egui::Slider::new(&mut p.rmax, 20.0..=200.0).text("range"));
                            ui.add(egui::Slider::new(&mut p.repel_core, 0.05..=0.8).text("core repel"));
                            ui.add(egui::Slider::new(&mut p.coulomb, -2.0..=2.0).text("charge"));
                            ui.horizontal(|ui| {
                                ui.label("edges");
                                ui.selectable_value(&mut p.boundary, Boundary::Bounce, "bounce");
                                ui.selectable_value(&mut p.boundary, Boundary::Wrap, "wrap");
                            });
                        }
                        ui.separator();
                        let count = run.sim.world().get::<Particles>().map_or(0, |p| p.len());
                        let mut c = count;
                        if ui.add(egui::Slider::new(&mut c, 500..=100_000).logarithmic(true).text("particles")).drag_stopped() {
                            run.rebuild(Some(c), None);
                        }
                        let gpu_ok = run.gpu.available();
                        ui.horizontal(|ui| {
                            ui.label("compute");
                            let cur = run.sim.compute_requested();
                            if ui.selectable_label(cur == Backend::Cpu, "CPU").clicked() {
                                run.sim.compute_mut().set_requested(Backend::Cpu);
                            }
                            if ui.add_enabled_ui(gpu_ok, |ui| ui.selectable_label(cur == Backend::Gpu, "GPU")).inner.clicked() {
                                run.sim.compute_mut().set_requested(Backend::Gpu);
                            }
                        });
                        }
                        ui.separator();
                        ui.label("Measurements");
                        let m = run.sim.collect_metrics();
                        if run.template == TEMPLATE_ORBITS {
                            let tick = run.sim.world().clock.tick;
                            ui.label(format!("tick {tick}   (day {:.0}, year {:.2})   {:.2} ms/frame", tick as f64 * 0.25, tick as f64 / 1461.0, run.ms));
                        } else {
                            ui.label(format!("tick {}   backend {:?}   {:.2} ms/frame", run.sim.world().clock.tick, run.sim.backend(), run.ms));
                        }
                        for (k, v) in &m {
                            ui.label(format!("{k}: {}", fmt_num(*v)));
                        }
                        if let Some(ke) = m.get("particles.kinetic_energy") {
                            let t = run.sim.world().clock.tick as f64;
                            if run.history.last().is_none_or(|h| t > h[0]) {
                                run.history.push([t, *ke]);
                                if run.history.len() > 400 {
                                    run.history.remove(0);
                                }
                            }
                            let (lo, hi) = run.history.iter().fold((f64::MAX, f64::MIN), |a, h| (a.0.min(h[1]), a.1.max(h[1])));
                            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 70.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 3.0, egui::Color32::from_gray(20));
                            let n = run.history.len();
                            if n > 1 && hi > lo {
                                let pts: Vec<egui::Pos2> = run
                                    .history
                                    .iter()
                                    .enumerate()
                                    .map(|(i, h)| {
                                        egui::pos2(
                                            rect.left() + rect.width() * i as f32 / (n - 1) as f32,
                                            rect.bottom() - 4.0 - (rect.height() - 8.0) * ((h[1] - lo) / (hi - lo)) as f32,
                                        )
                                    })
                                    .collect();
                                ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, egui::Color32::LIGHT_GREEN)));
                            }
                            ui.weak("kinetic energy over time");
                        }

                        ui.separator();
                        ui.strong("Experiment");
                        if run.template == TEMPLATE_ORBITS {
                            ui.weak("Run the universe as it is, and again with a different script, from this moment. Then compare.");
                            ui.horizontal(|ui| {
                                ui.label("name");
                                ui.add(egui::TextEdit::singleline(&mut run.exp_name).desired_width(170.0));
                            });
                            ui.label("script for the changed run");
                            ui.add(egui::TextEdit::multiline(&mut run.exp_script).code_editor().desired_rows(6).desired_width(f32::INFINITY));
                        } else {
                        ui.weak("Change one law, run both versions from this moment, and compare.");
                        let before = run.exp_param;
                        egui::ComboBox::from_label("law").selected_text(simtara_store::PARAMETERS[run.exp_param].1).show_ui(ui, |ui| {
                            for (i, p) in simtara_store::PARAMETERS.iter().enumerate() {
                                ui.selectable_value(&mut run.exp_param, i, p.1);
                            }
                        });
                        let field = simtara_store::PARAMETERS[run.exp_param].0;
                        let current = simtara_store::get_parameter(&run.sim, field).unwrap_or(0.0);
                        if before != run.exp_param || run.exp_alt == 0.0 {
                            run.exp_alt = if current == 0.0 { 1.0 } else { current * 2.0 };
                        }
                        ui.horizontal(|ui| {
                            ui.label(format!("now {current:.2}  →  change to"));
                            ui.add(egui::DragValue::new(&mut run.exp_alt).speed(0.05));
                        });
                        }
                        let metrics: Vec<String> = run.sim.metric_names().iter().map(|m| m.to_string()).collect();
                        egui::ComboBox::from_label("measure").selected_text(run.exp_metric.clone()).show_ui(ui, |ui| {
                            for m in &metrics {
                                ui.selectable_value(&mut run.exp_metric, m.clone(), m);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("run for");
                            ui.add(egui::DragValue::new(&mut run.exp_ticks).range(10..=50_000).speed(10.0));
                            ui.label("ticks");
                            if run.template == TEMPLATE_ORBITS {
                                ui.weak(format!("(about {:.0} days)", run.exp_ticks as f64 * 0.25));
                            }
                        });
                        let busy = run.job.is_some();
                        if busy {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("running...");
                            });
                        } else if run.saved_as.is_some() {
                            if ui.button("▶ Run experiment").clicked() {
                                do_experiment = true;
                            }
                        } else {
                            ui.weak("Save the universe first (name and 💾 Save above), so results are kept with it.");
                        }
                        if !run.experiments.is_empty() {
                            ui.add_space(4.0);
                            ui.label("Past experiments");
                            for (i, e) in run.experiments.iter().enumerate() {
                                if ui.selectable_label(run.selected == Some(i), format!("{} · {}", e.spec.id, e.spec.name)).clicked() {
                                    run.selected = if run.selected == Some(i) { None } else { Some(i) };
                                }
                            }
                        }

                        ui.separator();
                        if ui.button("← Back to launcher").clicked() {
                            back = true;
                        }
                    });
                });

                if do_save {
                    run.save(&self.store);
                }
                if do_experiment {
                    run.run_experiment(&self.store);
                }

                if let Some(live) = &run.live {
                    egui::Panel::bottom("results").resizable(false).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            let have = live.result.series.first().map_or(0, |s| s.points.len());
                            ui.strong(format!("Running {} · {} samples so far", live.spec.name, have));
                        });
                        draw_chart(ui, live);
                    });
                } else if let Some(i) = run.selected {
                    let mut close = false;
                    egui::Panel::bottom("results").resizable(false).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Experiment {} · {}", run.experiments[i].spec.id, run.experiments[i].spec.name));
                            if let (Some(script), Some(u)) = (&run.experiments[i].spec.alt_script, &run.saved_as) {
                                if ui.button("➕ Keep the changed version as a new universe").clicked() {
                                    branch = Some((u.clone(), run.experiments[i].spec.id.clone(), script.clone()));
                                }
                            }
                            if ui.small_button("✕ close").clicked() {
                                close = true;
                            }
                        });
                        draw_chart(ui, &run.experiments[i]);
                    });
                    if close {
                        run.selected = None;
                    }
                }

                let t0 = std::time::Instant::now();
                run.sim.run_frame();
                run.ms = t0.elapsed().as_secs_f32() * 1000.0;
                if run.template == TEMPLATE_ORBITS {
                    run.update_trails();
                    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| draw_orbits(ui, run));
                } else {
                    let img = render(&run.sim);
                    match &mut run.tex {
                        Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                        None => run.tex = Some(ctx.load_texture("universe", img, egui::TextureOptions::NEAREST)),
                    }
                    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
                        if let Some(t) = &run.tex {
                            let side = ui.available_width().min(ui.available_height());
                            ui.centered_and_justified(|ui| {
                                ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(side, side)));
                            });
                        }
                    });
                }
                ctx.request_repaint();
                if let Some((universe, id, script)) = branch {
                    let made = (|| -> Result<Running, String> {
                        let snap = self.store.load_experiment_start(&universe, &id)?;
                        let mut sim = simtara_store::from_snapshot(&snap)?;
                        simtara_script::set_source(sim.world_mut(), &script)?;
                        sim.set_speed(3.0);
                        sim.play();
                        let mut r = Running::fresh(sim, ParticleConfig::default(), snap.config.seed, GpuModule::new(), self.store.suggest_name(), TEMPLATE_ORBITS, snap.config.clone());
                        r.status = format!("new universe from experiment {id} of '{universe}'. Press Save to keep it.");
                        Ok(r)
                    })();
                    match made {
                        Ok(r) => {
                            self.autosave();
                            self.screen = Screen::Universe(Box::new(r));
                            self.refresh();
                        }
                        Err(e) => self.notice = e,
                    }
                } else if back {
                    self.autosave();
                    self.screen = Screen::Launcher;
                    self.refresh();
                }
            }
        }
    }

    fn on_exit(&mut self) {
        self.autosave();
    }
}

fn main() -> eframe::Result {
    let g = GpuModule::new();
    let store = Store::default_location();
    let saved = store.list();
    // `simtara-studio --open "My universe"` skips the launcher.
    let args: Vec<String> = std::env::args().collect();
    let mut notice = String::new();
    let mut screen = Screen::Launcher;
    if let Some(name) = args.iter().position(|a| a == "--open").and_then(|i| args.get(i + 1)) {
        match Running::open(&store, name, GpuModule::new(), g.available()) {
            Ok(run) => screen = Screen::Universe(Box::new(run)),
            Err(e) => notice = format!("could not open '{name}': {e}"),
        }
    }
    let studio = Studio {
        launch: Launch { template: TEMPLATE_PARTICLES, count: 8000, kinds: 4, seed: 1, spread: 0.8, base: 0.0, use_gpu: g.available() },
        screen,
        gpu_desc: g.description(),
        gpu_ok: g.available(),
        store,
        saved,
        confirm_delete: None,
        notice,
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1250.0, 920.0]).with_title("SimTara Studio"),
        ..Default::default()
    };
    eframe::run_native("SimTara Studio", opts, Box::new(|_| Ok(Box::new(studio))))
}
