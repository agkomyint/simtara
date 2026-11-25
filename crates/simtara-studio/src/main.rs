//! SimTara Studio: launcher (new + saved universes) and a live universe with a control panel,
//! saving, and experiments.

use eframe::egui;
use simtara_core::{Backend, Config, Simulation, Snapshot};
use simtara_gpu::GpuModule;
use simtara_particles::{simulation, Boundary, ParticleConfig, ParticleParams, Particles};
use simtara_store::{Experiment, Store, UniverseMeta};
use std::sync::mpsc::{channel, Receiver};

const VIEW: usize = 900;

struct Launch {
    count: usize,
    kinds: usize,
    seed: u64,
    spread: f32,
    base: f32,
    use_gpu: bool,
}

type ExperimentJob = Receiver<Result<(Experiment, Snapshot), String>>;

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
    exp_param: usize,
    exp_alt: f64,
    exp_ticks: u64,
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
    fn fresh(sim: Simulation, initial: ParticleConfig, seed: u64, gpu: GpuModule, name: String) -> Running {
        Running {
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
            exp_param: 0,
            exp_alt: 0.0,
            exp_ticks: 2000,
        }
    }

    fn start(l: &Launch, gpu: GpuModule, name: String) -> Running {
        let mut pc = ParticleConfig::default();
        pc.count = l.count;
        pc.params.kinds = l.kinds;
        pc.spread = l.spread;
        pc.base_attraction = l.base;
        let mut sim = simulation(Config { seed: l.seed, ..Config::default() }, &pc);
        sim.add_module(gpu.clone());
        sim.compute_mut().set_requested(if l.use_gpu { Backend::Gpu } else { Backend::Cpu });
        sim.play();
        Running::fresh(sim, pc, l.seed, gpu, name)
    }

    /// Reopen a saved universe at the tick it was saved.
    fn open(store: &Store, name: &str, gpu: GpuModule, use_gpu: bool) -> Result<Running, String> {
        let (meta, snap) = store.load(name)?;
        let mut sim = simtara_store::build_simulation(&snap.config);
        sim.add_module(gpu.clone());
        sim.restore(&snap)?;
        sim.compute_mut().set_requested(if use_gpu && gpu.available() { Backend::Gpu } else { Backend::Cpu });
        sim.play();
        let mut run = Running::fresh(sim, simtara_store::particle_config(&snap), snap.config.seed, gpu, meta.name.clone());
        run.saved_as = Some(meta.name.clone());
        run.experiments = store.list_experiments(&meta.name);
        run.selected = run.experiments.len().checked_sub(1);
        run.status = format!("opened '{}' at tick {}", meta.name, meta.tick);
        Ok(run)
    }

    fn save(&mut self, store: &Store) {
        match store.save(&self.name_input, &self.sim, simtara_store::TEMPLATE_PARTICLES) {
            Ok(meta) => {
                self.status = format!("saved '{}' at tick {}", meta.name, meta.tick);
                self.name_input = meta.name.clone();
                self.saved_as = Some(meta.name);
            }
            Err(e) => self.status = format!("could not save: {e}"),
        }
    }

    fn rebuild(&mut self, count: Option<usize>, new_seed: Option<u64>) {
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
            let spec = simtara_store::new_spec(id, &self.sim, field, self.exp_alt, self.exp_ticks)?;
            let start = self.sim.snapshot()?;
            Ok((spec, start))
        })();
        match result {
            Ok((spec, start)) => {
                let (tx, rx) = channel();
                std::thread::spawn(move || {
                    let out = simtara_store::run_experiment(&start, &spec).map(|result| (Experiment { spec, result }, start));
                    let _ = tx.send(out);
                });
                self.job = Some(rx);
                self.status = "experiment running...".into();
            }
            Err(e) => self.status = format!("could not start experiment: {e}"),
        }
    }

    /// Collect a finished experiment (if any) and write it to disk.
    fn poll_experiment(&mut self, store: &Store) {
        let Some(rx) = &self.job else { return };
        let Ok(out) = rx.try_recv() else { return };
        self.job = None;
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
            ui.colored_label(c, format!("● {} ({label} = {:.2})", s.label, s.value));
        }
    });
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), (ui.available_height() - 24.0).max(80.0)), egui::Sense::hover());
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
        for (s, c) in exp.result.series.iter().zip(colors) {
            let pts: Vec<egui::Pos2> = s
                .points
                .iter()
                .map(|p| egui::pos2(plot.left() + plot.width() * ((p[0] - x0) / (x1 - x0)) as f32, plot.bottom() - plot.height() * ((p[1] - lo) / (hi - lo)) as f32))
                .collect();
            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(2.0, c)));
        }
        let font = egui::FontId::proportional(11.0);
        let grey = egui::Color32::GRAY;
        ui.painter().text(egui::pos2(rect.left() + 4.0, plot.top()), egui::Align2::LEFT_TOP, format!("{hi:.0}"), font.clone(), grey);
        ui.painter().text(egui::pos2(rect.left() + 4.0, plot.bottom()), egui::Align2::LEFT_BOTTOM, format!("{lo:.0}"), font.clone(), grey);
        ui.painter().text(egui::pos2(plot.left(), rect.bottom() - 1.0), egui::Align2::LEFT_BOTTOM, format!("tick {x0:.0}"), font.clone(), grey);
        ui.painter().text(egui::pos2(plot.right(), rect.bottom() - 1.0), egui::Align2::RIGHT_BOTTOM, format!("tick {x1:.0}"), font, grey);
    }
    let r = &exp.result.record;
    ui.weak(format!("metric {} · engine v{} · seed {} · {} · {} ms", exp.spec.metric, r.engine_version, r.seed, r.backend, r.millis));
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
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add_space(20.0);
                        ui.heading("SimTara");

                        ui.add_space(8.0);
                        ui.strong("My universes");
                        if self.saved.is_empty() {
                            ui.weak("Nothing saved yet. Launch a universe below, then press Save.");
                        }
                        for m in &self.saved {
                            ui.horizontal(|ui| {
                                if ui.button("Open").clicked() {
                                    open = Some(m.name.clone());
                                }
                                ui.strong(&m.name);
                                ui.weak(format!("{} · tick {} · saved {}", m.summary, m.tick, age(m.updated)));
                                if self.confirm_delete.as_deref() == Some(m.name.as_str()) {
                                    ui.colored_label(egui::Color32::LIGHT_RED, "Delete for good?");
                                    if ui.button("Yes, delete").clicked() {
                                        delete = Some(m.name.clone());
                                    }
                                    if ui.button("No").clicked() {
                                        self.confirm_delete = None;
                                    }
                                } else if ui.small_button("Delete").clicked() {
                                    self.confirm_delete = Some(m.name.clone());
                                }
                            });
                        }
                        if !self.notice.is_empty() {
                            ui.colored_label(egui::Color32::LIGHT_RED, &self.notice);
                        }
                        ui.weak(format!("Saved in {}", self.store.root().display()));

                        ui.add_space(16.0);
                        ui.separator();
                        ui.add_space(8.0);
                        ui.strong("New universe");
                        ui.label("Choose the starting conditions, then launch.");
                        ui.add_space(8.0);
                        let l = &mut self.launch;
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
                        ui.add_space(20.0);
                        if ui.add(egui::Button::new("Launch universe").min_size(egui::vec2(180.0, 36.0))).clicked() {
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
                            if ui.button("🎲 New seed").clicked() {
                                let s = run.seed.wrapping_add(1);
                                run.rebuild(None, Some(s));
                            }
                        });
                        let mut speed = run.sim.speed() as f32;
                        if ui.add(egui::Slider::new(&mut speed, 0.1..=10.0).logarithmic(true).text("speed")).changed() {
                            run.sim.set_speed(speed as f64);
                        }
                        ui.separator();
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
                        ui.separator();
                        ui.label("Measurements");
                        let m = run.sim.collect_metrics();
                        ui.label(format!("tick {}   backend {:?}   {:.2} ms/frame", run.sim.world().clock.tick, run.sim.backend(), run.ms));
                        for (k, v) in &m {
                            ui.label(format!("{k}: {v:.2}"));
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
                        ui.horizontal(|ui| {
                            ui.label("run for");
                            ui.add(egui::DragValue::new(&mut run.exp_ticks).range(10..=50_000).speed(10.0));
                            ui.label("ticks");
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

                if let Some(i) = run.selected {
                    let mut close = false;
                    egui::Panel::bottom("results").min_size(230.0).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Experiment {} · {}", run.experiments[i].spec.id, run.experiments[i].spec.name));
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
                ctx.request_repaint();
                if back {
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
        launch: Launch { count: 8000, kinds: 4, seed: 1, spread: 0.8, base: 0.0, use_gpu: g.available() },
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
