//! SimTara Studio: launcher + live universe with a control panel.

use eframe::egui;
use simtara_core::{Backend, Config, Simulation};
use simtara_gpu::GpuModule;
use simtara_particles::{simulation, Boundary, ParticleConfig, ParticleParams, Particles};

const VIEW: usize = 900;

struct Launch {
    count: usize,
    kinds: usize,
    seed: u64,
    spread: f32,
    base: f32,
    use_gpu: bool,
}

struct Running {
    sim: Simulation,
    initial: ParticleConfig,
    seed: u64,
    gpu: GpuModule,
    tex: Option<egui::TextureHandle>,
    ms: f32,
    history: Vec<[f64; 2]>,
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

impl Running {
    fn start(l: &Launch, gpu: GpuModule) -> Running {
        let mut pc = ParticleConfig::default();
        pc.count = l.count;
        pc.params.kinds = l.kinds;
        pc.spread = l.spread;
        pc.base_attraction = l.base;
        let mut sim = simulation(Config { seed: l.seed, ..Config::default() }, &pc);
        sim.add_module(gpu.clone());
        sim.compute_mut().set_requested(if l.use_gpu { Backend::Gpu } else { Backend::Cpu });
        sim.play();
        Running { sim, initial: pc, seed: l.seed, gpu, tex: None, ms: 0.0, history: Vec::new() }
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
}

impl eframe::App for Studio {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        match &mut self.screen {
            Screen::Launcher => {
                let mut go = false;
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.add_space(20.0);
                    ui.heading("SimTara");
                    ui.label("Choose the starting conditions of your universe, then launch.");
                    ui.add_space(12.0);
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
                if go {
                    let gpu = GpuModule::new();
                    self.screen = Screen::Universe(Box::new(Running::start(&self.launch, gpu)));
                }
            }
            Screen::Universe(run) => {
                let mut back = false;
                egui::Panel::left("controls").min_size(300.0).show(ui, |ui| {
                    ui.heading("Controls");
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
                    if ui.button("← Back to launcher").clicked() {
                        back = true;
                    }
                });

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
                    self.screen = Screen::Launcher;
                }
            }
        }
    }
}

fn main() -> eframe::Result {
    let g = GpuModule::new();
    let studio = Studio {
        launch: Launch { count: 8000, kinds: 4, seed: 1, spread: 0.8, base: 0.0, use_gpu: g.available() },
        screen: Screen::Launcher,
        gpu_desc: g.description(),
        gpu_ok: g.available(),
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1250.0, 920.0]).with_title("SimTara Studio"),
        ..Default::default()
    };
    eframe::run_native("SimTara Studio", opts, Box::new(|_| Ok(Box::new(studio))))
}
