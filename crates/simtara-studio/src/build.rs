//! The Build screen: write a universe as a script, run it, look inside it, change anything.
//!
//! Left: the script editor (colours, line numbers, errors shown at the line).
//! Right: what the universe is doing: measurements over time, every thing and its
//! properties (editable while it runs), the rules (switch on and off), and a reference.

use eframe::egui;
use serde_json::{json, Value};
use simtara_core::{RunState, Simulation};
use simtara_store::{Experiment, Store, TEMPLATE_BUILD};
use std::sync::mpsc::{channel, Receiver};
use simtara_things::Things;
use crate::view3d;
use std::collections::{BTreeMap, BTreeSet};

pub const EXAMPLES: [(&str, &str, &str); 14] = [
    ("Empty universe", "A blank page with a few hints. Describe what exists, then the rules.", EMPTY),
    ("An idea spreads", "1,500 people, 3 of them know an idea. Each tells friends. No physics at all.", IDEA),
    ("Forest fire (physics + chemistry)", "Fuel, heat and oxygen. Turn oxygen, rain and ignition temperature while it burns.", include_str!("../../../examples/fire.luau")),
    ("Chemical reaction patterns", "Two chemicals react and diffuse: spots, stripes and mazes grow. Turn feed and kill.", include_str!("../../../examples/reaction.luau")),
    ("Bouncing things (3D)", "120 balls, cubes, pyramids and ships bouncing. Turn the camera and light, change gravity and wind, shake as a human.", include_str!("../../../examples/balls3d.luau")),
    ("Smoke and wind around a building (3D)", "A real fluid in metres and seconds: wind flows around a building, a hot chimney's smoke rises and swirls in the wake.", include_str!("../../../examples/smoke.luau")),
    ("Fire plume (3D smoke and heat)", "A fire heats the air; the hot column rises, curls into a mushroom and breaks into swirls around a beam.", include_str!("../../../examples/plume.luau")),
    ("Water: a dam breaks (3D)", "75,000 water particles (FLIP): a wall of water crashes into a block, splashes up the far wall and settles.", include_str!("../../../examples/water.luau")),
    ("Pour a real liquid into a glass (3D)", "Real viscosity and density: honey, water, olive oil, milk, mercury... poured into a glass at real size. Change LIQUID at the top.", include_str!("../../../examples/pour.luau")),
    ("Airflow around a building (3D)", "A wind tunnel of 6,000 air cells. Smoke streaks show the air going around a building. Turn wind, viscosity and gusts.", include_str!("../../../examples/airflow.luau")),
    ("Heat on a plate", "A 40x40 plate with a hot square in the middle. Heat spreads to neighbours. Watch the picture.", HEAT),
    ("Towns and a disease", "Four towns of people, mostly meeting their own town. Watch the disease jump between towns.", TOWNS),
    ("Typed (Luau types)", "The same crowd, written with Luau type annotations so mistakes are caught early.", TYPED),
    ("An epidemic", "A disease moving through contacts: people get sick, then immune.", EPIDEMIC),
];

const EMPTY: &str = r##"-- An empty universe. Describe what exists in setup(); it runs once, at tick 0.
-- Rules then run on every thing of a kind, every tick. See the Reference tab for all commands.

function setup()
  -- spawn{kind = "person", props = {mood = 0.5}}
  -- add_rule{name = "calm", kind = "person", set = {mood = "mood * 0.99"}}
  -- define_metric{name = "average mood", kind = "person", prop = "mood", op = "mean"}
end

-- Optional: runs every tick, for things a rule can't say.
function on_tick(t)
end
"##;

const IDEA: &str = r##"-- An idea spreads through a crowd. Only data and rules: no bodies, no physics.
-- Change a number, then press "Run from start".
local PEOPLE = 1500
local FRIENDS = 6

function setup()
  local ids = {}
  for i = 1, PEOPLE do
    ids[i] = spawn{kind = "person", props = {knows = (i <= 3)}}
  end
  for i = 1, PEOPLE do
    for _ = 1, FRIENDS do
      relate{from = ids[i], to = ids[1 + math.floor(random() * PEOPLE)], kind = "friend"}
    end
  end
  define_metric{name = "people who know", kind = "person", prop = "knows", op = "sum"}
  -- every tick, each person who doesn't know yet may hear it from friends who do
  add_rule{name = "spread", kind = "person",
           when = "knows == 0 and random() < 0.12 * ncount(friend, knows)",
           set = {knows = "true"}}
end
"##;

const HEAT: &str = r##"-- Heat spreading across a plate. Each cell sits at (x, y) and touches its 4 neighbours.
-- view{...} says how to draw it: x and y place the cells, color shows the property "heat".
local N = 40

function setup()
  local id = {}
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local hot = gx >= 18 and gx < 22 and gy >= 18 and gy < 22
      id[gy * N + gx] = spawn{kind = "cell", props = {x = gx, y = gy, heat = hot and 100 or 0}}
    end
  end
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local me = id[gy * N + gx]
      if gx > 0 then relate{from = me, to = id[gy * N + gx - 1], kind = "next"} end
      if gx < N - 1 then relate{from = me, to = id[gy * N + gx + 1], kind = "next"} end
      if gy > 0 then relate{from = me, to = id[(gy - 1) * N + gx], kind = "next"} end
      if gy < N - 1 then relate{from = me, to = id[(gy + 1) * N + gx], kind = "next"} end
    end
  end
  add_rule{name = "diffuse", kind = "cell", set = {heat = "heat + 0.22 * (nmean(next, heat) - heat)"}}
  define_metric{name = "total heat", kind = "cell", prop = "heat", op = "sum"}
  define_metric{name = "hottest", kind = "cell", prop = "heat", op = "max"}
  view{x = "x", y = "y", color = "heat"}
end
"##;

const TOWNS: &str = r##"-- Four towns. state: 0 healthy, 1 sick, 2 immune. x,y place people, colour is the state.
local PER_TOWN = 400
local CENTRES = {{-6, -6}, {6, -6}, {-6, 6}, {6, 6}}

function setup()
  local towns = {}
  for t = 1, 4 do
    towns[t] = {}
    for i = 1, PER_TOWN do
      local sick = (t == 1 and i <= 4)
      towns[t][i] = spawn{kind = "person", props = {
        x = CENTRES[t][1] + (random() - 0.5) * 8, y = CENTRES[t][2] + (random() - 0.5) * 8,
        sick = sick, immune = false, state = sick and 1 or 0}}
    end
  end
  for t = 1, 4 do
    for i = 1, PER_TOWN do
      for _ = 1, 5 do
        -- 92% of contacts are in your own town, 8% travel
        local town = (random() < 0.92) and t or (1 + math.floor(random() * 4))
        relate{from = towns[t][i], to = towns[town][1 + math.floor(random() * PER_TOWN)], kind = "contact"}
      end
    end
  end
  add_rule{name = "infect", kind = "person", when = "sick == 0 and immune == 0 and random() < 0.07 * ncount(contact, sick)", set = {sick = "true"}}
  add_rule{name = "recover", kind = "person", when = "sick == 1 and random() < 0.04", set = {sick = "false", immune = "true"}}
  add_rule{name = "label", kind = "person", set = {state = "sick + immune * 2"}}
  define_metric{name = "sick", kind = "person", prop = "sick", op = "sum"}
  define_metric{name = "immune", kind = "person", prop = "immune", op = "sum"}
  view{x = "x", y = "y", color = "state"}
end
"##;

const TYPED: &str = r##"-- Luau lets you describe your data with types. They are checked when the script loads.
type Person = {knows: boolean, friends: number}

local PEOPLE: number = 800

local function makePerson(i: number): Person
  return {knows = i <= 2, friends = 5}
end

function setup()
  local ids: {number} = {}
  for i = 1, PEOPLE do
    local p = makePerson(i)
    ids[i] = spawn{kind = "person", props = {knows = p.knows}}
  end
  for i = 1, PEOPLE do
    local p = makePerson(i)
    for _ = 1, p.friends do
      relate{from = ids[i], to = ids[1 + math.floor(random() * PEOPLE)], kind = "friend"}
    end
  end
  define_metric{name = "people who know", kind = "person", prop = "knows", op = "sum"}
  add_rule{name = "spread", kind = "person",
           when = "knows == 0 and random() < 0.15 * ncount(friend, knows)",
           set = {knows = "true"}}
end
"##;

const EPIDEMIC: &str = r##"-- A disease in a crowd. sick = infectious now, immune = had it.
-- Try: lower the 0.06 and see how the peak changes.
local PEOPLE = 1500

function setup()
  local ids = {}
  for i = 1, PEOPLE do
    ids[i] = spawn{kind = "person", props = {sick = (i <= 5), immune = false}}
  end
  for i = 1, PEOPLE do
    for _ = 1, 5 do
      relate{from = ids[i], to = ids[1 + math.floor(random() * PEOPLE)], kind = "contact"}
    end
  end
  define_metric{name = "sick", kind = "person", prop = "sick", op = "sum"}
  define_metric{name = "immune", kind = "person", prop = "immune", op = "sum"}
  add_rule{name = "infect", kind = "person",
           when = "sick == 0 and immune == 0 and random() < 0.06 * ncount(contact, sick)",
           set = {sick = "true"}}
  add_rule{name = "recover", kind = "person",
           when = "sick == 1 and random() < 0.05",
           set = {sick = "false", immune = "true"}}
end
"##;

/// (title, what it does, what to type)
const REFERENCE: [(&str, &str, &str); 23] = [
    ("spawn", "Create a thing. Returns its id.", "local id = spawn{kind = \"person\", props = {age = 30, mood = \"calm\"}}"),
    ("despawn", "Remove a thing (and its relations).", "despawn(id)"),
    ("set", "Change properties of one thing.", "set{id = id, props = {age = 31}}"),
    ("set_all", "Change properties of every thing of a kind.", "set_all{kind = \"person\", props = {mood = 0}}"),
    ("get", "Read one thing.", "local p = get(id)"),
    ("things", "List things of a kind (text means kind).", "for _, p in ipairs(things(\"person\")) do end"),
    ("count", "How many things of a kind.", "log(count(\"person\"))"),
    ("relate", "Connect two things: knows, inside, next_to ...", "relate{from = a, to = b, kind = \"friend\", weight = 1}"),
    ("unrelate", "Remove a connection.", "unrelate{from = a, to = b, kind = \"friend\"}"),
    ("related", "Things this one points at.", "local friends = related{id = id, kind = \"friend\"}"),
    ("define_metric", "A measurement to chart. op: count, sum, mean, min, max.", "define_metric{name = \"average age\", kind = \"person\", prop = \"age\", op = \"mean\"}"),
    ("add_rule", "A formula run natively on every thing of a kind, each tick. Fast for huge crowds.", "add_rule{name = \"age\", kind = \"person\", set = {age = \"age + 1\"}, remove_when = \"age > 90\"}"),
    ("rule formulas", "Numbers, + - * / %, == != < > <= >=, and or not, if(c,a,b), min max abs sqrt floor clamp, random(), tick(), id, kind.", "when = \"energy < 1 and random() < 0.1\""),
    ("neighbours", "ncount(rel[,prop]) nsum nmean nmax nmin wsum wmean over the things this one points at.", "when = \"ncount(friend, knows) > 2\""),
    ("param", "A control you can turn while it runs. Rules use it by name like a property.", "param{name = \"spread\", value = 0.1, min = 0, max = 1, about = \"how contagious\"}"),
    ("owner", "Who may change a control. Add owner = \"nature\" (or \"cat,dog\") to param{}; Studio's Play as menu locks the others.", "param{name = \"rain\", value = 0, min = 0, max = 1, owner = \"nature\"}"),
    ("play_as", "Act as a role: only that role's controls can be changed. Empty = see and change all.", "play_as{role = \"firefighters\"}"),
    ("camera", "Place the 3D camera (it belongs to the universe and is saved). distance 1 fits everything; fov 0 is flat; follow a thing by id.", "camera{yaw = 35, pitch = 25, distance = 0.9, fov = 50, follow = 0}"),
    ("light", "Aim the 3D light. ambient is the light everywhere, strength the light from the lamp.", "light{yaw = 300, pitch = 55, ambient = 0.3, strength = 0.8, ground = true, grid = true}"),
    ("set_param", "Change a control from the script.", "set_param{name = \"spread\", value = 0.2}"),
    ("view", "How to draw things in the Universe tab. Each field names a property; links draws relations.", "view{x = \"x\", y = \"y\", color = \"heat\", size = \"mass\", links = true}"),
    ("random", "A repeatable random number 0..1 (never math.random).", "if random() < 0.5 then end"),
    ("log", "Print a line to the Messages box below the editor.", "log(\"tick\", t)"),
];

/// A model that ships with SimTara; it is copied into a project's models/ folder when it is saved.
const SHIP_OBJ: &str = include_str!("../../../examples/models/ship.obj");

const DEFAULT_EXP_NAME: &str = "what if it were different";

const KEYWORDS: [&str; 22] = [
    "function", "end", "if", "then", "else", "elseif", "for", "while", "do", "in", "return", "local", "and", "or", "not", "true", "false", "nil", "repeat", "until", "break", "continue",
];

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    World,
    Charts,
    Things,
    Rules,
    Test,
    Experiment,
    Reference,
}

pub struct BuildView {
    sim: Simulation,
    seed: u64,
    pub script: String,
    pub name_input: String,
    pub saved_as: Option<String>,
    pub status: String,
    history: BTreeMap<String, Vec<[f64; 2]>>,
    shown: BTreeSet<String>,
    shown_ready: bool,
    tab: Tab,
    kind_filter: String,
    last_tick: Option<u64>,
    ms: f32,
    commands: Vec<String>,
    pan: egui::Vec2,
    zoom: f32,
    /// x, y, colour, size chosen in the picture (None = what the script said; empty = none)
    pick: [Option<String>; 6],
    mode3d: Option<bool>,
    /// Draw the 3D view on the graphics card (falls back to the CPU painter when there is none).
    gpu3d_on: bool,
    gpu3d: Option<crate::view3d_gpu::GpuView>,
    /// Time the last 3D picture took to prepare and hand over, in ms (CPU or GPU).
    draw_ms: f32,
    /// The fluid step last drawn and when it arrived; the usual time between steps (seconds).
    step_seen: (u64, std::time::Instant),
    step_gap: f32,
    /// frames per second of the window (smoothed), and when the last frame started
    fps: f32,
    frame_at: std::time::Instant,
    models: std::collections::HashMap<String, view3d::Mesh>,
    models_for: Option<String>,
    model_errors: Vec<String>,
    show_links: Option<bool>,
    selected: Option<u64>,
    exp_name: String,
    exp_script: String,
    exp_metric: String,
    exp_ticks: u64,
    /// controls changed in the second run: name -> new value
    exp_overrides: BTreeMap<String, f64>,
    experiments: Vec<Experiment>,
    exp_sel: Option<usize>,
    exp_live: Option<Experiment>,
    exp_job: Option<Receiver<crate::JobMsg>>,
    test_ticks: u64,
    test_checks: String,
    report: Option<Value>,
}

/// Water is blue, and white where it is spray (`foam` from `Fluid::water_points`).
pub fn water_colour(foam: f32) -> egui::Color32 {
    let foam = foam.clamp(0.0, 1.0);
    let mix = |a: f32, b: f32| (a + (b - a) * foam) as u8;
    egui::Color32::from_rgb(mix(30.0, 225.0), mix(105.0, 238.0), mix(215.0, 255.0))
}

fn make(seed: u64, script: &str, speed: f64) -> Simulation {
    let mut sim = simtara_store::build_simulation(&simtara_store::new_build_config(seed, script));
    sim.set_speed(speed);
    // Start paused after setup() has built the world, so there is time to look and turn controls.
    sim.advance(1);
    sim.pause();
    sim
}

impl BuildView {
    pub fn new(example: usize, seed: u64, name: String) -> BuildView {
        let script = EXAMPLES[example.min(EXAMPLES.len() - 1)].2.to_string();
        BuildView::with(make(seed, &script, 1.0), seed, script, name)
    }

    fn with(sim: Simulation, seed: u64, script: String, name: String) -> BuildView {
        let commands = sim.world().command_names();
        BuildView {
            sim,
            seed,
            script,
            name_input: name,
            saved_as: None,
            status: String::new(),
            history: BTreeMap::new(),
            shown: BTreeSet::new(),
            shown_ready: false,
            tab: Tab::World,
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            pick: Default::default(),
            mode3d: None,
            gpu3d_on: true,
            gpu3d: None,
            draw_ms: 0.0,
            step_seen: (0, std::time::Instant::now()),
            step_gap: 0.1,
            fps: 0.0,
            frame_at: std::time::Instant::now(),
            models: std::collections::HashMap::new(),
            models_for: None,
            model_errors: Vec::new(),
            show_links: None,
            selected: None,
            kind_filter: String::new(),
            last_tick: None,
            ms: 0.0,
            commands,
            exp_name: DEFAULT_EXP_NAME.into(),
            exp_script: String::new(),
            exp_metric: String::new(),
            exp_ticks: 300,
            exp_overrides: BTreeMap::new(),
            experiments: Vec::new(),
            exp_sel: None,
            exp_live: None,
            exp_job: None,
            test_ticks: 200,
            test_checks: "people who know > 10".into(),
            report: None,
        }
    }

    pub fn open(store: &Store, name: &str) -> Result<BuildView, String> {
        let (meta, snap) = store.load(name)?;
        let mut sim = simtara_store::build_simulation(&snap.config);
        sim.restore(&snap)?;
        sim.pause();
        let script = simtara_script::source(sim.world());
        let mut v = BuildView::with(sim, snap.config.seed, script, meta.name.clone());
        v.saved_as = Some(meta.name.clone());
        v.experiments = store.list_experiments(&meta.name);
        v.exp_sel = v.experiments.len().checked_sub(1);
        v.status = format!("opened '{}' at tick {}", meta.name, meta.tick);
        Ok(v)
    }

    pub fn save(&mut self, store: &Store) {
        match store.save(&self.name_input, &self.sim, TEMPLATE_BUILD) {
            Ok(meta) => {
                self.status = format!("saved '{}' at tick {}", meta.name, meta.tick);
                self.name_input = meta.name.clone();
                if self.script.contains("\"ship\"") {
                    let _ = store.add_model_if_missing(&meta.name, "ship", SHIP_OBJ);
                }
                self.saved_as = Some(meta.name);
                self.models.clear();
            }
            Err(e) => self.status = format!("could not save: {e}"),
        }
    }

    fn restart(&mut self) {
        self.sim = make(self.seed, &self.script, self.sim.speed());
        self.history.clear();
        self.last_tick = None;
        self.shown_ready = false;
    }

    /// Give the running universe the edited script without starting over.
    fn apply_live(&mut self) {
        match simtara_script::set_source(self.sim.world_mut(), &self.script) {
            Ok(()) => self.status = "script applied to the running universe (setup() does not run again)".into(),
            Err(e) => self.status = e,
        }
    }

    fn record(&mut self) {
        let tick = self.sim.world().clock.tick;
        if self.last_tick == Some(tick) {
            return;
        }
        self.last_tick = Some(tick);
        for (k, v) in self.sim.collect_metrics() {
            if !self.shown_ready && !k.starts_with("things.") {
                self.shown.insert(k.clone());
            }
            let h = self.history.entry(k).or_default();
            h.push([tick as f64, v]);
            if h.len() > 4000 {
                h.drain(..1000);
            }
        }
        if !self.history.is_empty() && !self.shown_ready {
            if self.shown.is_empty() {
                self.shown.extend(self.history.keys().take(4).cloned());
            }
            self.shown_ready = true;
        }
    }

    /// Draw the whole screen. Returns true when the user wants to go back.
    pub fn show(&mut self, ui: &mut egui::Ui, store: &Store) -> bool {
        let mut back = false;
        let (mut run_start, mut run_live, mut do_save) = (false, false, false);
        let t0 = std::time::Instant::now();
        self.poll_experiment(store);
        let dt = self.frame_at.elapsed().as_secs_f32().max(1e-4);
        self.frame_at = std::time::Instant::now();
        self.fps = if self.fps == 0.0 { 1.0 / dt } else { self.fps * 0.9 + 0.1 / dt };
        // a fluid makes what the 3D view draws with each step, on its own thread
        let sun = self.sim.world().get::<Things>().map(|t| {
            let (y, p) = ((t.light.yaw as f32).to_radians(), (t.light.pitch as f32).to_radians());
            [p.cos() * y.cos(), p.cos() * y.sin(), p.sin()]
        });
        if let (Some(sun), Some(f)) = (sun, self.sim.world_mut().get_mut::<simtara_fields::Fluid>()) {
            f.want_render = true;
            f.sun = sun;
        }
        // Fluids compute their next step on another thread. While it is not ready, skip the tick
        // and keep drawing, so turning the camera stays smooth however slow the fluid is.
        let fluid_busy = self.sim.world().get::<simtara_fields::Fluid>().is_some_and(|f| f.busy());
        if self.sim.run_state() == RunState::Running && fluid_busy {
            ui.ctx().request_repaint();
        } else {
            self.sim.run_frame();
            self.ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
        self.record();

        egui::Panel::top("build_top").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("← Back").clicked() {
                    back = true;
                }
                ui.separator();
                ui.add(egui::TextEdit::singleline(&mut self.name_input).desired_width(150.0));
                if ui.button("💾 Save").clicked() {
                    do_save = true;
                }
                ui.separator();
                let playing = self.sim.run_state() == RunState::Running;
                let fresh = self.sim.world().clock.tick <= 1;
                let label = if playing { "⏸ Pause" } else if fresh { "▶ Start" } else { "▶ Play" };
                let btn = egui::Button::new(label);
                let btn = if fresh && !playing { btn.fill(egui::Color32::from_rgb(40, 130, 70)) } else { btn };
                if ui.add(btn).clicked() {
                    self.sim.toggle();
                }
                if ui.add_enabled(!playing, egui::Button::new("⏭ Step")).clicked() {
                    self.sim.advance(1);
                }
                let mut speed = self.sim.speed() as f32;
                if ui.add(egui::Slider::new(&mut speed, 0.05..=20.0).logarithmic(true).text("speed")).changed() {
                    self.sim.set_speed(speed as f64);
                }
                ui.separator();
                ui.label(format!("tick {}", self.sim.world().clock.tick));
                ui.weak(format!("{:.2} ms/frame", self.ms));
            });
            if !self.status.is_empty() {
                ui.weak(&self.status);
            }
            ui.add_space(2.0);
        });

        egui::Panel::left("build_editor").resizable(true).default_size(560.0).show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("▶ Run from start").fill(egui::Color32::from_rgb(40, 90, 160))).on_hover_text("Rebuild the universe at tick 0 from this script").clicked() {
                    run_start = true;
                }
                if ui.button("⟳ Apply to running").on_hover_text("Swap the script in without restarting (setup() does not run again)").clicked() {
                    run_live = true;
                }
                egui::ComboBox::from_id_salt("examples").selected_text("Examples").show_ui(ui, |ui| {
                    for (title, about, text) in EXAMPLES {
                        if ui.selectable_label(false, title).on_hover_text(about).clicked() {
                            self.script = text.to_string();
                            run_start = true;
                        }
                    }
                });
            });
            let (error, log) = simtara_script::status(self.sim.world());
            let error_line = error.as_deref().and_then(error_line);
            ui.add_space(4.0);
            let editor_height = (ui.available_height() - 150.0).max(150.0);
            egui::ScrollArea::both().max_height(editor_height).auto_shrink([false, false]).show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let lines = self.script.lines().count().max(1) + 1;
                    let mut gutter = egui::text::LayoutJob::default();
                    for n in 1..=lines {
                        let bad = error_line == Some(n);
                        gutter.append(
                            &format!("{n:>3}\n"),
                            0.0,
                            egui::TextFormat { font_id: egui::FontId::monospace(13.0), color: if bad { egui::Color32::LIGHT_RED } else { egui::Color32::from_gray(110) }, ..Default::default() },
                        );
                    }
                    ui.add(egui::Label::new(gutter).selectable(false));
                    let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, _wrap: f32| {
                        let job = highlight(text.as_str());
                        ui.fonts_mut(|f| f.layout_job(job))
                    };
                    ui.add(egui::TextEdit::multiline(&mut self.script).font(egui::FontId::monospace(13.0)).code_editor().desired_width(f32::INFINITY).desired_rows(30).layouter(&mut layouter));
                });
            });
            ui.add_space(6.0);
            match &error {
                Some(e) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ {e}"));
                    ui.weak("The script is switched off until you change it. Fix it, then Run from start.");
                }
                None => {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, "● script running without errors");
                }
            }
            egui::ScrollArea::vertical().id_salt("messages").max_height(70.0).stick_to_bottom(true).show(ui, |ui| {
                for l in &log {
                    ui.monospace(l);
                }
            });
        });

        if run_start {
            self.restart();
        }
        if run_live {
            self.apply_live();
        }
        if do_save {
            self.save(store);
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (tab, label) in [(Tab::World, "🌍 Universe"), (Tab::Charts, "📈 Measurements"), (Tab::Things, "🧩 Things"), (Tab::Rules, "📏 Rules"), (Tab::Test, "✅ Test"), (Tab::Experiment, "🧪 Experiment"), (Tab::Reference, "📖 Reference")] {
                    ui.selectable_value(&mut self.tab, tab, label);
                }
            });
            ui.separator();
            match self.tab {
                Tab::World => {
                    self.load_models(store);
                    self.world_view(ui, store)
                }
                Tab::Charts => self.charts(ui),
                Tab::Things => self.things(ui),
                Tab::Rules => self.rules(ui),
                Tab::Test => self.test(ui),
                Tab::Experiment => self.experiment(ui, store),
                Tab::Reference => self.reference(ui),
            }
        });
        back
    }

    /// A picture of the things: x and y place them, colour and size show a property.
    /// (Re)read the project's `models/*.obj` the first time, or when asked.
    fn load_models(&mut self, store: &Store) {
        if self.models_for == self.saved_as && !self.models.is_empty() {
            return;
        }
        self.models.clear();
        self.model_errors.clear();
        for n in ["cube", "pyramid"] {
            if let Some(m) = view3d::builtin(n) {
                self.models.insert(n.to_string(), m);
            }
        }
        if let Ok(m) = view3d::parse_obj(SHIP_OBJ) {
            self.models.insert("ship".into(), m);
        }
        if let Some(name) = &self.saved_as {
            for (n, text) in store.load_models(name) {
                match view3d::parse_obj(&text) {
                    Ok(m) => {
                        self.models.insert(n, m);
                    }
                    Err(e) => self.model_errors.push(format!("{n}.obj: {e}")),
                }
            }
        }
        self.models_for = self.saved_as.clone();
    }

    fn world_view(&mut self, ui: &mut egui::Ui, store: &Store) {
        let Some(t) = self.sim.world().get::<Things>() else {
            ui.weak("This universe has no things to draw.");
            return;
        };
        let fluid = self.sim.world().get::<simtara_fields::Fluid>().filter(|f| f.ready());
        if t.things.is_empty() && fluid.is_none() {
            ui.weak("Nothing exists yet. Create things with spawn{...} or a fluid{...} in setup().");
            return;
        }
        let mut param_edits: Vec<(String, f64)> = Vec::new();
        let mut role_edit: Option<String> = None;
        let role = t.role.clone();
        if !t.params.is_empty() {
            egui::CollapsingHeader::new(format!("🎛 Controls ({})", t.params.len())).default_open(true).show(ui, |ui| {
                let mut roles: Vec<String> = t.params.values().flat_map(|p| p.owner.split(',').map(|o| o.trim().to_string()).collect::<Vec<_>>()).filter(|o| !o.is_empty()).collect();
                roles.sort();
                roles.dedup();
                if !roles.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Play as");
                        egui::ComboBox::from_id_salt("play_as").selected_text(if role.is_empty() { "everyone (see all)".to_string() } else { role.clone() }).show_ui(ui, |ui| {
                            if ui.selectable_label(role.is_empty(), "everyone (see all)").clicked() {
                                role_edit = Some(String::new());
                            }
                            for r in &roles {
                                if ui.selectable_label(&role == r, r).clicked() {
                                    role_edit = Some(r.clone());
                                }
                            }
                        });
                        ui.weak("Controls that belong to someone else are locked 🔒");
                    });
                }
                egui::Grid::new("controls").num_columns(2).spacing([14.0, 4.0]).show(ui, |ui| {
                    for (name, p) in &t.params {
                        let mut v = p.value;
                        let mine = p.can_change(&role);
                        let who = if p.owner.is_empty() { String::new() } else { format!("  [{}]", p.owner) };
                        ui.label(format!("{}{}{}", if mine { "" } else { "🔒 " }, name, who)).on_hover_text(&p.about);
                        let step = (p.max - p.min) / 400.0;
                        if ui.add_enabled(mine, egui::Slider::new(&mut v, p.min..=p.max).step_by(step).custom_formatter(|n, _| fmt(n)).text(&p.about)).changed() {
                            param_edits.push((name.clone(), v));
                        }
                        ui.end_row();
                    }
                });
            });
        }
        let mut props: BTreeSet<&str> = BTreeSet::new();
        for th in t.things.values().take(500) {
            props.extend(th.props.keys().map(String::as_str));
        }
        let from_view = [t.view.x.clone(), t.view.y.clone(), t.view.color.clone(), t.view.size.clone(), t.view.z.clone(), t.view.shape.clone()];
        let mut chosen: [Option<String>; 6] = Default::default();
        for i in 0..6 {
            chosen[i] = match &self.pick[i] {
                Some(p) if p.is_empty() => None,
                Some(p) => Some(p.clone()),
                None => from_view[i].clone(),
            };
        }
        // properties literally called x and y make a map without being asked
        for (i, name) in [(0usize, "x"), (1, "y"), (4, "z")] {
            if chosen[i].is_none() && self.pick[i].is_none() && props.contains(name) {
                chosen[i] = Some(name.to_string());
            }
        }
        let mut links_on = self.show_links.unwrap_or(t.view.links);
        let mut new_pick = self.pick.clone();
        ui.horizontal_wrapped(|ui| {
            let mut three = self.mode3d.unwrap_or(match t.view.mode.as_deref() {
                Some("3d") => true,
                Some(_) => false,
                None => chosen[4].is_some(),
            });
            if ui.checkbox(&mut three, "3D").on_hover_text("Draw the universe in 3D: drag to turn the camera, scroll to move closer").changed() {
                self.mode3d = Some(three);
            }
            for (i, label) in ["x", "y", "colour", "size", "z", "shape"].iter().enumerate() {
                if (i >= 4) && !three {
                    continue;
                }
                ui.label(*label);
                let none_label = if i < 2 { "grid layout" } else { "none" };
                egui::ComboBox::from_id_salt(("pick", i)).width(90.0).selected_text(chosen[i].clone().unwrap_or_else(|| none_label.to_string())).show_ui(ui, |ui| {
                    if ui.selectable_label(chosen[i].is_none(), none_label).clicked() {
                        new_pick[i] = Some(String::new());
                    }
                    for p in &props {
                        if ui.selectable_label(chosen[i].as_deref() == Some(*p), *p).clicked() {
                            new_pick[i] = Some(p.to_string());
                        }
                    }
                });
            }
            ui.checkbox(&mut links_on, "links");
            if ui.small_button("reset view").clicked() {
                self.pan = egui::Vec2::ZERO;
                self.zoom = 1.0;
            }
        });
        for i in 0..6 {
            if new_pick[i] != self.pick[i] {
                chosen[i] = new_pick[i].clone().filter(|s| !s.is_empty());
            }
        }
        let num = |th: &simtara_things::Thing, p: &Option<String>| -> Option<f64> {
            match th.props.get(p.as_deref()?)? {
                Value::Number(n) => n.as_f64(),
                Value::Bool(b) => Some(*b as u8 as f64),
                _ => None,
            }
        };
        let n = t.things.len();
        let side = (n as f64).sqrt().ceil().max(1.0);
        let pts: Vec<(u64, [f64; 2], &simtara_things::Thing)> = t
            .things
            .values()
            .enumerate()
            .map(|(i, th)| {
                let gx = (i as f64) % side;
                let gy = (i as f64 / side).floor();
                (th.id, [num(th, &chosen[0]).unwrap_or(gx), num(th, &chosen[1]).unwrap_or(gy)], th)
            })
            .collect();
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for (_, p, _) in &pts {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        // colour: a number gets a gradient; a few whole numbers or text get distinct colours
        let (mut lo, mut hi, mut distinct) = (f64::MAX, f64::MIN, BTreeSet::new());
        let mut all_small_ints = true;
        if chosen[2].is_some() {
            for (_, _, th) in pts.iter().take(5000) {
                if let Some(v) = num(th, &chosen[2]) {
                    lo = lo.min(v);
                    hi = hi.max(v);
                    all_small_ints &= v.fract() == 0.0;
                    if distinct.len() < 20 {
                        distinct.insert(v as i64);
                    }
                }
            }
        }
        let discrete = all_small_ints && distinct.len() <= 8;
        let (mut slo, mut shi) = (f64::MAX, f64::MIN);
        if chosen[3].is_some() {
            for (_, _, th) in pts.iter().take(5000) {
                if let Some(v) = num(th, &chosen[3]) {
                    slo = slo.min(v);
                    shi = shi.max(v);
                }
            }
        }
        let color_of = |th: &simtara_things::Thing| -> egui::Color32 {
            match &chosen[2] {
                None => hash_color(&th.kind),
                Some(p) => match th.props.get(p) {
                    Some(Value::String(s)) => hash_color(s),
                    _ => match num(th, &chosen[2]) {
                        Some(v) if discrete => PALETTE[(v as i64).rem_euclid(PALETTE.len() as i64) as usize],
                        Some(v) => heat_color(if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }),
                        None => egui::Color32::GRAY,
                    },
                },
            }
        };
        let base = if n > 50_000 { 1.2 } else if n > 5_000 { 2.0 } else if n > 800 { 3.0 } else { 5.0 };

        // smoke and water only make sense in 3D
        let three = fluid.is_some() || self.mode3d.unwrap_or(match t.view.mode.as_deref() {
            Some("3d") => true,
            Some(_) => false,
            None => chosen[4].is_some(),
        });
        if three {
            let mut cam = t.camera.clone();
            let mut light = t.light.clone();
            let (cam0, light0) = (cam.clone(), light.clone());
            ui.horizontal_wrapped(|ui| {
                ui.label("camera");
                cam.yaw = view3d::wrap_degrees(cam.yaw);
                ui.add(egui::Slider::new(&mut cam.yaw, -180.0..=180.0).text("turn"));
                ui.add(egui::Slider::new(&mut cam.pitch, -89.0..=89.0).text("tilt"));
                ui.add(egui::Slider::new(&mut cam.distance, 0.05..=5.0).logarithmic(true).text("distance"));
                ui.add(egui::Slider::new(&mut cam.fov, 0.0..=120.0).text("lens").custom_formatter(|n, _| if n < 1.0 { "flat".into() } else { format!("{n:.0}°") }));
                // jump straight to a side: the camera on the -x side, the -y side, or above
                for (label, yaw, pitch, tip) in [("x", 180.0, 8.0, "Look along x (from the -x side)"), ("y", -90.0, 8.0, "Look along y (from the -y side)"), ("top", -90.0, 89.0, "Look down from above")] {
                    if ui.small_button(label).on_hover_text(tip).clicked() {
                        cam.yaw = yaw;
                        cam.pitch = pitch;
                    }
                }
                let has_gpu = ui.ctx().data(|d| d.get_temp::<eframe::egui_wgpu::RenderState>(egui::Id::unique(crate::RENDER_STATE))).is_some();
                ui.add_enabled(has_gpu, egui::Checkbox::new(&mut self.gpu3d_on, "GPU"))
                    .on_hover_text("Draw on the graphics card (fast, real depth). Off = the CPU painter.")
                    .on_disabled_hover_text("No graphics card available: drawing on the CPU.");
                if let Some(id) = self.selected {
                    if ui.selectable_label(cam.follow == id, format!("follow #{id}")).clicked() {
                        cam.follow = if cam.follow == id { 0 } else { id };
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("light");
                ui.add(egui::Slider::new(&mut light.yaw, 0.0..=360.0).text("direction"));
                ui.add(egui::Slider::new(&mut light.pitch, 0.0..=90.0).text("height"));
                ui.add(egui::Slider::new(&mut light.ambient, 0.0..=1.0).text("ambient"));
                ui.add(egui::Slider::new(&mut light.strength, 0.0..=1.5).text("strength"));
                ui.checkbox(&mut light.ground, "ground");
                ui.checkbox(&mut light.grid, "grid");
                if let Some(f) = self.sim.world().get::<simtara_fields::Fluid>().filter(|f| f.ready() && !f.particles.is_empty()) {
                    if ui.small_button("💧 export water mesh").on_hover_text("Save the water surface of this moment as an .obj (metres, z up) for Blender, Maya, Houdini ...").clicked() {
                        let mesh = f.surface_mesh(2);
                        let dir = store.root().join("exports");
                        let tick = self.sim.world().clock.tick;
                        let name = simtara_store::clean_name(&self.name_input).unwrap_or_else(|_| "universe".into());
                        let path = dir.join(format!("{name}-water-{tick}.obj"));
                        self.status = match std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, mesh.to_obj("water"))) {
                            Ok(()) => format!("saved {} ({} triangles)", path.display(), mesh.triangles.len()),
                            Err(e) => format!("could not save the mesh: {e}"),
                        };
                    }
                }
                if let Some(name) = self.saved_as.clone() {
                    if ui.small_button("📁 models").on_hover_text("Open this project's models folder; drop .obj files there, then reload").clicked() {
                        if let Ok(d) = store.models_dir(&name) {
                            let _ = std::fs::create_dir_all(&d);
                            let _ = std::process::Command::new("explorer").arg(d).spawn();
                        }
                    }
                    if ui.small_button("⟳ reload models").clicked() {
                        self.models.clear();
                    }
                }
            });
            for e in &self.model_errors {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
            let mut items: Vec<view3d::Item> = pts
                .iter()
                .map(|(id, p, th)| {
                    let z = num(th, &chosen[4]).unwrap_or(0.0);
                    let sz = match num(th, &chosen[3]) {
                        Some(v) if shi > slo => (0.4 + 2.2 * (v - slo) / (shi - slo)) as f32,
                        _ => 1.0,
                    };
                    let shape = match th.props.get(chosen[5].as_deref().unwrap_or("shape")) {
                        Some(Value::String(s)) => s.clone(),
                        _ => String::new(),
                    };
                    let heading = th.props.get("heading").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    view3d::Item { id: *id, pos: [p[0] as f32, p[1] as f32, z as f32], color: color_of(th), size: sz, heading, shape }
                })
                .collect();
            let size = ui.available_size_before_wrap();
            let (resp, _) = ui.allocate_painter(egui::vec2(size.x, (size.y - 10.0).max(200.0)), egui::Sense::click_and_drag());
            // GPU when the window has a wgpu device and the user has not switched it off
            let render_state = ui.ctx().data(|d| d.get_temp::<eframe::egui_wgpu::RenderState>(egui::Id::unique(crate::RENDER_STATE)));
            if !self.gpu3d_on {
                self.gpu3d = None;
            } else if self.gpu3d.is_none() {
                self.gpu3d = render_state.clone().map(crate::view3d_gpu::GpuView::new);
            }
            let t0 = std::time::Instant::now();
            // fluids: what to draw is made once per step on the fluid's thread (Fluid::render);
            // here it is only handed over, so a frame stays cheap however big the fluid is
            let sun = {
                let (y, p) = ((light.yaw as f32).to_radians(), (light.pitch as f32).to_radians());
                [p.cos() * y.cos(), p.cos() * y.sin(), p.sin()]
            };
            let on_gpu = self.gpu3d.is_some();
            let mut volume = None;
            if let Some(f) = fluid {
                let d = f.dims();
                let water = f.kind == simtara_fields::Kind::Water;
                let made;
                let r: &simtara_fields::RenderData = match f.render.as_deref() {
                    Some(r) => r,
                    None => {
                        let mut copy = simtara_fields::Fluid::default();
                        copy.clone_from(f);
                        copy.want_render = true;
                        copy.sun = sun;
                        copy.make_render();
                        made = copy.render.unwrap_or_default();
                        &made
                    }
                };
                volume = Some(view3d::Volume {
                    dims: [d.nx as u32, d.ny as u32, d.nz as u32],
                    size: f.extent(),
                    data: r.rgba.clone(),
                    water_radius: if water { f.cell * 0.55 } else { 0.0 },
                    caustics: r.caustics.clone(),
                    liquid: (water && !r.liquid.is_empty()).then(|| r.liquid.clone()),
                    generation: r.generation,
                    blend: {
                        // a new step: remember when it came and how long steps take
                        if self.step_seen.0 != r.generation {
                            let gap = self.step_seen.1.elapsed().as_secs_f32();
                            self.step_gap = if self.step_seen.0 == 0 { gap.min(0.2) } else { self.step_gap * 0.7 + gap.min(1.0) * 0.3 };
                            self.step_seen = (r.generation, std::time::Instant::now());
                        }
                        let playing = self.sim.run_state() == RunState::Running;
                        if playing { (self.step_seen.1.elapsed().as_secs_f32() / self.step_gap.max(1e-3)).min(1.0) } else { 1.0 }
                    },
                });
                if self.sim.run_state() == RunState::Running {
                    ui.ctx().request_repaint();
                }
                if on_gpu {
                    for (i, p) in r.spray.iter().enumerate() {
                        items.push(view3d::Item { id: u64::MAX - i as u64, pos: *p, color: water_colour(1.0), size: 1.0, heading: 0.0, shape: "spray".into() });
                    }
                } else if water {
                    // the CPU painter cannot draw a surface: it shows the particles
                    for (i, (p, foam)) in f.water_points().into_iter().enumerate() {
                        items.push(view3d::Item { id: u64::MAX - i as u64, pos: p, color: water_colour(foam), size: 1.0, heading: 0.0, shape: String::new() });
                    }
                }
            }
            let out = view3d::draw(ui, resp.rect, &resp, &items, &mut cam, &light, &self.models, self.selected, self.gpu3d.as_mut(), volume.as_ref());
            self.draw_ms = self.draw_ms * 0.9 + t0.elapsed().as_secs_f32() * 1000.0 * 0.1;
            if resp.clicked() {
                self.selected = out.hovered;
            }
            let painter = ui.painter_at(resp.rect);
            if let Some(at) = out.selected_at {
                painter.circle_stroke(at, 9.0, egui::Stroke::new(1.5, egui::Color32::WHITE));
            }
            if let Some(th) = out.hovered.or(self.selected).and_then(|id| t.things.get(&id)) {
                let props: Vec<String> = th
                    .props
                    .iter()
                    .map(|(k, v)| match v {
                        Value::Number(n) => format!("{k}={}", fmt(n.as_f64().unwrap_or(0.0))),
                        other => format!("{k}={}", other.to_string().trim_matches('"')),
                    })
                    .collect();
                painter.text(resp.rect.left_bottom() + egui::vec2(80.0, -8.0), egui::Align2::LEFT_BOTTOM, format!("#{} {}   {}", th.id, th.kind, props.join("  ")), egui::FontId::monospace(12.0), egui::Color32::WHITE);
            }
            painter.text(resp.rect.right_top() + egui::vec2(-8.0, 6.0), egui::Align2::RIGHT_TOP, format!("{n} things · 3D · {} {:.1} ms · {:.0} fps · drag to turn, scroll to zoom", if self.gpu3d.is_some() { "GPU" } else { "CPU" }, self.draw_ms, self.fps), egui::FontId::monospace(11.0), egui::Color32::from_gray(150));
            self.pick = new_pick;
            self.show_links = Some(links_on);
            let cam_changed = (cam.yaw, cam.pitch, cam.distance, cam.fov, cam.follow) != (cam0.yaw, cam0.pitch, cam0.distance, cam0.fov, cam0.follow);
            let light_changed = (light.yaw, light.pitch, light.ambient, light.strength, light.ground, light.grid) != (light0.yaw, light0.pitch, light0.ambient, light0.strength, light0.ground, light0.grid);
            if let Some(r) = role_edit {
                let _ = self.sim.world_mut().call("play_as", &json!({"role": r}));
            }
            for (name, value) in param_edits {
                let _ = self.sim.world_mut().call("set_param", &json!({"name": name, "value": value}));
            }
            if cam_changed {
                let _ = self.sim.world_mut().call("camera", &json!({"yaw": cam.yaw, "pitch": cam.pitch, "distance": cam.distance, "fov": cam.fov, "follow": cam.follow}));
            }
            if light_changed {
                let _ = self.sim.world_mut().call("light", &json!({"yaw": light.yaw, "pitch": light.pitch, "ambient": light.ambient, "strength": light.strength, "ground": light.ground, "grid": light.grid}));
            }
            return;
        }
        let size = ui.available_size_before_wrap();
        let (resp, painter) = ui.allocate_painter(egui::vec2(size.x, (size.y - 10.0).max(200.0)), egui::Sense::click_and_drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 4.0, egui::Color32::from_gray(14));
        if resp.dragged() {
            self.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let f = (1.0 + scroll * 0.0015).clamp(0.5, 2.0);
                let anchor = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                let before = anchor - rect.center() - self.pan;
                self.zoom = (self.zoom * f).clamp(0.2, 400.0);
                self.pan += before - before * f;
            }
        }
        let (w, h) = (((x1 - x0).max(1e-9)) as f32, ((y1 - y0).max(1e-9)) as f32);
        let fit = ((rect.width() - 30.0) / w).min((rect.height() - 30.0) / h);
        let scale = fit * self.zoom;
        let (cx, cy) = ((x0 + x1) as f32 / 2.0, (y0 + y1) as f32 / 2.0);
        let pan = self.pan;
        let to_screen = |p: [f64; 2]| egui::pos2(rect.center().x + pan.x + (p[0] as f32 - cx) * scale, rect.center().y + pan.y - (p[1] as f32 - cy) * scale);

        let painter = painter.with_clip_rect(rect);
        if links_on && n <= 30_000 {
            let pos: std::collections::HashMap<u64, egui::Pos2> = pts.iter().map(|(id, p, _)| (*id, to_screen(*p))).collect();
            let stroke = egui::Stroke::new(0.6, egui::Color32::from_white_alpha(28));
            for r in t.relations.iter().take(40_000) {
                if let (Some(a), Some(b)) = (pos.get(&r.from), pos.get(&r.to)) {
                    painter.line_segment([*a, *b], stroke);
                }
            }
        }
        let mut mesh = egui::Mesh::default();
        let mut hovered: Option<(f32, u64)> = None;
        let hover = resp.hover_pos();
        for (id, p, th) in pts.iter().take(300_000) {
            let c = to_screen(*p);
            if !rect.expand(10.0).contains(c) {
                continue;
            }
            let mut r = base;
            if chosen[3].is_some() {
                if let Some(v) = num(th, &chosen[3]) {
                    r = base * (0.4 + 2.2 * if shi > slo { ((v - slo) / (shi - slo)) as f32 } else { 0.5 });
                }
            }
            let r = r.max(0.8);
            mesh.add_colored_rect(egui::Rect::from_center_size(c, egui::vec2(r * 2.0, r * 2.0)), color_of(th));
            if let Some(hp) = hover {
                let d = c.distance(hp);
                if d < (r + 6.0) && hovered.is_none_or(|(bd, _)| d < bd) {
                    hovered = Some((d, *id));
                }
            }
        }
        painter.add(egui::Shape::mesh(mesh));
        if resp.clicked() {
            self.selected = hovered.map(|h| h.1);
        }
        if let Some(sel) = self.selected.and_then(|id| pts.iter().find(|p| p.0 == id)) {
            painter.circle_stroke(to_screen(sel.1), base + 5.0, egui::Stroke::new(1.5, egui::Color32::WHITE));
        }
        let show = hovered.map(|h| h.1).or(self.selected);
        if let Some(th) = show.and_then(|id| t.things.get(&id)) {
            let props: Vec<String> = th
                .props
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        Value::Number(n) => fmt(n.as_f64().unwrap_or(0.0)),
                        other => other.to_string().trim_matches('"').to_string(),
                    };
                    format!("{k}={val}")
                })
                .collect();
            painter.text(rect.left_bottom() + egui::vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, format!("#{} {}   {}", th.id, th.kind, props.join("  ")), egui::FontId::monospace(12.0), egui::Color32::WHITE);
        }
        let legend = match &chosen[2] {
            Some(p) if !discrete && hi > lo => format!("colour: {p}  {} → {}", fmt(lo), fmt(hi)),
            Some(p) => format!("colour: {p}"),
            None => "colour: kind".to_string(),
        };
        painter.text(rect.right_top() + egui::vec2(-8.0, 6.0), egui::Align2::RIGHT_TOP, format!("{n} things · {legend} · scroll to zoom, drag to move"), egui::FontId::monospace(11.0), egui::Color32::from_gray(150));
        self.pick = new_pick;
        self.show_links = Some(links_on);
        if let Some(r) = role_edit {
            let _ = self.sim.world_mut().call("play_as", &json!({"role": r}));
        }
        for (name, value) in param_edits {
            let _ = self.sim.world_mut().call("set_param", &json!({"name": name, "value": value}));
        }
    }

    fn charts(&mut self, ui: &mut egui::Ui) {
        if self.history.is_empty() {
            ui.weak("Nothing measured yet. Press Play, and use define_metric{...} in setup() to measure things.");
            return;
        }
        ui.collapsing("choose what to chart", |ui| {
            for k in self.history.keys() {
                let mut on = self.shown.contains(k);
                if ui.checkbox(&mut on, k).changed() {
                    if on {
                        self.shown.insert(k.clone());
                    } else {
                        self.shown.remove(k);
                    }
                }
            }
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (i, k) in self.shown.iter().enumerate() {
                if let Some(h) = self.history.get(k) {
                    chart(ui, k, h, PALETTE[i % PALETTE.len()]);
                }
            }
        });
    }

    fn things(&mut self, ui: &mut egui::Ui) {
        let Some(t) = self.sim.world().get::<Things>() else {
            ui.weak("This universe has no things.");
            return;
        };
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for th in t.things.values() {
            *kinds.entry(th.kind.clone()).or_default() += 1;
        }
        let relations = t.relations.len();
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("{} things, {} relations", t.things.len(), relations));
            ui.separator();
            if ui.selectable_label(self.kind_filter.is_empty(), "all").clicked() {
                self.kind_filter.clear();
            }
            for (k, n) in &kinds {
                if ui.selectable_label(self.kind_filter == *k, format!("{k} ({n})")).clicked() {
                    self.kind_filter = k.clone();
                }
            }
        });
        let fluid = self.sim.world().get::<simtara_fields::Fluid>().filter(|f| f.ready());
        if t.things.is_empty() && fluid.is_none() {
            ui.weak("Nothing exists yet. Create things with spawn{...} or a fluid{...} in setup().");
            return;
        }
        let rows: Vec<simtara_things::Thing> = t.of_kind(if self.kind_filter.is_empty() { None } else { Some(self.kind_filter.as_str()) }).take(200).cloned().collect();
        let cols: BTreeSet<String> = rows.iter().flat_map(|r| r.props.keys().cloned()).collect();
        ui.weak("Showing the first 200. Change any value and the universe uses it from the next tick.");
        let mut edits: Vec<(u64, String, Value)> = Vec::new();
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("things_grid").striped(true).spacing([14.0, 4.0]).show(ui, |ui| {
                ui.strong("id");
                ui.strong("kind");
                for c in &cols {
                    ui.strong(c);
                }
                ui.end_row();
                for r in &rows {
                    ui.label(r.id.to_string());
                    ui.label(&r.kind);
                    for c in &cols {
                        match r.props.get(c) {
                            Some(Value::Number(n)) => {
                                let mut v = n.as_f64().unwrap_or(0.0);
                                if ui.add(egui::DragValue::new(&mut v).speed(0.05)).changed() {
                                    edits.push((r.id, c.clone(), json!(v)));
                                }
                            }
                            Some(Value::Bool(b)) => {
                                let mut v = *b;
                                if ui.checkbox(&mut v, "").changed() {
                                    edits.push((r.id, c.clone(), json!(v)));
                                }
                            }
                            Some(Value::String(s)) => {
                                let mut v = s.clone();
                                if ui.add(egui::TextEdit::singleline(&mut v).desired_width(80.0)).changed() {
                                    edits.push((r.id, c.clone(), json!(v)));
                                }
                            }
                            _ => {
                                ui.weak("—");
                            }
                        }
                    }
                    ui.end_row();
                }
            });
        });
        for (id, prop, v) in edits {
            let _ = self.sim.world_mut().call("set", &json!({"id": id, "props": {prop: v}}));
        }
    }

    fn rules(&mut self, ui: &mut egui::Ui) {
        let rules: Vec<simtara_things::Rule> = self.sim.world().get::<Things>().map(|t| t.rules.clone()).unwrap_or_default();
        if rules.is_empty() {
            ui.weak("No rules yet. Add one in setup(): add_rule{name = \"...\", kind = \"...\", set = {...}}");
            return;
        }
        let mut toggles = Vec::new();
        let warnings = self.sim.world().get::<Things>().map(|t| t.warnings()).unwrap_or_default();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for w in &warnings {
                ui.colored_label(egui::Color32::from_rgb(240, 200, 90), format!("⚠ {w}"));
            }
            for r in &rules {
                egui::Frame::group(ui.style()).inner_margin(8.0).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let mut on = r.enabled;
                        if ui.checkbox(&mut on, "").changed() {
                            toggles.push((r.name.clone(), on));
                        }
                        ui.strong(&r.name);
                        ui.weak(format!("on every {}{}", r.kind, if r.every > 1 { format!(", every {} ticks", r.every) } else { String::new() }));
                    });
                    if let Some(w) = &r.when {
                        ui.monospace(format!("when  {w}"));
                    }
                    for (p, f) in &r.set {
                        ui.monospace(format!("set   {p} = {f}"));
                    }
                    if let Some(x) = &r.remove_when {
                        ui.monospace(format!("remove when  {x}"));
                    }
                    if let Some(e) = &r.error {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ switched off: {e}"));
                    }
                });
            }
        });
        for (name, on) in toggles {
            let _ = self.sim.world_mut().call("enable_rule", &json!({"name": name, "on": on}));
        }
    }

    /// Build and test: run this script from scratch, headless, and check the result.
    fn test(&mut self, ui: &mut egui::Ui) {
        ui.weak("Runs your script in a fresh universe (separate from the one on the left), then checks the results. Same engine as `simtara script` on the command line.");
        ui.horizontal(|ui| {
            ui.label("ticks");
            ui.add(egui::DragValue::new(&mut self.test_ticks).range(1..=20000));
        });
        ui.label("Checks, one per line, like  people who know >= 100  (operators: >= <= == != > <)");
        ui.add(egui::TextEdit::multiline(&mut self.test_checks).desired_rows(3).desired_width(f32::INFINITY).font(egui::FontId::monospace(13.0)));
        if ui.add(egui::Button::new("▶ Build & test").fill(egui::Color32::from_rgb(40, 110, 70))).clicked() {
            let checks: Vec<String> = self.test_checks.lines().map(str::to_string).filter(|l| !l.trim().is_empty()).collect();
            self.report = Some(simtara_store::run_script_report(&self.script, self.seed, self.test_ticks, &checks));
        }
        let Some(r) = &self.report else { return };
        ui.add_space(8.0);
        let ok = r["ok"] == json!(true);
        ui.label(egui::RichText::new(if ok { "✔ PASSED" } else { "✘ FAILED" }).size(20.0).strong().color(if ok { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED }));
        ui.weak(format!("{} ticks in {:.0} ms · repeatable with the same seed: {}", r["ticks"], r["ms"].as_f64().unwrap_or(0.0), if r["repeatable"] == json!(true) { "yes" } else { "NO" }));
        egui::ScrollArea::vertical().show(ui, |ui| {
            if let Some(e) = r["error"].as_str() {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ script error: {e}"));
            }
            for c in r["checks"].as_array().into_iter().flatten() {
                let pass = c["pass"] == json!(true);
                let detail = c["why"].as_str().map(str::to_string).or_else(|| c["got"].as_f64().map(|g| format!("got {}", fmt(g)))).unwrap_or_default();
                ui.colored_label(if pass { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED }, format!("{} {}   {}", if pass { "✔" } else { "✘" }, c["check"].as_str().unwrap_or(""), detail));
            }
            for rule in r["rules"].as_array().into_iter().flatten() {
                if let Some(e) = rule["error"].as_str() {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ rule '{}' broke: {e}", rule["name"].as_str().unwrap_or("")));
                }
            }
            for w in r["warnings"].as_array().into_iter().flatten() {
                ui.colored_label(egui::Color32::from_rgb(240, 200, 90), format!("⚠ {}", w.as_str().unwrap_or("")));
            }
            ui.add_space(6.0);
            ui.strong("Final measurements");
            if let Some(m) = r["metrics"].as_object() {
                for (k, v) in m {
                    ui.monospace(format!("{k}: {}", fmt(v.as_f64().unwrap_or(0.0))));
                }
            }
        });
    }

    /// Start "original vs changed script", both from tick 0 with the same seed, on a background thread.
    fn run_experiment(&mut self, store: &Store) {
        self.save(store);
        let Some(name) = self.saved_as.clone() else { return };
        let made = (|| -> Result<_, String> {
            let id = store.next_experiment_id(&name)?;
            let mut spec = simtara_store::new_script_spec(id, &self.exp_name, &self.exp_script, &self.exp_metric, self.exp_ticks);
            spec.param_overrides = self.exp_overrides.clone();
            spec.role = self.sim.world().get::<Things>().map(|t| t.role.clone()).unwrap_or_default();
            if self.exp_name == DEFAULT_EXP_NAME && !self.exp_overrides.is_empty() {
                let list: Vec<String> = self.exp_overrides.iter().map(|(k, v)| format!("{k} = {}", fmt(*v))).collect();
                spec.name = list.join(", ");
            }
            if self.exp_overrides.is_empty() && self.exp_script == self.script {
                return Err("nothing is different yet: change a control below or edit the changed script".into());
            }
            Ok((spec, self.sim.snapshot()?))
        })();
        match made {
            Ok((spec, start)) => {
                let live = Experiment {
                    spec: spec.clone(),
                    result: simtara_store::ExperimentResult {
                        schema_version: 0,
                        series: Vec::new(),
                        record: simtara_store::RunRecord { engine_version: String::new(), seed: 0, backend: "cpu".into(), start_tick: 0, millis: 0 },
                    },
                };
                let (tx, rx) = channel();
                std::thread::spawn(move || {
                    let txp = tx.clone();
                    let out = simtara_store::run_experiment_live(&start, &spec, &mut |s| {
                        let _ = txp.send(crate::JobMsg::Progress(s.to_vec()));
                    })
                    .map(|result| (Experiment { spec, result }, start));
                    let _ = tx.send(crate::JobMsg::Done(out));
                });
                self.exp_live = Some(live);
                self.exp_sel = None;
                self.exp_job = Some(rx);
                self.status = "experiment running...".into();
            }
            Err(e) => self.status = format!("could not start experiment: {e}"),
        }
    }

    fn poll_experiment(&mut self, store: &Store) {
        let Some(rx) = &self.exp_job else { return };
        let mut finished = None;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                crate::JobMsg::Progress(series) => {
                    if let Some(l) = self.exp_live.as_mut() {
                        l.result.series = series;
                    }
                }
                crate::JobMsg::Done(out) => {
                    finished = Some(out);
                    break;
                }
            }
        }
        let Some(out) = finished else { return };
        self.exp_job = None;
        self.exp_live = None;
        match out {
            Ok((exp, start)) => {
                let name = self.saved_as.clone().unwrap_or_default();
                self.status = match store.save_experiment(&name, &exp, &start) {
                    Ok(()) => format!("experiment finished in {} ms and saved", exp.result.record.millis),
                    Err(e) => format!("experiment finished but could not be saved: {e}"),
                };
                self.experiments.push(exp);
                self.exp_sel = Some(self.experiments.len() - 1);
            }
            Err(e) => self.status = format!("experiment failed: {e}"),
        }
    }

    fn experiment(&mut self, ui: &mut egui::Ui, store: &Store) {
        if self.exp_script.is_empty() {
            self.exp_script = self.script.clone();
        }
        if self.exp_metric.is_empty() {
            self.exp_metric = self.history.keys().find(|k| !k.starts_with("things.")).cloned().unwrap_or_else(|| "things.total".into());
        }
        let running = self.exp_job.is_some();
        ui.weak("Run your universe twice from tick 0 with the same seed: once as saved, once with the changed script below. Then compare one measurement.");
        ui.horizontal(|ui| {
            ui.label("name");
            ui.add(egui::TextEdit::singleline(&mut self.exp_name).desired_width(220.0));
            ui.label("measure");
            egui::ComboBox::from_id_salt("exp_metric").selected_text(self.exp_metric.clone()).show_ui(ui, |ui| {
                for k in self.history.keys() {
                    ui.selectable_value(&mut self.exp_metric, k.clone(), k);
                }
            });
            ui.label("ticks");
            ui.add(egui::DragValue::new(&mut self.exp_ticks).range(1..=20000));
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(!running, egui::Button::new("▶ Run experiment").fill(egui::Color32::from_rgb(40, 90, 160))).clicked() {
                self.run_experiment(store);
            }
            if ui.button("copy current script").on_hover_text("Replace the changed script with what is in the editor").clicked() {
                self.exp_script = self.script.clone();
            }
            if running {
                ui.spinner();
            }
        });
        let params: Vec<(String, simtara_things::Param)> = self.sim.world().get::<Things>().map(|t| t.params.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
        if params.is_empty() {
            ui.weak("Tip: declare controls in your script with param{name = \"spread\", value = 0.1, min = 0, max = 1}. They appear here (and in the Universe tab) as sliders, and you can compare different values without editing code.");
        } else {
            ui.add_space(4.0);
            ui.strong("Change a control for the second run");
            let role = self.sim.world().get::<Things>().map(|t| t.role.clone()).unwrap_or_default();
            if role.is_empty() {
                ui.weak("Tick the controls that should be different. The first run uses the script's own values.");
            } else {
                ui.weak(format!("You are playing '{role}': only your controls can be changed (🔒 = not yours). The first run uses the script's own values."));
            }
            egui::Grid::new("exp_params").num_columns(3).spacing([14.0, 4.0]).show(ui, |ui| {
                for (name, p) in &params {
                    let mut on = self.exp_overrides.contains_key(name);
                    let mine = p.can_change(&role);
                    if ui.add_enabled(mine, egui::Checkbox::new(&mut on, format!("{}{}", if mine { "" } else { "🔒 " }, name))).on_hover_text(&p.about).changed() {
                        if on {
                            self.exp_overrides.insert(name.clone(), p.value);
                        } else {
                            self.exp_overrides.remove(name);
                        }
                    }
                    ui.weak(&p.about);
                    match self.exp_overrides.get_mut(name) {
                        Some(v) => {
                            ui.add(egui::Slider::new(v, p.min..=p.max).step_by((p.max - p.min) / 400.0).custom_formatter(|n, _| fmt(n)));
                        }
                        None => {
                            ui.weak(format!("{}", fmt(p.value)));
                        }
                    }
                    ui.end_row();
                }
            });
            ui.add_space(4.0);
        }
        egui::CollapsingHeader::new("changed script (edit what is different)").default_open(false).show(ui, |ui| {
            let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, _wrap: f32| {
                let job = highlight(text.as_str());
                ui.fonts_mut(|f| f.layout_job(job))
            };
            egui::ScrollArea::vertical().max_height(260.0).id_salt("exp_script").show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut self.exp_script).font(egui::FontId::monospace(13.0)).code_editor().desired_width(f32::INFINITY).desired_rows(14).layouter(&mut layouter));
            });
        });
        ui.separator();
        if let Some(l) = &self.exp_live {
            ui.strong(format!("Running: {}", l.spec.name));
            if !l.result.series.is_empty() {
                crate::draw_chart(ui, l);
            }
        } else if let Some(e) = self.exp_sel.and_then(|i| self.experiments.get(i)) {
            ui.strong(format!("Experiment {} · {}", e.spec.id, e.spec.name));
            crate::draw_chart(ui, e);
        }
        if !self.experiments.is_empty() {
            ui.add_space(6.0);
            ui.label("Past experiments");
            for (i, e) in self.experiments.iter().enumerate() {
                if ui.selectable_label(self.exp_sel == Some(i), format!("{} · {} · {}", e.spec.id, e.spec.name, e.spec.metric)).clicked() {
                    self.exp_sel = Some(i);
                }
            }
        }
    }

    fn reference(&mut self, ui: &mut egui::Ui) {
        let mut insert: Option<&str> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.weak("Click Insert to add an example to the end of your script. Heavy work belongs in rules, not loops.");
            ui.add_space(4.0);
            for (title, about, example) in REFERENCE {
                egui::Frame::group(ui.style()).inner_margin(8.0).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.strong(title);
                        if ui.small_button("Insert").clicked() {
                            insert = Some(example);
                        }
                    });
                    ui.label(about);
                    ui.monospace(example);
                });
            }
            ui.add_space(6.0);
            ui.weak(format!("Every command in this universe: {}", self.commands.join(", ")));
        });
        if let Some(e) = insert {
            if !self.script.ends_with('\n') {
                self.script.push('\n');
            }
            self.script.push_str(&format!("{e}\n"));
        }
    }
}

const PALETTE: [egui::Color32; 6] = [
    egui::Color32::from_rgb(110, 200, 255),
    egui::Color32::from_rgb(255, 170, 90),
    egui::Color32::from_rgb(140, 220, 130),
    egui::Color32::from_rgb(230, 130, 200),
    egui::Color32::from_rgb(240, 220, 100),
    egui::Color32::from_rgb(170, 150, 255),
];

fn hash_color(s: &str) -> egui::Color32 {
    let h = s.bytes().fold(1469598103934665603u64, |a, b| (a ^ b as u64).wrapping_mul(1099511628211));
    PALETTE[(h % PALETTE.len() as u64) as usize]
}

/// 0 = cold blue, 0.5 = yellow, 1 = hot red.
fn heat_color(t: f64) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    let (r, g, b) = if t < 0.5 {
        let u = t * 2.0;
        (30.0 + 225.0 * u, 40.0 + 190.0 * u, 160.0 - 100.0 * u)
    } else {
        let u = (t - 0.5) * 2.0;
        (255.0, 230.0 - 190.0 * u, 60.0 - 30.0 * u)
    };
    egui::Color32::from_rgb(r as u8, g as u8, b as u8)
}

fn fmt(v: f64) -> String {
    if v.abs() >= 1e6 || (v != 0.0 && v.abs() < 1e-3) {
        format!("{v:.3e}")
    } else if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.3}")
    }
}

fn chart(ui: &mut egui::Ui, name: &str, pts: &[[f64; 2]], color: egui::Color32) {
    let last = pts.last().map_or(0.0, |p| p[1]);
    ui.horizontal(|ui| {
        ui.colored_label(color, "■");
        ui.strong(name);
        ui.label(fmt(last));
    });
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 90.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 3.0, egui::Color32::from_gray(20));
    if pts.len() > 1 {
        let (x0, x1) = (pts[0][0], pts[pts.len() - 1][0]);
        let (lo, hi) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        let span = if hi > lo { hi - lo } else { 1.0 };
        let line: Vec<egui::Pos2> = pts
            .iter()
            .map(|p| egui::pos2(rect.left() + rect.width() * ((p[0] - x0) / (x1 - x0).max(1.0)) as f32, rect.bottom() - 6.0 - (rect.height() - 12.0) * ((p[1] - lo) / span) as f32))
            .collect();
        ui.painter().add(egui::Shape::line(line, egui::Stroke::new(1.8, color)));
        ui.painter().text(rect.left_top() + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP, fmt(hi), egui::FontId::monospace(10.0), egui::Color32::from_gray(130));
        ui.painter().text(rect.left_bottom() + egui::vec2(4.0, -2.0), egui::Align2::LEFT_BOTTOM, fmt(lo), egui::FontId::monospace(10.0), egui::Color32::from_gray(130));
    }
    ui.add_space(8.0);
}

/// The line number in an error like `script:12: ...`.
fn error_line(e: &str) -> Option<usize> {
    let i = e.find("script:")? + "script:".len();
    let digits: String = e[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

fn highlight(text: &str) -> egui::text::LayoutJob {
    use egui::Color32 as C;
    let font = egui::FontId::monospace(13.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;
    let mut put = |s: &str, color: C| {
        if !s.is_empty() {
            job.append(s, 0.0, egui::TextFormat { font_id: font.clone(), color, ..Default::default() });
        }
    };
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        let start = i;
        if c == '-' && cs.get(i + 1) == Some(&'-') {
            while i < cs.len() && cs[i] != '\n' {
                i += 1;
            }
            put(&cs[start..i].iter().collect::<String>(), C::from_rgb(105, 140, 105));
        } else if c == '"' || c == '\'' {
            i += 1;
            while i < cs.len() && cs[i] != c && cs[i] != '\n' {
                i += 1;
            }
            i = (i + 1).min(cs.len());
            put(&cs[start..i].iter().collect::<String>(), C::from_rgb(225, 175, 110));
        } else if c.is_ascii_digit() {
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            put(&cs[start..i].iter().collect::<String>(), C::from_rgb(150, 200, 255));
        } else if c.is_alphabetic() || c == '_' {
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            let w: String = cs[start..i].iter().collect();
            let color = if KEYWORDS.contains(&w.as_str()) {
                C::from_rgb(205, 135, 235)
            } else if REFERENCE.iter().any(|r| r.0 == w) || ["setup", "on_tick", "ipairs", "pairs"].contains(&w.as_str()) {
                C::from_rgb(110, 205, 175)
            } else {
                C::from_gray(220)
            };
            put(&w, color);
        } else {
            i += 1;
            put(&c.to_string(), C::from_gray(200));
        }
    }
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_example_runs_clean_and_is_measured() {
        for (i, (title, _, _)) in EXAMPLES.iter().enumerate() {
            let mut v = BuildView::new(i, 7, "t".into());
            v.sim.advance(60);
            let r = simtara_store::run_script_report(&v.script, 7, 40, &[]);
            assert!(r["ok"] == json!(true) && r["warnings"].as_array().unwrap().is_empty(), "{title}: {r}");
            let (err, _) = simtara_script::status(v.sim.world());
            assert!(err.is_none(), "{title}: {err:?}");
        }
        let mut v = BuildView::new(1, 7, "t".into());
        v.sim.advance(80);
        assert!(v.sim.collect_metrics()["people who know"] > 3.0, "the idea should spread");
    }
}
