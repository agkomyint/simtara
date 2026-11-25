//! The Build screen: write a universe as a script, run it, look inside it, change anything.
//!
//! Left: the script editor (colours, line numbers, errors shown at the line).
//! Right: what the universe is doing: measurements over time, every thing and its
//! properties (editable while it runs), the rules (switch on and off), and a reference.

use eframe::egui;
use serde_json::{json, Value};
use simtara_core::{RunState, Simulation};
use simtara_store::{Store, TEMPLATE_BUILD};
use simtara_things::Things;
use std::collections::{BTreeMap, BTreeSet};

pub const EXAMPLES: [(&str, &str, &str); 3] = [
    ("Empty universe", "A blank page with a few hints. Describe what exists, then the rules.", EMPTY),
    ("An idea spreads", "1,500 people, 3 of them know an idea. Each tells friends. No physics at all.", IDEA),
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
const REFERENCE: [(&str, &str, &str); 16] = [
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
    ("random", "A repeatable random number 0..1 (never math.random).", "if random() < 0.5 then end"),
    ("log", "Print a line to the Messages box below the editor.", "log(\"tick\", t)"),
];

const KEYWORDS: [&str; 22] = [
    "function", "end", "if", "then", "else", "elseif", "for", "while", "do", "in", "return", "local", "and", "or", "not", "true", "false", "nil", "repeat", "until", "break", "continue",
];

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Charts,
    Things,
    Rules,
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
}

fn make(seed: u64, script: &str, speed: f64) -> Simulation {
    let mut sim = simtara_store::build_simulation(&simtara_store::new_build_config(seed, script));
    sim.set_speed(speed);
    sim.play();
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
            tab: Tab::Charts,
            kind_filter: String::new(),
            last_tick: None,
            ms: 0.0,
            commands,
        }
    }

    pub fn open(store: &Store, name: &str) -> Result<BuildView, String> {
        let (meta, snap) = store.load(name)?;
        let mut sim = simtara_store::build_simulation(&snap.config);
        sim.restore(&snap)?;
        sim.play();
        let script = simtara_script::source(sim.world());
        let mut v = BuildView::with(sim, snap.config.seed, script, meta.name.clone());
        v.saved_as = Some(meta.name.clone());
        v.status = format!("opened '{}' at tick {}", meta.name, meta.tick);
        Ok(v)
    }

    pub fn save(&mut self, store: &Store) {
        match store.save(&self.name_input, &self.sim, TEMPLATE_BUILD) {
            Ok(meta) => {
                self.status = format!("saved '{}' at tick {}", meta.name, meta.tick);
                self.name_input = meta.name.clone();
                self.saved_as = Some(meta.name);
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
        self.sim.run_frame();
        self.ms = t0.elapsed().as_secs_f32() * 1000.0;
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
                if ui.button(if playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
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
                for (tab, label) in [(Tab::Charts, "📈 Measurements"), (Tab::Things, "🧩 Things"), (Tab::Rules, "📏 Rules"), (Tab::Reference, "📖 Reference")] {
                    ui.selectable_value(&mut self.tab, tab, label);
                }
            });
            ui.separator();
            match self.tab {
                Tab::Charts => self.charts(ui),
                Tab::Things => self.things(ui),
                Tab::Rules => self.rules(ui),
                Tab::Reference => self.reference(ui),
            }
        });
        back
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
        if t.things.is_empty() {
            ui.weak("Nothing exists yet. Create things with spawn{...} in setup().");
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
        egui::ScrollArea::vertical().show(ui, |ui| {
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
            let (err, _) = simtara_script::status(v.sim.world());
            assert!(err.is_none(), "{title}: {err:?}");
        }
        let mut v = BuildView::new(1, 7, "t".into());
        v.sim.advance(80);
        assert!(v.sim.collect_metrics()["people who know"] > 3.0, "the idea should spread");
    }
}
