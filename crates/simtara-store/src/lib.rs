//! Saving and loading universes and experiments as plain folders.
//!
//! Layout (default root `Documents/SimTara`, override with `SIMTARA_HOME`):
//!
//! ```text
//! universes/<name>/
//!   universe.json          name, template, summary, tick, times, schema_version
//!   state.snap.json        the saved world (an engine Snapshot)
//!   experiments/<id>/
//!     experiment.json      what was changed, ticks, metric
//!     results.json         the measured series + a run record
//!     start.snap.json      the exact state the experiment started from
//! ```
//!
//! No UI code lives here, so the Studio, the CLI and the API can all use it.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use simtara_core::{Backend, Config, Simulation, Snapshot};
use simtara_particles::{ParticleConfig, ParticleParams, MODULE};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Version written into every file we create. Bump when a file's shape changes.
pub const SCHEMA_VERSION: u32 = 1;
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const TEMPLATE_PARTICLES: &str = "particles";
pub const TEMPLATE_ORBITS: &str = "orbits";
/// A universe described entirely by a script: things, relations and rules (plus orbits if the config asks).
pub const TEMPLATE_BUILD: &str = "build";

/// Parameters of the particles template that an experiment may change: (field, label).
pub const PARAMETERS: [(&str, &str); 5] =
    [("gravity", "gravity"), ("friction", "friction"), ("rmax", "range"), ("repel_core", "core repel"), ("coulomb", "charge")];

pub const DEFAULT_METRIC: &str = "particles.kinetic_energy";

/// The metric an experiment charts unless told otherwise.
pub fn default_metric(template: &str) -> &'static str {
    if template == TEMPLATE_BUILD {
        "things.total"
    } else if template == TEMPLATE_ORBITS {
        "orbits.distance_km.Earth"
    } else {
        DEFAULT_METRIC
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniverseMeta {
    pub schema_version: u32,
    pub name: String,
    pub template: String,
    pub summary: String,
    pub seed: u64,
    pub tick: u64,
    pub created: u64,
    pub updated: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentSpec {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    /// Field name inside the template's laws, e.g. "gravity". Empty for script experiments.
    #[serde(default)]
    pub parameter: String,
    #[serde(default)]
    pub base_value: f64,
    #[serde(default)]
    pub alt_value: f64,
    /// If set, the "changed" run uses this Lua script instead of the universe's own.
    #[serde(default)]
    pub alt_script: Option<String>,
    /// Controls (`param{...}`) set to different values in the changed run of a script universe.
    #[serde(default)]
    pub param_overrides: std::collections::BTreeMap<String, f64>,
    /// Who is acting in the changed run; they may only change controls they own ("" = anyone).
    #[serde(default)]
    pub role: String,
    pub ticks: u64,
    pub sample_every: u64,
    pub metric: String,
    pub created: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Series {
    pub label: String,
    pub value: f64,
    /// `[tick, metric value]` pairs.
    pub points: Vec<[f64; 2]>,
}

/// What is needed to trust and repeat a result later.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub engine_version: String,
    pub seed: u64,
    pub backend: String,
    pub start_tick: u64,
    pub millis: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub schema_version: u32,
    pub series: Vec<Series>,
    pub record: RunRecord,
}

#[derive(Clone, Debug)]
pub struct Experiment {
    pub spec: ExperimentSpec,
    pub result: ExperimentResult,
}

// ------------------------------------------------------------ building

/// Which template a config describes.
pub fn template_of(config: &Config) -> &'static str {
    if config.modules.contains_key(simtara_things::MODULE) {
        TEMPLATE_BUILD
    } else if config.modules.contains_key(simtara_orbits::MODULE) {
        TEMPLATE_ORBITS
    } else {
        TEMPLATE_PARTICLES
    }
}

/// A fresh simulation for whatever template the config describes (CPU only;
/// clients may add more modules, such as the GPU backend).
pub fn build_simulation(config: &Config) -> Simulation {
    let mut sim = Simulation::new(config.clone());
    if template_of(config) == TEMPLATE_BUILD {
        sim.add_module(simtara_things::ThingsModule);
        if config.modules.contains_key(simtara_orbits::MODULE) {
            sim.add_module(simtara_orbits::OrbitModule);
        }
        sim.add_module(simtara_script::ScriptModule);
    } else if template_of(config) == TEMPLATE_ORBITS {
        sim.add_module(simtara_orbits::OrbitModule);
        sim.add_module(simtara_script::ScriptModule);
    } else {
        sim.add_module(simtara_particles::ParticleModule);
    }
    sim
}

/// A new orbits universe (Sun, Earth, Moon) with an empty-by-default script.
pub fn new_orbits_config(seed: u64, script: &str) -> Config {
    let mut config = Config { seed, ..Config::default() };
    config.set_module(simtara_orbits::MODULE, &simtara_orbits::OrbitConfig::sun_earth_moon());
    config.set_module(simtara_script::MODULE, &simtara_script::ScriptConfig { source: script.to_string() });
    config
}

/// A new universe that exists only as a script (see `simtara-things`).
pub fn new_build_config(seed: u64, script: &str) -> Config {
    let mut config = Config { seed, ..Config::default() };
    config.set_module(simtara_things::MODULE, &serde_json::json!({}));
    config.set_module(simtara_script::MODULE, &simtara_script::ScriptConfig { source: script.to_string() });
    config
}

/// Rebuild a simulation exactly as it was when the snapshot was taken.
pub fn from_snapshot(snap: &Snapshot) -> Result<Simulation, String> {
    let mut sim = build_simulation(&snap.config);
    sim.restore(snap)?;
    Ok(sim)
}

/// The particle settings a snapshot was created with.
pub fn particle_config(snap: &Snapshot) -> ParticleConfig {
    snap.config.module(MODULE)
}

// ------------------------------------------------------------ files

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    write_atomic(path, text.as_bytes())
}

/// Write to a temp file then rename, so a crash never leaves a half-written save.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn check_version(v: u32, path: &Path) -> Result<(), String> {
    if v > SCHEMA_VERSION {
        Err(format!("{} was made by a newer SimTara (file version {v}, this build understands {SCHEMA_VERSION})", path.display()))
    } else {
        Ok(())
    }
}

/// Make a user-typed name safe to use as a folder name.
pub fn clean_name(name: &str) -> Result<String, String> {
    let s: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        Err("give the universe a name".into())
    } else if s.chars().count() > 60 {
        Err("name is too long (60 characters max)".into())
    } else {
        Ok(s)
    }
}

// ------------------------------------------------------------ store

pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Use `root` as the SimTara folder (universes go in `root/universes`).
    pub fn new(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into() }
    }

    /// `$SIMTARA_HOME`, else `Documents/SimTara` in the user's home folder.
    pub fn default_location() -> Store {
        if let Some(p) = std::env::var_os("SIMTARA_HOME") {
            return Store::new(p);
        }
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        Store::new(home.join("Documents").join("SimTara"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn universes_dir(&self) -> PathBuf {
        self.root.join("universes")
    }
    fn dir(&self, name: &str) -> Result<PathBuf, String> {
        Ok(self.universes_dir().join(clean_name(name)?))
    }

    /// The folder of one project: its state, `script.luau`, `models/` and `experiments/`.
    pub fn project_dir(&self, name: &str) -> Result<PathBuf, String> {
        self.dir(name)
    }

    /// Where a project keeps its 3D models (`.obj` files).
    pub fn models_dir(&self, name: &str) -> Result<PathBuf, String> {
        Ok(self.dir(name)?.join("models"))
    }

    /// Read every `.obj` in the project's models folder: (model name, file text).
    pub fn load_models(&self, name: &str) -> Vec<(String, String)> {
        let Ok(dir) = self.models_dir(name) else { return Vec::new() };
        let mut out: Vec<(String, String)> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("obj")))
            .filter_map(|e| Some((e.path().file_stem()?.to_string_lossy().to_string(), std::fs::read_to_string(e.path()).ok()?)))
            .collect();
        out.sort();
        out
    }

    /// Put a model into the project's models folder (does not overwrite one that is already there).
    pub fn add_model_if_missing(&self, name: &str, model: &str, obj: &str) -> Result<(), String> {
        let dir = self.models_dir(name)?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.obj", clean_name(model)?));
        if !path.exists() {
            std::fs::write(&path, obj).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn exists(&self, name: &str) -> bool {
        self.dir(name).map(|d| d.join("universe.json").is_file()).unwrap_or(false)
    }

    /// A name like "Universe 3" that is not used yet.
    pub fn suggest_name(&self) -> String {
        (1..).map(|n| format!("Universe {n}")).find(|n| !self.exists(n)).unwrap_or_else(|| "Universe".into())
    }

    /// Save the whole universe (overwrites a previous save of the same name).
    pub fn save(&self, name: &str, sim: &Simulation, template: &str) -> Result<UniverseMeta, String> {
        let name = clean_name(name)?;
        let dir = self.dir(&name)?;
        let snap = sim.snapshot()?;
        let created = read_json::<UniverseMeta>(&dir.join("universe.json")).map(|m| m.created).unwrap_or_else(|_| now());
        let summary = match (sim.world().get::<simtara_things::Things>(), sim.world().get::<simtara_orbits::Bodies>()) {
            (Some(t), _) => format!("{} things, {} rules", t.things.len(), t.rules.len()),
            (None, Some(b)) => format!("{} bodies", b.list.len()),
            (None, None) => format!("{} particles", sim.world().get::<simtara_particles::Particles>().map_or(0, |p| p.len())),
        };
        let meta = UniverseMeta {
            schema_version: SCHEMA_VERSION,
            name,
            template: template.to_string(),
            summary,
            seed: snap.config.seed,
            tick: snap.tick,
            created,
            updated: now(),
        };
        let text = serde_json::to_vec(&snap).map_err(|e| e.to_string())?;
        write_atomic(&dir.join("state.snap.json"), &text)?;
        write_json(&dir.join("universe.json"), &meta)?;
        // A script universe is also a plain folder you (or a coding agent) can edit: script.luau + models/.
        if template == TEMPLATE_BUILD {
            write_atomic(&dir.join("script.luau"), simtara_script::source(sim.world()).as_bytes())?;
            let _ = std::fs::create_dir_all(dir.join("models"));
        }
        Ok(meta)
    }

    pub fn load(&self, name: &str) -> Result<(UniverseMeta, Snapshot), String> {
        let dir = self.dir(name)?;
        let meta_path = dir.join("universe.json");
        let meta: UniverseMeta = read_json(&meta_path)?;
        check_version(meta.schema_version, &meta_path)?;
        let snap: Snapshot = read_json(&dir.join("state.snap.json"))?;
        Ok((meta, snap))
    }

    /// All saved universes, most recently saved first. Unreadable folders are skipped.
    pub fn list(&self) -> Vec<UniverseMeta> {
        let mut out: Vec<UniverseMeta> = std::fs::read_dir(self.universes_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| read_json::<UniverseMeta>(&e.path().join("universe.json")).ok())
            .collect();
        out.sort_by(|a, b| b.updated.cmp(&a.updated).then(a.name.cmp(&b.name)));
        out
    }

    pub fn delete(&self, name: &str) -> Result<(), String> {
        let dir = self.dir(name)?;
        if !dir.join("universe.json").is_file() {
            return Err(format!("no universe named '{name}'"));
        }
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())
    }

    // ---- experiments ----

    fn experiments_dir(&self, universe: &str) -> Result<PathBuf, String> {
        Ok(self.dir(universe)?.join("experiments"))
    }

    /// Next free id: exp-001, exp-002, ...
    pub fn next_experiment_id(&self, universe: &str) -> Result<String, String> {
        let used = std::fs::read_dir(self.experiments_dir(universe)?)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_string_lossy().strip_prefix("exp-").and_then(|n| n.parse::<u32>().ok()))
            .max()
            .unwrap_or(0);
        Ok(format!("exp-{:03}", used + 1))
    }

    pub fn save_experiment(&self, universe: &str, exp: &Experiment, start: &Snapshot) -> Result<(), String> {
        let dir = self.experiments_dir(universe)?.join(&exp.spec.id);
        let text = serde_json::to_vec(start).map_err(|e| e.to_string())?;
        write_atomic(&dir.join("start.snap.json"), &text)?;
        write_json(&dir.join("experiment.json"), &exp.spec)?;
        write_json(&dir.join("results.json"), &exp.result)
    }

    /// Saved experiments of a universe, oldest first.
    pub fn list_experiments(&self, universe: &str) -> Vec<Experiment> {
        let Ok(dir) = self.experiments_dir(universe) else { return Vec::new() };
        let mut out: Vec<Experiment> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let spec: ExperimentSpec = read_json(&e.path().join("experiment.json")).ok()?;
                let result: ExperimentResult = read_json(&e.path().join("results.json")).ok()?;
                (spec.schema_version <= SCHEMA_VERSION).then_some(Experiment { spec, result })
            })
            .collect();
        out.sort_by(|a, b| a.spec.id.cmp(&b.spec.id));
        out
    }

    /// Load the exact start state of a saved experiment (to repeat it).
    pub fn load_experiment_start(&self, universe: &str, id: &str) -> Result<Snapshot, String> {
        read_json(&self.experiments_dir(universe)?.join(id).join("start.snap.json"))
    }
}

// ------------------------------------------------------------ experiments

/// Set one named law of the particles template on a running simulation.
pub fn set_parameter(sim: &mut Simulation, field: &str, value: f64) -> Result<(), String> {
    if !PARAMETERS.iter().any(|(f, _)| *f == field) {
        return Err(format!("unknown parameter '{field}' (try: {})", PARAMETERS.map(|p| p.0).join(", ")));
    }
    let p = sim.world_mut().get_mut::<ParticleParams>().ok_or("this universe has no particle laws")?;
    let mut v = serde_json::to_value(&*p).map_err(|e| e.to_string())?;
    let slot = v.get_mut(field).ok_or_else(|| format!("no parameter '{field}'"))?;
    *slot = serde_json::json!(value);
    *p = serde_json::from_value(v).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn get_parameter(sim: &Simulation, field: &str) -> Option<f64> {
    let p = sim.world().get::<ParticleParams>()?;
    serde_json::to_value(p).ok()?.get(field)?.as_f64()
}

fn arm_sim(start: &Snapshot, spec: &ExperimentSpec, value: f64, changed: bool) -> Result<Simulation, String> {
    if template_of(&start.config) == TEMPLATE_BUILD {
        // A script universe is described by its script, so both runs start again from tick 0 with
        // the same seed: the original script, and the changed script.
        let original = simtara_script::source(from_snapshot(start)?.world());
        let script = match (&spec.alt_script, changed) {
            (Some(alt), true) => alt.as_str(),
            _ => original.as_str(),
        };
        let mut sim = build_simulation(&new_build_config(start.config.seed, script));
        if changed && !spec.role.is_empty() {
            let known = from_snapshot(start)?;
            if let Some(t) = known.world().get::<simtara_things::Things>() {
                for name in spec.param_overrides.keys() {
                    if let Some(p) = t.params.get(name) {
                        if !p.can_change(&spec.role) {
                            return Err(format!("as '{}' you cannot change '{name}': it belongs to '{}'", spec.role, p.owner));
                        }
                    }
                }
            }
        }
        if changed {
            if let Some(t) = sim.world_mut().get_mut::<simtara_things::Things>() {
                t.overrides = spec.param_overrides.clone();
            }
        }
        return Ok(sim);
    }
    let mut sim = from_snapshot(start)?;
    // CPU on purpose: CPU runs repeat exactly, GPU runs are only approximately repeatable.
    sim.compute_mut().set_requested(Backend::Cpu);
    match (&spec.alt_script, changed) {
        (Some(script), true) => simtara_script::set_source(sim.world_mut(), script)?,
        (Some(_), false) => {}
        (None, _) => set_parameter(&mut sim, &spec.parameter, value)?,
    }
    Ok(sim)
}

fn sample(sim: &Simulation, metric: &str) -> Result<[f64; 2], String> {
    let m = sim.collect_metrics();
    // A script universe defines its measurements in setup(), which runs on the first tick.
    let v = match m.get(metric) {
        Some(v) => *v,
        None if sim.world().clock.tick == 0 && template_of(sim.config()) == TEMPLATE_BUILD => 0.0,
        None => return Err(format!("unknown metric '{metric}'")),
    };
    Ok([sim.world().clock.tick as f64, v])
}

/// Run the original and the changed version side by side. `progress` is called after every sample
/// with the two series so far, so a screen can draw the lines as they grow.
pub fn run_experiment_live(start: &Snapshot, spec: &ExperimentSpec, progress: &mut dyn FnMut(&[Series])) -> Result<ExperimentResult, String> {
    let t0 = Instant::now();
    let mut sims = [arm_sim(start, spec, spec.base_value, false)?, arm_sim(start, spec, spec.alt_value, true)?];
    let mut series = vec![
        Series { label: "original".into(), value: spec.base_value, points: vec![sample(&sims[0], &spec.metric)?] },
        Series { label: "changed".into(), value: spec.alt_value, points: vec![sample(&sims[1], &spec.metric)?] },
    ];
    progress(&series);
    let every = spec.sample_every.max(1);
    let mut done = 0;
    while done < spec.ticks {
        let n = every.min(spec.ticks - done);
        for (sim, s) in sims.iter_mut().zip(series.iter_mut()) {
            sim.advance(n);
            s.points.push(sample(sim, &spec.metric)?);
        }
        done += n;
        progress(&series);
    }
    Ok(ExperimentResult {
        schema_version: SCHEMA_VERSION,
        series,
        record: RunRecord {
            engine_version: ENGINE_VERSION.to_string(),
            seed: start.config.seed,
            backend: "cpu".into(),
            start_tick: if template_of(&start.config) == TEMPLATE_BUILD { 0 } else { start.tick },
            millis: t0.elapsed().as_millis() as u64,
        },
    })
}

pub fn run_experiment(start: &Snapshot, spec: &ExperimentSpec) -> Result<ExperimentResult, String> {
    run_experiment_live(start, spec, &mut |_| {})
}

pub fn new_spec(id: String, sim: &Simulation, parameter: &str, alt_value: f64, ticks: u64) -> Result<ExperimentSpec, String> {
    let base_value = get_parameter(sim, parameter).ok_or_else(|| format!("unknown parameter '{parameter}'"))?;
    let label = PARAMETERS.iter().find(|p| p.0 == parameter).map_or(parameter, |p| p.1);
    Ok(ExperimentSpec {
        schema_version: SCHEMA_VERSION,
        id,
        name: format!("{label} {base_value:.2} vs {alt_value:.2}"),
        parameter: parameter.to_string(),
        base_value,
        alt_value,
        alt_script: None,
        param_overrides: Default::default(),
        role: String::new(),
        ticks: ticks.max(1),
        sample_every: (ticks / 200).max(1),
        metric: DEFAULT_METRIC.to_string(),
        created: now(),
    })
}

/// "Run this universe as it is, and again with this script": e.g. remove the Moon.
pub fn new_script_spec(id: String, name: &str, script: &str, metric: &str, ticks: u64) -> ExperimentSpec {
    ExperimentSpec {
        schema_version: SCHEMA_VERSION,
        id,
        name: name.trim().to_string(),
        parameter: String::new(),
        base_value: 0.0,
        alt_value: 0.0,
        alt_script: Some(script.to_string()),
        param_overrides: Default::default(),
        role: String::new(),
        ticks: ticks.max(1),
        sample_every: (ticks / 200).max(1),
        metric: metric.to_string(),
        created: now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_sim(seed: u64) -> Simulation {
        let pc = ParticleConfig { count: 300, ..ParticleConfig::default() };
        simtara_particles::simulation(Config { seed, ..Config::default() }, &pc)
    }

    fn temp_store(tag: &str) -> Store {
        let dir = std::env::temp_dir().join(format!("simtara-store-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Store::new(dir)
    }

    #[test]
    fn save_then_load_gives_the_same_universe() {
        let store = temp_store("roundtrip");
        let mut sim = small_sim(7);
        sim.advance(25);
        set_parameter(&mut sim, "gravity", 7.5).unwrap();
        store.save("My first", &sim, TEMPLATE_PARTICLES).unwrap();

        let (meta, snap) = store.load("My first").unwrap();
        assert_eq!(meta.tick, 25);
        assert_eq!(meta.seed, 7);
        let mut back = from_snapshot(&snap).unwrap();
        assert_eq!(back.world().clock.tick, 25);
        assert_eq!(get_parameter(&back, "gravity"), Some(7.5));
        // Same state means the same future.
        sim.advance(10);
        back.advance(10);
        assert_eq!(sim.collect_metrics(), back.collect_metrics());
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn list_and_delete_work() {
        let store = temp_store("list");
        let sim = small_sim(1);
        store.save("a", &sim, TEMPLATE_PARTICLES).unwrap();
        store.save("b", &sim, TEMPLATE_PARTICLES).unwrap();
        assert_eq!(store.list().len(), 2);
        assert_eq!(store.suggest_name(), "Universe 1");
        store.delete("a").unwrap();
        assert_eq!(store.list().len(), 1);
        assert!(store.delete("a").is_err());
        assert!(store.load("a").is_err());
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn names_are_made_safe() {
        assert_eq!(clean_name("  ../evil/name ").unwrap(), "___evil_name");
        assert!(clean_name("   ").is_err());
        assert!(clean_name(&"x".repeat(61)).is_err());
    }

    #[test]
    fn newer_files_are_refused_politely() {
        let store = temp_store("version");
        let sim = small_sim(1);
        let mut meta = store.save("v", &sim, TEMPLATE_PARTICLES).unwrap();
        meta.schema_version = SCHEMA_VERSION + 1;
        write_json(&store.dir("v").unwrap().join("universe.json"), &meta).unwrap();
        let err = store.load("v").unwrap_err();
        assert!(err.contains("newer SimTara"), "{err}");
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn experiment_shows_a_difference_and_repeats_exactly() {
        let mut sim = small_sim(3);
        sim.advance(10);
        let start = sim.snapshot().unwrap();
        let spec = new_spec("exp-001".into(), &sim, "gravity", 12.0, 120).unwrap();

        let a = run_experiment(&start, &spec).unwrap();
        let b = run_experiment(&start, &spec).unwrap();
        assert_eq!(a.series.len(), 2);
        assert_ne!(a.series[0].points.last(), a.series[1].points.last(), "changing gravity should change the outcome");
        assert_eq!(a.series[0].points, b.series[0].points, "same start, same numbers");
        assert_eq!(a.series[1].points, b.series[1].points);
        assert_eq!(a.series[0].points[0][0], 10.0, "starts at the saved tick");
        assert_eq!(a.record.backend, "cpu");
    }

    #[test]
    fn experiments_are_saved_and_come_back() {
        let store = temp_store("exp");
        let mut sim = small_sim(5);
        sim.advance(5);
        store.save("u", &sim, TEMPLATE_PARTICLES).unwrap();
        let start = sim.snapshot().unwrap();
        let id = store.next_experiment_id("u").unwrap();
        assert_eq!(id, "exp-001");
        let spec = new_spec(id, &sim, "friction", 3.0, 40).unwrap();
        let result = run_experiment(&start, &spec).unwrap();
        store.save_experiment("u", &Experiment { spec, result: result.clone() }, &start).unwrap();

        assert_eq!(store.next_experiment_id("u").unwrap(), "exp-002");
        let back = store.list_experiments("u");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].result.series[1].points, result.series[1].points);
        assert!(store.load_experiment_start("u", "exp-001").is_ok());
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn bad_parameters_are_rejected() {
        let mut sim = small_sim(1);
        assert!(set_parameter(&mut sim, "banana", 1.0).unwrap_err().contains("unknown parameter"));
    }

    #[test]
    fn remove_the_moon_experiment_saves_reopens_and_differs() {
        let store = temp_store("moon");
        let mut sim = build_simulation(&new_orbits_config(1, "function on_tick(t) end"));
        sim.advance(20);
        let meta = store.save("Sun and Earth", &sim, TEMPLATE_ORBITS).unwrap();
        assert_eq!(meta.template, TEMPLATE_ORBITS);
        assert_eq!(meta.summary, "3 bodies");

        let (_, snap) = store.load("Sun and Earth").unwrap();
        assert_eq!(template_of(&snap.config), TEMPLATE_ORBITS);
        let spec = new_script_spec(store.next_experiment_id("Sun and Earth").unwrap(), "Remove the Moon", "function on_tick(t) remove_body('Moon') end", "orbits.distance_km.Earth", 1461);
        let result = run_experiment(&snap, &spec).unwrap();
        let (a, b) = (&result.series[0], &result.series[1]);
        assert_eq!(a.points.len(), b.points.len());
        assert!(a.points.iter().zip(&b.points).any(|(p, q)| p[1] != q[1]), "removing the Moon should change Earth's path");
        // Small next to the 150 million km orbit, but real: the Earth keeps its own ~12 m/s motion
        // around the Earth-Moon centre of mass, so without the Moon it drifts onto a slightly
        // different orbit (by how much depends on where the Moon was when it vanished).
        let worst = a.points.iter().zip(&b.points).map(|(p, q)| (p[1] - q[1]).abs()).fold(0.0, f64::max);
        assert!(worst > 1000.0 && worst < 500_000.0, "difference {worst} km");
        // exact repeat
        let again = run_experiment(&snap, &spec).unwrap();
        assert_eq!(again.series[1].points, b.points);

        store.save_experiment("Sun and Earth", &Experiment { spec, result }, &snap).unwrap();
        let back = store.list_experiments("Sun and Earth");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].spec.alt_script.as_deref(), Some("function on_tick(t) remove_body('Moon') end"));
    }
}

/// One check like `people who know >= 100` against the final measurements.
fn check_expectation(spec: &str, m: &std::collections::BTreeMap<String, f64>) -> serde_json::Value {
    use serde_json::json;
    let spec = spec.trim();
    for op in [">=", "<=", "==", "!=", ">", "<"] {
        if let Some(i) = spec.rfind(op) {
            let (name, rhs) = (spec[..i].trim(), spec[i + op.len()..].trim());
            let Ok(want) = rhs.parse::<f64>() else { continue };
            let Some(&got) = m.get(name) else {
                return json!({"check": spec, "pass": false, "why": format!("no measurement named '{name}'; have: {}", m.keys().cloned().collect::<Vec<_>>().join(", "))});
            };
            let pass = match op {
                ">=" => got >= want,
                "<=" => got <= want,
                "==" => got == want,
                "!=" => got != want,
                ">" => got > want,
                _ => got < want,
            };
            return json!({"check": spec, "pass": pass, "got": got});
        }
    }
    json!({"check": spec, "pass": false, "why": "write a check like: name >= 10 (operators: >= <= == != > <)"})
}

/// Build a universe from a script, run it, and describe what happened as JSON: the build-and-test
/// loop for people and coding agents. `expects` are checks like `people who know >= 100`.
pub fn run_script_report(script: &str, seed: u64, ticks: u64, expects: &[String]) -> serde_json::Value {
    use serde_json::json;
    let t0 = std::time::Instant::now();
    let mut sim = build_simulation(&new_build_config(seed, script));
    let samples = 60u64.min(ticks.max(1));
    let every = (ticks / samples).max(1);
    let mut series: std::collections::BTreeMap<String, Vec<[f64; 2]>> = Default::default();
    let mut sample = |sim: &Simulation, series: &mut std::collections::BTreeMap<String, Vec<[f64; 2]>>| {
        let tick = sim.world().clock.tick as f64;
        for (k, v) in sim.collect_metrics() {
            series.entry(k).or_default().push([tick, v]);
        }
    };
    sample(&sim, &mut series);
    let mut done = 0;
    while done < ticks {
        let n = every.min(ticks - done);
        sim.advance(n);
        done += n;
        sample(&sim, &mut series);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (error, log) = simtara_script::status(sim.world());
    let metrics = sim.collect_metrics();
    let (mut kinds, mut relations, mut rules, mut warnings) = (std::collections::BTreeMap::<String, usize>::new(), 0, vec![], vec![]);
    if let Some(t) = sim.world().get::<simtara_things::Things>() {
        for th in t.things.values() {
            *kinds.entry(th.kind.clone()).or_default() += 1;
        }
        relations = t.relations.len();
        rules = t.rules.iter().map(|r| json!({"name": r.name, "kind": r.kind, "enabled": r.enabled, "error": r.error})).collect();
        warnings = t.warnings();
    }
    // The same script and seed must give the same universe.
    let mut again = build_simulation(&new_build_config(seed, script));
    again.advance(ticks);
    let repeatable = match (sim.snapshot(), again.snapshot()) {
        (Ok(a), Ok(b)) => a.state == b.state,
        _ => false,
    };
    let checks: Vec<serde_json::Value> = expects.iter().map(|e| check_expectation(e, &metrics)).collect();
    let rule_broke = rules.iter().any(|r: &serde_json::Value| !r["error"].is_null());
    let ok = error.is_none() && !rule_broke && checks.iter().all(|c| c["pass"] == json!(true));
    json!({
        "ok": ok, "error": error, "ticks": ticks, "seed": seed, "ms": ms, "repeatable": repeatable,
        "log": log.iter().take(100).collect::<Vec<_>>(),
        "metrics": metrics, "series": series,
        "things": {"by_kind": kinds, "relations": relations},
        "rules": rules, "warnings": warnings, "checks": checks,
    })
}

#[cfg(test)]
mod build_experiment_tests {
    use super::*;

    #[test]
    fn a_control_can_be_changed_for_the_second_run_without_touching_the_script() {
        let base = "function setup()
  param{name = \"step\", value = 1, min = 0, max = 5}
  spawn{kind = \"p\", props = {v = 0}}
  add_rule{name = \"up\", kind = \"p\", set = {v = \"v + step\"}}
  define_metric{name = \"v\", kind = \"p\", prop = \"v\", op = \"sum\"}
end
";
        let start = build_simulation(&new_build_config(1, base)).snapshot().unwrap();
        let mut spec = new_script_spec("e".into(), "step 3", base, "v", 10);
        spec.param_overrides.insert("step".into(), 3.0);
        let r = run_experiment(&start, &spec).unwrap();
        assert_eq!(r.series[0].points.last().unwrap()[1], 10.0);
        assert_eq!(r.series[1].points.last().unwrap()[1], 30.0);
    }

    #[test]
    fn an_experiment_refuses_a_control_the_role_does_not_own() {
        let base = "function setup()
  param{name = \"rain\", value = 0, min = 0, max = 1, owner = \"nature\"}
  spawn{kind = \"p\", props = {v = 0}}
  define_metric{name = \"v\", kind = \"p\", prop = \"v\", op = \"sum\"}
end
";
        let mut sim = build_simulation(&new_build_config(1, base));
        sim.advance(1);
        let start = sim.snapshot().unwrap();
        let mut spec = new_script_spec("e".into(), "rain", base, "v", 5);
        spec.param_overrides.insert("rain".into(), 1.0);
        spec.role = "firefighters".into();
        assert!(run_experiment(&start, &spec).unwrap_err().contains("belongs to 'nature'"));
        spec.role = "nature".into();
        assert!(run_experiment(&start, &spec).is_ok());
    }

    #[test]
    fn a_changed_script_is_compared_with_the_original_from_tick_zero() {
        let base = "function setup()
  for i = 1, 10 do spawn{kind = \"p\", props = {v = 0}} end
  add_rule{name = \"up\", kind = \"p\", set = {v = \"v + 1\"}}
  define_metric{name = \"total\", kind = \"p\", prop = \"v\", op = \"sum\"}
end
";
        let alt = base.replace("v + 1", "v + 2");
        let mut sim = build_simulation(&new_build_config(3, base));
        sim.advance(17); // the universe has already been running; the experiment still starts at 0
        let start = sim.snapshot().unwrap();
        let spec = new_script_spec("e1".into(), "faster", &alt, "total", 10);
        let r = run_experiment(&start, &spec).unwrap();
        assert_eq!(r.series[0].points.last().unwrap()[1], 100.0);
        assert_eq!(r.series[1].points.last().unwrap()[1], 200.0);
        assert_eq!(r.record.start_tick, 0);
    }
}

#[cfg(test)]
mod script_report_tests {
    use super::*;

    const S: &str = "function setup()\n  for i = 1, 20 do spawn{kind = \"p\", props = {v = i}} end\n  define_metric{name = \"total v\", kind = \"p\", prop = \"v\", op = \"sum\"}\n  add_rule{name = \"grow\", kind = \"p\", set = {v = \"v + 1\"}}\nend\n";

    #[test]
    fn a_script_reports_checks_and_repeats() {
        let r = run_script_report(S, 1, 10, &["total v == 410".into(), "total v > 1000".into(), "nope > 1".into()]);
        assert_eq!(r["error"], serde_json::Value::Null);
        assert_eq!(r["repeatable"], serde_json::json!(true));
        assert_eq!(r["checks"][0]["pass"], serde_json::json!(true), "{r}");
        assert_eq!(r["checks"][1]["pass"], serde_json::json!(false));
        assert!(r["checks"][2]["why"].as_str().unwrap().contains("total v"));
        assert_eq!(r["ok"], serde_json::json!(false));
        let bad = run_script_report("function setup() oops( end", 1, 3, &[]);
        assert_eq!(bad["ok"], serde_json::json!(false));
        assert!(bad["error"].is_string());
    }
}
