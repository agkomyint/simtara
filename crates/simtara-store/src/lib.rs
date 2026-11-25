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

/// Parameters of the particles template that an experiment may change: (field, label).
pub const PARAMETERS: [(&str, &str); 5] =
    [("gravity", "gravity"), ("friction", "friction"), ("rmax", "range"), ("repel_core", "core repel"), ("coulomb", "charge")];

pub const DEFAULT_METRIC: &str = "particles.kinetic_energy";

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
    /// Field name inside the template's laws, e.g. "gravity".
    pub parameter: String,
    pub base_value: f64,
    pub alt_value: f64,
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

/// A fresh simulation of the particles template (CPU only; clients may add more modules).
pub fn build_simulation(config: &Config) -> Simulation {
    let mut sim = Simulation::new(config.clone());
    sim.add_module(simtara_particles::ParticleModule);
    sim
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
        let count = sim.world().get::<simtara_particles::Particles>().map_or(0, |p| p.len());
        let meta = UniverseMeta {
            schema_version: SCHEMA_VERSION,
            name,
            template: template.to_string(),
            summary: format!("{count} particles"),
            seed: snap.config.seed,
            tick: snap.tick,
            created,
            updated: now(),
        };
        let text = serde_json::to_vec(&snap).map_err(|e| e.to_string())?;
        write_atomic(&dir.join("state.snap.json"), &text)?;
        write_json(&dir.join("universe.json"), &meta)?;
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

fn run_arm(start: &Snapshot, spec: &ExperimentSpec, label: &str, value: f64) -> Result<Series, String> {
    let mut sim = from_snapshot(start)?;
    // CPU on purpose: CPU runs repeat exactly, GPU runs are only approximately repeatable.
    sim.compute_mut().set_requested(Backend::Cpu);
    set_parameter(&mut sim, &spec.parameter, value)?;
    let every = spec.sample_every.max(1);
    let sample = |sim: &Simulation| -> Result<[f64; 2], String> {
        let v = *sim.collect_metrics().get(&spec.metric).ok_or_else(|| format!("unknown metric '{}'", spec.metric))?;
        Ok([sim.world().clock.tick as f64, v])
    };
    let mut points = vec![sample(&sim)?];
    let mut done = 0;
    while done < spec.ticks {
        let n = every.min(spec.ticks - done);
        sim.advance(n);
        done += n;
        points.push(sample(&sim)?);
    }
    Ok(Series { label: label.to_string(), value, points })
}

/// Run both arms (original value and changed value) from the same start state.
pub fn run_experiment(start: &Snapshot, spec: &ExperimentSpec) -> Result<ExperimentResult, String> {
    let t0 = Instant::now();
    let series = vec![run_arm(start, spec, "original", spec.base_value)?, run_arm(start, spec, "changed", spec.alt_value)?];
    Ok(ExperimentResult {
        schema_version: SCHEMA_VERSION,
        series,
        record: RunRecord {
            engine_version: ENGINE_VERSION.to_string(),
            seed: start.config.seed,
            backend: "cpu".into(),
            start_tick: start.tick,
            millis: t0.elapsed().as_millis() as u64,
        },
    })
}

/// Fill in a spec for "change `parameter` from its current value to `alt_value`".
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
        ticks: ticks.max(1),
        sample_every: (ticks / 200).max(1),
        metric: DEFAULT_METRIC.to_string(),
        created: now(),
    })
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
}
