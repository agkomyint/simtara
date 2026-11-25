//! SimTara control API.
//!
//! One interface for every client — the CLI, Studio, scripts, AI agents. A client sends a
//! JSON request and gets a JSON response; the `Engine` owns any number of named worlds and
//! drives the runtime. The runtime knows nothing about this crate.
//!
//! Request:  `{"id": 1, "cmd": "experiment.run", "args": {"name": "u1", "ticks": 500}}`
//! Response: `{"id": 1, "ok": true, "result": {...}}` or `{"id": 1, "ok": false, "error": "..."}`
//!
//! Send `{"cmd": "capabilities"}` to discover everything.

use serde::Deserialize;
use serde_json::{json, Map, Value};
use simtara_core::{Backend, Config, Simulation, Snapshot};
use simtara_particles::ParticleModule;
use std::collections::BTreeMap;
use std::time::Instant;

type Ctor = fn(Config) -> Simulation;

fn particles_ctor(cfg: Config) -> Simulation {
    let mut sim = Simulation::new(cfg);
    sim.add_module(ParticleModule);
    sim
}

struct SavedSnapshot {
    id: String,
    label: String,
    snap: Snapshot,
}

struct Session {
    module: String,
    sim: Simulation,
    snapshots: Vec<SavedSnapshot>,
    next_snapshot: u32,
}

#[derive(Deserialize)]
pub struct Request {
    #[serde(default)]
    pub id: Value,
    pub cmd: String,
    #[serde(default)]
    pub args: Value,
}

pub struct Engine {
    worlds: BTreeMap<String, Session>,
    modules: BTreeMap<String, Ctor>,
    use_gpu: bool,
    #[cfg(feature = "gpu")]
    gpu: Option<simtara_gpu::GpuModule>,
}

impl Engine {
    /// `use_gpu`: allow the GPU backend (initialised lazily, shared by all worlds).
    pub fn new(use_gpu: bool) -> Self {
        let mut modules: BTreeMap<String, Ctor> = BTreeMap::new();
        modules.insert("particles".into(), particles_ctor);
        Engine {
            worlds: BTreeMap::new(),
            modules,
            use_gpu,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Make another kind of world available to `world.create`.
    pub fn register_module(&mut self, name: &str, ctor: Ctor) {
        self.modules.insert(name.to_string(), ctor);
    }

    /// Handle one JSON line, return one JSON line.
    pub fn handle_line(&mut self, line: &str) -> String {
        match serde_json::from_str::<Request>(line) {
            Ok(req) => self.handle(req).to_string(),
            Err(e) => json!({"ok": false, "error": format!("invalid request: {e}")}).to_string(),
        }
    }

    pub fn handle(&mut self, req: Request) -> Value {
        let args = if req.args.is_null() { Value::Object(Map::new()) } else { req.args };
        match self.dispatch(&req.cmd, &args) {
            Ok(result) => json!({"id": req.id, "ok": true, "result": result}),
            Err(error) => json!({"id": req.id, "ok": false, "error": error}),
        }
    }

    fn dispatch(&mut self, cmd: &str, a: &Value) -> Result<Value, String> {
        match cmd {
            "capabilities" => Ok(self.capabilities()),
            "world.create" => self.world_create(a),
            "world.list" => Ok(self.world_list()),
            "world.delete" => {
                let name = str_arg(a, "name")?;
                self.worlds.remove(&name).ok_or_else(|| no_world(&name))?;
                Ok(json!({"deleted": name}))
            }
            "world.reset" => {
                let s = self.session(a)?;
                s.sim.reset();
                Ok(json!({"tick": s.sim.world().clock.tick}))
            }
            "world.configure" => self.world_configure(a),
            "backend.set" => {
                let b = parse_backend(&str_arg(a, "backend")?)?;
                let s = self.session(a)?;
                s.sim.compute_mut().set_requested(b);
                Ok(json!({"backend": format!("{:?}", s.sim.backend())}))
            }
            "state.observe" => self.state_observe(a),
            "metrics.collect" => {
                let s = self.session(a)?;
                Ok(json!({"tick": s.sim.world().clock.tick, "metrics": s.sim.collect_metrics()}))
            }
            "experiment.run" => self.experiment_run(a),
            "experiment.branch" => self.experiment_branch(a),
            "snapshot.save" => {
                let s = self.session(a)?;
                let label = a.get("label").and_then(Value::as_str).unwrap_or("").to_string();
                let snap = s.sim.snapshot()?;
                s.next_snapshot += 1;
                let id = format!("s{}", s.next_snapshot);
                let tick = snap.tick;
                s.snapshots.push(SavedSnapshot { id: id.clone(), label, snap });
                Ok(json!({"id": id, "tick": tick}))
            }
            "snapshot.restore" => {
                let id = str_arg(a, "id")?;
                let s = self.session(a)?;
                let snap = s.snapshots.iter().find(|x| x.id == id).ok_or(format!("no snapshot '{id}'"))?.snap.clone();
                s.sim.restore(&snap)?;
                Ok(json!({"restored": id, "tick": snap.tick}))
            }
            "snapshot.list" => {
                let s = self.session(a)?;
                Ok(json!(s.snapshots.iter().map(|x| json!({"id": x.id, "label": x.label, "tick": x.snap.tick})).collect::<Vec<_>>()))
            }
            other => Err(format!("unknown command '{other}'. Send {{\"cmd\":\"capabilities\"}} for the list.")),
        }
    }

    fn session(&mut self, a: &Value) -> Result<&mut Session, String> {
        let name = str_arg(a, "name")?;
        self.worlds.get_mut(&name).ok_or_else(|| no_world(&name))
    }

    #[cfg(feature = "gpu")]
    fn gpu_module(&mut self) -> Option<simtara_gpu::GpuModule> {
        if !self.use_gpu {
            return None;
        }
        let g = self.gpu.get_or_insert_with(simtara_gpu::GpuModule::new);
        g.available().then(|| g.clone())
    }

    /// Build a world from a config (used by create and branch).
    fn build(&mut self, module: &str, config: Config, backend: Backend) -> Result<Session, String> {
        let ctor = *self
            .modules
            .get(module)
            .ok_or_else(|| format!("unknown module '{module}' (available: {:?})", self.modules.keys().collect::<Vec<_>>()))?;
        let mut sim = ctor(config);
        #[cfg(feature = "gpu")]
        if backend == Backend::Gpu {
            if let Some(g) = self.gpu_module() {
                sim.add_module(g);
            }
        }
        sim.compute_mut().set_requested(backend);
        Ok(Session { module: module.to_string(), sim, snapshots: Vec::new(), next_snapshot: 0 })
    }

    fn summary(name: &str, s: &Session) -> Value {
        json!({
            "name": name,
            "module": s.module,
            "tick": s.sim.world().clock.tick,
            "backend": format!("{:?}", s.sim.backend()),
            "metrics_available": s.sim.metric_names(),
            "systems": s.sim.system_names(),
        })
    }

    fn world_create(&mut self, a: &Value) -> Result<Value, String> {
        let name = str_arg(a, "name")?;
        if self.worlds.contains_key(&name) {
            return Err(format!("world '{name}' already exists"));
        }
        let module = a.get("module").and_then(Value::as_str).unwrap_or("particles").to_string();
        let mut cfg = Config::default();
        if let Some(seed) = a.get("seed").and_then(Value::as_u64) {
            cfg.seed = seed;
        }
        if let Some(dt) = a.get("dt").and_then(Value::as_f64) {
            if dt <= 0.0 {
                return Err("dt must be > 0".into());
            }
            cfg.dt = dt;
        }
        if let Some(c) = a.get("config") {
            cfg.modules.insert(module.clone(), c.clone());
        }
        let backend = parse_backend(a.get("backend").and_then(Value::as_str).unwrap_or("auto"))?;
        let session = self.build(&module, cfg, backend)?;
        let out = Self::summary(&name, &session);
        self.worlds.insert(name, session);
        Ok(out)
    }

    fn world_list(&self) -> Value {
        json!(self
            .worlds
            .iter()
            .map(|(n, s)| json!({"name": n, "module": s.module, "tick": s.sim.world().clock.tick, "backend": format!("{:?}", s.sim.backend())}))
            .collect::<Vec<_>>())
    }

    /// Patch any saved resource with a JSON merge patch. Unknown field names are rejected
    /// (the particle structs default missing fields, so typos would otherwise do nothing).
    fn world_configure(&mut self, a: &Value) -> Result<Value, String> {
        let resource = str_arg(a, "resource")?;
        let patch = a.get("patch").ok_or("missing 'patch'")?;
        let s = self.session(a)?;
        let state = s.sim.world().save_persistent()?;
        let mut current = state.get(&resource).cloned().ok_or_else(|| {
            format!("no resource '{resource}' (available: {:?})", state.keys().collect::<Vec<_>>())
        })?;
        check_keys(patch, &current, &resource)?;
        merge(&mut current, patch);
        let mut update = BTreeMap::new();
        update.insert(resource.clone(), current);
        s.sim.world_mut().load_persistent(&update)?;
        let now = s.sim.world().save_persistent()?;
        Ok(json!({"resource": resource, "value": truncate(&now[&resource], 8)}))
    }

    fn state_observe(&mut self, a: &Value) -> Result<Value, String> {
        let limit = a.get("limit").and_then(Value::as_u64).unwrap_or(5) as usize;
        let only = a.get("resource").and_then(Value::as_str).map(str::to_string);
        let s = self.session(a)?;
        let state = s.sim.world().save_persistent()?;
        if let Some(r) = &only {
            if !state.contains_key(r) {
                return Err(format!("no resource '{r}' (available: {:?})", state.keys().collect::<Vec<_>>()));
            }
        }
        let resources: Map<String, Value> = state
            .iter()
            .filter(|(k, _)| only.as_ref().is_none_or(|o| o == *k))
            .map(|(k, v)| (k.clone(), truncate(v, limit)))
            .collect();
        let c = s.sim.world().clock;
        Ok(json!({"clock": {"tick": c.tick, "time": c.time}, "resources": resources}))
    }

    fn experiment_run(&mut self, a: &Value) -> Result<Value, String> {
        let ticks = a.get("ticks").and_then(Value::as_u64).ok_or("missing 'ticks' (integer)")?;
        let every = a.get("sample_every").and_then(Value::as_u64).unwrap_or((ticks / 10).max(1)).max(1);
        let s = self.session(a)?;
        let names: Vec<String> = match a.get("metrics").and_then(Value::as_array) {
            Some(list) => list.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            None => s.sim.metric_names().iter().map(|n| n.to_string()).collect(),
        };
        let mut sample_ticks = Vec::new();
        let mut series: BTreeMap<String, Vec<f64>> = names.iter().map(|n| (n.clone(), Vec::new())).collect();
        let sample = |sim: &Simulation, ticks_out: &mut Vec<u64>, series: &mut BTreeMap<String, Vec<f64>>| {
            let m = sim.collect_metrics();
            ticks_out.push(sim.world().clock.tick);
            for n in &names {
                series.get_mut(n).unwrap().push(m.get(n).copied().unwrap_or(f64::NAN));
            }
        };
        if let Some(bad) = names.iter().find(|n| !s.sim.metric_names().contains(n)) {
            return Err(format!("unknown metric '{bad}' (available: {:?})", s.sim.metric_names()));
        }
        let start = s.sim.world().clock.tick;
        sample(&s.sim, &mut sample_ticks, &mut series);
        let t0 = Instant::now();
        let mut done = 0;
        while done < ticks {
            let n = every.min(ticks - done);
            s.sim.advance(n);
            done += n;
            sample(&s.sim, &mut sample_ticks, &mut series);
        }
        let el = t0.elapsed().as_secs_f64();
        Ok(json!({
            "from_tick": start,
            "to_tick": s.sim.world().clock.tick,
            "backend": format!("{:?}", s.sim.backend()),
            "elapsed_ms": (el * 1000.0).round(),
            "series": {"ticks": sample_ticks, "metrics": series},
            "final": s.sim.collect_metrics(),
        }))
    }

    fn experiment_branch(&mut self, a: &Value) -> Result<Value, String> {
        let new_name = str_arg(a, "new_name")?;
        if self.worlds.contains_key(&new_name) {
            return Err(format!("world '{new_name}' already exists"));
        }
        let (module, backend, snap) = {
            let s = self.session(a)?;
            let snap = match a.get("snapshot").and_then(Value::as_str) {
                Some(id) => s.snapshots.iter().find(|x| x.id == id).ok_or(format!("no snapshot '{id}'"))?.snap.clone(),
                None => s.sim.snapshot()?,
            };
            (s.module.clone(), s.sim.compute_requested(), snap)
        };
        let mut session = self.build(&module, snap.config.clone(), backend)?;
        session.sim.restore(&snap)?;
        let out = Self::summary(&new_name, &session);
        self.worlds.insert(new_name, session);
        Ok(out)
    }

    fn capabilities(&self) -> Value {
        #[cfg(feature = "gpu")]
        let gpu = self.use_gpu;
        #[cfg(not(feature = "gpu"))]
        let gpu = false;
        json!({
            "modules": self.modules.keys().collect::<Vec<_>>(),
            "gpu_enabled": gpu,
            "commands": {
                "world.create": "{name, module?='particles', seed?, dt?, backend?='auto'|'gpu'|'cpu', config?: module config}",
                "world.list": "{}",
                "world.delete": "{name}",
                "world.reset": "{name}  rebuild from its original config, tick 0",
                "world.configure": "{name, resource, patch}  JSON merge-patch a saved resource, e.g. resource='particle_params', patch={gravity:2}; unknown fields are rejected",
                "backend.set": "{name, backend}",
                "state.observe": "{name, resource?, limit?=5}  saved state; long arrays are truncated to {_len,_head}",
                "metrics.collect": "{name}  current value of every metric",
                "experiment.run": "{name, ticks, sample_every?, metrics?}  advance time, returns metric time-series",
                "experiment.branch": "{name, new_name, snapshot?}  copy a world (current state or a snapshot) to explore an alternative",
                "snapshot.save": "{name, label?} -> id",
                "snapshot.restore": "{name, id}  rewind",
                "snapshot.list": "{name}"
            },
            "notes": "Worlds are independent. Use experiment.branch + world.configure to compare alternatives from the same starting point."
        })
    }
}

fn str_arg(a: &Value, key: &str) -> Result<String, String> {
    a.get(key).and_then(Value::as_str).map(String::from).ok_or_else(|| format!("missing '{key}' (string)"))
}

fn no_world(name: &str) -> String {
    format!("no world named '{name}'")
}

fn parse_backend(s: &str) -> Result<Backend, String> {
    match s {
        "cpu" => Ok(Backend::Cpu),
        "gpu" | "auto" => Ok(Backend::Gpu), // falls back to CPU if no GPU
        other => Err(format!("unknown backend '{other}' (use 'auto', 'gpu' or 'cpu')")),
    }
}

fn merge(base: &mut Value, patch: &Value) {
    match (&mut *base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                merge(b.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        _ => *base = patch.clone(),
    }
}

fn check_keys(patch: &Value, original: &Value, path: &str) -> Result<(), String> {
    if let (Value::Object(p), Value::Object(o)) = (patch, original) {
        for (k, v) in p {
            match o.get(k) {
                None => return Err(format!("unknown field '{path}.{k}' (valid: {:?})", o.keys().collect::<Vec<_>>())),
                Some(ov) => check_keys(v, ov, &format!("{path}.{k}"))?,
            }
        }
    }
    Ok(())
}

fn truncate(v: &Value, limit: usize) -> Value {
    match v {
        Value::Array(a) if a.len() > limit => {
            json!({"_len": a.len(), "_head": a[..limit].iter().map(|x| truncate(x, limit)).collect::<Vec<_>>()})
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| truncate(x, limit)).collect()),
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), truncate(x, limit))).collect()),
        _ => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(e: &mut Engine, cmd: &str, args: Value) -> Value {
        let r = e.handle(Request { id: Value::Null, cmd: cmd.into(), args });
        assert_eq!(r["ok"], true, "{cmd} failed: {r}");
        r["result"].clone()
    }
    fn fail(e: &mut Engine, cmd: &str, args: Value) -> String {
        let r = e.handle(Request { id: Value::Null, cmd: cmd.into(), args });
        assert_eq!(r["ok"], false, "{cmd} should fail: {r}");
        r["error"].as_str().unwrap().to_string()
    }
    fn create(e: &mut Engine, name: &str) {
        call(e, "world.create", json!({"name": name, "seed": 7, "backend": "cpu", "config": {"count": 600}}));
    }

    #[test]
    fn agent_loop_create_run_observe_branch_compare() {
        let mut e = Engine::new(false);
        create(&mut e, "base");

        let run = call(&mut e, "experiment.run", json!({"name": "base", "ticks": 100, "sample_every": 25}));
        assert_eq!(run["series"]["ticks"], json!([0, 25, 50, 75, 100]));
        assert_eq!(run["series"]["metrics"]["particles.count"].as_array().unwrap().len(), 5);

        // branch, change one law in the branch only, run both
        call(&mut e, "experiment.branch", json!({"name": "base", "new_name": "hot"}));
        call(&mut e, "world.configure", json!({"name": "hot", "resource": "particle_params", "patch": {"gravity": 4.0}}));
        let a = call(&mut e, "experiment.run", json!({"name": "base", "ticks": 60}));
        let b = call(&mut e, "experiment.run", json!({"name": "hot", "ticks": 60}));
        assert_ne!(a["final"]["particles.kinetic_energy"], b["final"]["particles.kinetic_energy"]);

        let obs = call(&mut e, "state.observe", json!({"name": "base", "resource": "particles", "limit": 3}));
        assert_eq!(obs["resources"]["particles"]["x"]["_len"], 600);
        assert_eq!(obs["clock"]["tick"], 160);
    }

    #[test]
    fn snapshot_rewind_is_exact() {
        let mut e = Engine::new(false);
        create(&mut e, "u");
        call(&mut e, "experiment.run", json!({"name": "u", "ticks": 20}));
        let id = call(&mut e, "snapshot.save", json!({"name": "u", "label": "t20"}))["id"].as_str().unwrap().to_string();
        let first = call(&mut e, "experiment.run", json!({"name": "u", "ticks": 30}))["final"].clone();
        call(&mut e, "snapshot.restore", json!({"name": "u", "id": id}));
        assert_eq!(call(&mut e, "metrics.collect", json!({"name": "u"}))["tick"], 20);
        let second = call(&mut e, "experiment.run", json!({"name": "u", "ticks": 30}))["final"].clone();
        assert_eq!(first, second, "deterministic replay after restore");
    }

    #[test]
    fn typos_and_bad_input_give_helpful_errors() {
        let mut e = Engine::new(false);
        create(&mut e, "u");
        let err = fail(&mut e, "world.configure", json!({"name": "u", "resource": "particle_params", "patch": {"gravty": 2}}));
        assert!(err.contains("unknown field") && err.contains("gravity"), "{err}");
        assert!(fail(&mut e, "world.configure", json!({"name": "u", "resource": "nope", "patch": {}})).contains("available"));
        assert!(fail(&mut e, "experiment.run", json!({"name": "ghost", "ticks": 1})).contains("no world"));
        assert!(fail(&mut e, "world.create", json!({"name": "u"})).contains("already exists"));
        assert!(fail(&mut e, "world.create", json!({"name": "x", "module": "chem"})).contains("unknown module"));
        assert!(fail(&mut e, "experiment.run", json!({"name": "u", "ticks": 5, "metrics": ["bogus"]})).contains("unknown metric"));
        // a patch that makes the resource invalid must not corrupt the world
        fail(&mut e, "world.configure", json!({"name": "u", "resource": "particle_params", "patch": {"gravity": "lots"}}));
        call(&mut e, "experiment.run", json!({"name": "u", "ticks": 5}));
    }

    #[test]
    fn json_line_protocol() {
        let mut e = Engine::new(false);
        let r: Value = serde_json::from_str(&e.handle_line(r#"{"id":"a","cmd":"capabilities"}"#)).unwrap();
        assert_eq!(r["id"], "a");
        assert!(r["result"]["commands"]["experiment.run"].is_string());
        let bad: Value = serde_json::from_str(&e.handle_line("not json")).unwrap();
        assert_eq!(bad["ok"], false);
    }
}
