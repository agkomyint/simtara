use crate::compute::{Backend, Compute};
use crate::config::Config;
use crate::world::World;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Ordered phases of a tick. Within a stage, systems run in registration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    PreUpdate,
    Update,
    PostUpdate,
}

pub struct Context<'a> {
    pub world: &'a mut World,
    /// Timestep for this tick, in simulated seconds.
    pub dt: f64,
    pub tick: u64,
    pub compute: &'a Compute,
}

/// A rule/behavior. Runs once per tick.
pub trait System: Send {
    fn name(&self) -> &str;
    fn run(&mut self, ctx: &mut Context);
}

struct FnSystem<F: FnMut(&mut Context) + Send> {
    name: String,
    f: F,
}
impl<F: FnMut(&mut Context) + Send> System for FnSystem<F> {
    fn name(&self) -> &str {
        &self.name
    }
    fn run(&mut self, ctx: &mut Context) {
        (self.f)(ctx)
    }
}

#[derive(Default)]
struct Schedule {
    systems: Vec<(Stage, Box<dyn System>)>,
}

/// Handed to a module so it can install state and systems. This is the ONLY
/// way features enter the runtime.
pub struct Builder<'a> {
    pub world: &'a mut World,
    pub config: &'a Config,
    metrics: &'a mut Vec<(String, MetricFn)>,
    metric_sources: &'a mut Vec<MetricSource>,
    /// Modules that provide a compute backend (e.g. GPU) register it here.
    pub compute: &'a mut Compute,
    schedule: &'a mut Schedule,
}

/// A named measurement of the world (population, energy, ...). Agents and tools read these.
pub type MetricFn = Box<dyn Fn(&World) -> f64 + Send>;

/// Measurements whose names are only known while the universe runs (for example ones a script defined).
pub type MetricSource = Box<dyn Fn(&World) -> Vec<(String, f64)> + Send>;

impl Builder<'_> {
    /// Register a metric under `name` (convention: `module.metric`).
    pub fn add_metric(&mut self, name: &str, f: impl Fn(&World) -> f64 + Send + 'static) {
        self.metrics.push((name.to_string(), Box::new(f)));
    }
    /// Register a provider of measurements whose set can change while running.
    pub fn add_metric_source(&mut self, f: impl Fn(&World) -> Vec<(String, f64)> + Send + 'static) {
        self.metric_sources.push(Box::new(f));
    }
    pub fn add_system(&mut self, stage: Stage, s: impl System + 'static) {
        self.schedule.systems.push((stage, Box::new(s)));
        // stable sort keeps registration order inside each stage
        self.schedule.systems.sort_by_key(|(st, _)| *st);
    }
    pub fn add_fn(&mut self, name: &str, stage: Stage, f: impl FnMut(&mut Context) + Send + 'static) {
        self.add_system(stage, FnSystem { name: name.to_string(), f });
    }
}

/// A pluggable capability: particles, chemistry, biology, agents, ...
pub trait Module: Send {
    fn name(&self) -> &str;
    fn build(&self, b: &mut Builder);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    Running,
    Paused,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub tick: u64,
    pub time: f64,
    pub config: Config,
    pub state: BTreeMap<String, Value>,
}

pub struct Simulation {
    config: Config,
    modules: Vec<Box<dyn Module>>,
    world: World,
    schedule: Schedule,
    metrics: Vec<(String, MetricFn)>,
    metric_sources: Vec<MetricSource>,
    compute: Compute,
    state: RunState,
    speed: f64,
    accum: f64,
}

impl Simulation {
    pub fn new(config: Config) -> Self {
        let compute = Compute::new(config.backend);
        Simulation {
            config,
            modules: Vec::new(),
            world: World::new(),
            schedule: Schedule::default(),
            metrics: Vec::new(),
            metric_sources: Vec::new(),
            compute,
            state: RunState::Paused,
            speed: 1.0,
            accum: 0.0,
        }
    }

    /// Install a module. The core is untouched; the module registers its own state and systems.
    pub fn add_module(&mut self, m: impl Module + 'static) -> &mut Self {
        let m: Box<dyn Module> = Box::new(m);
        self.build_module(&*m);
        self.modules.push(m);
        self
    }
    fn build_module(&mut self, m: &dyn Module) {
        let mut b = Builder { world: &mut self.world, config: &self.config, compute: &mut self.compute, schedule: &mut self.schedule, metrics: &mut self.metrics, metric_sources: &mut self.metric_sources };
        m.build(&mut b);
    }

    /// Rebuild the universe from its config (tick 0). Keeps run state and speed.
    pub fn reset(&mut self) {
        self.world = World::new();
        self.schedule = Schedule::default();
        self.metrics.clear();
        self.metric_sources.clear();
        self.accum = 0.0;
        let modules = std::mem::take(&mut self.modules);
        for m in &modules {
            self.build_module(&**m);
        }
        self.modules = modules;
    }

    // ---- time ----
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let tick = self.world.clock.tick;
        for (_, s) in self.schedule.systems.iter_mut() {
            let mut ctx = Context { world: &mut self.world, dt, tick, compute: &self.compute };
            s.run(&mut ctx);
        }
        self.world.clear_events();
        self.world.clock.tick += 1;
        self.world.clock.time += dt;
    }
    pub fn advance(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }
    /// Call once per rendered frame: steps according to run state and speed (0.1x..10x).
    pub fn run_frame(&mut self) {
        if self.state != RunState::Running {
            return;
        }
        self.accum += self.speed;
        let mut n = 0;
        while self.accum >= 1.0 && n < 64 {
            self.step();
            self.accum -= 1.0;
            n += 1;
        }
        self.accum = self.accum.min(1.0);
    }

    // ---- control ----
    pub fn play(&mut self) {
        self.state = RunState::Running;
    }
    pub fn pause(&mut self) {
        self.state = RunState::Paused;
    }
    pub fn toggle(&mut self) {
        self.state = if self.state == RunState::Running { RunState::Paused } else { RunState::Running };
    }
    pub fn run_state(&self) -> RunState {
        self.state
    }
    pub fn speed(&self) -> f64 {
        self.speed
    }
    pub fn set_speed(&mut self, s: f64) {
        self.speed = s.clamp(0.1, 10.0);
    }

    // ---- access ----
    pub fn world(&self) -> &World {
        &self.world
    }
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }
    pub fn compute_mut(&mut self) -> &mut Compute {
        &mut self.compute
    }
    /// The backend that will actually run (requested one if available, else CPU).
    /// The backend the user asked for (may not be available).
    pub fn compute_requested(&self) -> Backend {
        self.compute.requested()
    }
    pub fn backend(&self) -> Backend {
        self.compute.active()
    }
    pub fn system_names(&self) -> Vec<&str> {
        self.schedule.systems.iter().map(|(_, s)| s.name()).collect()
    }

    /// Evaluate every registered metric on the current world.
    pub fn collect_metrics(&self) -> BTreeMap<String, f64> {
        let mut out: BTreeMap<String, f64> = self.metrics.iter().map(|(n, f)| (n.clone(), f(&self.world))).collect();
        for src in &self.metric_sources {
            out.extend(src(&self.world));
        }
        out
    }
    pub fn metric_names(&self) -> Vec<String> {
        self.collect_metrics().into_keys().collect()
    }

    // ---- persistence ----
    pub fn snapshot(&self) -> Result<Snapshot, String> {
        Ok(Snapshot {
            version: 1,
            tick: self.world.clock.tick,
            time: self.world.clock.time,
            config: self.config.clone(),
            state: self.world.save_persistent()?,
        })
    }
    pub fn restore(&mut self, s: &Snapshot) -> Result<(), String> {
        self.world.load_persistent(&s.state)?;
        self.world.clock.tick = s.tick;
        self.world.clock.time = s.time;
        self.accum = 0.0;
        Ok(())
    }
    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let json = serde_json::to_string(&self.snapshot()?).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }
    pub fn read_snapshot(path: impl AsRef<Path>) -> Result<Snapshot, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Default)]
    struct Counter(u64);

    /// A module defined entirely outside the core's knowledge.
    struct CounterModule;
    impl Module for CounterModule {
        fn name(&self) -> &str {
            "counter"
        }
        fn build(&self, b: &mut Builder) {
            b.world.insert_persistent("counter", Counter(0));
            b.add_fn("counter.inc", Stage::Update, |ctx| {
                ctx.world.get_mut::<Counter>().unwrap().0 += 1;
                ctx.world.emit(7u32);
            });
        }
    }

    #[test]
    fn module_plugs_in_and_time_advances() {
        let mut sim = Simulation::new(Config::default());
        sim.add_module(CounterModule);
        sim.advance(10);
        assert_eq!(sim.world().get::<Counter>().unwrap().0, 10);
        assert_eq!(sim.world().clock.tick, 10);
        assert!((sim.world().clock.time - 10.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn stages_order_systems() {
        struct M;
        impl Module for M {
            fn name(&self) -> &str {
                "m"
            }
            fn build(&self, b: &mut Builder) {
                b.add_fn("late", Stage::PostUpdate, |_| {});
                b.add_fn("early", Stage::PreUpdate, |_| {});
                b.add_fn("mid", Stage::Update, |_| {});
            }
        }
        let mut sim = Simulation::new(Config::default());
        sim.add_module(M);
        assert_eq!(sim.system_names(), vec!["early", "mid", "late"]);
    }

    #[test]
    fn pause_resume_reset() {
        let mut sim = Simulation::new(Config::default());
        sim.add_module(CounterModule);
        sim.run_frame();
        assert_eq!(sim.world().clock.tick, 0, "paused sim must not advance");
        sim.play();
        sim.run_frame();
        sim.run_frame();
        assert_eq!(sim.world().clock.tick, 2);
        sim.reset();
        assert_eq!(sim.world().clock.tick, 0);
        assert_eq!(sim.world().get::<Counter>().unwrap().0, 0);
    }

    #[test]
    fn snapshot_roundtrip() {
        let mut sim = Simulation::new(Config::default());
        sim.add_module(CounterModule);
        sim.advance(5);
        let snap = sim.snapshot().unwrap();
        sim.advance(5);
        sim.restore(&snap).unwrap();
        assert_eq!(sim.world().get::<Counter>().unwrap().0, 5);
        assert_eq!(sim.world().clock.tick, 5);
    }

    #[test]
    fn events_are_per_tick() {
        let mut w = World::new();
        w.emit(1u8);
        w.emit(2u8);
        assert_eq!(w.drain_events::<u8>(), vec![1, 2]);
        assert!(w.drain_events::<u8>().is_empty());
    }

    #[test]
    fn entities_generation_checked() {
        let mut w = World::new();
        let mut hp = crate::ComponentStore::<i32>::default();
        let a = w.spawn();
        hp.insert(a, 10);
        w.despawn(a);
        let b = w.spawn();
        assert_eq!(a.index, b.index);
        assert!(!w.is_alive(a));
        assert!(hp.get(b).is_none() && hp.get(a).is_some());
    }

    #[test]
    fn compute_falls_back_to_cpu() {
        let mut c = Compute::new(Backend::Gpu);
        assert_eq!(c.resolve(&[Backend::Gpu, Backend::Cpu]), Backend::Cpu);
        c.register(Backend::Gpu);
        assert_eq!(c.resolve(&[Backend::Gpu, Backend::Cpu]), Backend::Gpu);
    }
}
