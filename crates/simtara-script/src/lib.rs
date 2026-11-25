//! Lua scripting for universes.
//!
//! A script is a piece of Lua text saved with the universe. Once per tick the
//! runtime calls its `on_tick(tick)` function, and the script can ask the
//! universe to do things through *commands* that modules offer
//! (`remove_body("Moon")`, `set_mass("Moon", 1e22)`, ...):
//!
//! ```lua
//! function on_tick(t)
//!   if t == 1461 then remove_body("Moon") end   -- after one year
//! end
//! ```
//!
//! Rules of the road:
//! - Top-level code should only define functions (it runs again whenever the script is loaded).
//! - The script has no files, no network, no clock and no randomness, so a run repeats exactly.
//! - Each call may use a limited number of Lua instructions; a runaway loop is stopped
//!   and reported instead of freezing the run.
//! - Lua is for *events and setup*. Heavy per-particle math belongs in compiled modules.
//!
//! Knows nothing about particles or planets: everything it can do comes from the
//! commands registered with `World::register_command`.

use mlua::serde::SerializeOptions;
use mlua::{HookTriggers, Lua, LuaOptions, LuaSerdeExt, StdLib, Value as LuaValue, VmState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use simtara_core::{Builder, Context, Module, Stage, System, World};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const MODULE: &str = "script";

/// How many Lua instructions one call may use (about a millisecond or two of work per 100k).
const BUDGET: u64 = 2_000_000;
const HOOK_EVERY: u32 = 1000;
const MEMORY_LIMIT: usize = 64 * 1024 * 1024;
const LOG_KEEP: usize = 50;

/// Script settings in the universe config.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScriptConfig {
    pub source: String,
}

/// The script text and what happened the last time it ran. The text is saved in snapshots.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScriptState {
    pub source: String,
    /// Set when the script failed; the script then stays off until its text changes.
    #[serde(skip)]
    pub error: Option<String>,
    /// Lines the script printed with `log(...)`, newest last.
    #[serde(skip)]
    pub log: Vec<String>,
}

/// Replace the script of a running universe.
pub fn set_source(world: &mut World, source: &str) -> Result<(), String> {
    let st = world.get_mut::<ScriptState>().ok_or("this universe has no scripting")?;
    st.source = source.to_string();
    st.error = None;
    Ok(())
}

pub fn source(world: &World) -> String {
    world.get::<ScriptState>().map(|s| s.source.clone()).unwrap_or_default()
}

/// `(error, log lines)` from the most recent run.
pub fn status(world: &World) -> (Option<String>, Vec<String>) {
    world.get::<ScriptState>().map(|s| (s.error.clone(), s.log.clone())).unwrap_or_default()
}

const PRELUDE: &str = r#"
function call() error("call() only works while on_tick runs") end
function log() end
function remove_body(name) return call("remove_body", {name = name}) end
function hide_body(name) return call("hide_body", {name = name}) end
function show_body(name) return call("show_body", {name = name}) end
function set_mass(name, mass) return call("set_mass", {name = name, mass = mass}) end
function body(name) return call("body", {name = name}) end
function bodies() return call("bodies") end
"#;

struct Runtime {
    lua: Lua,
    on_tick: Option<mlua::Function>,
    used: Arc<AtomicU64>,
}

impl Runtime {
    fn new(source: &str) -> Result<Runtime, String> {
        let lua = Lua::new_with(StdLib::MATH | StdLib::STRING | StdLib::TABLE | StdLib::UTF8, LuaOptions::default()).map_err(|e| e.to_string())?;
        lua.set_memory_limit(MEMORY_LIMIT).map_err(|e| e.to_string())?;
        let used = Arc::new(AtomicU64::new(0));
        let u = used.clone();
        lua.set_hook(HookTriggers::new().every_nth_instruction(HOOK_EVERY), move |_, _| {
            if u.fetch_add(HOOK_EVERY as u64, Ordering::Relaxed) > BUDGET {
                Err(mlua::Error::runtime("script used too many steps in one tick (is there an endless loop?)"))
            } else {
                Ok(VmState::Continue)
            }
        })
        .map_err(|e| e.to_string())?;
        let g = lua.globals();
        // No way out of the sandbox, and nothing that differs between runs.
        for name in ["dofile", "loadfile", "load", "loadstring", "require", "collectgarbage", "print"] {
            g.set(name, LuaValue::Nil).map_err(|e| e.to_string())?;
        }
        if let Ok(m) = g.get::<mlua::Table>("math") {
            for name in ["random", "randomseed"] {
                m.set(name, LuaValue::Nil).map_err(|e| e.to_string())?;
            }
        }
        lua.load(PRELUDE).exec().map_err(|e| e.to_string())?;
        used.store(0, Ordering::Relaxed);
        lua.load(source).set_name("script").exec().map_err(|e| e.to_string())?;
        let on_tick = g.get::<Option<mlua::Function>>("on_tick").map_err(|e| e.to_string())?;
        Ok(Runtime { lua, on_tick, used })
    }

    fn run(&mut self, world: &mut World, tick: u64, log: &mut Vec<String>) -> Result<(), String> {
        let Some(f) = self.on_tick.clone() else { return Ok(()) };
        self.used.store(0, Ordering::Relaxed);
        let lua = &self.lua;
        let opts = SerializeOptions::new().serialize_none_to_null(false).serialize_unit_to_null(false);
        lua.scope(|scope| {
            let call = scope.create_function_mut(|lua, (name, arg): (String, LuaValue)| {
                let args: Value = if arg.is_nil() { Value::Null } else { lua.from_value(arg)? };
                match world.call(&name, &args) {
                    Ok(v) => lua.to_value_with(&v, opts),
                    Err(e) => Err(mlua::Error::runtime(e)),
                }
            })?;
            let print = scope.create_function_mut(|_, parts: mlua::Variadic<LuaValue>| {
                let line = parts.iter().map(|p| p.to_string().unwrap_or_default()).collect::<Vec<_>>().join(" ");
                log.push(line);
                Ok(())
            })?;
            let g = lua.globals();
            g.set("call", call)?;
            g.set("log", print)?;
            f.call::<()>(tick)
        })
        .map_err(|e| e.to_string())
    }
}

struct ScriptSystem {
    loaded: Option<String>,
    runtime: Option<Runtime>,
}

impl System for ScriptSystem {
    fn name(&self) -> &str {
        "script.on_tick"
    }
    fn run(&mut self, ctx: &mut Context) {
        let Some(mut st) = ctx.world.take::<ScriptState>() else { return };
        if self.loaded.as_deref() != Some(st.source.as_str()) {
            self.loaded = Some(st.source.clone());
            self.runtime = None;
            st.error = None;
            if !st.source.trim().is_empty() {
                match Runtime::new(&st.source) {
                    Ok(r) => self.runtime = Some(r),
                    Err(e) => st.error = Some(e),
                }
            }
        }
        if st.error.is_none() {
            if let Some(rt) = self.runtime.as_mut() {
                let mut lines = Vec::new();
                if let Err(e) = rt.run(ctx.world, ctx.tick, &mut lines) {
                    st.error = Some(format!("tick {}: {e}", ctx.tick));
                }
                if !lines.is_empty() {
                    st.log.extend(lines.into_iter().map(|l| format!("[{}] {l}", ctx.tick)));
                    let extra = st.log.len().saturating_sub(LOG_KEEP);
                    st.log.drain(..extra);
                }
            }
        }
        ctx.world.put(st);
    }
}

pub struct ScriptModule;

impl Module for ScriptModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        let cfg: ScriptConfig = b.config.module(MODULE);
        b.world.insert_persistent("script", ScriptState { source: cfg.source, error: None, log: Vec::new() });
        // PreUpdate: events happen before the physics of the same tick.
        b.add_system(Stage::PreUpdate, ScriptSystem { loaded: None, runtime: None });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use simtara_core::{Config, Simulation};
    use simtara_orbits::{Bodies, OrbitConfig, OrbitModule};

    fn universe(script: &str) -> Simulation {
        let mut config = Config::default();
        config.set_module(simtara_orbits::MODULE, &OrbitConfig::sun_earth_moon());
        config.set_module(MODULE, &ScriptConfig { source: script.into() });
        let mut sim = Simulation::new(config);
        sim.add_module(OrbitModule);
        sim.add_module(ScriptModule);
        sim
    }

    fn bodies(sim: &Simulation) -> usize {
        sim.world().get::<Bodies>().unwrap().list.len()
    }

    #[test]
    fn script_removes_the_moon_at_the_right_tick() {
        let mut sim = universe("function on_tick(t) if t == 50 then remove_body('Moon') end end");
        sim.advance(50);
        assert_eq!(bodies(&sim), 3, "not yet");
        sim.advance(1);
        assert_eq!(bodies(&sim), 2, "removed at tick 50");
        assert!(status(sim.world()).0.is_none());
    }

    #[test]
    fn a_hidden_body_can_be_brought_back() {
        let src = "function on_tick(t) if t == 10 then hide_body('Moon') end if t == 200 then show_body('Moon') end end";
        let mut sim = universe(src);
        sim.advance(50);
        assert_eq!(sim.collect_metrics()["orbits.bodies"], 2.0);
        assert_eq!(bodies(&sim), 3, "still there, just not pulling");
        sim.advance(200);
        assert_eq!(sim.collect_metrics()["orbits.bodies"], 3.0);
        assert!(status(sim.world()).0.is_none());
    }

    #[test]
    fn script_can_read_and_log() {
        let mut sim = universe("function on_tick(t) if t == 3 then local m = body('Moon'); log('moon mass', m.mass) end end");
        sim.advance(5);
        let (err, log) = status(sim.world());
        assert!(err.is_none(), "{err:?}");
        assert_eq!(log.len(), 1);
        assert!(log[0].contains("moon mass") && log[0].contains("7342"), "{log:?}");
    }

    #[test]
    fn mistakes_are_reported_and_do_not_stop_the_universe() {
        let mut sim = universe("function on_tick(t) remove_body(42) end");
        sim.advance(3);
        let (err, _) = status(sim.world());
        assert!(err.unwrap().contains("tick 0"));
        assert_eq!(sim.world().clock.tick, 3);
        // a syntax error is reported too
        let mut sim = universe("function on_tick(t) +++ end");
        sim.advance(1);
        assert!(status(sim.world()).0.is_some());
        // and fixing the text turns it back on
        set_source(sim.world_mut(), "function on_tick(t) remove_body('Moon') end").unwrap();
        sim.advance(1);
        assert!(status(sim.world()).0.is_none());
        assert_eq!(bodies(&sim), 2);
    }

    #[test]
    fn endless_loops_are_stopped() {
        let mut sim = universe("function on_tick(t) while true do end end");
        let t0 = std::time::Instant::now();
        sim.advance(2);
        assert!(t0.elapsed().as_secs() < 5);
        assert!(status(sim.world()).0.unwrap().contains("too many steps"));
    }

    #[test]
    fn the_sandbox_has_no_way_out() {
        for code in ["os.execute('echo hi')", "io.open('x')", "dofile('x')", "require('os')", "return load('return 1')()", "math.random()"] {
            let mut sim = universe(&format!("function on_tick(t) {code} end"));
            sim.advance(1);
            assert!(status(sim.world()).0.is_some(), "{code} should fail");
        }
    }

    #[test]
    fn script_is_saved_with_the_universe_and_runs_identically() {
        let src = "function on_tick(t) if t == 20 then set_mass('Moon', 1e22) end end";
        let mut a = universe(src);
        a.advance(10);
        let snap = a.snapshot().unwrap();
        assert_eq!(snap.state["script"]["source"], json!(src));
        let mut b = Simulation::new(snap.config.clone());
        b.add_module(OrbitModule);
        b.add_module(ScriptModule);
        b.restore(&snap).unwrap();
        a.advance(30);
        b.advance(30);
        assert_eq!(a.snapshot().unwrap().state, b.snapshot().unwrap().state);
        assert_eq!(a.world().get::<Bodies>().unwrap().find("Moon").unwrap().mass, 1e22);
    }

    #[test]
    fn no_script_means_nothing_happens() {
        let mut sim = universe("");
        sim.advance(5);
        assert_eq!(bodies(&sim), 3);
        assert!(status(sim.world()).0.is_none());
    }
}
