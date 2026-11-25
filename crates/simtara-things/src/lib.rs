//! Things: the domain-neutral way to describe a universe.
//!
//! A *thing* has a kind (`"person"`, `"idea"`, `"water"`, `"place"` ...) and any
//! number of named properties (numbers, text, true/false). Things can be
//! *related* to each other (`knows`, `believes`, `inside`, `next_to`). Nothing
//! here knows what a person or water is: names are chosen by whoever builds
//! the universe, and the same rules work for all of them.
//!
//! Commands offered to scripts and tools:
//! `spawn {kind, props?}`, `despawn {id}`, `set {id, props}`, `get {id}`,
//! `things {kind?}`, `count {kind?}`, `relate {from, to, kind, weight?}`,
//! `unrelate {from, to, kind?}`, `related {id, kind?}`,
//! `define_metric {name, kind, prop?, op}` (op: count, sum, mean, min, max),
//! `set_all {kind, props}`, `random {}` (0..1, repeatable from the universe seed), and rules that
//! run natively on every thing of a kind each tick: `add_rule {name, kind, when?, set?, remove_when?, every?}`,
//! `remove_rule {name}`, `enable_rule {name, on?}`, `rules` (see `rules.rs` for the formula language).

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
mod rules;
pub use rules::{compile, Rule};

use simtara_core::{Builder, Context, Module, Stage, System, World};
use std::collections::BTreeMap;

pub const MODULE: &str = "things";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Thing {
    pub id: u64,
    pub kind: String,
    pub props: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Relation {
    pub from: u64,
    pub to: u64,
    pub kind: String,
    pub weight: f64,
}

/// A measurement defined by name, evaluated natively (fast, no script needed).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricDef {
    pub name: String,
    pub kind: String,
    pub prop: Option<String>,
    pub op: String,
}

/// How a picture of the universe should be drawn. A script sets it with `view{...}`; every field
/// names a property of the things (x and y make a map, color and size make it readable).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct View {
    pub x: Option<String>,
    pub y: Option<String>,
    pub color: Option<String>,
    pub size: Option<String>,
    pub links: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Things {
    #[serde(default)]
    pub view: View,
    next_id: u64,
    pub things: BTreeMap<u64, Thing>,
    pub relations: Vec<Relation>,
    pub metrics: Vec<MetricDef>,
    pub rules: Vec<Rule>,
    #[serde(skip)]
    compiled: std::collections::HashMap<String, std::sync::Arc<rules::Compiled>>,
    rng: u64,
}

impl Things {
    fn random(&mut self) -> f64 {
        rules::unit_random(&mut self.rng)
    }

    pub fn of_kind<'a>(&'a self, kind: Option<&'a str>) -> impl Iterator<Item = &'a Thing> {
        self.things.values().filter(move |t| kind.is_none_or(|k| t.kind == k))
    }

    pub fn evaluate(&self, m: &MetricDef) -> f64 {
        let things = self.of_kind(Some(&m.kind));
        if m.op == "count" {
            return things.count() as f64;
        }
        let vals: Vec<f64> = things.filter_map(|t| m.prop.as_ref().and_then(|p| t.props.get(p)).and_then(as_number)).collect();
        if vals.is_empty() {
            return 0.0;
        }
        match m.op.as_str() {
            "sum" => vals.iter().sum(),
            "mean" => vals.iter().sum::<f64>() / vals.len() as f64,
            "min" => vals.iter().copied().fold(f64::INFINITY, f64::min),
            "max" => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            _ => 0.0,
        }
    }
}

/// Numbers count as themselves and true/false as 1/0, so "how many believe it" is a plain sum.
fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    }
}

struct RuleSystem;

impl System for RuleSystem {
    fn name(&self) -> &str {
        "things.rules"
    }
    fn run(&mut self, ctx: &mut Context) {
        if let Some(mut t) = ctx.world.take::<Things>() {
            t.run_rules(ctx.tick);
            ctx.world.put(t);
        }
    }
}

pub struct ThingsModule;

impl Module for ThingsModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        b.world.insert_persistent("things", Things { rng: b.config.seed, ..Things::default() });
        register_commands(b.world);
        b.add_system(Stage::Update, RuleSystem);
        b.add_metric("things.total", |w| w.get::<Things>().map_or(0.0, |t| t.things.len() as f64));
        b.add_metric_source(|w| {
            let Some(t) = w.get::<Things>() else { return Vec::new() };
            let mut out: Vec<(String, f64)> = t.metrics.iter().map(|m| (m.name.clone(), t.evaluate(m))).collect();
            let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
            for th in t.things.values() {
                *kinds.entry(th.kind.as_str()).or_default() += 1;
            }
            out.extend(kinds.into_iter().map(|(k, n)| (format!("things.count.{k}"), n as f64)));
            out
        });
    }
}

fn thing_json(t: &Thing) -> Value {
    let mut m = Map::new();
    for (k, v) in &t.props {
        m.insert(k.clone(), v.clone());
    }
    m.insert("id".into(), json!(t.id));
    m.insert("kind".into(), json!(t.kind));
    Value::Object(m)
}

fn id_arg(a: &Value, key: &str) -> Result<u64, String> {
    a.get(key).and_then(Value::as_u64).ok_or_else(|| format!("needs '{key}' (a thing's id, e.g. {{{key} = 3}})"))
}

fn str_arg<'a>(a: &'a Value, key: &str) -> Result<&'a str, String> {
    a.get(key).and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or_else(|| format!("needs '{key}' as text, e.g. {{{key} = \"person\"}}"))
}

fn props_of(a: &Value) -> Result<BTreeMap<String, Value>, String> {
    match a.get("props") {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(m)) => {
            for (k, v) in m {
                if !matches!(v, Value::Number(_) | Value::String(_) | Value::Bool(_)) {
                    return Err(format!("property '{k}' must be a number, text or true/false"));
                }
            }
            Ok(m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        }
        Some(_) => Err("'props' must be a table like {age = 30, mood = \"calm\"}".into()),
    }
}

fn things_mut(w: &mut World) -> Result<&mut Things, String> {
    w.get_mut::<Things>().ok_or_else(|| "this universe has no things".to_string())
}

fn register_commands(w: &mut World) {
    w.register_command("spawn", |w, a| {
        let kind = str_arg(a, "kind")?.to_string();
        let props = props_of(a)?;
        let t = things_mut(w)?;
        t.next_id += 1;
        let id = t.next_id;
        t.things.insert(id, Thing { id, kind, props });
        Ok(json!(id))
    });
    w.register_command("despawn", |w, a| {
        let id = id_arg(a, "id")?;
        let t = things_mut(w)?;
        t.relations.retain(|r| r.from != id && r.to != id);
        Ok(json!(t.things.remove(&id).is_some()))
    });
    w.register_command("set", |w, a| {
        let id = id_arg(a, "id")?;
        let props = props_of(a)?;
        let t = things_mut(w)?;
        let th = t.things.get_mut(&id).ok_or_else(|| format!("no thing with id {id}"))?;
        th.props.extend(props);
        Ok(json!(true))
    });
    w.register_command("get", |w, a| {
        let id = id_arg(a, "id")?;
        Ok(things_mut(w)?.things.get(&id).map_or(Value::Null, thing_json))
    });
    w.register_command("things", |w, a| {
        let kind = a.get("kind").and_then(Value::as_str);
        let t = things_mut(w)?;
        Ok(Value::Array(t.of_kind(kind).map(thing_json).collect()))
    });
    w.register_command("count", |w, a| {
        let kind = a.get("kind").and_then(Value::as_str);
        let t = things_mut(w)?;
        Ok(json!(t.of_kind(kind).count()))
    });
    w.register_command("relate", |w, a| {
        let (from, to) = (id_arg(a, "from")?, id_arg(a, "to")?);
        let kind = str_arg(a, "kind")?.to_string();
        let weight = a.get("weight").and_then(Value::as_f64).unwrap_or(1.0);
        let t = things_mut(w)?;
        for id in [from, to] {
            if !t.things.contains_key(&id) {
                return Err(format!("no thing with id {id}"));
            }
        }
        match t.relations.iter_mut().find(|r| r.from == from && r.to == to && r.kind == kind) {
            Some(r) => r.weight = weight,
            None => t.relations.push(Relation { from, to, kind, weight }),
        }
        Ok(json!(true))
    });
    w.register_command("unrelate", |w, a| {
        let (from, to) = (id_arg(a, "from")?, id_arg(a, "to")?);
        let kind = a.get("kind").and_then(Value::as_str);
        let t = things_mut(w)?;
        let before = t.relations.len();
        t.relations.retain(|r| !(r.from == from && r.to == to && kind.is_none_or(|k| r.kind == k)));
        Ok(json!(t.relations.len() < before))
    });
    w.register_command("related", |w, a| {
        let id = id_arg(a, "id")?;
        let kind = a.get("kind").and_then(Value::as_str);
        let t = things_mut(w)?;
        let out: Vec<Value> = t
            .relations
            .iter()
            .filter(|r| r.from == id && kind.is_none_or(|k| r.kind == k))
            .map(|r| json!({"id": r.to, "kind": r.kind, "weight": r.weight}))
            .collect();
        Ok(Value::Array(out))
    });
    w.register_command("view", |w, a| {
        let text = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
        let v = View { x: text("x"), y: text("y"), color: text("color"), size: text("size"), links: a.get("links").and_then(Value::as_bool).unwrap_or(false) };
        things_mut(w)?.view = v;
        Ok(json!(true))
    });
    w.register_command("define_metric", |w, a| {
        let name = str_arg(a, "name")?.to_string();
        let kind = str_arg(a, "kind")?.to_string();
        let op = str_arg(a, "op")?.to_string();
        if !["count", "sum", "mean", "min", "max"].contains(&op.as_str()) {
            return Err(format!("op must be one of count, sum, mean, min, max (got '{op}')"));
        }
        let prop = a.get("prop").and_then(Value::as_str).map(str::to_string);
        if op != "count" && prop.is_none() {
            return Err(format!("op '{op}' needs 'prop', the property to measure"));
        }
        let t = things_mut(w)?;
        t.metrics.retain(|m| m.name != name);
        t.metrics.push(MetricDef { name, kind, prop, op });
        Ok(json!(true))
    });
    w.register_command("add_rule", |w, a| {
        let name = str_arg(a, "name")?.to_string();
        let kind = str_arg(a, "kind")?.to_string();
        let text = |key: &str| -> Result<Option<String>, String> {
            match a.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(_) => Err(format!("'{key}' must be a formula in quotes, e.g. {key} = \"age > 90\"")),
            }
        };
        let mut set = BTreeMap::new();
        match a.get("set") {
            None | Some(Value::Null) => {}
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    let f = match v {
                        Value::String(f) => f.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        _ => return Err(format!("set.{k} must be a formula in quotes, e.g. {k} = \"{k} + 1\"")),
                    };
                    set.insert(k.clone(), f);
                }
            }
            Some(_) => return Err("'set' must be a table like {age = \"age + 1\"}".into()),
        }
        let every = a.get("every").and_then(Value::as_u64).unwrap_or(1).max(1);
        let rule = Rule { name, kind, when: text("when")?, set, remove_when: text("remove_when")?, every, enabled: true, error: None };
        things_mut(w)?.add_rule(rule)?;
        Ok(json!(true))
    });
    w.register_command("remove_rule", |w, a| {
        let name = str_arg(a, "name")?;
        let t = things_mut(w)?;
        let before = t.rules.len();
        t.rules.retain(|r| r.name != name);
        Ok(json!(t.rules.len() < before))
    });
    w.register_command("enable_rule", |w, a| {
        let name = str_arg(a, "name")?;
        let on = a.get("on").and_then(Value::as_bool).unwrap_or(true);
        let t = things_mut(w)?;
        let r = t.rules.iter_mut().find(|r| r.name == name).ok_or_else(|| format!("no rule named '{name}'"))?;
        r.enabled = on;
        if on {
            r.error = None;
        }
        Ok(json!(true))
    });
    w.register_command("warnings", |w, _| Ok(json!(things_mut(w)?.warnings())));
    w.register_command("rules", |w, _| {
        let t = things_mut(w)?;
        Ok(Value::Array(
            t.rules
                .iter()
                .map(|r| json!({"name": r.name, "kind": r.kind, "when": r.when, "set": r.set, "remove_when": r.remove_when, "every": r.every, "enabled": r.enabled, "error": r.error}))
                .collect(),
        ))
    });
    w.register_command("set_all", |w, a| {
        let kind = str_arg(a, "kind")?.to_string();
        let props = props_of(a)?;
        let t = things_mut(w)?;
        let mut n = 0;
        for th in t.things.values_mut().filter(|x| x.kind == kind) {
            th.props.extend(props.clone());
            n += 1;
        }
        Ok(json!(n))
    });
    w.register_command("random", |w, _| Ok(json!(things_mut(w)?.random())));
}

#[cfg(test)]
mod tests {
    use super::*;
    use simtara_core::{Config, Simulation};

    fn sim() -> Simulation {
        let mut s = Simulation::new(Config::default());
        s.add_module(ThingsModule);
        s
    }

    #[test]
    fn people_ideas_and_water_are_all_just_things() {
        let mut s = sim();
        let w = s.world_mut();
        let a = w.call("spawn", &json!({"kind": "person", "props": {"believes": true}})).unwrap();
        let b = w.call("spawn", &json!({"kind": "person", "props": {"believes": false}})).unwrap();
        w.call("spawn", &json!({"kind": "water", "props": {"depth": 2.5}})).unwrap();
        w.call("relate", &json!({"from": a, "to": b, "kind": "knows"})).unwrap();
        assert_eq!(w.call("count", &json!({"kind": "person"})).unwrap(), json!(2));
        assert_eq!(w.call("related", &json!({"id": a, "kind": "knows"})).unwrap().as_array().unwrap().len(), 1);
        w.call("define_metric", &json!({"name": "believers", "kind": "person", "prop": "believes", "op": "sum"})).unwrap();
        let m = s.collect_metrics();
        assert_eq!(m["believers"], 1.0);
        assert_eq!(m["things.count.water"], 1.0);
    }

    #[test]
    fn despawn_drops_relations_and_bad_input_explains_itself() {
        let mut s = sim();
        let w = s.world_mut();
        let a = w.call("spawn", &json!({"kind": "x"})).unwrap();
        let b = w.call("spawn", &json!({"kind": "x"})).unwrap();
        w.call("relate", &json!({"from": a, "to": b, "kind": "r"})).unwrap();
        w.call("despawn", &json!({"id": b})).unwrap();
        assert!(w.call("related", &json!({"id": a})).unwrap().as_array().unwrap().is_empty());
        assert!(w.call("spawn", &json!({})).unwrap_err().contains("kind"));
        assert!(w.call("spawn", &json!({"kind": "x", "props": {"p": [1]}})).is_err());
        assert!(w.call("define_metric", &json!({"name": "n", "kind": "x", "op": "mean"})).unwrap_err().contains("prop"));
    }

    #[test]
    fn random_repeats_from_the_seed_and_survives_a_snapshot() {
        let draw = |s: &mut Simulation| (0..3).map(|_| s.world_mut().call("random", &json!({})).unwrap().as_f64().unwrap()).collect::<Vec<_>>();
        let (mut a, mut b) = (sim(), sim());
        assert_eq!(draw(&mut a), draw(&mut b));
        let snap = a.snapshot().unwrap();
        let next = draw(&mut a);
        a.restore(&snap).unwrap();
        assert_eq!(draw(&mut a), next);
    }

    fn rule(w: &mut World, v: Value) -> Result<Value, String> {
        w.call("add_rule", &v)
    }

    #[test]
    fn rules_update_everything_together_and_explain_mistakes() {
        let mut s = sim();
        let w = s.world_mut();
        for age in [1, 50, 89] {
            w.call("spawn", &json!({"kind": "person", "props": {"age": age}})).unwrap();
        }
        rule(w, json!({"name": "age", "kind": "person", "set": {"age": "age + 1"}, "remove_when": "age > 90"})).unwrap();
        w.call("define_metric", &json!({"name": "mean_age", "kind": "person", "prop": "age", "op": "mean"})).unwrap();
        s.advance(1);
        assert_eq!(s.collect_metrics()["mean_age"], (2.0 + 51.0 + 90.0) / 3.0);
        s.advance(2);
        assert_eq!(s.collect_metrics()["things.count.person"], 2.0, "the oldest left");
        let w = s.world_mut();
        assert!(rule(w, json!({"name": "bad", "kind": "x", "set": {"a": "age +"}})).unwrap_err().contains("age +"));
        assert!(rule(w, json!({"name": "bad", "kind": "x", "set": {"a": "nope(1)"}})).unwrap_err().contains("unknown function"));
        assert!(rule(w, json!({"name": "bad", "kind": "x"})).is_err());
    }

    #[test]
    fn neighbours_drive_rules_and_rules_can_be_switched_off() {
        let mut s = sim();
        let w = s.world_mut();
        let ids: Vec<u64> = (0..3).map(|i| w.call("spawn", &json!({"kind": "p", "props": {"v": if i == 0 { 10.0 } else { 0.0 }}})).unwrap().as_u64().unwrap()).collect();
        w.call("relate", &json!({"from": ids[1], "to": ids[0], "kind": "near"})).unwrap();
        w.call("relate", &json!({"from": ids[2], "to": ids[1], "kind": "near"})).unwrap();
        rule(w, json!({"name": "flow", "kind": "p", "when": "ncount(near) > 0", "set": {"v": "nmean(near, v)"}})).unwrap();
        s.advance(1);
        let v = |s: &Simulation, i: usize| s.world().get::<Things>().unwrap().things[&ids[i]].props["v"].as_f64().unwrap();
        assert_eq!((v(&s, 0), v(&s, 1), v(&s, 2)), (10.0, 10.0, 0.0), "synchronous: 2 saw the old value of 1");
        s.world_mut().call("enable_rule", &json!({"name": "flow", "on": false})).unwrap();
        s.advance(5);
        assert_eq!(v(&s, 2), 0.0, "switched off, nothing moves");
        s.world_mut().call("enable_rule", &json!({"name": "flow"})).unwrap();
        s.advance(1);
        assert_eq!(v(&s, 2), 10.0);
    }

    #[test]
    fn a_rule_that_breaks_turns_itself_off_and_says_why() {
        let mut s = sim();
        let w = s.world_mut();
        w.call("spawn", &json!({"kind": "p", "props": {"v": 1.0}})).unwrap();
        rule(w, json!({"name": "div", "kind": "p", "set": {"v": "1 / (v - 1)"}})).unwrap();
        s.advance(2);
        let r = s.world_mut().call("rules", &json!({})).unwrap();
        assert_eq!(r[0]["enabled"], json!(false));
        assert!(r[0]["error"].as_str().unwrap().contains("division by zero"));
    }

    #[test]
    fn a_hundred_thousand_things_with_random_rules_repeat_exactly() {
        let build = || {
            let mut s = sim();
            let w = s.world_mut();
            for i in 0..100_000 {
                w.call("spawn", &json!({"kind": "c", "props": {"e": (i % 7) as f64, "alive": true}})).unwrap();
            }
            rule(w, json!({"name": "drift", "kind": "c", "set": {"e": "e * 0.99 + random()"}, "remove_when": "random() < 0.001"})).unwrap();
            s
        };
        let (mut a, mut b) = (build(), build());
        let t0 = std::time::Instant::now();
        a.advance(5);
        let took = t0.elapsed();
        b.advance(5);
        assert_eq!(a.snapshot().unwrap().state, b.snapshot().unwrap().state);
        let n = a.collect_metrics()["things.count.c"];
        assert!(n < 100_000.0 && n > 99_000.0, "{n}");
        eprintln!("100k things x 5 ticks: {took:?}");
    }

    #[test]
    fn a_misspelt_property_is_flagged() {
        let mut s = sim();
        let w = s.world_mut();
        w.call("spawn", &json!({"kind": "p", "props": {"energy": 1.0}})).unwrap();
        rule(w, json!({"name": "r", "kind": "p", "set": {"energy": "enrgy + 1"}})).unwrap();
        let warn = w.call("warnings", &json!({})).unwrap();
        assert!(warn[0].as_str().unwrap().contains("enrgy"), "{warn}");
        rule(w, json!({"name": "r", "kind": "p", "set": {"energy": "energy + 1"}})).unwrap();
        assert_eq!(w.call("warnings", &json!({})).unwrap(), json!([]));
    }
}
