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
//! `random {}` (0..1, repeatable from the universe seed).

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use simtara_core::{Builder, Module, World};
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Things {
    next_id: u64,
    pub things: BTreeMap<u64, Thing>,
    pub relations: Vec<Relation>,
    pub metrics: Vec<MetricDef>,
    rng: u64,
}

impl Things {
    fn random(&mut self) -> f64 {
        // splitmix64: tiny, repeatable, and its state is one number that snapshots save.
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
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

pub struct ThingsModule;

impl Module for ThingsModule {
    fn name(&self) -> &str {
        MODULE
    }
    fn build(&self, b: &mut Builder) {
        b.world.insert_persistent("things", Things { rng: b.config.seed, ..Things::default() });
        register_commands(b.world);
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
}
