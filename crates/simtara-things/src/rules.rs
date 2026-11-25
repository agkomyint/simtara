//! Rules: small formulas that run natively on every thing of a kind, every tick.
//!
//! ```text
//! add_rule{name = "spread", kind = "person",
//!          when = "believes == 0 and random() < 0.3 * nmean(knows, believes)",
//!          set  = {believes = "1"}}
//! add_rule{name = "age", kind = "person", set = {age = "age + 1"}, remove_when = "age > 90"}
//! ```
//!
//! Updates are *synchronous*: every thing reads the state from before the rule ran, then all
//! changes are applied together, so the result never depends on the order things are visited.
//! `random()` is derived from the thing's id, so runs repeat exactly, whether on one core or many.
//!
//! Expressions: numbers, `"text"`, `true`/`false`, property names, `+ - * / %`,
//! `== != < <= > >=`, `and or not`, `( )`, and functions
//! `random() tick() min(a,b) max(a,b) abs(x) sqrt(x) floor(x) clamp(x,lo,hi) if(cond,a,b)`.
//! Neighbours (things this one points at with a relation): `ncount(rel)`, `ncount(rel, prop)`,
//! `nsum(rel, prop)`, `nmean(rel, prop)`, `nmax(rel, prop)`, `nmin(rel, prop)`, `wsum(rel, prop)`
//! (weighted by the relation's weight). `id` and `kind` are properties too.

use crate::{Relation, Thing, Things};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// Run in parallel only when there are enough things for it to pay off.
const PARALLEL_FROM: usize = 4000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    pub kind: String,
    pub when: Option<String>,
    pub set: BTreeMap<String, String>,
    pub remove_when: Option<String>,
    /// Run only on ticks that are a multiple of this (1 = every tick).
    pub every: u64,
    pub enabled: bool,
    /// Set when the rule failed; the rule then stays off until it is enabled again.
    #[serde(skip)]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    N(f64),
    S(String),
}

impl Val {
    fn num(&self) -> f64 {
        match self {
            Val::N(x) => *x,
            Val::S(_) => 0.0,
        }
    }
    fn from_json(v: &Value) -> Val {
        match v {
            Value::Number(n) => Val::N(n.as_f64().unwrap_or(0.0)),
            Value::Bool(b) => Val::N(*b as u8 as f64),
            Value::String(s) => Val::S(s.clone()),
            _ => Val::N(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    Str(String),
    Prop(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(&'static str, Vec<Expr>),
    /// Neighbour aggregate: function, relation kind, property.
    Agg(&'static str, String, Option<String>),
}

/// A rule's formulas, parsed once.
#[derive(Debug)]
pub struct Compiled {
    when: Option<Expr>,
    set: Vec<(String, Expr)>,
    remove: Option<Expr>,
    uses_neighbours: bool,
}

// ---------- parsing ----------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Id(String),
    Sym(&'static str),
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && cs.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            if i < cs.len() && (cs[i] == 'e' || cs[i] == 'E') {
                let save = i;
                i += 1;
                if i < cs.len() && (cs[i] == '+' || cs[i] == '-') {
                    i += 1;
                }
                if i < cs.len() && cs[i].is_ascii_digit() {
                    while i < cs.len() && cs[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let t: String = cs[st..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number '{t}'"))?));
        } else if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            out.push(Tok::Id(cs[st..i].iter().collect()));
        } else if c == '"' || c == '\'' {
            let st = i + 1;
            i += 1;
            while i < cs.len() && cs[i] != c {
                i += 1;
            }
            if i >= cs.len() {
                return Err("text is missing its closing quote".into());
            }
            out.push(Tok::Str(cs[st..i].iter().collect()));
            i += 1;
        } else {
            let two: String = cs[i..(i + 2).min(cs.len())].iter().collect();
            let sym = match two.as_str() {
                "==" => Some("=="),
                "!=" => Some("!="),
                "<=" => Some("<="),
                ">=" => Some(">="),
                "&&" => Some("and"),
                "||" => Some("or"),
                _ => None,
            };
            if let Some(s) = sym {
                out.push(Tok::Sym(s));
                i += 2;
                continue;
            }
            let s = match c {
                '(' => "(",
                ')' => ")",
                ',' => ",",
                '+' => "+",
                '-' => "-",
                '*' => "*",
                '/' => "/",
                '%' => "%",
                '<' => "<",
                '>' => ">",
                '!' => "not",
                _ => return Err(format!("unexpected '{c}'")),
            };
            out.push(Tok::Sym(s));
            i += 1;
        }
    }
    Ok(out)
}

struct Parser {
    t: Vec<Tok>,
    p: usize,
    neighbours: bool,
}

const AGGS: [&str; 7] = ["ncount", "nsum", "nmean", "nmax", "nmin", "wsum", "wmean"];
/// (name, number of arguments)
const FUNCS: [(&str, usize); 9] = [("random", 0), ("tick", 0), ("min", 2), ("max", 2), ("abs", 1), ("sqrt", 1), ("floor", 1), ("clamp", 3), ("if", 3)];

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.p)
    }
    fn eat_sym(&mut self, s: &str) -> bool {
        if self.peek() == Some(&Tok::Sym(match s {
            "(" => "(",
            ")" => ")",
            "," => ",",
            _ => return false,
        })) {
            self.p += 1;
            true
        } else {
            false
        }
    }
    fn eat_word(&mut self, w: &str) -> bool {
        let hit = match self.peek() {
            Some(Tok::Id(s)) => s == w,
            Some(Tok::Sym(s)) => *s == w,
            _ => false,
        };
        if hit {
            self.p += 1;
        }
        hit
    }
    fn or(&mut self) -> Result<Expr, String> {
        let mut l = self.and()?;
        while self.eat_word("or") {
            l = Expr::Bin(Op::Or, Box::new(l), Box::new(self.and()?));
        }
        Ok(l)
    }
    fn and(&mut self) -> Result<Expr, String> {
        let mut l = self.cmp()?;
        while self.eat_word("and") {
            l = Expr::Bin(Op::And, Box::new(l), Box::new(self.cmp()?));
        }
        Ok(l)
    }
    fn cmp(&mut self) -> Result<Expr, String> {
        let l = self.add()?;
        let op = [("==", Op::Eq), ("!=", Op::Ne), ("<=", Op::Le), (">=", Op::Ge), ("<", Op::Lt), (">", Op::Gt)].into_iter().find(|(s, _)| self.peek() == Some(&Tok::Sym(s)));
        if let Some((_, op)) = op {
            self.p += 1;
            return Ok(Expr::Bin(op, Box::new(l), Box::new(self.add()?)));
        }
        Ok(l)
    }
    fn add(&mut self) -> Result<Expr, String> {
        let mut l = self.mul()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Sym("+")) => Op::Add,
                Some(Tok::Sym("-")) => Op::Sub,
                _ => return Ok(l),
            };
            self.p += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.mul()?));
        }
    }
    fn mul(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Sym("*")) => Op::Mul,
                Some(Tok::Sym("/")) => Op::Div,
                Some(Tok::Sym("%")) => Op::Rem,
                _ => return Ok(l),
            };
            self.p += 1;
            l = Expr::Bin(op, Box::new(l), Box::new(self.unary()?));
        }
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(&Tok::Sym("-")) {
            self.p += 1;
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat_word("not") {
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        self.primary()
    }
    fn ident(&mut self, what: &str) -> Result<String, String> {
        match self.t.get(self.p).cloned() {
            Some(Tok::Id(s)) => {
                self.p += 1;
                Ok(s)
            }
            _ => Err(format!("expected {what}")),
        }
    }
    fn primary(&mut self) -> Result<Expr, String> {
        let Some(tok) = self.t.get(self.p).cloned() else { return Err("the formula ends too early".into()) };
        self.p += 1;
        match tok {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::Sym("(") => {
                let e = self.or()?;
                if !self.eat_sym(")") {
                    return Err("missing ')'".into());
                }
                Ok(e)
            }
            Tok::Id(name) if self.peek() == Some(&Tok::Sym("(")) => {
                self.p += 1;
                if let Some(agg) = AGGS.iter().find(|a| **a == name) {
                    let rel = self.ident("a relation name, e.g. ncount(knows)")?;
                    let prop = if self.eat_sym(",") { Some(self.ident("a property name")?) } else { None };
                    if prop.is_none() && *agg != "ncount" {
                        return Err(format!("{agg}(relation, property) needs a property"));
                    }
                    if !self.eat_sym(")") {
                        return Err("missing ')'".into());
                    }
                    self.neighbours = true;
                    return Ok(Expr::Agg(agg, rel, prop));
                }
                let Some((fname, arity)) = FUNCS.iter().find(|(f, _)| *f == name) else {
                    return Err(format!("unknown function '{name}'"));
                };
                let mut args = Vec::new();
                if !self.eat_sym(")") {
                    loop {
                        args.push(self.or()?);
                        if self.eat_sym(")") {
                            break;
                        }
                        if !self.eat_sym(",") {
                            return Err("expected ',' or ')'".into());
                        }
                    }
                }
                if args.len() != *arity {
                    return Err(format!("{fname} takes {arity} value(s), got {}", args.len()));
                }
                Ok(Expr::Call(fname, args))
            }
            Tok::Id(name) => Ok(match name.as_str() {
                "true" => Expr::Num(1.0),
                "false" => Expr::Num(0.0),
                _ => Expr::Prop(name),
            }),
            Tok::Sym(s) => Err(format!("unexpected '{s}'")),
        }
    }
}

fn parse(src: &str, uses_neighbours: &mut bool) -> Result<Expr, String> {
    let wrap = |e: String| format!("in \"{src}\": {e}");
    let mut p = Parser { t: lex(src).map_err(wrap)?, p: 0, neighbours: false };
    let e = p.or().map_err(wrap)?;
    if p.p < p.t.len() {
        return Err(wrap(format!("unexpected {:?} after the formula", p.t[p.p])));
    }
    *uses_neighbours |= p.neighbours;
    Ok(e)
}

pub fn compile(rule: &Rule) -> Result<Compiled, String> {
    let mut n = false;
    let when = rule.when.as_deref().map(|s| parse(s, &mut n)).transpose()?;
    let remove = rule.remove_when.as_deref().map(|s| parse(s, &mut n)).transpose()?;
    let mut set = Vec::new();
    for (prop, src) in &rule.set {
        set.push((prop.clone(), parse(src, &mut n).map_err(|e| format!("set.{prop} {e}"))?));
    }
    if set.is_empty() && remove.is_none() {
        return Err("a rule needs 'set' (what changes) or 'remove_when'".into());
    }
    Ok(Compiled { when, set, remove, uses_neighbours: n })
}

// ---------- evaluation ----------

struct Env<'a> {
    things: &'a Things,
    index: &'a HashMap<u64, Vec<&'a Relation>>,
    tick: u64,
    rng: u64,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(crate) fn unit_random(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn prop(t: &Thing, name: &str) -> Val {
    match name {
        "id" => Val::N(t.id as f64),
        "kind" => Val::S(t.kind.clone()),
        _ => t.props.get(name).map_or(Val::N(0.0), Val::from_json),
    }
}

fn truthy(v: &Val) -> bool {
    v.num() != 0.0
}

fn eval(e: &Expr, me: &Thing, env: &mut Env) -> Val {
    match e {
        Expr::Num(n) => Val::N(*n),
        Expr::Str(s) => Val::S(s.clone()),
        Expr::Prop(p) => match env.things.params.get(p.as_str()) {
            Some(par) if !me.props.contains_key(p.as_str()) && p != "id" && p != "kind" => Val::N(par.value),
            _ => prop(me, p),
        },
        Expr::Neg(x) => Val::N(-eval(x, me, env).num()),
        Expr::Not(x) => Val::N(!truthy(&eval(x, me, env)) as u8 as f64),
        Expr::Bin(Op::And, a, b) => Val::N((truthy(&eval(a, me, env)) && truthy(&eval(b, me, env))) as u8 as f64),
        Expr::Bin(Op::Or, a, b) => Val::N((truthy(&eval(a, me, env)) || truthy(&eval(b, me, env))) as u8 as f64),
        Expr::Bin(op, a, b) => {
            let (x, y) = (eval(a, me, env), eval(b, me, env));
            let eq = x == y;
            let (p, q) = (x.num(), y.num());
            Val::N(match op {
                Op::Add => p + q,
                Op::Sub => p - q,
                Op::Mul => p * q,
                Op::Div => p / q,
                Op::Rem => p % q,
                Op::Eq => eq as u8 as f64,
                Op::Ne => (!eq) as u8 as f64,
                Op::Lt => (p < q) as u8 as f64,
                Op::Le => (p <= q) as u8 as f64,
                Op::Gt => (p > q) as u8 as f64,
                Op::Ge => (p >= q) as u8 as f64,
                Op::And | Op::Or => unreachable!(),
            })
        }
        Expr::Call(name, args) => match *name {
            "random" => Val::N(unit_random(&mut env.rng)),
            "tick" => Val::N(env.tick as f64),
            "if" => {
                if truthy(&eval(&args[0], me, env)) {
                    eval(&args[1], me, env)
                } else {
                    eval(&args[2], me, env)
                }
            }
            _ => {
                let v: Vec<f64> = args.iter().map(|a| eval(a, me, env).num()).collect();
                Val::N(match *name {
                    "min" => v[0].min(v[1]),
                    "max" => v[0].max(v[1]),
                    "abs" => v[0].abs(),
                    "sqrt" => v[0].sqrt(),
                    "floor" => v[0].floor(),
                    _ => v[0].max(v[1]).min(v[2]),
                })
            }
        },
        Expr::Agg(f, rel, p) => {
            let empty = Vec::new();
            let near = env.index.get(&me.id).unwrap_or(&empty);
            let mut vals = Vec::new();
            for r in near.iter().filter(|r| r.kind == *rel) {
                let Some(other) = env.things.things.get(&r.to) else { continue };
                let x = p.as_ref().map_or(1.0, |p| prop(other, p).num());
                vals.push((x, r.weight));
            }
            let n = vals.len() as f64;
            Val::N(match *f {
                "ncount" if p.is_none() => n,
                "ncount" => vals.iter().filter(|(x, _)| *x != 0.0).count() as f64,
                "nsum" => vals.iter().map(|(x, _)| x).sum(),
                "nmean" => if n == 0.0 { 0.0 } else { vals.iter().map(|(x, _)| x).sum::<f64>() / n },
                "nmax" => vals.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max).max(if n == 0.0 { 0.0 } else { f64::NEG_INFINITY }),
                "nmin" => vals.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min).min(if n == 0.0 { 0.0 } else { f64::INFINITY }),
                "wsum" => vals.iter().map(|(x, w)| x * w).sum(),
                _ => {
                    let w: f64 = vals.iter().map(|(_, w)| w).sum();
                    if w == 0.0 { 0.0 } else { vals.iter().map(|(x, w)| x * w).sum::<f64>() / w }
                }
            })
        }
    }
}

type Outcome = Result<(Vec<(String, Value)>, bool), String>;

fn eval_thing(c: &Compiled, t: &Thing, env: &Env) -> Outcome {
    // Each thing gets its own random stream, from the rule's seed and its id.
    let mut e = Env { things: env.things, index: env.index, tick: env.tick, rng: env.rng ^ t.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) };
    if let Some(w) = &c.when {
        if !truthy(&eval(w, t, &mut e)) {
            return Ok((Vec::new(), false));
        }
    }
    let mut out = Vec::with_capacity(c.set.len());
    for (name, ex) in &c.set {
        let v = eval(ex, t, &mut e);
        let j = match v {
            Val::N(x) if !x.is_finite() => return Err(format!("'{name}' became {x} for thing {} (check for division by zero)", t.id)),
            // a true/false property stays true/false
            Val::N(x) if matches!(t.props.get(name), Some(Value::Bool(_))) => json!(x != 0.0),
            Val::N(x) => json!(x),
            Val::S(s) => json!(s),
        };
        out.push((name.clone(), j));
    }
    let remove = c.remove.as_ref().is_some_and(|r| truthy(&eval(r, t, &mut e)));
    Ok((out, remove))
}

impl Things {
    /// Add or replace a rule. The formulas are checked straight away.
    pub fn add_rule(&mut self, rule: Rule) -> Result<(), String> {
        let c = compile(&rule)?;
        self.compiled.insert(rule.name.clone(), std::sync::Arc::new(c));
        match self.rules.iter_mut().find(|r| r.name == rule.name) {
            Some(r) => *r = rule,
            None => self.rules.push(rule),
        }
        Ok(())
    }

    /// Run every enabled rule once, in the order they were added.
    pub fn run_rules(&mut self, tick: u64) {
        for ri in 0..self.rules.len() {
            let rule = self.rules[ri].clone();
            if !rule.enabled || (rule.every > 1 && tick % rule.every != 0) {
                continue;
            }
            let compiled = match self.compiled.get(&rule.name) {
                Some(c) => c.clone(),
                None => match compile(&rule) {
                    Ok(c) => {
                        let c = std::sync::Arc::new(c);
                        self.compiled.insert(rule.name.clone(), c.clone());
                        c
                    }
                    Err(e) => {
                        self.rules[ri].error = Some(e);
                        self.rules[ri].enabled = false;
                        continue;
                    }
                },
            };
            let seed = splitmix(&mut self.rng);
            let result: Result<Vec<(u64, Outcome)>, String> = {
                let mut index: HashMap<u64, Vec<&Relation>> = HashMap::new();
                if compiled.uses_neighbours {
                    for r in &self.relations {
                        index.entry(r.from).or_default().push(r);
                    }
                }
                let env = Env { things: self, index: &index, tick, rng: seed };
                let targets: Vec<&Thing> = self.of_kind(Some(&rule.kind)).collect();
                let run = |t: &&Thing| (t.id, eval_thing(&compiled, t, &env));
                let outs: Vec<(u64, Outcome)> = if targets.len() >= PARALLEL_FROM { targets.par_iter().map(run).collect() } else { targets.iter().map(run).collect() };
                Ok(outs)
            };
            let outs = result.unwrap_or_default();
            if let Some(e) = outs.iter().find_map(|(_, o)| o.as_ref().err()) {
                self.rules[ri].error = Some(format!("tick {tick}: {e}"));
                self.rules[ri].enabled = false;
                continue;
            }
            let mut gone = Vec::new();
            for (id, o) in outs {
                let Ok((updates, remove)) = o else { continue };
                if remove {
                    gone.push(id);
                } else if let Some(t) = self.things.get_mut(&id) {
                    for (p, v) in updates {
                        t.props.insert(p, v);
                    }
                }
            }
            if !gone.is_empty() {
                for id in &gone {
                    self.things.remove(id);
                }
                self.relations.retain(|r| self.things.contains_key(&r.from) && self.things.contains_key(&r.to));
            }
        }
    }
}

// ---------- warnings ----------

fn reads(e: &Expr, out: &mut Vec<(String, bool)>) {
    // (property, read from a neighbour)
    match e {
        Expr::Prop(p) if p != "id" && p != "kind" => out.push((p.clone(), false)),
        Expr::Neg(a) | Expr::Not(a) => reads(a, out),
        Expr::Bin(_, a, b) => {
            reads(a, out);
            reads(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| reads(a, out)),
        Expr::Agg(_, _, Some(p)) => out.push((p.clone(), true)),
        _ => {}
    }
}

impl Things {
    /// Plain-language hints about likely mistakes: a rule reads a property that no thing has and
    /// no rule sets (usually a typo; it silently reads as 0).
    pub fn warnings(&self) -> Vec<String> {
        let mut known: std::collections::BTreeSet<&str> = self.things.values().flat_map(|t| t.props.keys().map(String::as_str)).collect();
        for r in &self.rules {
            known.extend(r.set.keys().map(String::as_str));
        }
        known.extend(self.params.keys().map(String::as_str));
        let mut out = Vec::new();
        for r in &self.rules {
            let Ok(c) = compile(r) else { continue };
            let mut used = Vec::new();
            for e in c.when.iter().chain(c.remove.iter()).chain(c.set.iter().map(|(_, e)| e)) {
                reads(e, &mut used);
            }
            let mut seen = std::collections::BTreeSet::new();
            for (p, _) in used {
                if !known.contains(p.as_str()) && seen.insert(p.clone()) {
                    out.push(format!("rule '{}' reads '{}', which no thing has and no rule sets (typo? it reads as 0)", r.name, p));
                }
            }
        }
        out
    }
}
