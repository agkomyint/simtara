use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};

type Boxed = Box<dyn Any + Send>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Clock {
    pub tick: u64,
    pub time: f64,
}

#[derive(Default)]
struct Entities {
    generations: Vec<u32>,
    alive: Vec<bool>,
    free: Vec<u32>,
}

/// Dense per-entity storage for one component type. Generation-checked,
/// so a stale `Entity` handle never reads a recycled slot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentStore<T> {
    slots: Vec<Option<(u32, T)>>,
}

impl<T> Default for ComponentStore<T> {
    fn default() -> Self {
        ComponentStore { slots: Vec::new() }
    }
}

impl<T> ComponentStore<T> {
    pub fn insert(&mut self, e: Entity, v: T) {
        let i = e.index as usize;
        if self.slots.len() <= i {
            self.slots.resize_with(i + 1, || None);
        }
        self.slots[i] = Some((e.generation, v));
    }
    pub fn get(&self, e: Entity) -> Option<&T> {
        match self.slots.get(e.index as usize)? {
            Some((g, v)) if *g == e.generation => Some(v),
            _ => None,
        }
    }
    pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
        match self.slots.get_mut(e.index as usize)? {
            Some((g, v)) if *g == e.generation => Some(v),
            _ => None,
        }
    }
    pub fn remove(&mut self, e: Entity) -> Option<T> {
        let slot = self.slots.get_mut(e.index as usize)?;
        match slot {
            Some((g, _)) if *g == e.generation => slot.take().map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.as_ref().map(|(g, v)| (Entity { index: i as u32, generation: *g }, v))
        })
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            s.as_mut().map(|(g, v)| (Entity { index: i as u32, generation: *g }, v))
        })
    }
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct Persistent {
    type_id: TypeId,
    save: fn(&(dyn Any + Send)) -> Result<Value, String>,
    load: fn(Value) -> Result<Boxed, String>,
}

fn save_impl<T: Serialize + 'static>(a: &(dyn Any + Send)) -> Result<Value, String> {
    let t = a.downcast_ref::<T>().ok_or("resource type mismatch")?;
    serde_json::to_value(t).map_err(|e| e.to_string())
}
fn load_impl<T: serde::de::DeserializeOwned + Send + 'static>(v: Value) -> Result<Boxed, String> {
    serde_json::from_value::<T>(v).map(|t| Box::new(t) as Boxed).map_err(|e| e.to_string())
}

/// All simulation state. Type-keyed resources plus entities and typed event queues.
/// Resources registered with `insert_persistent` are included in snapshots.
#[derive(Default)]
pub struct World {
    pub clock: Clock,
    entities: Entities,
    resources: HashMap<TypeId, Boxed>,
    persistent: BTreeMap<String, Persistent>,
    events: HashMap<TypeId, Boxed>,
    commands: BTreeMap<String, Command>,
}

/// A named action a module offers to the outside (scripts, the API, tools).
/// Takes JSON in, gives JSON back, so the core stays ignorant of what it does.
pub type Command = std::sync::Arc<dyn Fn(&mut World, &Value) -> Result<Value, String> + Send + Sync>;

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- entities ----
    pub fn spawn(&mut self) -> Entity {
        if let Some(i) = self.entities.free.pop() {
            self.entities.alive[i as usize] = true;
            Entity { index: i, generation: self.entities.generations[i as usize] }
        } else {
            self.entities.generations.push(0);
            self.entities.alive.push(true);
            Entity { index: (self.entities.generations.len() - 1) as u32, generation: 0 }
        }
    }
    pub fn despawn(&mut self, e: Entity) -> bool {
        if !self.is_alive(e) {
            return false;
        }
        self.entities.alive[e.index as usize] = false;
        self.entities.generations[e.index as usize] += 1;
        self.entities.free.push(e.index);
        true
    }
    pub fn is_alive(&self, e: Entity) -> bool {
        let i = e.index as usize;
        self.entities.alive.get(i).copied().unwrap_or(false) && self.entities.generations[i] == e.generation
    }
    pub fn entity_count(&self) -> usize {
        self.entities.alive.iter().filter(|a| **a).count()
    }

    // ---- resources ----
    /// Insert state that is NOT saved in snapshots (caches, scratch data).
    pub fn insert<T: Send + 'static>(&mut self, v: T) {
        self.resources.insert(TypeId::of::<T>(), Box::new(v));
    }
    /// Insert state that IS saved/restored under `name`.
    pub fn insert_persistent<T>(&mut self, name: &str, v: T)
    where
        T: Serialize + serde::de::DeserializeOwned + Send + 'static,
    {
        self.persistent.insert(
            name.to_string(),
            Persistent { type_id: TypeId::of::<T>(), save: save_impl::<T>, load: load_impl::<T> },
        );
        self.insert(v);
    }
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.resources.get(&TypeId::of::<T>())?.downcast_ref()
    }
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut(&TypeId::of::<T>())?.downcast_mut()
    }
    /// Temporarily remove a resource so it can be mutated alongside others.
    /// Always `put` it back.
    pub fn take<T: 'static>(&mut self) -> Option<T> {
        self.resources.remove(&TypeId::of::<T>())?.downcast().ok().map(|b| *b)
    }
    pub fn put<T: Send + 'static>(&mut self, v: T) {
        self.insert(v);
    }

    // ---- events (same-tick; undrained events are cleared at end of tick) ----
    pub fn emit<E: Send + 'static>(&mut self, e: E) {
        self.events
            .entry(TypeId::of::<E>())
            .or_insert_with(|| Box::new(Vec::<E>::new()))
            .downcast_mut::<Vec<E>>()
            .expect("event queue type")
            .push(e);
    }
    pub fn drain_events<E: Send + 'static>(&mut self) -> Vec<E> {
        match self.events.get_mut(&TypeId::of::<E>()).and_then(|b| b.downcast_mut::<Vec<E>>()) {
            Some(v) => std::mem::take(v),
            None => Vec::new(),
        }
    }
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    // ---- commands ----
    /// Offer an action under `name` (convention: `verb_noun`, e.g. `remove_body`).
    pub fn register_command(&mut self, name: &str, f: impl Fn(&mut World, &Value) -> Result<Value, String> + Send + Sync + 'static) {
        self.commands.insert(name.to_string(), std::sync::Arc::new(f));
    }
    pub fn call(&mut self, name: &str, args: &Value) -> Result<Value, String> {
        let c = self.commands.get(name).cloned().ok_or_else(|| format!("unknown command '{name}' (known: {})", self.command_names().join(", ")))?;
        c(self, args)
    }
    pub fn command_names(&self) -> Vec<String> {
        self.commands.keys().cloned().collect()
    }

    // ---- persistence ----
    pub fn save_persistent(&self) -> Result<BTreeMap<String, Value>, String> {
        let mut out = BTreeMap::new();
        for (name, p) in &self.persistent {
            let r = self.resources.get(&p.type_id).ok_or_else(|| format!("missing resource '{name}'"))?;
            out.insert(name.clone(), (p.save)(r.as_ref())?);
        }
        Ok(out)
    }
    pub fn load_persistent(&mut self, data: &BTreeMap<String, Value>) -> Result<(), String> {
        // Validate everything first so a bad snapshot leaves the world untouched.
        let mut staged = Vec::new();
        for (name, v) in data {
            let p = self.persistent.get(name).ok_or_else(|| format!("no module registered state '{name}'"))?;
            staged.push((p.type_id, (p.load)(v.clone()).map_err(|e| format!("{name}: {e}"))?));
        }
        for (id, b) in staged {
            self.resources.insert(id, b);
        }
        Ok(())
    }
}
