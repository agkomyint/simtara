use crate::compute::Backend;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Universe configuration. The core only owns seed/dt/backend;
/// each module reads its own section by name.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub seed: u64,
    /// Fixed timestep per tick, in simulated seconds.
    pub dt: f64,
    pub backend: Backend,
    pub modules: BTreeMap<String, Value>,
}

impl Default for Config {
    fn default() -> Self {
        Config { seed: 1, dt: 1.0 / 60.0, backend: Backend::Cpu, modules: BTreeMap::new() }
    }
}

impl Config {
    /// Read a module's section, falling back to its `Default` if absent or invalid.
    pub fn module<T: DeserializeOwned + Default>(&self, name: &str) -> T {
        self.modules
            .get(name)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default()
    }
    pub fn set_module<T: Serialize>(&mut self, name: &str, value: &T) {
        if let Ok(v) = serde_json::to_value(value) {
            self.modules.insert(name.to_string(), v);
        }
    }
}
