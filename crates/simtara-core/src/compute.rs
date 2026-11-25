use serde::{Deserialize, Serialize};

/// Where a system's heavy math may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    Cpu,
    /// Provided by a GPU module (e.g. a future `simtara-gpu` crate using wgpu).
    Gpu,
}

/// Tells systems which backends exist and which one the user asked for.
/// A GPU module calls `register(Backend::Gpu)`; systems call `resolve`
/// with what they support and get the best available choice, falling back to CPU.
#[derive(Clone, Debug)]
pub struct Compute {
    requested: Backend,
    available: Vec<Backend>,
}

impl Compute {
    pub fn new(requested: Backend) -> Self {
        Compute { requested, available: vec![Backend::Cpu] }
    }
    pub fn register(&mut self, b: Backend) {
        if !self.available.contains(&b) {
            self.available.push(b);
        }
    }
    pub fn requested(&self) -> Backend {
        self.requested
    }
    pub fn set_requested(&mut self, b: Backend) {
        self.requested = b;
    }
    pub fn is_available(&self, b: Backend) -> bool {
        self.available.contains(&b)
    }
    /// The backend that will actually run: the requested one if available, else CPU.
    /// Competing implementations of one rule (CPU system, GPU system) each gate on this.
    pub fn active(&self) -> Backend {
        if self.available.contains(&self.requested) {
            self.requested
        } else {
            Backend::Cpu
        }
    }
    /// Pick the backend a system should use, given the ones it implements.
    pub fn resolve(&self, supported: &[Backend]) -> Backend {
        if supported.contains(&self.requested) && self.available.contains(&self.requested) {
            return self.requested;
        }
        supported
            .iter()
            .copied()
            .find(|b| self.available.contains(b))
            .unwrap_or(Backend::Cpu)
    }
}
