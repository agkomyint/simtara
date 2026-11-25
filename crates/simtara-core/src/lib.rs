//! SimTara core runtime: a domain-neutral engine.
//!
//! The core knows how to hold state (`World`), advance time (`Simulation`),
//! run ordered rules (`System`), pick a compute backend (`Compute`), emit
//! events, and snapshot/restore state. It does not know what a particle,
//! cell or atom is. Everything domain-specific arrives through a `Module`.

pub mod compute;
pub mod config;
pub mod rng;
pub mod sim;
pub mod world;

pub use compute::{Backend, Compute};
pub use config::Config;
pub use rng::Rng;
pub use sim::{Builder, Context, Module, RunState, Simulation, Snapshot, Stage, System};
pub use world::{Clock, ComponentStore, Entity, World};
