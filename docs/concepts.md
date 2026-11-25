# Concepts in plain English

Each term says what it is, then where it lives in the code.

## The big idea

SimTara is a **runtime**: an engine that holds a simulated world and moves time forward. It does not know what a particle or a bird is. Specific things (particles, later boids or ecosystems) are added as **modules**. The rule we keep: *new capabilities plug in; they never require changing the engine.*

Think of a game console (runtime) and games (modules). The console does not know any game.

## Words you will see

**World.** All the data of one universe at one moment: entities, their components, shared resources. `simtara-core/src/world.rs`.

**Entity / component.** An entity is just an id. A component is a piece of data attached to it (a position, a health). Entity ids carry a "generation" so a deleted entity's id can't be mistaken for a new one.

**Resource.** Data that belongs to the world as a whole, not to one entity. Our particles live in a resource named `particles`, and the laws live in `particle_params`. Resources registered as *persistent* are saved in snapshots.

**Tick.** One step of simulated time (default 1/60 s). Everything advances one tick at a time. Same start + same ticks = same result.

**System.** A rule that runs once per tick, like "apply forces" or "move particles". Systems run in a fixed order inside three stages: PreUpdate, Update, PostUpdate. `simtara-core/src/sim.rs`.

**Module.** A bundle that installs resources, systems and metrics into a universe. Particles are a module (`simtara-particles`). The GPU backend is a module too (`simtara-gpu`).

**Simulation (in the code).** The engine object that owns a world plus its systems and runs ticks: play, pause, speed, reset, snapshot, restore. `Simulation` in `sim.rs`.

**Config.** The starting settings: seed, tick length, preferred backend, and each module's own settings. Rebuilding from the config gives tick 0 again.

**Seed.** A number that makes randomness repeatable. Same seed = same universe.

**Snapshot.** The whole saved state at one tick, as JSON. You can restore it later, or copy it to make a branch.

**Backend (compute).** Where the heavy math runs: CPU or GPU. If the GPU isn't available the engine falls back to CPU. `simtara-core/src/compute.rs`.

**Metric.** A named number measured from the world, such as `particles.kinetic_energy`. Metrics are how experiments compare results. A module registers them.

**Event.** A short-lived message that systems can send each other within one tick. Cleared at the end of the tick.

## Words from the product plan

**Template.** A ready-made starting point you pick in the launcher, such as "Particle Life". Technically: a module plus a preset config.

**Universe (what you see in the app).** The thing you create from a template, tune, save and reopen. It is a world plus its settings. This is the word users see; in the code it is a `Simulation` holding a `World`.

**Saved universe.** A folder on disk with the universe's settings and its snapshot. Open it later and it continues from the same tick. Lives in `simtara-store`.

**Experiment.** A change applied to a universe's starting state plus the runs and the measured results. "Gravity 4 instead of 2, run 5,000 ticks."

**Run.** One execution of an experiment. A run record keeps the engine version, seed and backend so it can be repeated.

**Branch.** A copy of a world that can then be changed separately. Snapshots make this cheap.

**Project.** A folder of related universes and experiments (planned, see the vision doc).

**Client.** Anything that talks to the runtime: the CLI, the Studio, an AI agent, a future web app. They are separate from the runtime.

**API.** The command interface (JSON in, JSON out) that clients use. `simtara-api`. `simtara serve` exposes it over stdin/stdout.

## Common questions

**Why are experiments run on the CPU?** CPU runs repeat exactly: same start state, same numbers. GPU runs are very close but not guaranteed identical, so for results people may cite we use the CPU.

**What makes a result trustworthy later?** Each experiment keeps its exact start state and a run record (engine version, seed, backend, time).

**Is the runtime the same as the world?** No. The runtime is the engine (code). The world is the data it is running. You can have many worlds in one runtime.

**Where do the rules live?** In compiled Rust systems. Users can change parameters of existing rules but can't write new rules yet. That would need a scripting or rules-as-data feature, which is a later idea.

**Is a GPU run identical to a CPU run?** Very close but not guaranteed bit-for-bit. CPU runs are exactly repeatable. This matters for results people will cite.
