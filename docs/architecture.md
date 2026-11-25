# Architecture

## Crates (one folder each in `crates/`)

| Crate | Job | Depends on |
|---|---|---|
| `simtara-core` | The runtime: world, ticks, systems, modules, snapshots, metrics, compute backend choice. Knows nothing about particles or UI. | nothing of ours |
| `simtara-particles` | First module: particle physics (attract/repel, friction, bounds), its metrics, the CPU force code. | core |
| `simtara-gpu` | A module that adds a GPU version of the particle forces using wgpu + a WGSL shader. | core, particles |
| `simtara-api` | The control layer: JSON commands to create/run/observe/configure/branch worlds. | core, particles, gpu |
| `simtara-store` | Saves and loads universes and experiments as folders; runs "original vs changed" experiments. No UI. | core, particles |
| `simtara-cli` | The `simtara` program: `run`, `view`, `serve`. | core, particles, gpu, api |
| `simtara-studio` | The desktop app: launcher with saved universes, live view, control panel, experiments and chart. | core, particles, gpu, store |

Direction of dependency: clients (cli, studio) sit on top; the core depends on nothing above it.

```
   Studio (GUI)      CLI        AI agent / scripts
        |             |               |
        |             +--- serve -----+   (JSON lines)
        |                     |
        |                simtara-api  (Engine)
        |                     |
        +----------+----------+
                   |
             simtara-core  <--- modules plug in here
              ^        ^
     simtara-particles  simtara-gpu
```

## One tick, step by step

1. `Simulation::step` goes through its systems in stage order.
2. With the GPU module active: `gpu.forces` (PreUpdate) computes accelerations on the GPU and stores them.
3. `particles.forces` (Update) computes accelerations on the CPU, but only if the CPU backend is active.
4. `particles.integrate` updates velocities and positions from the accelerations.
5. `particles.boundary` keeps particles inside (bounce) or wraps them.
6. Events are cleared and the clock advances.

## The API flow

`{"cmd":"experiment.run","args":{...}}` -> `Engine::handle` -> finds the named world -> runs ticks, sampling metrics -> returns `{"ok":true,"result":{...}}`. The engine holds many named worlds, so a client can branch one and compare. Every command is listed by `{"cmd":"capabilities"}`.

## Saving and experiments (0.3.0)

```
Documents/SimTara/universes/<name>/
  universe.json        name, template, summary, tick, seed, times, schema_version
  state.snap.json      the saved world (engine Snapshot)
  experiments/exp-001/
    experiment.json    which law, original vs changed value, ticks, metric
    results.json       two series of [tick, value] + run record
    start.snap.json    the exact state the experiment began from
```

Save = write the snapshot (to a temp file, then rename, so a crash can't leave half a save). Open = build the template from the snapshot's config, then restore the state. An experiment copies the start state twice, sets the law to each value, runs on CPU, and samples a metric. Code: `crates/simtara-store/src/lib.rs`.

## Rules we keep

1. Core never imports a module, the API, or any UI.
2. New features arrive as modules or clients.
3. Same seed + same config + same ticks = same result (on CPU).
4. Saved files carry a version number.

## Where things are

- Runtime loop and Module trait: `crates/simtara-core/src/sim.rs`
- World storage: `crates/simtara-core/src/world.rs`
- Particle physics: `crates/simtara-particles/src/lib.rs`
- GPU shader: `crates/simtara-gpu/src/forces.wgsl`
- API commands: `crates/simtara-api/src/lib.rs`
- Saving and experiments: `crates/simtara-store/src/lib.rs`
- Studio UI: `crates/simtara-studio/src/main.rs`

## Run it

- Tests: `cargo test --workspace --release`
- Studio: `cargo run --release -p simtara-studio`
- Headless: `cargo run --release -p simtara-cli -- run --particles 8000 --steps 600`
- Open a saved universe directly: `cargo run --release -p simtara-studio -- --open "Universe 1"`
- API: `cargo run --release -p simtara-cli -- serve`, then type a JSON line.
