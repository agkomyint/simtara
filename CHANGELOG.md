# Changelog

## Unreleased

- **Build screen in Studio.** "Build your own" template: Luau editor (colours, line numbers, errors at the line), live charts of your own measurements, a table of every thing with editable values, a rules panel (toggle, errors, typo warnings), a command reference with Insert buttons, examples (empty, idea spreads, typed Luau, epidemic), and a **Build & test** tab with checks like `people who know > 100`.
- **Build and test from the command line.** `simtara script FILE --ticks N --expect "metric >= 10"` prints a JSON report (errors, measurements over time, things, rules, warnings, checks, repeatability) and exits 1 on failure. `simtara describe` lists everything a script can use, for coding agents.
- **Things and rules (`simtara-things`).** Domain-neutral things with properties and relations, script-defined measurements, `setup()`, and native rules (formulas run in parallel on every thing of a kind each tick; 100,000 things x 5 ticks about 120 ms, repeatable). A `warnings` command flags properties a rule reads that nothing sets.
- **Scripting moved from Lua 5.4 to Luau** (typed, built-in sandbox).
- **Orbital gravity.** New `simtara-orbits` crate: Sun, Earth and Moon in real SI units, leapfrog integration, energy conserved. One tick = 6 hours, so 1461 ticks = one year. Studio gets a "Sun, Earth & Moon" template with follow/zoom view and trails.
- **Event scripting (Lua, now Luau).** New `simtara-script` crate: a sandboxed Lua script saved with the universe, `on_tick(t)` runs every tick (for example `remove_body("Moon")`). No files, network or randomness; runaway loops are stopped and reported. Modules expose actions through a new generic command registry in core (`World::register_command` / `call`).
- **Script experiments.** An experiment can now change the script instead of a law ("What happens if the Moon disappears?"), and compare any measurement of the two runs.
- **Save and reopen universes.** New `simtara-store` crate stores a universe as a folder (`universe.json` + `state.snap.json`, with a `schema_version`) under `Documents/SimTara/universes/` (override with `SIMTARA_HOME`). Studio launcher gets a "My universes" list (open, delete) and a Save button; leaving a saved universe autosaves it. `simtara-studio --open "<name>"` skips the launcher.
- **Experiments.** In the Studio, change one law (gravity, friction, range, core repel, charge), run the original and the changed version from the same moment on CPU, and see a two-line chart. Results are saved with the universe (`experiment.json`, `results.json`, the exact `start.snap.json`, and a run record: engine version, seed, backend, time) and shown again on reopen.
- `serde_json` now uses `float_roundtrip`, so saved numbers reload bit-for-bit (found by a test: results changed in the last digit after a save).
- **License changed from MIT to Elastic License 2.0** (no hosted/managed-service use). v0.1.0 and v0.2.0 remain MIT. Added ENTERPRISE.md describing the planned commercial tier.

## v0.2.0

- New `simtara-studio` app: launcher with parameter sliders, then live universe with control panel (play/pause, live laws, particle count, CPU/GPU, metrics graph).

- New `simtara-api` crate: JSON control interface (create/run/observe/configure/branch/snapshot worlds) for CLI, Studio and AI agents. `simtara serve` speaks it as JSON lines over stdin/stdout.
- Core: metric registry (`Builder::add_metric`, `Simulation::collect_metrics`); particles register `particles.count`, `particles.kinetic_energy`, `particles.mean_neighbors`.
- New `simtara-gpu` crate: wgpu + WGSL particle force kernel, registered as `Backend::Gpu`; automatic CPU fallback.
- Core: `Builder.compute` (modules can register backends), `Compute::active()`.
- Particles: shared `build_grid` / `accelerations_cpu` / `apply_acceleration` so CPU and GPU share one contract.
- CLI: `--backend gpu|cpu`, `G` key to switch live, ms/tick reporting.

## v0.1.0

First release.

- `simtara-core`: domain-neutral runtime — world, entities/components, ticks, staged systems,
  events, config, snapshots, module interface, compute-backend selection (CPU only for now).
- `simtara-particles`: particle module with attraction/repulsion, friction, bounce/wrap boundaries,
  live-tunable parameters.
- `simtara-cli`: `simtara run` (headless, save/load/PNG) and `simtara view` (live window).

Not yet implemented in v0.1.0: GPU backend, slider UI (both added in v0.2.0).
