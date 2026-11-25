# Changelog

## Unreleased

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

Not yet implemented: GPU backend (wgpu), slider UI.
