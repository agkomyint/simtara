# Changelog

## v0.1.0

First release.

- `simtara-core`: domain-neutral runtime — world, entities/components, ticks, staged systems,
  events, config, snapshots, module interface, compute-backend selection (CPU only for now).
- `simtara-particles`: particle module with attraction/repulsion, friction, bounce/wrap boundaries,
  live-tunable parameters.
- `simtara-cli`: `simtara run` (headless, save/load/PNG) and `simtara view` (live window).

Not yet implemented: GPU backend (wgpu), slider UI.
