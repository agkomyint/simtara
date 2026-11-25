# SimTara

A domain-neutral simulation runtime. The core knows nothing about particles, cells or atoms —
it manages state, runs ordered systems, advances time, and snapshots the world. Everything
else is a **module** that plugs in.

```
crates/
  simtara-core       World · Time · Systems · Events · Compute backend · Config · Snapshots · Module trait
  simtara-particles  Particle System v0: attraction / repulsion / collision (first module)
  simtara-gpu        GPU backend: wgpu + WGSL particle force kernel, CPU fallback
  simtara-cli        `simtara run` (headless) and `simtara view` (live window)
```

## Try it

```
cargo run --release -p simtara-cli -- view
cargo run --release -p simtara-cli -- run --particles 5000 --backend gpu --steps 600 --png out.png --save u.snap.json
cargo run --release -p simtara-cli -- run --load u.snap.json --steps 300
cargo test --workspace
```

Viewer keys: `G` gpu/cpu · `Space` play/pause · `R` reset · `Up/Down` gravity · `PgUp/PgDn` particle count ·
`Left/Right` speed (0.1x–10x) · `B` bounce/wrap · `S` save · `L` load · `Esc` quit.

## Adding a module (no core changes)

```rust
struct Chemistry;
impl Module for Chemistry {
    fn name(&self) -> &str { "chemistry" }
    fn build(&self, b: &mut Builder) {
        b.world.insert_persistent("molecules", Molecules::default()); // saved in snapshots
        b.add_system(Stage::Update, ReactionSystem);                  // runs each tick
    }
}
sim.add_module(Chemistry);
```

## Status

Core v0.1 checklist: world lifecycle ✅ · entities/components ✅ · ticks & stepping ✅ · system
registration with stages ✅ · CPU execution (rayon) ✅ · event system ✅ · snapshots ✅ · config ✅ ·
pause/resume/reset ✅ · visualization ✅ · module interface ✅.

**GPU:** `simtara-gpu` (wgpu + WGSL) registers `Backend::Gpu` and a GPU version of the particle force rule. The CPU rule stays and takes over automatically if no GPU is found. Neither the core nor the particle module depends on wgpu. Measured on an RTX 3050 Ti laptop: 20k particles 15 ms -> 2 ms per tick, 100k particles 371 ms -> 43 ms per tick. A test checks the GPU kernel against the CPU reference.

Current GPU design: the CPU builds the spatial grid and applies velocities; the GPU computes the pair forces and the result is read back every tick. Keeping state resident on the GPU (integrate + draw there too) is the next optimization.

## License

Elastic License 2.0 - see [LICENSE](LICENSE). Free to use and modify, but you may not offer SimTara as a hosted or managed service. Organization features are planned under a separate commercial license, see [ENTERPRISE.md](ENTERPRISE.md).

Versions v0.1.0 and v0.2.0 were released under the MIT license and remain available under those terms.

## Control API (for tools and AI agents)

`simtara serve` reads one JSON request per line on stdin and writes one JSON response per line on stdout:

```
{"id":1,"cmd":"world.create","args":{"name":"u1","seed":7,"config":{"count":2000}}}
{"id":2,"cmd":"experiment.branch","args":{"name":"u1","new_name":"u2"}}
{"id":3,"cmd":"world.configure","args":{"name":"u2","resource":"particle_params","patch":{"gravity":4}}}
{"id":4,"cmd":"experiment.run","args":{"name":"u2","ticks":500,"sample_every":100}}
```

Send `{"cmd":"capabilities"}` for the full command list. Worlds are independent, so branch, change one law, run both, and compare metrics.
