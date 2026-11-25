# SimTara

A domain-neutral simulation runtime. The core knows nothing about particles, cells or atoms —
it manages state, runs ordered systems, advances time, and snapshots the world. Everything
else is a **module** that plugs in.

```
crates/
  simtara-core       World · Time · Systems · Events · Compute backend · Config · Snapshots · Module trait
  simtara-particles  Particle System v0: attraction / repulsion / collision (first module)
  simtara-cli        `simtara run` (headless) and `simtara view` (live window)
```

## Try it

```
cargo run --release -p simtara-cli -- view
cargo run --release -p simtara-cli -- run --particles 5000 --steps 600 --png out.png --save u.snap.json
cargo run --release -p simtara-cli -- run --load u.snap.json --steps 300
cargo test --workspace
```

Viewer keys: `Space` play/pause · `R` reset · `Up/Down` gravity · `PgUp/PgDn` particle count ·
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

**GPU:** the `Compute` abstraction (`Backend::Cpu | Gpu`, with CPU fallback) is in the core, but
there is no GPU backend yet. Next step is a `simtara-gpu` crate (wgpu + WGSL) that registers
`Backend::Gpu` and a GPU force kernel for the particle module.
