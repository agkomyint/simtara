# Learning log

A running notebook of decisions and questions. Newest first. Add to it whenever we decide something or get confused.

## Open questions

- Who is the very first customer: a teacher, a lab, or a studio? (decides which templates come first)
- How should hosted compute be priced?
- When do customers get to write their own rules? (needs a scripting or rules-as-data design)
- Do we need to guarantee that GPU results match CPU results, or is "very close" enough?

## Decisions

### 2026-11-25: 0.3.0 built (save/reopen + experiments)
**Decision:** new `simtara-store` crate holds all the file logic with no UI; the Studio only calls it. Experiments always start from a copy of the current state, which is saved with the experiment (`start.snap.json`), and run on CPU.
**Why:** keeping files out of the UI lets the CLI, API and a future hosted service reuse them. Saving the start state makes a result repeatable even after the universe changes. CPU keeps results exact.
**Learned:** a test caught that serde_json's default float parsing is not exact, so reloaded numbers differed in the last digit. Fixed with the `float_roundtrip` feature. Worth remembering for anything that must repeat bit-for-bit.
**Not done yet:** the universe screen was only smoke-tested (it ran 10 s with a saved experiment open, no crash), not clicked through by a person. Try it and note anything odd under Open questions.

### 2026-11-25: Scripting: Python outside, Lua inside
**Decision:** a Python SDK over the existing JSON API controls SimTara from outside (run experiments, analyze). Lua is the scripting language inside the runtime, for setup and events ("at tick 5000 remove the Moon"). WASM stays an option later for sandboxed packages.
**Why:** Lua is the lightest to embed (about 200 KB, one Rust crate `mlua`) and easy to learn. Python is what scientists and agents already use, but is too slow and hard to sandbox inside the runtime.
**Notes:** Lua is too slow to call once per particle, so heavy per-particle math stays in compiled modules or data-described rules. For hosted compute, run Lua in an isolated process with time and memory limits. Not built yet; comes after 0.3.0.

### 2026-11-25: Keep the word "universe" (we tried "simulation" and went back)
**Decision:** user-facing words are Template (ready-made starting point) -> Universe (what the user creates and owns) -> Experiment -> Run, inside a Project. "Universe type" is retired; say "template".
**Why:** "universe" first felt vague because it meant three things (the kind, your copy, the running world). Fixed by giving it one meaning: *your creation*. "Simulation" sounds like copying something that already exists, but SimTara lets people invent rules and ask "what if", which fits "universe" better. It also suits the brand.
**Notes:** the code types `World` and `Simulation` keep their names; `concepts.md` maps the words.
**Example project:** "What happens if the Moon disappears?" Model the Sun, Earth and Moon with real physics, remove the Moon in an experiment, compare against the unchanged universe.

### 2026-11-25: Shrink 0.3.0 to two features
**Decision:** 0.3.0 = save/reopen a universe + run an experiment and see a two-line result. Everything else moved to `roadmap/vision.md`.
**Why:** the first plan (projects, templates, docking panels, sweeps, sharing, server) was too many new ideas at once to understand or test.
**Rejected:** building the project layer and templates first. They are useful but are not needed to prove the core loop.

### 2026-11-25: License changed to Elastic License 2.0
**Decision:** core is source-available under ELv2 (no hosting it as a service for others). Enterprise features will be under a separate commercial license (see `ENTERPRISE.md`).
**Why:** the plan is a free desktop app with paid hosted compute, and MIT would let anyone host it and compete.
**Notes:** v0.1.0 and v0.2.0 stay MIT. This is not "open source" in the OSI sense. A lawyer should review before selling.

### 2026-11-25: Sell the experiment platform first
**Decision:** first product = ready-made universes users tune, run, compare. User-defined rules come later.
**Why:** it builds directly on what exists. Writing new rules needs a scripting layer, which is a big separate project.

### 2026-11 : Control API before more features
**Decision:** build `simtara-api` (JSON commands) and `simtara serve` before the Studio grows.
**Why:** the CLI, Studio and AI agents should all use one interface, and a later HTTP server only wraps it.

### 2026-11: GPU as a module, not part of the core
**Decision:** `simtara-gpu` registers a backend and a system through the normal module interface.
**Why:** proves the "plug in, don't modify the runtime" rule. If the GPU is missing the engine uses CPU.

## Things we learned

- Particles interact only within a radius, so a grid (cell size >= radius) turns an N-squared problem into near-linear work. It is built on the CPU and shared by both backends.
- For this workload the GPU was about 7 to 9 times faster than the CPU in our test.
- wgpu 30 returns a Result from `get_mapped_range()`; egui 0.37 uses `fn ui` and `Panel` instead of the older `update` and `SidePanel`. Library APIs change between versions, so read the compiler errors and the library source.
- A saved `.snap.json` is the key to branching, resuming and (later) sharing.
