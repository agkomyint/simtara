# Technical direction

Where the engineering should go next, and why. Companion to `possibilities.md` (what could exist) and `AGENTS.md` (how the code works today).

## Principles

1. **The core stays domain-neutral.** Fire, people and planets are modules and scripts, never core code.
2. **Core never depends on Studio, CLI or API.** They are clients of the same command registry.
3. **One way in.** Everything a human, a script, an agent or the API does is a command (`World::call`). If it is not a command, it cannot be scripted, tested or recorded.
4. **Deterministic by default.** Same seed + same commands at the same ticks = same universe, bit for bit. This is what makes experiments, replays, sweeps and videos trustworthy.
5. **Heavy work is native.** Luau describes and steers; formula rules, aggregates and rendering run in Rust.
6. **A mistake must never end a running universe.** (See live coding below.)

## Direction 1: Live code to visual (live coding that never falls over)

**Goal.** You type Luau while the universe runs. The picture changes as you type. A typo, a runtime error or an infinite loop never stops the universe or throws away its state. You shape the universe in real time, like a musician live-coding.

**What exists today**
- "⟳ Apply to running" swaps the script into the running universe without restarting (`simtara_script::set_source`); `setup()` does not run again.
- Script errors are reported with a line number; runaway loops are stopped by the sandbox; rules show their own errors and can be toggled.
- Controls (`param{}`) can be turned live; things can be edited live in the Things tab.
- Snapshots exist for the whole world.

**What is missing**
- Applying is manual, one button; it is not continuous.
- An error in a swapped script can still leave the universe in an unclear state.
- `setup()` creates things once, so editing the setup part of a script has no effect until a restart.

**Design**

1. **Last-good script.** The runtime keeps two scripts: *running* (last version that compiled and passed its dry run) and *draft* (what is in the editor). Only a good draft is promoted. A bad draft shows its error in the editor at the line; the universe keeps running on the last good script. The tick loop never sees a broken script.
2. **Dry run before promotion.** On each edit (debounced, ~300 ms) compile the draft in a throwaway Luau state, with a step/time budget. Check syntax, that referenced commands exist, formula rules parse (`warnings` already flags unknown properties). Only then swap.
3. **Transactional swap.** Take a cheap checkpoint (snapshot) before promoting. If the first tick after a swap fails (error, NaN storm, budget exceeded), roll back to the checkpoint and the last good script, and show why. Checkpoints are kept in a short ring (for example the last 10 seconds) so "undo" works on the universe, not only on text.
4. **Reconcile, don't restart.** Split a script into declarations (`param`, `add_rule`, `define_metric`, `view`, `camera`, `light`) and creation (`spawn`, `relate`). On a live edit:
   - declarations are **diffed by name** and replaced in place: a changed rule formula takes effect next tick, a new param appears as a slider (keeping its current value), a removed one disappears;
   - creation runs only for **new** lines/loops, or when the user asks for a "rebuild" (restart from tick 0). Never silently re-spawn everything.
   - to make the diff possible, `setup()` is run in a *recording* mode that returns a list of declarations; the engine applies the difference.
5. **Errors are values, not stops.** A runtime error inside `on_tick` skips that call, records the error with tick and line, and the universe continues. Repeated errors from the same site are collapsed and, after N in a row, that hook is paused (shown as ⚠, one click to resume) so it cannot flood the log or the frame time.
6. **Budgets.** Per-tick Luau instruction/time budget; a rule or hook that exceeds it is cut off for that tick. The frame never blocks on user code.
7. **Visual feedback.** Show what the edit changed: highlight the line that last took effect, flash changed rules, keep a small timeline of "script version N applied at tick T" so charts can be read against code changes (a vertical marker on every chart).
8. **Everything is recorded.** The sequence of (tick, script version / command) is stored with the run. Replaying it with the same seed reproduces the session exactly, which is also the basis for videos of live sessions.

**Phases**
1. Last-good script + dry run + error stays in the editor (small, high value; no new concepts).
2. Auto-apply on edit (toggle "live"), chart markers for script versions.
3. Checkpoint ring and rollback on first-tick failure.
4. Declaration diffing / reconcile instead of a full `set_source`.
5. Per-hook budgets and pausing; session recording and replay.

**Tests that must exist**
- A syntactically broken edit leaves tick count advancing and state identical to a run without the edit.
- A runtime error in a new rule rolls back and the universe state equals the pre-swap checkpoint.
- Changing a rule formula live changes behaviour from the next tick; the same session replayed from its record gives identical results.
- An infinite loop in a hook does not freeze a frame beyond the budget.

**Risks.** Divergence between "live-edited" and "script from scratch" universes (a live session can reach states a fresh run of the final script cannot). Mitigation: always show whether the current state is "script-pure" (no manual edits or partial reconciles since tick 0) and offer "re-run final script from tick 0" as a check.

## Direction 2: Headless rendering and video

See `possibilities.md` section 1. Move the 3D drawing in `view3d.rs` into a crate that renders to an image without a window; `simtara render` writes PNG frames and calls `ffmpeg`. Camera keyframes via `shot{}`. This also supplies screenshots for docs and tests (golden-image checks of the renderer).

## Direction 3: Scale

- **Profile the fire example first** (about 9 ms/tick for 3,600 cells against 121 ms for 5 ticks of 100,000 things): find what is slow before adding features. Candidates: relation lookups, per-rule allocation, `gsum` recomputation per rule, drawing.
- **Bulk creation** (`grid{}`, `spawn_many`) so setup loops leave Luau.
- **Spatial index** for neighbour-by-distance queries.
- **Columnar storage** for numeric properties (fast rules, easy GPU upload), keeping the thing/property API unchanged.
- **GPU rules** for formula rules over large kinds, behind the existing compute backend with CPU fallback and a test that CPU and GPU agree.

## Direction 4: Modules ("batteries")

`use("people")`, `use("water")`, `use("maps")`: a module is a Rust crate or a Luau package that registers commands, rules and views through the same registry. Needs a manifest (name, version, required commands, params it adds), namespacing so two modules cannot clash, and a test harness that runs a module's example headlessly.

## Direction 5: Observability and roles

Roles already own controls. Next: what each role can **see** (a view filter on properties and kinds), with optional delay and noise, applied in the view layer and in the API so an agent playing a role cannot read more than a human would.

## Direction 6: Packages, API, hosted compute

- `.simtara` package: universe + script + models + experiments + results + engine version, one file, reproducibility check on open.
- Expose things, rules, params, roles and camera over the HTTP API with the same command names; add a Python SDK as a thin wrapper.
- Hosted compute runs the same headless runner on sweeps; the licence (Elastic 2.0) forbids others offering SimTara as a managed service.

## Suggested order

1. Live coding phases 1-2 (last-good script, auto-apply): makes Studio feel alive and safe.
2. Profile the fire example (cheap, may reveal an easy win).
3. Headless renderer + `simtara render`.
4. Bulk spawn + live reconcile (phase 4).
5. Parameter sweeps in Studio.
6. Modules, observability, packages.
