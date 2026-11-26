# SimTara: notes for coding agents

SimTara is a domain-neutral simulation runtime in Rust. The core knows nothing about fire, people or planets. Domains are modules plus Luau scripts. License: Elastic License 2.0 (no hosted/managed-service use).

Vocabulary: Template -> Universe -> Experiment -> Run. A Project is a folder. Packages and hosted compute are planned, not built.

## Crates (core must never depend on CLI, Studio or API)

| crate | job |
|---|---|
| `simtara-core` | `Simulation`, `World` (clock, systems, events, snapshots, command registry: `register_command` / `call` / `command_names`), compute backend |
| `simtara-things` | Things (properties, relations), rules (formula language), metrics, params/roles, view, camera, light |
| `simtara-fields` | Fluids in SI units: gases (low-Mach, ideal gas law, CPU + GPU), water (APIC, CPU + GPU), floods and tsunamis (shallow water, `shallow.rs`); commands `fluid solid emitter water gas object fluid_metric fluid_info` and `flood ground water_level rain river gate quake wave flood_metric flood_info`; any number may name a param |
| `simtara-script` | Luau sandbox: `setup()` runs once at tick 0, `on_tick(t)` each tick; every registered command is a Luau function taking one table |
| `simtara-store` | Saves universes and experiments as folders |
| `simtara-studio` | egui desktop app (launcher, Build screen, 3D view) |
| `simtara-cli` | binary `simtara` (headless build/test, `describe`) |
| `simtara-particles`, `simtara-orbits`, `simtara-gpu` | example domain modules and GPU backend |
| `simtara-api` | HTTP API (so far only for the older modules; things/rules not exposed yet) |

## Fastest way to work: headless

```
cargo build --release
./target/release/simtara describe                       # every command, rule function, metric op (JSON)
./target/release/simtara script FILE.luau --ticks 300   # JSON report: ok, error, metrics, series, rules, warnings, repeatable
./target/release/simtara script FILE.luau --ticks 300 --expect "metric >= 10"   # exit 1 on failure
```

Rebuild the whole workspace (`cargo build --release`) before using the CLI after changing library crates; a stale binary reports "attempt to call a nil value" for new commands. Check `warnings` in the report: it flags properties a rule reads that nothing sets. Runs are deterministic for a seed (`repeatable`).

Tests: `cargo test --release --workspace`. The examples test requires every bundled example to run with no errors and no warnings.

## Script model (Luau)

```lua
function setup()
  param{name="oxygen", value=1, min=0, max=1, about="...", owner="nature"}   -- slider; rules read it by name
  local id = spawn{kind="cell", props={x=1, y=2, fuel=1}}                    -- props: number, text, bool only
  relate{from=id, to=other, kind="next"}
  add_rule{name="burn", kind="cell", set={fuel="max(0, fuel - 0.05*oxygen)"}}
  define_metric{name="fuel left", kind="cell", prop="fuel", op="sum"}        -- ops: count sum mean min max
  view{x="x", y="y", color="temp", size="fuel"}                               -- 2D; add z, shape, mode="3d" for 3D
  camera{yaw=35, pitch=25, distance=0.9, fov=50, follow=0}
  light{yaw=300, pitch=55, ambient=0.3, strength=0.8, ground=true, grid=true}
end
```

Rules are formula strings, not Luau: this is why large universes stay fast (native, parallel per kind). Luau only does setup and glue, so avoid per-thing Luau loops each tick.

Formula language: properties, params, `+ - * / %`, comparisons, `and or not`, `random() tick() min max abs sqrt floor clamp(x,lo,hi) if(c,a,b)`. Neighbour aggregates over a relation: `ncount nsum nmean nmax nmin wsum wmean`, e.g. `nmean(next, temp)`. Global aggregates over a whole kind: `gcount gsum gmean gmax gmin`, e.g. `gsum(cell, water_used)`, computed before the rule runs (use these for finite shared resources). Rules in a kind run in the order added and see earlier rules' results on the same thing; neighbour reads see the previous state of neighbours.

Quirks: params resolve after thing properties (a property with the same name wins). Rule `set` values for strings are not supported; use numbers. Missing `view` picks x, y, z properties by name.

## Roles and experiments

- `param{owner="nature"}` (comma-separated for several). `play_as{role}` locks every control not owned by that role (empty = everyone). `set_param` enforces it; Studio shows a lock.
- Experiment on a Build universe: both runs rebuilt from tick 0 with the same seed. The second run may change the script and/or `param_overrides`; an override of a control the role does not own is refused.

## Studio and files

- Pictures and meshes without a window: `simtara-studio --picture FILE.luau --ticks 30,80 --out frame [--obj] [--size 960x600]` (PNG of the 3D view; `--obj` adds the liquid surface mesh). Look at the PNGs to check visual changes.
- Run Studio: `simtara-studio` (`--open "<name>"`, or `--new NAME --example "title substring"`). Kill a running copy before rebuilding on Windows: `taskkill //F //IM simtara-studio.exe`.
- Saved universes live in `$SIMTARA_HOME` or `Documents/SimTara/universes/<name>/`: `universe.json`, `state.snap.json`, `script.luau`, `models/*.obj`, experiment files (`experiment.json`, `results.json`, `start.snap.json`).
- 3D: drawn on the GPU (`view3d_gpu.rs`, wgpu offscreen texture shown as an egui image) with the CPU egui painter (`view3d.rs`) as fallback and behind the GPU checkbox. Both share the camera maths and mouse picking in `view3d.rs`; GPU tests read pixels back and skip when there is no adapter. `shape` property is `sphere | cube | pyramid | <model file name>`; `heading` (degrees) rotates a thing. `.obj` models are z-up, normalised to size 1; put `# simtara up=y` in the header for y-up files. Over 20,000 things it draws squares; models are capped near 350k triangles per frame.
- Examples live in `examples/*.luau` (`fire`, `reaction`, `idea`, `balls3d`, `airflow`, `smoke`, `plume`, `water`, `pour`, `play`, `tsunami`, `flood`) and are embedded in `crates/simtara-studio/src/build.rs` (`EXAMPLES`, `REFERENCE` arrays have fixed lengths: update the number when adding).

## Known gaps / backlog

Fire example costs ~9 ms/tick for 3,600 cells (unexplained; 100k things x 5 ticks is ~120 ms); no bulk spawn (`grid{}`, `spawn_many`); no spatial neighbour-by-distance query; liquids run on the GPU (`gpu_liquid.wgsl`, CPU fallback; GPU and CPU agree closely but not bit-for-bit), gases on the GPU too (~13 ms/step at 128k cells, CPU ~140 ms); floods on the CPU only (~100 ms/tick at 30k cells); fluids are not coupled to things; fluid snapshots are large JSON; no `use("module")`; things/rules not in the HTTP API; no CLI `--set name=value`; crew cost is paid even when no fire; people/water/maps modules, Python SDK, `.simtara` packages and hosted compute are not started.

## Conventions

- Keep the core domain-neutral; put domain ideas in modules or scripts.
- Do not push or publish without the owner's say-so; commits are local until asked.
- Windows machine: Git Bash and PowerShell both work; Python in Windows cannot read bash `/tmp`.
- Keep `CHANGELOG.md` current for user-visible changes.
