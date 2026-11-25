# SimTara long-term vision (not a plan for one release)

This is the big picture we are steering toward. The next concrete release is described in [v0.3.0.md](v0.3.0.md). Do not try to build all of this at once.

Status: draft for discussion. Goal: turn SimTara from a runtime with a viewer into a product people can work in and pay for.

## The product model

```
Workspace
 └─ Project            "Crowd safety study"
     └─ Universe       a configured instance of a template (particles, boids, ...)
         └─ Experiment a recorded change + runs, e.g. "gravity 4, 20 seeds, 5000 ticks"
             └─ Results  metric series, comparisons, snapshots, exports
```

- **Module**: the code for one kind of universe (particles today; boids, life, ecosystem next).
- **Template**: a module + a ready-made config + description + preview. What the launcher shows.
- **Universe**: one instance created from a template. Has a fixed starting state (snapshot 0) and a seed.
- **Experiment**: starting point (universe snapshot) + changes (parameter patches) + run plan (ticks, seeds, sample rate) + the metrics to record. Immutable once run, so results stay trustworthy.
- **Run**: one execution of an experiment with a manifest (engine version, module version, config hash, seed, backend, timings).

Most of the engine for this exists: worlds, `experiment.branch`, `experiment.run`, metrics, snapshots, deterministic seeding. v0.3.0 is mostly persistence, structure and UI.

## Who buys it (decide before building too far)

| Direction | Customer does | Needs | Effort |
|---|---|---|---|
| **A. Experiment platform** (recommended first) | Pick a shipped template, tune it, run experiments, compare results | Projects, experiments, results, templates | Medium; builds on current engine |
| **B. Rule builder** | Define their own rules/templates | Rules-as-data or scripting, sandboxing, validation | Large; later (v0.5+) |
| **C. Agent/API platform** | Connect their own AI agents to run experiments | Server, auth, quotas | Medium; builds on `simtara-api` |

v0.3.0 delivers A. B comes after we know which templates customers actually want. C starts in v0.4.0.

## v0.3.0 scope

### 1. Module parameter descriptions
Each module describes its parameters (name, type, range, default, group, help text). The launcher and control panel are generated from this, so a new template needs no hand-written UI. The same description feeds the API's `capabilities` and validation.

### 2. Template system
- Templates are JSON files: `{id, title, description, module, config, tags}`.
- Launcher shows template cards; ships with several Particle Life presets, plus Boids and Life once those modules land.
- "Custom" remains for free parameter choice.

### 3. Project storage
Folder-based, human-readable, git-friendly:

```
SimTara/projects/<project>/
  project.json            name, description, created, schema_version
  universes/<universe>/
    universe.json         module, config, seed, template id
    snapshots/            start.snap.json, autosave, named checkpoints
    experiments/<exp>/
      experiment.json     changes, run plan, metrics, status
      runs/<run>/manifest.json, series.json, final.snap.json (optional)
```

- Launcher gets a "My projects" view: open, duplicate, delete, new universe, resume from last checkpoint.
- Autosave on a timer and on exit; named checkpoints on demand.

### 4. Experiments
- Create an experiment from a universe: choose changes (parameter patches), ticks, seeds (single or a list), metrics.
- Run executes from the universe's start snapshot; results written to the run folder.
- Parameter **sweeps**: a list or range for one or more parameters expands into runs.
- Runs execute in the background with progress and cancel; CPU runs may run in parallel across seeds.

### 5. Results view
- Metric charts over time per run; overlay several experiments/runs to compare.
- Summary table (final value, mean, min/max across seeds).
- Export: CSV and JSON of series; PNG of the final frame; copy of the manifest.

### 6. Reproducibility
- Run manifest recorded for every run (engine and module versions, config hash, seed, backend).
- "Re-run" reproduces the same series on CPU bit-for-bit; GPU reproducibility documented as approximate.
- Explicit schema version on snapshot, project, universe and experiment files, with a migration hook.

### 7. API parity
Everything in the Studio is available through `simtara-api` (`project.*`, `universe.*`, `experiment.*`, `run.*`, `results.*`), so agents and scripts use the same model as the UI.

## Enterprise-readiness items (v0.3.0 foundations)

| Item | In v0.3.0 | Later |
|---|---|---|
| Versioned file formats + migrations | Yes | |
| Run manifests / provenance | Yes | Signed manifests |
| Structured logging + error codes | Yes | Telemetry export |
| CI: build, test, clippy, fmt on every push | Yes | Benchmarks gate |
| Release builds Windows/macOS/Linux | Yes | Installers, code signing |
| Docs: user guide + API reference | Yes | |
| License review of dependencies | Yes | SBOM |
| Auth, roles, audit log | | v0.4.0 (with server) |
| HTTP/WebSocket server | | v0.4.0 |
| Resource quotas for agents/users | | v0.4.0 |
| Plugin API stability guarantee | | v0.5.0 (with builder) |

## Milestones

1. **M1 – Foundations:** parameter descriptions, template system, versioned formats, CI.
2. **M2 – Projects:** project/universe storage, launcher "My projects", autosave, checkpoints.
3. **M3 – Experiments:** experiment records, runner with manifests, sweeps, background runs.
4. **M4 – Results:** charts, comparison, export; API parity.
5. **M5 – Content and polish:** Boids and Life modules plus templates; docs; cross-platform release.

## Acceptance for v0.3.0

A new user can: pick a template, create a project, launch a universe, define two experiments that differ in one parameter, run both with several seeds, see the compared metric curves, close the app, reopen the project and find everything intact, and re-run an old experiment to get the same numbers.

## Licensing (decided)

- Core: Elastic License 2.0 - free to use and modify, hosting it as a service for others is not allowed.
- Enterprise features (auth/roles/audit, hosted compute, quotas, team workspaces): separate commercial license in `enterprise/`. See ENTERPRISE.md.
- Hosted compute is the paid product; the desktop app stays free.

## Open decisions

1. First customer segment (education, research labs, game/film studios, enterprise analysts)? It decides which templates come first.
2. Pricing model: per-seat license, per-run cloud compute, or both?
3. Desktop-only for now, or hosted (needs the v0.4.0 server) sooner?
4. Where do we draw the line on user-defined rules in the first paid version?
