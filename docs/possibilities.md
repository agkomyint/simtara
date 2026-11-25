# New possibilities

Ideas that fit SimTara's design but are not built yet. Each says what already exists, what is missing, and a rough first step. Nothing here is promised.

## 1. Lua to movie (render a universe to video)

**Idea.** Write a script, press render, get an `.mp4`: a directed film of a simulation, with scripted camera moves and objects.

**Already there**
- Camera and light belong to the universe: `camera{yaw, pitch, distance, fov, follow}`, `light{...}`.
- A script can call `camera{}` from `on_tick(t)` to move it each tick (an orbit is `yaw = t * 0.5`; `follow = id` tracks a thing). Not yet tested for this use.
- Objects are driven by rules, params and `set{}`, so scripts control them.
- Runs are deterministic for a seed, so every frame is reproducible.
- Projection, lighting, shapes and `.obj` parsing exist in `simtara-studio/src/view3d.rs`.

**Missing**
- A headless renderer. Drawing currently happens through egui inside the Studio window.
- A way to write frames out.

**First steps**
1. Move the 3D drawing into its own crate with a small CPU rasteriser that outputs an RGBA image (no window, no GPU).
2. `simtara render script.luau --ticks 600 --fps 30 --size 1280x720 --out movie.mp4`: run the sim, set the camera, render, save `frame_0001.png`..., then call `ffmpeg` if installed (otherwise stop at PNGs).
3. `shot{from_tick, to_tick, yaw, pitch, distance, ...}` keyframes, so camera moves are declared instead of hand-coded in `on_tick`.
4. Optional later: title text, fades, a chart overlay of a metric, audio.

**Limits.** Flat shading, no shadows, textures or anti-aliasing at first. Offline rendering only: a few frames per second at 720p with thousands of things. Good for science and explainer videos, not film.

**Why it matters.** Experiments become shareable: the original run and the changed run, side by side, as a video. It is also a clear demo of the "universe as code" idea.

## 2. Related ideas

- **Screenshot / GIF from the CLI.** The first slice of the above: `simtara render --ticks 1 --out frame.png`.
- **Side-by-side experiment videos.** Render both arms of an experiment in one frame with a live chart.
- **Scripted interaction.** Replay a recorded set of control changes ("at tick 200 the crew turns water to 5") so a movie shows a human playing a role.
- **Agent-made films.** Because the camera, objects and controls are all script commands, a coding agent can write, render, look at the result and adjust, like it already does with headless `simtara script` runs.
- **GPU renderer.** For large universes and real-time playback, using the existing wgpu setup in `simtara-gpu`; the CPU renderer stays as the reproducible reference.

## 3. More possibilities

**Using the universe**
- **Role-play games and training.** "Play as" already locks controls by role. Add scoring, a budget and a goal ("keep burning cells under 100") and a universe becomes a game or a classroom exercise: be the fire chief, the city mayor, the virus.
- **Observability per role.** Each role sees only part of the universe, delayed or noisy, like real life. Needed for honest decision training.
- **Multiplayer.** Several people in different roles in one hosted universe, each turning only their own controls.
- **Agents as players.** An AI agent plays a role through the same commands a human uses, and the run is scored. Compare agents or compare humans with agents.

**Finding answers**
- **Parameter sweeps in Studio.** Pick controls and ranges, run hundreds of headless runs in parallel, get a heat map ("which oxygen and rain stop the fire?"). Already done by hand with scripts; make it a button, and a hosted-compute feature later.
- **Goal search.** "Find the cheapest crew size that stops the fire": an optimiser calls the sweep.
- **Calibration to real data.** Load a CSV (real cases, real temperatures) and fit controls so the universe matches it.
- **Uncertainty.** Run the same universe with many seeds and show ranges, not one line.

**Building universes**
- **Plug-in modules ("batteries").** `use("people")`, `use("water")`, `use("maps")`: ready-made things and rules you combine, such as disease plus towns plus travel.
- **Maps and places.** Real geography as the space things live in (roads, rivers, terrain heights drive the 3D ground).
- **Bulk spawn and spatial queries.** `grid{}`, `spawn_many`, "neighbours within distance", so big worlds are quick to write and run.
- **Agents describe a universe in plain words**, the script is generated, tested headlessly, and fixed until checks pass.

**Sharing**
- **`.simtara` packages.** A universe plus its models, experiments and results as one file; open it, rerun it, verify it reproduces.
- **Reproducible papers and lessons.** A figure links to the package that regenerates it.
- **Web viewer.** Open a package in the browser, orbit the camera, turn the controls, no install.
- **Python and HTTP API.** Drive universes from notebooks or other apps; things and rules are not yet in the API.

**Look and feel**
- **Tools in the 3D view:** select and move a thing with the mouse, draw a region and set a value inside it (rain on this area), trails and arrows for velocity, labels.
- **Sound.** Map a metric to a tone, or add sound to rendered videos.
- **VR / AR viewing** of a universe, later.

**Business**
- Hosted compute for big sweeps, an enterprise tier with access control, and a library of paid modules and universes.
