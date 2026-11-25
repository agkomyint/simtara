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

## 4. Story games on a living universe

**Idea.** Use a universe's current state as the world of a story game. Each player action is a command; the simulation reacts, so outcomes differ without a writer scripting every branch. Same seed + same choices = same story, which makes runs replayable and shareable.

**How it maps onto what exists**
- World state: things, relations, rules.
- Player actions: commands (`set_param`, `spawn`, `set`), limited by roles ("you are the fire chief").
- Story beats: `on_tick` reads metrics and logs messages ("the fire crossed the river").
- Endings: checks such as `cells burning > 100` become win and lose conditions.
- Determinism: a seed and a record of choices reproduce a story exactly.

**Missing for a real game**
1. A player view instead of the developer Studio: narration text, a few choices or sliders, a map or 3D picture.
2. A story and event layer: named events, conditions, text templates, a queue so only a few beats show per turn. Today there is only `log`.
3. Turn structure: "run until something happens or N ticks, then report".
4. Save points and undo: checkpoints plus the input log (same foundation as branching, see `technical-direction.md`), so a player can rewind to a choice.
5. What the player can see: per-role visibility with delay and noise, so a character knows only what they would know.
6. Optional AI narrator: a language model turns structured events and state into prose. The simulation stays the source of truth; the narrator describes, never decides, and is given only structured facts to limit invented details.

**Can it be done now?** A rough prototype is possible today with Luau, a role, `on_tick` beats and checks. A real game needs items 1 to 5 above; the player view and the event layer are the largest costs.

**Design tension.** Pure emergence can feel flat because simulations lack dramatic timing. A *director* rule layer (raise the fire when the player is winning too easily, hold back disasters after one) adds pacing without scripting outcomes, and fits the rule system.

**Related branching features** (also in `technical-direction.md`)
- *Intervene*: a recorded one-off state edit.
- *Branch*: fork at a checkpoint, run independently, compare, adopt or discard. In a game this is "what would have happened if I had chosen the other way?"
- *Twin*: a branch that keeps following the main universe with a persistent difference; later.

**Open question.** Text-and-choices or a visual (3D) world? That decides the amount of player-view work more than anything else.

## 5. Science use

**Perturbation modeling: a natural fit.** A perturbation study is a baseline run, one change, and a comparison. That is already the experiment feature: gene knockout, drug dose, temperature change, removing a species, crew size in the fire example. Determinism, same-seed branches, compare charts and sweeps are what make such a study trustworthy.

Needed to make it a real study tool:
- Sensitivity analysis: sweep many controls and rank which matter most.
- Many seeds, with uncertainty ranges instead of one line.
- Fitting controls to measured data (CSV in, parameters out).
- Reproducible `.simtara` packages, so a result can be rerun and checked.

**Protein folding: a stretch.** Real folding simulation is molecular dynamics: force fields, tiny timesteps, neighbour lists, long-range forces, GPU kernels, validated over decades (GROMACS, OpenMM). SimTara's particles module and GPU backend are nowhere near that, and validation is a research effort of its own. AlphaFold-style prediction is machine learning, not simulation. Realistic near-term uses are toy and teaching models: coarse-grained chains, HP lattice folding, "what if this bond is stronger?".

**What science needs in the platform**
1. Numerical methods: ODE solvers, stochastic simulation (Gillespie), proper integrators, not only per-tick rules.
2. Units and dimensions, so a model cannot mix metres and seconds.
3. Performance: bulk spawn, spatial queries, columnar storage, GPU rules.
4. Data in and out: CSV and standard formats.
5. Validation: each science module must reproduce a known result and ship with that test.

**Honest framing.** SimTara is a platform, not a model. Scientific value comes from each module's correctness. The claim to make is "reproducible, comparable what-if experiments for models people build", not "discovers things". A fast, clear loop for building and testing models is a real contribution; the discovery comes from the model and its data.

## 6. AI and learning

SimTara should not compete with ML frameworks. Training a real model needs tensor math, automatic differentiation and GPU matrix kernels (PyTorch, JAX); SimTara's rules are scalar formulas over things.

**Realistic uses**
- **Gradient descent as a visible simulation.** A loss surface is a landscape, each thing a point with `w = w - lr * grad(w)`. Sliders for learning rate, momentum and noise; compare runs. A good teaching tool.
- **Toy networks as things and relations.** A few neurons with hand-written gradients: slow, but it shows learning step by step.
- **Evolution and search.** Populations that mutate and get selected: optimisation without gradients, a natural fit for things and rules.
- **A tokenizer** (BPE) is easy to write but is not a simulation; it would be a separate tool.

**Where SimTara fits better**
1. **Environments for AI agents.** A universe is a world an agent acts in through commands, with roles, scores and deterministic replay. SimTara is the arena; PyTorch or an API is the brain.
2. **Evaluating agents.** Same universe, many agents and seeds, compare (branches and sweeps).
3. **A language model as narrator or player** inside a story game; it describes or plays but never decides outcomes.
4. **Optimising universe parameters** by search or finite-difference gradients across sweeps ("find the crew size that minimises burned area").

## 7. More uses

**Practical what-if tools**
- **Policy and planning:** school closures in an epidemic, a new bus route, a bigger crew budget. Roles and limits already model who can do what and what runs out.
- **Crowds, evacuation and traffic:** where to put an exit before building it.
- **Supply chains and logistics:** "what if this port closes?"
- **Economies and markets:** many agents with simple rules, for teaching and policy tests.
- **Digital twin of a real system** (factory, server cluster, hospital) fed with real data for what-if runs.

**Software and systems**
- **Chaos testing for distributed systems:** servers, networks and failures simulated deterministically; a bug found with seed 4812 replays exactly.
- **Game worlds:** a living world (weather, ecology, towns) that keeps running between sessions, so player changes last.

**People and culture**
- **Teaching kits:** a universe, controls to turn, and checks that grade predictions ("will the fire reach the town?", then run it).
- **Competitions:** a fixed universe and goal; people or agents submit strategies; a leaderboard compares them. Determinism makes it fair.
- **Interactive explainers:** an article embeds a universe in the browser; readers turn the controls (needs the web viewer).
- **Generative art and music:** systems that grow and drift, rendered as images or video, with sound mapped from metrics.
- **Wargaming and scenario planning:** roles for opposing sides, hidden information, a facilitator changing conditions mid-game.

**Ecosystem**
- **A marketplace** for paid universes and modules, with `.simtara` packages as the unit.
- **More languages:** Python and others for modules and controllers, not only Luau.

**Best near-term fit**
1. Competitions and teaching: little new engineering (checks, packages, a leaderboard) and they build a community.
2. Chaos testing for software: a market with money, matching the determinism the core already has.
3. Policy and planning demos: good stories for people who do not write code.

## 8. The long list

Everything below fits the same shape: things with properties, relations, rules, controls, roles, metrics, and experiments to compare changes. "Have" means an example or module exists today; the rest would be a script or a module, sometimes with the engine work listed in `technical-direction.md`.

**Physics and maths**
- Heat on a plate (have), waves and ripples, diffusion, percolation, sandpiles, Ising model and phase changes.
- Cellular automata: Game of Life, rule 110, forest-fire models, traffic automata.
- Orbits and gravity (have), n-body clusters, tides, simple planetary systems.
- Particles, charges and boundaries (have), gas in a box, Brownian motion.
- Toy fluids and smoke on a grid; pendulums and chaos; springs and cloth with a coarse model.
- Random walks, Monte Carlo estimation, queueing theory, network flow.

**Chemistry and biology**
- Reaction-diffusion patterns (have), reaction networks, simple enzyme kinetics.
- Forest fire with fuel, heat, oxygen, crews and water (have).
- Population dynamics: predator and prey, competition, carrying capacity, extinction.
- Evolution: mutation, selection, drift, speciation on a landscape.
- Epidemics with towns, vaccines, masks, travel (have: an epidemic and towns and a disease).
- Cells and tissue: growth, wound healing, tumour growth, morphogenesis, gene regulatory networks.
- Ecosystems and food webs; invasive species; fisheries and quotas.
- Neurons and simple circuits (toy models for teaching).
- Pollination, pests, crop growth, irrigation, soil and weather.

**Earth and environment**
- Toy climate and carbon cycle; heat islands; sea-level and flooding on a height map.
- Wildfire, flood, drought and landslide scenarios on real terrain (needs maps).
- Water systems: reservoirs, pipes, leaks, rationing.
- Energy grids: supply and demand, storage, outages, price rules.
- Waste, recycling and pollution spread.

**Society and economy**
- An idea spreads (have); rumours and misinformation; fact-checking as a role.
- Opinions and voting, polarisation, coalition formation.
- Markets, auctions, bubbles and crashes; banks and contagion; inflation toy models.
- Cities: land use, housing, commuting, zoning, gentrification.
- Traffic, public transport, bike lanes, parking, congestion pricing.
- Crowds, evacuation, stadium exits, queues at airports and hospitals.
- Supply chains, warehouses, delivery routes, breakdowns, shortages.
- Labour and organisations: teams, handoffs, bottlenecks, burnout.
- Tragedy of the commons, public goods, cooperation, trust, fairness games.
- Language and culture: words that spread, dialects, naming trends, fashion cycles.
- History what-ifs and archaeology scenarios (settlements, trade routes, collapse).

**Health and public services**
- Hospital beds, triage and staffing; ambulance placement; waiting lists.
- Vaccination and screening policy; contact tracing; outbreak drills.
- Disaster response: roles for command, crews and the public, with limited water, money and people.

**Software, security and systems**
- Chaos testing of distributed systems with deterministic failures.
- Load tests and capacity planning: users, servers, caches, queues.
- Worm and malware spread, attack and defence games, patch policy.
- A/B test simulators and statistical power checks.
- Synthetic data generators with known ground truth, for testing analytics and models.
- Robot swarms, warehouse robots, drone coverage, multi-agent path finding.

**Games and creativity**
- Story games on a living world (section 4), roguelike worlds, city builders, tycoon games.
- Procedural worlds: terrain, rivers, towns, histories, then play them.
- Game balance testing: run thousands of seeded matches and compare strategies.
- Generative art, evolving patterns, music from metrics, videos (section 1).
- Interactive fiction where weather, markets and rumours keep running.

**Education and research**
- Classroom kits for physics, biology, economics, civics, and computing, with predict-then-run checks.
- Reproducible research packages: a paper's model, data and seeds in one `.simtara` file.
- Sensitivity analysis, ensembles, calibration to data, uncertainty ranges.
- Peer comparison: two labs run the same package and compare results.
- Open competitions and challenges with leaderboards.

**Agents and AI**
- Arenas where AI agents act through commands and are scored; human versus agent runs.
- Language models as narrators, game masters or role-players, never deciding outcomes.
- Multi-agent negotiation, markets and cooperation tests.
- Safety drills: how an agent behaves when a control is locked, a budget runs out, or information is delayed.

**Business and platform**
- Digital twins of factories, networks and buildings fed by real data.
- Planning tools for governments, utilities and companies; scenario libraries.
- A marketplace for universes and modules; enterprise access control; hosted compute for sweeps.
- Embeds: a universe inside a web page, a notebook, a slide or a dashboard.

**What decides what is next.** Pick uses that reuse the same engine work: the input log, branching and compare, the headless renderer, bulk spawn and spatial queries, and per-role visibility. Each of those unlocks many items above at once.
