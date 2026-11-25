# The start

Why SimTara exists, and the first big goal it is pointed at.

> **The plan: model a cell that has died, and search for a way to bring it back.**

Everything else in this repository (the engine, rules, controls, roles, experiments, the 3D view, the future renderer, packages and hosted compute) is there so this question can be asked carefully, repeatedly, and by many people. A story game, a fire, an epidemic and a bouncing ball are practice. This is the goal.

---

## 1. Read this first: honesty about the goal

- Nobody can currently revive an arbitrary dead cell. If it were solved, it would not be a software project.
- A simulation does **not** discover biology by itself. A model of a cell is only as true as the biology and data put into it. A model that "revives" a simulated cell proves nothing about a real one.
- What a simulator *can* do is cheaper and still valuable:
  1. force us to write down exactly what "dead" and "alive" mean;
  2. turn vague ideas ("restore energy, then repair the membrane") into testable, comparable experiments;
  3. search a huge space of interventions and ordering of steps that no one would try by hand;
  4. rank what to test in a real lab, so a wet experiment is chosen on evidence instead of guesswork;
  5. keep every result reproducible, so others can check it.
- So the success claim is modest and real: **"here is a model, here is what it says is reversible and what is not, here are the interventions it ranks, here is how to test them."** Not "we revived a cell".
- Real-world work belongs with qualified biologists, in proper laboratories, under the rules of their institutions. This project produces models and hypotheses.

## 2. What does "dead" mean? (the first job)

The whole plan depends on this definition, and it is not one thing. A cell can fail in different ways, and some are reversible:

| state | what has happened | reversible? |
|---|---|---|
| **Stressed** | low energy, ion imbalance, still intact | usually yes |
| **Dormant / frozen / dried** | metabolism stopped on purpose (cold, dryness) | yes, in many cells; that is the point of cryopreservation and dormancy |
| **Energy-failed** | ATP gone, pumps stopped, ions leak, swelling | often yes if caught early (this is ischemia and reperfusion) |
| **Damaged** | membrane, mitochondria, DNA or proteins damaged but present | sometimes; depends on what and how much |
| **Committed to death** | apoptosis or necrosis programme has run past its checkpoints | probably no |
| **Dissolved** | membrane burst, contents lost, structure gone | no; the information is gone |

The key concept is the **point of no return**: the line after which no intervention works. Finding where that line is in the model, and which interventions move it, is the main scientific output of this plan.

**A working definition for the model (to be argued and revised):**
A cell is *alive* if it has a self-maintaining loop: it makes energy, uses energy to hold its ion gradients and repair itself, and can copy or at least keep its own state. A cell is *dead* when that loop has stopped and cannot restart without outside help. A cell is *revived* when, after intervention, the loop runs again without further help and stays running.

## 3. Why a simulator is the right first tool

- A cell is many things (molecules, organelles, compartments) with relations and rules about how they change. That is the shape SimTara already has: things, relations, rules, and measurements.
- The question is a comparison question: *same cell, different intervention, what changes?* That is an experiment.
- The answer has many interacting parameters and an order of steps, which is what sweeps, branches and search handle.
- Failures must be reproducible: the same seed gives the same death, so we can compare rescue attempts from the exact same moment.
- Different people hold different parts (membrane experts, metabolism experts, gene expression experts). Modules and roles let them each own a layer and plug it in.

## 4. The model, layer by layer

Start small and coarse. Add detail only when the layer below behaves sensibly. Each layer is a **module** (a set of things, rules and controls) that can be switched on or off, so we can see what each one contributes.

**Layer 0. The bag of chemistry (well-mixed).**
Quantities, not things: ATP, ADP, glucose, oxygen, sodium, potassium, calcium, protons, water volume. A handful of reactions with rates. Death and rescue appear as trajectories. This is closer to ordinary differential equations than agents, so it needs the numerical methods listed in `technical-direction.md` (an ODE solver, then stochastic simulation).

**Layer 1. Energy and ions.**
Mitochondria make ATP from fuel and oxygen. Pumps (the sodium-potassium pump is the classic one) use ATP to hold ion gradients. If ATP falls, pumps slow, ions leak, water enters, the cell swells, calcium rises. This is the classic cascade of energy failure and is the best-understood place to start.

**Layer 2. Membrane and compartments.**
A membrane with a leak and a tolerance for swelling. Compartments for cytoplasm, mitochondria, and later others. A membrane that bursts is the "dissolved" state: the model must make that irreversible.

**Layer 3. Damage and repair.**
Damage accumulates in the membrane, mitochondria, proteins and DNA (each as a number or a population of damaged items). Repair machinery needs energy and raw materials. Rule: repair can only happen while energy is above a minimum. This creates the point of no return honestly: too little energy, no repair, more damage, less energy.

**Layer 4. Signals and programmed death.**
Calcium and damage signals can trigger apoptosis or necrosis programmes with checkpoints. Interventions can block them (inhibitors) at certain times only.

**Layer 5. Gene expression and state.**
A regulatory network that decides which proteins are made. Restarting a cell needs the right genes switched on in the right order. First as a small switch-network, later from real data.

**Layer 6. Space.**
Where things are inside the cell: diffusion, crowding, transport along structures. Uses the 3D view and spatial queries (see the scaling work).

Always keep the simplest model that answers the question. A bigger model is not a truer one unless it is validated.

## 5. Interventions: what we can try

Interventions are controls in the model, owned by a role ("lab" or "treatment"), with limits and costs. Examples to model:

- **Supply:** restore oxygen, glucose or specific metabolites; supply ATP precursors.
- **Temperature:** cool to slow damage and buy time; rewarm at different rates.
- **Ion and water control:** correct sodium, potassium, calcium; change osmotic conditions.
- **Block cascades:** inhibit specific death signals or calcium entry.
- **Support repair:** add repair factors, antioxidants, chaperones.
- **Reperfusion strategy:** restoring oxygen too fast can cause its own injury (reperfusion injury); the model must be able to show that.
- **Ordering and timing:** the same steps in a different order, or a minute later, may fail. This is the heart of the search.
- **Replacement:** swap in healthy mitochondria or other parts (as a modelled intervention, with its cost).

Every intervention has a **cost** and a **limit** (limited supply, limited time), as in the fire example where water runs out. A rescue that works only with infinite resources is not a rescue.

## 6. The experiments we would run

1. **Time-to-death curve.** Stop the energy supply in a healthy cell and watch which things fail in which order. Does the model reproduce the known sequence? (Validation.)
2. **Point-of-no-return map.** Start rescue at tick T for many values of T. For each, record whether the cell recovers. Plot recovery against delay and against the severity of damage.
3. **Single-intervention sweeps.** For each intervention alone, find the window in which it helps, and the window in which it hurts.
4. **Combination and ordering search.** Search over sequences of interventions (for example restore ions, then energy, then lift the block) to find the sequences that move the point of no return furthest, within the resource limits. Use branches: fork from the same dying cell, apply different plans, compare.
5. **Sensitivity analysis.** Which parameters, if wrong, change the answer? Those are what the real lab should measure first.
6. **Robustness.** Run many seeds and many parameter variations. A plan that works for one exact setting is a coincidence; a plan that works across the plausible range is a lead.
7. **Counterfactuals.** "What if the mitochondria had been protected?" Branch from the past and compare.

## 7. How SimTara supports this (what exists, what is needed)

| need | exists today | needed |
|---|---|---|
| things, relations, rules | yes | bulk creation and spatial queries for many molecules |
| controls, owners, costs and limits | yes (params, roles, shared totals) | observability per role |
| deterministic runs and seeds | yes | stable per-thing random streams |
| experiments (change a control, compare) | yes | run from "now" (fork), many at once |
| headless runs and checks | yes (`simtara script`) | sweep runner with parallelism |
| 3D view | yes | cell-scale models and labels |
| measurements and charts | yes | marker for "point of no return" |
| ODE and stochastic simulation | no | a numerical-methods module (see technical direction) |
| units | no | units and dimensions, so rates cannot be mixed up |
| search over plans | no | an optimiser that calls sweeps and branches |
| data import and fitting | no | CSV in, fit parameters to measurements |
| packages and reproducibility | partial (saved projects) | `.simtara` package with versions |
| live editing without crashing | planned | last-good script and rollback (see technical direction) |

## 8. Staged plan

**Stage 0: Write the question down.**
Define "dead", "alive" and "revived" for one chosen cell type and one chosen cause of death (for example energy failure by oxygen and glucose loss). Without this, nothing else is measurable. Deliverable: a one-page definition and a list of what we do and do not claim.

**Stage 1: A toy cell that can die and be saved.**
Layer 0 and 1 only: ATP, ions, water, a pump, a membrane that bursts. Numbers are rough and labelled "toy". It must show a clear point of no return and at least one rescue that works only if started in time. Deliverable: a script, checks ("rescue before tick 200 recovers; after 400 it does not"), and the first chart.

**Stage 2: Make it honest.**
Replace invented numbers with published ones (with citations in the module), add units, and compare the sequence of failure against what is known. A model that fails this check is fixed or thrown away. Deliverable: validation notes inside the module.

**Stage 3: Damage, repair and signals.**
Layers 2 to 4. Add reperfusion injury, repair that needs energy, and death programmes with checkpoints. Deliverable: the point-of-no-return map and the single-intervention windows.

**Stage 4: Search for plans.**
Branching, sweeps and ordering search. Deliverable: a ranked list of plans, each with its resource cost and the range of conditions where it works, plus the sensitivity ranking.

**Stage 5: Data.**
Bring in real measurements from collaborators or public data. Fit parameters, report uncertainty, and say which results survive.

**Stage 6: Share and test with biologists.**
Release the package. Biologists try to break it. Choose a small number of the ranked plans for a laboratory to test. Feed the results back into the model, whatever they show.

**Stage 7: Bigger cells, more types, tissues.**
Only after the first results stand up: gene expression, space, other cell types, groups of cells.

## 9. What would count as success, and as failure

**Success (any of these is a real result):**
- A model whose failure sequence matches known biology.
- A clear, defended point-of-no-return map.
- A ranked list of interventions and timings, with the conditions where each works.
- A sensitivity ranking that tells experimenters which numbers to measure.
- A reproducible package others can run, criticise and improve.

**A useful failure:** the model shows that, under realistic limits, no tested plan moves the point of no return, or that revival is only possible in a narrow window. That is still knowledge: it says where not to spend lab time.

**A bad outcome:** a pretty model nobody can check. The guard against it is validation, citations, honest labels ("toy", "fitted", "measured") and open packages.

## 10. Risks

- **False confidence.** A smooth simulation looks like truth. Mitigate with labels, uncertainty ranges and independent checks.
- **Wrong abstraction.** The real thing may depend on details the model leaves out. Mitigate by adding layers one at a time and testing each against data.
- **Overfitting.** A model tuned to one dataset can "revive" anything. Keep test data separate; report failures.
- **Scope creep.** Whole-cell models are research programmes by themselves. Stay with one cell type and one cause of death until it works.
- **Performance.** Many molecules and small timesteps are heavy. The scaling work in `technical-direction.md` (bulk creation, spatial index, columnar storage, GPU rules) comes before large models.
- **Responsibility.** Anything that touches real living material belongs to qualified people under their institution's safety and ethics rules. This project's outputs are models, hypotheses and ranked suggestions, clearly marked as such.

## 11. First concrete steps (small, soon)

1. Pick the cell type and the cause of death for Stage 0, and write the definition page.
2. Write the toy cell script (Layers 0 and 1) as `examples/cell.luau` with checks for "dies without rescue", "recovers with early rescue", "does not recover with late rescue".
3. Add the numerical module for ODE and stochastic simulation so rates are not forced into per-tick formulas.
4. Add the sweep runner and the point-of-no-return chart marker.
5. Find a collaborator in cell biology to review the definition and the numbers before anything is claimed.

## 12. Why this matters beyond one cell

The cell is the hardest and most meaningful test of the platform's promise: *a clear, reproducible, shareable way to ask "what if we change this?"* If the same engine can run a fire, a story, an epidemic and a dying cell with the same tools (controls, roles, experiments, branches, packages), then SimTara is a real general-purpose runtime. The other uses in `possibilities.md` come along for free.
