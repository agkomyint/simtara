# Business model: what we could sell

Status: ideas for discussion, not decided. Decisions get recorded in [learning-log.md](learning-log.md).

## Words (one meaning each)

- **Template**: a ready-made starting point (for example "Particle Life").
- **Universe**: what a user creates and owns, built from a template.
- **Experiment**: a change to a universe, run and measured.
- **Package**: a universe or experiment exported as a file so others can run it.
- **Hosted compute**: running universes and experiments on our machines.

## What we could sell

| # | Product | Who pays | Notes |
|---|---|---|---|
| 1 | Runtime and Studio app | Free | Free download. Brings people in and builds trust. |
| 2 | Hosted compute | Labs, studios, schools | Pay to run big universes or many experiments on our machines. First real revenue. |
| 3 | Universe packages | Builders sell, others buy or license | An expert models a universe (for example a solar system with real physics) and packages it. Others run it without building it. |
| 4 | Experiment packages | Researchers, teachers | A saved experiment ("remove the Moon at tick 5000, compare these metrics") that anyone can load onto a universe package and re-run. Reproducible and shareable. |
| 5 | Enterprise features | Organizations | Accounts, roles, audit logs, team workspaces, support. See [../ENTERPRISE.md](../ENTERPRISE.md). |

## How they fit together

A package is a universe plus its template and rules. An experiment is a change plus a measurement, aimed at a package. Hosted compute runs the experiment. Packages and experiment exports are the sharing formats on top of things we already planned.

Long term this can become a marketplace: people who build universes earn money, people who want answers pay, and we sell the compute and take a cut. The value then comes from what the community makes, not only from our code.

## Example project

"What happens if the Moon disappears?" Model the Sun, Earth and Moon with real physics, remove the Moon in an experiment, compare Earth's orbit and tilt against the unchanged universe. Needs an orbital-gravity module (not built yet). Tides and climate need far more physics, so the first honest answer is about orbit and tilt.

## Cautions

- **Marketplaces need both sides.** With no universes to buy and no buyers it stays empty. Build the experiment loop first, then hosted compute, then packaging, then a marketplace.
- **Packages need a safe format.** Today rules are compiled Rust, so a package can only carry settings and a start state for modules we ship. Packages with their own new rules need a scripting or rules-as-data layer plus sandboxing. This is the biggest technical piece in the plan.
- **Trust.** Re-run results are only believable with a run record (engine version, seed, backend). This is in the 0.3.0 plan.
- **License.** Elastic License 2.0 stops others from offering SimTara as a hosted service, which protects hosted compute and any marketplace. A lawyer should review before selling.

## Suggested order

1. 0.3.0: save universes and run experiments.
2. A `.simtara` package file (universe plus experiment, shipped modules only) with export and import.
3. Hosted compute that runs a package.
4. Lua scripting for events, a Python SDK, user-written rules, sandboxing, then a marketplace. See the scripting decision in [learning-log.md](learning-log.md).

## Open questions

- Who is the first customer: education, research labs, or studios?
- How is hosted compute priced (per hour, per run, subscription)?
- Do package sellers get a revenue share, and how large?
