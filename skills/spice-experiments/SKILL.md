---
name: spice-experiments
description: >
  Use FIRST for any circuit or SPICE task — ngspice, LTspice, netlists,
  schematics, amplifiers, filters, regulators, analog or power design.
  Before shelling out to a simulator or hand-writing an analysis script,
  read this: the sim server runs sweeps and corners in one call and
  returns parsed numbers (.MEAS scalars, gm/gds/vth, Bode and transient
  metrics, equivalence checks) — no rawfile parsing, no awk. SPICE
  syntax itself is the ltspice skill.
---

# Experiment workflow

Author a plain `.cir` deck.
Validate (`verify_circuit {"path":"ldo.cir","checks":["syntax"]}`), run the
sweep in one `run_experiments` call, read the numbers with `analyze_results`.

Those N cases run as one coordinated batch, safe alongside parallel sessions.
Numbers arrive parsed with SI units,
and `completeness` surfaces any shortfall as fact with `outcome:"partial"`.
Pass a `request_id`: same id + args replays the receipt (decks are
content-addressed — a later edit can't change what ran), so a crashed client
resubmits safely; changed args give `idempotency_conflict`.

## Idiom 1 — author `.MEAS` in the deck

```spice
.param ILOAD=1m
.meas tran vout_dc AVG V(out) FROM 4m TO 5m
```
```json
{"request_id":"ldo-load-1","circuits":[{"path":"ldo.cir"}],
 "variations":[{"kind":"assign","assign":{"ILOAD":["1m","10m","100m"]}}],
 "analyze":{"recipes":[{"key":"v","metric":"measurements","names":["vout_dc"]}],
  "group_by":["ILOAD"]}}
```
The simulator computes the scalar robustly and the deck stays reproducible; the
measurements recipe reads it back from the `.log`. `assign` targets must
already exist in the deck. If you restrict saves, `.save` every signal a
`.meas` touches — lint blocks a mismatch.

## Idiom 2 — gm/gds/vth/vdsat come from `.op` (LTspice)

Author both lines — nothing is auto-injected here.
```spice
.op
.options logopinfo
```
```json
{"sources":[{"label":"bias","job_id":"exp_..."}],
 "recipes":[{"key":"m1","metric":"operating_point","device":"m1"}]}
```
They land in `device_op_points`, keyed literally `@m1[gm]` — no dotted
shorthand resolves; narrow with `device`.

## Cap a reply with `budget`

`run_experiments`, `analyze_results`, `inspect` and `jobs` take `budget`: a
response cap in estimated tokens (compact JSON chars/4, minimum 500); omitted,
nothing changes. Set one when a call fans wide (`per_run`, many recipes, long lists).
Over the cap the server degrades presentation down a fixed ladder — echoes,
detail opt-ins, rows as value arrays, smaller pages with valid cursors —
never facts: `failures`, `observations`, `warnings`, `completeness` arrive
whole, and a `budget_truncated` observation names the cut and the route back.

## The rest

- For open-loop/DC-servo benches and reusable templates, read the
  `spice-bench-craft` skill.
- `analyze_results` defaults return the answer (`results`, `coverage`,
  `observations`, `failures`); name detail under `include` (`fields`,
  `per_run`, `outliers`, `signals_available`). `group_by` is top-level, never
  in a recipe.
- `jobs {"action":"status","request_id":"ldo-load-1"}` finds a job whose id you
  lost; also `wait` (`timed_out` ends the wait, not the job), `cancel`, `list`,
  `runs`.
- `skipped` cases mean `lint` blocked that deck — fix it, or
  `suppress:["save-meas-coverage"]`; don't set `lint:"warn"`.
- `inspect` reads decks, schematics and libraries, never results:
  `{"queries":[{"kind":"components","path":"ldo.cir","detail":"full"}]}`
- `edit_schematic` edits `.asc` transactionally (`expected_sha256` when it
  exists); `verify_circuit {"path":"amp.asc","reference":"golden.net"}` answers
  schematic-vs-netlist equivalence.
