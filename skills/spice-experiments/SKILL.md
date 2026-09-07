---
name: spice-experiments
description: >
  Use for any circuit or SPICE task: ngspice, LTspice, netlists,
  schematics, amplifiers, filters, regulators, analog or power design.
  Before running a simulator or writing an analysis script yourself, read
  this: the sim server runs sweeps and corners in one call and returns
  parsed numbers (.MEAS scalars, gm/gds/vth, Bode and transient metrics,
  equivalence checks), no rawfile parsing. SPICE syntax: the ltspice skill.
---

# Experiment workflow

Author a plain `.cir` deck.
Validate (`verify_circuit {"path":"ldo.cir","checks":["syntax"]}`), run the
sweep in one `run_experiments` call, read the numbers with `analyze_results`.

The N cases run as one batch; other sessions can run at the same time.
Values come back parsed, with SI units; if a case produced no result,
`completeness` reports it and `outcome` is `"partial"`.
Charts: `plot_waveform` (interactive, where the client supports it); the
`plot` recipe is the static fallback.
Loops over many runs, numpy on the samples, or a script you will keep:
`from ltspice_mcp.api import Api` runs the same six ops in-process and returns
complete results (no paging, no budget); `api.reference("<op>")` gives an
op's arguments before you guess them. If the server lists a `run_code` tool,
the same snippet runs there with `api` already in scope and the engine warm.
Pass a `request_id`: the same id and args returns the original receipt
instead of re-running (decks are content-addressed, so a later edit does not
change what ran); different args return `idempotency_conflict`.

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
The simulator computes the scalar and the `measurements` recipe reads it back
from the `.log`. `assign` targets must exist in the deck. If you restrict
saves, `.save` every signal a `.meas` uses; lint blocks a mismatch.

## Idiom 2 — gm/gds/vth/vdsat come from `.op` (LTspice)

Put `.op` in the deck; the server adds `.options logopinfo` to LTspice `.op`
runs (writing it yourself is harmless).
```spice
.op
.options logopinfo
```
```json
{"sources":[{"label":"bias","job_id":"exp_..."}],
 "recipes":[{"key":"m1","metric":"operating_point","device":"m1"}]}
```
Results are in `device_op_points`, keyed by the literal `@m1[gm]` (no `m1.gm`
shorthand here); `device` limits the output to one device.

## Cap a reply with `budget`

`run_experiments`, `analyze_results`, `inspect` and `jobs` take `budget`, a
response cap in estimated tokens (compact JSON chars/4, minimum 500);
omitted, the response is unchanged. Set one when a call can return a lot
(`per_run`, long lists). Over the cap the server drops presentation in a fixed
order (echoes, detail opt-ins, smaller pages with valid cursors) and never
facts: `failures`, `observations`, `warnings` and `completeness` arrive
complete, and a `budget_truncated` observation says what was cut and how to get
it back.

## Other notes

- Open-loop/DC-servo benches and templates: the `spice-bench-craft` skill.
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
  exists); `verify_circuit {"path":"amp.asc","reference":"golden.net"}` checks
  that a schematic matches a netlist.
