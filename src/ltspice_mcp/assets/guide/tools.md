---
name: tools
description: >
  Calling the tools directly: which call does what, a sweep in one call,
  notes per tool, progress on a running job, the reference lookup, the
  recipes the schema lists by name only, and the response budget.
---

# Using the tools

The loop: write the deck to a file, run it with `run_experiments` (attach
`analyze.recipes` and the numbers come back in the same response), follow a
receipt with `jobs`, and measure a finished job with `analyze_results`.

| To … | Call |
|-|-|
| run a deck — once, swept, or perturbed | `run_experiments(circuits=[{"path": …}], variations=[…])` |
| get the measurements without a second round trip | `run_experiments(analyze={"recipes": […]})` |
| follow, cancel, or page a job's runs | `jobs(action="status"\|"wait"\|"cancel"\|"list"\|"runs")` |
| measure a finished job | `analyze_results(sources=[{"job_id": …}], recipes=[…])` |
| read `.meas` results | recipe `{"metric": "measurements"}` |
| a scalar, a trace, a chart | recipes `value`, `waveform`, `plot` |
| device operating points (gm/gds/vth) | recipe `{"metric": "operating_point", "device": "M1"}` |
| AC corner, gain, slope, crossing, stability | recipes `bode_filter`, `bode_point`, `bode_slope`, `bode_crossing`, `stability`, `ac_structure` |
| transient stats, edges, timing, THD | recipes `signal_stats`, `edges`, `timing`, `periodic`, `transient_response`, `thd` |
| symbol geometry, a net, a component list, a model | `inspect(kind="symbol"\|"net"\|"components"\|"model")` |
| nested devices, scoped ports, effective parameters | `inspect(kind="hierarchy", path=..., simulator=...)` |
| find the recipe, op or check for a job, and its fields | `inspect(kind="reference", query="phase margin")` |
| create or mutate an `.asc` | `edit_schematic(target=…, ops=[…])` |
| check a sheet against its netlist, or render it | `verify_circuit(path=…)` |

A case has no time limit unless `execution.run_timeout_s` (or the server's
`[simulation] run_timeout`) sets one. While a job runs, each `jobs(status)` or
`jobs(wait)` receipt carries a `run_progress` observation per running case:
points written and the last time or frequency reached, to compare with the
deck's own stop value. A simulator writes in buffered blocks, so compare
readings a few minutes apart: if `points` has not moved, the case is not
advancing, and `jobs(action="cancel")` stops it.

`inspect(kind="reference")` finds a recipe, schematic op, variation kind,
check or job action from plain words (`query="phase margin"`) and returns its
fields, types, defaults and units; with no `query` it lists them all. Use it
instead of guessing a name, and always on the compact tool listing, where the
argument descriptions are not on the wire.

## A sweep in one call

Put the scalar in the deck as a `.meas`, sweep with an `assign` variation, and
attach the recipe that reads it back:

```spice
.param ILOAD=1m
.meas tran vout_dc AVG V(out) FROM 4m TO 5m
```

```json
{"request_id": "ldo-load-1", "circuits": [{"path": "ldo.cir"}],
 "variations": [{"kind": "assign", "assign": {"ILOAD": ["1m", "10m", "100m"]}}],
 "analyze": {"recipes": [{"key": "v", "metric": "measurements", "names": ["vout_dc"]}],
             "group_by": ["ILOAD"]}}
```

The simulator computes the scalar and the `measurements` recipe reads it back
from the log, parsed and with SI units. On ngspice, measure the trace with a
recipe instead (guide section 'ngspice'). An `assign` target must exist in the
deck. If the deck restricts what it saves, `.save` every signal a `.meas` uses;
lint blocks a mismatch. A case that produced nothing is counted in
`completeness`, and `outcome` is `"partial"`.

## Notes per tool

- `run_experiments`: a reused `request_id` returns `idempotency_conflict` when
  the arguments differ, a deck has changed since the job ran, or the simulator
  is now a different build; submit under a new id to run again. A `skipped`
  case means lint blocked its deck: fix the deck, or drop that one rule with
  `suppress` (the `rule_id` from the finding, such as `"save-meas-coverage"`)
  rather than setting `lint: "warn"`.
- `jobs`: `{"action": "status", "request_id": "ldo-load-1"}` finds a job whose
  id you lost. A `wait` that returns `timed_out` ended the wait, not the job.
- `analyze_results`: the default reply is the answer (`results`, `coverage`,
  `observations`, `failures`); ask for more under `include` (`fields`,
  `per_run`, `outliers`, `signals_available`). `group_by` is a top-level
  argument, never inside a recipe. Results of the `operating_point` recipe are
  in `device_op_points`, keyed by the simulator's literal names (`@m1[gm]`).
  For a staircase signal (DAC steps, line reflections), read each level with a
  `value` recipe on its plateau, or take the whole table with a `waveform`
  recipe at `"format": "csv"`; the inline waveform's bucket statistics blur
  the levels.
- `inspect` reads decks, schematics and libraries, never results:
  `{"queries": [{"kind": "components", "path": "ldo.cir", "detail": "full"}]}`.
- `edit_schematic` edits one `.asc` in a transaction; pass `expected_sha256`
  to commit to a sheet that exists (`inspect` and a dry run report it). `verify_circuit` checks a
  schematic against a netlist with
  `{"path": "amp.asc", "compare": {"reference": "golden.net"}}`.
- Charts: `plot_waveform` draws an interactive chart (`attach_plot` adds a PNG
  you can look at); the `plot` recipe writes a static chart file.

## Recipes the schema lists by name only

Three recipes appear in the tool schema by name only; their arguments are
documented here (every other recipe field — `key`, `sources`, `reduce`,
`field`, `spec` — applies to them unchanged; as with any multi-field recipe,
`reduce`/`spec` on `periodic` or `return_loss` needs `field`):

- `periodic` — `{"metric": "periodic", "signal": …}` plus an optional
  `window` `{start, end}`; returns `period`, `frequency`, `duty_cycle`
  (reducible) from a settled repetitive `.tran` signal.
- `noise_integral` — `{"metric": "noise_integral"}` with optional `signal`,
  `from_hz`, `to_hz`; returns the integrated RMS noise of a `.noise` run over
  that band.
- `return_loss` — `{"metric": "return_loss", "signal": "V(in)/I(Rs)"}` with
  optional `z0` (default 50); returns `return_loss_db`, `vswr`,
  `reflection_coefficient` vs frequency (reducible) from an `.AC` impedance
  trace.

## The response budget

`run_experiments`, `jobs`, `analyze_results` and `inspect` take a `budget` in
estimated tokens (compact characters / 4, minimum 500). If the assembled
response is over it, the server applies the rungs of a fixed reduction ladder
in order until it fits:

| rung | what is removed |
|-|-|
| 0 trim | empty presentation blocks and the identity echo (`source`, `source_hashes`) |
| 1 answer | your detail opt-ins — `include.provenance`, `outliers`, `detail:"full"` |
| 2 shrink | page size, with cursors minted against the smaller page so paging still walks every row |

Facts are never cut: `failures`, `observations`, `warnings`, `completeness`
and spec verdicts always come back whole, and a budget too small for them
returns them anyway and says so.

If you omit `budget`, the server applies its own default
(`[analysis] default_budget`, 4000 tokens) at rung 0 only: a large response
loses empty blocks and the identity echo, nothing else. Set the config value
to `0` to turn that off.
