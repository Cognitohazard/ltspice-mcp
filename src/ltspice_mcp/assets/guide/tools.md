---
name: tools
description: >
  Calling the tools directly: which tool does what, progress on a running
  job, the reference lookup, the recipes the schema lists by name only, and
  the response budget.
---

# Using the tools

Six tools: `run_experiments`, `jobs`, `analyze_results`, `inspect`,
`edit_schematic`, `verify_circuit`. The loop is: write the deck to a file, run it
with `run_experiments` (attach `analyze.recipes` and the numbers come back in the
same response), follow a receipt with `jobs`, measure a finished job with
`analyze_results`.

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
points written and the last time or frequency reached, to set against the
deck's own stop value. A simulator writes in buffered blocks, so judge over looks minutes
apart: if `points` has not moved across them, the case is not advancing, and
`jobs(action="cancel")` stops it.

`inspect(kind="reference")` is the lookup for this surface's own vocabulary.
Each tool holds many capabilities behind a discriminator — twenty-one
`analyze_results` recipes, eleven `edit_schematic` ops, the variation kinds,
the `verify_circuit` checks, the `jobs` actions — and a `query` in plain words
returns the closest ones with their fields, types, defaults and units. With no
`query` it returns the table of contents. Reach for it instead of guessing a
name or re-reading the guide, and always when the server is serving the
compact tool listing, where the per-argument descriptions are not on the wire.

## Recipes the schema lists by name only

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

Rows keep their shape at every rung: a row is always an object with the same
keys, so a tight budget returns fewer rows, never differently shaped ones.

Facts are never cut at any rung: `failures`, `observations`, `warnings`,
`completeness` and spec verdicts always come back whole, and a budget too small
for them returns them anyway and says so. The budget is presentation only — it
is not part of a result's identity, so the same request at two budgets shares
one result set and one set of cursors.

If you omit `budget`, the server applies its own default
(`[analysis] default_budget`, 4000 tokens) at **rung 0 only**: a large
response loses empty blocks and the identity echo, nothing else. Detail you
asked for is never removed by the default. Set the config value to `0` to
turn that off.
