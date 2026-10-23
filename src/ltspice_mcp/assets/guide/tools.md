---
name: tools
description: >
  Calling the tools directly: which call does what, a sweep in one call,
  notes per tool, progress and recovery on a job, RAW plots and log facts,
  the reference lookup, recipes the schema lists by name only, and the
  response budget.
---

# Using the tools

Write the deck to a file and run it with `run_experiments`; attach
`analyze.recipes` to get the numbers in the same response. Follow a receipt
with `jobs`, and measure a finished job with `analyze_results`.

| To … | Call |
|-|-|
| run a deck — once, swept, or perturbed | `run_experiments(circuits=[{"path": …}], variations=[…])` |
| get the measurements without a second round trip | `run_experiments(analyze={"recipes": […]})` |
| follow, cancel, resume, or page a job's runs | `jobs(action="status"\|"wait"\|"cancel"\|"resume"\|"list"\|"runs")` |
| measure a finished job | `analyze_results(sources=[{"job_id": …}], recipes=[…])` |
| read `.meas` results | recipe `{"metric": "measurements"}` |
| page RAW plots, trace descriptors, or printed log facts | `inspect(queries=[{"kind": "results", "path": …, "view": …}])` |
| a scalar, a trace, a chart | recipes `value`, `waveform`, `plot` |
| show a run in the user's own LTspice, when they ask to see it there | `plot_waveform(job_id=…, signals=[…], in_ltspice=true)` |
| let the user plot nets by clicking the sheet in LTspice | `plot_waveform(job_id=…, in_ltspice=true)` on a job that ran the sheet (guide section 'schematics') |
| device operating points (gm/gds/vth) | recipe `{"metric": "operating_point", "device": "M1"}` |
| AC corner, gain, slope, crossing, stability | recipes `bode_filter`, `bode_point`, `bode_slope`, `bode_crossing`, `stability`, `ac_structure` |
| transient stats, edges, timing, THD | recipes `signal_stats`, `edges`, `timing`, `periodic`, `transient_response`, `thd` |
| amplitude and phase at one frequency | recipe `tone` (guide section 'signals') |
| symbol geometry, a net, a component list, a model | `inspect(kind="symbol"\|"net"\|"components"\|"model")` |
| nested devices, scoped ports, effective parameters | `inspect(kind="hierarchy", path=..., simulator=...)` |
| find the recipe, op or check for a job, and its fields | `inspect(kind="reference", query="phase margin")` |
| what the user has open in LTspice, and which sheet is in front | `inspect(kind="open_in_ltspice")` |
| LTspice's own reference: shortcuts, menus, the waveform viewer, `.MEAS` | `inspect(kind="simulator_docs")`, then `name=` one of them |
| create or mutate an `.asc` | `edit_schematic(target=…, ops=[…])` |
| check a sheet against its netlist, or render it | `verify_circuit(path=…)` |
| open a sheet in the user's own LTspice, when they ask to see it there | `verify_circuit(path=…, in_ltspice=true)` |

A case has no time limit unless `execution.run_timeout_s` (or the server's
`[simulation] run_timeout`) sets one. While a job runs, each `jobs(status)` or
`jobs(wait)` receipt contains a `run_progress` observation for each running
case. It reports the points written and the last time or frequency reached,
which you can compare with the deck's own stop value. A simulator writes in
buffered blocks, so compare readings a few minutes apart. If `points` has not
changed, the case is not advancing, and `jobs(action="cancel")` stops it. A
stopped case's raw is deleted unless the run set
`execution.keep_partial_raw: true`. With that set, its results stay readable
up to the point where the run stopped, and every read reports a `partial_raw`
observation.

`inspect(kind="reference")` finds a recipe, schematic op, variation kind,
check or job action from plain words (`query="phase margin"`) and returns its
fields, types, defaults and units; with no `query` it lists them all. Use it
instead of guessing a name. Always use it with the compact tool listing, which
omits the argument descriptions.

## Recovering a job

Submit with `execution={"simulator": "ngspice", "recoverable": true}` and save
the receipt's `control_token`, even if the job finishes immediately. Recovery
captures validated electrical inputs and controlled startup. It currently
supports ngspice on Linux and native Windows, and the audited LTspice 26.0.2
executable on native Windows. A job that uses live dependencies, caller
control scripts, external modules or unbound simulator randomness is refused
before submission. For static randomness, see guide section 'ngspice'.

For LTspice, use `execution={"simulator":"ltspice","recoverable":true}`.
The established LTspice settings come from `simulator.ltspice_ini`
(environment `LTSPICE_MCP_LTSPICE_INI`) or `%APPDATA%/LTspice.ini`. Each
attempt runs with a private copy, and the original file is not changed. The
captured profile needs a real LTspice update-query time less than 15 days old,
at submission and again at resume. Resolve startup prompts in LTspice first,
because the server does not answer dialogs or change consent. Other LTspice
builds are refused until their startup behavior is verified.

Once the lineage head is terminal, recover it with
`jobs(action="resume", job_id=..., resume_request_id="retry-1", control_token=...)`.
Interrupted cases are retried by default; add `retry_failed=true` for simulator
failures or `retry_cancelled=true` for explicit cancellation. `case_ids` selects
eligible unfinished cases; `wait_s` defaults to zero and is capped at 120 s.
Successful cases keep their original artifacts. `completeness.reused` counts
them, and each run's `attempt` identifies the execution that produced it.

The child has a new job id and token. Repeating the same resume request returns
that same child, and a repeat with a different selection or different retry
flags returns a conflict. If no case is eligible, the resume is a no-op
(`resumed=false`) and returns no child token. To retry only the analysis, call
`analyze_results`. Read-only job replies never return tokens. For a child with
its own owner process, see guide section 'python'.

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
from the log, parsed and with SI units. Each case's row contains it as
`value.measured.vout_dc`. If that case's `.meas` failed, the value is null and
`value.failed_measurements` lists it. On ngspice, measure the trace with a
recipe instead (guide section 'ngspice'). An `assign` target must exist in the
deck. If the deck restricts what it saves, `.save` every signal a `.meas` uses;
lint blocks a mismatch. A case that produced nothing is counted in
`completeness`, and `outcome` is `"partial"`.

## Notes per tool

- `run_experiments`: a reused `request_id` returns `idempotency_conflict` when
  the arguments differ, a deck has changed since the job ran, or the simulator
  is now a different build; submit under a new id to run again. For changed
  arguments, the refusal names the job created under that id, which `jobs`
  reads by its `job_id`. A `skipped` case means lint blocked its deck. Fix the
  deck, or drop that one rule with `suppress` (the `rule_id` from the finding,
  such as `"save-meas-coverage"`) rather than setting `lint: "warn"`.
- `jobs`: `{"action": "status", "request_id": "ldo-load-1"}` finds a job whose
  id you lost. A `wait` that returns `timed_out` ended only the wait; the job
  is still running. A receipt's `status` describes the job (`completed`,
  `completed_with_failures`, `failed`, ...). Each run row's `status` describes
  its case (`produced` when it ran; `failed`, `cancelled` or `skipped` when it
  did not).
- `analyze_results`: the default reply contains `results`, `coverage`,
  `observations` and `failures`; ask for more under `include` (`fields`,
  `per_run`, `outliers`, `signals_available`). Each failure row is one reason.
  A reason that affected several runs has `count` and `wheres` (the first 10
  places). `group_by` is a top-level argument, never inside a recipe. Results
  of the `operating_point` recipe are in `device_op_points`, keyed by the
  simulator's literal names (`@m1[gm]`); a device inside a subcircuit is
  found by its path (`"device": "x1.m1"`, guide section 'operating-points').
  For a staircase signal (DAC steps, line reflections), read each level with a
  `value` recipe on its plateau, or take the whole table with a `waveform`
  recipe at `"format": "csv"`; the inline waveform's bucket statistics blur
  the levels. Runs with no number for a reduced or spec'd field (a failed
  `.meas`) are counted in `no_value_count`, and a spec stays `indeterminate`
  while there are any, unless you set `allow_incomplete`.
- `inspect` reads decks, schematics, libraries and result facts:
  `{"queries": [{"kind": "components", "path": "ldo.cir", "detail": "full"}]}`.
- `edit_schematic` edits one `.asc` in a transaction; pass `expected_sha256`
  to commit to a sheet that exists (`inspect` and a dry run report it).
- `verify_circuit` checks a schematic against a netlist with
  `{"path": "amp.asc", "compare": {"reference": "golden.net"}}`.
- Charts: `plot_waveform` draws an interactive chart (`attach_plot` adds a PNG
  you can look at); the `plot` recipe writes a static chart file.

## Reading RAW plots and logs

A RAW artifact can contain several plots. `plot_index` selects one independently
of the job's case and the plot's step; omit it to read the first plot. When
supplied, it must be a nonnegative integer. Page through the file's plots with
`inspect(queries=[{"kind": "results", "path": "run.raw", "view": "plots"}])`.
Use `view:"signals"` for the selected trace descriptors or `view:"table"` for
native quantities without a sampled axis. Tables keep their first quantity;
complex values are `{real, imag}`, and units come from the selected descriptor.
A read by job takes `job_id` plus `run_index` or `case_id` instead of `path`.

Each `analyze_results` source, `plot_waveform` and `run_experiments.analyze`
take the same plot selection. `dialect` is null or `ltspice`, `ngspice`,
`qspice`, `xyce`; a `dialect` you name must agree with the producer and the
captured header. Step metadata comes from that capture and is not re-read
from a sibling log later. Guide section 'python' covers Python arrays and
native tables.

To read a log that has no RAW, use the `measurements` recipe:

```json
{"sources": [{"log_path": "result.log", "label": "imported"}],
 "recipes": [{"metric": "measurements", "key": "meas"}]}
```

The recipe reads measurements once over the whole log, with no plot or step
identity. `step` and `all_steps` are refused for these facts, because a
measurement vector's ordinal is not a RAW step. In a mixed request, a missing
or malformed RAW fails the RAW recipes but keeps the valid measurements. A
direct log import does not guess a sibling RAW file.

For pages of literal log facts, use
`inspect(queries=[{"kind": "results", "path": "result.log", "view": "measurements"}])`
for values and range/AT metadata, or `view:"native_tables"` for printed entries
and frequency rows. Omit `plot_index` on log views. Native rows keep physical
section boundaries, closure facts and printed labels; complex values keep
real/imaginary components, unknown units stay null, and no plot or step identity
is invented. Reading the whole log does not prove that the solve completed. A
read by job still requires a terminal job and a produced case.

A cursor is bound to the source snapshot and the selection, including the RAW,
log and console bytes and an explicit record of any that are absent. If a
companion file changes, start a new read. Analysis captures source identities
and diagnostics before resumable recipe work starts. An initialization timeout
returns `analysis_deadline` without a result set or cursor; retry the original
request with fewer sources or a larger configured analysis time budget. The
timeout is not saved as a permanent fault of the source. After
initialization, the existing minimum-work and continuation policy applies.

## Recipes the schema lists by name only

Three recipes appear in the tool schema by name only, so their arguments are
listed here. Every other recipe field (`key`, `sources`, `reduce`, `field`,
`spec`) applies to them unchanged. As with any multi-field recipe, `spec` on
`periodic` or `return_loss` needs `field`, and a `reduce` without `field`
covers every field.

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
| 0 trim | empty presentation blocks and the per-run identity echo (`source_hashes`, an attached analysis's included) |
| 1 answer | your detail opt-ins — `include.provenance`, `outliers`, `detail:"full"` |
| 2 shrink | page size; cursors are created for the smaller page, so paging still reaches every row |

Rung 2 lowers one page limit on every list in the reply at once, to the
largest limit at which all their rows fit, so the values of three recipes
share the budget instead of each taking all of it. If even one row per list
is over the budget, a `run_experiments` or `jobs` receipt contains no per-case
rows at all, so its minimum size is the same for 4 cases or 400.
`completeness` and `runs.total` still count the runs, `jobs(runs)` pages them
from `runs.next_cursor`, and `analyze_results` over the `job_id` returns the
attached analysis's rows. Reductions and verdicts are kept.

`failures`, `observations`, `warnings`, `completeness` and spec verdicts are
never cut. A budget too small for them still returns them in full, with an
observation that the reply is over the budget.

If you omit `budget`, the server applies its own default
(`[analysis] default_budget`, 4000 tokens) at rung 0 only, so a large response
loses only its empty blocks and the identity echo. Set the config value to `0`
to turn that off.
