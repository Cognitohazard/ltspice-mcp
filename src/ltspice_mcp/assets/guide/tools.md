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
instead of guessing a name. Always use it when your tool schema has no
argument descriptions, as with the server's default
`[tools] tool_listing = "compact"`, which leaves them out.

## Recovering a job

Submit with `execution={"simulator": "ngspice", "recoverable": true}` and save
the receipt's `control_token`, even if the job finishes immediately. Recovery
saves the job's validated inputs and the simulator's startup settings, so a
resumed case runs with the same inputs. It currently supports ngspice on Linux
and native Windows. On native Windows it also supports one LTspice
executable, the LTspice 26.0.2 build whose startup behavior was checked,
matched by the SHA-256 hash of its file. Any other LTspice executable, a later
version included, is refused (`recovery_startup_unsupported`).

A recoverable request is refused before submission, with an error code that
names the reason, when a deck reads an include in place at run time
(`allow_live_includes`, `recovery_live_dependency`), has a `.control` block
(`recovery_control_unsupported`), loads an external module with `.load`,
`.osdi` and the like or has an `A` device (`recovery_external_module`), or
calls one of the random functions `rand`, `random`, `gauss`, `agauss`, `unif`,
`aunif`, `flat`, `sgauss`, `limit`, `mc`, `poisson`, `white`, `flicker`,
`trnoise` or `trrandom` (`recovery_random_unsupported`). A source that reads a
file and a function the server does not know are refused the same way. On
ngspice, `simulator_seed` allows `agauss`, `gauss`, `aunif`, `unif` and
`limit` (guide section 'ngspice').

For LTspice, use `execution={"simulator":"ltspice","recoverable":true}`.
Each case starts LTspice with its own copy of an LTspice settings file, so the
original is not changed. The file is `simulator.ltspice_ini` (environment
`LTSPICE_MCP_LTSPICE_INI`) when that is set, and `%APPDATA%/LTspice.ini`
otherwise. It must already hold the answers LTspice saves from its startup
prompts, which are an `[Options]` `UUID`, a `CaptureAnalytics` choice and a
`LastWebUpdateQuery` time no more than 15 days old. The server checks them at
submission, at resume and before each case starts. A missing value returns
`recovery_startup_unsupported`, and an older time `recovery_startup_stale`.
Resolve startup prompts in LTspice first, because the server does not answer
dialogs or write these values.

A resume creates a child job. The original job and its children form a
lineage, and its head is the newest of them, which a resume reply returns as
`head_job_id`. Only the head can be resumed; an older job returns
`recovery_stale_parent`. Once the head is terminal, recover it with
`jobs(action="resume", job_id=..., resume_request_id="retry-1", control_token=...)`.
Cases that failed with code `server_restarted` (the owning process exited) or
`job_deadline` are retried by default; add `retry_failed=true` for other
failed cases or `retry_cancelled=true` for explicit cancellation.
`case_ids` limits the retry to those of the eligible cases, and naming an
unknown or `produced` case returns `recovery_case_selection`. `wait_s`
defaults to zero and is capped at 120 s. The child copies the parent's
`produced` cases with their original `.raw` and `.log` files instead of
running them again. `completeness.reused` counts them, each run's
`attempt.execution_job_id` names the job that ran it, and `attempt.reused` is
true for a copied case.

The child has a new job id and token. Repeating the same resume request returns
that same child, and a repeat with a different selection or different retry
flags returns `idempotency_conflict`. If no case is eligible, the resume is a
no-op (`resumed=false`) and returns no child token. To retry only the analysis,
call `analyze_results`. Of the `jobs` replies, only `resume` returns a
`control_token`. To give the child an owner process of its own, so it outlives
the process that resumed it, see guide section 'python'.

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
- `edit_schematic` edits one `.asc` in a transaction. Changing a sheet that
  exists requires `expected_sha256`, the `sha256` that `inspect` or a dry run
  reports; if the file has changed since, the edit returns `revision_conflict`
  and writes nothing.
- `verify_circuit` checks a schematic against a netlist with
  `{"path": "amp.asc", "compare": {"reference": "golden.net"}}`.
- Charts: `plot_waveform` draws an interactive chart (`attach_plot` adds a PNG
  you can look at); the `plot` recipe writes a static chart file.

## Reading RAW plots and logs

A RAW artifact can contain several plots. `plot_index` selects one by its
zero-based position in the file; omit it to read the first plot. It is
separate from the case (`case_id` or `run_index`) and from `step`, which
selects a step inside the plot. When supplied, it must be a nonnegative
integer. Page through the file's plots with
`inspect(queries=[{"kind": "results", "path": "run.raw", "view": "plots"}])`.
Use `view:"signals"` to list the selected plot's traces with their declared
type and unit. In a plot with a sampled axis, the first quantity is that axis
(time, frequency or the swept value). `view:"table"` reads a plot without
one, such as an `.op` or `.tf` result, and returns `no_table` for a plot that
has one. Its rows are every value of every quantity, the first included;
complex values are `{real, imag}`, and each row's `unit` is the one
`view:"signals"` lists for its quantity. A read by job takes `job_id` plus
`run_index` or `case_id` instead of `path`.

Each `analyze_results` source, `plot_waveform` and `run_experiments.analyze`
take the same `plot_index` and `dialect`. `dialect` is null or `ltspice`,
`ngspice`, `qspice`, `xyce`, and names the simulator that wrote the RAW. Leave
it null unless a read fails because the file's header does not name its
simulator; a `dialect` that contradicts the job's simulator or the header is
refused. Step parameters are read at the same time as the RAW, from the RAW
itself or the `.log` beside it, and are not read again later. Guide section
'python' covers Python arrays and tables.

To read a log that has no RAW, use the `measurements` recipe:

```json
{"sources": [{"log_path": "result.log", "label": "imported"}],
 "recipes": [{"metric": "measurements", "key": "meas"}]}
```

The recipe reads every measurement in the log in one pass, so its rows name no
plot or step. It refuses `step` and `all_steps`, because a measurement lists
one value per `.step` iteration in the order the log prints them, and that
position is not a RAW step index. In a request that also has RAW recipes, a
missing or malformed RAW fails only those recipes, and the `measurements`
recipe still returns its values. A `log_path` source does not look for a RAW
beside the log. A `raw_path` source reads the RAW and the `.log` of the same
name, so use it to read both from imported files.

For pages of log values as printed, use
`inspect(queries=[{"kind": "results", "path": "result.log", "view": "measurements"}])`
for each measurement's values with its `range_from` and `range_to` (a
`FROM`/`TO` window) or its `at` point, or `view:"native_tables"` for the value
blocks printed in the log, as `name = value` entries and `Index frequency`
table rows. Omit `plot_index` on log views. Each native row includes its
block's first and last line numbers (`line_start`, `line_end`), the reason
the block ended (`closed_by`) and its printed title; complex values are given
as `real` and `imag`, a unit the log does not print is null, and rows are not
assigned to a plot or step. A log that reads without error does not show that
the simulation finished. A read by `job_id` works once all the job's runs have
finished, and only on a case that is `produced` or kept a partial raw
(`execution.keep_partial_raw`).

A page cursor works only with the same query, and only while the RAW, its
`.log` and the simulator's console log (`.exe.log`) are unchanged, including
whether each of them exists. If any of these files changes, appears or is
deleted, the next page is refused, so start a new read.

`analyze_results` reads every source before it evaluates any recipe. If that
first read runs out of time, the call returns `analysis_deadline` with no
`result_set_id` or cursor; retry the original request with fewer sources or a
larger configured analysis time budget. A retry reads the sources again,
because the timeout is not recorded against them. After the first read, every
call completes at least one recipe evaluation. Work left over when the time
budget runs out continues when you pass the reply's `next` block
(`result_set_id` and `cursor`) as `continue`.

## Recipes the schema lists by name only

Three recipes appear in the tool schema by name only, so their arguments are
listed here. Every other recipe field (`key`, `sources`, `reduce`, `field`,
`spec`) applies to them unchanged. As with any multi-field recipe, `spec` on
`periodic` or `return_loss` needs `field`, and a `reduce` without `field`
reduces every field.

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
| 0 trim | empty presentation blocks and `source_hashes`, the row per source that repeats its label, plot and file snapshot (an attached analysis's included) |
| 1 answer | your detail opt-ins — `include.provenance`, `outliers`, `detail:"full"` |
| 2 shrink | page size; cursors are created for the smaller page, so paging still reaches every row |

Rung 2 lowers one page limit on every list in the reply at once, to the
largest limit at which all their rows fit, so the values of three recipes
share the budget instead of each taking all of it. If even one row per list
is over the budget, a `run_experiments` or `jobs` receipt contains no per-case
rows at all, so its minimum size is the same for 4 cases or 400.
`completeness` and `runs.total` still count the runs, `jobs(runs)` pages them
from `runs.next_cursor`, and `analyze_results` over the `job_id` returns the
attached analysis's rows. `reduce` results and spec verdicts are kept.

`failures`, `observations`, `warnings`, `completeness` and spec verdicts are
never cut. A budget too small for them still returns them in full, with an
observation that the reply is over the budget.

If you omit `budget`, the server applies its own default
(`[analysis] default_budget`, 4000 tokens) at rung 0 only, so a large response
loses only its empty blocks and `source_hashes`. Set the config value to `0`
to turn that off.
