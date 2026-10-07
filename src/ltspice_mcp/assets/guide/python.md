---
name: python
description: >
  Working in Python, through run_code or your own script: where `api` comes
  from, the six operations as methods, running and waiting on jobs, raw
  traces and log facts, the analysis primitives, errors, detached and resumed
  jobs, and more than one build of a simulator.
---

# Working in Python

`api` holds the six tool operations as methods, over the same engine, job
records and store as the tools. The core's example shows a sweep read back in
one loop; this section is the reference behind it.

## Where `api` comes from

**`run_code`** (when the server serves it) runs a snippet in a warm worker with
`api` already open on the server's working directory. Also in scope: `np`,
`load_raw`, `measurements`, `reference`, `window_and_clean`,
`compute_signal_stats` and `time_weighted_quantiles`. `print()` output and the
repr of a trailing expression come back, so print the result you need, such as
the crossing or the worst corner, rather than the whole table. Each call is a
fresh namespace around the same engine, so keep state on disk: a file, or a
job you find again by its `request_id`. A snippet is bounded by `timeout_s`
(60 s by default, 600 at most), and one runs at a time. It runs with the
server process's own file and process authority, outside the path sandbox.

**Your own Python:**

```python
from ltspice_mcp.api import Api

with Api(working_dir=".") as api:
    print(api.reference())
```

One `Api` per process, and not inside a process that serves the MCP server.
Relative paths resolve against `working_dir`. Closing it, or exiting the
process, cancels the jobs it owns.

## The six operations

`api.run_experiments`, `api.jobs`, `api.analyze_results`, `api.inspect`,
`api.edit_schematic` and `api.verify_circuit` take the tool's arguments as
keywords, nested ones as plain dicts and lists, and return the reply as a dict.

`api.reference()` lists them. `api.reference("analyze_results")` prints one
operation's whole argument tree, with every field's type, default and allowed
values, and a worked example; `help(api.analyze_results)` prints the same.
Read it before guessing an argument. From a shell, without starting the engine:
`python -m ltspice_mcp.api reference analyze_results`.

## Running and waiting

- `run_experiments` waits until the job is terminal and returns the whole
  receipt: `outcome` (`"complete"`, `"partial"` or `"failed"`), `completeness`
  (declared, produced, failed, cancelled, skipped, reused), one row per case under
  `runs["items"]` (`case_id`, `assignments`, and a `status` of `"produced"`
  when the case ran), `failures`, `observations`, and the `analysis` block
  when you attached `analyze`.
- `wait=False` returns the receipt at once; `api.wait(job_id, timeout=None)`
  blocks until the job is terminal. A timeout returns the current snapshot with
  `timed_out: true`; the job keeps running.
- Ctrl-C during a wait raises `ApiInterrupted` carrying the receipt and job id;
  the job keeps running.
- A job belongs to the process that submitted it and stops when that process
  exits. In `run_code` that process is the worker: a worker restart interrupts
  the jobs it owned. `run_experiments(wait=False, detach=True)` hands the job to
  an owner process of its own, so it outlives yours; `api.wait(job_id)` or
  `jobs(action="wait")` reads it from anywhere.

For recovery, submit with `execution={"simulator": "ngspice", "recoverable": True}`
and keep the root receipt's `control_token`. Once the job is terminal, use
`api.jobs(action="resume", job_id=parent_id, resume_request_id="retry-1",
control_token=parent_token)`. Save the child's token too, including when it
finishes immediately. Selection, retry flags, reused successes and no-op receipts
are described in guide section 'tools'; static simulator seeds are in guide
section 'ngspice'.
The audited native Windows LTspice startup contract and its `ltspice_ini`
configuration are also described in guide section 'tools'.

`api.jobs(detach=True, action="resume", ...)` gives the child its own supervising
process. Detach applies only to resume and cannot accompany `raw_page=True`.
The owner independently authorizes the request; a handoff does not grant
authority. The resume's `wait_s` is honored, defaults to zero and is capped at
120 s. For a full wait, call `api.wait` on the child receipt's job id.

## Reading results

```python
# job_id from the receipt, case_id from one of its run rows
r = api.load_raw(job_id=job_id, case_id=case_id)  # or api.load_raw("run.raw")
r.signals  # trace names
r.steps  # one entry per .step iteration
t = r.axis(step=0)  # time or frequency
v = r.trace("V(out)", step=0)  # numpy array, complex on an .AC run
meas = api.measurements(job_id=job_id, case_id=case_id)  # parsed .meas values
```

`api.load_raw(..., plot_index=1, dialect=None)` selects a plot independently of
case/run and step. `plot_index` defaults to zero and takes a nonnegative integer;
`dialect` is `None` or `ltspice`, `ngspice`, `qspice`, `xyce`, and must agree
with captured producer/header evidence. `r.plots`, `r.descriptor` and
`r.plot_index` expose detached inventory and selected descriptor facts. The
descriptor owns analysis type, axis and physical units. `r.table(step=0)` reads
native quantities, keeping the first quantity and complex values as real/imaginary
components; `r.axis()` refuses a native table. Step metadata comes from the same
captured snapshot.

For logs without RAW:

```python
meas = api.measurements(log_path="result.log")  # log_path XOR job_id
facts = api.inspect(queries=[{
    "kind": "results", "path": "result.log", "view": "native_tables",
}])
measured = api.analyze_results(
    sources=[{"log_path": "result.log", "label": "imported"}],
    recipes=[{"metric": "measurements", "key": "meas"}],
)
```

The `measurements` inspect view gives values and recorded range/AT metadata;
`native_tables` gives literal printed entries and frequency rows. These are
detached plain facts for caller code to arrange or export, with unknown units
left unknown and no invented plot or step identity. Omit `plot_index` on log
views. Whole-log measurements reject `step`/`all_steps`. Direct log imports
guess no RAW sibling; malformed RAW in a mixed request fails only RAW recipes.
Capture identity, companion drift and initialization timeouts follow guide
section 'tools'.

Arrays are copies, safe to modify. Trace math across steps and its statistics
are in guide section 'signals'. `api.analyze_results(...)` returns every row of
every recipe; name `include={"per_run": True}` for the per-run rows. Variation
values are SPICE literals (`"5p"`): `parse_spice_value` reads one.

## Analysis primitives

Importable from `ltspice_mcp.api`; arrays in, dicts out.

- Transient: `window_and_clean`, `compute_signal_stats`,
  `time_weighted_quantiles`, `analyze_edge`, `analyze_pulse_response`,
  `analyze_disturbance_response`, `analyze_timing_between`,
  `analyze_periodic`, `analyze_thd`, `analyze_tone`,
  `compute_measurement_stats`.
- AC: `prepare_ac_arrays`, `gain_at_frequencies`, `detect_crossings`,
  `find_crossings_any_quantity`, `compute_filter_metrics`,
  `compute_stability_metrics`, `compute_roll_off`, `compute_resonances`,
  `compute_return_loss`, `integrate_noise`, `classify_filter`,
  `analyze_ac_structure`, `unwrap_phase_safe`, `log_interp`,
  `log_interp_complex`.

## Errors

- `ApiValidationError` (a `ValueError`): the arguments did not validate; the
  message is the one the tool would give.
- `ApiCallError`: the call failed. `.code` names why, `.job_id` and the full
  payload keep every recovery handle.
- A per-item failure, or `outcome` `"partial"` or `"failed"`, is returned
  data, not an exception: read `failures`, `observations` and `warnings`.
- `ApiSessionError` (with `ApiClosedError`): a second `Api` in one process, a
  call after `close()`, or a call from a forked child.
- A refused path raises the engine's path error, with a note naming the setting
  that widens the sandbox.

## Controls that exist only on the wire

`budget` and `execution.wait_s` are dropped, with a warning in the result.
A cursor or continuation is rejected. `raw_page=True` on a call accepts them
and returns exactly one page as the tool would.

## More than one build of a simulator

Name each further executable, in `[simulator.executables]` of the config
(`xvii = "C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"`) or as
`Api(working_dir=".", simulator_executables={"xvii": "C:/.../XVIIx64.exe"})`,
and select it per run with `execution={"simulator": "ltspice:xvii"}`. The plain
family name still runs the default build. `inspect(kind="capabilities")` lists
the named builds under `named_executables`. Each job records the executable
that ran it.
