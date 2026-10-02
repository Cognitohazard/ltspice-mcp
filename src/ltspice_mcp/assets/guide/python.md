---
name: python
description: >
  Working in Python, through run_code or your own script: where `api` comes
  from, the six operations as methods, running and waiting on jobs, raw
  traces and measurements, the analysis primitives, errors, detached jobs,
  and a second LTspice build.
---

# Working in Python

The six tools are also methods on one Python object, `api`, over the same
engine, job records and store. A loop, a decision between runs, or numpy on
the traces is one Python call instead of a tool call per step, and every result
comes back whole: no pages, no cursors, no response budget.

## Where `api` comes from

**`run_code`** (when the server serves it) runs a snippet in a warm worker with
`api` already open on the server's working directory. Also in scope: `np`,
`load_raw`, `measurements`, `reference`, `window_and_clean`,
`compute_signal_stats` and `time_weighted_quantiles`. `print()` output and the
repr of a trailing expression come back. Each call is a fresh namespace around
the same engine, so keep state on disk: a file, or a job you find again by its
`request_id`. A snippet is bounded by `timeout_s` (60 s by default, 600 at
most), and one runs at a time. It runs with the server process's own file and
process authority, outside the path sandbox.

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

```python
receipt = api.run_experiments(
    request_id="rc-sweep-1",
    circuits=[{"path": "rc.cir"}],
    variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k", "4k"]}}],
)
receipt["outcome"]  # "complete", "partial" or "failed"
receipt["completeness"]  # declared, produced, failed, cancelled, skipped
for row in receipt["runs"]["items"]:
    row["case_id"], row["assignments"], row["status"]  # status "produced" ran
```

- `run_experiments` waits until the job is terminal and returns every run row,
  plus `failures`, `observations` and, when you attached `analyze`, the
  `analysis` block.
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
- The same `request_id` with the same arguments returns the original receipt
  instead of running again.

## Reading results

```python
job = receipt["job_id"]
cid = receipt["runs"]["items"][0]["case_id"]
r = api.load_raw(job_id=job, case_id=cid)  # or api.load_raw("path/to/run.raw")
r.signals  # trace names
r.steps  # one entry per .step iteration
t = r.axis(step=0)  # time or frequency
v = r.trace("V(out)", step=0)  # numpy array, complex on an .AC run
meas = api.measurements(job_id=job, case_id=cid)  # parsed .meas values
```

Arrays are copies, safe to modify. A step's traces share that step's axis;
never combine traces across steps. `api.analyze_results(...)` returns every
row of every recipe; name `include={"per_run": True}` for the per-run rows.
Variation values are SPICE literals (`"5p"`): `parse_spice_value` reads one.

## Analysis primitives

Importable from `ltspice_mcp.api`; arrays in, dicts out.

- Transient: `window_and_clean`, `compute_signal_stats`,
  `time_weighted_quantiles`, `analyze_edge`, `analyze_pulse_response`,
  `analyze_disturbance_response`, `analyze_timing_between`,
  `analyze_periodic`, `analyze_thd`, `compute_measurement_stats`.
- AC: `prepare_ac_arrays`, `gain_at_frequencies`, `detect_crossings`,
  `find_crossings_any_quantity`, `compute_filter_metrics`,
  `compute_stability_metrics`, `compute_roll_off`, `compute_resonances`,
  `compute_return_loss`, `integrate_noise`, `classify_filter`,
  `analyze_ac_structure`, `unwrap_phase_safe`, `log_interp`,
  `log_interp_complex`.

Take statistics of a transient trace with `compute_signal_stats` and
`time_weighted_quantiles`, not `np.mean` or `np.percentile`: the timestep
varies and samples pack around edges, so a plain sample average over-weights
them (guide section 'signals').

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

## A second LTspice build

A process runs one executable per simulator family. To run a second build
beside the server, open a second process on the same working directory:
`Api(working_dir=".", simulator_exe="C:/path/to/XVIIx64.exe")`. Each job
records the executable that ran it, so the two builds' results stay apart in
the shared records.
