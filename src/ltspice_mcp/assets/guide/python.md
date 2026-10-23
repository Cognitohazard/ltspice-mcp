---
name: python
description: >
  Working in Python, through run_code or your own script: where `api` comes
  from, the six operations as methods, running and waiting on jobs, raw
  traces and log facts, the analysis primitives, errors, detached and resumed
  jobs, and more than one build of a simulator.
---

# Working in Python

`api` provides the six tool operations as methods, on the same engine, job
records and store as the tools. The core has an example that reads a sweep
back in one loop.

## Where `api` comes from

**`run_code`** (when the server serves it) runs a snippet in a warm worker with
`api` already open on the server's working directory. The snippet also has
`np`, `load_raw`, `measurements`, `reference`, `window_and_clean`,
`compute_signal_stats` and `time_weighted_quantiles` in scope. `print()`
output and the repr of a trailing expression are returned, so print the result
you need, such as the crossing or the worst corner, rather than the whole
table. Each call gets a fresh namespace around the same engine, so keep state
on disk, in a file or in a job you can find again by its `request_id`. A
snippet is bounded by `timeout_s` (60 s by default, 600 at most), and one runs
at a time. A blocking wait in it (`run_experiments` with `wait=True`, or
`api.wait`) ends just before `timeout_s` and raises `ApiInterrupted` naming
its job. The job keeps running, so read it with `api.wait(job_id)` in a later
snippet, or submit a long run with `wait=False` from the start. The snippet
runs with the server process's own file and process authority. A sandbox
(`[security] allowed_paths`) applies only to the paths passed to `api` calls
and does not restrict the snippet's own code.

**Your own Python:**

```python
from ltspice_mcp.api import Api

with Api(working_dir=".") as api:
    print(api.reference())
```

Create one `Api` per process, and not inside a process that serves the MCP
server.
Relative paths resolve against `working_dir`. Closing it, or exiting the
process, cancels the jobs it owns.

## The six operations

`api.run_experiments`, `api.jobs`, `api.analyze_results`, `api.inspect`,
`api.edit_schematic` and `api.verify_circuit` take the tool's arguments as
keywords, nested ones as plain dicts and lists, and return the reply as a dict.

`api.reference()` lists them. `api.reference("analyze_results")` prints one
operation's whole argument tree, with every field's type, default and allowed
values, and a worked example; `help(api.analyze_results)` prints the same.
Read it before guessing an argument. From a shell,
`python -m ltspice_mcp.api reference analyze_results` prints it without
starting the engine.

## Running and waiting

- `run_experiments` waits until the job is terminal and returns the whole
  receipt, with `outcome` (`"complete"`, `"partial"` or `"failed"`),
  `completeness` (declared, produced, failed, cancelled, skipped, reused), one
  row per case under `runs["items"]` (`case_id`, `assignments`, and the case's
  `status`), `failures`, `observations`, and the `analysis` block when you
  attached `analyze`. A row's status describes its case. A case that ran is
  `"produced"` while the job is `"completed"`, so select rows on `"produced"`.
- `wait=False` returns the receipt at once; `api.wait(job_id, timeout_s=None)`
  blocks until the job is terminal. A timeout returns the current snapshot with
  `timed_out: true`; the job keeps running.
- Ctrl-C during a wait raises `ApiInterrupted` with the receipt and job id;
  the job keeps running.
- A job belongs to the process that submitted it and stops when that process
  exits. In `run_code` that process is the worker, so a worker restart
  interrupts the jobs it owned. `run_experiments(wait=False, detach=True)`
  hands the job to an owner process of its own, so it outlives yours;
  `api.wait(job_id)` or `jobs(action="wait")` reads it from anywhere.

For recovery, submit with `execution={"simulator": "ngspice", "recoverable": True}`
and keep the root receipt's `control_token`. Once the job is terminal, use
`api.jobs(action="resume", job_id=parent_id, resume_request_id="retry-1",
control_token=parent_token)`. Save the child's token too, including when it
finishes immediately. Guide section 'tools' describes selection, retry flags,
reused successes and no-op receipts, and the audited LTspice startup on native
Windows with its `ltspice_ini` setting. Guide section 'ngspice' covers static
simulator seeds.

`api.jobs(detach=True, action="resume", ...)` gives the child its own supervising
process. Detach applies only to resume and cannot be combined with
`raw_page=True`. The owner process authorizes the request itself, because a
handoff does not grant authority. The resume's `wait_s` is honored, defaults
to zero and is capped at 120 s. For a full wait, call `api.wait` on the child
receipt's job id.

## Reading results

```python
# job_id from the receipt, case_id from one of its run rows
r = api.load_raw(job_id=job_id, case_id=case_id)  # or api.load_raw("run.raw")
r.signals  # trace names
r.steps  # one entry per .step iteration
t = r.axis(step=0)  # time or frequency
v = r.trace("V(out)", step=0)  # numpy array, complex on an .AC run
vop = r.value("V(out)", step=0)  # one number, from an operating point (.op)
meas = api.measurements(job_id=job_id, case_id=case_id)  # parsed .meas values
ops = api.device_op_points(job_id=job_id, case_id=case_id)  # ops["m1"]["gm"], .op log, 3 digits
d = r.compare(api.load_raw("archived.raw"))  # d["signals"]["V(out)"]["max_abs"], per signal
```

`r.compare(other, signals=None, step=0, other_step=None)` measures how far
another run's traces are from these. For each signal it reports the largest
difference, that difference as a share of this run's peak, and where it
occurs, and it lists the signals that only one run has. Signals are paired by
name, ignoring case, and by their path inside subcircuits, so an LTspice 26
run can be compared with an XVII run. Steps are paired by their parameters.
The other run is read at this run's points over the span both cover,
interpolated where its axis differs (as in a compressed LTspice raw, guide
section 'ltspice'). You choose the tolerance.

`api.load_raw(..., plot_index=1, dialect=None)` selects a plot independently of
case/run and step. `plot_index` defaults to zero and takes a nonnegative integer;
`dialect` is `None` or `ltspice`, `ngspice`, `qspice`, `xyce`, and must agree
with captured producer/header evidence. `r.plots` returns a detached list of
the file's plots, and `r.descriptor` and `r.plot_index` describe the selected
one. The descriptor contains the analysis type, axis and physical units.
`r.table(step=0)` reads native quantities; it keeps the first quantity and
returns complex values as real and imaginary components. `r.axis()` refuses a
native table. Step metadata comes from the same captured snapshot.

For a log without a RAW file:

```python
meas = api.measurements(log_path="result.log")  # log_path XOR job_id
facts = api.inspect(
    queries=[
        {
            "kind": "results",
            "path": "result.log",
            "view": "native_tables",
        }
    ]
)
measured = api.analyze_results(
    sources=[{"log_path": "result.log", "label": "imported"}],
    recipes=[{"metric": "measurements", "key": "meas"}],
)
```

The `measurements` inspect view returns values and recorded range/AT metadata,
and `native_tables` returns the literal printed entries and frequency rows.
Both are detached plain data for your code to arrange or export. Unknown units
stay unknown, and no plot or step identity is invented. Omit `plot_index` on
log views. Whole-log measurements reject `step`/`all_steps`. A direct log
import does not guess a sibling RAW file, and in a mixed request a malformed
RAW fails only the RAW recipes. Guide section 'tools' covers capture identity,
changed companion files and initialization timeouts.

Arrays are copies, so you can modify them. Trace math across steps and its
statistics are in guide section 'signals'. `api.analyze_results(...)` returns
every row of every recipe; pass `include={"per_run": True}` for the per-run
rows. Variation values are SPICE literals (`"5p"`), and `parse_spice_value`
parses one.

To keep one run's numbers for later, write them to a file you choose, in a
layout you decide. Each read takes one line:

```python
import json
import numpy as np

op = api.load_raw(job_id=op_job, case_id=case_id)  # an .op run
ac = api.load_raw(job_id=ac_job, case_id=case_id)  # an .ac run
h = ac.trace("V(out)")  # complex; JSON takes it as magnitude and phase
row = {
    "case_id": case_id,
    "op": {name: op.value(name) for name in ("V(out)", "I(R1)")},
    "devices": api.device_op_points(job_id=op_job, case_id=case_id),
    "ac": {
        "f": ac.axis().tolist(),
        "db": (20 * np.log10(np.abs(h))).tolist(),
        "deg": np.degrees(np.unwrap(np.angle(h))).tolist(),
    },
}
with open("results.jsonl", "a", encoding="utf-8") as out:
    out.write(json.dumps(row) + "\n")
```

A stepped run stores one value per step. Read each with `step=k`, and keep the
step's parameters, `op.steps[k]`, beside it. A stepped `.ac` at one frequency
is stored the same way, one point per step with no frequency axis, so
`value(name, step=k)` reads it too.

## Analysis primitives

Import these from `ltspice_mcp.api`. They take arrays and return dicts.

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
- `ApiCallError`: the call failed. `.code` names the reason, and `.job_id` and
  the full payload keep every recovery handle.
- A per-item failure, or an `outcome` of `"partial"` or `"failed"`, does not
  raise. It is returned in the data, so read `failures`, `observations` and
  `warnings`.
- `ApiSessionError` (with `ApiClosedError`): a second `Api` in one process, a
  call after `close()`, or a call from a forked child.
- A refused path raises the engine's path error with a note on what to change.
  When the path is outside the sandbox, the note names the setting that
  defines the sandbox.

## Controls that exist only on the wire

`budget` and `execution.wait_s` are dropped, with a warning in the result, and
a cursor or continuation is rejected. With `raw_page=True`, a call accepts them
and returns exactly one page, as the tool would.

## More than one build of a simulator

Name each further executable, in `[simulator.executables]` of the config
(`xvii = "C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"`) or as
`Api(working_dir=".", simulator_executables={"xvii": "C:/.../XVIIx64.exe"})`,
and select it per run with `execution={"simulator": "ltspice:xvii"}`. The plain
family name still runs the default build. `inspect(kind="capabilities")` lists
the named builds under `named_executables`. Each job records the executable
that ran it.
