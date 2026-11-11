# Python API reference — `ltspice_mcp.api`

`ltspice_mcp.api` is the in-process Python interface to the code the MCP
server exposes.

The source is authoritative. Where this document disagrees with it, fix the
document.

---

## 1. Relation to the MCP contract

**The API has no contract of its own.** It exposes the six operations of the
MCP contract (`docs/design/mcp_surface.md`) as Python methods. It has the same
operations, argument shapes, validation (through the same Pydantic input
models), and the same meaning for `completeness` and `observations`. Only the
presentation differs. Calls are synchronous, returns are complete and
structured, call-level errors raise exceptions, and the `budget` argument,
which caps the size of an MCP reply, is dropped (§7).

## 2. Limits

- No MCP resources, and no `budget` reply-size cap (§7 describes how `budget`
  is dropped, and the single-page mode).
- No analysis the MCP tools do not offer.
- `ltspice_mcp.api` ships in the same package as the server.
- An API process and an MCP server process may share a working directory. They
  coordinate as separate server sessions do. A lock on each circuit file stops
  two processes from editing it at once. A process treats another's job as
  abandoned only after checking that the job's owning process has exited
  (§11, "If the owner dies"). A cancel kills only the simulator processes whose
  command line contains that job's id. **One process cannot run both** (§4).

## 3. Configuration

Constructor keyword arguments override environment variables, which override
the TOML file, which overrides the defaults. `Api(working_dir=...)` reads the
TOML **under that directory**, not the process CWD, and takes `[security]
allowed_paths` from that file. If that setting is unset, there is no sandbox.
Besides `working_dir` and `config_path`, the constructor accepts these
overrides: `simulator`, `enabled_simulators`, `simulator_exe`,
`simulator_executables`, `ltspice_ini`, `hidden_desktop`, `ngbehavior`,
`allowed_paths`, `max_parallel_sims`, `max_experiment_cases`,
`default_timeout`, `run_timeout`, `max_estimated_points`, `max_raw_mb`,
`max_points_returned`, `analysis_budget_s`, `result_set_ttl_hours`,
`symbol_paths`, `sync_open_window`, `start_ltspice`, `persist_jobs` and
`preload_recent_count`. Any other keyword raises `TypeError`, including the
server-only settings in `[logging]` and `[tools]`, and `tool_profile`.

The precedence holds for the life of the session. The session re-reads
`[security] allowed_paths` when the config file changes, so a sandbox can be
set, widened or removed without a restart. An explicit `Api(allowed_paths=...)`
pins the sandbox, and an empty list pins the session to no sandbox. A TOML
file written or edited later does not replace it, including the default config
that a server session in the same directory writes on its first tool call. In
that session, the note on a `PathSecurityError` (§6) tells the caller to open a
new `Api` with the directory added to `allowed_paths`, rather than to edit the
config file.

The API does not configure logging. It never calls `logging.basicConfig` and
writes only through module loggers.

## 4. The `Api` object

```python
from ltspice_mcp.api import Api

with Api(working_dir="~/designs/ldo") as api:
    receipt = api.run_experiments(...)
```

**One live session per process.** A second concurrent `Api`, or an
`Api` inside a process already running the MCP server, raises
`ApiSessionError`. Closing a session stops the jobs it runs and closes what
the whole process shares, such as the hidden desktop LTspice runs on, so one
session closing would stop work another depends on.

Each `Api` runs its session on a private event-loop thread. Calls from several
threads run concurrently there, so one thread can cancel a job while another is
blocked waiting on it.

**Fork guard.** After a `fork()`, a call on the inherited `Api` raises
`ApiSessionError`. The child must create a new `Api`.

**Loop-thread re-entry.** Every synchronous method, `close()` included, raises
`ApiSessionError` when called from the `Api`'s own loop thread, where it would
deadlock.

**`close()`** stops the jobs its session runs, as server shutdown does, and
each reads as it would after a crash: `interrupted` while runs were left
(`completed_with_failures` when only the attached analysis was), its
unfinished cases failed with code `server_restarted`, which
`jobs(action="resume")` retries by default, and an unfinished attached
analysis failed. Nobody cancelled them, so none reads `cancelled`.
Calls made after it starts raise `ApiClosedError`. It is idempotent, and a
second `close()` from another thread returns once the first has finished. A
call already running in another thread when `close()` starts is cancelled and
raises `ApiClosedError`, except a submission, `jobs(cancel)`, `jobs(resume)` or
`edit_schematic`, which `close()` lets finish first. After it returns, the
process can open a new `Api`.

**Live jobs an `Api`'s session runs do not survive `close()` or process exit.**
Persisted records and results survive. To keep work running after the
interpreter exits, submit it through a long-lived server process or a per-job
detached owner (`run_experiments(wait=False, detach=True)`, §11), which is a
separate process and so a separate owner.

The API states this rule in three places:

- `reference()` and `reference('run_experiments')` state that `wait=False`
  returns a receipt *and* that the submitting process must outlive the run
  unless `detach=True` hands the job to its own owner, because a job is
  interrupted when the `Api` that submitted it closes or its process exits.
- The receipt includes a `process_owned_job` observation.
- When the process running a job stops before a case finished or wrote its
  results, whether it exited, crashed or closed its `Api`, the case fails. Its
  `failures` row has code `server_restarted` and a message that says the
  process running the job stopped and must stay running until the job
  finishes.

## 5. The six operations

Each method takes the operation's arguments as keyword arguments, validates
them with the operation's own input model, and returns a `dict`. A validation
failure raises `ApiValidationError` (a `ValueError`) with the message the MCP
server returns for the same arguments: each error's field path and message,
the field table of each branch the error names, and, for a field this
operation does not accept, the operations that do. It is not a raw
`pydantic.ValidationError`, which renders differently.

Every other argument a method takes but cannot use raises `ApiValidationError`
too: both spellings of one argument (`wait`'s `timeout_s` and `timeout`, a
result read's `path` and its older keyword), a value `wait` cannot use, a
`wait` or `detach` control that is not a bool, and a value a detached call
cannot hand to its owner as given: one that is neither JSON, a path nor a
model, a dictionary key that is not a string (JSON would write a number's as
a string, a key the call never had), and a value that holds itself. A
keyword a method's signature does not name raises
Python's own `TypeError`, as for the constructor; the six operations take any
keyword and refuse one they do not accept with `ApiValidationError`. The
catalogue lookups `reference()` and `guide()` run no call and raise Python's
own errors: a `ValueError` for a name they do not have, and `reference()` a
`TypeError` for an operation name that is not a string.

```python
api.run_experiments(**args)   # two-phase; see below
api.jobs(**args)
api.analyze_results(**args)
api.inspect(**args)
api.edit_schematic(**args)
api.verify_circuit(**args)
```

**Fields left out.** A result leaves out the fields that say nothing (an
empty list or object, a null, a false flag), by the same rule and with the
same kept fields as the tool's reply (`docs/design/mcp_surface.md`, §2,
"Fields left out"), so a script and a tool call read one shape. Read a field
that may be empty with a default, such as `row.get("assignments", {})`.

**Complete results.** Where an MCP tool pages or caps a result, the method
returns all of it:

- `analyze_results` runs the evaluation to completion in passes of up to
  `[analysis] analysis_budget_s` each. Where the MCP tool would return partial
  results and a `result_set_id`, the method starts the next pass. If reading
  the sources and recording their content hashes and log diagnostics does not
  finish within the first pass, the call raises `AnalysisDeadlineExceeded` and
  creates no result set or cursor; retry the original request with fewer
  sources or a larger `analysis_budget_s`. `result["results"]` has an entry
  for every recipe, so `result["results"][key]` never raises for a recipe
  that failed: its rows are empty and a `No rows:` warning names the
  `failures` places that explain it.
- `verify_circuit` returns every finding for each rule. The MCP tool lists at
  most 25 findings of a rule and gives a rule with more a `findings_omitted`
  entry, whose cursor `findings_cursor` takes to list the next ones; the
  method takes `findings_cursor` only with `raw_page=True`.
- `edit_schematic` applies the batch once and returns its views whole. The
  views are built inside the edit from the sheet it wrote, whose hash is the
  returned `sha256`, so a later edit by another session cannot appear in them.
  `dry_run` returns full views the same way and writes nothing.
- `inspect` follows each query's cursor to the end. A query whose source
  changes during collection starts again once; a second change raises
  `ApiCallError`.
- `run_experiments`, `jobs(status|wait)` and `jobs(runs)` return one
  consistent snapshot of the job. The snapshot contains the rows keyed by
  `(case_id, run_index)`, status, completeness, `outcome`, `failures`,
  `observations`, `artifacts`, and the attached analysis's status, result,
  error and observations, all read at the same moment. `outcome` and `hint`
  are derived from that snapshot. Every row has the fields `run_fields`
  selects, or all fields when it is unset. The rows come in the same page
  object as on MCP, with `returned == total` and neither `truncated` nor
  `next_cursor`, since there is no further page.
- `jobs(list)` returns every circuit group. `jobs(cancel)` returns its kill
  receipts, which neither interface pages.

If assembling a complete result fails after the submission, `ApiCallError`
contains the original receipt and control token.

**Two-phase `run_experiments`.** The API always submits with
`execution.wait_s=0`; a `wait_s` the caller passes is dropped with a warning
(§7). A Ctrl-C during submission does not cancel it. The submission runs to
its end, so it either fails before anything was submitted or records the job,
and in the second case the call raises `ApiInterrupted` containing the
receipt. Without an interrupt, `wait=True` (the default) then blocks in
successive `jobs(wait)` calls:

- Ctrl-C cancels only the current wait. The job keeps running, and the API
  raises `ApiInterrupted` (a `KeyboardInterrupt`) **that contains the receipt
  and job id**, so the caller keeps the handle.
- `api.wait(job_id, timeout_s=None)` waits for a job. The older spelling
  `timeout=` is still accepted; giving both raises `ApiValidationError`. On
  timeout it returns the current `jobs(wait)` snapshot with `timed_out: true`
  and leaves the job running.
- `wait=False` returns the submission receipt immediately, with the
  `process_owned_job` observation.
- `wait=False, detach=True` is the only combination that does not submit
  in-process. A spawned per-job owner submits the job, so the job survives
  this process (§11).
- Leaving the `Api` context still stops the live jobs its session runs, which
  then read `interrupted` (§4).

**Recovery through `Api.jobs`.** Opt in at initial submission with
`execution={"recoverable": True}`. `api.jobs(action="resume", job_id=parent_id,
resume_request_id="retry-1", control_token=parent_token)` returns the complete
resume receipt. It includes the job's `lineage` (`root_job_id`,
`parent_job_id`, `attempt_index`), each run's `attempt` (which execution
produced it and whether it was `reused`), and `completeness.reused`, the count
of successful cases reused from earlier attempts. By default the resume
retries interrupted cases; `retry_failed=True` adds failed cases,
`retry_cancelled=True` adds cancelled ones, and `case_ids` limits the retry to
the cases it names. `wait_s` bounds how long the call waits and defaults to
zero. The receipt of a resume that created a child job contains the child's
control token, even once the child has finished. A resume with nothing to do
returns `resumed=false` and no child token; to retry only the analysis, call
`analyze_results` on its own.

A recoverable job saves the simulator's startup settings with the job, so
every attempt starts the simulator the same way. For ngspice, the server
disables user startup files and supplies a system startup file that sets
nothing; this works on Linux and native Windows. For LTspice, it works on
native Windows with one LTspice 26.0.2 executable, identified by its SHA-256
hash, whose startup behaviour was examined for this purpose. Other platforms
and other LTspice executables are refused with `recovery_startup_unsupported`
before submission. For LTspice, recovery copies a settings file LTspice has
already used, one that holds the identifier and usage-data choice LTspice
records on its first start. The file is `simulator.ltspice_ini` (environment
`LTSPICE_MCP_LTSPICE_INI`, API override `ltspice_ini`) or
`%APPDATA%/LTspice.ini`, and a relative `ltspice_ini` resolves from the
working directory. The original file is not changed. The job keeps a copy,
the template, and each attempt runs LTspice on its own writable copy of the
template.

The LTspice template must contain the time of the last update query, written
by LTspice itself within the preceding 15 days. This is checked again on
resume and before each launch. If the check fails (`recovery_startup_stale`),
resolve the update reminder in LTspice before submitting. Recovery never
changes the timestamp, chooses usage consent or answers dialogs. The build
restriction and the freshness check rely on how the update reminder behaved
when that build was inspected. They are not a permanent, vendor-supported way
to suppress update queries.

To make ngspice's static random functions, the ones it evaluates once when it
loads a deck, reproducible, pass
`execution={"simulator": "ngspice", "recoverable": True, "simulator_seed": 17}`
to `api.run_experiments`. The API keeps this field when it sets `wait_s` to
zero. The seed must be a strict integer in `1..2147483646`. ngspice is
reseeded with it (`setseed`) immediately before each case's deck is loaded, so
every case and retry starts from the same seed. The seed sets no Monte Carlo
run count or distribution, selects no PDK statistical models and changes no
startup files. It cannot be combined with a `pdk_native` variation
(`recovery_seed_unsupported`).

A seeded deck may use the static random functions `agauss`, `gauss`, `aunif`,
`unif` and `limit`, and must have exactly one `.op`, `.ac`, `.dc` or `.tran`
analysis. A `.noise` analysis, a `.step` or more than one analysis is refused
with `recovery_seed_analysis_unsupported` before the job is recorded, so the
same `request_id` can be submitted again once the deck is fixed. The deck is
also refused if it calls a function that the server does not know to be
deterministic and that the deck does not define with `.func`
(`recovery_expression_unsupported`), uses any other random function, such as
transient noise (`trnoise`, `trrandom`) (`recovery_random_unsupported`),
contains a `.control` block (`recovery_control_unsupported`), or loads an
external module with `.load`, `.osdi`, `.pre_osdi`, `.hdl`, `.verilog` or an
`A` device (`recovery_external_module`). Every attempt, resumes included, runs
in the ngspice compatibility mode (`[simulator] ngbehavior`) recorded at
submission. Statistical two-argument `limit` works in `hsa`, while the default
`kiltpsa` supplies a different, three-argument clamping function. Configure
the mode the deck needs separately; the seed does not override it.

Of the `jobs` actions, only resume accepts `api.jobs(detach=True, ...)`, and
not with `raw_page=True`. It starts a detached owner as `run_experiments` does
(§11). Before starting the owner, the API re-reads the parent's job record
from disk and checks that the caller may resume it; a refused caller gets
`resume_not_authorized`. A caller without a token must be the session that
runs the parent, which that session's job registry answers, as it does for a
cancel; the PID and creation time the parent's record names are never
compared with the calling process's, nor is a PID given in the request. Once
the check passes, the API passes the parent token privately to the new owner,
which checks the resume again with that token; being started by the caller
gives the owner no authority of its own. Repeating a resume with the same
`resume_request_id` returns the same result even when the control token,
`detach` or `wait_s` differ, because only the parent, `case_ids` and the retry
flags decide whether it is the same request. Save root and child tokens from
submission and resume receipts, because `status`, `list` and `runs` are
read-only and do not return them.

This example sweeps a divider over corner, temperature and supply, cancels the
first attempt, resumes the cancelled cases and checks every operating point
against its analytic value. Caller code maps cases to conditions and computes
expected values; the server keeps each case's `case_id` across the resume and
reruns a resumed case from the deck and include files it saved at submission.
The library sections are synthetic. They are not measured process data
or foundry models.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from ltspice_mcp.api import Api

with TemporaryDirectory() as directory:
    folder = Path(directory)
    (folder / "corners.lib").write_text(
        ".lib low\n.param rbase=800\n.endl low\n"
        ".lib high\n.param rbase=1200\n.endl high\n",
        encoding="utf-8",
    )
    circuits, conditions = [], {}
    for corner, rbase in (("low", 800), ("high", 1200)):
        for temperature in (-20, 27, 77):
            label = f"{corner}_{temperature}"
            deck = folder / f"{label}.cir"
            deck.write_text(
                f'* divider\n.lib "corners.lib" {corner}\n'
                f".param supply=1\n.temp {temperature}\n"
                "V1 in 0 {supply}\nR1 in out {rbase} tc1=0.01\n"
                "R2 out 0 1000\n.op\n.end\n",
                encoding="utf-8",
            )
            circuits.append({"path": str(deck), "id": label})
            conditions[label] = (rbase, temperature)
    with Api(working_dir=folder, allowed_paths=[folder],
             simulator="ngspice", ngbehavior="hsa", max_parallel_sims=1) as api:
        root = api.run_experiments(
            wait=False, request_id="divider-campaign", circuits=circuits,
            variations=[{"kind": "assign", "assign": {"supply": [0.9, 1.1]}}],
            execution={"recoverable": True},
        )
        api.jobs(action="cancel", job_id=root["job_id"])
        api.wait(root["job_id"])
        resumed = api.jobs(
            action="resume", job_id=root["job_id"], resume_request_id="retry-1",
            control_token=root["control_token"], retry_cancelled=True,
        )
        api.wait(resumed["job_id"])
        result = api.analyze_results(
            sources=[{"label": "pvt", "job_id": resumed["job_id"]}],
            recipes=[{"key": "out", "metric": "value", "expr": "V(out)"}],
            include={"per_run": True},
        )
        rows = result["results"]["out"]["per_run"]["items"]
        assert len(rows) == 12 and not result["failures"]
        for row in rows:
            rbase, temperature = conditions[row["circuit"]]
            resistance = rbase * (1 + 0.01 * (temperature - 27))
            expected = row["assignments"]["supply"] * 1000 / (1000 + resistance)
            assert abs(row["value"]["value"] - expected) < 1e-10
```

**`Api.reference(op=None) -> str`** returns the argument catalogue.
`reference()` returns the index of the six operations.
`reference('edit_schematic')` returns that operation's resolved argument tree.
The tree writes out every field with its type, default, enum members and union
branches, flattens nested models onto dotted paths, and includes one worked
example. It contains no JSON Schema syntax or `$ref` chains. It is a
**staticmethod** and needs no session, so it works before any `Api` is
opened. It renders from the Pydantic models the call validates against, so it
always matches them.

**`Api.guide(section=None) -> str`** returns the guide the MCP server serves.
With no argument it returns the core, which a session reads first, and the
core's index of topic sections and task playbooks. Given a name from that
index, it returns that part. It is a staticmethod for the same reason as
`reference()`, and returns the same text as `inspect(kind: "guide")` and the
`spice://guide` resources. Its `python` section covers this interface.

`python -m ltspice_mcp.api reference [OP]` prints the same catalogue without
starting a session, and `python -m ltspice_mcp.api guide [SECTION]` prints the
guide.

Each of the six methods has its `reference()` text as its `__doc__`, so
`help(api.edit_schematic)` and `inspect.getdoc` show it.

**Relative path arguments resolve from `working_dir`**, not the process CWD,
as does any relative entry in `allowed_paths` (such as a config file's
`allowed_paths = ["."]`). The MCP server resolves a relative path from the
first allowed directory when there is a sandbox, and from its working
directory otherwise; its `spice://netlists/` resource lists that same folder.

## 6. Errors

- When an operation raises one of the server's own exceptions (subclasses of
  `LTSpiceMCPError` in `ltspice_mcp.errors`, such as `PathSecurityError` or
  `AnalysisDeadlineExceeded`), the exception propagates unchanged; the MCP
  server would return it as an `isError` result instead. A
  `PathSecurityError` gets one note (PEP 678) with the same guidance as a tool
  call's `hint`, so the traceback shows what to change (with a sandbox, the
  setting that defines it); its type and message are not changed. The note
  names `working_dir` as the folder a relative path is taken from, since that
  is where the API takes it from.
- A result with `isError=True` raises `ApiCallError`, which contains the
  complete structured payload and has the attributes `code`, `commit_state`,
  `job_id` and `control_token`. When the error comes after a job was submitted
  or a schematic edit was written, the payload still contains what identifies
  the result, such as the receipt's `job_id` and `control_token`.
- Per-item failures, and `outcome="partial"|"failed"` envelopes with
  `isError=False`, are **returned data** and raise no exception, as on the MCP
  interface.
- An error code is never turned into a typed exception; read it from
  `ApiCallError.code`.
- A result with no structured payload raises `ApiInternalError`; a method never
  returns `None`.

`ApiError` is the base of the exception hierarchy. `ApiCallError`,
`ApiSessionError` (with `ApiClosedError` beneath it) and `ApiInternalError`
derive from it.
`ApiValidationError` derives from `ValueError` and `ApiInterrupted` from
`KeyboardInterrupt`, so ordinary `except ValueError` / Ctrl-C handling still
works.

## 7. Policy for wire-only controls

Default (automatic) mode **drops** the two presentation controls, `budget`
and `execution.wait_s`, and lists each dropped field in the result's
`warnings`. Neither is part of a request's idempotency fingerprint, so an MCP
call replayed through the API with them attached is the same request. It
**rejects** any of the operation's own fields, at any depth, named `cursor`,
`continuation`, `continue` or `view_cursors` or ending in `_cursor` or
`_cursors`, with an `ApiValidationError` that names `raw_page=True` as the fix.
A key that is the caller's own data, such as a parameter named `budget` or
`bias_cursor` in an `assign` variation, is not a control and passes as
written. It never changes
request fields that are part of the fingerprint. It passes `per_run`,
`outliers` and `signals_available` exactly as the caller wrote them, so
replaying an MCP request through the API cannot cause an idempotency conflict.
Presentation-only fields (`include.fields`, `run_fields`, `provenance`) pass
through.

`raw_page=True` accepts every wire control as given and returns exactly one
page, the one the MCP tool would return for the same arguments. It is the
preview mode, and the only way to get paging identical to the MCP tools'.

## 8. Result primitives

```python
raw = api.load_raw(path, plot_index=0, dialect=None)  # XOR: path | job_id
raw = api.load_raw(job_id=..., run_index=0, case_id=None, other_raw=None)  # "fra_1" | "op"
raw = api.load_raw(path, signals=["V(out)", "V(a,b)"])  # only these, the axis and step parameters
raw = api.load_raw(path, window=("1m", "2m"))  # only the points of the axis from 1 ms to 2 ms
raw.signals                    # list[str]
raw.trace("V(out)", step=0)    # np.ndarray — complex values preserved
raw.value("V(out)", step=0)    # one point a step (.op): float, or complex
raw.axis(step=0)               # real sampled axis; refuses a native table
raw.step_count; raw.steps      # counted as the tools count them; refused when steps are not told apart
raw.plots; raw.descriptor; raw.plot_index  # inventory and selected plot facts
raw.table(step=0)              # native table rows; complex values use real/imag
raw.compare(other, signals=None, step=0, other_step=None)  # per-signal differences
raw.analysis_type; raw.dialect; raw.source          # provenance
raw.observations               # [{code, kind, detail, evidence}]; partial_raw, coarse_compressed_waveform
api.measurements(job_id=..., run_index=0, case_id=None)
api.measurements("result.log")  # XOR: path | job_id; RAW is optional
api.device_op_points(job_id=..., case_id=None)  # {"m1": {"gm": ...}}; same sources
ops.observations               # a log reader's dict (LogResult) says the same of its log
```

- `RawResult` is this project's type; the API returns no spicelib types.
- `signals` reads only the traces those names read, by the rule `trace` reads
  a name with: one trace, or a node pair's two node voltages. The plot's
  first variable (its axis, where it has one) and any step parameters are
  always kept. The read's numeric limit (512 MiB at the widths the file
  stores) applies to the traces kept, so a file too large to read whole can
  be read for a few signals. A file stored a point at a time, as LTspice
  stores a run, is still decoded whole while it is read, and a read holds
  that to 1 GiB of numbers; a file in LTspice's Fast Access layout (`LTspice
  -FastAccess <file>.raw` converts one) is decoded only for the traces kept.
  A name is refused before the read for the reason `trace` would refuse it
  (no such trace, a step value, an expression, a pair naming ground twice, a
  node pair of a noise result), with the plot's traces; the result's
  `observations` include `signals_chosen`, and `trace` of a trace not read
  says how to read it. A whole-file read over the limit is refused with how
  many bytes of numbers it counted (at least that many, in a file of several
  plots), the limit and these ways around it.
- `window=(start, end)` reads only the points of the plot's axis from `start`
  to `end`, in each step, by the rule the exports' windows keep points with:
  every point at or after the start and at or before the end, with nothing
  interpolated at a bound. The bounds are in the axis's unit (seconds of deck
  time, hertz, the swept value), each a number or SPICE text such as `"1m"`,
  and either may be None for an open end. The numeric limit applies to the
  points kept, so a file with too many points for one read can be read a
  stretch at a time. Decoding is held to 1 GiB as above, and a window does
  not lessen it: a Fast Access file's kept traces are decoded along their
  whole axis before the window cuts them, so a window alone of a file over
  that is refused, naming `signals`. It combines with `signals`. A plot with
  no axis (an operating point), an axis that decreases within a step (a
  downward sweep, or a stepped run whose steps were not told apart), and a
  window that meets no point of a step are refused. The result's
  `observations` include `window_chosen`; on a compressed LTspice transient
  it says that whether compression left the run coarse, which is judged over
  all of a run's points, was not judged from the window.
- Each of these reads takes exactly one of `path` and `job_id`. Neither or
  both, an empty `path`, or `run_index`, `case_id` or `other_raw` with a
  `path` raises `ApiValidationError`, as the operations do.
- `observations` describes the result as it was read. It contains
  `partial_raw` for a stopped case's raw kept with
  `execution.keep_partial_raw`, whose values end where the run was stopped,
  and `coarse_compressed_waveform` for an LTspice transient that waveform
  compression left with few points, as `analyze_results` reports it. It is
  empty for a whole result. `measurements` and `device_op_points` return a
  `LogResult`, a plain `dict` that has the same `observations` for the log it
  read.
- `step_count` counts a plot's steps as the tools count them, 1 for a run that
  is not stepped, and `steps` holds one parameter dict a step. A plot whose
  steps cannot be told apart is loaded, so its `signals`, `descriptor` (with
  its `step_status`) and `plots` can be read, but `step_count`, `steps` and
  every read of a step refuse it with the reason `analyze_results` gives:
  the operating-point raw LTspice 26 writes beside a stepped `.tran` or
  `.ac` holds one point and does not say which step it is, and a stepped
  sweep read without the log that holds its step values has none to split
  it by.
- `value` reads a plot with one point per step, such as an operating point, as
  a number. It returns a float, or a complex number for a complex trace. A
  trace with more points is refused rather than cut to its first point; read
  it with `trace`.
- `compare` reports how far `other`'s traces are from this result's and
  returns `{signals: {name: {max_abs, max_rel, at, value, other}}, only_here,
  only_there, span, points}`. A signal named in `signals` is looked up in each
  result the way `trace` looks it up, a node pair `V(a,b)` included, and is
  reported under the name `trace` resolves it to; one either result lacks is
  refused with the reason. By default it compares every signal both
  results hold except the axis. It pairs signals by name, ignoring case; a
  node or device inside subcircuits is matched by its path, however each
  build spells it (LTspice 26's `V(xa:xin:mid)` is XVII's `V(a:in:mid)`).
  `other_step` defaults to the step of `other` whose parameters match this
  step's, each value matched as a step selector matches it
  (`step_values.step_with`; step 0 of a result that is not stepped); a step
  with no match is refused with the steps `other` has. On a sampled axis,
  each signal is compared at this result's points within the span both axes
  cover. When `other`'s axis differs, `other` is
  read at those points by linear interpolation, which on a frequency axis is
  in log frequency, dB and unwrapped phase, as `log_interp_complex` does it. A
  plot with no axis is compared point by point and must have the same number
  of points. `max_rel` is `max_abs` divided by the signal's largest magnitude
  in this result, and is null when that magnitude is 0. Complex values are
  compared as complex numbers. No tolerance is applied. A result read with
  `signals` holds no other trace, so when `other` was, the traces only this
  result holds are `unread_there` and `only_here` is empty, and when this
  result was, the other's are `unread_here` and `only_there` is empty.
- `device_op_points` reads, by device and parameter, the semiconductor
  operating points that LTspice prints in the log of an `.op` run. LTspice 26
  prints them under `.options logopinfo`; XVII always prints them and refuses
  that option. The table has three significant digits. A node voltage or a
  device current read from the raw with `value` has the raw's own precision
  (single, about seven digits, in the operating-point recordings of both
  builds). A device in a subcircuit is keyed by its path (`x1.m1`), where the
  table names it `x1:m1` on LTspice 26 and `m:1:1` on XVII (the device's
  letter, the instance's name without its X, and the device's name without
  its letter). A value the log did not print as a finite number is `None`.
  The result is empty when the log has no table. An ngspice log never has
  one, because ngspice writes the same parameters as traces of its raw; read
  them with `raw.value("@m1[gm]")`.
- `plot_index` is a strict nonnegative integer, independent of case/run and
  step selection. `dialect` is `None` or `ltspice`, `ngspice`, `qspice`,
  `xyce`; an explicit dialect must agree with the producing simulator and the
  captured header. The selected plot's descriptor gives the analysis type,
  axis, trace units and step status. Only a plot over time, frequency or a DC
  sweep variable has a sampled axis, its first variable. In any other plot,
  such as an operating point or transfer function (a native table), the first
  variable is an ordinary quantity that `table()` returns with the others, and
  `axis()` refuses the plot rather than inventing an axis. Inventory and
  descriptor metadata are detached dictionaries; table values are the numbers
  recorded, with no engineering formulas applied.
- **Arrays are detached copies.** Changing a returned array does not change
  what a later call returns, through either interface.
- A job id resolves to one experiment case; a caller-supplied path goes
  through the same checks as any other path argument. Both read through the
  same parser and result cache as the tools, so a primitive and a tool read a
  file the same way. Step metadata (`raw.steps`) comes from the files read
  with the result, the `.raw` and the `.log` beside it; no other log is opened
  to fill it in. Each step names its values as the deck's `.step` does, a
  stepped temperature `temp`, whichever of the two files holds them.
- RAW decoding runs in a separate parser process with a deadline of 120
  seconds (`AnalysisDeadlineExceeded` when it runs out) and a memory limit of
  2 GiB; the returned arrays are fully in memory. `measurements` returns
  detached copies of the `.meas` values in the log; it does not read values
  from a RAW table plot.

For a log with no RAW, use `inspect` and `analyze_results`:

```python
facts = api.inspect(queries=[{
    "kind": "results", "view": "native_tables", "path": "result.log",
}])
measured = api.analyze_results(
    sources=[{"path": "result.log", "label": "imported"}],
    recipes=[{"metric": "measurements", "key": "meas"}],
)
```

The `measurements` inspect view pages measured values with their recorded
range and `AT` metadata, and marks a value LTspice printed as failed
`failed: True`. `native_tables` pages literal scalar entries or
printed frequency rows, keeping section boundaries, printed labels and
real/imaginary components. A unit the log does not establish is left out, and a
block's `analysis_extent`, whether its rows cover the whole analysis, is
`"unknown"` unless the log establishes it. Rows carry no plot or step index. A
`measurements` row's `value_index` counts the values printed for that `.meas`
and is not a RAW step index. Rows and metadata are detached plain values,
which caller code can turn into arrays or export without another result
class.

The `measurements` recipe's rows are the MCP rows, rendered by the same code.
For a run with one value per `.meas`, such as one case of a sweep, the value
is `row["value"]["measured"][name]`. It is null where the log holds no number
for that name, and the name is then in one of three places, by why.
`row["value"]["failed_measurements"]` lists it where LTspice printed it as
failed at the step the row reads (`FAIL'ed`, or `failed` in a stepped
table), which is what `failed_measurements` means wherever it appears,
`measurements()` included. Where the log printed something other
than a number for it (a WHEN threshold written as a parameter, an
expression or, on LTspice 26, a braced expression),
`row["value"]["unreadable"][name]` gives the log line and the
reason, as `measurements()` does on the entry; only that name fails. Where it
printed a number that is not finite (a gain margin of `infdB`),
`row["value"]["not_finite"]` lists it. Any crossing or `AT` point the log
printed is under `row["value"]["at"][name]`. The per-name `stats` block is left
off the default row; `include={"fields": ["value"]}` returns it. A stepped
run's row contains `stats`, with `failed_measurements` and `unreadable` over
all its steps; `measurements()` gives an entry whose stepped table printed
`failed` the steps it did, from 0, in `failed_steps`. Reductions read `stats` either way, so
`reduced` does not depend on how a row was rendered. A name the log holds no
number for is counted rather than skipped. Its reduced rows include
`no_value_count`, and a spec on it reports the same count beside `pass_count`
and `fail_count` and is `"indeterminate"` unless `allow_incomplete` is set.
The API returns the same dict as the MCP result.

A source whose `path` is a `.log` reads only the log; a `.raw` beside it is
not read, even when one exists. Log views take no `plot_index`. In analysis,
including attached analysis, an omitted plot selector selects plot zero; this
applies only to RAW recipes. Whole-log measurements give statistics over every
step, or with `step` or `all_steps` each step's row with its `step_values`.
The six operations take exactly what their tools take, so an argument a tool
renamed is refused there with the name that replaced it. The helpers keep an
older keyword they had, as `wait` keeps `timeout`: `load_raw` takes
`raw_path`, and `measurements` and `device_op_points` take `log_path`, in
place of `path`; giving both raises `ApiValidationError`. A malformed or
missing RAW fails only the RAW recipes. A result set's manifest records, for
each source,
whether its RAW, log and console files existed and a hash of each one's
content, so an empty file is distinguishable from a missing one. A new call
reads a changed source again. A continuation (`continue` with a
`result_set_id`) refuses a source whose RAW, log or console file changed or
appeared since the result set recorded it (the `analyze_results` section of
`docs/design/mcp_surface.md` lists the changes that a check of file metadata
cannot detect). Under `include.provenance`, each source reports
`explicit_dialect`, the dialect the caller passed, separately from
`producing_dialect`, the simulator the records show produced the files.
In a whole-log row, the dialect of the producing simulator can be absent.
A log alone does not show that a simulator solve completed or that a case was
produced; reads by job still check that the job is terminal and the case was
produced.

`ltspice_mcp.api` exports the AC and transient metric functions listed below.
They take arrays and return dict- or TypedDict-shaped mappings; AC
functions take complex `H`, and axes are real.

- AC: `prepare_ac_arrays`, `unwrap_phase_safe`, `log_interp`,
  `log_interp_complex`, `detect_crossings`, `find_crossings_any_quantity`,
  `gain_at_frequencies`, `compute_filter_metrics`, `compute_stability_metrics`,
  `compute_roll_off`, `compute_resonances`, `compute_return_loss`,
  `integrate_noise`, `classify_filter`, `analyze_ac_structure`.
- Transient: `window_and_clean`, `analyze_edge`, `analyze_pulse_response`,
  `analyze_disturbance_response`, `analyze_timing_between`, `analyze_periodic`,
  `analyze_thd`, `analyze_tone`, `compute_signal_stats`,
  `time_weighted_quantiles`, `compute_measurement_stats`.
- `parse_spice_value` parses a SPICE literal such as `'5p'`. Variation values
  are passed and returned as SPICE literals, and no other function in
  `ltspice_mcp.api` parses one.

**Typing policy.** Every *project-defined* alias, TypedDict or nested output
type in a public annotation is importable from `ltspice_mcp.api`.
Standard-library and dependency types (`numpy.ndarray`, `Sequence`, ...) are
exempt. A type's defining module is not part of the contract, so its
`__module__`, repr, documentation links and pickling behavior may change. Only
the import from `ltspice_mcp.api` is stable. `typing.get_type_hints` resolves
on every public function.

## 9. `__all__` — the stability boundary

The names in `__all__` are the package's stable interface. Adding a name is a
minor change; removing or renaming one is a major change. Dict return shapes
are the shapes the MCP tools return, so they change only when the tools'
outputs change and are versioned with them. The docstrings of the §8
functions fix their signatures, units and array conventions.

`__all__` contains `Api`, `RawResult`, the exception types, the §8 functions,
and these typing names:

- aliases: `Quantity`, `SearchDirection`, `CrossingDirection` (defined
  identically in both metric modules and re-exported once), `FilterType`,
  `StabilityLabel`, `CornerKind`;
- outputs and their nested types: `CrossingWithQuantity`, `GainAtPoint`,
  `ReturnLossOutput`, `FilterMetricsOutput`, `StabilityMetricsOutput`
  (`Crossover`, `PhaseMargin`, `GainMargin`), `RollOffOutput`,
  `ResonancesOutput` (`ResonancePeak`), `NoiseIntegralOutput`,
  `EdgeMetricsOutput`, `PulseResponseOutput`, `DisturbanceResponseOutput`,
  `TimingBetweenOutput`, `PeriodicMetricsOutput`, `SignalStatsOutput`,
  `TimeWeightedQuantilesOutput`, `ThdOutput` (`HarmonicEntry`), `ToneOutput`,
  `MeasurementStatsEntry`, `HistogramBin`, `AcStructureResult`, `Corner`,
  `Observation`.

## 10. `ltspice_mcp.api.types` — the argument models

A separate module, **`ltspice_mcp.api.types`**, re-exports the *argument* models
the six operations validate against. These are the render and compare policies
(`CompareSpec`, which `edit_schematic` takes, `VerifyRenderPolicy` and
`VerifyCompareSpec`, which `verify_circuit` takes, and `RenderPolicy`, the base
the verify render policy subclasses), the recipe union and its members, the
inspect query kinds, the variation rules, and public aliases for the schematic
op models (`AddComponentOp`, `WirePinsOp`, ...). Validation errors name these
types; import them from this module. It is not part of `__all__` (§9).

## 11. Detached owners

`run_experiments(wait=False, detach=True)` submits an experiment that outlives
the calling process. `detach` defaults to `False`, so a call without it submits
in this process, which owns the job (§4).

**Only with `wait=False`.** `detach=True, wait=True` raises an
`ApiValidationError` that names two fixes, dropping `detach` or passing
`wait=False` and waiting later with `api.wait(job_id)`. `detach=True` with
`raw_page=True` is also refused, because `raw_page` returns the reply to a
submission made in this process, and a detached submission is made in
another process.

**Submission.** The calling process validates the arguments against
`RunExperimentsInput`, so a malformed request raises in the caller's own
traceback before any process starts. It then starts

```
sys.executable -m ltspice_mcp.detached_owner <request-file>
```

in a new session (on POSIX), with its stdin closed and its stdout and stderr
appended to a log file. The owner runs on the same interpreter and install as
the caller. It opens an ordinary `Api` on the same working directory, config
file and constructor overrides, submits with `run_experiments(wait=False)`,
sends the receipt back, waits until the job is terminal, and exits. The caller
returns that receipt. The owner stages and submits the job. Until its
submission writes the job record, a lookup by `request_id` finds no job; after
that, the record names the owner's pid.

On Windows, each owner holds a Job Object containing itself and its simulator
descendants. Windows closes that handle when the owner exits, including a
forced exit, and terminates the remaining descendants. An owner started from
`run_code` leaves the worker's Job Object, so resetting the worker does not
cancel a detached experiment. The owner requests breakaway only when the
enclosing job permits it, so a job imposed from outside can still end the
owner. In a virtual environment on Windows the owner runs on the underlying
interpreter with `__PYVENV_LAUNCHER__` set, as CPython's multiprocessing does,
so the environment is kept without the launcher's extra process.

**If the owner dies.** If it exits before writing the job record, the caller's
wait for the receipt ends when the exit is seen and raises `ApiCallError`
naming the log file. If it dies after writing the record, including a kill
mid-run, the record names a pid that no longer exists. The next process that
reads the job checks that the pid has exited and then marks the job
`interrupted`. It marks each unfinished case whose `.raw` and `.log` are on
disk `produced`, and fails the other unfinished cases with code
`server_restarted`, as it does after a server crash.

**If the caller stops waiting.** The caller waits up to 360 seconds for the
receipt, plus `wait_s` on a resume. That is the 300 seconds the owner may
itself wait for another submission of the same `request_id` to finish, plus
60 seconds for the owner to start. After that, the owner and the processes it
started are stopped and the call raises `ApiCallError` with code
`detached_owner_timeout`. A Ctrl-C during the wait raises
`KeyboardInterrupt` and leaves the owner running. On POSIX the owner is in its
own session, so the signal does not reach it. In both cases, calling again
with the same `request_id` replays whatever the owner submitted.

**Ownership.** No session of the calling process runs a detached job.
`close()`, leaving a `with` block and interpreter exit stop only the jobs the
`Api`'s session runs, so they do not stop a detached job. Read it as you would
any job another process owns, with `jobs(action="status"|"wait"|"runs")` by
`job_id` or `request_id`, from this process, a later one, or a running MCP
server. Cancel it the same way. `jobs(action="cancel")` with the receipt's
`control_token` writes a cancellation file for the job in the store
(`experiments/cancellations/<job_id>.json`). The owner checks for that file
while the job runs, and on finding it kills only the simulator processes
started for this job's cases (§2).

**Replay.** A detached call whose `request_id` was already submitted starts an
owner that treats it as any repeated `request_id`. With the same payload the
owner returns the existing job's receipt and submits nothing; with a different
payload it fails with `idempotency_conflict`. The receipt's owner pid is read
from the record, so it names the process that owns the job, which may not be
the owner just started.

**The receipt.** The caller returns the owner's receipt, replacing its
`process_owned_job` observation (correct in the owner but wrong in the caller)
with:

```
code:     "detached_owner"
kind:     "lifecycle"
detail:   names the owning pid, says this Api's session does not run the job,
          so closing the Api does not stop it, and gives the log path
evidence: {"owner_pid": <int>, "log_file": "<path>"}
```

For a replay of a job that is already terminal, the detail states instead that
the pid is the process the record names as having run the job, that no process
supervises it now, and that there is nothing to cancel.

**Requires persisted jobs.** With `[state] persist_jobs` off, a session keeps
its jobs in memory, so the caller could not see the owner's job. `detach=True`
then raises `ApiValidationError`, on `run_experiments` and on a resume.

**Limitations.** `max_parallel_sims` is a per-process cap, so each detached
owner has its own cap, and N detached jobs can run up to N times the cap in
total, as N separate sessions can. On Windows the owner starts in the caller's
console process group, so a Ctrl-C in that console also reaches it.
Ownership, records and cancellation work the same on both platforms.

## Coexistence with a server

The API and an MCP server on the same working directory share the job records
and the store, and each process holds one session (§4). A job submitted
through the API with `run_experiments(wait=False)` is run by the submitting
`Api`'s session and ends `interrupted` when that `Api` closes or its process
exits; adding `detach=True` hands it
to a detached owner started for that job, which outlives the script (§11).
Either way, the other interface reads the same record by `job_id`.

One process can run several builds of a simulator family. Each build other
than the family's default is a named executable, configured in
`[simulator.executables]` or
`Api(working_dir=..., simulator_executables={"xvii": ".../XVIIx64.exe"})`, and
a call selects it with `execution={"simulator": "ltspice:xvii"}`
(`docs/design/mcp_surface.md`, "Selecting a build"). A second process on the
same working directory with its own `simulator_exe` also works; use it when
the two builds must not share a process. Either way, each job records the
executable its cases launched (`simulator_executable`) and each run records
the build it reported (`simulator_version`), so the shared records show which
build produced each result. A `request_id` reused from the other process, or
under another simulator name, replays only when it would launch the same build;
otherwise the call fails with `idempotency_conflict`
(`docs/design/mcp_surface.md`, "Replay is scoped to the simulator build").

The store is `<working_dir>/.ltspice-mcp/` unless `LTSPICE_MCP_STORE_DIR` is
set. With it set, the store is one directory per working directory under the
named directory, so records are still scoped by working directory, but a script
sees the server's jobs (and replays its `request_id`s) only if it has the same
setting. An `Api` started without it opens `<working_dir>/.ltspice-mcp/` and
sees none of them. A detached owner and a `run_code` worker inherit the setting
from the process that starts them.
