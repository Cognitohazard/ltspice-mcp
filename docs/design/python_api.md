# The Python API contract — `ltspice_mcp.api`

The in-process interface onto the same engine the MCP server exposes. This is
its contract: what it promises, what it does not do, and why.

The code is the authority. Where this document and the source disagree, the
source wins and this document is the thing to fix.

---

## 1. Goal, and the one design move that follows from it

Give a Python-native caller — an agent writing code in its own interpreter — the
same engine the MCP server exposes, with its loops and intermediate data
outside model context.

**The API does not get a new contract.** It binds the consolidated six-operation
MCP contract (`docs/design/mcp_surface.md`) to Python: the same ops language,
the same argument shapes, the same validation (literally the same Pydantic
input models), the same completeness and observation semantics. What differs is
presentation — synchronous calls, complete structured returns, exceptions for
call-level errors, no response-budget negotiation.

One evaluator per capability, reached through two interfaces (MCP and Python).
Anything that forks semantics between them is a defect, not a feature.

## 2. Non-goals

- No MCP resources, and no response-budget negotiation surface (see §7 for the
  single-page mode).
- No new analysis capability.
- No physical package split: `ltspice_mcp.api` ships in the existing package.
- No server-side arbitrary code execution.
- No cross-process ownership changes. The existing filesystem mechanisms —
  circuit-file locks, owner-pid liveness, token-scoped kill — are untouched, so
  a separate-process API and MCP server sharing a working directory is
  supported. **Same-process coexistence is not** (see §4).

## 3. Shared engine bootstrap

`server_lifespan` and `Api.__init__` call one bootstrap function
(`engine.bootstrap_server_engine` / `engine.bootstrap_library_engine`) that
owns, in order: config load, simulator detection, `SessionState.create`,
`_configure_asc_editor` symbol-path setup, result-set cleanup, and persisted
job preload. The last three used to live only in server startup; an `Api` that
skipped them would silently lose schematic support and job recovery.

Config precedence is explicit: constructor keyword arguments beat environment,
which beats TOML, which beats defaults. `Api(working_dir=...)` resolves the
TOML **under that directory**, not the process CWD, and the `allowed_paths`
defaults follow it. An unknown or irrelevant constructor override raises
`TypeError` rather than being silently ignored (a wire-only or server-only name
such as `tool_profile` is rejected here rather than quietly accepted).

The precedence holds for the life of the session. The engine re-reads
`[security] allowed_paths` when the config file changes, so an agent can widen
the sandbox without a restart, but an explicit `Api(allowed_paths=...)` pins
the sandbox: a TOML written or edited later (a server session in the same
directory writes its default config on its first tool call) does not replace
it, and a refusal in that session names the argument rather than the file.

Library mode never calls `logging.basicConfig`. The server's `force=True`
logging setup stays in server startup; the library uses module loggers only.

## 4. The `Api` object

```python
from ltspice_mcp.api import Api

with Api(working_dir="~/designs/ldo") as api:
    receipt = api.run_experiments(...)
```

**One live engine session per process**, enforced by an atomic PID-scoped
lease. A second concurrent `Api` — or an `Api` inside a process already running
the MCP server — raises `ApiSessionError`, because shutdown ownership is
owner-pid scoped and two same-PID sessions would cancel each other's jobs.
Lease semantics:

- acquired under a process-local lock, so two racing constructor threads cannot
  both pass;
- a lease recorded by *another* PID is inherited and stale (a post-`fork()`
  child) and is replaced without touching the parent's resources;
- the lease lock itself is fork-safe: an `os.register_at_fork(after_in_child=)`
  hook resets the lock in the child, because a lock held by a parent thread at
  fork time is inherited *locked* by a child that has no thread to release it,
  while the lease record is retained so the stale-PID replacement rule can run;
- released only when instance identity and PID both match;
- released on bootstrap *failure* as well as on a successful `close()`;
- the server lifespan acquires and releases the same lease.

**Private event loop thread.** One persistent loop thread per `Api`; handler
coroutines are marshalled onto it with `run_coroutine_threadsafe`. They run
concurrently on that loop rather than being serialized to completion —
`jobs(cancel)` has to run while another caller thread is blocked in a wait.
This is load-bearing: a per-call loop would invalidate the entire runner cache
on every call (loop invalidation lives in `RunnerManager._get_or_create`) and
split the `max_parallel_sims` semaphore. Editors are touched only on that loop,
so the `tools/_base.py` contract holds unchanged.

**Fork guard.** `Api` records its creator PID and re-checks on every call,
raising `ApiSessionError` after a `fork()` — an inherited loop thread is dead in
the child. The child creates a fresh `Api`, which the lease's stale-PID rule
lets succeed.

**Loop-thread re-entry.** Every synchronous method, `close()` included, detects
being called *from* the private loop thread and raises `ApiSessionError`
immediately. Blocking on `run_coroutine_threadsafe(...).result()` from that
thread deadlocks, and `close()` cannot join its own thread.

**`close()` state machine.** Task ownership is split, and draining "everything
state-touching" would deadlock behind a running simulation, because
runner- and registry-owned tasks live until `cancel_running()` asks them to
stop. So:

1. Mark closing; new calls raise `ApiClosedError`.
2. Cancel cancelable **bridge invocation tasks** — waits, pagination and
   collector loops, reads. Never cancel effectful handler tasks.
3. Drain **bridge invocation tasks only**: handlers, evaluators, reads, waits
   and collectors, with effectful ones run to their safe return boundary and
   cancelled ones awaited to settled cancellation. Do *not* drain
   runner- or registry-owned background tasks (submission pipelines, experiment
   coordinators, case tasks, deadline watchers, attached-analysis tasks) —
   those are `state.shutdown()`'s to cancel.
4. Run `state.shutdown()` on the private loop: cache clears, `cancel_running`,
   `drain_pending`, in that existing order. Step 3 guarantees no bridge task is
   still using the caches it clears.
5. Bounded-drain the residual loop tasks shutdown left behind. Shared artifact
   loaders settle their contained parser processes before releasing ownership,
   and `state.shutdown()` closes the session's kept parser tree; unconfirmed
   cleanup retains parser admission and scratch rather than claiming the
   worker exited. Then `shutdown_asyncgens`, stop the loop, close it on its
   own thread, join.
6. `close()` is idempotent. Concurrent callers get a deterministic
   `ApiClosedError` (or a `CancelledError` mapped to one). The session lease is
   released last.

**Live jobs this process owns do not survive `close()` or process exit.**
Durability covers persisted identity and results; execution belongs to the
owning process. Work that must outlive the interpreter goes to a long-lived
server process, or to a per-job detached owner —
`run_experiments(wait=False, detach=True)`, §11 — which is a separate process
and therefore a separate owner.

That rule is stated where it applies, not only here:

- `reference()` and `reference('run_experiments')` say that `wait=False`
  returns a receipt *and* that the submitting process must outlive the run
  unless `detach=True` hands the job to its own owner, because a job is
  cancelled when its owning process exits. The receipt itself carries a
  `process_owned_job` observation, but the catalogue is where a caller looks
  *before* submitting.
- During interpreter teardown the registry's async persist would raise
  `RuntimeError: cannot schedule new futures after shutdown`; that path falls
  back to a synchronous persist (blocking is fine during teardown), so the
  record gets written instead of an alarming and irrelevant traceback getting
  printed.
- A case abandoned because its owner died recovers with an error naming the
  owning process and the rule, rather than "Server restarted" — which is a
  guess about a mechanism the store cannot see, and wrong on this interface.

## 5. The six operations

Methods take a `dict` in and return a `dict` out, validated by the operation's
own input model. Validation failures raise `ApiValidationError` (a
`ValueError`) whose message comes from `compact_validation_error(...)` — the
same renderer the server dispatch uses, with the same `field_owners` context
passed, so locations, semantics and cross-tool referrals match the wire exactly.
A raw `pydantic.ValidationError` renders differently and is not what is
promised.

```python
api.run_experiments(**args)   # two-phase; see below
api.jobs(**args)
api.analyze_results(**args)
api.inspect(**args)
api.edit_schematic(**args)
api.verify_circuit(**args)
```

**Handler caps are real, so generic depagination is not possible.**
`budget=None` is already the fullest handler rendering, and several operations
cap irreversibly (analyze failures, verify findings per rule, 50-run receipt
pages, paged pin legends on a non-idempotent mutation). So each operation gets
its own completion strategy:

| Operation | Strategy |
|-|-|
| `analyze_results` | A **bounded resumable neutral evaluator**. The seam returns neutral work — rows, reductions, facts, failures, missing cases — plus an internal continuation position, honoring `analysis_budget_s` per drive so the whole-call bound stays (untrusted artifacts get a hard bound either way). Source capture and diagnostics must finish before a resumable set exists: an initialization timeout raises `AnalysisDeadlineExceeded`, creates no set/cursor, and requires retrying the original request with fewer sources or a larger `[analysis] analysis_budget_s`. MCP renders a completed initialization's continuation position as its opaque cursor; Python drives the evaluator repeatedly, accumulating neutral results until the position is exhausted. "No cursor merging" means no merging of *rendered* MCP pages; accumulating neutral, unprojected, unrendered work is well-defined by construction. |
| `verify_circuit` | Evaluator seam: uncapped findings per rule. The MCP interface keeps its per-rule cap plus a truncation observation on top. |
| `edit_schematic` | The mutation executes once and is never replayed. The seam is the **neutral in-memory view the handler already computes while holding the edit guard**: MCP paginates that view, Python returns it whole. Views are produced inside the edit transaction and bound to the committed `sha256`, never from a post-guard file re-read — a peer session's next revision could interleave. `dry_run` gets full views the same way, in memory, with nothing on disk to read. |
| `run_experiments`, `jobs(status\|wait)`, `jobs(runs)` | A **loop-atomic neutral receipt snapshot**: one non-suspending evaluation on the private loop copies every mutable job-derived receipt field together — canonical rows keyed `(case_id, run_index)`, status, completeness, `outcome`, `failures`, `observations`, `artifacts`, and the attached analysis's status, result, error and observations. Never multi-page collection over live mutable state: time-A completeness beside time-C rows violates the completeness rule, and a stale `outcome` or `hint` beside a fresh failed row is the same fork one level up. `outcome` and `hint` are derived *from* the snapshot after the copy; static submission fields and a wait's historical `timed_out` fact come from the same evaluation that produced the snapshot. For the three `jobs` actions this is literal: one `evaluate_jobs` call does the control-plane work, and the wire page and the Python dict are two renderings of that one evaluation, chosen by a presentation argument. Rendering by invoking the wire handler and then reading the job again — which is how this interface once assembled its complete receipt — reports two reads as one answer, with a window in between for the job to move. MCP pages the snapshot created for its current invocation, and a continuation request takes a fresh atomic snapshot and applies its existing offset, so live-status semantics are unchanged. Python takes one whole snapshot because it returns one whole response, then applies the original receipt's projection policy (`run_fields`, or the lean default) to it whole — returning the existing page shape with `returned == total`, `truncated == false`, `next_cursor == null`. No `assembled` field, no shape fork, no provenance change. If snapshot or assembly fails after submission, `ApiCallError` carries the original receipt and control token. A direct `api.jobs(action="runs")` goes through the same seam: treating it as an "other action" would recreate the mixed-time inventory the seam eliminates. |
| `jobs(list)` | The same single evaluation: the circuit-group inventory is read once and rendered whole, where the wire renders one offset page of it. |
| `jobs(cancel)` | One evaluation, no collection — the kill receipts are the acknowledgement itself and are never paged on either interface. |

**Two-phase `run_experiments`.** The API always submits with
`execution.wait_s=0`, because the wire dwell is a presentation constant and is
rejected as caller input (§7). The submission future is never cancelled on
Ctrl-C; it settles either to a pre-submit failure or to a durable receipt.
`wait=True` (the default) then blocks in successive `jobs(wait)` calls:

- Ctrl-C cancels only the current wait task. The job keeps running, and the API
  raises `ApiInterrupted` (a `KeyboardInterrupt`) **carrying the receipt and
  job id**, so the handle is never lost.
- `api.wait(job_id, timeout=None)` is the public wait. On timeout it returns
  the current `jobs(wait)` snapshot with `timed_out: true` and leaves the job
  running.
- `wait=False` returns the submission receipt immediately, annotated with the
  process-owned-job observation.
- `wait=False, detach=True` submits from a spawned per-job owner instead, so
  the job survives this process (§11). It is the one combination that does not
  submit in-process.
- Exiting the `Api` context still cancels owned live jobs (§4).

**Recovery through `Api.jobs`.** Opt in at initial submission with
`execution={"recoverable": True}`. `api.jobs(action="resume", job_id=parent_id,
resume_request_id="retry-1", control_token=parent_token)` returns the complete
resume receipt, including lineage, per-case attempt provenance and reused
accounting. Optional `case_ids`, `retry_failed`, and `retry_cancelled` select
eligible cases; `wait_s` bounds dwell and defaults to zero. A child receipt keeps
its control token even after completion. A no-op has `resumed=false` and no child
token; use standalone `analyze_results` when only analysis needs retrying.

Controlled startup supports ngspice on Linux and native Windows, and the audited
LTspice 26.0.2 executable on native Windows. Other LTspice builds and platforms
refuse recovery before submission. LTspice captures established settings from
`simulator.ltspice_ini` (environment `LTSPICE_MCP_LTSPICE_INI`, API override
`ltspice_ini`) or `%APPDATA%/LTspice.ini`. Relative configured paths resolve
from the working directory. The source remains unchanged; every attempt gets
its own writable copy of the captured template.

The LTspice template must contain the application's actual update-query time
within the preceding 15 days, checked again on resume and before launch. Resolve
the update reminder in LTspice before submitting if this check refuses. Recovery
never changes the timestamp, chooses usage consent or answers dialogs. The build
restriction and freshness check bind the inspected reminder behavior; they are
not a permanent vendor-supported update suppression mechanism.

For reproducible static ngspice randomness, pass
`execution={"simulator": "ngspice", "recoverable": True, "simulator_seed": 17}`
to `api.run_experiments`. The API preserves this execution field when removing
the wire dwell. The seed must be a strict integer in `1..2147483646`; every case
and retry uses the same initial seed. It does not define a Monte Carlo sampling
policy, select PDK statistics, or change startup files. Explicit seeds cannot
mix with native statistical families.

Seeded recovery admits static `agauss`, `gauss`, `aunif`, `unif`, and `limit`
with exactly one `.op`, `.ac`, `.dc`, or `.tran` analysis. Noise, stepped inputs
and combinations refuse before claim with `recovery_seed_analysis_unsupported`.
Unknown functions, transient random functions, caller controls and external
modules remain unsupported. The recorded compatibility mode remains in force:
statistical two-argument `limit` works in `hsa`; default `kiltpsa` supplies a
different three-argument clamping function. Configure the appropriate mode for
the deck separately; the seed does not override it.

Only resume accepts `api.jobs(detach=True, ...)`; `raw_page=True` cannot accompany
detach. It reuses the existing detached owner and its operation discriminant.
Before spawning, the caller authorizes against a fresh durable parent record. A
tokenless caller must match both the owning process's PID and creation time;
its parent token is then carried privately to the new owner. That owner
independently authorizes the resume under the lineage lock. The handoff and a
caller-supplied PID grant no authority. Token, detach and dwell do not change the
resume fingerprint. Save root and child tokens from submission/resume receipts:
`status`, `list`, and `runs` remain read-only and do not return them.

For example, this synthetic corner/temperature/supply campaign interrupts its
initial attempt, resumes cancelled cases, and checks every operating point.
The condition mapping and analytic oracle belong to caller code; the server
preserves case identity and captured electrical inputs. These synthetic library
sections are not measured process data or foundry models.

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

**`Api.reference(op=None) -> str`** is the argument catalogue.
`reference()` returns the six-operation index; `reference('edit_schematic')`
returns that operation's resolved argument tree — every field with type,
default, enum members and union branches written out, nested models flattened
onto dotted paths, and one worked example. No JSON Schema syntax, no `$ref`
chains. It is a **staticmethod**: reading the catalogue must not require an
engine session nor take the process's single session lease, so
`Api.reference('inspect')` works before anything is opened. It renders from the
same Pydantic models the call validates against, so it cannot drift.

**`Api.guide(section=None) -> str`** is the guide the MCP server serves
(`lib/guide.py`): with no argument, the core a session reads first and its index
of topic sections and task playbooks; with a name from that index, that part. A
staticmethod for the same reason as `reference()`, and the same text as
`inspect(kind: "guide")` and the `spice://guide` resources. The guide carries
this interface's usage (its `python` section), because this document is not in
the package and a model cannot read it.

`python -m ltspice_mcp.api reference [OP]` prints the same catalogue with no
engine boot, and `python -m ltspice_mcp.api guide [SECTION]` the guide. The package's lazy `__init__` (PEP 562) plus a stdlib-and-pydantic
catalogue module keep scipy and the MCP SDK out of the interpreter; a
cold-subprocess test pins that they stay out of `sys.modules`.

The six methods carry that same text as their `__doc__`, installed at
class-definition time from the same renderer, so `help(api.edit_schematic)` and
`inspect.getdoc` answer directly.

**Relative path arguments are taken from `working_dir`**, not the process CWD.
The resolve chain carries an optional base directory in a context variable, and
the `Api` sets it around every marshalled call, anchoring both the user path and
any relative entry in `allowed_paths` (a config file's
`allowed_paths = ["."]` is what exposed the CWD behavior). **MCP server
resolution is unchanged**, and a test pins that; the base is this interface's opt-in
only.

**For a subclass or a test double:** every public method marshals through
`ApiMethodsMixin._marshal`, which wraps the coroutine with the working-dir
anchor before handing it to `_call`. A host that mixes in `ApiMethodsMixin`
inherits the anchoring; `_call`'s contract is unchanged.

## 6. Errors

- A typed engine exception that *escapes* a handler propagates unchanged. A
  `PathSecurityError` gains one note (PEP 678) carrying the sandbox guidance a
  tool call's `hint` carries, so a traceback names the setting that widens the
  sandbox; its type and message are untouched.
- `result.isError=True` raises `ApiCallError`, carrying the complete structured
  payload plus convenience attributes `code`, `commit_state`, `job_id` and
  `control_token` — a post-submit or post-commit payload preserves every
  recovery handle.
- Per-item failures, and `outcome="partial"|"failed"` envelopes with
  `isError=False`, are **returned data**, not exceptions. Identical to the MCP
  interface's semantics.
- The bridge never synthesizes a typed exception from an error code.
- `structuredContent` is asserted present before unwrapping; a missing one is
  an `ApiInternalError`, not a silent `None`.

Exception hierarchy: `ApiError` is the base; `ApiCallError`, `ApiSessionError`
(with `ApiClosedError` beneath it) and `ApiInternalError` derive from it.
`ApiValidationError` derives from `ValueError` and `ApiInterrupted` from
`KeyboardInterrupt`, so ordinary `except ValueError` / Ctrl-C handling still
works.

## 7. Policy for wire-only controls

Default (automatic) mode **drops** the two presentation controls, `budget`
and `execution.wait_s`, and says so in the result's `warnings`: neither is
part of a request's identity (the fingerprint excludes both), so an MCP call
replayed through this door with them still attached is the same request, and
a caller carrying the MCP habit over loses nothing but the field. It
**rejects** any cursor or continuation field and `view_cursors`, with a
message naming `raw_page=True` as the remedy, because a dropped paging control
would change what the caller gets. It never auto-flips request fields that
participate in the idempotency fingerprint: `per_run`, `outliers` and
`signals_available` stay exactly as the caller wrote them, so an API replay of
an MCP request never becomes an idempotency conflict. Detail beyond the
handler's rendering comes from the §5 collectors, not from mutating the
submitted request. Presentation-only fields — `include.fields`, `run_fields`,
`provenance` — pass through freely.

`raw_page=True` accepts every wire control verbatim and returns exactly one
handler page. It is the preview mode, and the one way to get MCP-identical paging.

## 8. Curated primitives

```python
raw = api.load_raw(raw_path=..., plot_index=0, dialect=None)  # XOR: raw_path | job_id
raw = api.load_raw(job_id=..., run_index=0, case_id=None)
raw.signals                    # list[str]
raw.trace("V(out)", step=0)    # np.ndarray — complex values preserved
raw.axis(step=0)               # real sampled axis; refuses a native table
raw.step_count; raw.steps      # captured metadata aligned per step
raw.plots; raw.descriptor; raw.plot_index  # inventory and selected plot facts
raw.table(step=0)              # native table rows; complex values use real/imag
raw.analysis_type; raw.dialect; raw.source          # provenance
api.measurements(job_id=..., run_index=0, case_id=None)
api.measurements(log_path="result.log")  # XOR: log_path | job_id; RAW is optional
```

- `RawResult` is this project's wrapper; spicelib types never cross the
  boundary.
- `plot_index` is a strict nonnegative integer, independent of case/run and
  step selection. `dialect` is `None` or `ltspice`, `ngspice`, `qspice`, `xyce`;
  explicit dialect evidence must agree with the producer and captured header.
  The selected descriptor owns analysis type, axis, trace units, and step
  status. A native table retains its first quantity and has no fabricated axis.
  Inventory and descriptor metadata are detached dictionaries; table values
  are numeric facts, with no inferred engineering formulas.
- **Arrays are detached copies.** The parsed object is shared with the handler
  cache, which assumes immutability; caller mutation must not fork later results
  between the two interfaces.
- A job id resolves one experiment case; a caller-supplied path passes through
  path admission. Both routes carry one `AnalysisSource` into the shared loader.
  Step metadata belongs to the captured snapshot; consumers do not reopen a
  parent or sibling log to reconstruct it.
- RAW decoding runs in the contained parser worker and returns fully resident
  numeric views after validation and process cleanup. `measurements` reads
  detached captured log facts; it does not interpret native RAW table quantities.

Logs with no RAW use the existing programmable operations:

```python
facts = api.inspect(queries=[{
    "kind": "results", "view": "native_tables", "path": "result.log",
}])
measured = api.analyze_results(
    sources=[{"log_path": "result.log", "label": "imported"}],
    recipes=[{"metric": "measurements", "key": "meas"}],
)
```

The `measurements` inspect view pages value observations and their recorded
range/AT metadata. `native_tables` pages literal scalar entries or printed
frequency rows, retaining section boundaries, printed labels and real/imaginary
components. Units and analysis extent remain unknown where the log does not
prove them. There are no fabricated plot or step identities; measurement
vector ordinals are not RAW steps. Rows and metadata are detached plain values,
so agent code can arrange arrays or export them without another result class.

Direct log imports capture no RAW sibling, even when one exists. Log views
require `plot_index` omission. Analysis and attached analysis accept an omitted
selector; it selects plot zero only for a RAW recipe. Whole-log measurements run
once per source and reject `step`/`all_steps`; RAW recipes in a mixed batch can
still use those selectors. Malformed or missing RAW fails only RAW recipes.
Manifests bind captured RAW/log/console presence and bytes, including an empty
file versus absence. Snapshot hashing runs in the contained worker; the parent
does not reread whole artifacts to hash them. Repeated references share captured
work within a call. Initialization records all captured artifact-role hashes.
A source check answers from the resident cache only while every role still
carries the stat stamp (identity, size, modification and change time) recorded
when that content was hashed, and recaptures in the worker otherwise; the
`analyze_results` section of the MCP surface design gives the settle margin
and what a stamp cannot see.
Continuations reject companion/content drift. Manifests
keep explicit dialect hints separate from recorded producing evidence under
`include.provenance`; a whole-log row's producer dialect can remain null.
Log facts alone do not establish that a simulator solve completed or a case was admitted
as produced; job readers retain the existing terminal/produced gates.

The AC and transient metric functions are re-exported under their existing
names: arrays in, dict- or TypedDict-shaped mappings out, complex `H` for AC
and real axes.

- AC: `prepare_ac_arrays`, `unwrap_phase_safe`, `log_interp`,
  `log_interp_complex`, `detect_crossings`, `find_crossings_any_quantity`,
  `gain_at_frequencies`, `compute_filter_metrics`, `compute_stability_metrics`,
  `compute_roll_off`, `compute_resonances`, `compute_return_loss`,
  `integrate_noise`, `classify_filter`, `analyze_ac_structure`.
- Transient: `window_and_clean`, `analyze_edge`, `analyze_pulse_response`,
  `analyze_disturbance_response`, `analyze_timing_between`, `analyze_periodic`,
  `analyze_thd`, `analyze_tone`, `compute_signal_stats`,
  `time_weighted_quantiles`, `compute_measurement_stats`.
- Also `parse_spice_value`, which is not a metric but a value reader: variation
  values cross the boundary as SPICE literals (`'5p'`) in both directions, and
  nothing else on the facade parses one.

**Typing-surface policy.** Every *project-defined* alias, TypedDict or nested
output type appearing in a public annotation is importable from
`ltspice_mcp.api`. Standard-library and dependency types (`numpy.ndarray`,
`Sequence`, ...) are exempt. The type's defining module is an implementation
detail: re-binding cannot change `__module__` or postponed-annotation globals,
so `__module__`, reprs, documentation links and pickling behavior are
observable but explicitly **not stable** — only importability from the facade
is promised. A `typing.get_type_hints` test per public function pins that every
referenced type resolves and is importable from the facade.

## 9. `__all__` — the stability boundary

`__all__` is what this package promises. Additions are minor; removals and
renames are major. Dict return shapes track the MCP contract's versioning,
because they are the same shapes. Signatures, units and array conventions for
the §8 functions are frozen by their docstrings and pinned by an
`__all__`-coverage test.

It contains `Api`, `RawResult`, the exception types, the §8 function names, and
the literal typing surface:

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

A separate module, **`ltspice_mcp.api.types`**, re-exports the *argument* models
the six operations validate against: the render and compare policies
(`CompareSpec`, which `edit_schematic` takes, `VerifyRenderPolicy` and
`VerifyCompareSpec`, which `verify_circuit` takes, and `RenderPolicy`, the base
the verify render policy subclasses), the recipe union and its
members, the inspect query kinds, the variation rules, and public aliases for
the schematic op models (`AddComponentOp`, `WirePinsOp`, ...) that the applier
keeps private. A validation error names one of these types; without the module
there was no way to import the thing the message pointed at, and the observed
recovery was reflecting over a private module. It is a separate
module so that `__all__` stays the pinned stability boundary and does not move.

## 10. What the tests must cover

- **Parity.** `raw_page=True` with identical presentation arguments equals the
  handler's `structuredContent` verbatim, with a drift pin per operation.
  Default-mode results compare against an explicit composition of handler calls.
- **Three-part parity oracle for capped data**, since "equals fully-collected
  MCP pages" is impossible past an irreversible cap: (a) neutral evaluator
  output equals Python output; (b) MCP output equals the documented projection
  or cap of that neutral output; (c) every omitted record reconciles through
  totals and truncation observations.
- **Collectors.** More than 50 runs; more than 100 analyze rows or failures;
  more than 25 verify findings per rule; more than 100 pin rows on one edit; a
  batched `inspect` with several live cursors; an `.asc` net with pins *and*
  coordinates paged; an analyze continuation with per-run rows, missing cases,
  multiple recipes and a view-carrying `include.fields`; projected
  (`run_fields`) and lean-default run assembly both preserving the receipt's
  projection; a collector failure after a durable receipt keeping receipt and
  control token in the raised error; an inspect restart discarding
  pre-staleness partials (the two-revision mix test).
- **Edit views.** Revision interleaving — a peer session commits between our
  commit and any read, and the views still match *our* `sha256`; `dry_run`
  returns full views with nothing on disk.
- **Lease.** Concurrent constructors with exactly one winner; bootstrap failure
  releasing the lease; a forked child constructing fresh over the inherited
  stale lease, including a controlled fork *while another thread holds the
  lease lock* (an unlocked happy-path fork does not pin the failure); server
  lifespan and `Api` mutually exclusive in one process.
- **Shutdown against live jobs.** Closing after a durable receipt while a job
  the test keeps running is active must reach `cancel_running`, proving
  step 3 does not wait for natural completion.
- **Snapshot coherence.** The assembled receipt is internally consistent across
  all mutable fields: a job transitioning queued to produced during collection
  can never yield produced rows beside `produced == 0`; a case failing between
  invocation and snapshot can never yield `status="completed_with_failures"`
  rows beside a stale `outcome="in_progress"`, a missing failure entry, or a
  keep-waiting hint; an attached-analysis transition is captured atomically
  with the rows it describes.
- **Typing surface.** `typing.get_type_hints` resolves every public function,
  and every referenced type is importable from `ltspice_mcp.api`.
- **Lifecycle.** Close during pre-receipt submission, during a shielded rename,
  during a jobs wait, during a bounded parse; concurrent calls plus close;
  double close; calls after close; the fork guard; loop-thread re-entry.
- **Interrupt.** Ctrl-C before and after the durable receipt, with the handle
  preserved and the job not cancelled; `api.wait` timing out leaves the job
  running.
- **Parallel processes.** API and MCP server sharing a working directory, each
  shutdown cancelling only its own jobs.
- **Detached owners**, driven through a real spawned process: a job that
  survives the submitting `Api`'s `close()` and is read back complete by a
  fresh one; an idempotent replay of the same `request_id` returning the same
  job without a second submission; a cancel from another process ending the
  job and the owner; the owner killed mid-run leaving a job that classifies as
  interrupted rather than running; `detach=True, wait=True` refused; and parent
  and child holding their own engine leases at the same time.
- **`RawResult`.** Step slicing, experiment-case resolution, captured step metadata,
  array-mutation isolation (mutate a returned array, cached result unchanged),
  parse-deadline propagation, and AC complex-dtype pins on recorded fixtures.
- **Door policy.** Budget and cursor rejection in automatic mode; exact
  one-page behavior in raw mode; fingerprint identity — the same request via
  MCP and then via the API replays idempotently.
- **Bootstrap parity.** Symbol paths, working-directory config selection,
  allowed paths, cleanup and job preload identical between a server boot and an
  `Api` boot; no root-logger mutation in library mode.
- The archetype battery once through the Python API (build, run, analyze, verify)
  as an integration smoke test.

## 11. Detached owners, and the roadmap past them

### The per-job detached owner (`detach=True`)

`run_experiments(wait=False, detach=True)` submits an experiment that outlives
the calling process. The keyword defaults to `False`, so nothing about an
existing call changes.

**Only with `wait=False`.** `detach=True, wait=True` is an `ApiValidationError`
naming the two ways forward (drop `detach`, or pass `wait=False` and wait later
with `api.wait(job_id)`) — waiting in this process for a job this process does
not own is the shape the keyword exists to avoid. `detach=True` with
`raw_page=True` is likewise refused: `raw_page` returns exactly one handler page
of a submission this process performed, and a detached submission is performed
somewhere else.

**Who does what.** The calling process validates the arguments against
`RunExperimentsInput` (so a malformed request still raises in the caller's own
traceback, before any process is spawned), writes a request file, and spawns

```
sys.executable -m ltspice_mcp.detached_owner <request-file>
```

with `start_new_session=True`, its stdin closed, and stdout and stderr appended
to a log file. Same interpreter, same install, so there is no version skew. The
child opens an ordinary `Api` on the same working directory, config file and
constructor overrides, calls `run_experiments(wait=False)`, writes the receipt
to a handshake file, then blocks in `api.wait(job_id)` until the job is
terminal and exits. The parent polls for the handshake, annotates the receipt
and returns it.

On Windows, each owner holds a Job Object containing itself and its simulator
descendants. Windows closes that handle when the owner exits, including a
forced exit, and terminates the remaining descendants. An owner spawned from
`run_code` explicitly leaves the worker's Job Object, so resetting the worker
does not cancel a detached experiment. Breakaway is requested only when the
enclosing job permits it; an externally imposed job can still constrain the
process's lifetime. Windows virtual-environment launches use the underlying
interpreter with `__PYVENV_LAUNCHER__`, following CPython multiprocessing. This
preserves the environment without the redirector's additional process and
restrictive Job Object.

**The record is written by the process that owns it.** This is the ordering
choice, and it is the reason the API prepares nothing on disk beyond the
request file: staging and submission both happen in the child, so the job
record and its `request_id` index entry are first written by the existing
submission pipeline with `owner_pid` already set to the child's live pid.

What a reader sees in each interval:

| Interval | On disk | What any process reads |
|-|-|-|
| Parent has spawned the child; the child is booting and staging | request file only | Nothing. `jobs(status, request_id=...)` reports no job is indexed, which is true. |
| Child's submission pipeline has taken the request lock and written the index and the record | index + record, `owner_pid` = child, alive | A live foreign job, exactly like another session's — the existing owner-pid liveness path. |
| Child is supervising | record, updated by the child | Same, refreshed from disk by `refresh_foreign_job`. |
| Child reached terminality and exited | terminal record | A terminal job. |

There is no interval in which the record names a dead or unrelated owner. The
rejected alternative — parent writes the record, spawns, child rewrites
`owner_pid` — has exactly that interval: between the write and the rewrite the
record names the *parent*, a live process that will never advance the job, and
the parent's own `close()` would cancel it, because `cancel_running` matches on
`owner_pid == os.getpid()`.

**A child that dies.** Before it writes the record there is nothing to
misread; the parent's handshake wait ends when the child's exit is seen and
raises `ApiCallError` naming the log file. After the record exists — a kill
mid-run included — the record names a pid that is gone, and the existing
restart reconciliation classifies the job `interrupted`, promoting any case
whose artifacts are on disk. That is the same path a crashed server takes.

That reconciliation depends on an owner-liveness answer the pid alone cannot
give: a process that has exited but has not been collected by the process that
started it keeps its pid in the table, and the caller of a detached submission
is exactly that process. `owner_liveness` therefore asks what the process is
doing, and reads an exited one as dead.

**A caller that stops waiting.** The wait for the handshake bounds itself at
the owner's own wait for the request gate plus a boot allowance — an owner
blocked on that gate is waiting to replay whatever holds the id, and a shorter
bound would kill it for doing the right thing. Past that the owner and the
processes it started are stopped and the call raises. A Ctrl-C
during it raises `KeyboardInterrupt` and leaves the owner running — it is in
its own session, so the signal never reached it. Neither loses the job: asking
again with the same `request_id` replays whatever it submitted.

**Ownership.** The calling process never owns a detached job. `close()`,
leaving a `with` block and interpreter exit cancel jobs whose `owner_pid` is
this process, so they leave a detached job alone. Reading it back is the
ordinary foreign-job route (`jobs(action="status"|"wait"|"runs")` by `job_id`
or `request_id`), from this process, a later one, or a running MCP server.
Cancelling it is the ordinary foreign-owner route: `jobs(action="cancel")` with
the `control_token` from the receipt writes the durable cancellation marker,
the owner's coordinator sees it and stops its cases with a token-scoped kill.

**Replay.** A detached call with a `request_id` that already submitted spawns an
owner that takes the ordinary idempotent-replay path: it returns the existing
job's receipt and submits nothing. The receipt's owner pid is read from the
record, so it names whichever process actually owns the job, not the owner
just spawned.

**The receipt.** The returned receipt is the child's, with the
`process_owned_job` observation (true in the child, misleading in the caller)
replaced by:

```
code:     "detached_owner"
kind:     "lifecycle"
detail:   names the owning pid, says this process does not own the job and
          that closing the Api will not cancel it, and gives the log path
evidence: {"owner_pid": <int>, "log_file": "<path>"}
```

On a replay of a job that is already terminal the detail says instead that the
pid is the process the record names as having run it, that nothing is
supervising it now and there is nothing to cancel — the sentence about a live
supervisor would be an instruction to act on a process that has exited.

**Refused when nothing is persisted.** A session with `[state] persist_jobs`
off keeps its jobs in memory, so an owner's job would be invisible to the
caller that asked for it. `detach=True` is an `ApiValidationError` there rather
than a receipt for a job nothing can read back.

**Known costs, stated not fixed.** `max_parallel_sims` is a per-process cap
held by a runner instance, so every detached owner carries its own; N detached
jobs run at up to N x the cap between them, which widens the multi-session
residual already recorded in `CLAUDE.md`. And `start_new_session` is a POSIX
call: on Windows the owner starts in the caller's console process group, so a
Ctrl-C in that console reaches it as well. Everything else — ownership,
records, cancellation — is the same on both.

### Still ahead

A full broker daemon — a socket-addressed detached server mode — remains the
possible end state behind the detached owner. It inverts the
shutdown-cancels-jobs invariant and adds the usual daemon costs (version skew,
stale sockets, config drift), so it gets built only if the per-job owner proves
insufficient.

`api.log_diagnostics` is deferred unless callers are seen re-parsing logs by
hand.

## Coexistence with a server

The design intent is that the Python API and the MCP server share one
working directory and one set of job records, and that a long-lived server
process is the owner of jobs that must outlive a call, the provider of
resources (the guide, job resources), and the renderer of the
waveform widget — while scripts use the API for loops and complete results.
A job either interface starts is readable by the other by `job_id`.

What is implemented: shared records, shared store, one engine lease per
process, and hand-off through a per-job detached owner. A job submitted
through the API with `run_experiments(wait=False)` is owned by the submitting
process and is cancelled when that process exits; adding `detach=True` hands it
to a supervisor process spawned for that one job, which outlives the script
(§11). Either way the record is the same record, and the other interface reads
it by `job_id`.

One process can run several builds of a simulator family. Each build past the
family's own is a named executable, from `[simulator.executables]` or
`Api(working_dir=..., simulator_executables={"xvii": ".../XVIIx64.exe"})`, and
a call selects it with `execution={"simulator": "ltspice:xvii"}`
(`docs/design/mcp_surface.md`, "Selecting a build"). A second process on the
same working directory with its own `simulator_exe` still works, and is the
route when the two builds must not share a process. Either way each job
records the executable its cases launched (`simulator_executable`) and each run
the build it reported (`simulator_version`), so the builds' results stay
distinguishable in the shared records. A `request_id` reused from the other
process, or under another name, replays only when it would launch the same
build; otherwise it is an `idempotency_conflict` (`docs/design/mcp_surface.md`,
"Replay is scoped to the simulator build").

The records are shared through the working directory's store, which is
`<working_dir>/.ltspice-mcp/` unless `LTSPICE_MCP_STORE_DIR` moves it. With the
setting, the store is one directory per working directory under the named
directory, so the working directory still scopes the records, but a script
finds the server's jobs (and replays its `request_id`s) only when it carries
the same setting: an `Api` started without it opens
`<working_dir>/.ltspice-mcp/` and sees none of them. A detached owner and a
`run_code` worker inherit it from the process that starts them.
