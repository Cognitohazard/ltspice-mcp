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
`TypeError` rather than being silently ignored (`tool_profile` is irrelevant
here and is rejected).

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
5. Bounded-drain the residual loop tasks shutdown left behind, *without*
   joining abandoned bounded-parser worker threads — those are unjoinable by
   design, their loop-facing task settles, and a late worker completion cannot
   re-enter a closed loop. Then `shutdown_asyncgens`, stop the loop, close it on
   its own thread, join.
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
| `analyze_results` | A **bounded resumable neutral evaluator**. The seam returns neutral work — rows, reductions, facts, failures, missing cases — plus an internal continuation position, honoring `analysis_budget_s` per drive so the whole-call bound stays (untrusted artifacts get a hard bound either way). MCP renders that position as its opaque cursor; Python drives the evaluator repeatedly, accumulating neutral results until the position is exhausted. "No cursor merging" means no merging of *rendered* MCP pages; accumulating neutral, unprojected, unrendered work is well-defined by construction. |
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

**`Api.reference(op=None) -> str`** is the argument catalogue.
`reference()` returns the six-operation index; `reference('edit_schematic')`
returns that operation's resolved argument tree — every field with type,
default, enum members and union branches written out, nested models flattened
onto dotted paths, and one worked example. No JSON Schema syntax, no `$ref`
chains. It is a **staticmethod**: reading the catalogue must not require an
engine session nor take the process's single session lease, so
`Api.reference('inspect')` works before anything is opened. It renders from the
same Pydantic models the call validates against, so it cannot drift.

`python -m ltspice_mcp.api reference [OP]` prints the same catalogue with no
engine boot. The package's lazy `__init__` (PEP 562) plus a stdlib-and-pydantic
catalogue module keep scipy and the MCP SDK out of the interpreter; a
cold-subprocess test pins that they stay out of `sys.modules`.

The six methods carry that same text as their `__doc__`, installed at
class-definition time from the same renderer, so `help(api.edit_schematic)` and
`inspect.getdoc` answer directly.

**Relative path arguments are taken from `working_dir`**, not the process CWD.
The resolve chain carries an optional base directory in a context variable, and
the `Api` sets it around every marshalled call, anchoring both the user path and
any relative entry in `allowed_paths` (the generated TOML ships
`allowed_paths = ["."]`, which is what exposed the CWD behavior). **MCP server
resolution is unchanged**, and a test pins that; the base is this interface's opt-in
only.

**For a subclass or a test double:** every public method marshals through
`ApiMethodsMixin._marshal`, which wraps the coroutine with the working-dir
anchor before handing it to `_call`. A host that mixes in `ApiMethodsMixin`
inherits the anchoring; `_call`'s contract is unchanged.

## 6. Errors

- A typed engine exception that *escapes* a handler propagates unchanged.
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

Default (automatic) mode **rejects** `budget`, any cursor or continuation
field, `view_cursors`, and `execution.wait_s`, with a message naming the
remedy. It never auto-flips request fields that participate in the idempotency
fingerprint: `per_run`, `outliers` and `signals_available` stay exactly as the
caller wrote them, so an API replay of an MCP request never becomes an
idempotency conflict. Detail beyond the handler's rendering comes from the §5
collectors, not from mutating the submitted request. Presentation-only fields —
`include.fields`, `run_fields`, `provenance` — pass through freely.

`raw_page=True` accepts every wire control verbatim and returns exactly one
handler page. It is the preview mode, and the one way to get MCP-identical paging.

## 8. Curated primitives

```python
raw = api.load_raw(raw_path=...)                    # XOR: raw_path | job_id
raw = api.load_raw(job_id=..., run_index=0, case_id=None)
raw.signals                    # list[str]
raw.trace("V(out)", step=0)    # np.ndarray — complex preserved for .AC
raw.axis(step=0)               # real array (time or frequency)
raw.step_count; raw.steps      # metadata list aligned per step (log fallback)
raw.analysis_type; raw.dialect; raw.source          # provenance
api.measurements(job_id=..., run_index=0, case_id=None)
```

- `RawResult` is this project's wrapper; spicelib types never cross the
  boundary.
- **Arrays are detached copies.** The parsed object is shared with the handler
  cache, which assumes immutability; caller mutation must not fork later results
  between the two interfaces.
- Experiment cases resolve through `resolve_experiment_run` and legacy jobs
  through `resolve_raw_file`. `load_raw` routes on job type and never feeds an
  experiment job to the legacy resolver.
- All parsing goes through the bounded-parse wrapper, and `measurements`
  performs a bounded log parse — the synchronous inline loader is not called.

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
  `analyze_thd`, `compute_signal_stats`, `compute_measurement_stats`.
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
  `ThdOutput` (`HarmonicEntry`), `MeasurementStatsEntry`, `HistogramBin`,
  `AcStructureResult`, `Corner`, `Observation`.

A separate module, **`ltspice_mcp.api.types`**, re-exports the *argument* models
the six operations validate against: the render and compare policies
(`RenderPolicy` and `CompareSpec`, which `edit_schematic` takes, plus
`VerifyRenderPolicy` and `VerifyCompareSpec`, the subclasses `verify_circuit`
takes), the recipe union and its
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
- **`RawResult`.** Step slicing, experiment-case resolution, log fallback,
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
resources (`spice://guide`, job resources), and the renderer of the
waveform widget — while scripts use the API for loops and complete results.
A job either interface starts is readable by the other by `job_id`.

What is implemented: shared records, shared store, one engine lease per
process, and hand-off through a per-job detached owner. A job submitted
through the API with `run_experiments(wait=False)` is owned by the submitting
process and is cancelled when that process exits; adding `detach=True` hands it
to a supervisor process spawned for that one job, which outlives the script
(§11). Either way the record is the same record, and the other interface reads
it by `job_id`.
