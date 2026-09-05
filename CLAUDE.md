# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

Start with `CONTRIBUTING.md` (setup, gate, review rules) and `docs/DESIGN.md` (architecture and rationale). This file is the working map of the source: what lives where, and which patterns are load-bearing.

## Project Overview

MCP server exposing SPICE circuit simulation to LLMs. LTspice is the primary simulator; ngspice/qspice/xyce are supported but secondary. Built on the low-level `mcp.server.lowlevel.Server` API (not FastMCP) with spicelib as the simulation backend. The same engine is also importable in-process as `ltspice_mcp.api`.

## Commands

Standard `uv` invocations (`uv sync`, `uv run pyright`, `uv run ruff check --fix src/ tests/`, `uv run pytest tests/ -v`) work as expected; the dev tools are declared in `pyproject.toml`. The non-obvious ones:

```bash
# Debug a single failure (disable parallelism for readable output)
uv run pytest -n0 tests/test_pathutil.py::TestName::test_case -v

# Run the server without the console script
python -m ltspice_mcp
```

No Makefile. CI: `.github/workflows/publish.yml` (test + publish to PyPI on version tags).

`pytest-xdist` is available, but the suite runs serially by default. Pass `-n auto` to parallelize locally when you do not need deterministic output order or debugger attachment.

**Working practice** (these cost real time when ignored):
- **Don't use `uv run python -c` to explore an API.** Read the source with `Read`/`Grep` instead — a throwaway interpreter session answers one question and teaches nothing that survives.
- **A regression test must fail before the fix and pass after.** If it never failed, it does not pin the behavior; run it against the unfixed code to prove it bites.
- **Test through real code paths, never a bypass**, and assert on real values — a test that stubs the mechanism it claims to cover passes for the wrong reason.
- **A green suite is green.** Do not re-run it to feel sure; re-run it only after the tree changes.

`docs/TESTING.md` is the testing-practice doc: the absence-class blind spot (a missing capability, or one unusable for an input class nobody fed it) and the mechanisms that catch it — inverse-op closure (`test_dispatch.py::TestOpInverseClosure`), the archetype build battery (`test_circuit_asc.py::TestArchetypeBuildCoverage`), task-down coverage, and blind-artifact judging. Read it before adding a tool or an op.

## Comments, docstrings, commit messages, and branch names

These are read by people outside this project — keep internal shorthand out of them. Do **not** use, in code comments, docstrings, commit messages, or branch names: severity codes, internal codenames for findings, backlog item numbers, or internal batch/version labels. Name a branch for what it changes (`fix/asc-export-lock`). Describe the actual behavior, condition, or bug in plain technical terms instead. Internal planning notes may use that shorthand; shipped code, git history, and branch names may not.

## spicelib bugs

spicelib is a pinned third-party dependency we cannot fix in place, so we work around its bugs and remove the workaround once upstream is fixed. **Whenever you hit a spicelib bug or limitation** — a wrong parse, a hang/infinite loop, a silently dropped field, an unapplied header value — **record it in `docs/spicelib_bugs.md`** as a self-contained, upstream-ready section: summary, affected code + version, reproduction, impact, proposed fix, a suggested upstream test, and a cross-reference to our workaround and the test that pins it. Do this even when you also ship a workaround — the file is the record of what to delete once upstream lands, and the reproduction is what lets someone confirm the fix. Follow the format of the existing entries.

## Architecture

All source lives under `src/ltspice_mcp/`.

```
MCP Protocol Layer    server.py — lifespan, dispatch, request routing
                      resources.py — MCP resources & URI templates
Tool Layer            tools/*.py — tool definitions + handlers
Core Logic Layer      lib/*.py — see below
Config/State          config.py, state.py, errors.py
```

Key `lib/` modules:
- `services.py` — application-level service layer shared by tools and resources. Owns job resolution, cached result loading, and reusable extraction logic. Sits between MCP adapters and pure parsers. **Runs are case-addressed:** `resolve_experiment_run(job_id, ...)` / `experiment_run_context(job, ...)` resolve one case of an experiment to an `AnalysisSource` (raw, log, deck, dialect, identity), gated on the job being terminal and the case having produced. Case raws bypass `safe_path` — they are trusted server artifacts, and a reloaded job's raw can live outside `allowed_paths` (a WSL temp dir).
- **SPICE lexer/validator** — `spice_lex.py` (foundation netlist lexer → `list[SpiceCard]` tokens), `spice_lex_ops.py` (cross-card transform passes), `spice_lex_views.py` (typed views over cards), `spice_validator.py` (pre-flight directive + arity validation). The `.cir`/`.net` read / list paths and `verify_circuit`'s arity checks run on this pipeline, not spicelib's `SpiceEditor`. **Known gap:** `validate_netlist_dangling_nodes`, `validate_netlist_directive_refs` and `validate_netlist_bias_topology` still have their own unit tests but no caller — `verify_circuit`'s syntax check runs only `validate_directive` and `validate_netlist_arity`, so the dangling-node, directive-reference and bias-topology findings the removed `validate_netlist` tool surfaced now reach nobody.
- **The store** — `store.py` owns the on-disk layout: **every path the server writes is a method on `Store`**, so the set of directories it can create is the set of methods there. Read its module docstring for the tree. Three rules that are not obvious: run artifacts group under `runs/{job_id}/` (staged decks included) while the *runner's* output folder stays the one `runs/` root, because `RunnerManager` caches a runner per output folder and a runner owns the `max_parallel` semaphore — the grouping rides on `run_filename`, which the simulator layer joins onto the unchanged folder; the WSL rule that moves LTspice's whole tree to a Windows-native disk is decided once in `Store.artifact_base`; and one `store_version`, stamped in `store.json` at the root, covers every record the store writes (job, request index, circuit index, cancellation, result set, attached-analysis snapshot, recent index) — bump it as one. `tests/test_store_layout.py` pins both ends: what a real session creates, and what the `Store` API can create.
- **Job subsystem** — `job_types.py` (`LegacyJobRecord` plus the status vocabulary and `legacy_record_observation`, re-exported from `state` to break import cycles), `job_registry.py` (in-memory registry + disk persistence + `preload_recent`; `get_or_load` is the one route from an id to a job, and `_adopt` is the one place that decides whether an off-loop caller may register what it read), `job_store.py` (reads the per-circuit sidecars at `{circuit_dir}/.ltspice-mcp/jobs/{job_id}.json` that releases before 0.6 wrote — read, never written, and no longer sharing a directory with anything this build writes), `job_lifecycle.py` (declarative status state machine, `transition()`).
- **Experiment subsystem** (backs every job the server runs) — `experiment_types.py` (the domain dataclasses `ExperimentJob`/`ExperimentCase`/`Completeness`; `ExperimentJob` also carries the live `asyncio.Event`s and coordinator `Task` the runner waits on, so it is not asyncio-free), `experiment_store.py` (the record's shape: serialization, restart reconciliation, the request index behind durable idempotency, and the per-circuit index behind `jobs(list, circuit=...)`), `experiment_runner.py` (the standalone coordinator: per-case futures, cancel gates, retained permits on unconfirmed kills), `variations.py` (variation expansion, target ladder param→ref→@model), `deck_staging.py` (manifest staging; asks the store where a simulator's decks may go), `lint_rules.py`. Cases run on the job-agnostic `submit_netlist` primitive in `runner_base`.
- **Analysis result plumbing** — `recipes.py` (the typed recipe union behind `analyze_results`), `result_store.py` (immutable result sets, composite raw+log digest manifests), `cursor_codec.py` (the checksummed cursor a surface resuming into stored work mints; also used by `pin_legend.py` and `inspect`), `pagination.py` (the page shape every list surface returns, and the plain `"o:<offset>"` cursor a recomputed listing uses).
- **Schematic rendering** — `schematic_scene.py` (`.asc` → absolute-coordinate `Scene`), `schematic_renderer.py` (SVG), `raster.py` (PNG via the optional `raster` extra), `netlist_graph.py` (structural comparison behind `verify_circuit`), `pin_legend.py` (per-component pin/net table).
- `runner_base.py` — the shared spicelib runner scaffolding (`submit_netlist`, outcome classification, token-scoped kill) that `experiment_runner` builds on; `montecarlo.py` is the pure perturbation engine `variations.py` draws from
- `deck_prep.py` — getting one deck ready to run: path resolution, the `.asc` → `.net` export and its per-schematic lock, and the content-addressed deck snapshot. Distinct from `deck_staging.py`, which stages a whole include manifest for an experiment
- `projection.py` — dotted-path field projection (`keep_plan`/`project_row`), shared by `analyze_results` and `run_experiments`
- `schematic_ops.py` — the `.asc` edit engine: op models, appliers, geometry, net tracing. Every name another module imports from it is public
- `runner_manager.py` — centralized runner lifecycle (see Key Patterns)
- `simulator.py` — simulator detection, WSL/Wine selection
- `ltspice_wsl.py`, `wsl.py` — WSL path conversion and interop
- `metrics.py` — the numeric core behind every `analyze_results` recipe: one function per discriminant, each taking a resolved `AnalysisSource` explicitly, with `METRICS` mapping recipe class to function. `tools/analyze.py` looks a recipe up there rather than branching on its type, and a recipe class with no entry fails at import. Also home to the readers those functions share — axis and window handling, the AC signal loader, the log relay, operating-point device matching, .MEAS aggregation
- `ac_analysis.py`, `signal_analysis.py`, `ac_structure.py` — pure-function analysis primitives for frequency-domain (.AC) and transient (.tran) `.raw` data; the metric functions in `metrics.py` are what put results on the wire
- `raw_parser.py`, `log_parser.py` — simulation `.raw` / `.log` result parsing
- `library_manager.py`, `library_parser.py`, `encoding.py` — component library handling + library/netlist encoding detection
- `component_value.py` — element-class-typed dispatcher behind the `set_component_value` op
- `cache.py` — `FileCache` for editor and result instances
- `pathutil.py` — path security (`safe_path()`, `resolve_safe_path()`); `filelock.py` — cross-process advisory file locks
- `recent.py` — global recently-touched-circuit index (`recent.json`); backs the startup job preload
- `symbol_geometry.py`, `geometry.py` — .asy symbol parsing (pin positions, rotation transforms, bounding boxes) + shared 2D / bbox helpers
- `mcp_logging.py`, `observability.py` — MCP protocol log notifications + structured job-lifecycle events

Self-describing helpers not listed above (`format.py`, `sweep_utils.py`, `desktop.py`, `plot_html.py`) do what their names say — read them when you need them.

### Tool Module Convention

Registered tools use a decorator-based registry via `@registry.tool()`:

```python
@registry.tool(
    name="foo",
    description="...",
    input_model=FooInput,           # subclass of ToolInput (Pydantic)
    annotations=RO_ANNOTATIONS,     # or custom ToolAnnotations
    output_schema={...},            # optional: JSON Schema for structuredContent
)
async def handle_foo(args: FooInput, state: SessionState) -> types.CallToolResult:
    ...
```

`tools/__init__.py` simply imports all tool modules to trigger registration, then exposes `get_tools()` which delegates to `registry.get_tools()`. `SessionState.create()` calls this during lifespan init. `get_tools` raises if the registry resolves to zero tools — a working handshake advertising nothing is the failure mode that guard exists for.

**Registered tools and delegated handlers.** Only the seven registered tools carry `@registry.tool`. A handler one of them delegates to declares its own contract with `@declare_output_schema(...)` (same schema argument forms as `registry.tool`): the contract belongs to the handler, not to its registration, and the test suite's conformance hook validates every emission at the first stack frame carrying one — so a delegated emission is checked against the DELEGATE's schema, never the delegating tool's envelope. A new delegation without a declaration fails `test_conformance_hook_armed.py`'s closure test. A handler with no caller at all is dead: the pre-0.6.0 tool handlers were kept as a rollback seam through 0.6 development and are gone, and so is the adapter layer `analyze_results` used to reach its numbers through — the recipes call `lib/metrics.py` directly. Both come back only from git history.

**Shared helpers in `_base.py`**: `text_response()`, `json_response()`, `format_response()` for building `CallToolResult`; `StrictModel` as the Pydantic base for strict validation config; `ToolInput(StrictModel)` as the base for top-level tool input models; `RO_ANNOTATIONS` for read-only tools; the shared response `Envelope` (`outcome_of`, `outcome_schema`, `FINDING_SCHEMA`, `failures_schema`, `page_schema`); the shared `RenderPolicy`/`CompareSpec` argument models; and `PIN_SCHEMA`/`BBOX_SCHEMA` for reusable output schema fragments. The offset-paged envelope itself (`{items, total, returned, truncated, next_cursor}` and the `o:<offset>` cursor) is `lib/pagination.py` — below the tool layer, because `analyze_results`' pages and the experiment records use it too; `lib/pin_legend.py`'s checksummed, binding-scoped cursor is a separate, stricter contract.

**`_schema.py`** holds everything about the JSON Schema a tool *publishes*, so a change to how schemas are shrunk is not a change to the module every tool imports: `ToolInput` and `build_input_schema` (title strip → type-keyword compaction → shared-fragment `$defs` hoist), `strip_wire_prose` (the advertised copy, keeping only load-bearing field prose per `WIRE_PROSE_KEEP`), and `schema_from_typeddict` (the output-schema generator). `_base.py` re-exports the names it uses plus a short, deletable compatibility block.

**Output schemas**: Tools that return `structuredContent` (via `format_response()`) declare an `output_schema` (or an `output_model` TypedDict) for client introspection. Text-only confirmation tools (`text_response()`) don't need one. A dispatcher that delegates to another handler may omit it — the sub-handler's structuredContent carries the shape. Tools with `output_schema` must return `structuredContent` on every code path — never fall back to `text_response()`.

**Every registered tool's `output_schema` must be an object schema at the top level.** A bare `oneOf`/`anyOf` makes strict MCP clients (Claude Code included) reject the entire `tools/list` response, disabling every tool on the server — pinned by `test_dispatch.py::TestSchemaPostProcessing::test_output_schema_top_level_is_object`.

**Structured self-sufficiency**: structured-aware clients (Claude Code included) render only `structuredContent` when it is present and drop the text channel entirely, so the data dict must carry everything the caller needs to act on. Any caller-guidance in a `format_response()` text (hints, referrals, recovery steps, caveats) must be mirrored into `data` — conventionally an optional `hint` string key declared in the `output_schema`. The text channel is presentation only.

### Schematic Editing (.asc)

Direct editing of LTspice `.asc` schematics is a first-class feature. The MCP surface for it is `edit_schematic` (typed op batch), `inspect` (symbol/net/components reads), and `verify_circuit` (lint/export/compare/render). The shared implementation lives in **`lib/schematic_ops.py`** — extension-based dispatch picks `AscEditor` or `SpiceEditor` automatically. Every name another module imports from it is public; read its module docstring for the map. What lives there:

- `plan_connect_route` wires two pins by reference (e.g., `M1.D` → `M4a.D`) with waypoint routing, behind the `wire_pins` op. Validates before writing: refuses diagonal wires, pin collisions, wire junction overlaps, and a same-instance tie LTspice would drop. Warns on long runs and bbox crossings.
- The `add_component` op returns pin positions (with direction), bounding box, and overlap warnings; `inspect(kind:"symbol")` provides the same geometry non-destructively for pre-placement planning.
- `handle_trace_net` (in `tools/inspect_tools.py`, behind `inspect(kind:"net")`) reports every pin/label/wire vertex on the net at a pin/`net:NAME`/`(x,y)`, flagging multi-label shorts. Built on `schematic_ops.net_partition`, the union-find that also backs `trace_nets`.
- There is no session-side undo. `edit_schematic` is revision-guarded and transactional (a failed batch writes nothing), so recovery from a committed edit is the caller's own copy of the sheet.
- Every mutation is an op in `edit_schematic`'s batch, applied by `apply_op_inplace` under `run_op_batch`; `post_op_warnings` runs the validation pass over the mutated editor afterwards, and `collapse_result_warnings` folds an advisory repeated across ops into one entry with a count.
- What is left in `tools/analysis.py` is the `plot_waveform` tool and the two artifact writers `analyze_results` shares with it (`build_waveform_csv`, `build_plot_file`); every recipe's numbers come from `lib/metrics.py`.
- All path-taking surfaces use `"path"` as the file parameter name.

AscEditor requires `.asy` symbol library files. Platform handling in `server.py:_configure_asc_editor()`:

| Platform | How symbol paths are resolved |
|-|-|
| Windows native | `AscEditor.prepare_for_simulator()` (spicelib built-in) |
| Linux native (LTspice via Wine) | `AscEditor.prepare_for_simulator()` (spicelib handles Wine paths) |
| WSL + LTspice on Windows | `wsl.get_ltspice_lib_paths()` resolves `%LOCALAPPDATA%` via `cmd.exe` |
| Any platform without LTspice | No .asc support (no .asy symbol files available) |

Users can override via `[schematic] symbol_paths` in TOML or `LTSPICE_MCP_SYMBOL_PATHS` env var.

### Key Patterns

- **Heavy blocking work is offloaded with `asyncio.to_thread`**: the MCP SDK dispatches each request as its own task on one shared event loop, so anything that blocks the loop stalls every in-flight request (including a cancel). Slow filesystem work is therefore awaited via `asyncio.to_thread` at its call sites — raw parses (`services.load_raw` is async; `load_raw_sync` is the worker-side residual), batch raw/log loops, the recent-index lock + durable write (`SessionState.note_recent_circuit`), WSL `cmd.exe` interop (`Store.artifact_base`, reached off-loop by the staging route), and whole MCP resource reads (`server.py:read_resource`). Two invariants: response building (`format_response`/`json_response`) stays in the handler coroutine (the test suite's schema-conformance hook attributes emissions by walking the handler frame on the current thread), and mutable cached editors are touched only on the loop — editor parses/mutations stay inline by contract (see the contract comment in `tools/_base.py`). Simulation runners still use their own background threads for long-lived simulator work. Do not reintroduce inline heavy work citing thread-deadlock concerns — that old claim had no recorded basis and `tests/test_loop_responsiveness.py` pins the offloaded behavior.
  - **Offload is not a wedge guard — bounding untrusted work is a separate obligation.** `asyncio.to_thread` stops a *slow* operation from stalling the loop; it does nothing for a *runaway* one, because a CPU-bound loop inside a third-party parser holds the GIL and stalls the loop from any thread. So treat every artifact a simulator hands back (raw/log) as untrusted input: a parse through a dependency you don't control can hang or loop indefinitely on a shape you didn't anticipate, and the only real protection is a hard bound (wall-clock or subprocess) that fails that one call instead of every session. Offloading heavy work and bounding untrusted work are orthogonal — a hot parse on a shared path needs both, and "we moved it to a thread" is not evidence it's safe.
- **Path security**: All user-provided paths go through `safe_path()` → `resolve_safe_path()`, which validates against `config.allowed_paths`. Raises `PathSecurityError` on violation.
- **Lifespan context**: `server_lifespan()` creates `SessionState` (config + detected simulators + `JobRegistry` + the tool dispatch built once from `tools.get_tools()`). Handlers receive state via `server.request_context.lifespan_context["state"]`.
- **Job lifecycle & persistence**: `SessionState` delegates all job state to `JobRegistry` (`lib/job_registry.py`); `state.experiment_jobs`/`state.add_experiment_job`/etc. are thin delegators. Experiments round-trip through `experiment_store`; `preload_recent()` reloads recently-touched circuits' records at startup, including the pre-0.6 sidecars `job_store` reads. Resolution has one route — `services.resolve_job` asks the registry, the registry asks the store — so "not in memory" and "not on disk" are one answer rather than a fallback chain each caller re-assembles. Status transitions go through the `job_lifecycle.transition()` state machine. On shutdown, `cancel_running()` cancels this process's own experiments and `drain_pending()` flushes persistence.
- **Records an earlier release wrote**: a pre-0.6 sidecar loads as an inert `LegacyJobRecord` — id, circuit, claimed kind, and the status that release persisted, with any live-looking status reported as `interrupted`. Every read of one surfaces `legacy_record_observation` and produces no results; `analyze_results` refuses it with the same sentence. Loading must never crash the registry or the preload, and must never silently skip: a caller told nothing waits on a job that will never report.
- **Structured errors**: Use the hierarchy in `errors.py` (PathSecurityError, NetlistError, SimulationError variants). Handlers catch `LTSpiceMCPError` subtypes and return error text; unknown exceptions propagate to the MCP SDK. `server.py:_ERROR_HINTS` maps an error type to one recovery hint, appended to the message and mirrored into `structuredContent["hint"]`; the hints name only tools the client can see.
- **Log diagnostics**: `log_parser.py:extract_log_diagnostics()` extracts structured warnings and errors from LTspice log files (parse errors with caret pointers, Fatal Error, convergence messages, etc.). Consumed by `runner_base` at run completion, by `services` for the run-level solve-failure relay, and by `lib/metrics.py` and `tools/analyze.py` when reading results — so a failed run surfaces its errors instead of silently returning empty results.
- **Runner lifecycle**: `RunnerManager` (`lib/runner_manager.py`) owns the experiment runner instances. Accessed via `state.runners.get_experiment_runner(loop, simulator_class, output_folder)`. Runners are cached per (kind, simulator class, output folder) with a small LRU cap — a per-run simulator override depends on a second simulator's runner NOT evicting the first's in-flight launch permits/cancel state, so eviction skips runners reporting `has_active_work()` (the cache may transiently exceed the cap). Only an event-loop change invalidates everything. **`max_parallel_sims` is the RUNNER's cap, not a job's:** `RunnerBase` holds one `asyncio.Semaphore(max_parallel)` and every case acquires a permit for the life of its simulator process (`acquire_launch_slot`/`release_launch_slot`), so concurrent `run_experiments` calls share one quota instead of each getting a private one. A request's own `execution.max_parallel` is a second semaphore layered on top that divides that job's share — it can only lower it (`ExperimentRunner._case_capacity` clamps to the runner's cap), which is why `tools/experiments.py` passes the runner `config.max_parallel_sims` and never the request's value. Cancel resolves the coordinator by ownership (`get_experiment_runner_for(job)` — the cancel state lives only on the instance that launched the job). Result parsing likewise follows the job: `services.dialect_for_job`/`raw_dialect_for` derive the raw dialect from the simulator the job ran on, not the session default. Never create runners directly.
- **Parallel sessions**: independent server processes may share a working directory; they coordinate only via the filesystem. Three mechanisms (all pinned by `tests/test_parallel_sessions.py`): (1) every circuit-file mutation/export runs inside `circuit_file_lock` (cross-process `file_lock` under `.ltspice-mcp/locks/`), acquired BEFORE the cached-editor fetch so the stat-on-fetch sees a peer's completed write — keep that ordering, and resolve any geometry a mutation depends on (pin positions, routes) INSIDE the guard, never from a pre-lock editor fetch; exports lock both the `.asc` and the sidecar `.net` they overwrite (fixed order `.asc`→`.net`); (2) job sidecars carry the owning server's `pid` — a running job with a live owner loads as `running` in other sessions, `services.resolve_job` and the job listings refresh it from disk via `refresh_foreign_job` (which swaps the registry entry only on the event loop; off-loop callers like resource-read worker threads get a read-only fresh view), and `cancel_running` only cancels `owner_pid == os.getpid()` jobs; (3) process kills are token-scoped (`lib/proc_kill.py` psutil + the WSL taskkill), matching simulator executable AND job_id at a filename boundary in the command line — never reintroduce a name-global kill (spicelib's `kill_all_spice` is deliberately unused: it's a no-op in the pinned spicelib and collateral once "fixed"). Known residuals: `max_parallel_sims` is enforced per runner instance, and runners are keyed by (kind, simulator, output folder) — so a process driving two simulators holds two caps, and N sessions multiply it again, same-file edit coherence on coarse-mtime filesystems (`/mnt/c` DrvFs) can still miss a same-size same-tick rewrite, and a pid recycled by an unrelated process reads as a live owner. **Artifacts are not scoped the way records are:** the experiment runs root is one directory per box (Windows temp under WSL), and every session's jobs land under it. Each job now gets its own `runs/{job_id}/` subdirectory, so its `.raw`/`.log`/staged decks are enumerable as a set instead of by guessing at a filename prefix — but the root is still shared and still readable by anything running as that user. An agent asked to recover a crashed session will inventory that folder with ordinary shell, find another session's job directories, and analyze them believing they are its own. An artifact's provenance comes from a job record that claims it, never from its presence in the folder.

### Result-trust: surface, don't judge

Rules for any field that rates/flags/classifies a result (consumer is an LLM; rationale + canonical impl in `lib/result_observations.py`):

- **No trust verdicts** (`confidence`/`unreliable`/`suspect`/`degraded`): surface facts, let the model judge. Observation/`quality` codes name an input condition or provenance fact (window shape, which rail-samples were used, a value past a salience cutoff).
- **Severity is only relayed** — attach one only when the simulator already assigned it.
- **`status` ≠ trust:** lifecycle `status` (set early from raw size) stays separate from the diagnostics-derived trust signal.
- **Thresholds prefer** relay > topology > signature > relative > bare magnitude; magnitude stays advisory; emit a null/flag over a meaningless number; one constant, one meaning.
- **A batch reports completeness, not just status.** An aggregate over N requested runs must reconcile produced + failed against N and surface any shortfall as a fact — a terminal `completed` that silently omits dropped runs is data loss dressed as success, and the consumer has no way to know a run vanished. Count what was promised, not just what was collected: if the number that came back is less than the number asked for, that gap is itself a result to surface, never a rounding-down to "done".
- **`observations` vs `warnings` are two distinct channels — don't merge them.** `observations` = is the data/solve trustworthy (relayed log errors, non-finite/extreme values, coverage gaps; structured). `warnings` (on the analysis tools) = did this measurement make assumptions (clamped window, ambiguous edge, FFT-window approximation; free-text, actionable). A run-level solve failure is a data fact: read tools relay it into `observations` where they have that channel, else into their `warnings`. The metric tools lack an `observations` channel; add one (and move their solve-failure relay onto it) only when a second data-fact needs a home — never via a breaking `warnings`→`observations` migration. Full rationale in `lib/result_observations.py`.

### WSL Support

On WSL, LTspice.exe runs via Windows interop (not Wine). Key adaptations:
- `lib/ltspice_wsl.py`: `LTspiceWSL` subclass overrides `run()` to convert paths via `wslpath` instead of Wine's `Z:` prefix. Auto-selected by `lib/simulator.py` when `is_wsl()` is True.
- `simulator.path` in `ltspice-mcp.toml` must be set to the Windows-side path (e.g., `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) since spicelib can't auto-detect across the WSL boundary.
- The simulation **runner output folder is kept stable** (one `runs/` root per box, from `Store.runs_root`) so the cached runner, cancellation, and the global `max_parallel` cap stay valid — `RunnerManager` keys its cache on the output folder and a runner owns the launch permits that enforce `max_parallel_sims`, so a folder that varied per job would hand every job its own runner and its own full quota, and losing an in-flight runner loses its process handles. **Per-job grouping happens a level down**: `store.run_filename_in(job_id, name)` returns a relative sub-path that spicelib joins onto the unchanged folder (it copies the deck to `output_folder / run_filename` and derives the raw and log from that copy's own path), so a job's artifacts land in `runs/{job_id}/` without the runner moving. `ExperimentJob.output_folder` records that per-job directory, which is what the crash reconciliation and the kill-path cleanup both reconstruct from. **WSL + LTspice** moves the whole tree to a Windows-native temp dir, because LTspice (a Windows process reaching the Linux fs over a `wsl.localhost` UNC share) can't write the SQLite `.db` behind `.MEAS` over UNC — that decision lives in `Store.artifact_base` and every writer asks it. Cancellation resolves the runner by ownership, so its folder always matches the launching runner.
- LTspice requires netlist files to have an extension (`.cir`, `.net`, `.sp`). `runner_base.submit_netlist` preserves the original extension in `run_filename`.

### Configuration

`ltspice-mcp.toml` in working directory (auto-generated if missing). Environment variables with `LTSPICE_MCP_` prefix override TOML values. See `config.py:ServerConfig` for all options. On WSL, set `simulator.path` to the LTspice Windows executable path.

TOML sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[logging]`, `[schematic]`, `[tools]`, `[state]` (`persist_jobs`).

**ngspice compatibility-mode gotcha:** spicelib runs ngspice in `kiltpsa` mode by default, whose `lt`/`ps` tokens make it read a sectioned `.lib <file> <section>` (the PDK corner-select idiom) as two plain includes and drop the section → "could not find include file". `[simulator] ngbehavior` (or `LTSPICE_MCP_NGBEHAVIOR`) overrides it (e.g. `"hsa"`), applied at startup by `lib/simulator.py:_apply_ngbehavior`. Runtime default unchanged (spicelib's). **Known gap:** the remediation hint that used to name this fix on that exact failure reached a caller only through the removed `run_simulation` tool, so nothing surfaces it now — a caller has to know the knob exists. **Known gap:** the remediation hint that used to name this fix on that exact failure reached a caller only through `run_simulation`, so nothing surfaces it now — a caller has to know the knob exists.

### Tool Profile

Since 0.6.0 the server has ONE profile, `consolidated` — 7 registered tools: six ops over three planes — EXECUTE (`run_experiments`, `jobs`), UNDERSTAND (`analyze_results`, `inspect`), AUTHOR (`edit_schematic`, `verify_circuit`) — all sharing one response envelope, plus `plot_waveform` (the interactive MCP Apps waveform widget, which predates the envelope and stays outside its contract). The six's handlers live in their own `tools/` modules (`experiments.py`, `jobs.py`, `analyze.py`, `schematic_edit.py`, `verify.py`, `inspect_tools.py`), with the receipt `run_experiments` and `jobs` share — its shape, schema, and renderers — in `receipts.py`. `tools/__init__.py`'s import list is the advertised tool order and is deliberately not alphabetised. **The envelope, the per-tool argument shapes, and the rationale are in `docs/design/mcp_surface.md`** — read it before changing any of the six; this file deliberately does not duplicate it.

The former `full` (49-tool) and `agentic` (41-tool) profiles were removed in 0.6.0. For one release, `[tools] profile` stays a *recognized* key: the values `"full"`/`"agentic"` produce a warning naming the removal and the `ltspice-mcp==0.5.*` pin, then the consolidated surface is served (the key is deleted in 0.7.0). Nothing else keys on the profile: registration and lookup have no profile parameter, because there is one surface to serve. Tool defs and dispatch live on `SessionState` (`state.tool_defs`, `state.tool_dispatch`), built once at lifespan init from `tools.get_tools()`.

### Public Python API (`ltspice_mcp.api`)

The same six ops are importable: `Api(working_dir=...)` boots the engine in-process (`engine.bootstrap_library_engine` — the same bootstrap `server_lifespan` enters via `bootstrap_server_engine`) and exposes them as synchronous methods with **complete** results where the wire pages or caps, plus `load_raw`/`measurements` (numpy access, detached copies) and the AC/transient metric functions under their existing names. One evaluator, two doors (MCP / Python) — the handlers and the API consume the same evaluate/render seams, so anything that would fork semantics between doors is a defect.

Things that will bite you if unknown: one live engine session per PID (an atomic lease shared with the server lifespan — an `Api` inside a server process raises); the `Api` owns a private persistent event loop (per-call loops would invalidate the runner cache); `close()` cancels jobs this process owns, exactly like server shutdown; default mode *rejects* wire-only controls (budget, cursors, dwell) instead of rewriting them so API/MCP replays stay idempotent (`raw_page=True` is the single-page parity hatch). **The full contract is `docs/design/python_api.md`**; `__all__` is the stability boundary and is pinned.
