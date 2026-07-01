# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP server that exposes LTSpice circuit simulation to LLMs via the Model Context Protocol. LTspice is the primary simulator; ngspice/qspice/xyce are supported but secondary. Built on the low-level `mcp.server.lowlevel.Server` API (not FastMCP) with spicelib as the simulation backend.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the server (stdio transport)
uv run ltspice-mcp

# Type checking
uv run pyright

# Lint
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_pathutil.py -v

# Debug a single failure (disable parallelism for readable output)
uv run pytest -n0 tests/test_pathutil.py::TestName::test_case -v

# Run directly
python -m ltspice_mcp
```

No Makefile. CI: `.github/workflows/publish.yml` (test + publish to PyPI on version tags).

`pytest-xdist` is available, but the suite runs serially by default. Pass `-n auto` to parallelize locally when you do not need deterministic output order or debugger attachment.

`docs/TESTING.md` is the testing-practice doc: the absence-class blind spot path-walking stress tests cannot see (a missing or unusable-for-a-class capability), and the mechanisms that catch it — inverse-op closure (`test_dispatch.py::TestOpInverseClosure`), the archetype build battery (`test_circuit_asc.py::TestArchetypeBuildCoverage`), task-down coverage, and blind-artifact judging. Read it before adding a tool or an op.

## Comments, docstrings, and commit messages

These are read by people outside this repo's internal process — keep internal jargon out of them. Do **not** use, in code comments, docstrings, or commit messages: severity codes (P0–P3), internal codenames for stress-test findings (e.g. J-KILL, J-MAXPAR), backlog item numbers (`open_followups` item N), or stress-test version numbers (v9/v10). Describe the actual behavior, condition, or bug in plain technical terms instead. Internal planning docs and the backlog may use that shorthand; shipped code and git history may not.

## Architecture

All source lives under `src/ltspice_mcp/`.

### Layered Design

```
MCP Protocol Layer    server.py — lifespan, dispatch, request routing
                      resources.py — MCP resources & URI templates
Tool Layer            tools/*.py — tool definitions + handlers
Core Logic Layer      lib/*.py — see below
Config/State          config.py, state.py, errors.py
```

Key `lib/` modules:
- `services.py` — application-level service layer shared by tools and resources. Owns job resolution, cached result loading, and reusable extraction logic. Sits between MCP adapters and pure parsers. **Unified result read-model:** `runs_of(job)` projects either a `SimulationJob` or a `BatchJob` into a uniform `list[RunRef]` (a single run = batch-of-one); `resolve_run(job_id, run_index)` + `resolve_raw_file`/`resolve_log_file` address any run through it (gated on `completed`). This is the one place that knows the two physical result layouts (one multi-step raw vs N single-point raws), so extraction stays job-agnostic. `query_value`/`bode_metrics` accept `job_id`+`run_index` to analyze a sweep/MC run like a standalone raw (job-run raws bypass `safe_path` — trusted server artifacts, like `batch_results`).
- **SPICE lexer/validator** — `spice_lex.py` (foundation netlist lexer → `list[SpiceCard]` tokens), `spice_lex_ops.py` (cross-card transform passes), `spice_lex_views.py` (typed views over cards), `spice_validator.py` (pre-flight directive + arity validation). The `.cir`/`.net` read / list / value-edit paths and `validate_netlist` arity checks run on this pipeline, not spicelib's `SpiceEditor` (which is still used for some other `.cir`/`.net` editor ops).
- **Job subsystem** — `job_types.py` (domain dataclasses `SimulationJob`/`BatchJob`/`RunRef`/configs, re-exported from `state` to break import cycles), `job_registry.py` (in-memory registry + disk persistence + interrupted-job recovery + `preload_recent`), `job_store.py` (per-circuit JSON sidecar at `{circuit_dir}/.ltspice-mcp/jobs/{job_id}.json`), `job_lifecycle.py` (declarative status state machine, `transition()`).
- `sim_runner.py`, `sweep_runner.py`, `montecarlo_runner.py`, `runner_base.py` — spicelib runner wrappers (`runner_base` = shared scaffolding); `montecarlo.py` is the pure perturbation engine behind `montecarlo_runner`
- `runner_manager.py` — centralized runner lifecycle (see Key Patterns)
- `simulator.py` — simulator detection, WSL/Wine selection
- `ltspice_wsl.py`, `wsl.py` — WSL path conversion and interop
- `ac_analysis.py`, `signal_analysis.py` — pure-function analysis primitives for frequency-domain (.AC) and transient (.tran) `.raw` data; back the structured analysis tools (`bode_metrics`, `signal_stats`, etc.)
- `raw_parser.py`, `log_parser.py` — simulation `.raw` / `.log` result parsing
- `library_manager.py`, `library_parser.py`, `encoding.py` — component library handling + library/netlist encoding detection
- `batch_results.py` — sweep/MC batch result extraction
- `component_value.py` — element-class-typed dispatcher behind `set_component_value`
- `cache.py` — `FileCache` for editor and result instances
- `pathutil.py` — path security (`safe_path()`, `resolve_safe_path()`); `filelock.py` — cross-process advisory file locks
- `recent.py` — global recently-touched-circuit index (`recent.json`); backs the `recent` tool + job preload
- `format.py`, `sweep_utils.py` — formatting + sweep helpers
- `symbol_geometry.py`, `geometry.py` — .asy symbol parsing (pin positions, rotation transforms, bounding boxes) + shared 2D / bbox helpers
- `mcp_logging.py`, `observability.py` — MCP protocol log notifications + structured job-lifecycle events

### Tool Module Convention

Tool modules (circuit, simulation, analysis, advanced, library, status) use a decorator-based registry. Each tool is registered via `@registry.tool()` in its module:

```python
@registry.tool(
    name="foo",
    description="...",
    input_model=FooInput,          # subclass of ToolInput (Pydantic)
    annotations=RO_ANNOTATIONS,    # or custom ToolAnnotations
    profiles=("full", "agentic"),  # which profiles expose this tool
    output_schema={...},           # optional: JSON Schema for structuredContent
)
async def handle_foo(args: FooInput, state: SessionState) -> types.CallToolResult:
    ...
```

`tools/__init__.py` simply imports all tool modules to trigger registration, then exposes `get_tools_for_profile()` which delegates to `registry.get_for_profile()`. `SessionState.create()` calls this during lifespan init.

To add a new tool: define it with `@registry.tool()` in the appropriate module and ensure the module is imported in `tools/__init__.py`. Set `profiles=("full", "agentic")` if it should appear in both profiles, or `profiles=("full",)` for full-only.

**Shared helpers in `_base.py`**: `text_response()`, `json_response()`, `format_response()` for building `CallToolResult`; `StrictModel` as the Pydantic base for strict validation config; `ToolInput(StrictModel)` as the base for top-level tool input models; `RO_ANNOTATIONS` for read-only tools; `paginate()` + `pagination_metadata()` for list endpoints; `PAGINATION_SCHEMA`, `PIN_SCHEMA`, `BBOX_SCHEMA` for reusable output schema fragments.

**Output schemas**: Tools that return `structuredContent` (via `format_response()`) declare an `output_schema` (or an `output_model` TypedDict) for client introspection. Text-only confirmation tools (`text_response()`) don't need one. Dispatcher tools that delegate to another handler (e.g. `bode_metrics`) may omit it — the sub-handler's structuredContent carries the shape. Tools with `output_schema` must return `structuredContent` on every code path — never fall back to `text_response()`.

### Schematic Editing (.asc)

Direct editing of LTspice `.asc` schematics is a first-class feature. All circuit tools live in **`tools/circuit.py`** — extension-based dispatch picks `AscEditor` or `SpiceEditor` automatically:

- **`read_circuit`** works on both `.cir` and `.asc` — returns raw netlist for `.cir`, or schematic layout (positions, labels, wires) for `.asc`.
- **`list_components`** lists components (with optional `prefix` filter) or looks up a single component's value via `reference` param.
- **`set_component_value`** handles both single (`reference`+`value`) and batch (`values` dict) modes.
- **`parameter`** reads all .PARAM values (no args) or sets one (`name`+`value`).
- **`edit_directive`** adds or removes SPICE directives via `action: "add"|"remove"`.
- **Schematic-only tools** (`export_netlist`, `connect`, `symbol_info`, `component_info`, `reset_schematic`) validate `.asc` extension and use `_get_asc_editor()`.
- **`reset_schematic`** reverts an `.asc` to the byte snapshot taken before its first in-session mutation (recovery hatch). `_snapshot_asc()` captures it at the `_editing` choke point and in `apply_schematic_ops`; the snapshot lives on `state.asc_snapshots` (per-session only).
- **`connect`** wires two pins by reference (e.g., `M1.D` → `M4a.D`) with waypoint routing. Validates before writing: refuses diagonal wires, pin collisions, and wire junction overlaps. Warns on long runs and bbox crossings.
- **`add_component`** returns pin positions (with direction), bounding box, and overlap warnings.
- **`symbol_info`** / **`component_info`** provide pin geometry for layout planning.
- **`trace_net`** reports every pin/label/wire vertex on the net at a pin/`net:NAME`/`(x,y)`, flagging multi-label shorts. Built on the shared `_net_partition` union-find (also backs `_trace_nets`).

#### Standalone tool vs. apply_schematic_ops op

A schematic mutation is exposed as a **standalone MCP tool only when its result returns information the model acts on** (structured `format_response`/`output_schema`). An ack-only mutation — one that just confirms "done" — lives **only as an `apply_schematic_ops` op**: a standalone tool's schema costs the model context whether or not it's ever called, and that cost is only earned by a useful return. MCP best practice is fewer, more capable tools (Anthropic's "writing tools for agents"; tool-selection accuracy degrades past roughly 15 tools), and this server is over budget. The rule:

- **Standalone (structured return the model uses):** `add_component` (pin positions, bounding box, overlap warnings), `connect` (orthogonal routing result, collision/junction checks).
- **Ops-only (ack-only mutations):** `move_component`, `remove_component`, `set_component_attribute`, `add_net_label`, `remove_net_label`, `remove_wire`. Their `handle_*` functions stay as unregistered internal handlers (their tests call them directly); the `apply_schematic_ops` op path (`_apply_op_inplace`) is what the MCP surface reaches.
- **Lifecycle / entry exception:** `create_schematic` and `reset_schematic` stay standalone regardless of return shape — they have no batch home.
- **Reads** (`read_circuit`, `list_components`, `symbol_info`, `component_info`, `trace_net`) stay standalone.

**Consolidated AC / step tools (clean break — no aliases):** `bode_metrics(mode="filter"|"slope"|"point"|"crossing")` is the single public AC tool; it dispatches to the now-unregistered internal compute adapters `handle_filter_metrics`/`handle_roll_off`/`handle_gain_at`/`handle_find_crossing` (still in `tools/analysis.py`, still unit-tested directly). `query_value(step_axis=, step_value=)` absorbs the former `step_get` (the `handle_step_get` adapter stays internal in `tools/circuit.py`; `query_value` imports it lazily). To re-expose any adapter, re-add its `@registry.tool(...)`. `bode_metrics(all_steps=true)` runs the chosen mode for every step of a `.step` sweep (per-step dispatch via the shared `_bode_dispatch`), returning a `steps` list.
- **`export_netlist`** shows diff against previous export.
- All tools use `"path"` as the file parameter name.

AscEditor requires `.asy` symbol library files. Platform handling in `server.py:_configure_asc_editor()`:

| Platform | How symbol paths are resolved |
|-|-|
| Windows native | `AscEditor.prepare_for_simulator()` (spicelib built-in) |
| Linux native (LTspice via Wine) | `AscEditor.prepare_for_simulator()` (spicelib handles Wine paths) |
| WSL + LTspice on Windows | `wsl.get_ltspice_lib_paths()` resolves `%LOCALAPPDATA%` via `cmd.exe` |
| Any platform without LTspice | No .asc support (no .asy symbol files available) |

Users can override via `[schematic] symbol_paths` in TOML or `LTSPICE_MCP_SYMBOL_PATHS` env var.

### Key Patterns

- **Heavy blocking work is offloaded with `asyncio.to_thread`**: the MCP SDK dispatches each request as its own task on one shared event loop, so anything that blocks the loop stalls every in-flight request (including `cancel_job`). Slow filesystem work is therefore awaited via `asyncio.to_thread` at its call sites — raw parses (`services.load_raw` is async; `load_raw_sync` is the worker-side residual), batch raw/log loops, the recent-index lock + durable write (`SessionState.note_recent_circuit`), WSL `cmd.exe` interop (`resolve_output_folder`), and whole MCP resource reads (`server.py:read_resource`). Two invariants: response building (`format_response`/`json_response`) stays in the handler coroutine (the test suite's schema-conformance hook attributes emissions by walking the handler frame on the current thread), and mutable cached editors are touched only on the loop — editor parses/mutations stay inline by contract (see the contract comment in `tools/_base.py`). Simulation runners still use their own background threads for long-lived simulator work. Do not reintroduce inline heavy work citing thread-deadlock concerns — that old claim had no recorded basis and `tests/test_loop_responsiveness.py` pins the offloaded behavior.
- **Path security**: All user-provided paths go through `safe_path()` → `resolve_safe_path()`, which validates against `config.allowed_paths`. Raises `PathSecurityError` on violation.
- **Lifespan context**: `server_lifespan()` creates `SessionState` (config + detected simulators + `JobRegistry` + profile-filtered tool dispatch). Handlers receive state via `server.request_context.lifespan_context["state"]`.
- **Job lifecycle & persistence**: `SessionState` delegates all job state to `JobRegistry` (`lib/job_registry.py`); `state.jobs`/`state.add_job`/etc. are thin delegators. When `state.persist_jobs` is on, jobs round-trip through per-circuit JSON sidecars (`{dir}/.ltspice-mcp/jobs/`) via `job_store.py`, and `preload_recent()` reloads jobs for recently-touched circuits at startup. Status transitions go through the `job_lifecycle.transition()` state machine; `run_simulation` enforces `config.max_parallel_sims`. On shutdown, `cancel_running()` kills live simulators and `drain_pending()` flushes persistence.
- **Structured errors**: Use the hierarchy in `errors.py` (PathSecurityError, NetlistError, SimulationError variants). Handlers catch `LTSpiceMCPError` subtypes and return error text; unknown exceptions propagate to MCP SDK.
- **Log diagnostics**: `log_parser.py:extract_log_diagnostics()` extracts structured warnings and errors from LTspice log files (parse errors with caret pointers, Fatal Error, convergence messages, etc.). Used by `run_simulation`, `check_job`, `measurement_stats`, and `simulation_summary` to surface errors instead of silently returning empty results.
- **Runner lifecycle**: `RunnerManager` (`lib/runner_manager.py`) owns all runner instances (sim, sweep, MC). Accessed via `state.runners.get_sim_runner(loop, simulator_class, output_folder)` etc. The manager auto-invalidates cached runners when the event loop, simulator class, or output folder changes. Never create runners directly.

### Result-trust: surface, don't judge

Rules for any field that rates/flags/classifies a result (consumer is an LLM; rationale + canonical impl in `lib/result_observations.py`):

- **No trust verdicts** (`confidence`/`unreliable`/`suspect`/`degraded`): surface facts, let the model judge. Observation/`quality` codes name an input condition or provenance fact (window shape, which rail-samples were used, a value past a salience cutoff).
- **Severity is only relayed** — attach one only when the simulator already assigned it.
- **`status` ≠ trust:** lifecycle `status` (set early from raw size) stays separate from the diagnostics-derived trust signal.
- **Thresholds prefer** relay > topology > signature > relative > bare magnitude; magnitude stays advisory; emit a null/flag over a meaningless number; one constant, one meaning.
- **`observations` vs `warnings` are two distinct channels — don't merge them.** `observations` = is the data/solve trustworthy (relayed log errors, non-finite/extreme values, coverage gaps; structured, doctrine-governed). `warnings` (on the analysis tools) = did this measurement make assumptions (clamped window, ambiguous edge, FFT-window approximation; free-text, actionable). A run-level solve failure is a data fact: read tools relay it into `observations` where they have that channel, else into their `warnings`. The metric tools lack an `observations` channel; add one (and move their solve-failure relay onto it) only when a second data-fact needs a home — never via a breaking `warnings`→`observations` migration. Full rationale in `lib/result_observations.py`.

### WSL Support

On WSL, LTspice.exe runs via Windows interop (not Wine). Key adaptations:
- `lib/ltspice_wsl.py`: `LTspiceWSL` subclass overrides `run()` to convert paths via `wslpath` instead of Wine's `Z:` prefix. Auto-selected by `lib/simulator.py` when `is_wsl()` is True.
- `simulator.path` in `ltspice-mcp.toml` must be set to the Windows-side path (e.g., `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) since spicelib can't auto-detect across WSL boundary.
- The simulation **runner output folder is kept stable** (one `{working_dir}/.ltspice-mcp/runs` sidecar) so the single cached runner, `cancel_job`, and the global `max_parallel` cap stay valid — a per-deck output dir would change the folder on every run in a different directory, and `RunnerManager._ensure_fresh` invalidates the whole runner cache when the folder changes (losing in-flight process handles and splitting the concurrency semaphore per directory). Each run's artifacts are uniquely named (`{job_id}.*`) so they stay isolated within that shared folder; callers find them through the result path `check_job` reports (the MCP analysis tools resolve by `job_id` and don't care about location). `resolve_output_folder` (`tools/_base.py`) has two overrides: a deck with a relative `.include`/`.lib` runs in its own dir (the simulator resolves the include from the staged netlist's dir, so it can't move — applies to single runs and sweeps/MC alike), and **WSL + LTspice + a Linux-fs source** routes to a Windows-native temp dir, because LTspice (a Windows process reaching the Linux fs over a `wsl.localhost` UNC share) can't write the SQLite `.db` behind `.MEAS` over UNC. `cancel_job` resolves the runner via the **job's own netlist** so its folder matches the launching runner.
- LTspice requires netlist files to have an extension (`.cir`, `.net`, `.sp`). `sim_runner.py` preserves the original extension in `run_filename`.

### Configuration

`ltspice-mcp.toml` in working directory (auto-generated if missing). Environment variables with `LTSPICE_MCP_` prefix override TOML values. See `config.py:ServerConfig` for all options. On WSL, set `simulator.path` to the LTspice Windows executable path.

TOML sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[logging]`, `[schematic]`, `[tools]`, `[state]` (`persist_jobs`).

**ngspice compatibility-mode gotcha:** spicelib runs ngspice in `kiltpsa` mode by default, whose `lt`/`ps` tokens make it read a sectioned `.lib <file> <section>` (the PDK corner-select idiom) as two plain includes and drop the section → "could not find include file". `[simulator] ngbehavior` (or `LTSPICE_MCP_NGBEHAVIOR`) overrides it (e.g. `"hsa"`), applied at startup by `lib/simulator.py:_apply_ngbehavior`; `services.ngbehavior_lib_hint` surfaces the fix on that failure. Runtime default unchanged (spicelib's).

### Tool Profiles

`config.tool_profile` controls which tools are exposed. Set via `[tools] profile` in TOML or `LTSPICE_MCP_TOOL_PROFILE` env var.

| Profile | Tools | Use case |
|-|-|-|
| `full` (default) | All 49 | Any MCP client, automation, non-agent LLMs |
| `agentic` | 41 | LLM agents with native file access (Read/Edit/Write) |

The "agentic" profile removes 8 tools: the netlist-editing wrappers (`create_netlist`, `read_circuit`, `set_component_value`, `parameter`, `edit_directive`) and library session management (`load_library`, `unload_library`, `list_libraries`) — things capable agents do natively. It deliberately **keeps** `configure_sweep`/`configure_montecarlo`: the only producers of the `config_id` that `run_sweep`/`run_montecarlo` consume, and Monte Carlo perturbation + N-run aggregation (and the batch-sweep route) are not something an agent reproduces with native file edits the way it can a plain LTspice `.step`. It keeps simulation lifecycle, binary `.raw` parsing, batch run/results, `find_model` search, and the schematic-construction + wiring + inspection set (`create_schematic`, `add_component`, `apply_schematic_ops`, `connect`, `export_netlist`, `reset_schematic`, `symbol_info`, `component_info`, `trace_net`) — geometry-aware .asc editing (orthogonal routing, pin-collision and junction checks) that hand-writing the file can't match. The ack-only mutations (`move_component`, `remove_component`, `set_component_attribute`, `add_net_label`, `remove_net_label`, `remove_wire`) are `apply_schematic_ops` ops, not standalone tools (see the standalone-vs-op rule above).

Profile-filtered tool defs and dispatch live on `SessionState` (`state.tool_defs`, `state.tool_dispatch`). Each tool's `profiles` frozenset (set at registration via `@registry.tool(profiles=...)`) determines visibility. Error hints in `server.py` are profile-aware (tuples of `(full_hint, agentic_hint)`) so they don't reference tools the client can't see.
