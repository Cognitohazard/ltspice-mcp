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

The suite runs in parallel by default via `pytest-xdist` (`-n auto` in `pyproject.toml`). Pass `-n0` to serialize when you need deterministic output order or to attach a debugger.

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
- `sim_runner.py`, `sweep_runner.py`, `montecarlo_runner.py` — spicelib runner wrappers
- `runner_manager.py` — centralized runner lifecycle (see Key Patterns)
- `simulator.py` — simulator detection, WSL/Wine selection
- `ltspice_wsl.py`, `wsl.py` — WSL path conversion and interop
- `raw_parser.py`, `log_parser.py` — simulation result parsing
- `library_manager.py`, `library_parser.py` — component library handling
- `batch_results.py` — sweep/MC batch result extraction
- `cache.py` — `FileCache` for editor and result instances
- `pathutil.py` — path security (`safe_path()`, `resolve_safe_path()`)
- `format.py`, `plotting.py`, `sweep_utils.py` — formatting and plot helpers
- `symbol_geometry.py` — .asy symbol parsing, pin positions, rotation transforms, bounding boxes

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
- **Schematic-only tools** (`remove_component`, `move_component`, `set_component_attribute`, `export_netlist`, `connect`, `add_net_label`, `add_text`, `symbol_info`, `component_info`, `reset_schematic`) validate `.asc` extension and use `_get_asc_editor()`.
- **`reset_schematic`** reverts an `.asc` to the byte snapshot taken before its first in-session mutation (recovery hatch). `_snapshot_asc()` captures it at the `_editing` choke point and in `apply_schematic_ops`; the snapshot lives on `state.asc_snapshots` (per-session only).
- **`connect`** wires two pins by reference (e.g., `M1.D` → `M4a.D`) with waypoint routing. Validates before writing: refuses diagonal wires, pin collisions, and wire junction overlaps. Warns on long runs and bbox crossings.
- **`add_component`** returns pin positions (with direction), bounding box, and overlap warnings.
- **`symbol_info`** / **`component_info`** provide pin geometry for layout planning.
- **`add_net_label`** supports `pin="M3.S"` for placement at pin coordinates.
- **`schematic_from_netlist`** generates a full `.asc` from SPICE netlist text: grid-places supported 2-terminal elements (R/C/L/V/I/D) on their LTspice symbols and connects pins by net label (FLAGs carrying the node name) so the result is electrically identical to the netlist. Multi-terminal/controlled/subckt elements (M/Q/J/X/E/G/F/H) are returned in `skipped`. Reuses `_parse_netlist_for_synth` (pure) + `_layout_synth_components`.
- **`trace_net`** reports every pin/label/wire vertex on the net at a pin/`net:NAME`/`(x,y)`, flagging multi-label shorts. Built on the shared `_net_partition` union-find (also backs `_trace_nets`).

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

- **Blocking calls use `run_sync()`**: `tools/_base.py:run_sync()` wraps synchronous spicelib calls. Currently executes inline (not threaded) because threaded filesystem access deadlocks in this environment. Simulation runners still use their own background threads for long-lived simulator work.
- **Path security**: All user-provided paths go through `safe_path()` → `resolve_safe_path()`, which validates against `config.allowed_paths`. Raises `PathSecurityError` on violation.
- **Lifespan context**: `server_lifespan()` creates `SessionState` (config + detected simulators + batch job tracking + profile-filtered tool dispatch). Handlers receive state via `server.request_context.lifespan_context["state"]`.
- **Structured errors**: Use the hierarchy in `errors.py` (PathSecurityError, NetlistError, SimulationError variants). Handlers catch `LTSpiceMCPError` subtypes and return error text; unknown exceptions propagate to MCP SDK.
- **Log diagnostics**: `log_parser.py:extract_log_diagnostics()` extracts structured warnings and errors from LTspice log files (parse errors with caret pointers, Fatal Error, convergence messages, etc.). Used by `run_simulation`, `check_job`, `get_measurements`, and `get_simulation_summary` to surface errors instead of silently returning empty results.
- **Runner lifecycle**: `RunnerManager` (`lib/runner_manager.py`) owns all runner instances (sim, sweep, MC). Accessed via `state.runners.get_sim_runner(loop, simulator_class, output_folder)` etc. The manager auto-invalidates cached runners when the event loop, simulator class, or output folder changes. Never create runners directly.

### WSL Support

On WSL, LTspice.exe runs via Windows interop (not Wine). Key adaptations:
- `lib/ltspice_wsl.py`: `LTspiceWSL` subclass overrides `run()` to convert paths via `wslpath` instead of Wine's `Z:` prefix. Auto-selected by `lib/simulator.py` when `is_wsl()` is True.
- `simulator_exe` in `ltspice-mcp.toml` must be set to the Windows-side path (e.g., `/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) since spicelib can't auto-detect across WSL boundary.
- Simulation output goes to a Windows-native temp dir when working dir is on the Linux filesystem. This is required for `.MEAS` results — LTspice's SQLite `.db` writes fail on UNC paths (`\\wsl.localhost\...`), which causes measurement data to be lost from `.log` files.
- LTspice requires netlist files to have an extension (`.cir`, `.net`, `.sp`). `sim_runner.py` preserves the original extension in `run_filename`.

### Configuration

`ltspice-mcp.toml` in working directory (auto-generated if missing). Environment variables with `LTSPICE_MCP_` prefix override TOML values. See `config.py:ServerConfig` for all options. On WSL, set `simulator.path` to the LTspice Windows executable path.

TOML sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`, `[plotting]`, `[logging]`, `[schematic]`, `[tools]`.

### Tool Profiles

`config.tool_profile` controls which tools are exposed. Set via `[tools] profile` in TOML or `LTSPICE_MCP_TOOL_PROFILE` env var.

| Profile | Tools | Use case |
|-|-|-|
| `full` (default) | All 48 | Any MCP client, automation, non-agent LLMs |
| `agentic` | 32 | LLM agents with native file access (Read/Edit/Write) |

The "agentic" profile removes netlist-editing wrapper tools (e.g., `create_netlist`, `read_circuit`, `set_component_value`, `parameter`, `edit_directive`) and library session management — these are things capable agents do natively. It keeps simulation lifecycle, binary `.raw` parsing, batch orchestration, AscEditor-dependent ops, and library search.

Profile-filtered tool defs and dispatch live on `SessionState` (`state.tool_defs`, `state.tool_dispatch`). Each tool's `profiles` frozenset (set at registration via `@registry.tool(profiles=...)`) determines visibility. Error hints in `server.py` are profile-aware (tuples of `(full_hint, agentic_hint)`) so they don't reference tools the client can't see.
