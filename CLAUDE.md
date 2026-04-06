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

# Run directly
python -m ltspice_mcp
```

No Makefile. CI: `.github/workflows/publish.yml` (test + publish to PyPI on version tags).

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
- `services.py` — application-level service layer shared by tools and resources. Owns job resolution, cached result loading, and reusable extraction logic. Sits between MCP adapters and pure parsers.
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
    name="ltspice_foo",
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

**Output schemas**: All 19 tools that return `structuredContent` (via `format_response()`) declare `output_schema` for client introspection. The remaining 16 tools return text-only confirmations via `text_response()` and don't need output schemas. Tools with `output_schema` must return `structuredContent` on every code path — never fall back to `text_response()`.

### Schematic Editing (.asc)

Direct editing of LTspice `.asc` schematics is a first-class feature. All circuit tools live in **`tools/circuit.py`** — extension-based dispatch picks `AscEditor` or `SpiceEditor` automatically:

- **`ltspice_read_circuit`** works on both `.cir` and `.asc` — returns raw netlist for `.cir`, or schematic layout (positions, labels, wires) for `.asc`.
- **`ltspice_list_components`** lists components (with optional `prefix` filter) or looks up a single component's value via `reference` param.
- **`ltspice_set_component_value`** handles both single (`reference`+`value`) and batch (`values` dict) modes.
- **`ltspice_parameter`** reads all .PARAM values (no args) or sets one (`name`+`value`).
- **`ltspice_edit_directive`** adds or removes SPICE directives via `action: "add"|"remove"`.
- **Schematic-only tools** (`ltspice_remove_component`, `ltspice_move_component`, `ltspice_set_component_attribute`, `ltspice_export_netlist`, `ltspice_connect`, `ltspice_add_net_label`, `ltspice_add_text`, `ltspice_get_symbol_info`, `ltspice_get_component_info`) validate `.asc` extension and use `_get_asc_editor()`.
- **`ltspice_connect`** wires two pins by reference (e.g., `M1.D` → `M4a.D`) with waypoint routing. Validates before writing: refuses diagonal wires, pin collisions, and wire junction overlaps. Warns on long runs and bbox crossings.
- **`ltspice_add_component`** returns pin positions (with direction), bounding box, and overlap warnings.
- **`ltspice_get_symbol_info`** / **`ltspice_get_component_info`** provide pin geometry for layout planning.
- **`ltspice_add_net_label`** supports `pin="M3.S"` for placement at pin coordinates.
- **`ltspice_export_netlist`** shows diff against previous export.
- All tools use `"path"` as the file parameter name.

AscEditor requires `.asy` symbol library files. Platform handling in `server.py:_configure_asc_editor()`:

| Platform | How symbol paths are resolved |
|-|-|
| Windows native | `AscEditor.prepare_for_simulator()` (spicelib built-in) |
| Linux native (LTspice via Wine) | `AscEditor.prepare_for_simulator()` (spicelib handles Wine paths) |
| WSL + LTspice on Windows | `wsl.get_ltspice_lib_paths()` resolves `%LOCALAPPDATA%` via `cmd.exe` |
| Any platform without LTspice | No .asc support (no .asy symbol files available) |

Users can override via `[schematic] symbol_paths` in TOML or `LTSPICE_MCP_SYMBOL_PATHS` env var. `state.asc_editor_available` tracks whether .asc editing is usable.

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
| `full` (default) | All 35 | Any MCP client, automation, non-agent LLMs |
| `agentic` | 21 | LLM agents with native file access (Read/Edit/Write) |

The "agentic" profile removes netlist-editing wrapper tools (e.g., `create_netlist`, `read_circuit`, `set_component_value`, `parameter`, `edit_directive`) and library session management — these are things capable agents do natively. It keeps simulation lifecycle, binary `.raw` parsing, batch orchestration, AscEditor-dependent ops, and library search.

Profile-filtered tool defs and dispatch live on `SessionState` (`state.tool_defs`, `state.tool_dispatch`). Each tool's `profiles` frozenset (set at registration via `@registry.tool(profiles=...)`) determines visibility. Error hints in `server.py` are profile-aware (tuples of `(full_hint, agentic_hint)`) so they don't reference tools the client can't see.
