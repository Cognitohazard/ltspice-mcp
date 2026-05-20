# ltspice-mcp — Design

Scope, architecture, and the design principles you'll encounter when
using the server.

For install + client setup see [README.md](../README.md). For the parser
architecture see [docs/spice_lex.md](spice_lex.md). For current bugs and
limitations, check the GitHub issue tracker.

## Scope

The server targets **structured, validated, geometry-aware,
LTspice-specific** operations. Each adjective constrains what the
project invests in:

- **Structured** — tool outputs declare schemas; results are typed, not
  free text.
- **Validated** — mutating tools refuse invalid states before touching
  the file.
- **Geometry-aware** — `.asc` schematic operations work in symbol
  coordinates, not just netlist tokens.
- **LTspice-specific** — `.asc`/`.asy`/`.raw` quirks and Windows
  interop are first-class. Other simulators are supported through
  spicelib's common interface but treated as secondary.

### Why an MCP server vs. asking the LLM to use spicelib directly

An LLM with code execution (Claude Code, Cursor) could write spicelib
scripts directly. MCP applies in specific contexts:

- **Works without code execution.** Claude Desktop, ChatGPT, web chat
  clients have no shell or Python. MCP tools are the only way to give
  them simulation capabilities. This is the primary audience.
- **Reliability.** `run_simulation(netlist="foo.cir")` is a tested code
  path. LLM-generated spicelib code makes import mistakes, calls wrong
  method names, forgets `save_netlist()`, etc.
- **Context efficiency.** A tool call is ~50 tokens; equivalent Python is
  20-40 lines. Over an iterative design session this compounds.
- **Images inline.** MCP returns base64 PNG in the protocol — Bode plots
  and waveforms appear in the conversation, not as files the model can't
  see.
- **No user setup.** Install the server once. No spicelib in the user's
  project venv, no Python knowledge required.

Where MCP is weaker: a fixed tool set is less flexible than arbitrary
Python, and there's an extra process to maintain.

## Comparison with alternatives

|tool|schematic editing|geometry validation|cross-run analysis|WSL|
|-|-|-|-|-|
|`ltspice-mcp` (this)|`.asc` first-class|pin coords, bbox, diagonal-wire refusal, named-net-short detection|sweep + Monte Carlo + batch metrics|first-class|
|spicelib `AscEditor`|text-level mutator|none|via SimStepper / Montecarlo (lower-level)|N/A|
|PyLTSpice|spicelib re-export|none|same as spicelib|N/A|
|`xuio/ltspice-mcp`|create/modify/lint|not documented|not documented|macOS-only|
|`daviditkin/ltspice-mcp`|netlist-level (9 tools)|none|none|not documented|
|SPICEAssistant (arxiv 2507.10639)|none|N/A|measurement extractors|N/A — research only|
|LTspice GUI|interactive|interactive|GUI-driven|N/A|

Geometry-aware editing tools — `ltspice_connect`, `ltspice_add_component`,
`ltspice_move_component`, `ltspice_add_net_label` (with `pin="M3.S"`),
`ltspice_apply_schematic_ops`, `ltspice_symbol_info`,
`ltspice_component_info` — validate against pin coordinates, bounding
boxes, and named-net topology. The validation refuses diagonal wires,
pin collisions, wire-junction overlaps, and named-net shorts before
touching the file, and returns structured state the agent can use in
its next call.

## Design principles

### Validate before write

Every mutating tool refuses invalid states before modifying the file.
`ltspice_connect` is the model: it returns a structured rejection payload
naming the specific segments, pins, or labels that blocked the write so
the agent can pick a new waypoint instead of guessing.

Agents are bad at undoing mistakes. Tools that refuse invalid states save
entire conversation turns of recovery.

### Refusal, not auto-routing

When a wire is refused, the server returns the **conflict set** — what
blocked it and where — but does **not** try to auto-route around the
obstacle. Routing is NP-hard; the conflict-set form is almost as useful
at a fraction of the cost, and keeps the agent in control of the
decision.

### `dry_run` is scoped, not universal

`dry_run` / preview is only on ops that are hard to undo and multi-step:
`apply_schematic_ops`, `move_component`, `connect`, `add_component`,
`edit_directive remove`. Atomic, trivially reversible ops
(`set_component_value`, `parameter`) don't need preview — adding it
would be API surface without value.

### Post-op validation pass

Even when a single op succeeds, schematic-level warnings (floating pins,
duplicate wire segments, dangling net labels) ride along in the
response. Cheap to compute during the same editor session, and saves
the agent a follow-up inspection turn.

### Structured outputs

Tools that produce inspectable state declare an `output_schema` and
return `structuredContent` alongside text. When MCP clients run code
execution against tool calls, the schemas serve as the types the
sandboxed code consumes.

## Architecture

```
MCP protocol layer    server.py — lifespan, dispatch, request routing
                      resources.py — MCP resources & URI templates
Tool layer            tools/*.py — tool definitions + handlers
Core logic            lib/*.py
Config / state        config.py, state.py, errors.py
```

Tool modules use a decorator-based registry (`@registry.tool(...)`).
Each registration declares `name`, `description`, `input_model` (Pydantic),
`annotations`, `profiles` (which tool profiles expose this tool), and an
optional `output_schema`. `SessionState.create()` filters tools by
profile during lifespan init; the dispatch table and tool definitions
both come from `registry`.

Key `lib/` modules:

|module|purpose|
|-|-|
|`services.py`|service layer shared by tools and resources — job resolution, cached result loading, reusable extraction|
|`sim_runner.py`, `sweep_runner.py`, `montecarlo_runner.py`|spicelib runner wrappers|
|`runner_manager.py`|centralized runner lifecycle with auto-invalidation on loop / simulator / output-folder change|
|`simulator.py`|simulator detection, WSL/Wine selection|
|`ltspice_wsl.py`, `wsl.py`|WSL path conversion and Windows interop|
|`raw_parser.py`, `log_parser.py`|simulation result parsing|
|`library_manager.py`, `library_parser.py`|component library handling|
|`symbol_geometry.py`|`.asy` symbol parsing, pin positions, rotation transforms, bounding boxes|
|`spice_lex/`|shared SPICE tokenizer — see [docs/spice_lex.md](spice_lex.md)|
|`pathutil.py`|path security (`safe_path()`, `resolve_safe_path()`)|

### Tool profiles

`config.tool_profile` controls which tools are exposed.

|profile|tool count|use case|
|-|-|-|
|`full` (default)|49|Claude Desktop, ChatGPT, web chat clients, non-agent LLMs, automation|
|`agentic`|33|Claude Code, Cursor, Windsurf, and other agents with native `Read`/`Edit`/`Write`|

The `agentic` profile drops netlist-editing wrappers (`create_netlist`,
`read_circuit`, `set_component_value`, `parameter`, `edit_directive`)
and library session-management tools — things a capable agent does
natively via filesystem access. It keeps simulation lifecycle, binary
`.raw` parsing, batch orchestration, AscEditor-dependent ops, and
library search.

Set via `[tools] profile` in `ltspice-mcp.toml` or the
`LTSPICE_MCP_TOOL_PROFILE` env var. Error hints adapt to the active
profile so they never point at unavailable tools.

## Backend: spicelib

`spicelib` (>= 1.4.7) is the core Python library for SPICE automation,
by Nuno Brum. PyLTSpice is a thin re-export wrapper over spicelib that
adds nothing — we depend on `spicelib` directly.

All four simulators share the same base `Simulator` ABC:

|simulator|class|platform|
|-|-|-|
|LTspice|`LTspice`|Windows native; Linux/macOS via Wine; WSL via Windows interop|
|NGspice|`NGspiceSimulator`|Linux/macOS/Windows native|
|QSPICE|`Qspice`|Windows; Wine limited|
|Xyce|`XyceSimulator`|Linux/Windows native|

Core spicelib components used:

|component|purpose|
|-|-|
|`SpiceEditor`|read/modify/write `.net`/`.cir` netlists|
|`AscEditor`|read/modify `.asc` schematics (LTspice)|
|`SimRunner`|batch execution with parallel-sim support|
|`RawRead`|parse binary `.raw`/`.qraw` waveform output (dialect auto-detection)|
|`LTSpiceLogReader`|extract `.MEAS`, step data, Fourier|
|`SimStepper`|multi-dimensional parameter sweeps (overcomes the 3-parameter `.STEP` limit)|
|`Montecarlo`|statistical tolerance analysis|
|`WorstCaseAnalysis`|exhaustive min/max enumeration|

Per-simulator notes:

- **LTspice** supports `.asc` → `.net` conversion via `create_netlist()`.
  macOS LTspice has no CLI switch support. Default switches: `-Run -b`.
- **NGspice** has a compatibility mode (`kiltpsa` default for
  KiCad/LTspice/PSPICE). Default switches: `-b -o -r -a`. Native on Linux.
- **QSPICE** uses `.qraw` (double precision). Windows-only, limited Wine.
- **Xyce** supports `-syntax` and `-norun` for validation without
  simulation.

## WSL support

On WSL, LTspice runs via Windows interop, not Wine. Operational
adaptations:

- **Path conversion**: `LTspiceWSL` subclass (`lib/ltspice_wsl.py`)
  overrides `run()` to convert paths via `wslpath` instead of Wine's
  `Z:` prefix. Auto-selected when `is_wsl()` is True.
- **`%LOCALAPPDATA%` resolution**: symbol-library paths under
  `%LOCALAPPDATA%` are resolved via `cmd.exe` since they vary by
  Windows user profile.
- **Windows-side output dir**: simulation output goes to a Windows-native
  temp dir when the working dir is on the Linux filesystem. Required for
  `.MEAS` results — LTspice's SQLite `.db` writes fail on UNC paths
  (`\\wsl.localhost\...`), which loses measurement data from `.log`
  files.
- **Extension preservation**: LTspice requires netlist files to have an
  extension (`.cir`, `.net`, `.sp`). `sim_runner.py` preserves the
  original extension in `run_filename`.

`simulator.path` in `ltspice-mcp.toml` must point at the
Windows-side executable (e.g.
`/mnt/c/Program Files/ADI/LTspice/LTspice.exe`) — spicelib can't
auto-detect across the WSL boundary.

## Diagnostics

`log_parser.py:extract_log_diagnostics()` extracts structured warnings
and errors from LTspice log files — parse errors with caret pointers,
fatal errors, convergence messages — and surfaces them through
`run_simulation`, `check_job`, `get_measurements`, and
`get_simulation_summary`. An agent driving raw `ngspice -b` gets a
200-line log dump and has to grep; here, the same failure surfaces as
typed feedback the agent can act on.

Common patterns:

|log pattern|surfaced as|suggested action|
|-|-|-|
|"Time step too small"|`ConvergenceError`|increase max timestep, add `.options`, simplify model|
|"Singular matrix"|`SingularMatrixError`|check for floating nodes, verify ground connection|
|"Unknown subcircuit"|`MissingModelError`|report missing model name, suggest alternatives|
|"Can't find"|`MissingModelError`|report missing file, check include paths|
|"Syntax error"|`NetlistError`|report line number and content|
|"Analysis: interrupted"|`SimulationError`|timeout or user abort|

**Scope:** diagnostic taxonomy depth is LTspice-only. ngspice / qspice /
xyce have different log formats — secondary simulators get
convergence and fatal-error parsing, not the full taxonomy. Expanding
to per-simulator parity would fragment maintenance for marginal value.

## Non-goals

These are intentional gaps, not pending features. Adopt accordingly.

- **Auto-routing / auto-placement.** Conflict-set returns on refusal
  are the substitute. Routing is a research project, not a tool.
- **Semantic part search** ("find me a low-Vgs-th NMOS under 100mΩ
  Rds(on)"). Requires parsing every `.model` card, normalizing units
  across vendors, building a queryable parameter index — a separate
  parameter-database effort.
- **Thin wrappers around file-edit operations.** The `agentic` profile
  already drops these for clients with native `Read`/`Edit`.
- **Per-simulator parity.** This is an LTspice product. NGspice,
  QSPICE, and Xyce are supported through spicelib's common interface,
  but LTspice-specific features (`.asc`, `.asy`, Windows interop,
  `.raw` quirks) are first-class. Secondary simulators get best-effort
  support.
- **Auth / multi-tenancy / remote-MCP plumbing.** Local-tool territory.
  No OAuth flows, no per-tenant isolation.
- **Generic cross-simulator EDA features** that PyLTSpice and ngspice's
  CLI already cover well.

## Roadmap

Forward-looking and **not commitments**. Direction reflects the same
spec at the top of this doc — investment compounds where structured +
validated + geometry-aware + LTspice-specific overlap.

- **Layout primitives**: `route_bus`, `align`, `distribute` for the
  rows of caps/resistors agents constantly need to place.
- **`ltspice-mcp doctor`**: one-shot diagnostic that checks simulator
  detection, symbol-path resolution, output-dir placement, Wine vs WSL
  selection, `.MEAS`-on-UNC risk. The `flutter doctor` / `brew doctor`
  pattern.
- **Diagnostic taxonomy expansion**: singular matrix, gmin stepping
  suggestions, timestep too small, missing model cards, unmatched
  subcircuit pins, `.STEP` parameter mismatches. Each gets a
  `suggested_fix` field. LTspice-scoped.
- **Cross-run analysis**: `compare_corners`, `find_worst_case`,
  `sensitivity_ranking` — tools that aggregate measurements across a
  set of runs and return structured deltas.
- **Pin-compatible alternate suggestions** for unknown parts.
- **`schematic` profile** (geometry tools only, no simulation) for
  layout-focused agents — pending a real user request, since each
  profile is a test-matrix axis.

## Configuration

`ltspice-mcp.toml` in the working directory (auto-generated if missing).
Environment variables with `LTSPICE_MCP_` prefix override TOML values.
See `config.py:ServerConfig` for all options.

TOML sections: `[simulator]`, `[security]`, `[simulation]`, `[analysis]`,
`[plotting]`, `[logging]`, `[schematic]`, `[tools]`.

## Verification recipe

End-to-end smoke test for a fresh install:

1. Create RC lowpass netlist (R=1k, C=100n → fc ~1.59kHz)
2. Add `.ac dec 100 1 1Meg` directive
3. Run AC simulation
4. Get Bode plot — verify -3dB point near 1.59kHz
5. Change R to 10k (fc → ~159Hz)
6. Re-simulate, re-plot — verify bandwidth shifted
7. Load a custom library, use a component from it in a new circuit

If any step fails, the doctor tool (see Roadmap) is the eventual answer;
until it ships, check `simulator.path` in TOML and confirm
`ltspice-mcp.toml`'s `[security] allowed_paths` includes your working
directory.
