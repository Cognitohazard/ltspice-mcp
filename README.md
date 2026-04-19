# ltspice-mcp

> **Work in progress.** Core functionality is usable but expect rough edges, missing features, and breaking changes.

MCP server that exposes LTspice circuit simulation to LLMs via the [Model Context Protocol](https://modelcontextprotocol.io/). Create netlists, edit schematics, run simulations, and analyze results through MCP tool calls.

Built on the low-level `mcp.server.lowlevel.Server` API with [spicelib](https://github.com/nunobrum/spicelib) for simulator interfacing and result parsing.

## Requirements

- Python 3.13+
- LTspice (for simulation — circuit editing works without it)
- An MCP-capable client (see [MCP clients](#mcp-clients) below)

### Platform support

| Platform | How LTspice runs |
|-|-|
| Windows | Native |
| WSL2 | Windows LTspice.exe via interop (not Wine) |
| Linux | Via Wine (spicelib handles this) |

## Install

Any Python package installer works — pick whichever you already use.

```bash
# uv (recommended — fast, installs in an isolated tool env and puts `ltspice-mcp` on PATH)
uv tool install ltspice-mcp

# or plain pip (inside a venv or with --user)
pip install ltspice-mcp

# or pipx (same isolated-tool model as `uv tool`, if you prefer PyPA tooling)
pipx install ltspice-mcp
```

After install, the `ltspice-mcp` command should be on your PATH. Verify with:

```bash
ltspice-mcp --help
```

### Install from source

```bash
git clone https://github.com/Cognitohazard/ltspice-mcp.git
cd ltspice-mcp
uv sync   # or: pip install -e .
```

### Configure

```bash
cp ltspice-mcp.example.toml ltspice-mcp.toml
# Set simulator.path if LTspice isn't auto-detected (required on WSL)
```

## MCP clients

MCP is an [open standard](https://modelcontextprotocol.io/), so this server works with any client that speaks it. Below are setup snippets for the major ones. All examples assume you've already installed `ltspice-mcp` so the executable is on your `PATH`.

> **Transport:** `ltspice-mcp` speaks stdio. Local desktop clients (Claude Desktop, Claude Code, Cursor, Windsurf, Gemini CLI, etc.) connect to stdio servers directly. Web clients that only accept remote MCP (claude.ai, ChatGPT) need a stdio→HTTP bridge — see [Remote / web clients](#remote--web-clients).

### Generic stdio config

Every client below uses the same JSON snippet — only the config file path differs. Paste this into the client's MCP config file:

```json
{
  "mcpServers": {
    "ltspice": {
      "command": "ltspice-mcp",
      "args": []
    }
  }
}
```

If `ltspice-mcp` isn't on the client's `PATH` (common on macOS GUI apps), use the absolute path — find it with `which ltspice-mcp` (Linux/macOS) or `where ltspice-mcp` (Windows).

### Claude Code (Anthropic CLI)

```bash
claude mcp add -s project ltspice -- ltspice-mcp
```

This writes a `.mcp.json` in the project root. Use `-s user` to install globally for your user instead.

### Claude Desktop

Paste the [generic stdio config](#generic-stdio-config) into:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Restart Claude Desktop. The LTspice tools appear under the MCP icon in the input box.

### Cursor

Paste the [generic stdio config](#generic-stdio-config) into `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in a project.

### Windsurf (Codeium)

Paste the [generic stdio config](#generic-stdio-config) into `~/.codeium/windsurf/mcp_config.json`.

### Gemini CLI

Merge the [generic stdio config](#generic-stdio-config) into the `mcpServers` object in `~/.gemini/settings.json`.

### Continue, Cline, Zed, and other clients

Any client that supports MCP stdio servers takes the same [generic stdio config](#generic-stdio-config) — check the client's docs for where its config file lives.

### Remote / web clients

**claude.ai** (Pro/Max "Connectors") and **ChatGPT** (Developer Mode connectors) only accept remote MCP servers over HTTPS, not local stdio processes. To use `ltspice-mcp` with them you need to bridge stdio to HTTP/SSE and expose it at a public URL. Common approaches:

- [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) — wraps a stdio server as an SSE endpoint
- Run behind a reverse tunnel (Cloudflare Tunnel, ngrok, Tailscale Funnel)

Because this server can write files, spawn simulators, and read arbitrary paths under `allowed_paths`, **only expose it on a network you fully control**. For remote use cases, tighten `[security] allowed_paths` and keep the tunnel private.

## WSL configuration

On WSL, LTspice.exe runs via Windows interop (not Wine), so spicelib can't auto-detect it across the WSL boundary. Set the Windows-side path explicitly in `ltspice-mcp.toml`:

```toml
[simulator]
path = "/mnt/c/Program Files/ADI/LTspice/LTspice.exe"
```

Simulation output is automatically redirected to a Windows temp directory. LTspice writes SQLite `.db` files alongside results, and these fail on UNC paths (`\\wsl.localhost\...`), which causes `.MEAS` data to be lost.

## Tool Profiles

`tool_profile` controls which tools the server exposes. Set via `[tools] profile` in TOML or `LTSPICE_MCP_TOOL_PROFILE` env var.

| Profile | Tools | Use case |
|-|-|-|
| `full` (default) | All 35 | Backwards-compatible default for any MCP client |
| `agentic` | 21 | **Recommended** when your client supports skill files |

The **agentic** profile removes netlist-editing wrappers, sweep/MC configuration tools, niche schematic operations, and library session management — things a capable LLM agent does better through direct file editing. It keeps simulation lifecycle, binary `.raw` parsing, batch orchestration, AscEditor-dependent ops, and library search — the tools that provide genuine leverage over what an LLM can do natively.

```toml
[tools]
profile = "agentic"
```

## Skills

The tools give LLMs *access* to simulation. The skills give them the *expertise* to use it well.

The `skills/` directory contains structured domain knowledge that teaches LLMs how to work with SPICE circuit simulators. Each skill is a self-contained reference covering SPICE fundamentals, simulator-specific gotchas, and MCP tool workflows — the kind of knowledge that takes years to build and that LLMs don't reliably have.

| Skill | Description |
|-|-|
| `skills/ltspice/SKILL.md` | LTspice: `.asc` schematic editing, Windows/WSL paths, LTspice-specific directives |
| `skills/ngspice/SKILL.md` | ngspice: open-source workflow, control scripts, ngspice model compatibility |

Copy the relevant skill into whatever location your MCP client uses for persistent instructions, then set the `agentic` profile. The skill provides the domain knowledge that replaces the removed wrapper tools.

## Tools

All 35 tools are prefixed with `ltspice_` to avoid namespace conflicts with other MCP servers. Every tool declares MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) for client auto-approval decisions.

### Circuit editing (16 tools)

Work on both `.cir`/`.net` netlists and `.asc` schematics. Extension-based dispatch picks the right editor automatically.

| Tool | Description |
|-|-|
| `ltspice_create_netlist` | Create a new netlist from a content string |
| `ltspice_read_circuit` | Read and parse a circuit file (netlist text for `.cir`, schematic layout for `.asc`) |
| `ltspice_list_components` | List components (optionally filtered by prefix) or look up a single component by reference |
| `ltspice_set_component_value` | Set one component value, or batch-set many via a `values` dict |
| `ltspice_parameter` | Get all `.PARAM` values (no args) or set one (`name` + `value`) |
| `ltspice_edit_directive` | Add or remove SPICE directives (`.tran`, `.ac`, `.lib`, etc.) |
| `ltspice_remove_component` | Remove a component (warns about orphaned wires) |
| `ltspice_move_component` | Move or rotate a component in an `.asc` schematic |
| `ltspice_set_component_attribute` | Set a component attribute (SpiceLine, Value2, etc.) |
| `ltspice_add_component` | Add a component with value, attributes, returns pin positions and bounding box |
| `ltspice_connect` | Connect two pins by reference with waypoint routing; validates for pin collisions, wire junctions, and diagonal wires before adding |
| `ltspice_add_net_label` | Add/remove net labels and ground flags (supports pin reference for placement) |
| `ltspice_add_text` | Add comment text annotations to schematic |
| `ltspice_get_symbol_info` | Get symbol pin positions, bounding box, pin directions, and description |
| `ltspice_get_component_info` | Get placed component pin positions, bounding box, and attributes |
| `ltspice_export_netlist` | Export `.asc` to `.net` netlist (shows diff against previous export) |

`.asc` schematic editing requires `.asy` symbol files. These are auto-detected on Windows and WSL; override with `[schematic] symbol_paths` in TOML or the `LTSPICE_MCP_SYMBOL_PATHS` env var.

### Simulation (3 tools)

| Tool | Description |
|-|-|
| `ltspice_run_simulation` | Run a simulation — sync for short runs, async (returns job ID) for long ones |
| `ltspice_check_job` | Check a job's status by ID, or list all jobs |
| `ltspice_cancel_job` | Cancel a running simulation |

### Analysis (5 tools)

| Tool | Description |
|-|-|
| `ltspice_get_signal_stats` | Signal statistics: min, max, mean, RMS, peak-to-peak (dB/phase for AC) |
| `ltspice_query_value` | Query a signal's value at a specific time or frequency |
| `ltspice_get_measurements` | Extract `.MEAS` directive results from the simulation log |
| `ltspice_get_operating_point` | DC operating point: all node voltages and branch currents |
| `ltspice_get_simulation_summary` | Full summary: simulation type, signals, measurements, warnings |

### Parametric analysis (5 tools)

| Tool | Description |
|-|-|
| `ltspice_configure_sweep` | Configure a multi-parameter sweep (linear or log, by step size or point count) |
| `ltspice_run_sweep` | Execute a configured sweep (async, returns job ID) |
| `ltspice_configure_montecarlo` | Configure Monte Carlo analysis with per-type component tolerances |
| `ltspice_run_montecarlo` | Execute a configured Monte Carlo analysis (async, returns job ID) |
| `ltspice_get_batch_results` | Query sweep/MC job progress, per-signal statistics, or per-run data |

### Library management (5 tools)

| Tool | Description |
|-|-|
| `ltspice_find_model` | Find model candidates by name (fuzzy by default; `exact=true` for exact case-insensitive match) |
| `ltspice_get_model_info` | Get model parameters and the `.include` directive to use it |
| `ltspice_load_library` | Load a `.lib`/`.mod` file or a directory of library files |
| `ltspice_unload_library` | Unload a previously loaded library |
| `ltspice_list_libraries` | List loaded libraries, optionally with their model names |

### Status (1 tool)

| Tool | Description |
|-|-|
| `ltspice_get_server_status` | Server status: detected simulators, config, sandbox paths, runtime state |

## Resources

Static resources and URI templates for browsing circuit files and simulation results.

| URI | Description |
|-|-|
| `ltspice://netlists/` | List netlist files in the working directory |
| `ltspice://netlists/{filename}` | Read the full text of a specific netlist |
| `ltspice://results/` | List all simulation and batch jobs with their status |
| `ltspice://results/{job_id}/signals` | Signal/trace names from a completed job's `.raw` file |
| `ltspice://results/{job_id}/measurements` | `.MEAS` results from a completed job's log |
| `ltspice://models/` | List loaded model libraries and their models |
| `ltspice://config` | Current server configuration and detected simulators |

## Configuration

Copy `ltspice-mcp.example.toml` to `ltspice-mcp.toml` and edit. All settings can be overridden with `LTSPICE_MCP_` prefixed environment variables (highest precedence). The config file path itself can be set with `--config PATH` or the `LTSPICE_MCP_CONFIG` env var.

```toml
[simulator]
default = "ltspice"            # ltspice, ngspice, qspice, xyce (null = auto-detect)
path = ""                      # Explicit executable path (required on WSL)

[security]
allowed_paths = ["."]          # Sandbox: only these directories are accessible

[simulation]
max_parallel = 4               # Concurrent simulation limit
timeout = 300.0                # Default timeout in seconds

[analysis]
max_points = 10000             # Max waveform data points per trace

[plotting]
dpi = 150                      # Plot resolution
style = "seaborn-v0_8-darkgrid"  # Matplotlib style

[schematic]
symbol_paths = []              # Custom .asy symbol paths (auto-detected on Windows/WSL)

[logging]
level = "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Architecture

```
MCP Protocol    server.py         — lifespan, dispatch, request routing
                resources.py      — MCP resources and URI templates

Tools           tools/circuit.py     — netlist and schematic editing
                tools/simulation.py  — simulation execution and job management
                tools/analysis.py    — waveform analysis and measurements
                tools/advanced.py    — parametric sweep and Monte Carlo
                tools/library.py     — component library management
                tools/status.py      — server diagnostics

Core            lib/sim_runner.py        — spicelib SimRunner async integration
                lib/sweep_runner.py      — parametric sweep execution
                lib/montecarlo_runner.py — Monte Carlo execution
                lib/runner_manager.py    — centralized runner lifecycle and caching
                lib/raw_parser.py        — .raw file parsing and statistics
                lib/log_parser.py        — .log parsing (errors, measurements, Fourier)
                lib/batch_results.py     — sweep/MC batch result extraction
                lib/ltspice_wsl.py       — WSL-aware LTspice subclass
                lib/wsl.py               — WSL detection and path conversion
                lib/simulator.py         — simulator detection and selection
                lib/library_manager.py   — SPICE model library management
                lib/library_parser.py    — .lib/.mod file parsing
                lib/cache.py             — FileCache for editor and result instances
                lib/pathutil.py          — path security (safe_path, resolve_safe_path)
                lib/format.py            — output formatting helpers
                lib/plotting.py          — matplotlib plot generation
                lib/sweep_utils.py       — sweep parameter utilities

Config          config.py  — TOML + env var configuration
                state.py   — session state (jobs, editors, caches)
                errors.py  — structured error hierarchy
```

### Design notes

- **Async wrapping**: spicelib is synchronous. Short-lived parser/editor calls run inline on the event loop (the MCP stdio transport processes one request at a time). Long-lived simulator work uses `asyncio.to_thread()` inside the runner layer (`sim_runner`, `sweep_runner`, `montecarlo_runner`).
- **Path sandbox**: User-provided paths are validated against `config.allowed_paths`. Paths outside the sandbox raise `PathSecurityError`.
- **Runner lifecycle**: `RunnerManager` owns all runner instances (sim, sweep, MC). It auto-invalidates cached runners when the event loop, simulator class, or output folder changes. Runners are never created directly.
- **stdin protection**: `main.py` redirects fd 0 to `/dev/null` before starting the server, passing the real stdin only to the MCP transport. This prevents subprocesses from consuming MCP protocol bytes — a workaround for [python-sdk#671](https://github.com/modelcontextprotocol/python-sdk/issues/671).
- **Tool annotations**: Every tool declares `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint` for client auto-approval decisions.
- **Output schemas**: All 19 data-returning tools declare `outputSchema` for client introspection of `structuredContent`. The remaining 16 text-only confirmation tools don't set `structuredContent` and omit `outputSchema`.
- **Structured errors**: Typed error hierarchy (`PathSecurityError`, `NetlistError`, `SimulationError` variants) in `errors.py`. Handlers catch `LTSpiceMCPError` subtypes and return error text; unknown exceptions propagate to the MCP SDK.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dev dependency management because it resolves the PEP 735 `dependency-groups` in `pyproject.toml` natively. Plain `pip` (25.1+) also supports this via `--group`.

```bash
# With uv (recommended for development)
uv sync                        # Install runtime + dev dependencies
uv run pytest tests/ -v        # Run tests
uv run pyright                 # Type checking
uv run ruff check src/ tests/  # Lint
uv run ltspice-mcp             # Run server (stdio)
uv run ltspice-mcp --config /path/to/config.toml  # Custom config

# With pip 25.1+
pip install -e . --group dev
pytest tests/ -v
pyright
ltspice-mcp
```

## License

GPL-3.0
