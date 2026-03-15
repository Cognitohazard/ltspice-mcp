# ltspice-mcp

MCP server that exposes LTspice circuit simulation to LLMs via the [Model Context Protocol](https://modelcontextprotocol.io/). Create netlists, edit schematics, run simulations, and analyze results through MCP tool calls.

Built on the low-level `mcp.server.lowlevel.Server` API with [spicelib](https://github.com/nunobrum/spicelib) as the simulation backend.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- LTspice (for simulation — circuit editing works without it)

### Platform support

| Platform | How LTspice runs |
|-|-|
| Windows | Native |
| WSL2 | Windows LTspice.exe via interop (not Wine) |
| Linux | Via Wine (spicelib handles this) |

## Setup

```bash
git clone https://github.com/Cognitohazard/ltspice-mcp.git
cd ltspice-mcp
uv sync

cp ltspice-mcp.example.toml ltspice-mcp.toml
# Set simulator.path if LTspice isn't auto-detected (required on WSL)
```

### Add to Claude Code

```bash
claude mcp add -s project ltspice -- uv run --directory /path/to/ltspice-mcp ltspice-mcp
```

Or add to `.mcp.json` manually:

```json
{
  "mcpServers": {
    "ltspice": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ltspice-mcp", "ltspice-mcp"]
    }
  }
}
```

### WSL configuration

Set the Windows-side LTspice path in `ltspice-mcp.toml`:

```toml
[simulator]
path = "/mnt/c/Program Files/ADI/LTspice/LTspice.exe"
```

Simulation output is automatically redirected to a Windows temp directory. LTspice writes SQLite `.db` files alongside results, and these fail on UNC paths (`\\wsl.localhost\...`), which causes `.MEAS` data to be lost.

## Tools

All 29 tools are prefixed with `ltspice_` to avoid namespace conflicts with other MCP servers.

### Circuit editing (10 tools)

Work on both `.cir`/`.net` netlists and `.asc` schematics.

| Tool | Description |
|-|-|
| `ltspice_create_netlist` | Create a new netlist from content string |
| `ltspice_read_circuit` | Read and parse a circuit file |
| `ltspice_list_components` | List components, optionally filtered by prefix |
| `ltspice_set_component_value` | Set one or many component values |
| `ltspice_parameter` | Get or set `.PARAM` directive values |
| `ltspice_edit_directive` | Add or remove SPICE directives |
| `ltspice_remove_component` | Remove component from `.asc` schematic |
| `ltspice_move_component` | Move/rotate component in `.asc` schematic |
| `ltspice_set_component_attribute` | Set component attribute (SpiceLine, etc.) |
| `ltspice_export_netlist` | Export `.asc` schematic to `.net` netlist |

### Simulation (3 tools)

| Tool | Description |
|-|-|
| `ltspice_run_simulation` | Run simulation (sync for short, async for long) |
| `ltspice_check_job` | Check job status or list all jobs |
| `ltspice_cancel_job` | Cancel a running simulation |

### Analysis (5 tools)

| Tool | Description |
|-|-|
| `ltspice_get_signal_stats` | Signal statistics (min, max, mean, RMS, dB/phase for AC) |
| `ltspice_query_value` | Query signal value at a specific time or frequency |
| `ltspice_get_measurements` | Extract `.MEAS` results from log file |
| `ltspice_get_operating_point` | DC operating point (node voltages, branch currents) |
| `ltspice_get_simulation_summary` | Full summary: signals, measurements, bandwidth, warnings |

### Parametric analysis (5 tools)

| Tool | Description |
|-|-|
| `ltspice_configure_sweep` | Configure multi-parameter sweep |
| `ltspice_run_sweep` | Execute sweep (async, returns job ID) |
| `ltspice_configure_montecarlo` | Configure Monte Carlo with component tolerances |
| `ltspice_run_montecarlo` | Execute Monte Carlo analysis (async) |
| `ltspice_get_batch_results` | Query sweep/MC status, statistics, or per-run data |

### Library management (5 tools)

| Tool | Description |
|-|-|
| `ltspice_search_library` | Search models/subcircuits by name |
| `ltspice_get_model_info` | Get model details and `.include` directive |
| `ltspice_load_library` | Load `.lib`/`.mod` file or directory |
| `ltspice_unload_library` | Unload a library from the session |
| `ltspice_list_libraries` | List loaded libraries |

### Status (1 tool)

| Tool | Description |
|-|-|
| `ltspice_get_server_status` | Server status: simulators, config, sandbox, runtime state |

## Resources

| URI | Description |
|-|-|
| `ltspice://netlists/` | List netlist files in working directory |
| `ltspice://netlists/{filename}` | Read a specific netlist file |
| `ltspice://results/` | List simulation jobs and results |
| `ltspice://results/{job_id}/signals` | Signal data for a completed job |
| `ltspice://results/{job_id}/measurements` | Measurement data for a completed job |
| `ltspice://models/` | List available model libraries |
| `ltspice://config` | Current server configuration |

## Prompts

Guided workflow prompts for common circuit design tasks:

- **`filter_design`** — Design filters (lowpass, highpass, bandpass, bandstop, allpass) with topology and order selection
- **`amplifier_analysis`** — Analyze amplifier circuits (common-emitter, common-source, op-amp) for bias, gain, and stability
- **`tolerance_analysis`** — Monte Carlo yield estimation with component tolerances
- **`simulation_debugging`** — Diagnose simulation errors (convergence, singular matrix, missing models)

## Configuration

Copy `ltspice-mcp.example.toml` to `ltspice-mcp.toml` and edit. All settings can be overridden with `LTSPICE_MCP_*` environment variables.

```toml
[simulator]
default = "ltspice"            # ltspice, ngspice, qspice, xyce
path = ""                      # Explicit executable path (required on WSL)

[security]
allowed_paths = ["."]          # Sandbox: only these directories are accessible

[simulation]
max_parallel = 4               # Concurrent simulation limit
timeout = 300.0                # Default timeout in seconds

[analysis]
max_points = 10000             # Max waveform data points per trace

[logging]
level = "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Architecture

```
MCP Protocol    server.py         — lifespan, dispatch, request routing
                resources.py      — MCP resources
                prompts.py        — MCP prompts

Tools           tools/circuit.py     — netlist and schematic editing
                tools/simulation.py  — simulation execution and job management
                tools/analysis.py    — waveform analysis and measurements
                tools/advanced.py    — parametric sweep and Monte Carlo
                tools/library.py     — component library management
                tools/status.py      — server diagnostics

Core            lib/sim_runner.py        — spicelib SimRunner async integration
                lib/sweep_runner.py      — parametric sweep execution
                lib/montecarlo_runner.py — Monte Carlo execution
                lib/raw_parser.py        — .raw file parsing and statistics
                lib/log_parser.py        — .log parsing (errors, measurements, Fourier)
                lib/ltspice_wsl.py       — WSL-aware LTspice subclass
                lib/wsl.py               — WSL detection and path conversion
                lib/simulator.py         — simulator detection and selection
                lib/library_manager.py   — SPICE model library management

Config          config.py  — TOML + env var configuration
                state.py   — session state (jobs, editors, caches)
                errors.py  — structured error hierarchy
```

### Design notes

- **Async wrapping**: All spicelib operations are synchronous. They run in `asyncio.to_thread()` via `run_sync()` to avoid blocking the event loop.
- **Path sandbox**: User-provided paths are validated against `config.allowed_paths`. Paths outside the sandbox raise `PathSecurityError`.
- **stdin protection**: `main.py` redirects fd 0 to `/dev/null` before starting the server, passing the real stdin only to the MCP transport. This prevents subprocesses from consuming MCP protocol bytes — a workaround for [python-sdk#671](https://github.com/modelcontextprotocol/python-sdk/issues/671).
- **Tool annotations**: Every tool declares `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint` for client auto-approval decisions.

## Development

```bash
uv sync                        # Install dependencies
uv run pytest tests/ -v        # Run tests
uv run pyright                 # Type checking
uv run ltspice-mcp             # Run server (stdio)
```

## License

MIT
