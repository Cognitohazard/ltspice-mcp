# ltspice-mcp

> **Work in progress.** Core functionality is usable but expect rough edges and breaking changes.

An MCP server that connects LLM assistants (Claude, and any other MCP client) to real circuit simulation: LTspice and ngspice, plus direct editing of LTspice `.asc` schematics. Simulation results come back as structured numbers — cutoff frequencies, overshoot, phase margin, rise times, and per-device small-signal internals (gm, gds, vth, …) read back **by name** — so the assistant can design, verify, and iterate on circuits in the same files you open in LTspice, without ever hand-parsing a rawfile. Built on [spicelib](https://github.com/nunobrum/spicelib).

## Quick start

1. Install the server:

```bash
uv tool install ltspice-mcp     # or: pip install ltspice-mcp / pipx install ltspice-mcp
```

2. Add it to your MCP client:

```bash
# Claude Code
claude mcp add -s project ltspice -- ltspice-mcp
```

```json
// Claude Desktop (claude_desktop_config.json) and most other clients
{
  "mcpServers": {
    "ltspice": { "command": "ltspice-mcp", "args": [] }
  }
}
```

3. Have LTspice or ngspice installed. Both are auto-detected on Windows, Linux, and macOS; on WSL the LTspice path must be set explicitly ([WSL notes](#configuration)). Circuit editing works with no simulator at all.

That's the setup. Python 3.11+ required. Verify with `ltspice-mcp --help`.

Claude Desktop config lives at `~/Library/Application Support/Claude/` (macOS), `%APPDATA%\Claude\` (Windows), or `~/.config/Claude/` (Linux). Cursor, Windsurf, Gemini CLI, Continue, Cline, Zed and others take the same JSON snippet in their respective config files. Web clients (claude.ai, ChatGPT) need a stdio→HTTP bridge such as [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) — only expose this server on a network you fully control, since it writes files and spawns processes inside `allowed_paths`.

### Install as a Claude Code plugin or Desktop Extension

Two lower-friction routes that wrap the same PyPI package:

- **Claude Code plugin** — `/plugin marketplace add cognitohazard/ltspice-mcp`, then `/plugin install ltspice-mcp`. Registers the server (via `uvx`) and bundles the LTspice/ngspice skills. To test against a local checkout instead, `/plugin marketplace add .`.
- **Claude Desktop extension** — build the `.mcpb` in [`packaging/mcpb/`](packaging/mcpb/) and drag it onto Claude Desktop for a one-click install with a native folder picker for your circuits directory.

Both wrap the server; they still need `uv` and a simulator on the host — they do not bundle LTspice or ngspice.

## Using it

Once connected, you ask for circuit work in plain language. The assistant designs the circuit and decides what to measure; the server runs the simulator, parses the binary output, and hands back the numbers. It reports what the run produced, the simulator's own warnings included, and leaves the call on whether a result is good to you and the assistant.

> **"Bias this NMOS common-source stage into saturation at the target drain current and report gm/ID."**

The assistant writes the netlist — with a `.save` for M1's operating-point internals — runs a `.op` on ngspice, and reads the bias point back by name: VDS against the device's own VDSAT so you can see it's saturated, the drain current, and gm and gds straight from the solver. gm/ID, the figure analog designers size to, follows from gm and ID. If the bias is off, it trims the gate reference or W/L and re-runs, a couple of seconds per pass. (On LTspice the same `.op` returns node voltages and branch currents; per-device internals are an ngspice feature, and they have to be `.save`d to appear.)

Other requests that work the same way:

- *"What's the overshoot and settling time of this regulator's step response?"* — runs a transient analysis and measures both from the waveform, plus rise time, ringing frequency, and the final value.
- *"Run a 200-run Monte Carlo with 5% resistors and tell me the output spread."* — perturbs components per run, simulates the batch, and reports mean, sigma, and worst-case values per measurement.
- *"Sweep the load from 100 Ω to 10 kΩ and find where efficiency drops."* — parameter sweep with per-run results.
- *"Characterize this NMOS: gm and gm/ID vs VGS."* — writes a `.dc Vgs` deck with `.save @m1[gm] @m1[id]`, runs it on ngspice, and returns the gm/ID table as one CSV (no `.control` block, no rawfile parsing).
- *"Find an N-channel power MOSFET for a low-side switch and measure the on-state drop."* — searches the loaded libraries for a part (`find_model`), drops it into a pulsed-gate transient, and reads Vds(on) and load current back from the `.meas` results.
- *"Build this differential pair as a schematic I can open in LTspice."* — places and wires the components into a real `.asc`, with orthogonal routing and pin-collision checks.
- *"Is this loop stable?"* — AC analysis of the loop gain; reports phase and gain margin at every crossover, not just the first.
- *"What's the resonant frequency and Q of this series RLC?"* — runs an AC sweep and reports each peak's center frequency, Q, and −3 dB bandwidth.

**The warning rides with the number it affects.** A simulator like ngspice can print "singular matrix" once, deep in a log you'd never open, then finish the run and write perfectly plausible numbers anyway — read them by hand and nothing looks off. Ask the server for one of those numbers and the buried line comes attached to it, in an `observations` field right next to the value, so the failure surfaces where you're already looking instead of where it's easy to scroll past.

### Co-design on the same files

Everything operates on ordinary LTspice and SPICE files, so the work passes back and forth between you and the assistant instead of living inside a chat:

- Sketch a schematic in LTspice, then hand it over: *"what's the bias point?"*, *"why doesn't the output move?"*, *"add compensation and check the phase margin."*
- Or the reverse: the assistant designs and verifies the circuit and writes the `.asc`; you open it in LTspice, inspect it, and tweak by hand. Your manual edits are simply the file's new state — the assistant picks up from there on the next request.
- Changes can flow either direction mid-design: adjust a value in the GUI and ask for re-verification, or have the assistant sweep a change you're considering before you commit to it.

## What it does

**Simulation and measurement.** Runs LTspice or ngspice and parses the binary output directly. Measurements are computed server-side and returned as numbers: time-domain (rise/fall, overshoot, settling, delay, period/duty/jitter, RMS, THD), frequency-domain (filter cutoffs and roll-off, gain and phase at any frequency, stability margins, resonance peaks with Q, integrated noise), DC operating points, and `.MEAS` directive results including the ones that failed. With ngspice, per-device small-signal internals (`gm`, `gds`, `vth`, …) are returned by name too — `.save` them and read the set across a `.dc` sweep as a gm/ID table with `export_waveform`, or a single bias point with `operating_point` (address them as `m1.gm` / `@m1[gm]`, no rawfile parsing).

**Schematic and netlist editing.** Creates and edits real LTspice `.asc` files — place components, wire pins, label nets — with validation before anything is written: wiring that would collide with a pin, overlap a junction, or run diagonally is refused, and every edit returns warnings about floating pins or dangling labels. A session's edits can be reverted. Plain netlists (`.cir`/`.net`) get the same operations at text level, plus a static validation pass that catches malformed cards before a simulation is spent.

**Sweeps and Monte Carlo.** Multi-dimensional parameter sweeps and Monte Carlo with per-component tolerances, `.MODEL` process variation, and Pelgrom W·L device mismatch. Per-measurement statistics are aggregated across runs, and any single run can be pulled out and analyzed like a standalone simulation.

**Jobs and trust.** Simulations run as cancellable jobs with timeouts and a concurrency cap; long runs return a job ID immediately and job state survives a server restart. Results report facts, not verdicts: a completed run carries the simulator's own warnings, measurements that produced nothing, and extreme node values as structured observations. Judging whether a result is trustworthy is left to the model reading it.

## Supported simulators

| Simulator | Status |
|-|-|
| LTspice | Primary. Windows native, WSL2 (Windows LTspice.exe via interop), Linux via Wine. Required for `.asc` schematic editing (needs `.asy` symbol libraries). |
| ngspice | First-class: simulate, parse, diagnose, analyze. Open-source path with no LTspice install. |
| QSPICE, Xyce | Supported but secondary. |

## Configuration

Works with defaults out of the box. To customize, copy `ltspice-mcp.example.toml` to `ltspice-mcp.toml`; any setting can be overridden with an `LTSPICE_MCP_`-prefixed environment variable, and `--config PATH` or `LTSPICE_MCP_CONFIG` picks the file. Key options:

```toml
[simulator]
default = "ltspice"      # ltspice, ngspice, qspice, xyce (null = auto-detect)
path = ""                # explicit executable path (required on WSL)

[security]
allowed_paths = ["."]    # sandbox: only these directories are accessible

[simulation]
max_parallel = 4
timeout = 300.0          # seconds

[tools]
profile = "full"         # or "agentic"

[state]
persist_jobs = true
```

See [`src/ltspice_mcp/config.py`](src/ltspice_mcp/config.py) for the full option list (`[analysis]`, `[schematic]`, `[logging]`, ...).

<details>
<summary><strong>WSL specifics</strong></summary>

On WSL, LTspice.exe runs via Windows interop (not Wine), and spicelib can't auto-detect it across the WSL boundary. Set the Windows-side path explicitly:

```toml
[simulator]
path = "/mnt/c/Program Files/ADI/LTspice/LTspice.exe"
```

Simulation output is automatically redirected to a Windows temp directory: LTspice's `.MEAS` results go through SQLite `.db` files that fail on UNC paths (`\\wsl.localhost\...`), and without the redirect measurement data silently disappears from the logs.

`.asy` symbol paths for `.asc` editing are auto-detected on Windows and WSL; override with `[schematic] symbol_paths` or `LTSPICE_MCP_SYMBOL_PATHS`.

</details>

### Tool profiles

| Profile | Tools | Use case |
|-|-|-|
| `full` (default) | 49 | Any MCP client, automation, non-agent LLMs |
| `agentic` | 41 | LLM agents with native file access (Read/Edit/Write) |

The `agentic` profile drops netlist-editing wrappers and library session management — work a capable agent does through direct file edits — and keeps simulation lifecycle, binary `.raw` parsing, batch orchestration, and the `.asc` geometry tools. The `skills/` directory (`skills/ltspice/SKILL.md`, `skills/ngspice/SKILL.md`) contains the domain knowledge that pairs with it: copy the relevant skill into your client's persistent-instructions location.

**Where it runs.** The server shells out to a local LTspice/ngspice and reads circuit files from disk, so it must run where the simulator and the files are. Two setups work: a local MCP host (Claude Desktop, Claude Code, Cursor, Gemini CLI, Codex, …) on your own machine, or a browser-based cloud agent whose sandbox can install ngspice and register the server (verified with Claude). LTspice is local-only (a Windows app); ngspice is open-source and works in either place. Consumer web chat with no sandbox has no simulator and no file access, so it can't run this server directly; bridge it to a machine you control (e.g. [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy)) if you want that UI.

## Under the hood: the tool-level loop

What the assistant actually does for "design a 1 kHz RC low-pass and verify it". It writes the netlist (R=1k, C=159.155n → fc = 1 kHz):

```spice
* rc.cir — RC low-pass
V1 in 0 AC 1
R1 in out 1k
C1 out 0 159.155n
.ac dec 50 1 1Meg
.end
```

then drives three tools:

```
validate_netlist(path="rc.cir")
  → OK: directives valid, element arities check out — safe to simulate

run_simulation(netlist="rc.cir")
  → {"job_id": "sim_a3f1", "status": "completed", "raw_file": ".../rc.raw", ...}

bode_metrics(raw_file=".../rc.raw", signal="V(out)", mode="filter")
```

and gets back scalars, not a plot:

```json
{
  "signal": "V(out)",
  "filter_type": "lowpass",
  "passband_gain_db": 0.0,
  "passband_ripple_db": 0.02,
  "cutoff_low_hz": null,
  "cutoff_high_hz": 1000.4,
  "stopband_rejection_db": 59.97,
  "rolloff_slope_db_per_decade": -19.9,
  "estimated_order": 1,
  "warnings": []
}
```

(abridged — the full response also includes passband bounds and transition bandwidth)

Off-target → `set_component_value`, re-run, re-measure. Long simulations return a job ID instead of blocking; `check_job`/`cancel_job` manage them. Job metadata persists in per-circuit sidecars (`{dir}/.ltspice-mcp/jobs/` — add `.ltspice-mcp/` to your `.gitignore`), and MCP resources (`spice://results/...`, `spice://netlists/...`, `spice://config`) expose jobs, signals, measurements, and config for browsing.

<details>
<summary><strong>All 49 tools</strong></summary>

Every tool declares MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`); data-returning tools declare an `outputSchema` for `structuredContent` introspection.

| Tool | Description |
|-|-|
| `create_netlist` | Create a new netlist from a content string |
| `create_schematic` | Create an empty `.asc` ready for incremental editing |
| `read_circuit` | Read a circuit file (netlist text for `.cir`, schematic layout for `.asc`) |
| `list_components` | List components (optional prefix filter) or look up one by reference |
| `set_component_value` | Set one component value, or batch-set many via a `values` dict |
| `parameter` | Read all `.PARAM` values or set one |
| `edit_directive` | Add or remove SPICE directives (`.tran`, `.ac`, `.lib`, ...) |
| `add_component` | Add a component; returns pin positions, bounding box, overlap warnings |
| `connect` | Wire two pins by reference with waypoint routing; validates pin collisions, junctions, diagonals |
| `symbol_info` | Symbol pin positions, directions, bounding box, description |
| `component_info` | Placed component pin positions, bounding box, attributes |
| `export_netlist` | Export `.asc` to `.net` via LTspice (with diff against previous export) |
| `validate_netlist` | Static pre-flight checks on a netlist or schematic before simulation |
| `trace_net` | Every pin/label/wire on a net at a pin / `net:NAME` / `(x,y)`; flags accidental shorts |
| `reset_schematic` | Revert an `.asc` to its pre-edit snapshot from this session |
| `diff_circuit` | Structural diff between two circuit files |
| `apply_schematic_ops` | Apply many `.asc` edits in one transaction; home for the ack-only mutation ops (`move_component`, `remove_component`, `set_component_attribute`, `add_net_label`, `remove_net_label`, `remove_wire`) |
| `run_simulation` | Run a simulation — sync for short runs, async (job ID) for long ones; sets batch flags, handles the ngspice headerless-raw dialect, routes raw/log artifacts, surfaces convergence/timeout errors (no hand-parsing a rawfile) |
| `check_job` | Check a job's status by ID, or list all jobs |
| `cancel_job` | Cancel a running simulation or batch; kills the simulator process(es) |
| `signal_stats` | Min, max, mean, RMS, peak-to-peak (dB/phase for AC) |
| `get_waveform` | Decimated min/max stat-envelope of a signal over a window — see the shape, then re-request a narrower window to zoom |
| `export_waveform` | Full-fidelity CSV egress of one or more signals to disk (all analysis types; tidy/long for `.step`); accepts device internals (`m1.gm`/`@m1[gm]`) — across a `.dc` sweep this is the gm/ID-table read; returns the path to compute on yourself |
| `plot_waveform` | Interactive HTML chart (transient / DC / Bode dual-panel with `ac_structure` corner + non-minimum-phase markers / noise / `.step` overlay) written next to the circuit and opened in your browser; for seeing shape, not measuring |
| `query_value` | Signal value at a specific time/frequency (or a device internal, `m1.gm`/`@m1[gm]`); `step_axis`+`step_value` picks a `.step` run |
| `operating_point` | DC operating point: all node voltages, branch currents, and per-device internals (gm/gds/vth/…, ngspice) addressable as `m1.gm`/`@m1[gm]`; `device=` scopes to one device |
| `simulation_summary` | Full summary: simulation type, signals, measurements, warnings |
| `edge_metrics` | Rise/fall time and slew rate for one transient edge |
| `pulse_response` | Overshoot, undershoot, settling time for a step response |
| `timing_between` | Propagation delay between two transient signals |
| `periodic_metrics` | Period, frequency, duty cycle, jitter of an oscillating signal |
| `thd` | Total harmonic distortion (THD/THD+N) of a periodic transient via FFT; coherent sampling for an exact result; surfaces every condition |
| `measurement_stats` | Aggregate `.MEAS` scalars across a sweep or Monte Carlo run |
| `bode_metrics` | AC/Bode analysis by `mode`: `filter`, `slope`, `point`, `crossing`; `all_steps=true` for per-step results |
| `stability_metrics` | Loop-gain stability: all unity-gain / -180° crossings with per-crossing margins |
| `resonance` | AC peaks with Q factor and -3 dB bandwidth per peak |
| `ac_structure` | Pole/zero structure of an AC response: net order, corner ranges + Q, non-minimum-phase / RHP-zero, transport delay (facts for human review) |
| `noise_integral` | Integrate a `.noise` spectral density to total RMS over a band (`sqrt(∫ density² df)`); reports the band and sample count |
| `configure_sweep` | Configure a multi-parameter sweep (linear or log) |
| `run_sweep` | Execute a configured sweep (async, returns job ID) |
| `configure_montecarlo` | Configure Monte Carlo: tolerances, `.MODEL` variation, Pelgrom mismatch |
| `run_montecarlo` | Execute a configured Monte Carlo analysis (async, returns job ID) |
| `batch_results` | Sweep/MC job progress, per-signal statistics, or per-run data |
| `find_model` | Find model candidates by name (fuzzy by default, `exact=true` for exact) |
| `load_library` | Load a `.lib`/`.mod` file or a directory of libraries |
| `unload_library` | Unload a previously loaded library |
| `list_libraries` | List loaded libraries, optionally with model names |
| `server_status` | Detected simulators, config, sandbox paths, runtime state |
| `recent` | Recently-used circuits and jobs from the persistent index |

</details>

## Development

```bash
uv sync                        # install runtime + dev dependencies
uv run pytest tests/ -v        # tests
uv run pyright                 # type checking
uv run ruff check src/ tests/  # lint
uv run ltspice-mcp             # run the server (stdio)
```

More: [docs/DESIGN.md](docs/DESIGN.md) (scope, architecture, non-goals) and [docs/spice_lex.md](docs/spice_lex.md) (SPICE parser internals).

## License

GPL-3.0
