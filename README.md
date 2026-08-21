# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **0.6.0 (upcoming) is a breaking release:** the tool surface consolidates to
> six operations plus a plot widget, and the same engine becomes importable as
> a Python library. The 0.5 series keeps the old 49-tool surface
> (`ltspice-mcp==0.5.*`).

Real circuit simulation for LLM assistants and Python code: LTspice and ngspice, plus direct editing of LTspice `.asc` schematics. Simulation results come back as structured numbers — cutoff frequencies, overshoot, phase margin, rise times, and per-device small-signal operating-point parameters (gm, gds, vth, …) read back **by name** — so an assistant (or your script) can design, verify, and iterate on circuits in the same files you open in LTspice, without ever hand-parsing a rawfile. Built on [spicelib](https://github.com/nunobrum/spicelib).

**One engine, two doors.** The same six operations are served over MCP to any
client, and importable in-process as a Python API — pick per task, mix freely:
an agent explores over MCP, then hands the 200-case sweep to a script.

## Quick start — Python library

```bash
pip install ltspice-mcp        # or: uv tool install / pipx install
```

```python
from ltspice_mcp.api import Api

with Api(working_dir="circuits") as api:
    result = api.run_experiments(
        circuits=[{"path": "rc.cir"}],
        variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k", "4k"]}}],
        analyze={"recipes": [
            {"key": "fc", "metric": "bode_filter", "signal": "V(out)",
             "reduce_field": "cutoff_high_hz", "reduce": ["min", "max"]},
        ]},
    )
    print(result["analysis"]["result"]["results"]["fc"]["reduced"])
```

One call declares a three-case sweep with an attached measurement and returns
the per-case cutoff extremes, each attributed to the assignment that produced
it. `api.reference()` lists the six operations and
`api.reference("run_experiments")` prints that operation's full argument tree
(`python -m ltspice_mcp.api reference [op]` from a shell); `api.load_raw()`
hands back numpy arrays when you want the waveforms themselves.

## Quick start — MCP server

In Claude Code, install the plugin:

```
/plugin marketplace add cognitohazard/ltspice-mcp
/plugin install ltspice-mcp
```

You also need LTspice or ngspice on the host (auto-detected on Windows, Linux, and macOS; on WSL set the LTspice path explicitly — [WSL notes](#configuration)). Netlist editing (`.cir`/`.net`) works with no simulator at all; `.asc` schematic editing needs LTspice's `.asy` symbol libraries. `uv` is required; the server itself is fetched from PyPI on first use.

### Manual install (any MCP client)

Install the server, then point your client at it:

```bash
uv tool install ltspice-mcp     # or: pip install ltspice-mcp / pipx install ltspice-mcp
```

**[Claude Code](https://code.claude.com/docs/en/mcp)** — one command (drop `-s project` to install it globally):

```bash
claude mcp add -s project ltspice -- ltspice-mcp
```

**Other clients** — [Claude Desktop](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop), [Cursor](https://cursor.com/docs/mcp), [Windsurf](https://docs.devin.ai/desktop/cascade/mcp), [Gemini CLI](https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html), [Continue](https://docs.continue.dev/customize/deep-dives/mcp), [Cline](https://docs.cline.bot/mcp/mcp-overview), [Zed](https://zed.dev/docs/ai/mcp) and others — add this `mcpServers` stanza to the client's MCP config file (each client documents its own path):

```json
{
  "mcpServers": {
    "ltspice": { "command": "ltspice-mcp", "args": [] }
  }
}
```

Python 3.11+ required. Verify with `ltspice-mcp --help`. The same server is also published under two alias names — `circuit-mcp` and `ngspice-mcp` — so `uvx circuit-mcp` / `uvx ngspice-mcp` are drop-in equivalents of `uvx ltspice-mcp` if one of those names is more discoverable for you.

**Make your agent actually reach for it.** Agent clients defer MCP tool schemas
until first use, so at the moment your agent decides *how* to simulate, it may
have seen nothing but bare tool names — and default to shelling out to a
simulator it knows from training. Two lines fix that. Keep the server named
`ltspice` (or `spice`) so every deferred tool name still carries the domain,
and add one rule to your project's `CLAUDE.md` (or your client's equivalent):

> Always use the ltspice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

Web clients (claude.ai, ChatGPT) need a stdio→HTTP bridge such as [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) — only expose this server on a network you fully control, since it writes files and spawns processes inside `allowed_paths`.

A **Claude Desktop extension** is also available: build the `.mcpb` in [`packaging/mcpb/`](packaging/mcpb/) and drag it onto Claude Desktop for a one-click install with a native folder picker for your circuits directory. Like the plugin, it wraps the PyPI package and needs `uv` and a simulator on the host (it does not bundle LTspice or ngspice).

## Using it

Once connected, you ask for circuit work in plain language. The assistant designs the circuit and decides what to measure; the server runs the simulator, parses the binary output, and hands back the numbers. It reports what the run produced, the simulator's own warnings included, and leaves the call on whether a result is good to you and the assistant.

> **"Bias this NMOS common-source stage into saturation at the target drain current and report gm/ID."**

The assistant writes the netlist, solves the bias point on LTspice, and reads the device's operating point back by name — drain current, gm, gds, VDS against VDSAT to confirm it's in saturation, and the gm/ID that analog designers size to. If the bias is off, it nudges the gate reference or W/L and re-runs, a couple of seconds per pass.

Other requests that work the same way:

- *"What's the overshoot and settling time of this regulator's step response?"* — runs a transient analysis and measures both from the waveform, plus rise time, ringing frequency, and the final value.
- *"Run a 200-run Monte Carlo with 5% resistors and tell me the output spread."* — perturbs components per run, simulates the batch, and reports mean, sigma, and worst-case values per measurement.
- *"Sweep the load from 100 Ω to 10 kΩ and find where efficiency drops."* — parameter sweep with per-run results.
- *"Characterize this NMOS: gm and gm/ID vs VGS."* — writes a `.dc Vgs` deck with `.save @m1[gm] @m1[id]`, runs it on ngspice, and returns the gm/ID table as one CSV (no `.control` block, no rawfile parsing).
- *"Find an N-channel power MOSFET for a low-side switch and measure the on-state drop."* — searches the libraries the deck pulls in for a part (`inspect(kind="model")`), drops it into a pulsed-gate transient, and reads Vds(on) and load current back from the `.meas` results.
- *"Build this differential pair as a schematic I can open in LTspice."* — places and wires the components into a real `.asc`, with orthogonal routing and pin-collision checks.
- *"Is this loop stable?"* — AC analysis of the loop gain; reports phase and gain margin at every crossover, not just the first.
- *"What's the resonant frequency and Q of this series RLC?"* — runs an AC sweep and reports each peak's center frequency, Q, and −3 dB bandwidth.

**The warning rides with the number it affects.** A simulator like ngspice can print "singular matrix" once, deep in a log you'd never open, then finish the run and write perfectly plausible numbers anyway — read them by hand and nothing looks off. Ask the server for one of those numbers and the buried line comes attached to it, in an `observations` field right next to the value, so the failure surfaces where you're already looking instead of where it's easy to scroll past.

### Co-design on the same files

Everything operates on ordinary LTspice and SPICE files, so the work passes back and forth between you and the assistant instead of living inside a chat:

- Sketch a schematic in LTspice, then hand it over: *"what's the bias point?"*, *"why doesn't the output move?"*, *"add compensation and check the phase margin."*
- Or the reverse: the assistant designs and verifies the circuit and writes the `.asc`; you open it in LTspice, inspect it, and tweak by hand. Your manual edits are simply the file's new state — the assistant picks up from there on the next request.
- Changes can flow either direction mid-design: adjust a value in the GUI and ask for re-verification, or have the assistant sweep a change you're considering before you commit to it.

### When to shell out instead

An agent with a shell should run quick one-off ngspice simulations itself — ngspice is scriptable, local runs take under a second, and wrapping that in a protocol adds cost without adding capability. The server's lane is everything the shell doesn't give you: LTspice execution (which has no native automation on any platform), parsing binary rawfiles into named numbers, declared sweep/corner/Monte-Carlo matrices with durable idempotent submission, jobs that outlive a call, and geometry-checked `.asc` editing. The analysis tools accept artifacts from simulations this server never ran — `analyze_results` takes a bare `raw_path` — so "simulate in the shell, analyze here" is a first-class workflow, not a workaround.

## The Python API, same engine

`Api` boots the identical engine in-process — same handlers, same semantics,
no server. The differences are exactly what an in-process caller wants:

- **Complete results.** Where the wire pages or caps a response, the API
  collects every page and returns the whole thing; wire-only controls
  (response budgets, pagination cursors, wait dwells) are rejected rather
  than silently rewritten, so a replayed call means the same thing at both
  doors.
- **Jobs live and die with your process.** `run_experiments(wait=False)`
  returns a receipt immediately, but the job is owned by the process that
  submitted it and is cancelled when it exits — `api.close()`, the end of a
  `with` block, or the interpreter shutting down. Keep the process alive
  until the job finishes, or run work that must outlive it through a
  long-lived server.
- **One live engine per process** (an `Api` inside a running server process
  refuses), and a cold `Api()` boots in well under a second — the heavy
  imports arrive with the first call that needs them.
- `api.load_raw()` / `api.measurements()` return numpy-backed data for your
  own post-processing, and `api.reference(op)` prints any operation's full
  argument tree.

## What it does

**Simulation and measurement.** Runs LTspice or ngspice and parses the binary output directly. Measurements are computed server-side and returned as numbers: time-domain (rise/fall, overshoot, settling, delay, period/duty/jitter, RMS, THD), frequency-domain (filter cutoffs and roll-off, gain and phase at any frequency, stability margins, resonance peaks with Q, integrated noise), DC operating points, and `.MEAS` directive results including the ones that failed. Per-device small-signal operating-point parameters (`gm`, `gds`, `vth`, …) come back by name on **both** simulators — LTspice via an auto-added `.options logopinfo` block in the log, ngspice via `.save @dev[param]` traces. Read the set across a `.dc` sweep as a gm/ID table with the `waveform` recipe in `format: "csv"`, or a single bias point with the `operating_point` recipe (address them as `m1.gm` / `@m1[gm]`, no rawfile parsing).

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
ngbehavior = "hsa"       # ngspice compat mode; unset = spicelib default, "hsa" fixes sectioned .lib corner select

[security]
allowed_paths = ["."]    # sandbox: only these directories are accessible

[simulation]
# max_parallel = 4       # default: number of CPU cores, capped at 8
timeout = 300.0          # seconds

[tools]
profile = "consolidated" # the only profile since 0.6.0

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

### The tool surface

The server exposes **7 tools**, six of them arranged over three planes, plus the waveform widget:

| Plane | Tool | What it does |
|-|-|-|
| Execute | `run_experiments` | Run one deck or a whole matrix — sweeps, corners, Monte Carlo — in one declarative call, optionally returning the measurements with the receipt |
| Execute | `jobs` | Follow, wait on, cancel, list, or page the runs of a submitted job |
| Understand | `analyze_results` | Measure a finished job (or a bare `.raw` this server never ran) through named recipes |
| Understand | `inspect` | Read decks, schematics, symbols, nets, models, and server capabilities — never results |
| Author | `edit_schematic` | Create and mutate `.asc` transactionally: place, move, wire, label, set attributes |
| Author | `verify_circuit` | Syntax, symbol, layout, and quality checks, schematic-vs-netlist equivalence, and rendering |
| — | `plot_waveform` | Interactive chart of a run's waveforms, in-chat where the client renders widgets, otherwise opened on your desktop |

Netlists are authored and edited with the agent's own file tools — the server no longer wraps text edits. The same six ops are importable in-process as `ltspice_mcp.api` (`Api(working_dir=...)`), so a Python script drives the identical engine without an MCP client.

The `skills/` directory carries the domain knowledge that pairs with the surface: `skills/spice-experiments/SKILL.md` (the experiment workflow), `skills/ltspice/SKILL.md` and `skills/ngspice/SKILL.md` (SPICE syntax per engine), `skills/spice-bench-craft/SKILL.md` (bench archetypes). Copy the relevant skill into your client's persistent-instructions location.

**Migration from 0.5.** The `full` (49-tool) and `agentic` (41-tool) profiles were removed in 0.6.0; the consolidated surface above replaces them. `[tools] profile` still accepts `"full"` and `"agentic"`, but each logs a warning and serves the consolidated surface. Pin `ltspice-mcp==0.5.*` if you need the old per-operation tools.

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

then drives two tools:

```
verify_circuit(path="rc.cir", checks=["syntax"])
  → outcome "pass": directives valid, element arities check out — safe to simulate

run_experiments(
  circuits=[{"path": "rc.cir"}],
  analyze={"recipes": [{"key": "lp", "metric": "bode_filter", "signal": "V(out)"}]},
)
```

and gets back scalars, not a plot — the `lp` recipe's result:

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

Off-target → edit the netlist, re-run, re-measure. Long simulations return a job ID instead of blocking; `jobs` (`action="status"|"wait"|"cancel"`) manages them. Job metadata persists in per-circuit sidecars (`{dir}/.ltspice-mcp/jobs/` — add `.ltspice-mcp/` to your `.gitignore`), and MCP resources (`spice://results/...`, `spice://netlists/...`, `spice://config`) expose jobs, signals, measurements, and config for browsing.

<details>
<summary><strong>The capability vocabulary</strong></summary>

Every tool declares MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) and an `outputSchema` for `structuredContent` introspection. The capabilities live one level down, as the named values each tool accepts:

| Surface | Values |
|-|-|
| `analyze_results` recipes | `summary`, `measurements`, `value`, `signal_stats`, `edges`, `timing`, `periodic`, `transient_response`, `thd`, `bode_filter`, `bode_point`, `bode_slope`, `bode_crossing`, `stability`, `ac_structure`, `resonance`, `return_loss`, `noise_integral`, `operating_point`, `waveform` (inline envelope or full-fidelity CSV), `plot` |
| `inspect` kinds | `capabilities`, `components`, `symbol`, `symbols`, `net`, `model` |
| `edit_schematic` ops | `add_component`, `set_component_value`, `set_component_attribute`, `move_component`, `remove_component`, `wire_pins`, `add_net_label`, `remove_net_label`, `remove_wire`, `add_directive`, `remove_directive` |
| `jobs` actions | `status`, `wait`, `cancel`, `list`, `runs` |
| `verify_circuit` checks | `syntax`, `symbols`, `export`, `layout`, `quality`, `compare` |

Sweeps, corners, and Monte Carlo are not separate tools: they are `run_experiments` `variations`, so one call declares the whole matrix.

</details>

## Development

```bash
uv sync                        # install runtime + dev dependencies
uv run pytest tests/ -v        # tests
uv run pyright                 # type checking
uv run ruff check src/ tests/  # lint
uv run ltspice-mcp             # run the server (stdio)
```

Release: `scripts/release.sh 0.5.1` stamps the plugin manifests, commits, and tags — one input keeps the manifest versions in lockstep with the git tag (the package version, via hatch-vcs). Push the tag to publish to PyPI.

More: [docs/DESIGN.md](docs/DESIGN.md) (scope, architecture, non-goals) and [docs/spice_lex.md](docs/spice_lex.md) (SPICE parser internals).

## License

GPL-3.0
