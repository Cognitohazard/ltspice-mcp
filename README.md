# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **0.6.0 (upcoming) is a breaking release:** the tool surface consolidates to
> six operations plus a plot widget, and the same engine becomes importable as
> a Python library. The 0.5 series keeps the old 49-tool surface
> (`ltspice-mcp==0.5.*`).

ltspice-mcp lets LLM assistants and Python code run LTspice and ngspice simulations and edit LTspice `.asc` schematics. It returns structured measurements such as cutoff frequency, overshoot, phase margin, rise time, and per-device small-signal operating-point parameters (`gm`, `gds`, `vth`, …). Callers access these values by name without parsing raw files. It works on the same files you open in LTspice. Built on [spicelib](https://github.com/nunobrum/spicelib).

## Two ways to use it

You can use the same six operations as an **MCP server** or as a **Python
library**. Both run the same engine: the same code handles each operation,
reads the same files, and writes the same job records to disk.

| | MCP server | Python library |
|-|-|-|
| Who calls it | an assistant in Claude Code, Claude Desktop, Cursor, or another MCP client | a script, notebook, or CI job |
| What a call looks like | a tool call in the conversation; large results are split into pages and continued with a cursor | a method call; results are returned in full, with waveforms as numpy arrays |
| Long runs | the server keeps the job running; check on it with `jobs` | the process owns the job; `api.close()` or normal interpreter shutdown cancels unfinished work |
| Good for | interactive work: explore, edit, run a few checks per turn | code: optimizers, custom post-processing, pipelines, full result sets |

A sweep or Monte Carlo matrix is one call through either interface. Use the
Python API when each run depends on code that processes the previous result,
such as an optimizer, curve fit, or CI check. The API returns the complete
result set, while MCP paginates large results. Both interfaces use the same
working directory and job records. An assistant can start a sweep over MCP,
and a script can read the completed job by its `job_id`. A script can also
run a batch for an assistant to analyze later.

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

`run_experiments` defines a three-case sweep, measures each case, and returns
the minimum and maximum cutoff frequencies with their assignments.
`api.reference()` lists the six operations. `api.reference("run_experiments")`
prints that operation's full argument tree. From a shell, use
`python -m ltspice_mcp.api reference [op]`. `api.load_raw()` returns numpy
arrays for direct waveform access.

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

**Configure the agent to use this server.** Some agent clients defer MCP tool
schemas until first use. When an agent chooses how to simulate, it may have
seen only the tool names and may use a simulator through the shell instead.
Name the server `ltspice` or `spice`, so the tool names carry the domain even
while the schemas are deferred, and add this rule to your project's
`CLAUDE.md` or the equivalent file for your client:

> Always use the ltspice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

Web clients (claude.ai, ChatGPT) need a stdio→HTTP bridge such as [`mcp-proxy`](https://github.com/sparfenyuk/mcp-proxy) — only expose this server on a network you fully control, since it writes files and spawns processes inside `allowed_paths`.

A **Claude Desktop extension** is also available: build the `.mcpb` in [`packaging/mcpb/`](packaging/mcpb/) and drag it onto Claude Desktop for a one-click install with a native folder picker for your circuits directory. Like the plugin, it wraps the PyPI package and needs `uv` and a simulator on the host (it does not bundle LTspice or ngspice).

## Using it

Once connected, you ask for circuit work in plain language. The assistant designs the circuit and decides what to measure; the server runs the simulator, parses the binary output, and returns the numbers. It returns the simulation results and includes the simulator's warnings. You and the assistant decide whether the results are acceptable.

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

**Warnings are returned with the measurements they affect.** A simulator such as ngspice can report a "singular matrix" warning in its log and still finish the run and write plausible values. The server includes that diagnostic in an `observations` field next to the returned value.

### Co-design on the same files

Everything operates on ordinary LTspice and SPICE files. You and the assistant can edit the same files:

- Sketch a schematic in LTspice, then hand it over: *"what's the bias point?"*, *"why doesn't the output move?"*, *"add compensation and check the phase margin."*
- Or the reverse: the assistant designs and verifies the circuit and writes the `.asc`; you open it in LTspice, inspect it, and tweak by hand. Your manual edits are simply the file's new state — the assistant picks up from there on the next request.
- Changes can flow either direction mid-design: adjust a value in the GUI and ask for re-verification, or have the assistant sweep a change you're considering before you commit to it.

### When to shell out instead

An agent with a shell should run quick one-off ngspice simulations directly. Local ngspice runs are scriptable and usually take under a second, so MCP adds little in that case. Use the server when you need LTspice execution, named values parsed from binary raw files, declared sweep/corner/Monte Carlo matrices with durable idempotent submission, jobs that outlive a call, or geometry-checked `.asc` editing. `analyze_results` can also read a bare `raw_path` produced outside the server, so a simulation can run in the shell and be analyzed here.

## Why it is shaped this way

Three published studies by other groups report results that support the main design choices.

- **Measurements are returned as named numbers.** SPICEAssistant (Nau, Krummenauer, Zimmermann, [arXiv:2507.10639](https://arxiv.org/abs/2507.10639)) exposes scalar LTspice results to the model. The paper reports that, in a five-case ripple test, GPT-4o returned the correct value in 2 cases from a raw numeric vector and in 1 case from a plot image. It also reports that o3's solve rate on a 269-task power-supply benchmark increased from 25.4% to 84.9% with its tools, while retrieval-augmented prompting alone added 18.7 percentage points. This server returns measurements as named numbers. `plot_waveform` displays signal shape, while the measurement recipes return scalar values. AnalogCoder-Pro ([arXiv:2508.02518](https://arxiv.org/abs/2508.02518)) also uses images for diagnosis and scalar values for measurement.
- **Schematic edits use typed operations and validation.** NetlistBench (Ma et al., [arXiv:2608.12197](https://arxiv.org/html/2608.12197)) evaluated LLM edits to SPICE netlists across 2,342 cases. It reports 96–100% accuracy for parameter changes and device removal, compared with 41–83% for device addition. For the strongest model evaluated, accuracy on compound edits fell from 80% with 3 dependent steps to 26% with 15. The authors conclude that "LLMs currently cannot serve as unverified netlist editors." The benchmark covers text netlists, while `edit_schematic` edits `.asc` schematics. `edit_schematic` accepts a batch of typed operations, validates them before writing, and returns the resulting geometry. `verify_circuit` can then compare the schematic with an exported netlist.
- **The tool interface is typed.** An RTL-to-GDS agent benchmark ([arXiv:2607.17528](https://arxiv.org/html/2607.17528v3)) reports that 31.7% of physical-design errors were tool-interface failures: valid commands that failed because of tool state or version. The authors recommend registered APIs, persistent sessions, normalized result structures instead of log parsing, and stateful validation. This result concerns physical-design tooling rather than SPICE simulation, so it is supporting context rather than direct evidence for this server. The server uses typed tool schemas, persistent session state, structured results, and syntax and arity validation before a simulation runs.

## The Python API, same engine

`Api` starts the same engine in the caller's process and does not require an
MCP server. Its interface differs from MCP in the following ways:

- **Complete results.** Large MCP responses may be paginated or capped. The
  API collects every page and returns the complete result. It rejects
  MCP-only controls such as response budgets, pagination cursors, and wait
  dwells instead of rewriting them. This keeps replayed calls consistent
  between MCP and Python.
- **The Python process owns its jobs.** `run_experiments(wait=False)` returns
  a receipt immediately. Unfinished jobs are cancelled by `api.close()`, at
  the end of a `with` block, or during normal interpreter shutdown. Keep the
  process running until the job finishes. Use a long-lived server for work
  that must continue after the process exits.
- **One live engine per process.** An `Api` created inside a running server
  process raises an error. A cold `Api()` starts in well under a second; the
  heavy imports are loaded by the first call that needs them.
- `api.load_raw()` / `api.measurements()` return numpy-backed data for your
  own post-processing, and `api.reference(op)` prints any operation's full
  argument tree.

## What it does

**Simulation and measurement.** Runs LTspice or ngspice and parses the binary output directly. Measurements are computed server-side and returned as numbers: time-domain (rise/fall, overshoot, settling, delay, period/duty/jitter, RMS, THD), frequency-domain (filter cutoffs and roll-off, gain and phase at any frequency, stability margins, resonance peaks with Q, integrated noise), DC operating points, and `.MEAS` directive results including the ones that failed. Per-device small-signal operating-point parameters (`gm`, `gds`, `vth`, …) come back by name on **both** simulators — LTspice via an auto-added `.options logopinfo` block in the log, ngspice via `.save @dev[param]` traces. Read the set across a `.dc` sweep as a gm/ID table with the `waveform` recipe in `format: "csv"`, or a single bias point with the `operating_point` recipe (address them as `m1.gm` / `@m1[gm]`, no rawfile parsing).

**Schematic and netlist editing.** Creates and edits LTspice `.asc` files by placing components, wiring pins, and labeling nets. It rejects wires that collide with pins, overlap junctions, or run diagonally. Edits report floating pins and dangling labels, and a session's edits can be reverted. Plain netlists (`.cir`/`.net`) support the same operations at text level. A static validation pass catches malformed cards before simulation begins.

**Sweeps and Monte Carlo.** Multi-dimensional parameter sweeps and Monte Carlo with per-component tolerances, `.MODEL` process variation, and Pelgrom W·L device mismatch. Per-measurement statistics are aggregated across runs, and any single run can be pulled out and analyzed like a standalone simulation.

**Jobs and trust.** Simulations run as cancellable jobs with timeouts and a concurrency cap; long runs return a job ID immediately and job state survives a server restart. Results include simulator warnings, missing measurements, and extreme node values as structured observations. The server does not assign a trust rating; the caller evaluates these observations.

## Supported simulators

| Simulator | Status |
|-|-|
| LTspice | Primary. Windows native, WSL2 (Windows LTspice.exe via interop), Linux via Wine. Required for `.asc` schematic editing (needs `.asy` symbol libraries). |
| ngspice | Supports simulation, parsing, diagnostics, and analysis. Does not require LTspice. |
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

Netlists are written and edited with the agent's own file tools; the server does not wrap text edits. The same six operations are importable as `ltspice_mcp.api` (`Api(working_dir=...)`), so a Python script can drive the same engine without an MCP client.

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

The `lp` recipe returns these scalar results:

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

If the result is off target, edit the netlist, run it again, and repeat the measurement. Long simulations return a job ID instead of blocking. Use `jobs` with `action="status"`, `action="wait"`, or `action="cancel"` to manage them. Job metadata persists in per-circuit sidecars (`{dir}/.ltspice-mcp/jobs/` — add `.ltspice-mcp/` to your `.gitignore`), and MCP resources (`spice://results/...`, `spice://netlists/...`, `spice://config`) expose jobs, signals, measurements, and config for browsing.

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

Release with `scripts/release.sh 0.5.1`. The script updates the plugin manifests, commits the changes, and creates the tag. The package version comes from hatch-vcs. Push the tag to publish to PyPI.

More: [docs/DESIGN.md](docs/DESIGN.md) (scope, architecture, non-goals) and [docs/spice_lex.md](docs/spice_lex.md) (SPICE parser internals).

## License

GPL-3.0
