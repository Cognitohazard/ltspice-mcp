# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **WIP**
> **0.6.0 was a breaking release:** the old per-operation tools were merged
> into the set described below, and the engine became importable as a Python
> API. Pin `ltspice-mcp==0.5.*` if you need the old tools.

ltspice-mcp lets AI assistants run LTspice and ngspice simulations and edit LTspice `.asc` schematics. Instead of raw output files, the assistant gets measurements back by name: cutoff frequency, overshoot, phase margin, rise time, and per-device operating-point values such as `gm`, `gds` and `vth`. It works on the same files you open in LTspice. Built on [spicelib](https://github.com/nunobrum/spicelib).

## Quick start

**Claude Code:**

```
/plugin marketplace add cognitohazard/ltspice-mcp
/plugin install ltspice-mcp
```

**Claude Desktop:** build the extension in [`packaging/mcpb/`](https://github.com/cognitohazard/ltspice-mcp/tree/master/packaging/mcpb)
and drag the `.mcpb` file onto Claude Desktop. It asks which folder your
circuits are in.

**Any other MCP client** ([Cursor](https://cursor.com/docs/mcp), [Windsurf](https://docs.devin.ai/desktop/cascade/mcp), [Gemini CLI](https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html), [Continue](https://docs.continue.dev/customize/deep-dives/mcp), [Cline](https://docs.cline.bot/mcp/mcp-overview), [Zed](https://zed.dev/docs/ai/mcp), …): install the server and add it to the client's MCP config.

```bash
uv tool install ltspice-mcp        # or: pipx install ltspice-mcp
```

```json
{
  "mcpServers": {
    "spice": { "command": "ltspice-mcp", "args": [] }
  }
}
```

This needs Python 3.11 or newer. In Claude Code you can skip the JSON with
`claude mcp add -s project spice -- ltspice-mcp`.

**You also need a simulator** on the same machine: LTspice or ngspice, found
automatically on Windows, Linux and macOS. On WSL, set the LTspice path
yourself ([WSL](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#wsl)). Editing `.asc` schematics needs LTspice,
because it uses LTspice's symbol libraries. Reading and checking netlists
works without a simulator. The plugin and the extension need
[`uv`](https://docs.astral.sh/uv/).

**Make sure the assistant uses it.** An assistant may run the simulator from
the shell out of habit. Keep the server's name `spice` (the plugin already
does; avoid `ltspice`, which LTspice's own MCP server uses), and add this to
your project's `CLAUDE.md` or your client's equivalent:

> Always use the spice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

[When the shell is fine](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#when-the-shell-is-fine) covers the
exceptions, if you prefer a softer rule.

## What you can ask

You describe the circuit work; the assistant writes the circuit and picks
what to measure; the server runs the simulator and returns the numbers.

> **"Bias this NMOS common-source stage into saturation at the target drain current and report gm/ID."**

The assistant writes the netlist, solves the bias point, and reads the
transistor's operating point back by name: drain current, gm, gds, and VDS
against VDSAT to confirm saturation. If the bias is off, it adjusts and runs
again, a couple of seconds per pass.

Other examples:

- *"What's the overshoot and settling time of this regulator's step response?"*
- *"Run a 200-run Monte Carlo with 5% resistors and tell me the output spread."*
- *"Sweep the load from 100 Ω to 10 kΩ and find where efficiency drops."*
- *"Characterize this NMOS: gm and gm/ID vs VGS."*
- *"Find an N-channel power MOSFET for a low-side switch and measure the on-state drop."*
- *"Build this differential pair as a schematic I can open in LTspice."*
- *"Is this loop stable?"* (phase and gain margin at every crossover)
- *"What's the resonant frequency and Q of this series RLC?"*

Simulator warnings come back with the numbers they affect. If ngspice reports
a singular matrix but still writes plausible values, the reply says so next
to the value.

## Working with LTspice

Everything works on ordinary LTspice and SPICE files, so you and the
assistant can take turns. Sketch a schematic in LTspice and ask *"why doesn't
the output move?"*, or let the assistant build one and open it yourself. The
assistant reads your hand edits from the file on its next request.

**With the schematic open in LTspice** (Windows, LTspice 26.1 or later), the
assistant's edits show up in the window, and Ctrl+Z undoes them. If the
window has unsaved changes, the assistant asks you to save first instead of
overwriting them. The assistant can also see which schematic you have open,
open a schematic or a simulation result in LTspice for you, and look things
up in LTspice's own documentation. See [Working with LTspice
open](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#working-with-ltspice-open) for details.

**LTspice's own MCP server.** LTspice 26.1 and later ships its own MCP server,
which registers as `ltspice`. You can install both. LTspice's server drives
the LTspice window and returns raw samples and logs. This server works on the
files: checked schematic edits, sweeps and Monte Carlo as background jobs,
measurements as numbers, and ngspice support. The features above don't need
LTspice's server. If you registered this server as `ltspice`, rename it to
`spice`. See [Alongside LTspice's own MCP
server](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#alongside-ltspices-own-mcp-server) for a setting worth
changing on LTspice's side.

## What it does

**Simulation and measurement.** Runs LTspice or ngspice and reads the binary
output directly. Measurements come back as numbers: time-domain (rise/fall,
overshoot, settling, delay, period, duty cycle, jitter, RMS, THD),
frequency-domain (filter cutoffs, gain and phase, stability margins,
resonance and Q, integrated noise), DC operating points, and `.MEAS` results,
including failed ones. Per-device operating-point values (`gm`, `gds`, `vth`,
…) are available by name on both simulators.

**Schematic editing.** Creates and edits LTspice `.asc` files: places
components, wires pins and labels nets. It refuses diagonal wires and wires
that cross pins or junctions, and reports floating pins and dangling labels.
An edit is refused if the file changed since the assistant last read it.
Plain netlists (`.cir`, `.net`, `.sp`) are checked before they run; the
assistant edits them with its own file tools.

**Sweeps and Monte Carlo.** Multi-dimensional parameter sweeps, model
swaps, and Monte Carlo with component tolerances, `.MODEL` process variation
and device mismatch. Statistics are aggregated across runs, and any single
run can be analyzed on its own.

**Jobs.** Simulations run as jobs that can be cancelled or given a time
limit, and that survive a server restart. Long runs return a job ID right
away. Results list simulator warnings, missing measurements and extreme
values; the server reports these and leaves the judgment to the assistant.

## Supported simulators

| Simulator | Status |
|-|-|
| LTspice | Primary. Windows, WSL2 (runs the Windows LTspice), Linux via Wine. Required for `.asc` schematic editing. |
| ngspice | Simulation and analysis. Does not need LTspice. |
| QSPICE, Xyce | Secondary, selected per run. Netlists only, not `.asc`. QSPICE only when the server runs natively on Windows. |

## Tools

The server has 8 tools:

| Tool | What it does |
|-|-|
| `run_experiments` | Run a circuit, or a sweep or Monte Carlo set of runs, optionally measuring each |
| `jobs` | Check on, wait for, cancel, or list submitted runs |
| `analyze_results` | Measure finished runs, or a `.raw` file produced elsewhere |
| `inspect` | Read circuits, schematics, symbols, nets, models, and server capabilities |
| `edit_schematic` | Create and edit `.asc` schematics: place, move, wire, label |
| `verify_circuit` | Check a circuit, compare a schematic with a netlist, draw a schematic |
| `plot_waveform` | Interactive waveform chart, in the chat where the client supports it, otherwise in a browser |
| `run_code` | Run a Python snippet against the engine, for loops and number crunching |

The assistant learns how to use them from the server itself:
`inspect(kind="guide")` for guidance on simulators and common tasks, and
`inspect(kind="reference")` for every option of every tool.

## Configuration

None is needed. To change a setting, copy
[`ltspice-mcp.example.toml`](https://github.com/cognitohazard/ltspice-mcp/blob/master/ltspice-mcp.example.toml) to `ltspice-mcp.toml`
in your working directory; every option is described there and can also be
set with an `LTSPICE_MCP_` environment variable. The server keeps job records
and results in `.ltspice-mcp/` in the working directory; add it to
`.gitignore`.

**`run_code` runs Python with the server's own permissions,** not inside the
folder sandbox. Approve it in your client as you would a shell command, and
set `[tools] run_code = false` if anyone else can reach the server.

**Schematic and plot images** for the assistant to look at need an optional
extra and the Cairo library: see [PNG rendering](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#png-rendering).

[docs/USAGE.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md) covers the common settings, WSL, which files
the server creates, and running it on another machine.

## Python API

The same engine is importable as `ltspice_mcp.api`, for scripts and notebooks:
optimizers, curve fits, CI checks, or anything where each run depends on the
last. It returns complete results and numpy arrays where the MCP tools return
pages. Scripts and the server share job records in the working directory, so
either can read a job the other ran.

```bash
pip install ltspice-mcp        # or: uv add ltspice-mcp
```

The plugin and the Desktop extension install the server in their own
environment, so a script needs this install. An assistant inside a session
doesn't: it can use the API through `run_code`.

With an RC low-pass in `circuits/rc.cir`:

```spice
* RC low-pass
V1 in 0 AC 1
R1 in out 1k
C1 out 0 159.155n
.ac dec 50 1 1Meg
.end
```

this sweeps R1 and returns the lowest and highest cutoff frequency:

```python
from ltspice_mcp.api import Api

with Api(working_dir="circuits") as api:
    result = api.run_experiments(
        circuits=[{"path": "rc.cir"}],
        variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k", "4k"]}}],
        analyze={"recipes": [
            {"key": "fc", "metric": "bode_filter", "signal": "V(out)",
             "field": "cutoff_high_hz", "reduce": ["min", "max"]},
        ]},
    )
    print(result["analysis"]["result"]["results"]["fc"]["reduced"])
```

`api.reference()` lists the operations and `api.guide()` returns the guide.
See [Python API](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#python-api) for how it differs from the tools,
and [docs/design/python_api.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/design/python_api.md) for the full
contract.

## Why it is shaped this way

Published studies by other groups support the main design choices:

- **Measurements come back as named numbers.** SPICEAssistant ([arXiv:2507.10639](https://arxiv.org/abs/2507.10639)) gave a model scalar LTspice results instead of raw output, and o3's solve rate on a 269-task power-supply benchmark rose from 25.4% to 84.9%.
- **Schematic edits are typed and checked.** NetlistBench ([arXiv:2608.12197](https://arxiv.org/html/2608.12197)) found that the strongest model's accuracy on compound netlist edits fell from 80% at 3 dependent steps to 26% at 15. That study tested unassisted edits to text netlists, so `edit_schematic` (typed operations, validated before writing) is a response to the finding, not a measured fix for it.
- **The tool interface is typed.** An RTL-to-GDS agent benchmark ([arXiv:2607.17528](https://arxiv.org/html/2607.17528v3)) traced 31.7% of physical-design errors to tool-interface failures. It is a different domain, so this is context rather than evidence; its recommendations (registered APIs, persistent sessions, structured results) are what the server does.

## Development

```bash
uv sync                        # install runtime + dev dependencies
uv run pytest tests/ -v        # tests
uv run pyright                 # type checking
uv run ruff check src/ tests/  # lint
uv run ltspice-mcp             # run the server (stdio)
```

To release, run `scripts/release.sh <version>` (it needs a dated
`CHANGELOG.md` section for that version) and push the tag it creates; the tag
publishes to PyPI.

## Contributing

The architecture is in [docs/DESIGN.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/DESIGN.md), the tool and Python API contracts in [docs/design/](https://github.com/cognitohazard/ltspice-mcp/tree/master/docs/design), and the test practice in [docs/TESTING.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/TESTING.md). Vendored components are listed in [THIRD_PARTY_NOTICES.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/THIRD_PARTY_NOTICES.md). The project is not taking outside contributions at this stage; bug reports with a reproduction are welcome as issues.

## License

GPL-3.0-or-later
