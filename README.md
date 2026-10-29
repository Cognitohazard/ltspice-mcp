# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **Work in progress.** Version 0.6.0 was a breaking release. It merged the
> old tools, each of which ran one operation, into the tools described below,
> and made the engine importable as a Python API. Pin `ltspice-mcp==0.5.*` if
> you need the old tools.

ltspice-mcp lets AI assistants run LTspice and ngspice simulations and edit
LTspice `.asc` schematics. The server reads the simulator's output files and
returns measurements by name, such as cutoff frequency, overshoot, phase
margin, rise time, and the operating-point values `gm`, `gds` and `vth` of
each device. It reads and writes the same files you open in LTspice. The
server is built on [spicelib](https://github.com/nunobrum/spicelib).

## Quick start

**Claude Code:**

```
/plugin marketplace add cognitohazard/ltspice-mcp
/plugin install ltspice-mcp
```

**Claude Desktop:** build the extension in [`packaging/mcpb/`](https://github.com/cognitohazard/ltspice-mcp/tree/master/packaging/mcpb)
and drag the `.mcpb` file onto Claude Desktop. The extension asks for an
optional sandbox folder. If you leave it blank, there is no sandbox.

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

The server needs Python 3.11 or newer. In Claude Code,
`claude mcp add -s project spice -- ltspice-mcp` adds it without editing the
JSON.

**You also need LTspice or ngspice** on the same machine. The server finds an
installed simulator automatically on Windows, Linux and macOS. On WSL, you
have to set the LTspice path yourself ([WSL](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#wsl)). Editing `.asc`
schematics needs LTspice, because the server uses LTspice's symbol libraries.
Reading and checking netlists works without a simulator. The plugin and the
extension need [`uv`](https://docs.astral.sh/uv/).

**Make sure the assistant uses the server.** Some assistants run the
simulator from the shell instead. Name the server `spice`, as the plugin
does. Do not name it `ltspice`, because LTspice's own MCP server uses that
name. Then add this rule to your project's `CLAUDE.md` or your client's
equivalent:

> Always use the spice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

If you prefer a less strict rule, [When the shell is fine](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#when-the-shell-is-fine)
lists the cases in which running the simulator from the shell is acceptable.

## What you can ask

You describe the task, and the assistant writes the circuit and chooses the
measurements. The server runs the simulator and returns the measured values.

> **"Bias this NMOS common-source stage into saturation at the target drain current and report gm/ID."**

The assistant writes the netlist and runs an operating-point analysis. It
reads the transistor's drain current, gm and gds by name, and compares VDS
with VDSAT to confirm saturation. If the bias is wrong, it adjusts the
circuit and runs it again, which takes a couple of seconds each time.

Other examples:

- *"What's the overshoot and settling time of this regulator's step response?"*
- *"Run a 200-run Monte Carlo with 5% resistors and tell me the output spread."*
- *"Sweep the load from 100 Ω to 10 kΩ and find where efficiency drops."*
- *"Characterize this NMOS: gm and gm/ID vs VGS."*
- *"Find an N-channel power MOSFET for a low-side switch and measure the on-state drop."*
- *"Build this differential pair as a schematic I can open in LTspice."*
- *"Is this loop stable?"* (phase and gain margin at every crossover)
- *"What's the resonant frequency and Q of this series RLC?"*

The reply lists simulator warnings next to the values they affect, such as a
singular-matrix warning from an ngspice run that still wrote plausible
values.

## Working with LTspice

The server reads and writes ordinary LTspice and SPICE files, so you and the
assistant can edit the same circuit. You can draw a schematic in LTspice and
ask *"why doesn't the output move?"*, or have the assistant build one and
open it in LTspice yourself. The assistant reads your edits from the file on
its next request.

**With the schematic open in LTspice** (Windows, LTspice 26.1 or later), the
assistant's edits appear in the LTspice window, and Ctrl+Z undoes them. If
the window has unsaved changes, the assistant asks you to save them and does
not overwrite them. The assistant can also tell which schematic you have
open, open a schematic or simulation result in LTspice, and read LTspice's
own documentation. See [Working with LTspice
open](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#working-with-ltspice-open) for details.

**LTspice's own MCP server.** LTspice 26.1 and later include an MCP server
that registers as `ltspice`, and you can install it next to this one.
LTspice's server controls the LTspice window and returns raw samples and
logs. This server does not depend on it, and it adds checked schematic edits,
sweeps and Monte Carlo as background jobs, measurements returned as numbers,
and ngspice support. If you registered this server as `ltspice`, rename it to
`spice`. [Alongside LTspice's own MCP
server](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#alongside-ltspices-own-mcp-server) describes an LTspice
setting you may want to change.

## What it does

**Simulation and measurement.** The server runs LTspice or ngspice and reads
the binary output files itself. It returns numeric measurements in the time
domain (rise and fall time, overshoot, settling, delay, period, duty cycle,
jitter, RMS, THD) and the frequency domain (filter cutoffs, gain and phase,
stability margins, resonance and Q, integrated noise). It also reports DC
operating points and `.MEAS` results, including failed ones. Operating-point
values of each device, such as `gm`, `gds` and `vth`, are available by name
on both simulators.

**Schematic editing.** The server creates and edits LTspice `.asc` files,
placing components, wiring pins and labeling nets. It reports floating pins
and dangling labels, and it refuses diagonal wires, wires that cross pins or
junctions, and any edit to a file that changed since the assistant last read
it. The assistant edits plain netlists (`.cir`, `.net`, `.sp`)
with its own file tools, and the server checks them before they run.

**Sweeps and Monte Carlo.** The server can sweep several parameters at once,
swap models, and run Monte Carlo analyses with component tolerances,
`.MODEL` process variation and device mismatch. It aggregates statistics
across the runs, and any single run can also be analyzed alone.

**Jobs.** Each simulation runs as a job, which can be cancelled or given a
time limit. A `run_experiments` call waits up to 60 seconds by default and
then returns a receipt with the job ID while the job keeps running. Job
records are saved in the working directory, so a restarted server can still
read earlier jobs. A run still in progress when the server stopped is marked
failed, and a job submitted with `execution.recoverable` set can run it again.
The results list simulator warnings, missing measurements and extreme values,
and the assistant decides whether they matter.

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
| `run_code` | Run a Python snippet that calls the server's Python API, for loops and calculations |

The assistant learns to use the tools from `inspect(kind="guide")`, which
covers the simulators and common tasks, and `inspect(kind="reference")`,
which lists every option of every tool.

## Configuration

The server runs without configuration. To change a setting, copy
[`ltspice-mcp.example.toml`](https://github.com/cognitohazard/ltspice-mcp/blob/master/ltspice-mcp.example.toml) to `ltspice-mcp.toml`
in your working directory. That file describes every option, and each option
can also be set with an `LTSPICE_MCP_` environment variable. The server
stores job records and results in `.ltspice-mcp/` in the working directory,
so add that folder to `.gitignore`.

**The tools can use any path unless you set a sandbox.**
`[security] allowed_paths` limits the tools to the folders it lists. It does
not limit `run_code`, which runs Python with the server's own permissions, so
approve `run_code` in your client as you would a shell command. Set
`[tools] run_code = false` if anyone else can reach the server.

**Schematic and plot images** that the assistant can view need the optional
`raster` extra ([PNG rendering](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#png-rendering)).

[docs/USAGE.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md) covers the common settings, WSL, the files the
server creates, and running the server on another machine.

## Python API

Scripts and notebooks can import the same engine as `ltspice_mcp.api`. Use it
for optimizers, curve fits, CI checks, or any work in which each run depends
on the previous one. The API returns complete results and numpy arrays, while
the MCP tools return results in pages. Scripts and the server share the job
records in the working directory, so each can read a job the other ran.

```bash
pip install ltspice-mcp        # or: uv add ltspice-mcp
```

The plugin and the Desktop extension install the server in an environment of
their own, so a script needs this separate install. An assistant in a session
does not need it, because it can use the API through `run_code`.

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
See [Python API](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#python-api) for how the API differs from the
tools, and [docs/design/python_api.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/design/python_api.md) for the full
specification.

## Research behind the design

Published studies by other groups informed three design choices:

- **Measurements are returned as named numbers.** SPICEAssistant ([arXiv:2507.10639](https://arxiv.org/abs/2507.10639)) gave a model scalar LTspice results instead of raw output, which raised o3's solve rate on a benchmark of 269 power-supply tasks from 25.4% to 84.9%.
- **Schematic edits are typed and checked.** NetlistBench ([arXiv:2608.12197](https://arxiv.org/html/2608.12197)) found that the strongest model's accuracy on netlist edits with dependent steps fell from 80% at 3 steps to 26% at 15. The study measured unassisted edits to text netlists. `edit_schematic` validates each typed operation before it writes the file and was designed in response to that finding, but its effect has not been measured.
- **The tool interface is typed.** A benchmark of agents that take chip designs from RTL to GDS ([arXiv:2607.17528](https://arxiv.org/html/2607.17528v3)) traced 31.7% of physical-design errors to failures at the tool interface. That work is in a different domain, so it does not show the same effect in circuit simulation. The server follows its recommendations of registered APIs, persistent sessions and structured results.

## Development

```bash
uv sync                        # install runtime + dev dependencies
uv run pytest tests/ -v        # tests
uv run pyright                 # type checking
uv run ruff check src/ tests/  # lint
uv run ltspice-mcp             # run the server (stdio)
```

To release, add a dated section for the version to `CHANGELOG.md`, run
`scripts/release.sh <version>`, and push the tag it creates. Pushing the tag
publishes the release to PyPI.

## Contributing

- [docs/DESIGN.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/DESIGN.md): the architecture
- [docs/design/](https://github.com/cognitohazard/ltspice-mcp/tree/master/docs/design): specifications of the MCP tools and the Python API
- [docs/TESTING.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/TESTING.md): testing practice
- [THIRD_PARTY_NOTICES.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/THIRD_PARTY_NOTICES.md): vendored components

The project does not take outside contributions yet. Bug reports with a
reproduction are welcome as issues.

## License

GPL-3.0-or-later
