# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **Work in progress.** Expect breaking changes across versions.

ltspice-mcp is an MCP server for circuit simulation. It lets an AI assistant
run LTspice or ngspice and get the results back as named measurements, such
as cutoff frequency, phase margin, overshoot, or a transistor's gm and vth.
Sweeps, corners and Monte Carlo run as background jobs, with statistics
across the runs.

The assistant can also build and edit LTspice `.asc` schematics. Edited as
text, a schematic goes wrong without any error: a wire drawn across a pin
connects it, and the mistake shows up only when a simulation gives the wrong
answer, several turns later. Instead, the assistant names the pins to
connect, and the server draws the wire. An edit that would connect
something by accident, such as a wire over another part's pin or a label that
shorts two nets, is refused with the reason before anything is written, so
the assistant can fix it on the next call. The server also reports parts and
text drawn on top of each other, so the finished sheet is easy to read. On
Windows, the edits appear in an LTspice window you have open.

The schematics and netlists are ordinary files that you can open in LTspice
yourself. The same engine is also a Python package, `ltspice_mcp.api`, for
scripts and optimizers. The server is built on [spicelib](https://github.com/nunobrum/spicelib).

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

**You also need LTspice or ngspice** on the same machine. The server finds
either one on Windows and Linux. On WSL, set the LTspice path yourself
([WSL](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#wsl)).
Editing `.asc` schematics needs LTspice, because the server uses LTspice's
symbol libraries. Reading and checking netlists works without a simulator. The plugin and the extension need
[`uv`](https://docs.astral.sh/uv/).

**macOS is not supported yet: the server cannot read simulation results
there.** It reads each `.raw` and `.log` file in a separate process with a
memory limit, and that has not been validated on macOS.

**Make sure the assistant uses the server.** Some assistants run the
simulator from the shell instead. Name the server `spice`, as the plugin
does, and not `ltspice`, which is the name LTspice's own MCP server takes.
Then add this rule to your project's `CLAUDE.md` or your client's equivalent:

> Always use the spice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

If you prefer a less strict rule, [When the shell is fine](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#when-the-shell-is-fine)
lists the cases in which running the simulator from the shell is acceptable.

## What you can ask

### Simulate and measure

> *"Bias this NMOS common-source stage into saturation at 100 µA and report gm/ID."*

The assistant writes a netlist, runs an operating point, and reads the
transistor's drain current, gm, gds, VDS and VDSAT from the reply. If VDS is
below VDSAT, it moves the bias and runs again.

The server reads the simulator's binary output itself and measures:

- **Transient:** rise and fall time, overshoot, settling, delay, period, duty
  cycle, jitter, RMS, THD, and the amplitude and phase of one tone.
- **AC and noise:** filter cutoffs, gain and phase at a frequency, stability
  margins, resonance and Q, poles and zeros, return loss and VSWR, integrated
  noise.
- **Operating point:** node voltages, branch currents, and each device's gm,
  gds, vth and other small-signal parameters, on both simulators.
- **`.meas`:** the deck's own measurements, including those that failed.

A measurement can be given a limit, and the reply says whether each run
meets it. The reply lists simulator warnings next to the values they affect,
such as a singular-matrix warning from an ngspice run that still wrote
plausible values. The server can also measure a `.raw` file simulated
elsewhere.

- *"What's the overshoot and settling time of this regulator's step response?"*
- *"Is this loop stable?"* (phase and gain margin at every crossover)
- *"What's the resonant frequency and Q of this series RLC?"*
- *"Characterize this NMOS: gm and gm/ID vs VGS."*
- *"Find an N-channel power MOSFET for a low-side switch and measure the on-state drop."*

### Sweeps, corners and Monte Carlo

One `run_experiments` call can sweep several parameters at once, swap models
for process corners, and run Monte Carlo with component tolerances, `.MODEL`
parameter spread and Pelgrom device mismatch. A parameter can be set on one
device inside nested subcircuits. With an installed Sky130 PDK, the server
samples process and mismatch variation from the PDK's own statistical
models. Only its 1.8 V NMOS is supported so far. The reply gives statistics across the runs,
and any single run can be measured on its own.

- *"Run a 200-run Monte Carlo with 5% resistors and tell me the output spread."*
- *"Sweep the load from 100 Ω to 10 kΩ and find where efficiency drops."*
- *"Does the phase margin stay above 45° at every corner?"*
- *"What does 5 mV of threshold mismatch in the input pair do to the offset?"*

Each simulation runs as a job, which can be cancelled or given a time limit.
`run_experiments` waits up to 60 seconds, then returns the job ID and leaves
the job running. Job records are kept in the working directory, so a
restarted server can still read earlier jobs. A job submitted with
`execution.recoverable` can rerun the cases a stopped server left unfinished.

### Schematics and netlists

The server creates and edits LTspice `.asc` files: it places, moves and
replaces parts, wires pins, labels nets and adds directives. A batch of edits
is written whole or not at all, so a whole build is one call, and a batch
that fails leaves the file as it was. Besides accidental connections, the
server refuses diagonal wires, which LTspice connects anywhere along their
length, and any edit to a file that changed since the assistant last read it,
so it never overwrites a change you made. It reports floating pins and
dangling wires and labels, which are usually unfinished work.

The assistant edits netlists (`.cir`, `.net`, `.sp`) with its own file tools,
and the server checks them before they run. It can trace a net or list the
parts in either kind of file, and compare a schematic with a netlist. For a
netlist built from subcircuits, it lists every device inside them under the
name the simulator gives it, such as `x1.m3`. It also searches the installed
model libraries for a part, and draws a schematic as a PNG the assistant can
look at.

You and the assistant can work on the same circuit: the server reads your
changes from the file on its next call, and you can open anything it writes.

- *"Build this differential pair as a schematic I can open in LTspice."*
- *"I drew this in LTspice. Why doesn't the output move?"*
- *"Does my schematic match this netlist?"*
- *"Replace Q1 with a 2N3906 and keep it wired."*

### With the schematic open in LTspice

On Windows with LTspice 26.1 or later, an edit to a schematic you have open
appears in the LTspice window, and Ctrl+Z undoes it. If the window has
unsaved changes, the server refuses the edit and the assistant asks you to
save first. The assistant can also see which schematic is in front, open a
schematic or a finished run in LTspice with the traces you named already
plotted, and read LTspice's own help files.

- *"Add a 10 pF load to the circuit I have open."*
- *"Show me V(out) and I(L1) from that run in LTspice."*

[Working with LTspice open](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#working-with-ltspice-open)
has the details.

**LTspice's own MCP server.** LTspice 26.1 and later include an MCP server,
which registers as `ltspice` and can run beside this one. It drives the
LTspice window and returns raw samples and logs, leaving measurements and
Monte Carlo setup to the assistant. This server does not need it. [Alongside
LTspice's own MCP server](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#alongside-ltspices-own-mcp-server)
describes a setting you may want to change on LTspice's side.

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

The assistant reads how to use them from `inspect(kind="guide")`, which
covers the simulators and common tasks, and looks up any option with
`inspect(kind="reference")`.

## Configuration

The server runs without configuration. To change a setting, copy
[`ltspice-mcp.example.toml`](https://github.com/cognitohazard/ltspice-mcp/blob/master/ltspice-mcp.example.toml) to `ltspice-mcp.toml`
in your working directory. That file describes every option, and each can
also be set with an `LTSPICE_MCP_` environment variable. The server keeps job
records and results in `.ltspice-mcp/` in the working directory; add that
folder to `.gitignore`.

**The tools can use any path unless you set a sandbox.**
`[security] allowed_paths` limits them to the folders it lists. It does not
limit `run_code`, which runs Python with the server's own permissions, so
approve `run_code` in your client as you would a shell command, and set
`[tools] run_code = false` if anyone else can reach the server.

[docs/USAGE.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md) covers the common settings, WSL, the files the
server creates, and running the server on another machine.

## Python API

Scripts and notebooks can import the same engine as `ltspice_mcp.api`. Use it
for optimizers, curve fits, CI checks, or any loop where each run depends on
the last. The API returns complete results and numpy arrays, where the tools
return pages. Scripts and the server share the job records in the working
directory, so each can read a job the other ran.

```bash
pip install ltspice-mcp        # or: uv add ltspice-mcp
```

A script needs this install even if you use the plugin or the Desktop
extension, which keep the server in an environment of their own. An
assistant in a session does not need it, because it can use the API through
`run_code`.

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
[Python API](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#python-api) explains how the API differs from the
tools, and [docs/design/python_api.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/design/python_api.md) is the full
specification.

## Research behind the design

Published studies by other groups informed three design choices:

- **Measurements are returned as named numbers.** A study of LLM workflows for switched-mode power supply design ([arXiv:2507.10639](https://arxiv.org/abs/2507.10639), first published as SPICEAssistant) found that GPT-4o could not reliably read a ripple value from a waveform given as numbers or as an image, so its tools extract such values from each LTspice run and return them as text. With that simulation feedback, o3 solved 84.9% of 269 power-supply design tasks, against 25.4% with no simulation. The study compared simulation feedback with none, not named numbers with raw output.
- **Schematic edits are typed and checked.** NetlistBench ([arXiv:2608.12197](https://arxiv.org/html/2608.12197)) found that the strongest model's accuracy on netlist edits with dependent steps fell from 80% at 3 steps to 26% at 15. The study measured unassisted edits to text netlists. `edit_schematic`, which checks each typed operation before it writes the file, was designed in response; its effect has not been measured.
- **The tool interface is typed.** A benchmark of agents that take chip designs from RTL to GDS ([arXiv:2607.17528](https://arxiv.org/html/2607.17528v3)) traced 31.7% of physical-design errors to failures at the tool interface. That is a different domain, and the effect has not been shown for circuit simulation. The server follows the benchmark's recommendations: registered APIs, persistent sessions and structured results.

## Development

```bash
uv sync                        # install runtime + dev dependencies
uv run pytest tests/ -v        # tests
uv run pyright                 # type checking
uv run ruff check src/ tests/  # lint
uv run ltspice-mcp             # run the server (stdio)
```

To release, run `scripts/release.sh <version>` and push the tag it creates.
The script renders the fragments in `changelog.d/` into `CHANGELOG.md`,
stamps the version into the plugin and Desktop-extension manifests, and
tags the commit. The pushed tag publishes to PyPI.

## Contributing

- [docs/DESIGN.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/DESIGN.md): the architecture
- [docs/design/](https://github.com/cognitohazard/ltspice-mcp/tree/master/docs/design): specifications of the MCP tools and the Python API
- [docs/TESTING.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/TESTING.md): testing practice
- [THIRD_PARTY_NOTICES.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/THIRD_PARTY_NOTICES.md): vendored components

The project does not take outside contributions yet. Bug reports with a
reproduction are welcome as issues.

## License

GPL-3.0-or-later
