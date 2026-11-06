# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **Work in progress.** Expect breaking changes across versions.

ltspice-mcp is an MCP server for circuit simulation. It lets an AI assistant
run LTspice or ngspice and get the results back as named measurements, such
as cutoff frequency, phase margin, overshoot, or a transistor's gm and vth.
Sweeps, corners and Monte Carlo run as background jobs, with statistics
across the runs.

The assistant can also build and edit LTspice `.asc` schematics. Without the
server, an assistant edits the `.asc` text directly and can connect a pin by
mistake, for example by running a wire across it. Nothing reports this until
a simulation gives a wrong answer, often several turns later. This server
routes each wire itself, between pins the assistant names. An edit that would
connect something by accident, such as a wire over another part's pin or a
label that shorts two nets, is refused with the reason before anything is
written, so the assistant can fix it on the next call. The server also
reports parts and text drawn on top of each other, which make a sheet hard to
read. With LTspice 26.1 or later on Windows, the edits appear in an LTspice window you
have open ([Working with LTspice open](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#working-with-ltspice-open)).

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
symbol libraries. Reading and checking netlists works without a simulator.
The plugin and the extension need [`uv`](https://docs.astral.sh/uv/).

**macOS is not supported yet: the server cannot read simulation results
there.** It reads each `.raw` and `.log` file in a separate process with a
memory limit, which has not been validated on macOS.

**Make sure the assistant uses the server.** Some assistants run the
simulator from the shell instead. Name the server `spice`, as the plugin
does, and not `ltspice`, which is the name LTspice's own MCP server takes.
Then add this rule to your project's `CLAUDE.md` or your client's equivalent:

> Always use the spice MCP server for any SPICE/circuit simulation, sweep,
> or analysis. Do not invoke ngspice or LTspice from the shell, and do not
> hand-parse `.raw` files or `wrdata` output.

If you prefer a less strict rule, [When the shell is fine](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#when-the-shell-is-fine)
lists the cases in which running the simulator from the shell is acceptable.

**LTspice's own MCP server.** LTspice 26.1 and later include an MCP server,
which registers as `ltspice` and can run beside this one. It drives the
LTspice window and returns raw samples and logs, leaving measurements and
Monte Carlo setup to the assistant. This server does not need it. [Alongside
LTspice's own MCP server](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#alongside-ltspices-own-mcp-server)
describes an LTspice setting you may want to change.

## What you can ask

- *"Size this NMOS common-source stage for 100 µA at a gm/ID of 15, and check
  that it stays in saturation at the slow and fast corners."*
- *"Here's my LDO schematic. What are the overshoot and settling time for a
  1 mA to 100 mA load step, and how much phase margin does it have at full
  load?"*
- *"Run 500 Monte Carlo samples with 1% resistors and mismatch in the input
  pair, and tell me what fraction keep the output offset under 1 mV."*
- *"Find an N-channel MOSFET in the LTspice library for a 12 V, 5 A low-side
  switch, and compare the conduction loss of the three best candidates."*
- *"I have my filter open in LTspice and the output doesn't move. Find out
  why, fix it, and show me V(out)."*

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
in your working directory. That file describes every option. Any of them
can also be set with an `LTSPICE_MCP_` environment variable. The server keeps job
records and results in `.ltspice-mcp/` in the working directory; add that
folder to `.gitignore`.

**The tools can use any path unless you set a sandbox.**
`[security] allowed_paths` limits them to the folders it lists. It does not
limit `run_code`, which runs Python with the server's own permissions, so
approve `run_code` in your client as you would a shell command. If anyone
else can reach the server, set `[tools] run_code = false`.

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
tools. The full specification is in [docs/design/python_api.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/design/python_api.md).

## Research behind the design

Published studies by other groups informed three design choices:

- **Measurements are returned as named numbers.** A study of LLM workflows for switched-mode power supply design ([arXiv:2507.10639](https://arxiv.org/abs/2507.10639), first published as SPICEAssistant) found that GPT-4o could not reliably read a ripple value from a waveform given as numbers or as an image, so its tools extract such values from each LTspice run and return them as text. With that simulation feedback, o3 solved 84.9% of 269 power-supply design tasks, against 25.4% with no simulation. The study compared simulation feedback with none, not named numbers with raw output.
- **Schematic edits are typed and checked.** NetlistBench ([arXiv:2608.12197](https://arxiv.org/html/2608.12197)) found that the strongest model's accuracy on netlist edits with dependent steps fell from 80% at 3 steps to 26% at 15. The study measured unassisted edits to text netlists. `edit_schematic`, which checks each typed operation before it writes the file, was designed in response; its effect has not been measured.
- **The tool interface is typed.** A benchmark of agents that take chip designs from RTL to GDS ([arXiv:2607.17528](https://arxiv.org/html/2607.17528v3)) traced 31.7% of physical-design errors to failures at the tool interface. That work is in a different domain, so it does not show the same effect in circuit simulation. The server follows the benchmark's recommendations: registered APIs, persistent sessions and structured results.

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
