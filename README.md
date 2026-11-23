# ltspice-mcp

<!-- mcp-name: io.github.cognitohazard/ltspice-mcp -->

> **Work in progress.** Expect breaking changes across versions.

ltspice-mcp is an MCP server for circuit simulation with LTspice and ngspice. It provides tools for an AI to simulate circuits and to edit LTspice schematics.

## Functionalities

* Run a circuit, a sweep, a set of corners or a Monte Carlo analysis in one call. Jobs run in the background and across sessions.
* Well-tested tools for common measurements like gain and phase margins, step response, edge times, distortion, noise, operating points, etc.
* Plot waveforms in an interactive chart in the chat where the client supports it.
* Edit schematics in an aesthetic, human-readable form.
* Co-design with a human in [the same LTspice window](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#working-with-ltspice-open) (LTspice 26.1 or later, on Windows).
* Composable API that can be leveraged for complex custom tasks.

## Design Principles

The tools contain circuit design practice and the recorded behaviour of each simulator. Both are applied in every call, because AI often misses errors that are not directly related to the request.

* Mechanical sanity checks are run automatically without needing human/AI direction, catching errors without extra context bloat.
* Every check states what it did not cover, so a partial check is not mistaken for a pass.
* Analyses return measured values with units, so the AI works from a number and does not spend context on raw waveform data.
* Results are reported as facts without verdicts, leaving the engineering judgement to the AI and the human.
* Schematic layout is directed by the AI, with every edit validated before it is written. There is no auto-router, because auto-routed schematics are hard to read.
* Simulator and file-format quirks are recorded from real LTspice builds and covered by tests, so the tools do not depend on guesses.
* Jobs and results are stored on disk and requests are repeatable, so work continues across sessions without re-running simulations.
* A small set of composable tools keeps context use low, with the Python API for anything more complex.
* Refusals say how to fix the request, so mistakes usually cost no more than one turn.

[docs/DESIGN.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/DESIGN.md) has the full text.

## Quick start

**Claude Code:**

```
/plugin marketplace add cognitohazard/ltspice-mcp
/plugin install ltspice-mcp
```

**Claude Desktop:** build the extension in [`packaging/mcpb/`](https://github.com/cognitohazard/ltspice-mcp/tree/master/packaging/mcpb)
and drag the `.mcpb` file onto Claude Desktop. The extension asks for an
optional sandbox folder. If you leave it blank, there is no sandbox.

**Cursor, Codex and other MCP clients** ([Windsurf](https://docs.devin.ai/desktop/cascade/mcp), [Gemini CLI](https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html), [Continue](https://docs.continue.dev/customize/deep-dives/mcp), [Cline](https://docs.cline.bot/mcp/mcp-overview), [Zed](https://zed.dev/docs/ai/mcp), …): install the server and add it to the client's MCP config.

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

[Cursor](https://cursor.com/docs/mcp) reads this JSON from `.cursor/mcp.json`
in the project, or from `~/.cursor/mcp.json` for every project.
[Codex](https://developers.openai.com/codex/mcp) reads TOML from
`~/.codex/config.toml`:

```toml
[mcp_servers.spice]
command = "ltspice-mcp"
tool_timeout_sec = 660
```

Codex stops a tool call after 300 s by default; `run_code` can run for 600 s.

The server needs Python 3.11 or newer. If a client cannot find `ltspice-mcp`,
set `command` to its full path (on Windows, the `ltspice-mcp.exe` in the
folder `uv tool dir --bin` prints). In Claude Code,
`claude mcp add -s project spice -- ltspice-mcp` adds it without editing the
JSON.

**Development version.** To run the `master` branch, use this entry in the
MCP config. uvx needs Git on PATH to install from the repository.

```json
{
  "mcpServers": {
    "spice": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/cognitohazard/ltspice-mcp@master", "ltspice-mcp"]
    }
  }
}
```

The maintainer runs `master`, which gets each fix before a release. A change
to the code lands on `master` after the tests pass on Linux and Windows. A
tool's arguments, its reply fields and the Python API can change between two
commits. uvx fetches `master` again each time the client starts the server,
so each restart runs the newest commit. To stay on one commit, put its hash
in place of `master`, for example
`git+https://github.com/cognitohazard/ltspice-mcp@daab2255`.

The Claude Code plugin and the Claude Desktop extension install the release
from PyPI. To run the development version, remove the plugin or the extension
and add the entry above to the MCP config. In Claude Code,
`claude mcp add -s project spice -- uvx --from git+https://github.com/cognitohazard/ltspice-mcp@master ltspice-mcp`
adds it.

**Changes since the last release.** `CHANGELOG.md` lists the changes up to
the last release. Each later change is a short file in
[`changelog.d/`](https://github.com/cognitohazard/ltspice-mcp/tree/master/changelog.d),
and the type in its name, such as `breaking` or `fixed`, gives the kind of
change.

**Server version.** A bug report needs the version of the server. A build
from `master` reports a version such as `0.8.1.dev651+gc17aaf992`, where the
text after `+g` is the commit hash. The client gets it when the server starts
(`serverInfo.version` in the MCP `initialize` reply). The AI can also print it
with `run_code`: `import ltspice_mcp; print(ltspice_mcp.__version__)`.

**You also need LTspice or ngspice** on the same machine. The server finds
either one on Windows and Linux. On WSL, set the LTspice path yourself
([WSL](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#wsl)).
Editing `.asc` schematics needs LTspice, because the server uses LTspice's
symbol libraries. Reading and checking netlists works without a simulator.
The plugin and the extension need [`uv`](https://docs.astral.sh/uv/).

**macOS is not supported yet: the server cannot read simulation results
there.** It reads each `.raw` and `.log` file in a separate process with a
memory limit, which has not been validated on macOS.

**Make sure the AI uses the server.** Some AI clients run the
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
Monte Carlo setup to the AI. This server does not need it. [Alongside
LTspice's own MCP server](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md#alongside-ltspices-own-mcp-server)
describes an LTspice setting you may want to change.

## Examples

- *"Bias this common-source amplifier at 100 µA. What's its gain, and does the
  transistor stay in saturation at the slow and fast corners?"*
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
| QSPICE, Xyce | Secondary, selected per run, with the support that spicelib provides. Netlists only, not `.asc`. QSPICE only when the server runs natively on Windows. |

Simulator detection and launch are built on [spicelib](https://github.com/nunobrum/spicelib).

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

The AI reads how to use them from `inspect(kind="guide")`, which
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
`[security] allowed_paths` limits them to the folders it lists; a relative
entry, such as `"."`, is a folder in the working directory. It does not
limit `run_code`, which runs Python with the server's own permissions, so
approve `run_code` in your client as you would a shell command. If anyone
else can reach the server, set `[tools] run_code = false`.

[docs/USAGE.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/USAGE.md) covers the common settings, WSL, the files the
server creates, and running the server on another machine.

## Python API

The `ltspice_mcp.api` package provides the same functions as the tools to scripts and notebooks. Use it
for optimizers, curve fits, CI checks, or any loop where each run depends on
the last. The API returns complete results and numpy arrays, where the tools
return pages. Scripts and the server share the job records in the working
directory, so each can read a job the other ran.

```bash
pip install ltspice-mcp        # or: uv add ltspice-mcp
```

A script needs this install even if you use the plugin or the Desktop
extension, which keep the server in an environment of their own. An
AI in a session does not need it, because it can use the API through
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

Published studies by other groups informed these design choices:

- **Analyses return measured values with units.** A study of LLM workflows for switched-mode power supply design ([arXiv:2507.10639](https://arxiv.org/abs/2507.10639), first published as SPICEAssistant) found that GPT-4o could not reliably read a ripple value from a waveform given as numbers or as an image, so its tools extract such values from each LTspice run and return them as text. With that simulation feedback, o3 solved 84.9% of 269 power-supply design tasks, against 25.4% with no simulation. The study compared simulation feedback with none, not measured values with raw output.
- **Every schematic edit is validated before it is written.** NetlistBench ([arXiv:2608.12197](https://arxiv.org/html/2608.12197)) found that the strongest model's accuracy on netlist edits with dependent steps fell from 80% at 3 steps to 26% at 15. The study measured unassisted edits to text netlists. `edit_schematic`, which checks each typed operation before it writes the file, was designed in response; its effect has not been measured.
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

- [docs/DESIGN.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/DESIGN.md): the design principles and the architecture
- [docs/design/](https://github.com/cognitohazard/ltspice-mcp/tree/master/docs/design): specifications of the MCP tools and the Python API
- [docs/TESTING.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/docs/TESTING.md): testing practice
- [THIRD_PARTY_NOTICES.md](https://github.com/cognitohazard/ltspice-mcp/blob/master/THIRD_PARTY_NOTICES.md): vendored components

The project does not take outside contributions yet. Bug reports with a
reproduction are welcome as issues.

## License

GPL-3.0-or-later
