# ltspice-mcp — Design

An LLM agent can analyse a circuit. It can still make errors such as these:

- It checks the gain of a loop and does not check the phase.
- It moves a part and leaves a pin connected to nothing.
- It calculates the mean of a transient waveform and ignores that the time steps are unequal.
- It reads a measurement from the log and does not see that the solver reported a singular matrix.
- It expects two LTspice builds to write the same log. They use different encodings.
- It writes a script that reads a result file. No test covers the script, so a wrong value can look correct.

An agent often finds such an error later, but the search uses time and context. Sometimes the error stays, because a wrong value can look correct.

ltspice-mcp is an MCP server that runs SPICE simulations and edits LTspice schematics. Its tools report these cases in the first reply. The tools contain circuit design practice and recorded simulator behaviour, and tests cover them. Some examples of what a reply reports:

- a node at 1e30 V, or a value that is not finite
- a measurement that failed in every run of a sweep
- a waveform that LTspice stored in compressed form, with few points
- a `.step` card that ngspice ignores, so the netlist ran one time only
- a number with an upper-case `M`, which SPICE reads as milli
- two net names on one net, or two wires that cross and join
- a node name that ngspice changes, which connects two nodes with no warning
- the part of a circuit that a comparison did not cover

These are examples. The netlist lint has 34 rules and the schematic lint has 22. The analyses and the log reader add their own findings.

An agent can also write a script for each step. The tools differ from such a script:

- **Tests cover them.** Each file reader is tested against 418 recorded inputs on two LTspice builds. Each reply is tested against its schema.
- **They connect.** A call returns an id, and the next call uses that id. The Python API reads the same records.
- **They use little context.** A call is about 50 tokens. A reply omits empty fields. Long data stays on disk, and a later call reads it by its id.
- **They stay.** A later session uses the same tools and the same stored results. It does not write the script again.

In short, this project strives to direct the LLM's attention to issues before a person must correct them, to reveal potential problems the AI isn't looking for, and to increase efficiency with tested tools.

## Scope

The server runs locally and speaks MCP on stdio. Each tool takes the path of a circuit file. The server is for an agent in Claude Code, for a person in a chat client such as Claude Desktop, and for scripts that import the Python API.

LTspice comes first, and Windows is the primary platform. WSL2 and Wine also run LTspice. ngspice has full support for simulation and analysis on Linux and Windows, and the lint and the ngspice section of the packaged guide list where its dialect differs from LTspice. QSPICE and Xyce have the support that spicelib gives. Schematic editing is for LTspice schematics only, because it reads LTspice's symbol files.

Later, the server will work with xschem for IC design. No date is set.

## The two surfaces

A tool call `run_experiments(...)` and the Python call `api.run_experiments(...)` take the same arguments. They run the same validation and the same operation. The Python API and the tools are one contract.

A script gets complete results, numpy arrays and exceptions. A reply to a script has no size limit. A tool is for one intent whose reply the agent uses as it is. Python is for loops, for combining runs and for math on traces.

An agent that has only MCP gets the Python surface through `run_code`. A script and the server share the stored work in the working directory, so each can read a job that the other ran.

There are 8 tools. The operations are data: sweeps are variations, measurements are recipes, schematic changes are ops and reads are kinds. Each tool definition uses context in every turn, so a few tools cost less than one tool for each operation. A batch of ops is also one transaction.

No tool edits netlist text. An agent with file access writes a netlist well with its own file tools.

## The schematic

An agent adds a resistor at (320, 160) and draws a wire from its pin to an op-amp input along a path that it chooses. `edit_schematic` validates the edit before it writes the file. If the wire crosses a pin of another net, the call refuses the edit and returns each blocking element with its position.

There is no automatic router and no automatic layout. An automatic router gives a drawing that is hard to read, and a person must read the schematic later.

The server supplies the geometry. `inspect(kind="symbol")` returns the pins and the bounding box of a symbol before the agent places it. The server checks a proposed route by LTspice's rules for which points of a sheet are one net, and returns the conflicts.

`verify_circuit` has LTspice export the netlist of a saved sheet, and the server compares that netlist with a reference netlist or schematic. The file on disk is the reference, and an open LTspice window holds a copy. The server uses LTspice's own files and windows, and it does not draw a second schematic format.

## Results

An agent asks for the phase margin of a loop. `analyze_results` returns `phase_margin_deg` with the gain margin and the unity-gain frequency. The waveform comes only on request, from the `waveform` recipe or from `plot_waveform`.

A result is a fact with no conclusion. A reply can report a crest factor of 8.1 for one stretch of a waveform, and it does not call that stretch a spike. A reply with no finding means that no check fired, and it does not mean "verified".

`lib/result_observations.py` states the reason in the code. The agent has its own physics knowledge, so a trust label from the server adds no information. A wrong trust label is a false accusation, which a model can repeat or learn to ignore.

Sweeps, corners and Monte Carlo are data in one call. A `run_experiments` call lists its variations and attaches the analysis, and the reply returns the measured values of every case.

## The person

A person asks the agent for a plot of `V(out)`. `plot_waveform` draws an interactive chart in the chat where the client supports it, and in a browser otherwise. The same results can open in the LTspice window, and the server writes the `.plt` plot settings that LTspice reads.

On Windows, each LTspice run and export starts on a hidden desktop, so LTspice does not take the keyboard focus from the person. With LTspice 26.1 or later, the server connects to the open LTspice windows. `edit_schematic` refuses an edit when an open window differs from the file, because a save from that window would overwrite the edit.

## Stored work

An agent repeats a request with the same `request_id`, and the server returns the earlier job without a new simulation. A job records the simulator executable and its inputs, and each case records the build that its run reported. If the build has changed since the first request, the server refuses the repeat as an `idempotency_conflict`.

Results stay on disk behind an id. A later session reloads a job by its id and reads its cases without a new simulation.

## Principles

Each principle has a short explanation and one case. The first nine are the principles in the README.

### Limits of an answer

- **Every check reports what's not covered, so a partial check is not mistaken for a pass.** A check that did not run is reported as not run. A QSPICE run reports `measurements_coverage: simulator_unsupported`, because the server does not read QSPICE's measure output.
- **Results are reported as facts without conclusions, leaving the engineering judgement to the AI and the human.** A severity appears only where the simulator wrote one. LTspice XVII calls an ignored model parameter an error on a run that finished, and the reply keeps that word as the simulator's own.
- **The same question gets the same answer from each tool and from the Python API.** One function decides each question, and every tool and the Python API call that function. For example, every tool that reads a run's measurements gives the same reason when the log could not be read.

### Contents of a reply

- **Mechanical sanity checks are run automatically without needing human/AI direction, catching errors without extra context bloat.** The lint runs on every deck before the simulator starts. A `.param TEMP` line does not set the temperature, so `run_experiments` refuses the deck and names the line.
- **Analyses return measured values with units, so the AI works from a number and does not spend context on raw waveform data.** Each value carries its unit in its name. The `edges` recipe returns the rise and fall times of a pulse in seconds.
- **Refusals say how to fix the request, so mistakes usually cost no more than one turn.** A refusal names the field, the place and the way round. A misspelled argument is refused with the list of the fields that the tool accepts.
- **A finding gives its place.** The place is the part, the net, the line or the coordinates. Text drawn across a wire is reported with the anchor of the text and the two ends of the wire.

### Schematic edits

- **Schematic layout is directed by the AI, with every edit validated before it is written. There is no auto-router, because auto-routed schematics are hard to read.** A batch of ops is one transaction, so a refused op leaves the sheet as it was. `wire_pins` refuses a diagonal wire and a short between two named nets.
- **The file on disk is the reference.** LTspice reads a sheet once and keeps a copy, so a save from its window overwrites an edit made on disk. The server refuses to write while an open window's copy differs from the file.

### Simulator behaviour

- **Simulator and file-format quirks are recorded from real LTspice builds and covered by tests, so the tools do not depend on guesses.** Each recording is a file that LTspice wrote for a listed input. LTspice XVII prints 0 for a measurement that failed at one step, and the reply marks that step's value as not established.
- **A recording comes before the code that uses it, and behaviour without a recording is reported as not known.** Code that reads an LTspice format must name its recording. A run on a build with no recordings reports `diagnostics_coverage: build_unrecorded`.
- **The simulator does its own work, and the server does not copy a simulator function.** A copy can differ from the simulator in a case that no test covers. `verify_circuit` has LTspice export the netlist of a sheet, and the server does not write a netlister of its own.

### Stored work

- **Jobs and results are stored on disk and requests are repeatable, so work continues across sessions without re-running simulations.** A retry with the same `request_id` is safe. A renamed argument is renamed in one release, and the refusal of the old name gives the new one.

### Cost of a call

- **A small set of composable tools keeps context use low, with the Python API for anything more complex.** Each tool definition stays in context for every turn. A sweep, a corner set and a Monte Carlo run are three kinds of variation in one tool.
- **One call for one intent.** The server does the steps between the request and the answer, such as staging, locking, exporting and comparing. `jobs(action="place_beside_sheet")` puts the results of a run where LTspice opens them.
- **A small run is fast enough to try.** An agent can then run an experiment in place of a long chain of reasoning. Any work added to a simulation run is weighed against that cost. A script compares small runs with a local baseline.
- **A function is a tool only when it needs the server's knowledge, stored state or a transaction. If not, it is Python.** A job that continues after the call needs the server, and so does a schematic edit. The mean of a trace is a line of numpy in `run_code`.

## Architecture

The server has two surfaces and one core. The MCP tools and the Python API both call the core, and `run_code` gives the Python API to an MCP client.

```
  An agent or a chat client               A script or a notebook
            |                                        |
      MCP, on stdio                       import ltspice_mcp.api
            |                                        |
   +-----------------+       run_code       +------------------+
   |   The 8 tools   | -------------------> |  The Python API  |
   +-----------------+                      +------------------+
            \                                      /
             +------------------------------------+
             |              The core              |
             |                                    |
             |   Circuits     Runs      Results   |
             |   Checks       Records   Replies   |
             +------------------------------------+
                |              |               |
          Circuit files    Simulators      The store
          on disk          LTspice,        jobs, records
                           ngspice         and results
```

The core has six parts:

- **Circuits** read netlists, LTspice schematics and symbols. This part decides which points of a sheet are one net, and it applies schematic edits.
- **Runs** prepare a deck for each simulator and start the simulator. This part keeps the state of each job and limits the number of parallel simulations. The server perturbs the values for a Monte Carlo run itself, and each perturbed case runs on the same path as every other run.
- **Results** read `.raw` and `.log` files in a separate process with a deadline and a memory limit. The recipes compute the measured values from those results.
- **Checks** are the netlist lint, the schematic lint, the log reader's diagnostics and the observations on a result.
- **Records** are the store on disk. It keeps each job, its cases and its results by id.
- **Replies** leave out the fields that say nothing. A list that is too long is cut, and the reply gives the call that reads the rest.

The evidence for each LTspice behaviour is a recording. `tests/fixtures/ltspice_recorded/` holds the files that LTspice 26 and LTspice XVII wrote for the inputs in `inputs/cases.toml`. A test compares the code with each recording on any machine, without LTspice. [TESTING.md](TESTING.md), "Recorded LTspice behaviour", describes the recordings.

The server uses [spicelib](https://github.com/nunobrum/spicelib) to find and start the simulators. Its `.raw` reader and its log reader start from spicelib's readers, and the server adds its own decoding where a recording differs. The schematic editor writes the `.asc` file through spicelib's editor. The server depends on spicelib directly, not on PyLTSpice, which re-exports it.

The geometry, the net rules and the checks of a schematic edit are the server's own code. Each must agree with the recordings. [spicelib_bugs.md](spicelib_bugs.md) lists each spicelib bug that the server works around.

## Non-goals

- **Automatic layout or routing.** The agent sets the position of each part and the path of each wire.
- **Semantic part search**, such as a search for an NMOS with a low threshold voltage. It needs every `.model` card parsed and an index of parameters, which is a separate project.
- **Thin wrappers around file edits.** An agent with file tools edits a netlist directly, and an agent without them can pass a deck to `run_experiments`.
- **Auth, multi-tenancy and remote hosting.** The server is a local tool with no OAuth flow and no isolation between tenants.

## Other documents

[USAGE.md](USAGE.md) covers installation, configuration, WSL and the files that the server creates. [design/mcp_surface.md](design/mcp_surface.md) specifies the tool arguments and replies, and [design/python_api.md](design/python_api.md) specifies the Python API. [spice_lex.md](spice_lex.md) describes the netlist parser. [TESTING.md](TESTING.md) lists the tests of the principles.
