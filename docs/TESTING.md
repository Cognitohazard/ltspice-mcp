# Testing & quality practice

This document records how the project tests and, more importantly, a class
of bug the tests kept missing and the mechanisms added to catch it. Read it
before adding a tool, an op, or a test campaign.

## The bugs this addresses

Two defects shipped despite a long run of adversarial stress passes against a
live server:

- a netlist→schematic converter that was unusable for active circuits (it
  silently dropped every transistor / controlled source, so an active circuit
  could not even start); and
- the schematic editor had no way to remove a wire or a net label — you could
  add both, but not undo either.

Neither is a wrong-output bug. Both are **absence-class** defects: a capability
that is missing, or unusable for an input class the tests never fed it. They
survived because a stress pass exercises code that exists, and a missing
capability has no code to exercise. You cannot exercise a tool that does not
exist, and a tool that quietly skips an input class looks correct on every
input outside that class.

The deeper cause was how the evaluation was run: it started from the tool
list and drove the tools through a plausible workflow ("here are the tools,
does a reasonable sequence work?"). Three biases followed from that:

- **Passive-input bias.** The circuit batteries were almost entirely R/C/V.
  An active device was never pushed through the build or convert workflow, so
  the converter's active-device blindness never triggered.
- **Additive bias.** Every workflow built something up; none took anything
  apart. No test needed removal, so nobody noticed it was missing.
- **Self-grading.** The same agent that built a tool also chose the test
  circuit and judged the result — and chose inputs that showcased what it had
  built.

## What each instrument can and cannot catch

|test method|detects|does not detect|
|-|-|-|
|stress pass / workflow tests|wrong behavior in code that runs|a missing capability, or one unusable for an input class the tests never feed it|
|unit / contract tests|wrong behavior in a unit|whether the tool set as a whole can complete a user task|
|ground-truth numeric checks|silently-wrong numbers|missing capability|

The mechanisms below exist to cover the third column.

## The four mechanisms

### 1. Inverse-operation closure (mechanical)

`tests/test_dispatch.py::TestOpInverseClosure`. The schematic-editing op surface
must be **closed under inversion**: for every op that mutates the `.asc`, an
inverse op exists (or it is self-inverse). It checks that an undo capability
exists, not that state round-trips byte-for-byte (e.g.
`remove_component(cleanup_wires=true)` drops wires `add_component` won't restore). Each op in the `SchematicOp` union is either
paired with an inverse op that exists, or declared self-inverse (re-applying it
with the prior arguments reverts it). The pairing table forces the decision: a
new `add_*` / `wire_pins` / `create` op with no entry fails the test, so a
one-way mutation ships only as a reviewed decision, not by accident. Had this
check existed earlier, it would have caught `add_net_label` shipping without
`remove_net_label` and `wire_pins` without `remove_wire`. Building the table
also turned up one missing inverse (`add_directive` had no `remove_directive`
op), which was added with the check.

The standalone mutating tools live outside the op union, so a companion guard
(`tests/test_dispatch.py::TestMutatingToolsAreReversible`) requires every
non-read-only registered tool to declare a reversal path, or to be an explicitly
accepted one-way mutation (see *Accepted one-way mutations* below). A new
standalone mutate tool added with no entry fails the test the same way.

### 2. Archetype build battery (input distribution)

`tests/test_circuit_asc.py::TestArchetypeBuildCoverage` and the active-device
end-to-end test in `tests/test_ngspice_e2e.py`. Every workflow that claims to
handle "a circuit" must be exercised against each canonical device class, not
just passives:

- passive (R / C / V)
- two-terminal active (diode)
- three-terminal active (MOSFET / BJT)
- four-terminal controlled source (VCVS / VCCS)

The build battery places and wires each class through the real build path; the
ngspice test simulates an active circuit (a saturated NPN switch) end to end and
asserts physical ground truth (the collector is pulled to saturation, proving
the device conducts). An unusable-for-a-class regression now fails on the next
run instead of after it ships. Keep the archetype set; add to it when a new
device class becomes supported, and do not remove active-device coverage.

### 3. Task-down coverage pass (discipline)

Path-walking asks "does what exists work?" Coverage asks "does what exists let
me finish the job?" Only the second can find a missing capability.
Periodically, list the **user tasks** (build a circuit, edit one, **fix a
mistake**, analyze a result, convert between forms) and for each walk the
*minimal tool sequence*, asking at every step whether the step is possible.
The removal gap was an impossible step ("undo a misplaced label") that no
happy-path walk attempted, because a happy path never needs to undo anything.
Start from what a user needs to accomplish, not from the tool list.

### 4. Blind-artifact judging (part discipline, part regression test)

The agent that builds an artifact is not the sole judge of its quality. This has
two halves, and only the second is automated — keep them distinct.

- **The discipline (not automated).** Feed the **artifact alone** — the `.asc`,
  the plot, the netlist, with no build narrative — to an independent reviewer
  (a person, or a separate model) against a rubric. This came out of the
  plot-evaluation work: leaving the title and filename on a plot leaked the
  expected answer into the vision evaluation and inflated its scores;
  stripping them corrected it. There is no automated blind
  grader in the suite; this is a review step you run by hand.
- **The code-backed piece.** `tests/test_circuit_asc.py::TestSchematicReadability`
  is a deterministic artifact-readback regression: it reads the built `.asc` back
  from disk and asserts the result is actually wired (real `WIRE` records, net
  labels only on the terminal nets) rather than connected only through net
  labels, judged on the artifact rather than on the sequence of calls that
  produced it. It is not a
  blind reviewer; it is a fixed assertion that encodes one rubric item.

## Accepted one-way mutations

Closure under inversion is the rule for the schematic op surface. A few
tool-level mutations are *not* paired, and are recorded here so they
are not mistaken for the absence-class bug above:

- File creation (`edit_schematic` with `base: "blank"`; formerly the
  `create_netlist` / `create_schematic` tools) has no delete pair; removing a
  file is a native filesystem operation, intentionally out of scope for a
  circuit editor.
- The pre-0.6.0 `configure_sweep` / `configure_montecarlo` tools created a
  persisted config with no delete-config tool. A delete tool had low value (a
  stale config is inert). The question is moot now: sweeps are
  `run_experiments` variations.

These are decisions, not oversights. If one stops being acceptable, add it to
mechanism 1 or 2.

## The practice that was already working

### Shared rules and asynchronous checkpoints

When changing a shared rule, search for the behavior as well as the helper's
name. Call references cannot find a copied bucket list, a hand-written path
join, or a second header check. Identify every consumer before editing, and
exercise the changed case through each distinct output path. A parser test
does not establish that filtering, unit reporting and scalar reads preserve
what the parser returned. Include a counterexample where a fallback would
disagree with explicit metadata, such as a voltage-shaped name typed as an
impedance.

For asynchronous tests, wait for the checkpoint the assertion needs.
An in-memory status change does not establish that its disk write completed.
Do not mutate an object still owned by a running coordinator to simulate a
different process. Use a separately loaded record, and control the relevant
write or completion boundary with events when testing an interleaving. Test
both orders deliberately; extending sleeps or accepting a successful rerun
does not fix the race. A load test can reveal additional failures, but it is
not a substitute for a deterministic regression of a known interleaving.

That rule was written down and then broken, and each break lost only on the
Windows runner, one per run. Three mechanisms now carry it:

- **Work the server starts has an owner that can say when it is done.**
  `BackgroundTasks` (`lib/background.py`) holds every task the server starts
  without awaiting it, and `ExperimentRunner.settled(job)` and
  `SessionState.settled()` wait until none is left, counting work started
  while they wait. A test asserting on what that work produced awaits
  `settled`, not the disappearance of a file the work also touches. Nothing
  outside the process is waited on: a simulator's exit, a fake's callback or
  another process still needs its own event or handshake.
- **The rules are checked.** `tests/test_test_hygiene.py` fails on a direct
  task spawn in the source with no recorded owner, and on the test-side
  patterns every past race used. A line that must break a rule carries a
  `# timing: <reason>` comment saying why. A wait's timeout is
  `LIVENESS_S` (`tests/conftest.py`), a cap on a hang, never a claim about
  how fast the runner is, including a wait handed to `asyncio.to_thread`.
- **Fake work that must outlast a budget is held, not slept.** A stand-in
  for slow work blocks on an event the test sets once the call has returned,
  or runs until the call's own deadline has passed; a sleep sized past the
  budget guesses at it. Everything else inside the budget is made instant
  first (a source loaded beforehand under `settled_stamps`, say), because
  the half of such a test that has to fit inside the budget is the half that
  fails on a slow runner.
- **Races lose on Linux first.** `--jitter-seed=N` (`tests/schedule_jitter.py`)
  delays thread-to-loop hand-offs and process starts and fires timers up to
  15.6 ms early, as Windows does, with delays drawn from the seed and the
  test's id. The CI jitter leg and the release gate run seeds 1 and 2. A
  failure prints the seed; it usually reproduces under it, though the seed
  cannot fix the operating system's own scheduling.
- **Whether a repeated read starts a parser process depends on the clock.**
  A source's stat stamp is trusted only once its times are older than a
  timestamp tick, so with the real clock a test re-reading a file it just
  wrote may or may not reach a worker. A test that counts parser requests, or
  asserts on a worker's reply, pins the answer with the `settled_stamps` or
  `unsettled_stamps` fixture (`tests/conftest.py`).

### Existing coverage

The following practices were already sound. They are kept, and everything
above assumes them:

- **Real-path tests.** Tests drive actual code paths — `tests/test_e2e.py`
  launches the real server over stdio and speaks the client protocol, and the
  rest go through config and startup rather than constructing state by hand.
  Substitution is confined to a few named seams, and the suite does use
  `monkeypatch` for them:
  - **The simulator subprocess boundary.** `fake_simulator` and
    `recorded_fixture_simulator` (`tests/conftest.py`) replace
    `ExperimentRunner.submit_netlist` — the one call that spawns a simulator.
    The first hands back a minimal artifact pair on a controllable delay (that
    delay is what catches a caller who printed a receipt for a job still in
    flight); the second copies a recorded real-LTspice `.raw`/`.log` pair in,
    so an analysis stage parses genuine simulator output.
  - **Platform and environment seams.** WSL detection and path conversion
    (`lib/wsl.py`), simulator detection at bootstrap, desktop browser launch
    (`lib/desktop.py`), and the optional cairosvg raster backend — so one
    machine can exercise every platform branch.
  - **Timeouts, lowered.** Parse deadlines and the shutdown cancel timeout are
    dropped to fractions of a second, so a bound can be shown to fire inside
    the suite instead of only being asserted about.

  What is *not* substituted: handlers, the response path, the SPICE lexer and
  validator, the `.raw`/`.log` parsers, symbol and schematic geometry, and the
  job registry and store. A test for any of those runs the real thing.
- **Ground-truth-first numeric validation.** Numeric results are checked against
  closed-form expected values, not "it didn't crash."
- **Recorded-real fixtures.** Real simulator `.raw` / `.log` output is captured
  under `tests/fixtures/` so dialect and parse seams run against true output
  offline (see `tests/conftest.py`). For LTspice the recordings are a system
  of their own, with a recorder and an inventory: see *Recorded LTspice
  behaviour* below.
- **Tiered live tests.** `tests/test_ngspice_e2e.py` runs whenever `ngspice` is
  on PATH (so it runs in CI); `tests/test_e2e.py` runs un-gated in degraded
  mode; `tests/test_ltspice_integration.py` and the build comparison in
  `tests/test_ltspice_recorder.py` are opt-in via an environment flag.
- **Drift guards.** `tests/test_doc_drift.py` checks documented tool counts and
  names against the registry; `tests/test_guide_delivery.py` pins the guide's
  structure (the section list is the files present, the index lists every
  section under its kind, every pointer to a section resolves) and that its
  doors serve one text; `tests/test_skill_docs.py` keeps the plugin to the one
  skill that points at the guide.

See `CLAUDE.md` for the canonical `pytest` / `ruff` / `pyright` commands and
`docs/DESIGN.md` for the architecture and the end-to-end verification recipe.

## Measuring what cannot be run enough times

Some questions about the product are only fully posed by a long, expensive
run: a two-hour sizing project driven by a model, judged on the final design.
Such runs are too costly and too noisy to repeat until an average clears the
noise floor, and the model's own variance (how much it thinks, which path it
takes) dwarfs most product-caused differences. Treat these as a class and
answer them with instruments that do not need repetition.

- **Detect patterns; do not estimate means.** Estimating a mean over long runs
  is what is unaffordable. Detecting a named failure is not: a silently wrong
  number, a stale artifact replayed as fresh, a cancel that reports success
  while the run continues. Each of those was a finding at one occurrence. So
  pre-register the patterns that would count, judge live, and stop a run once
  its pattern is established rather than letting it finish for a score.
- **Decompose, then check the decomposition.** A long task is thinking plus a
  sequence of the short steps a scripted session samples (run, measure, edit,
  sweep, verify, decide). A fix at the step level carries over. What does not
  carry over is the class of failure that only exists at length: context
  growth, state drift across dozens of edits, the agent losing a decision it
  made an hour earlier, accumulated tool results crowding the window. Keep
  that list explicit. A restricted test finds general problems only where it
  samples the failure modes of the long task, so each scale-only class needs
  a probe of its own.
- **Prefer model-free instruments for scale.** Bytes per tool result, listing
  and instruction size, growth per step: deterministic, one run, attributable
  to the product. Transcript replay: record one long session once, then
  replay its tool-call sequence against a new tree with no model in the loop;
  any output difference is a product change, and a person judges whether it
  would have moved the agent. Long edit batteries: many edits on one sheet,
  then a coherence check, with no model at all.
- **Pair the comparisons that do use a model.** Same day, same client, same
  model and effort, arms alternating, two or more sessions a side, read
  against the within-arm spread rather than the means. Cost is comparable
  only within a day: the client's prompt size and cache warmth move it more
  than the product does. When the spread exceeds the delta, stop measuring
  cost and look at deterministic proxies; that is how one cost gap between
  two trees became a one-line description fix.
- **Run the full task once, as a falsifier.** One long session per release,
  blind judged, against a pre-registered list of what would count as a
  failure. It cannot say "ten percent better". It can say nothing on the
  list happened, or name what did.

The scripted designer session (a stream of small concrete requests over one
working schematic, scored per request on correctness, geometry legality,
cost and turns) is the workload this project uses for the paired
comparisons. It represents the step plane a long project decomposes into,
and holds the thinking plane constant; it does not represent authoring from
a blank sheet, topology choice, or long-horizon context growth, which need
the instruments above.

## Privacy checks before pushing

Enable the repository hooks once per clone with
`git config core.hooksPath .githooks`. The pre-push hook checks commit messages,
annotated tag messages, filenames and each file revision being introduced,
including content added in one commit and removed in a later one. It checks
the receiving remote's current refs for new branches and refuses to proceed
when the required history cannot be inspected. Unrelated remote refs absent
locally contribute no exclusions; they do not require an extra fetch. Findings
name their category and location without printing the matched private value.

Run `uv run python scripts/privacy_scan.py tracked` to check the tracked working
tree, or `uv run python scripts/privacy_scan.py local` to preview unpublished
history against local remote-tracking refs. The release gate runs both checks.
The local preview uses the last fetched refs; the pre-push check verifies the
actual destination. The tracked-file test uses the same scanner, including
both UTF-16 byte orders for simulator artifacts.

## Platforms and the release gate

The suite is developed on WSL2 and gated on Linux CI, but the users run
Windows natively, and neither host reproduces it: a Linux container on a
WSL2 box inherits the WSL kernel signature, so `is_wsl()` is true there and
every WSL branch is taken. The 0.6.0 release found the whole class at once:
dozens of Windows-only failures, among them a text-mode `os.open` that broke
the revision guard, a `/mnt` drive mapping applied on Windows, a `SIGKILL`
that does not exist there, and two races that only Windows' file semantics
expose. The same release then found the runner shapes one push at a time:
a checkout with line-ending conversion on, a second Windows interpreter, a
container without an init process, and finally the publisher's own metadata
checker, which lags the build backend's default metadata version. Seven red
runs in a row, each fixing the last failure and finding the next.

The lesson is not "push earlier"; holding the push was deliberate. It is
that a push must confirm a result, not test a guess. So the release gate is
a matrix of every shape the runners and the users have, run locally before
the push, and `scripts/release_gate.sh` runs it:

| shape | why it exists |
|-|-|
| Linux, serially | CI runs the suite on one xdist worker per core; the serial run keeps the one-process order covered, where state a test leaves behind reaches every later test |
| Linux with WSL detection forced off (`scripts/nonwsl_plugin.py`) | Linux CI is not WSL; this box is, so the non-WSL branch is otherwise never executed here |
| Linux under schedule jitter, seeds 1 and 2 (`tests/schedule_jitter.py`) | races in the suite used to lose only on the Windows runner, one per run; delaying thread hand-offs and process starts, and firing timers early as Windows does, makes them lose here first. CI runs the same two seeds |
| Ubuntu container, non-root, `--init` | a fresh machine with ngspice and libcairo2; `--init` because a container whose PID 1 is `bash` never reaps a killed child, and a zombie still answers `os.kill(pid, 0)` |
| Windows native, Python 3.12 and 3.13, checkout with conversion on | the primary platform, both supported interpreters (3.13 changed `Path.resolve` on a NUL byte), and the bytes a runner with `core.autocrlf=true` sees |
| the publisher's own metadata check | the PyPI action's bundled `twine` rejected a metadata version the build backend had started emitting by default, after the build job's own newer `twine` had passed it |

One Windows shape is not in the matrix, because no runner has it: a machine
whose code page is not cp1252. Text read or written without a named encoding
is in the machine's code page there, cp936 on a Simplified Chinese install
and cp932 on a Japanese one. A test that read a UTF-8 source file that way,
and one that wrote a log holding a degree sign that way, passed on every
runner and failed on such a machine. Name the encoding in every text read and
write, in a test as in the source; `PYTHONWARNDEFAULTENCODING=1` makes Python
warn at each place that does not.

The Windows shape needs a clone on a Windows disk with the Windows-side `uv`
on PATH; point `LTSPICE_MCP_WINDOWS_CLONE` at its WSL path. Without it the
script says SKIP, loudly, rather than passing by omission. The container
shape clones the local `master`, so it tests the last commit.

The native Windows integration tier also needs:

- LTspice installed, `LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`, and
  `LTSPICE_MCP_SYMBOL_PATHS` pointing to its symbol directory.
- For the comparison with the recordings (*Recorded LTspice behaviour*), each
  build to compare, started once so that it has a settings file: the current
  build and LTspice XVII in their standard install locations, or named in
  `LTSPICE_MCP_RECORDER_EXES`. A build that is missing is skipped by name.
- The console build of ngspice available as `ngspice.exe` on PATH, with its
  accompanying DLLs available. The GUI executable does not provide the console
  output these tests inspect.
- The `raster` extra and native Cairo DLLs with their dependencies. For a
  portable installation, set `CAIROCFFI_DLL_DIRECTORIES` to the DLL directory.

With those prerequisites configured in the test shell, run
`uv run --locked --extra raster --python 3.12 pytest tests/ -v -ra` and repeat
with Python 3.13 in a separate environment. Read the skip reasons: a green run
without these dependencies does not validate simulation or PNG rendering.
Worker timeout, cancellation and descendant cleanup tests require no simulator
and run in the ordinary suite.

Parser containment currently admits Linux and native Windows. macOS remains
refused before the supervisor writes control files or launches a process; the
bootstrap also refuses before reading its admission gate or importing a decoder.
`test_parser_process.py` checks both entry points. Its portable platform-string
checks establish refusal behavior only. The `test_native_macos_bootstrap_*`
checks use an actual macOS interpreter and otherwise skip; even a native pass
there establishes refusal, not a working macOS parser backend.

Enabling macOS requires native proof of the configured hard memory bound before
decoder import and cleanup after deadline, repeated cancellation and owner
death, including attempted child-process escape. Linux or Windows checks do
not establish these macOS guarantees.

The optional Sky130 integration tests use `LTSPICE_MCP_TEST_PDK_ROOT` to
locate a `sky130A` root containing `libs.ref` and `libs.tech`.
`TestOpenPdkDevice` in `test_subckt_mismatch.py` checks parameter forwarding and
an exact per-instance threshold shift. `test_nested_targeting.py` exercises
nested edits and operating-point reads against real devices.
`test_native_experiments.py` requires the pinned model acquisition named by
`sky130-e6f9c887-ngspice-v1`; it checks all four statistical modes against direct
ngspice runs, sample replay, included-device edits, input drift, cancellation
and persisted provenance. These numerical tests need console ngspice. The model
files are supplied separately; they are not bundled with the server. Native
experiments require the matching profile's model files at runtime too.

Two practices follow. A test that depends on a POSIX facility skips on
Windows with the reason (symlinks need a privilege; `wslpath` does not
exist), never fails. A fixture that must be byte-exact is written with
`newline="\n"` and read with `encoding="utf-8"`, because the platform
newline and codec differ there; and `.gitattributes` turns conversion off for
the whole repository, so a checkout holds the committed bytes everywhere.

The upload step itself has no local proxy. The publish workflow can be
dispatched by hand against TestPyPI with an explicit version, which
exercises trusted publishing and the metadata check without spending a
release tag.

## Recorded LTspice behaviour

Development and CI run on Linux, where there is no LTspice. For a long time
"correct" therefore meant "matches what we believe LTspice does", and a test
written from the same belief protects the belief. The M90 and M270 symbol
placements were swapped, and the tests' expected pin positions had been worked
out by hand from the swapped table, so the suite defended the bug until a
user's own export disagreed. Whether LTspice XVII reads a micro sign stored as
UTF-8 was inferred, never observed.

**The rule: every LTspice behaviour the server models has a recording behind
it, from the current build and from LTspice XVII, or a written reason it
cannot have one.** When code comes to encode "LTspice does X", add an input
that makes LTspice show X, record it, and write the test against the
recording. An expected value derived by hand from the assumption under test
is not evidence for it.

### What is where

Everything is under `tests/fixtures/ltspice_recorded/`.

- `inputs/` holds the decks and sheets, kept minimal so the fixtures stay
  small, and `inputs/cases.toml`, which is the inventory. Each
  `[behaviour.<key>]` table names a behaviour, the code that models it, and
  the inputs that record it. A sheet is exported with `-netlist`; anything
  else is run with `-Run -b`. A `plot` case runs a sheet in LTspice's window
  instead, the way the person it is handed to does (*Plot settings* below). A
  behaviour with no input says why:
  `evidence` when the manifest records it some other way, `unrecordable` when
  nothing can (LTspice saves a sheet only from its window, for one).
- `ltspice26/` and `ltspice17/` hold what each build wrote, one file per
  output, and a `manifest.json`: the executable's digest, size and version,
  the build as its own output names it, the build's defaults for the settings
  the recorder neutralises, facts about its library (where it is, how each
  `standard.*` file is encoded, the pins of the stock symbols), and per case
  the command line, the digest of every input, the exit code, what was
  written, and the text of any message box the build stopped on.

The directory is named for the build's major version; XVII is 17.

### What reads the recordings

These run everywhere, with no LTspice:

|test module|holds the server to|
|-|-|
|`test_recorded_ltspice_schematics.py`|pin positions in all eight placements, wire and label connectivity, the same-instance wire rule, and how an export is spelled and encoded|
|`test_recorded_ltspice_decks.py`|value suffixes, deck encodings, the title line and comments, the card forms lint and arity accept or refuse, and what a deck means where simulators differ|
|`test_recorded_ltspice_results.py`|every raw layout, stepped runs, measurements and the angle unit of trig inside them, Fourier and device operating-point blocks, and how a failed run is classified|
|`test_recorded_ltspice_plot_settings.py`|the plot settings file each build saves (its encoding and line ends, the pane order, the Log line) and what each build shows for one the server wrote|
|`test_ltspice_recorder.py`|the recorder itself, and the tree: every listed file present with its recorded digest, every input the one that was run, every behaviour recorded on every build or explained|

They go through the code a live result goes through: the contained decoder,
the schematic editor, the lexer. A sheet is staged with the symbols LTspice
resolved, the build's own stock symbols being rebuilt from the pins the
manifest recorded.

A difference between the server and a recording is a finding. Fix the server
if the fix is small, with the recording as the regression test, which must
fail before the fix. Otherwise pin what LTspice does and what the server does
side by side in the test, under a name that says so
(`READ_AS_CP1252_BY_THE_SERVER_ONLY` in `test_recorded_ltspice_decks.py`), so
the gap is written down where the next person will find it.

### Recording again

On a Windows machine with the builds installed, from the repository root:

```bash
uv run python scripts/record_ltspice_fixtures.py
```

With no argument it records every build it finds in the standard install
locations; pass executables to choose, `LTSPICE_MCP_RECORDER_EXES` to name
builds installed elsewhere, `--only 'raw/*'` to record some cases and keep the
rest, and `--check` to record into a temporary directory and print the
differences from what is committed. Recording the same build twice gives the
same bytes, so `git diff` after a re-record shows exactly what LTspice now
does differently.

What makes that true, and what a recording must never carry:

- **No settings of the person recording.** Each case runs against a copy of
  the build's settings file with the keys that change a result removed, so
  the build is on its own defaults; a case sets one back when it is the
  point (`ini = { NoGreekMus = "true" }`). The build must have been started
  once, so that it has a settings file to copy.
- **No path, name, date or duration.** The run directory, the home directory,
  dates, elapsed times and the thread count are rewritten to fixed values, in
  the file's own encoding, a raw's samples untouched. The recorder then
  refuses to finish if a user name, a host name or a local path survived.
  A path is looked for as each build spells it: LTspice 26 in UTF-8, and
  LTspice XVII in cp1252 on a machine of any code page, with a question mark
  for each character cp1252 lacks (so a home folder named in Chinese reaches
  an XVII export as `C:\Users\??`). The machine's own code page is searched
  too.
- **No person.** LTspice opens a window even for a batch run and takes the
  keyboard focus for as long as it lasts, and it answers some inputs with a
  message box that waits for OK. On the desktop someone is working at, a
  stray key press answers it and the run looks as if it had ended by itself.
  The recorder starts LTspice on a desktop of its own, where it cannot take
  focus and nobody can answer; a build that stops to ask is recorded as
  having done so, with what it asked. The launch is the server's own
  (`lib/hidden_desktop.py`), which is why the LTspice integration tier no
  longer takes the keyboard either: watching for an LTspice window on the
  desktop the tests run on is one of its tests
  (`test_ltspice_integration.py::TestWindowStaysOffTheDesktop`).

Things about the command line that cost an afternoon each: `-ini <file>` goes
after the input (given first, LTspice 26 exits 0 having run nothing and XVII
opens its window); the settings copy is never empty (a build that starts on
an empty one behaves as on first launch, and XVII then runs its updater);
`-ascii` is ignored when a settings file is also named, and when the deck's
own file name contains "ascii".

### Plot settings

LTspice writes a plot settings file (`.plt`) only from its waveform window,
so a `plot` case drives the window (`ltspice_recorder.drive_plot`). It runs
an RC sheet with `-Run`, as a person opening it does, and finds the waveform
window of the sheet's raw file. Its `steps` are sent as the window's own menu
commands, each by the id the build's menu resource gives its label (Add trace,
Add Plot Pane Below Active Pane), read from the executable rather than from
the running window. A trace is typed into the Add Traces dialog. The case ends
with the File menu's Save Plot Settings, which writes `<sheet>.plt`. A case
with a `plot` input puts that file beside the sheet first and saves straight
away, so what the build writes is what it read. Commands go to the waveform
window's own frame: after a run the schematic window can be the active one,
and its Save saves the sheet. A dialog the case did not open is a box the
build stopped on and is recorded as one, with no `.plt` kept. A window that
does none of this in the timeout fails the recording.

The committed plot cases were recorded under Wine 11, on the same executables
as the rest of the recording (the digests match the manifest's), and each
entry says so in `host`. A recording made on Windows has no `host`, so
recording them again there with
`uv run python scripts/record_ltspice_fixtures.py --only 'plot/*'`
replaces them. A partial recording like that keeps the library facts the rest
of the recording was made with, and says so when the machine's own differ: a
Wine prefix has the library the installer unpacked, not the one the committed
manifest describes. Every other committed case was recorded again under Wine
(`--check`) to see what the host changes. On LTspice 26 every file came out
as committed but those of the two cases that run on the recording user's own
settings, which differ by design. On XVII each log ended without the blank
line that follows the matrix compiler report. Neither touches a plot case,
whose recording is the build's own serialisation of the file. Under Wine a
desktop of the recorder's own is made but the windows on it cannot be listed,
so the recorder launches on Wine's display instead and looks for a box, or the
waveform window, among the windows of the process it started there. That is
how the box XVII stops on for the two sheets with a byte order mark is
recorded under Wine as it is on Windows.

### The opt-in tier

With `LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`,
`test_ltspice_recorder.py::test_an_installed_build_still_behaves_as_recorded`
records each installed build again, a group of cases at a time, and compares
the result with what is committed. A build of a newer major version than any
recorded stands in for the newest recording of its generation, so a release
that changes behaviour fails there by name. A build that is not installed
skips with that reason; it never fails for it. Text files are compared as
bytes, raw files by header and by samples to a part in a million (a solver's
last bits depend on the processor), and a run stopped part way only by its
header.

When it fails, look at the difference before recording over it: either
LTspice changed, in which case the model may need to follow, or the recorder
missed something that varies, in which case it belongs in the scrubber.

## Conventions

- **Behavior-named test files.** Tests are named for the behavior they cover,
  never for a test campaign, date, or version. A regression found in a test
  campaign goes in the existing behavior module it belongs to; its origin goes
  in a docstring or comment, not the filename.
- **Plain-language findings.** Shipped code, docstrings, commit messages, and
  this doc describe a bug by its actual behavior in plain technical terms — no
  internal severity codes, codenames, or pass numbers (those stay in the
  internal backlog).
- **Tables** use minimal separators (`|-|-|`); no box-drawing characters.
