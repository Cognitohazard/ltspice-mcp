# Testing

This document covers the checks to run before a push or a release, the LTspice
recordings and other committed records the suite checks against, and test
conventions. The standard `pytest`, `ruff` and `pyright` commands are in
`CLAUDE.md`; the architecture is in `docs/DESIGN.md`.

## Privacy checks before pushing

Enable the repository hooks once per clone:

```bash
git config core.hooksPath .githooks
```

The pre-push hook checks commit messages, annotated tag messages, filenames and
each file revision the push introduces, including content added in one commit
and removed in a later one. For a new branch it compares against the receiving
remote's current refs, and it refuses the push when it cannot read the history
it needs. Each finding names its category and location and does not print the
matched value.

To check without pushing:

```bash
uv run python scripts/privacy_scan.py tracked   # the tracked working tree
uv run python scripts/privacy_scan.py local     # unpublished history against local remote-tracking refs
```

`local` uses the last fetched refs; the pre-push hook checks the actual
destination. The release gate runs both commands. The tracked-file test uses
the same scanner, which reads simulator artifacts in both UTF-16 byte orders.

## Platforms and the release gate

The suite is developed on WSL2 and gated on Linux CI, but most users run
Windows natively, which neither host reproduces. A Linux container on a WSL2
machine inherits the WSL kernel signature, so `is_wsl()` is true there and
every WSL branch runs.

The release gate, `scripts/release_gate.sh`, runs the suite locally before a
push in each configuration the CI runners and the users have:

| configuration | reason |
|-|-|
| Linux, serially | CI runs one xdist worker per core; a serial run puts every test in one process, so state a test leaves behind reaches every later test |
| Linux with WSL detection forced off (`scripts/nonwsl_plugin.py`) | Linux CI is not WSL; on a WSL2 machine the non-WSL branches otherwise never run |
| Linux under schedule jitter, seeds 1 and 2 (`--jitter-seed=N`, `tests/schedule_jitter.py`) | delays thread hand-offs and process starts and fires timers early, as Windows does, so a race fails on Linux too. CI runs the same two seeds. A failure under a seed usually reproduces under the same seed; `--jitter-seed=random` picks a seed and prints it |
| Ubuntu container, non-root, `--init` | a fresh machine with ngspice and libcairo2. `--init` is needed because a container whose PID 1 is `bash` never reaps a killed child, and `os.kill(pid, 0)` succeeds on a zombie |
| Windows native, Python 3.12 and 3.13, checkout with line-ending conversion on | the primary platform; both supported interpreters (3.13 changed how `Path.resolve` handles a NUL byte); and the files as a checkout with `core.autocrlf=true` writes them |
| the publisher's own metadata check | the PyPI action's bundled `twine` can reject a metadata version the build backend emits by default |

The matrix has no Windows machine whose code page is not cp1252, because no
runner has one. On such a machine, text read or written without a named
encoding uses its code page: cp936 on a Simplified Chinese install, cp932 on a
Japanese one. Name the encoding in every text read and write, in tests and in
the source; `PYTHONWARNDEFAULTENCODING=1` makes Python warn at each one that
does not.

The Windows run needs a clone on a Windows disk and the Windows `uv` on PATH;
set `LTSPICE_MCP_WINDOWS_CLONE` to the clone's WSL path. Without it the script
reports SKIP, not a pass. The container run clones the local `master`, so it
tests that branch's last commit.

The native Windows integration tier also needs:

- LTspice installed, `LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`, and
  `LTSPICE_MCP_SYMBOL_PATHS` pointing to its symbol directory.
- For the comparison with the recordings (*Recorded LTspice behaviour*): the
  current build and LTspice XVII, each in its standard install location or
  named in `LTSPICE_MCP_RECORDER_EXES`, and each started once so that it has a
  settings file. A missing build is skipped, and the skip names it.
- The console build of ngspice on PATH as `ngspice.exe`, with its DLLs. The GUI
  executable does not write the console output these tests read.
- The `raster` extra. The tests of the `cairosvg` path also need native Cairo
  DLLs with their dependencies, and skip without them. For a portable
  installation, set `CAIROCFFI_DLL_DIRECTORIES` to the DLL directory.

With those set in the test shell, run

```bash
uv run --locked --extra raster --python 3.12 pytest tests/ -v -ra
```

and repeat with Python 3.13 in a separate environment. Read the skip reasons:
a passing run without these dependencies has not tested simulation or PNG
rendering. Worker timeout, cancellation and descendant cleanup tests need no
simulator and run in the ordinary suite.

Parser containment runs on Linux and native Windows. On macOS the supervisor
refuses before it writes control files or starts a process, and the bootstrap
refuses before it reads its admission gate or imports a decoder;
`test_parser_process.py` checks both. Its portable platform-string checks test
only the refusal. The `test_native_macos_bootstrap_*` checks need a native macOS
interpreter and skip elsewhere; a pass there also shows only the refusal, not a
working macOS parser. Supporting macOS needs native tests there of the hard
memory bound before decoder import, and of cleanup after a deadline, repeated
cancellation and owner death, including a child process that tries to escape.

The optional Sky130 integration tests find a `sky130A` root containing
`libs.ref` and `libs.tech` through `LTSPICE_MCP_TEST_PDK_ROOT`, and need
console ngspice. `TestOpenPdkDevice` in `test_subckt_mismatch.py` checks
parameter forwarding and an exact per-instance threshold shift.
`test_nested_targeting.py` tests nested edits and operating-point reads on real
devices. `test_native_experiments.py` needs the pinned model acquisition
`sky130-e6f9c887-ngspice-v1`; it checks all four statistical modes against
direct ngspice runs, sample replay, included-device edits, input drift,
cancellation and persisted provenance. The model files are not bundled with
the server; native experiments also need the matching profile's model files at
runtime.

A test that depends on a POSIX facility skips on Windows with the reason
(symlinks need a privilege; `wslpath` does not exist); it does not fail. Write
a fixture that must be byte-exact with `newline="\n"` and read it with
`encoding="utf-8"`, because the platform newline and codec differ on Windows.
`.gitattributes` turns line-ending conversion off for the whole repository, so
a checkout has the committed bytes on every platform.

The upload step has no local check. To test trusted publishing and the
metadata check without a release tag, dispatch the publish workflow by hand
against TestPyPI with an explicit version.

## Recorded LTspice behaviour

Development and CI run on Linux, where there is no LTspice, so the suite checks
the server against files LTspice wrote.

**Every LTspice behaviour the server models has a recording from the current
build and from LTspice XVII, or a written reason it cannot have one.** When
code newly encodes "LTspice does X", add an input that makes LTspice do X,
record it, and test against the recording. An expected value derived by hand
from the assumption under test is not evidence for it.

### What is where

All files are under `tests/fixtures/ltspice_recorded/`.

- `inputs/` holds the decks and sheets, kept small, and the inventory
  `inputs/cases.toml`. Each `[behaviour.<key>]` table names a behaviour, the
  code that models it, and the inputs that record it. A sheet is exported with
  `-netlist`; anything else is run with `-Run -b`. A `plot` case runs a sheet
  in LTspice's window instead (*Plot settings* below), and a `save` case opens
  a sheet there and saves it (*Saved sheets* below). A behaviour with no input
  gives the reason: `evidence` when the manifest records it another way,
  `unrecordable` when it cannot be recorded.
- `ltspice26/` and `ltspice17/` hold each build's output, one file per output,
  and a `manifest.json` with the executable's digest, size and version; the
  build name its own output reports; the build's defaults for the settings the
  recorder removes; facts about its library (its location, the encoding of each
  `standard.*` file, the pins of the stock symbols); and, per case, the command
  line, the digest of every input, the exit code, the files written, and the
  text of any message box the build stopped on.

The directory is named for the build's major version; XVII is 17.

### What reads the recordings

These tests run on any machine, without LTspice:

|test module|checks|
|-|-|
|`test_recorded_ltspice_schematics.py`|pin positions in all eight placements, wire and label connectivity, node names that differ only in case and a `GND` label as ground, the same-instance wire rule, the pins of one part that share a point, where a symbol is found beside a sheet and in a library, an export's spelling and encoding, how text of more than one line is exported, and the effect on an export of a data label, a bus tap, a bus label, an empty line and an unknown keyword|
|`test_recorded_ltspice_decks.py`|value suffixes, deck encodings, the title line and comments, the card forms that lint and arity checks accept or refuse, how a deck is interpreted where simulators differ, and how a three-node transistor card is read|
|`test_recorded_ltspice_results.py`|every raw layout, stepped runs, measurements and the angle unit of trigonometric functions in them, Fourier and device operating-point blocks, a frequency response analysis's results file, and how a failed run is classified|
|`test_recorded_ltspice_sheet_save.py`|the sheet each build writes on save (line endings, encoding, record order), and that `asc_document` reads it, writes it back byte-identical, and formats a record as the build does|
|`test_recorded_ltspice_plot_settings.py`|the plot settings file each build saves (encoding and line endings, pane order, the Log line) and what each build displays for a file the server wrote|
|`test_build_benchmark.py`|the benchmark's net reading of every sheet it builds, pin by pin; the fourth node of a transistor drawn with three pins; and that each export matches the benchmark's reference circuit|
|`test_ltspice_recorder.py`|the recorder, and the tree: every listed file is present with its recorded digest, every input matches the one that was run, and every behaviour is recorded on every build or has a reason|

They read the recordings through the same code as a live result: the contained
decoder, the schematic editor, the lexer. A sheet is staged with the symbols
LTspice resolved; the build's stock symbols are rebuilt from the pins in the
manifest.

A difference between the server and a recording is a finding. If the fix is
small, fix the server, with the recording as the regression test (it must fail
before the fix). Otherwise pin LTspice's behaviour and the server's side by
side in the test, under a name that states the difference
(`READ_AS_CP1252_BY_THE_SERVER_ONLY` in `test_recorded_ltspice_decks.py`;
`EXPORTED_BY_LTSPICE_AND_REFUSED_BY_THE_EDITOR` and
`JOINED_BY_A_BUS_LABEL_IN_LTSPICE_ONLY` in
`test_recorded_ltspice_schematics.py`).

### Recording again

On a Windows machine with the builds installed, from the repository root:

```bash
uv run python scripts/record_ltspice_fixtures.py
```

With no argument it records every build it finds in the standard install
locations. Pass executables to choose builds; set `LTSPICE_MCP_RECORDER_EXES`
to name builds installed elsewhere; pass `--only 'raw/*'` to re-record matching
cases and keep the rest; pass `--check` to record into a temporary directory
and print the differences from the committed files. Re-recording the same build
is byte-identical, so `git diff` after a re-record shows what changed in
LTspice's behaviour.

The recorder keeps a recording reproducible and free of personal data:

- **No settings of the person recording.** Each case runs against a copy of
  the build's settings file with the keys that change a result
  (`BEHAVIOUR_KEYS` in `tests/ltspice_recorder.py`) removed, so the build uses
  its defaults. A case that tests a setting sets it
  (`ini = { NoGreekMus = "true" }`). The build must have been started once so
  that it has a settings file to copy.
- **No path, name, date or duration.** The run directory, the home directory,
  dates, elapsed times and the thread count are replaced with fixed values in
  the file's own encoding; a raw file's samples are not changed. The recorder
  fails if a user name, host name or local path remains. It searches for a path
  in each build's encoding: UTF-8 for LTspice 26, and cp1252 for LTspice XVII
  on a machine of any code page, with a question mark for each character
  cp1252 lacks (a home folder named in Chinese appears in an XVII export as
  `C:\Users\??`). It also searches in the machine's own code page.
- **No interaction.** LTspice opens a window even for a batch run, takes the
  keyboard focus while it runs, and on some inputs shows a message box that
  waits for OK. The recorder starts LTspice on a separate desktop with the
  server's launch (`lib/hidden_desktop.py`), where it cannot take focus and
  nobody can answer a box; when a build stops on a message box, the recording
  notes that it stopped and keeps the box's text. The LTspice integration tier
  uses the same launch, and
  `test_ltspice_integration.py::TestWindowStaysOffTheDesktop` checks that no
  LTspice window appears on the desktop the tests run on.

Command-line details:

- `-ini <file>` goes after the input. Given before it, LTspice 26 exits 0
  without running anything and XVII opens its window.
- The settings copy is never empty. A build started on an empty one behaves as
  on first launch, and XVII then runs its updater.
- `-ascii` is ignored when a settings file is also named, and when the deck's
  file name contains "ascii".

### Plot settings

LTspice writes a plot settings file (`.plt`) only from its waveform window, so
a `plot` case drives that window (`ltspice_recorder.drive_plot`). It runs an RC
sheet with `-Run`, as opening it by hand does, and finds the waveform window of
the sheet's raw file. The case's `steps` are sent as the window's menu commands
(Add trace, Add Plot Pane Below Active Pane), each by the command id the
build's menu resource assigns to its label, read from the executable. A trace
is typed into the Add Traces dialog. The case ends with File > Save Plot
Settings, which writes `<sheet>.plt`. A case with a `plot` input first puts
that file beside the sheet and saves immediately, so the saved file shows what
the build read from it. Commands are sent to the waveform window's frame,
because after a run the schematic window can be active, and its Save saves the
sheet. A dialog the case did not open is recorded as a message box the build
stopped on, and no `.plt` is kept. If the window completes none of these steps
within the timeout, the recording fails.

The committed plot entries were recorded on Windows with
`uv run python scripts/record_ltspice_fixtures.py --only 'plot/*'`, and are
byte-identical to an earlier recording of the same executables under Wine 11.
An entry recorded under Wine records that in `host`; an entry recorded on
Windows has no `host`. A partial recording keeps the library facts and settings
defaults of the rest of the recording, reports when this machine's differ, and
adds any default a run writes back that the manifest lacks.

One user setting affects a plot settings file: the waveform window's grid (the
`grid` key of the settings file). With the grid on, each build writes a
`GridStyle` line in every pane it creates and extends the Y axis with fewer
divisions to match the other (`plot/ac_grid_on`, `plot/math_grid`). A pane read
from a file keeps the file's line (`plot/read_grid`,
`plot/read_two_panes_grid`). The recorder removes the key, so each build uses
its default, off.

Under Wine, the windows on the recorder's separate desktop cannot be listed, so
the recorder launches on Wine's display and searches the started process's
windows for a message box or the waveform window. Re-recorded under Wine with
`--check`, LTspice 26's files match the committed ones except for the two cases
that run on the recording user's own settings, which differ by design; XVII's
logs lack the blank line after the matrix compiler report.

### Saved sheets

A build writes a sheet only from its window, so a `save` case drives the window
(`ltspice_recorder.drive_save`). The recorder starts the build with only the
sheet, sends the schematic window's Save by its menu command id, and ends the
case when the file has been written and stops changing. The sheet is not
edited first. If the build stops on a message box, the box is recorded and no
sheet is kept.

These are the only sheets in the suite written by LTspice. Both builds write
the same, except for the sheet's first line:

- Line endings are LF, whatever the sheet had.
- The text is 8-bit: a UTF-16 sheet is saved as 8-bit text, and UTF-8 bytes
  are kept as they are, not decoded.
- Records are grouped by kind in a fixed order (wires, flags each with its
  port line, symbols, text, drawn lines, other shapes). The build reorders
  wires by its own rule; other records keep their order.
- LTspice 26 writes `Version 4.1` for a sheet that had `Version 4`.
- Neither build opens a sheet that starts with a byte order mark.

They were recorded on Windows, so their manifest entries have no `host`.

### An open window and the bridge

`edit_schematic` keeps a sheet open in an LTspice window in sync with its file
through the MCP bridge that LTspice ships from 26.1 (`lib/ltspice_bridge.py`,
`lib/ltspice_window.py`). Recording it needs an open window, so the bridge has
its own recorder:

```bash
uv run python scripts/record_ltspice_bridge.py
```

It starts LTspice with its window on a separate desktop and connects with the
server's client, whose bridge runs on another desktop, so every recording also
tests a bridge reaching a window on a different desktop. It opens each sheet in
`tests/fixtures/ltspice_bridge_recorded/inputs/` and writes under `ltspice26/`
the window's copy of each sheet (`sheets/`), every call and its reply
(`conversation.json`), and a manifest of digests. `cases.toml` lists these
behaviours with that tree as their `evidence`. LTspice XVII has no bridge, so
only LTspice 26 is recorded here.

`test_ltspice_bridge.py` checks `sheet_content` against the recorded pairs: a
sheet LTspice only opened must compare equal to its file even when the text
differs (of 700 of the example sheets LTspice ships, read back through the
bridge, 79 came back unchanged). It also replays the recorded conversation
against `tests/fake_ltspice_bridge.py`, the fake bridge the rest of the suite
uses.
When LTspice is found to rewrite a sheet in a new way on opening it, add an
input here; do not write the rule by hand.

The manifest also lists the install's reference documents by file name and
front matter key, the fields `lib/simulator_docs.py` reads; the documents
belong to the vendor and are not recorded. The opt-in tier checks that the
documents read from the install match the list LTspice's own server reports.

The recording also covers:

- **Which traces a pane draws** (`plot-settings-on-open`). The recorder writes
  plot settings beside a results file, opens the results, and reads back the
  plot settings the pane saves. The pane draws the named traces. If the
  results are opened again while the pane is open, it keeps its old traces;
  closed and reopened, it draws the new ones, which is why
  `plot_waveform(in_ltspice=true)` closes and reopens a pane. The opt-in tier
  checks the traces the reply names the same way.
- **A sheet closed from outside** (`sheet-closed-from-outside`), which the
  server does to make LTspice look for a sheet's results again. An unchanged
  sheet closes without a prompt, its plot closes with it, and once reopened it
  finds results added beside it in the meantime. For a sheet the bridge has
  written to, LTspice asks whether to save, even when the sheet matches the
  file. The recording keeps the box's text and records that the window accepts
  no input while the box is up, that the bridge still responds, that Cancel
  leaves the sheet open unchanged, and that No closes it without writing the
  file.
- **Whether a plot is tied to its sheet** (`plot-tied-to-sheet`), so that
  clicking a net plots it. The recorder posts a grid of clicks over the sheet's
  view and reads the plot's traces before and after. A plot opened by the
  sheet's Visible Traces command changes; a plot the bridge opens, by the
  results file's path or the sheet's, does not. A run made in the window also
  ties its plot; this was seen in a window capture and is not recorded.
- **The sheet menu's Visible Traces item.** LTspice disables it for a sheet
  with no associated results; the server uses this to decide which sheet to
  reopen before sending the command.
- **The Visible Traces command**, which `plot_waveform(in_ltspice=true)` sends
  for a run of a sheet (`lib/ltspice_frame.py`). The recorder queries the
  window's frame directly and records that panes are titled with their files'
  names; that the command does not open results put beside a sheet that was
  already open; that it opens, without a prompt, results that were beside the
  sheet when it was opened; and that with those results open it shows a
  dialog, whose title is recorded. A results pane keeps the data it read when
  its file is replaced; told to close, it closes without a prompt and saves no
  plot settings, with or without a trace added by hand; reopened from the
  sheet, it shows the file's current content. The manifest records the command
  id the build's sheet menu assigns to the label. `tests/test_ltspice_frame.py`
  checks `FakeFrame`, the frame the handler tests use, against these facts.
  Seen in two window captures but not recorded: with no pane of the results
  open, the command reads the file's current content.
- **Which copy a run in the window simulates.** The recorder runs the sheet
  with one resistor value in the window and another in the file, and records
  which value the netlist beside the sheet has, and that the sheet file was not
  written.
- **Undo after a replaced copy.** Replacing the window's copy adds one step to
  its undo history. The recorder posted Ctrl+Z to the sheet's view on the
  hidden desktop, with the Control key set down in that window's key state
  (`AttachThreadInput`, then `SetKeyboardState`), and read the window's copy
  back through the bridge. After two sheets were pushed in turn, one Ctrl+Z
  restored the first, a second restored the sheet as opened, and Ctrl+Shift+Z
  restored the first again; the file was never written.
- **LTspice started with no window open** (`ltspice_windows.start_in_view`),
  which the server does when a caller asks to show something in LTspice. The
  recorder starts LTspice the same way, with no document, and records that the
  bridge finds a window with no document open and none in front; `FakeStart`,
  the start the handler tests use, is checked against that. No test starts
  LTspice on the visible desktop: the opt-in tier starts it on a separate
  desktop, and the in-view start is tested only for the command line and flags
  it passes to Windows.

One behaviour is not recorded: given a document containing a NUL character,
LTspice 26.1.1 stops responding permanently. The client refuses to send such
text (`replace_design_text`), and the test checks that it is never sent.

The bridge recording also holds LTspice's own reading of results files. The
recorder passes every results file in the main recordings' `raw` group, from
both builds, to LTspice's reader and keeps each reply under `reader/`.
`tests/test_ltspice_reader_agreement.py` checks the server's decoder against
these replies on any machine, without LTspice: every sample to the last bit, a
stepped run split at the same points, and the same parameter values per step.
When a results file in the main recordings is added or re-recorded, run the
bridge recorder again on Windows with LTspice 26.1 or later; until then one
test fails and names the script.

The replies for `raw/fra` and its `fra_1` results file, for both builds'
`raw/step_ac_one_freq`, `raw/compress_*`, `raw/subckt_names_*` and
`raw/save_subckt_names*`, and for XVII's
`raw/op_save_device`, were made under Wine 11, by the recorder's own reading
code with the same pinned executables, and the bridge manifest has no field
that says so. Made the same way, the replies for
four files already recorded on Windows (`ac`, `step_tran`, XVII's `tran`, and
the refused `op`) were byte-identical to the committed ones.

The two readers agree on every file's content. They present six kinds of file
differently, and the tests pin LTspice's reading of each:

- XVII stores some time points of a compressed transient with the sign set.
  LTspice 26's reader returns them negative; the server returns the time with
  the sign cleared.
- A transient saved from a later start time (`.tran 0 2m 1m`) starts at zero in
  LTspice's reader; the server adds the header's `Offset`.
- LTspice's reader refuses a file of one point (an operating point, a transfer
  function); the server reads it.
- LTspice's reader refuses a file that holds only its axis, a run whose
  `.save` named nothing it stored, since it reads signals by name; the server
  reads its axis.
- LTspice's reader reads a stepped operating point, and a stepped AC analysis
  at one frequency, as one sweep with the stepped parameter as its axis; the
  server keeps one point per step. Both files store one point a step with the
  stepped parameter first and no frequency.
- LTspice's reader reads a run stopped part way up to its header's point count,
  which can lag the samples; the server's decoder refuses a file whose header
  and samples disagree, and reports how far the run got by other means.

Other bridges may be running on the machine, each with its own hidden LTspice
that starts and exits as windows open and close (an assistant session with
LTspice's own MCP server registered runs one). A recording includes only the
LTspice the recorder started.

The recording disables the bridge's own LTspice launch. The opt-in tier tests
that launch: `TestSheetOpenInAWindow` starts the bridge without blocking the
launch and checks that the LTspice it starts has its windows on the server's
desktop, none on the test's desktop, and exits when the session closes.

### The opt-in tier

With `LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1`,
`test_ltspice_recorder.py::test_an_installed_build_still_behaves_as_recorded`
re-records each installed build, one group of cases at a time, and compares the
result with the committed files. A build with a newer major version than any
recorded one is compared with the newest recording of its generation, so a
release that changes behaviour fails this test, which names the build. A build
that is not installed is skipped with that reason, not failed. Text files are
compared as bytes; raw files by header and by samples to one part in a million
(a solver's last bits depend on the processor); a run stopped part way by its
header only.

When it fails, read the difference before re-recording: either LTspice changed
and the server's model may need to change, or the recorder missed something
that varies. Add a path, date or duration to the scrubber (`Scrubber` in
`tests/ltspice_recorder.py`). A file that differs for another person on the
same build comes from one of their settings; add its key to `BEHAVIOUR_KEYS`
(the waveform grid was one, *Plot settings* above).

## Re-recording the suite's records

Several tests check the code against a committed JSON record of its answers on
every sheet in the suite. A difference is either a regression or an intended
change; for an intended change, re-record the file in the same commit, so
`git diff` shows what changed. Each record uses only the suite's own symbols,
so it is the same on every machine. To re-record one, run its test module with
its variable set to `1`, for example:

```bash
LTSPICE_MCP_RECORD_SHEET_FINDINGS=1 uv run pytest tests/test_sheet_findings_snapshot.py
```

|variable|test module|record|
|-|-|-|
|`LTSPICE_MCP_RECORD_SHEET_FINDINGS`|`tests/test_sheet_findings_snapshot.py`|`tests/fixtures/sheet_findings.json`: the findings `edit_schematic` and `verify_circuit` each report for every `.asc` under `tests/`|
|`LTSPICE_MCP_RECORD_ROUTE_PLANNER`|`tests/test_route_planner_record.py`|`tests/fixtures/route_planner_record.json`: the `wire_pins` route planner's results for a fixed set of proposals on every sheet|
|`LTSPICE_MCP_RECORD_LABELS_AND_MOVES`|`tests/test_label_and_move_record.py`|`tests/fixtures/label_and_move_record.json`: the `add_net_label` and `move_component` results for a fixed set of proposals on every sheet|
|`LTSPICE_MCP_RECORD_INSPECT_SHEETS`|`tests/test_inspect_sheet_record.py`|`tests/fixtures/inspect_sheet_record.json`: the `inspect(kind="net")` and `inspect(kind="components")` replies for every sheet|
|`LTSPICE_MCP_RECORD_BUILD_BENCHMARK`|`tests/test_build_benchmark.py`|`tests/fixtures/build_benchmark_record.json` and the built sheets (below)|

`tests/fixtures/sheet_edit_record.json` holds what every `edit_schematic` call
the suite makes answers: the committed sheet, read as records, and the reply.
`tests/sheet_edit_record.py`, a plugin loaded by `tests/conftest.py`, compares
each call with it and fails the test whose call answers differently; its
docstring says how a call is keyed. To re-record it, run the whole suite with
`LTSPICE_MCP_RECORD_SHEET_EDITS=1` with no LTspice library installed or
visible. A run that finds a symbol in an LTspice library writes nothing,
because CI has no library and would make different keys. Mark a test whose
`edit_schematic` replies differ on Windows and Linux with
`@pytest.mark.edit_reply_per_platform(reason)`. Its calls are then compared
only with a recording made on the same platform.

### The schematic build benchmark

`tests/test_build_benchmark.py` draws eleven reference netlists
(`tests/fixtures/build_benchmark/`) through `edit_schematic` with the scripted
builds in `tests/_benchmark_builds.py`, and `tests/_build_benchmark.py` scores
each build on equivalence with the netlist, on cost (calls, ops, refusals,
coordinates written, reply size) and on the drawing (each sheet rule's
findings, crossings, bends, wire length, pins reached only by a label). The
test checks the scores and the built sheets against the record.

`LTSPICE_MCP_RECORD_BUILD_BENCHMARK=1` writes the record and the sheets. The
sheets are stored under `tests/fixtures/ltspice_recorded/inputs/benchmark/` as
inputs of the LTspice recordings, so a changed sheet needs its export
re-recorded (`--only 'benchmark/*'`, *Recording again* above), and the four
records above that read every sheet under `tests/` gain or change entries for
it. To evaluate a new op, add a build that uses it to draw an existing
benchmark circuit, and compare its cost with the other builds in the record.

To print the record, or to score a sheet built outside the benchmark with the
same functions:

```bash
uv run python scripts/build_benchmark.py table
uv run python scripts/build_benchmark.py score SHEET.asc NETLIST.cir
```

## Conventions

- **Behavior-named test files.** Name a test file for the behavior it covers,
  never for a test campaign, date or version. Put a regression found in a test
  campaign in the existing module for its behavior, and note its origin in a
  docstring or comment, not the filename.
- **Plain-language findings.** Shipped code, docstrings, commit messages and
  this doc describe a bug by its actual behavior in plain technical terms, with
  no internal severity codes, codenames or pass numbers.
- **Tables** use minimal separators (`|-|-|`); no box-drawing characters.
