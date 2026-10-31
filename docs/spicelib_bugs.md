# spicelib bug reports

Upstream [spicelib](https://github.com/nunobrum/spicelib) bugs this project has
hit. Each section is a self-contained draft for an upstream pull request or
describes a known limitation that we accept.

spicelib is a pinned dependency we cannot fix in place, so we work around its
bugs and delete the workaround once upstream is fixed. This file lists the
workarounds to delete, and each reproduction lets someone confirm a fix.

If you hit a new one, add a section in this format: summary, affected code and
version, reproduction, impact, proposed fix, a suggested upstream test, and a
cross-reference to our workaround and the test that pins it. Do that even when
you also ship a workaround.

---

## Bug 1 — `.MEAS` AT/WHEN crossing time dropped on whitespace-padded lines

**Status:** draft for an upstream spicelib pull request. Filed 2026-06-28.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py`). The bug has
been in the single-line measurement regex for several releases.
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — `_RE_MEAS_AT_LINE`
plus the `at_overrides` backfill in `parse_measurements`. Remove it once
upstream is fixed. (The backfill fills `at` only when spicelib left it unset,
so it has no effect with a fixed spicelib, but it is then unused code.)

### Summary

`LTSpiceLogReader` parses the single-line (stepless) `.MEAS` result format with
a regex whose `AT` clause and `FROM`/`TO` clause are written with **literal
single spaces**. When LTspice pads the result line with more than one space (or
a tab) before `AT`/`FROM`, the optional clause fails to match. The measurement
is still returned, but **the crossing time (`AT`) and window bounds
(`FROM`/`TO`) are silently dropped**. The value is returned as the trigger
level alone, with no `_at` / `_FROM` / `_TO` companion column.

A transient WHEN measurement is the standard way to measure lock time, startup
time, propagation delay, settling time and threshold-crossing time. For such a
measurement the **crossing time, which is the result the user wants, is
lost**. The reader returns the trigger level the user typed (e.g. `0.5`)
instead, which looks plausible and is reported with no error.

### Affected code

`spicelib/log/ltsteps.py`, the stepless measurement regex (line ~312–315):

```python
regx = re.compile(
        r"^(?P<name>\w+)(:\s+.*)?=(?P<value>[\d(inf)E+\-\(\)dB,°(-/\w]+)( FROM (?P<from>[\d\.E+-]*) TO (?P<to>[\d\.E+-]*)|( at (?P<at>[\d\.E+-]*)))?",
        re.IGNORECASE)
```

The clause literals are `" FROM "`, `" TO "` and `" at "`, each with exactly one
leading/trailing space. `re.IGNORECASE` is set, so `AT` matches `at` and case
does not cause the failure. The **fixed single spaces** cause it. The
`(?P<value>…)` character class has no whitespace, so the value token ends at the
first space. The optional clause must then match starting at the *next*
character, which it expects to be a single space followed by `at`. A second
space (or a tab) there breaks the match. Because the whole optional group is
`?`-optional, it is skipped, including the `at` capture.

The stepped/table path (line ~440, `tokens[2] == "FROM" or tokens[2] == 'at'`)
is a separate parser and is **not** affected, since it splits on tabs.

### Reproduction

A real LTspice transient WHEN line, padded with a double space before `AT`
(observed from LTspice 26 output on
`.meas TRAN tcross WHEN V(out)=0.5 RISE=1`):

```
tcross: V(out)=0.5  AT 0.000693147672285
```

```python
from spicelib.log.ltsteps import LTSpiceLogReader
r = LTSpiceLogReader("with_double_space.log")
r.get_measure_names()          # -> ['tcross']  (no 'tcross_at')
r.dataset['tcross']            # -> [0.5]       (the trigger LEVEL, not the time)
r.dataset.get('tcross_at')     # -> None / KeyError  <-- crossing time lost
```

The single-space form parses correctly:

```
tcross: V(out)=0.5 AT 0.000693147672285     # single space -> tcross_at == 0.000693...
```

Whether the crossing time is kept therefore depends on how many spaces LTspice
happens to write on the result line, and a loss raises no error.

### Why LTspice produces the extra space

LTspice right-pads and aligns some result fields, and the exact spacing varies
with the value's sign and width, so the same `.meas` form can emit one space on
one run and two on another. A recorded single-space sample is stored in
`tests/fixtures/ltspice_sweep_meas_run0.log`; the double-space variant triggers
the loss.

### Impact

- Any transient `.MEAS … WHEN`/`AT` result whose line happens to be
  whitespace-padded loses its crossing time. These measurements are the
  standard way to measure lock, settling, delay and startup times.
- `FROM`/`TO` windowed measurements (`RMS(...) FROM a TO b`) lose their bounds
  the same way if padded.
- The failure is silent. The measurement still appears with a
  plausible-looking value (the level), and no error or warning is raised.

### Proposed fix

Make the clause separators whitespace-tolerant. Minimal change: replace the
literal spaces around `FROM`/`TO`/`at` with `\s+`, keeping `re.IGNORECASE`:

```python
regx = re.compile(
    r"^(?P<name>\w+)(:\s+.*)?=(?P<value>[\d(inf)E+\-\(\)dB,°(-/\w]+)"
    r"(\s+FROM\s+(?P<from>[\d\.E+-]*)\s+TO\s+(?P<to>[\d\.E+-]*)"
    r"|\s+at\s+(?P<at>[\d\.E+-]*))?",
    re.IGNORECASE)
```

Notes:

- The leading space before `FROM`/`at` becomes `\s+`, so the value token (which
  already stops at the first whitespace) is followed by one or more spaces or a
  tab.
- `[\d\.E+-]*` for the captured numbers already handles `0.000693147672285` and
  `6.98e-05` (lowercase `e` matches `E` under `IGNORECASE`).
- Consider also allowing the optional trailing `=> Interval` / `=> Point` /
  `=> Parameter` annotation explicitly rather than leaving it to the value
  class. This is outside the scope of this bug.

### Suggested upstream test

```python
def test_meas_at_clause_tolerates_extra_whitespace(tmp_path):
    log = tmp_path / "when.log"
    log.write_text(
        "Circuit: * t\n\n"
        "tcross: V(out)=0.5  AT 0.000693147672285\n"    # double space
        "vrms: RMS(v(out))=1.41109  FROM 0  TO 0.001\n" # double spaces
        "Total elapsed time: 0.001 seconds.\n"
    )
    r = LTSpiceLogReader(str(log))
    assert r.dataset['tcross_at'][0] == pytest.approx(0.000693147672285)
    assert r.dataset['vrms_from'][0] == pytest.approx(0.0)
    assert r.dataset['vrms_to'][0] == pytest.approx(0.001)
```

### Cross-reference

Downstream: `tests/test_log_parser.py::TestParseMeasurementsValid::test_when_crossing_with_padded_at_clause_backfilled`
and `::test_window_from_to_not_misread_as_at` pin our backfill behavior. They
will keep passing against a fixed spicelib. Once the upstream fix is released,
delete `_RE_MEAS_AT_LINE` and the `at_overrides` block in `parse_measurements`,
and point those tests at the spicelib path, which will then be correct.

---

## Bug 2 — `_get_text_space` places directives off-canvas on a low-content schematic

**Status:** draft for an upstream spicelib pull request. Filed 2026-07-04.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor._get_text_space`, ~line 590).
**Our workaround:** none needed, because we call neither `add_instruction` nor
`set_parameter`, the two callers of `_get_text_space`. Every directive or
comment our schematic editor adds goes through our own placement
(`_append_asc_text` in `src/ltspice_mcp/lib/schematic_ops.py`). It writes the
text at the coordinates the caller gives, or at `(16, 16)` when none are given,
and moves it down when another text already occupies that point. Unlike
`add_instruction`, it does not replace an existing analysis directive (with
`add_instruction`, adding `.ac` after `.tran` replaces the `.tran`).

### Summary

`AscEditor.add_instruction` places a new directive at the coordinate returned
by `_get_text_space()`, and `set_parameter` places a new `.param` line there
too. On a schematic with **no wires, labels,
directives or components**, that coordinate is far **off-canvas** — e.g.
`(880, -99976)` on the default 880 x 680 sheet — so the added text is invisible
in the LTspice view. (The netlist still exports correctly; position is
cosmetic.)

This happens because `_get_text_space` seeds its bounding box's **minimums**
from the sheet but never its **maximums**. `max_x` and `max_y` are initialized
to the sentinel `-100000` and are only updated by iterating over existing
wires, labels, directives and components. With no content to update them, `max_y`
stays `-100000`, and the method returns `(min_x, max_y + 24)` = `(880, -99976)`.

A second issue is that `min_x` is seeded from the sheet **width** (`880`)
instead of the left edge (`0`), so even the x coordinate is the far-right edge
rather than the "bottom left corner" that the code comment describes.

### Affected code

`spicelib/editor/asc_editor.py`, `_get_text_space` (~line 590):

```python
def _get_text_space(self):
    """Returns the coordinate ... where a text can be appended."""
    min_x = 100000
    max_x = -100000
    min_y = 100000
    max_y = -100000
    _, x, y = self.sheet.split()      # e.g. "1 880 680" -> x="880", y="680"
    min_x = min(min_x, int(x))        # seeds MIN from the sheet ...
    min_y = min(min_y, int(y))        # ... but max_x / max_y are NEVER seeded
    for wire in self.wires:           # only content updates the maxes
        ...
    for component in self.components.values():
        ...
    return min_x, max_y + 24          # empty schematic -> (880, -100000 + 24)
```

### Reproduction

```python
from spicelib.editor.asc_editor import AscEditor
open("empty.asc", "w").write("Version 4\nSHEET 1 880 680\n")
ed = AscEditor("empty.asc")
print(ed._get_text_space())     # (880, -99976)  <-- off-canvas
ed.add_instruction(".tran 5m")
print(ed._get_text_space())     # (880, -99952)  <-- still off-canvas
```

Both coordinates are about 100000 units above the 0..680 sheet, so the
directive renders far outside the visible schematic.

### Impact

- Adding a directive to a freshly created (empty) `.asc` via
  `add_instruction` places it off-canvas. Subsequent adds stagger downward
  (`max_y + 24` each time) but remain off-canvas until enough on-canvas content
  exists to dominate `max_y`.
- No error is raised, because the `.asc` and netlist are valid. The text is
  placed where a person cannot see it.
- The `min_x = sheet width` seeding also biases placement to the right edge even
  once content exists, contradicting the "bottom left corner" intent.

### Proposed fix

Seed the bounding box from the sheet **rectangle** (both corners), so an empty
schematic gets an on-canvas position at the bottom left instead of the
`-100000` sentinel:

```python
def _get_text_space(self):
    _, width, height = self.sheet.split()
    width, height = int(width), int(height)
    min_x, max_x = 0, width
    min_y, max_y = 0, height
    for wire in self.wires:
        ...
    ...
    return min_x, max_y + 24     # empty sheet -> (0, height + 24), on-canvas bottom-left
```

This also fixes the x bias (left edge `0` instead of the sheet width) and keeps
the existing "24 below the lowest content" behavior once content is present.

### Suggested upstream test

```python
def test_get_text_space_on_empty_sheet_is_on_canvas(tmp_path):
    p = tmp_path / "empty.asc"
    p.write_text("Version 4\nSHEET 1 880 680\n")
    ed = AscEditor(str(p))
    x, y = ed._get_text_space()
    assert 0 <= x <= 880
    assert 0 <= y <= 680 + 24     # just below the sheet, not ~-100000
```

### Cross-reference

Downstream: `tests/test_circuit_asc.py::test_standalone_directives_no_coords_do_not_stack`
and `::test_stacked_directives_auto_shift` pin that our placement gives
directives added without coordinates distinct positions. They do not touch
`_get_text_space`, so a fixed spicelib needs no downstream change.

---

## Bug 3 — multi-plot ASCII raw hangs the parser in a CPU-bound infinite loop

**Status:** draft for an upstream spicelib pull request. Filed 2026-07-13.
**Affected version:** spicelib 1.5.1 (`spicelib/raw/plot_data.py`,
`PlotData._read_ascii_vector`, the trailing-empty-line skip at ~line 401).
**Our workaround:** the server reads raw files with `decode_raw` in
`src/ltspice_mcp/lib/raw_decode.py`. It first finds each plot's
byte range with its own header scan (`preflight_raw` in `raw_header.py`), then
gives spicelib's `PlotData` a file view (`_PlotView`) that ends where that plot
ends. The skip loop therefore reads EOF where it would have met the next plot's
`Title:` line. Separately, `src/ltspice_mcp/lib/raw_parser.py` installs
`_MultiPlotAsciiGuard` over `_read_ascii_vector` when the module is imported
(`_install_multiplot_ascii_guard()`), so it also protects a direct `RawRead`
call in any process that has imported that module. The guard wraps the file
object passed to the ASCII reader. When the skip loop seeks back onto a line it
has just read as non-empty, the wrapper returns one empty read, which breaks the
loop and leaves the cursor on the next plot's header. It also adds a stall
backstop (`_STALL_LIMIT`) that raises `RuntimeError` once more than 128 reads
in a row fail to advance past the furthest byte offset reached so far. The
guard recognises the read and seek pattern rather than the loop's code, so it
has no effect on a spicelib that breaks the loop correctly.

### Summary

`PlotData._read_ascii_vector` ends each ASCII plot by skipping trailing blank
lines. The loop reads a line; if it is **non-empty** it seeks back to the line's
start (to leave it for the caller) — but then loops and **re-reads the same
line**, because it only `break`s on an **empty** line. When an ASCII raw
contains a **second plot with no blank-line separator** before it, the first
non-empty line the loop meets is the next plot's `Title:` header, so it seeks
back and re-reads that header forever in a CPU-bound infinite loop.

ngspice writes a `.noise` result this way, as two plots (noise spectral
density, then integrated noise) with no blank line between them. A single
successful `.noise` run therefore hangs any program that parses the raw.
Because a CPU-bound loop holds the GIL, moving the parse to a thread does not
help; the loop starves the whole process.

### Affected code

`spicelib/raw/plot_data.py`, end of `_read_ascii_vector` (~line 401):

```python
        # Remaining empty lines if exist are ignored
        while True:
            cursor = raw_file.tell()
            line = raw_file.readline().strip()
            if len(line) != 0:
                raw_file.seek(cursor)  # go back to the beginning of the line
            else:
                break
```

The `len(line) != 0` branch seeks back but does **not** `break`, so the next
iteration reads the same line again. The loop terminates only on an empty line
or EOF (`readline()` returns `''`, length 0, break); a non-empty, non-EOF line
loops forever.

### Reproduction

An ASCII raw with two plots and no blank line between them, as ngspice writes
for `.noise`. The second plot's `Title:`/`Plotname:` header directly follows
the first plot's last `Values:` row.

```
Title: * noise
Plotname: Noise Spectral Density
Flags: real
No. Variables: 2
No. Points: 1
Variables:
	0	frequency	frequency
	1	onoise_spectrum	voltage
Values:
0	1.0	2.0
Title: * noise            <-- next plot, no blank line before it
Plotname: Integrated Noise
Flags: real
...
```

```python
from spicelib.raw.raw_read import RawRead
RawRead("two_plot_noise.raw")   # never returns: CPU-bound infinite loop
```

### Impact

- Any multi-plot ASCII raw with no blank-line separator between plots hangs the
  parser. ngspice writes `.noise` results this way. The parser hangs
  instead of raising an error, so a caller that parses synchronously hangs
  until it is killed.
- Running the parse on a thread off the main loop does not help, because the
  loop is CPU-bound and holds the GIL.

### Proposed fix

`break` after seeking back on the next plot's header, and keep skipping across
multiple blank lines (stopping at the first content line or EOF, cursor left
before it):

```python
        # Skip trailing blank lines; stop at the next plot's first content line.
        while True:
            cursor = raw_file.tell()
            line = raw_file.readline()
            if line == "":            # EOF
                break
            if line.strip():          # next plot's header — leave it for the caller
                raw_file.seek(cursor)
                break
            # empty line: keep skipping
```

### Suggested upstream test

```python
def test_ascii_raw_with_adjacent_plots_terminates(tmp_path):
    p = tmp_path / "two_plot.raw"
    p.write_text(<two adjacent ASCII plots, no blank line — see Reproduction>)
    raw = RawRead(str(p))            # must return, not hang
    assert raw.get_trace("onoise_spectrum") is not None
    assert raw.get_trace("inoise_total") is not None
```

### Cross-reference

Downstream: `tests/test_raw_parser.py::TestMultiPlotNoiseRaw` pins the guard —
`test_guard_breaks_reread_of_next_plot_header` and
`test_two_plot_noise_raw_parses_without_hanging` (parsed in a worker thread with
a hard deadline, so a regression fails fast instead of hanging the suite), plus
`test_stall_backstop_aborts_a_nonprogressing_loop` and
`test_stall_backstop_allows_a_large_forward_read` for the stall backstop.
`tests/test_raw_decode.py::test_recorded_ngspice_ascii_preserves_both_plot_quantities`
decodes a recorded ngspice `.noise` raw
(`tests/fixtures/ngspice_noise_2plot.raw`) through the server's reader and
checks both plots. Once the upstream fix is released, delete
`_MultiPlotAsciiGuard` and `_install_multiplot_ascii_guard()` and point those
tests at the fixed reader. `_PlotView` stays, because it also refuses any read
outside the plot's checked byte range.

---

## Bug 4 — windowed-transient axis `Offset:` parsed but never applied

**Status:** draft for an upstream spicelib pull request. We work around this
known limitation.
**Affected version:** spicelib 1.5.1 (`spicelib/raw/raw_read.py`
`RawRead.get_axis` ~line 545, into `spicelib/raw/plot_data.py`
`PlotData.get_axis` ~line 779).
**Our workaround:** the server's raw reader (`decode_raw` in
`src/ltspice_mcp/lib/raw_decode.py`) takes each trace's stored samples from
`PlotData` unchanged and never calls `get_axis`. `DecodedPlot.__init__` in
`src/ltspice_mcp/lib/decoded_raw.py` then corrects the time axis of a transient
plot by taking the absolute value of the stored times and adding the header's
`Offset:` value (`transient_time_offset`). `OffsetAwareRawRead` in
`src/ltspice_mcp/lib/raw_parser.py`, a `RawRead` subclass that adds the offset
in `get_axis`, applies the same correction but has no caller in the server.

### Summary

For a **windowed** transient run — `.tran 0 <tstop> <tstart>` with a nonzero
`<tstart>` — LTspice writes the raw with the **time axis rebased to 0** and the
true start time recorded in the header's `Offset:` field. spicelib parses the
header (including `Offset:`) into `raw_params`, but `get_axis()` returns the
stored zero-based axis **without adding the offset back**. Every caller of the
axis therefore receives the rebased times. A `.tran 0 202u 196u` run, which the
user asked to observe over 196–202 us, reads back as **0–6 us**.

No error is raised. The data has the correct shape and only the absolute time
coordinates are wrong, so any value computed against absolute time (a
`.meas ... AT`, the start or end time of a signal-statistics window, an
exported waveform's time column) is off by `<tstart>`.

### Affected code

`spicelib/raw/raw_read.py`, `RawRead.get_axis` (~line 545):

```python
def get_axis(self, step: int = 0):
    ...
    return self._plots[0].get_axis(step)   # applies the 2nd-order-compression
                                           # abs() fix, but never raw_params["Offset"]
```

`Offset:` is documented in the header (line ~71) and read into `raw_params`, but
neither `RawRead.get_axis` nor `PlotData.get_axis` (~line 779) reads it back.

### Reproduction

```python
from spicelib.raw.raw_read import RawRead
raw = RawRead("windowed_tran.raw")     # from `.tran 0 202u 196u`
raw.raw_params["Offset"]               # '1.96e-04'  (parsed, present)
raw.get_axis()[0], raw.get_axis()[-1]  # ~0.0 .. ~6e-06  <-- should be 1.96e-4 .. 2.02e-4
```

### Impact

- Any windowed `.tran 0 tstop tstart` with `tstart > 0` reads back on a 0-based
  axis, so absolute-time measurements, windows and exports are shifted by
  `tstart`.
- No error is raised, because the shape is right and only the time
  coordinates are wrong.
- Only transient plots with a nonzero `Offset:` are affected. Other analyses'
  first axis is not time, and LTspice writes `Offset: 0` for unwindowed runs.

### Proposed fix

Apply `raw_params["Offset"]` to the time axis of transient plots in `get_axis`
(after the existing negative-value/compression fix), guarded to transient
plotnames and a nonzero offset:

```python
def get_axis(self, step=0):
    axis = self._plots[0].get_axis(step)
    offset = float(self.raw_params.get("Offset", 0) or 0)
    if offset and "transient" in str(self.raw_params.get("Plotname", "")).lower():
        return np.asarray(axis) + offset
    return axis
```

### Suggested upstream test

```python
def test_windowed_tran_axis_includes_offset(windowed_raw):
    # raw from `.tran 0 202u 196u`: stored axis 0..6u, header Offset: 1.96e-4
    raw = RawRead(str(windowed_raw))
    axis = raw.get_axis()
    assert axis[0] == pytest.approx(1.96e-4, rel=1e-3)
    assert axis[-1] == pytest.approx(2.02e-4, rel=1e-3)
```

### Cross-reference

Downstream: `tests/test_decoded_raw.py::test_transient_offset_once_on_later_plot_and_every_axis_accessor`
and `::test_mixed_case_offset_field_rebases_axis_without_changing_header` pin
that the server's decoded plot adds the offset exactly once, on every axis
accessor. `tests/test_raw_parser.py::TestOffsetAwareRawRead` pins that
`OffsetAwareRawRead` adds a nonzero offset to a transient axis and leaves
zero-offset and non-transient plots untouched. A fix in `RawRead.get_axis` does
not change the server's output, because the server reads the stored samples, so
the correction in `DecodedPlot` stays. Once the upstream fix is released, delete
`OffsetAwareRawRead` and its tests.

---

## Bug 5 — Fourier/THD parse crashes on a `-nan`/`inf` distortion line

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py` ~lines 354/357).
**Our workaround:** `make_log_reader` in `src/ltspice_mcp/lib/log_parser.py`.
When `LTSpiceLogReader` raises on a log, it retries on a temporary copy in which
`_sanitize_log_for_reader` (with the regex `_RE_FOURIER_NAN`) has rewritten
each `-nan%`/`inf%` THD or PHD line to `0.0%`.

### Summary

When LTspice runs a `.four`/THD analysis on an identically zero or
non-oscillating signal, it writes `Total Harmonic Distortion: -nan%` (or
`inf%`). spicelib parses that line with
`float(re.search(r"\d+.\d+", line).group())`. The regex needs a
digit-dot-digit run, which `-nan%` and `inf%` lack, so `re.search` returns
`None` and `None.group()` raises `AttributeError`. This **aborts the entire
`.log` parse**, so every `.MEAS` result and error line in the same log is lost
along with the Fourier block.

### Affected code

`spicelib/log/ltsteps.py` (~line 354):

```python
if line.startswith("Total Harmonic"):
    thd = float(re.search(r"\d+.\d+", line).group())   # None.group() on "-nan%"
elif line.startswith("Partial Harmonic"):
    phd = float(re.search(r"\d+.\d+", line).group())
```

### Reproduction

```python
from spicelib.log.ltsteps import LTSpiceLogReader
# a .log whose Fourier block reads: "Total Harmonic Distortion: -nan%"
LTSpiceLogReader("zero_signal_four.log")   # AttributeError: 'NoneType' has no 'group'
```

### Impact

- A single `-nan`/`inf` THD line, which is common for a zero or rail-pinned
  signal, crashes the whole log read, so all `.MEAS` scalars and diagnostics
  from that run are lost. The reader raises an uncaught exception instead of
  skipping the block.

### Proposed fix

Guard the search instead of unconditionally calling `.group()`:

```python
m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
thd = float(m.group()) if m else float("nan")
```

### Suggested upstream test

```python
def test_fourier_nan_thd_does_not_crash(tmp_path):
    log = tmp_path / "four.log"
    log.write_text("...\nTotal Harmonic Distortion: -nan%\n\n...\n")
    LTSpiceLogReader(str(log))   # must not raise
```

### Cross-reference

Downstream: `tests/test_log_parser.py::...::test_nan_thd_does_not_crash`. Once
the upstream fix is released, delete `_RE_FOURIER_NAN` and
`_sanitize_log_for_reader`.

---

## Bug 6 — empty `SYMATTR` value makes an `.asc` unreadable (`ValueError` on load)

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py` ~line 186).
**Our workaround:** we never write a two-token `SYMATTR` line, so spicelib can
always read back what we save. In `src/ltspice_mcp/lib/schematic_ops.py`,
`_reject_empty_attr_value` refuses an empty attribute value when a component is
added. Setting an existing attribute to an empty value removes its `SYMATTR`
line instead, and `_require_clearable_attr` refuses that for `InstName`.

### Summary

`AscEditor` reads each `SYMATTR` line with
`tag, ref, text = line.split(maxsplit=2)`, which assumes three tokens. LTspice
writes a valid empty attribute as a **two-token** line (`SYMATTR Value` with no
value), so the unpack raises
`ValueError: not enough values to unpack (expected 3, got 2)` and the whole
`.asc` fails to parse. Any file LTspice wrote with a blank attribute field is
unreadable by spicelib.

### Affected code

`spicelib/editor/asc_editor.py` (~line 186):

```python
elif line.startswith("SYMATTR"):
    ...
    tag, ref, text = line.split(maxsplit=2)   # ValueError on "SYMATTR Value"
```

### Reproduction

```python
open("empty_attr.asc", "w").write(
    "Version 4\nSHEET 1 880 680\n"
    "SYMBOL res 100 100 R0\nSYMATTR InstName R1\nSYMATTR Value\n")  # empty Value
from spicelib.editor.asc_editor import AscEditor
AscEditor("empty_attr.asc")   # ValueError: not enough values to unpack
```

### Impact

- A `.asc` with any blank SYMATTR field (LTspice writes these) cannot be
  opened, read, edited or netlisted through spicelib, because loading the file
  fails before any operation can run.

### Proposed fix

Split with a default for the missing value:

```python
parts = line.split(maxsplit=2)
tag, ref = parts[0], parts[1]
text = parts[2].strip() if len(parts) > 2 else ""
```

### Suggested upstream test

```python
def test_symattr_with_empty_value_loads(tmp_path):
    p = tmp_path / "e.asc"
    p.write_text("Version 4\nSHEET 1 880 680\nSYMBOL res 0 0 R0\n"
                 "SYMATTR InstName R1\nSYMATTR Value\n")
    AscEditor(str(p))   # must not raise
```

### Cross-reference

Downstream: `tests/test_circuit_asc.py::TestEmptyAttributeHandling` pins the
refusals and the removal of a cleared attribute. Our guard applies only when
writing. A fixed spicelib would also let us read such files, which we currently
cannot.

---

## Bug 7 — failed `.MEAS` names dropped from the reader (silent absence)

**Status:** known limitation; we work around it.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py`,
`LTSpiceLogReader`).
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — `_RE_MEAS_FAILED`
reads failed measurement names from the log text. `parse_measurements` returns
each one with a `None` value for every step and also lists the names under
`failed_measurements`, so a measurement that was requested and failed is
reported instead of missing.

### Summary

When a `.MEAS` fails, LTspice writes `Measurement "name" FAILed` to the log and
no result row. `LTSpiceLogReader` parses only successful result rows, so
`get_measure_names()` **omits the failed measure entirely**, so a measure that
was requested and failed cannot be told apart from one that was never
requested. A caller that checks whether its measurement passed gets no answer
and no error.

### Affected code

`spicelib/log/ltsteps.py` — the measurement parser matches result lines of the
form `name=value [...]`; there is no branch for the `Measurement "name" FAILed`
line, so failed names never enter `dataset` or `get_measure_names()`.

### Reproduction

```python
# .log containing:  Measurement "trise" FAILed
from spicelib.log.ltsteps import LTSpiceLogReader
r = LTSpiceLogReader("with_failed_meas.log")
"trise" in r.get_measure_names()   # False — the failure is invisible
```

### Impact

- A failed `.MEAS` is missing from the results and no error is raised.
  Downstream code cannot tell "failed" from "not requested", and may report
  success because no failure is listed.

### Proposed fix

Parse `Measurement "<name>" FAILed` lines and register `<name>` with a
`None`/`nan` value (and optionally a `failed` flag), so the name is present with
a distinguishable value.

### Suggested upstream test

```python
def test_failed_measurement_name_is_present(tmp_path):
    log = tmp_path / "f.log"
    log.write_text('...\nMeasurement "trise" FAILed\n...\n')
    r = LTSpiceLogReader(str(log))
    assert "trise" in r.get_measure_names()
```

### Cross-reference

Downstream: `tests/test_log_parser.py::TestParseMeasurementsFailed` pins the
`failed_measurements` list and the `None` values that `parse_measurements`
(`lib/log_parser.py`) returns. Once upstream reports failed names, delete
`_RE_MEAS_FAILED`.

---

## Bug 8 — cp1252 operating-point log misdetected as utf-16 and silently garbled

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/utils/detect_encoding.py`,
`detect_encoding` called with no `expected_pattern`).
**Our workaround:** `read_device_op_points` in
`src/ltspice_mcp/lib/log_parser.py` decodes the log with our own
BOM/UTF-16/cp1252 detector (`read_spice_text`), rewrites it to a UTF-8
temporary file, and passes that file to `opLogReader`, so its detection cannot
choose the wrong encoding.

### Summary

`detect_encoding()`, used by the device operating-point reader
`semi_dev_op_reader.opLogReader`, tries encodings in the order
`utf-8, utf-16, utf_16_le, windows-1252, cp1252, …` and returns the first that
decodes without raising. A cp1252 log containing a high byte (a degree sign,
micro sign or plus-minus sign, which are common in a `.options logopinfo`
operating-point dump) fails UTF-8. It is then tried as **utf-16**, which decodes
ASCII with a high byte to garbage **without raising**, so utf-16 is returned and
the block is garbled. utf-16 comes **before** cp1252 in the try order. The only
utf-16 guard (`lines[1] == '\x00'`) runs only when `encoding == 'utf-8'`, a
branch already skipped once UTF-8 raised, so nothing catches the misdetection.
`opLogReader` then finds no device parameters and returns an empty result, as
if the log held none.

### Affected code

`spicelib/utils/detect_encoding.py`:

```python
for encoding in ('utf-8', 'utf-16', 'utf_16_le', 'windows-1252', 'cp1252', ...):
    try:
        lines = open(file_path, encoding=encoding).read()
    except (UnicodeDecodeError, UnicodeError):
        continue
    ...
    if encoding == 'utf-8' and lines[1] == '\x00':   # utf-16 heuristic, utf-8 branch only
        continue
    return encoding      # utf-16 false-decodes a cp1252 file here
```

### Reproduction

```python
# a cp1252-encoded log with a degree-sign (0xB0) byte, even length
from spicelib.utils.detect_encoding import detect_encoding
detect_encoding("oppoint_cp1252.log")   # -> 'utf-16' (wrong); content garbled
```

### Impact

- Device operating-point dumps (gm, gds, vth, ...) in a cp1252 log decode to
  garbage, so the operating-point read silently returns no device parameters.
  No error is raised; the data is wrong or empty.

### Proposed fix

Try single-byte encodings before utf-16; or validate a utf-16 decode (reject it
if it yields a high proportion of non-text codepoints); or apply the
`\x00`-second-byte heuristic regardless of which branch reached the utf-16
candidate.

### Suggested upstream test

```python
def test_cp1252_with_high_byte_not_detected_as_utf16(tmp_path):
    p = tmp_path / "l.log"
    p.write_bytes("temp: 25°C\ngm: 1.0\n".encode("cp1252"))
    assert detect_encoding(str(p)) in ("windows-1252", "cp1252")
```

### Cross-reference

Downstream: `tests/test_device_op_points.py::TestReadDeviceOpPoints::test_recovers_op_points_from_nonutf8_log`
pins the operating-point read of a cp1252 log with a degree-sign byte.
`tests/test_log_parser.py::...::test_cp1252_degree_byte_in_step_value_recovered`
and the cp1252 cases in `tests/test_encoding.py` pin our decoder. Once upstream
fixes the order, delete the step that rewrites the log to a UTF-8 temporary
file before `opLogReader` reads it.

---

## Bug 9 — `remove_instruction` deletes the wrong directive by substring match

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`
`AscEditor.remove_instruction`, ~line 698).
**Our workaround:** `_remove_exact_directive` in `src/ltspice_mcp/lib/schematic_ops.py`.
On an `AscEditor`, it matches the typed `directives` list by **exact full-text
equality** and deletes exactly one entry, instead of calling `remove_instruction`.

### Summary

`AscEditor.remove_instruction` finds the directive to delete with
`if instruction in self.directives[i].text`, which is a **substring** test, and
removes the first match. `remove_instruction(".tran 1")` can therefore delete
`.tran 10m`, and removing one of several similar directives deletes whichever
comes first instead of the intended one. No error is raised.

### Affected code

`spicelib/editor/asc_editor.py` (~line 698):

```python
if instruction in self.directives[i].text:   # substring, first match
    del self.directives[i]
    ...
    return True
```

### Reproduction

```python
ed = AscEditor("two_directives.asc")   # has ".tran 10m" and ".tran 1"
ed.remove_instruction(".tran 1")        # removes ".tran 10m" (first substring hit)
```

### Impact

- Removing a directive can silently delete a different directive that
  contains the target as a substring, or the wrong one of several similar
  directives.

### Proposed fix

Match on exact (whitespace-normalized) equality instead of a substring:

```python
if self.directives[i].text.strip() == instruction.strip():
```

### Suggested upstream test

```python
def test_remove_instruction_exact_not_substring(tmp_path):
    ed = AscEditor(<asc with ".tran 10m" and ".tran 1">)
    ed.remove_instruction(".tran 1")
    assert ".tran 10m" in [d.text for d in ed.directives]   # untouched
```

### Cross-reference

Downstream: `tests/test_circuit_asc.py::...::test_remove_directive_round_trip`
and `::test_remove_directive_no_match_raises`. Once upstream matches exactly,
delete our exact-equality override.

---

## Bug 10 — `SimRunner.__del__` blocks for the whole simulation, so a dropped reference freezes the destroying thread

**Status:** draft for an upstream spicelib pull request. Filed 2026-08-07.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/sim_runner.py:333`
`__del__`, `:711` `wait_completion`). The code is unchanged across the pinned
range (`>=1.4.9,<1.6`).
**Our workaround:** `src/ltspice_mcp/lib/runner_base.py` —
`_NonBlockingSimRunner` overrides `__del__` to do nothing, and
`RunnerBase._build_sim_runner` constructs that subclass instead of `SimRunner`.
Nothing in this project needs the destructor. Completion is reported through
the run callback, and liveness is read from the task threads. The destructor's
only other action is `kill_all_spice()`, which kills simulator processes by
name across the machine, and this project never calls it. `submit_netlist`
still keeps every runner in `self._inflight_runners` and releases it in
`_retire_finished_runners`, because the code that kills runs and checks
liveness needs the handle. Delete the subclass once upstream's destructor no
longer waits.

### Summary

`SimRunner.__del__` calls `self.wait_completion(abort_all_on_timeout=True)`, and
`wait_completion` loops `while len(self.active_tasks) > 0: sleep(1)`. A
destructor therefore blocks for as long as the simulation runs, up to the
instance timeout.

**In one case the wait has no bound at all.** `wait_completion(timeout=None)`
recomputes its deadline every second from `_maximum_stop_time()`, which reads
`task.start_time + timeout` and **skips any task whose `start_time` is None**;
`update_completed()` likewise retires a task only `if not (is_alive() or
start_time is None)`. `run()` calls `active_tasks.append(t)` before
`t.start()`, so a task can be in `active_tasks` and never have started. Nothing
retires such a task, and it sets no deadline. The loop condition stays true,
the deadline stays `None`, and the destructor loops on `sleep(1)` forever. The
same predicate makes `RunTask.wait_results()` unbounded
(`while self.is_alive() or self.start_time is None or self.retcode == -1`).

Because the last reference is usually dropped by ordinary garbage collection,
the infinite wait runs on whichever thread's allocation triggered the
collection. On 2026-09-07 one of our CI jobs produced no output for 12 minutes
48 seconds because of it and was killed by the job timeout. The job reported
only "The operation was canceled", with no traceback, no failing test and no
simulator process left behind, because the simulation had finished and only
the waiting thread was stuck.

Under CPython's reference counting, the destructor can run at the end of the
statement that submits. `run()` starts the simulation on a run-task thread and
returns the `SimRunner`. A caller that submits **for the side effect** and does
not bind the result drops the last reference at once, and the submitting thread
blocks in the destructor until the simulation finishes. The API does not state
that the caller must keep the return value, which is documented as "For
internal use only", so the most natural way to call `run()` hangs.

The failure produces no diagnostic. There is no exception or log line, and the
simulation itself succeeds; only the *caller* is blocked. A stack dump shows
`__del__ -> wait_completion`, which looks like a shutdown path rather than a
submission.

### Affected code

`spicelib/sim/sim_runner.py`:

```python
    def __del__(self):
        """Class Destructor : Closes Everything"""
        self.wait_completion(abort_all_on_timeout=True)  # Kill all pending simulations

    def wait_completion(self, timeout=None, abort_all_on_timeout=False) -> bool:
        self.update_completed()
        ...
        while len(self.active_tasks) > 0:
            sleep(1)
            self.update_completed()
```

### Reproduction

```python
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ltspice_simulator import LTspice
import time

def submit_and_discard(netlist):
    # No binding: exactly how a caller submits for the side effect.
    SimRunner(simulator=LTspice, output_folder="out", parallel_sims=4).run(
        netlist, run_filename="probe.net", callback=lambda raw, log: None)

t0 = time.time()
submit_and_discard("long_transient.net")   # a deck that runs ~60 s
print(f"submit returned after {time.time() - t0:.1f}s")   # ~60 s, not ~0.1 s
```

Expected: submission returns as soon as the simulation thread is started.
Observed: it returns only when the simulation completes.

### Impact

The failure is severe and raises no error for any caller that submits from a
worker thread and expects to keep working, which every async wrapper does. In
this project, a coroutine submitted each simulation through `asyncio.to_thread`
and, once that call returned, marked the simulation running and started
watching for a cancel request. Because the worker thread was blocked in the
destructor, the coroutine never resumed and the cancel watch never started, so
**a cancellation could never reach the simulator**. The run continued to
completion, our job records listed a simulation that had started as never
submitted, and the completed result was left on disk with no record that
referred to it. The user saw that cancel did nothing, and the cause was the
destructor.

Because the destructor passes `abort_all_on_timeout=True`, a garbage collection
at an unlucky moment can also *abort* simulations that belong to a runner the
caller only stopped referencing.

### Proposed fix

A destructor must not block. Two options, in preference order:

0. **Give the wait a floor that always advances.** Whatever else changes,
   `_maximum_stop_time()` and `update_completed()` should treat a task that is
   not alive and has no `start_time` as finished (it can never start), or
   `wait_completion` should fall back to the instance `timeout` when no task
   offers a deadline. As written the loop has a reachable state with no exit.

1. **Drop the wait from `__del__`.** Leave `wait_completion()` as the explicit
   call it already is, and let `close()` or context-manager use handle teardown.
   Callers that want to wait can call it, and letting an object go out of scope
   no longer blocks.
2. **Bound it.** If teardown must wait, give the destructor a short, explicit
   timeout (`self.wait_completion(timeout=_DEL_TIMEOUT)`) and log when it
   expires, so the pause is bounded and visible rather than open-ended.

Either way, document that `run()`'s return value need not be retained — or, if
retention *is* required for correctness, say so in the `run()` docstring, since
the current text ("For internal use only") suggests the opposite.

### Suggested upstream test

```python
def test_submitting_without_binding_the_runner_returns_promptly(tmp_path):
    """A caller that submits for the side effect must not be frozen by GC."""
    started, release = threading.Event(), threading.Event()

    class SlowSim:                      # stands in for a long simulation
        @classmethod
        def run(cls, *a, **k):
            started.set()
            release.wait(10)
            return 0

    t0 = time.monotonic()
    SimRunner(simulator=SlowSim, output_folder=str(tmp_path)).run(
        str(deck), run_filename="probe.net", callback=lambda raw, log: None)
    elapsed = time.monotonic() - t0     # destructor runs at end of statement
    release.set()
    assert started.is_set()
    assert elapsed < 1.0, f"submission blocked {elapsed:.1f}s in the destructor"
```

### Cross-reference

Downstream: `tests/test_experiment_runner.py::TestSubmitPrimitive::test_submitted_runner_outlives_a_caller_that_discards_it`
submits through the real `RunnerBase.submit_netlist` with the return value
discarded, and asserts the runner is not destroyed while its task thread is
alive (and is released once it is not).

---

## Note A — LTspice `-netlist` marks subcircuit instances with U+00A7 (not a spicelib bug)

**Status:** **Not a bug, and not in spicelib.** This is documented LTspice
exporter behavior, recorded here so that nobody investigates the pattern again
the next time it appears in an exported netlist. No upstream pull request
applies.
**Affected version:** observed on LTspice 26.0.2 for Windows (via WSL interop).
spicelib is not involved; see "Why this is not a spicelib bug" below.
**Our handling:** `src/ltspice_mcp/lib/netlist_graph.py` already strips the
marker from each reference (`_strip_marker`), `canon_ref` drops it when two
references are compared, and the netlist lexer discards the trailing `;`
comment. There is no workaround to remove later.

### Summary

When LTspice exports a schematic with `-netlist`, every instance of a symbol
whose `SYMATTR Prefix` is `X` (that is, every subcircuit instance) is written
as:

```
X§<InstName> <nodes...> <subckt> [params] ;§pnba <pin1>)<pin2>)...
```

Two artifacts appear, both containing a literal U+00A7 SECTION SIGN (UTF-8
`0xC2 0xA7`):

1. the reference is `X` + `§` + the InstName (`X§U1`, not `XU1`), and
2. a trailing comment `;§pnba a)y)g` listing the symbol's pin names in
   SpiceOrder — LTspice's pin-name back-annotation aid, paired with `.backanno`.

Although it resembles a leaked placeholder or an encoding fault, it is
LTspice's normal output for subcircuit instances.

### Reproduction

This minimal case does **not use spicelib**; it runs LTspice.exe directly.

`attn.asy`, the smallest symbol that netlists as an X device:

```
Version 4
SymbolType CELL
RECTANGLE Normal 0 0 96 64
PIN 0 32 NONE 0
PINATTR PinName a
PINATTR SpiceOrder 1
PIN 96 32 NONE 0
PINATTR PinName y
PINATTR SpiceOrder 2
PIN 48 64 NONE 0
PINATTR PinName g
PINATTR SpiceOrder 3
SYMATTR Prefix X
SYMATTR Value attn
```

`min.asc`, one instance, no wires, no labels:

```
Version 4
SHEET 1 880 680
SYMBOL attn 208 176 R0
SYMATTR InstName U1
```

```console
$ LTspice.exe -netlist min.asc
$ cat min.net
* Generated by LTspice 26.0.2 for Windows.
X§U1 NC_01 NC_02 NC_03 attn ;§pnba a)y)g
.backanno
.end
```

Byte-level confirmation that both markers are U+00A7:

```console
$ grep -a U1 min.net | xxd | head -2
00000000: 58c2 a755 3120 4e43 5f30 3120 4e43 5f30  X..U1 NC_01 NC_0
00000010: 3220 4e43 5f30 3320 6174 746e 203b c2a7  2 NC_03 attn ;..
```

### Scope and trigger

Control cases determined the trigger:

| Symbol | `SYMATTR Prefix` | Exported line | Marker? |
|-|-|-|-|
| project-local generated `attn.asy` | `X` | `X§U1 NC_01 NC_02 NC_03 attn ;§pnba a)y)g` | yes |
| **stock** `bk_inv.asy` (ships with LTspice) | `X` | `X§U1 … bk_inv Wp=4u Wn=2u ;§pnba A)Y)VDD)VSS` | yes |
| stock `res.asy` | `R` | `R1 NC_01 NC_02 1k` | no |

The trigger is **`Prefix X` (subcircuit) instances, stock or custom alike**.
It is *not* specific to generated or project-local symbols, and a report that
describes it that way is wrong. Non-X devices are never marked. The annotation
comment follows any device parameters.

### Impact

None for this project. The `§` is stripped from the reference and the `;`
comment is discarded as a comment, so the card parses as intended. Both
reproductions above were checked:

```
min.net    -> ref='XU1' type='X' nodes=('nc_01','nc_02','nc_03') model='attn'  params=()
stock.net  -> ref='XU1' type='X' nodes=(...x4)                   model='bk_inv' params=(('Wp','4u'),('Wn','2u'))
```

The comment is read neither as a fourth or fifth node nor as a parameter, so
the node count and the parameters are right. This entry guards against a
*future* parser that splits the card before stripping comments, or that
compares references with the `§` still in them instead of through `canon_ref`.

### Why this is not a spicelib bug

`spicelib.simulators.ltspice_simulator.LTspice.create_netlist` does not build
a netlist itself; it runs LTspice, which writes the file:

```python
cmd_netlist = cls.spice_exe + ['-netlist'] + [circuit_file.as_posix()] + cmd_line_switches
...
netlist = circuit_file.with_suffix('.net')
```

The reproduction above bypasses spicelib entirely and produces byte-identical
output, so spicelib neither introduces nor can remove the marker.

### No workaround

There is deliberately no workaround. The marker is stripped where references
are parsed and compared, and the annotation is a SPICE comment. A separate
pass that removed them would be dead code and would hide a real change in
LTspice's output format.

### Cross-reference

A test pins this behavior so that it is deliberate rather than accidental.
`tests/test_netlist_graph.py::test_undefined_pdk_subckt_stays_a_black_box_leaf`
feeds the exact exported signature (both the `X§` reference and the `;§pnba`
comment, with device parameters) through `parse_netlist_graph` and asserts the
reference, node count, model and parameters.


## Bug 11 — `RawRead.get_axis()` returns an unsized 0-d array for a file with no plots, so `len()` raises `TypeError`

### Summary

For a `.raw` file that parsed but contains no plots, `RawRead.get_axis()`,
`get_wave()` and `get_time_axis()` each return `numpy.ndarray([])`. That is the
low-level constructor, whose first argument is the shape, so it builds an
uninitialized 0-dimensional array rather than an empty 1-d array.
`len(axis)` then raises `TypeError: len() of unsized object`
instead of returning `0`, and any caller that sizes the axis before
reading it has to handle, as a special case, a Python exception that gives
no information about the file.

### Affected code

spicelib 1.5.1, `spicelib/raw/raw_read.py`, the `if len(self._plots) == 0`
branch of `get_wave()`, `get_time_axis()` and `get_axis()` (~lines 531, 542 and
556). `get_trace()` raises `SpiceReadException` in the same case instead.

### Reproduction

```python
from spicelib import RawRead
raw = RawRead("empty_plot.raw")      # a header-only raw, e.g. a run that wrote no points
axis = raw.get_axis()
print(axis.ndim)                      # 0
len(axis)                             # TypeError: len() of unsized object
```

### Impact

A summary or metric that sizes the axis (`len(axis)`, `axis.shape[0]`) fails
with a `TypeError` that looks like a bug in the caller. Our summary builder
used to catch `Exception` around the read, which hid the file's real
condition; narrowing the catch revealed this bug.

### Proposed fix

Return `numpy.empty(0)` (a 1-d array of length 0) from every empty-data
branch, so `len()` is `0` and `.shape == (0,)`.

### Suggested upstream test

```python
def test_empty_axis_is_one_dimensional(tmp_path):
    raw = RawRead(write_header_only_raw(tmp_path / "empty.raw"))
    axis = raw.get_axis()
    assert axis.ndim == 1 and len(axis) == 0
```

### Cross-reference

Workaround: `build_simulation_summary` and `query_point_value` in
`src/ltspice_mcp/lib/raw_parser.py` catch `TypeError` alongside `RuntimeError`
at the axis read, for a caller that passes them a spicelib `RawRead` (the
comments there describe this bug). The server's own decoded plots raise
`NoAxisError` instead (`DecodedPlot.get_axis` in `lib/decoded_raw.py`). No
current test pins the catch. Remove `TypeError` from those catches once
upstream returns a 1-d empty array.

## Bug 12 — `AscEditor.save_netlist` silently drops hierarchical ports

### Summary and affected version

In spicelib 1.5.1, `editor/asc_editor.py::reset_netlist` parses each `IOPIN`
into a `Port` referencing its preceding label. `save_netlist` writes labels
but never writes `self.ports`. Editing an unrelated directive therefore removes
the hierarchical interface while leaving its net labels behind.

### Reproduction

Load this sheet with `AscEditor`, add an ordinary directive, and save it to
another ASC file or a `StringIO` sink:

```text
Version 4
SHEET 1 880 680
FLAG 0 0 IN
IOPIN 0 0 In
FLAG 160 0 OUT
IOPIN 160 0 Out
```

The output contains both `FLAG` lines and neither `IOPIN`. We reproduced this
through our `edit_schematic` tool, which saves a sheet through `save_netlist`,
by adding `.param marker=1` (with the sheet's current SHA-256, which the tool
requires before it changes an existing file). The tool reported a complete,
saved edit, and the port count fell from two to zero.

### Impact, proposed upstream fix and test

An ordinary edit to a reusable child sheet destroys its port declarations.
Write each port's `IOPIN` immediately after its associated `FLAG`, preserving
coordinates, direction and order. Association must use the label object,
not its text or coordinates, since distinct labels may have identical names.
Refuse an orphaned or ambiguous association instead of attaching it elsewhere.

An upstream test should round-trip multiple ports, including repeated label
names, then move/rename a label object and verify that its port follows it.
Also cover a removed label still referenced by a port.

### Workaround and regression

`tools/schematic_edit.py::_render_editor_text` renders the sheet with
`save_netlist` into a `StringIO`, then writes each port's `IOPIN` line after
its `FLAG` line, matching ports to labels by label object. It refuses the edit
before anything is written when a label belongs to more than one port or the
rendered `FLAG` lines no longer match the editor's labels.
`tests/test_edit_schematic.py::TestHierarchicalPortPreservation` covers an
ordinary edit through `edit_schematic`, a refused label removal, reordered or
changed labels, and ambiguous associations. Remove this workaround when the
pinned dependency preserves the same cases itself.

## Bug 13 — a `StringIO` schematic save can write modified child files

### Summary and affected version

In spicelib 1.5.1, `AscEditor.save_netlist` accepts a `StringIO` for rendering a
sheet in memory. While serializing an X component, it also calls
`save_netlist(child.asc_file_path)` on an updated `_SUBCKT` editor. The sink
controls only the parent; a child is written directly to its original path.

### Reproduction and impact

Load a parent ASC containing a BLOCK symbol backed by a child ASC. Obtain the
child with `parent.get_subcircuit("X1")`, call
`child.set_parameter("changed", 1)`, and render the parent into `StringIO`.
The child's on-disk file changes. The same branch is reached while our
`edit_schematic(dry_run=true)` renders the edited sheet before the dry run
returns. Our tool saves only the parent sheet, in one atomic write, so it cannot
safely include those child changes.

### Proposed upstream fix and test

Separate rendering one sheet from saving its dependency tree. A `StringIO`
render should neither write child files nor clear their pending-update state.
Any recursive save should be explicit and report which files it writes.
An upstream test should modify a loaded child, render its parent to `StringIO`,
and assert unchanged child bytes and retained pending changes. Include a
portless parent and a deeper loaded descendant.

### Workaround and regression

`tools/schematic_edit.py::_refuse_pending_child_edits` runs before every call
to `save_netlist`. It walks every loaded child sheet at any depth, including
sheets with no ports, and refuses the edit if any has pending changes
(`updated` set), without modifying editor flags or child files.
`TestHierarchicalPortPreservation.test_pending_child_changes_never_write_through_parent`
covers an ordinary edit and a dry run through `edit_schematic`, with real
loaded child editors. Files are written only when our tool saves the edited
sheet itself, which it does with an atomic rename after checking the caller's
SHA-256 of the file.

## Bug 14 — `SimRunner.run` cannot run without a timeout, though `None` is documented as "no timeout"

**Status:** draft for an upstream spicelib pull request. Filed 2026-09-29.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/sim_runner.py`, `SimRunner.run`,
the resource-wait loop after `_prepare_sim`). 1.4.9 has the same loop.
**Our workaround:** `src/ltspice_mcp/lib/runner_base.py` —
`RunnerBase._build_sim_runner` never passes `None` to spicelib. An unbounded
request becomes `SUBPROCESS_TIMEOUT_CEILING_S` (4,000,000 s), and any larger
value is clamped to it. The ceiling also keeps the bound inside what Windows
can wait for (below). Remove the `None` half of the substitution once upstream
runs with `None`; the clamp stays for as long as Windows waits in 32-bit
milliseconds.

### Summary

The `SimRunner` constructor documents `timeout` as "Timeout parameter as
specified on the OS subprocess.run() function. ... For no timeout, set to None."
`run()` then uses the same value as the deadline of its wait for a free slot,
and evaluates `timeout + 1` before launching anything:

```python
if timeout is None:
    timeout = self.timeout
t0 = clock()
while clock() - t0 < timeout + 1:
```

With `self.timeout = None` that is `None + 1`, so every `run()` raises
`TypeError` and no simulation starts, although `None` is the documented value
for no timeout.

Because the slot wait and the simulator process share one value, a long bound
on the process also gives the slot wait that bound. This is harmless for us,
because we create a fresh runner for each submission and its slot wait never
waits, but the process bound cannot be set separately.

### Affected code

`spicelib/sim/sim_runner.py`, `SimRunner.run`: the `while clock() - t0 < timeout + 1`
loop; `run_now` has the same `t.join(timeout + 1)`. `RunTask.run` passes the same
`timeout` to `Simulator.run`, which passes it to `subprocess.run`.

### Reproduction

```python
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

runner = SimRunner(simulator=NGspiceSimulator, output_folder="out", timeout=None)
runner.run("deck.cir")
# TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

Reproduced on spicelib 1.5.1, 2026-09-29, with any deck.

### Impact

A caller cannot leave the simulator bounded only by its own logic. Before we
found this bug we passed a fixed 600 s instead, which became a hidden limit on
every run. A run given a longer limit through our `run_timeout_s` setting was
killed at 600 s by `subprocess.run`, and the kill was reported as spicelib's
generic exit code -2 instead of as a timeout.

The obvious substitute for `None`, a very large number, is not safe on Windows.
There `subprocess` converts the timeout to integer milliseconds and passes it
to `WaitForSingleObject`, whose timeout is a 32-bit DWORD, so nothing past
about 49.7 days can be expressed. This comes from reading CPython's source and
was not tested on Windows. Any substitute has to stay below 4,294,967 s.

### Proposed fix

Treat `None` as no deadline in the slot wait, and keep passing `None` through to
`RunTask` so `subprocess.run` waits without a timeout:

```python
deadline = None if timeout is None else clock() + timeout + 1
while deadline is None or clock() < deadline:
```

`run_now` should `t.join(None if timeout is None else timeout + 1)`.

### Suggested upstream test

Construct `SimRunner(timeout=None)` with a stub simulator whose `run` records the
`timeout` it receives and returns 0; call `run()` and assert the stub ran with
`timeout=None`. Repeat through `run_now`.

### Cross-reference

Workaround: `runner_base.py` `SUBPROCESS_TIMEOUT_CEILING_S` and
`RunnerBase._build_sim_runner`. Pinned by
`tests/test_runners.py::TestSimulatorProcessBound::test_the_simulator_runs_under_the_callers_bound`,
which runs spicelib's real `SimRunner` and `RunTask` with a stub simulator and
checks the bound it receives for a given value, for `None`, and for a value past
the ceiling.

## Bug 15 — a stopped run's raw is rejected or silently cut short (limitation)

**Status:** known limitation; a feature request rather than a defect in reading
finished files. Recorded 2026-09-29.
**Affected version:** spicelib 1.5.1 (`spicelib/raw/plot_data.py`, `PlotData.__init__`
and `read_trace_data`; `spicelib/sim/run_task.py`, `RunTask.run`).
**Our workaround:** `src/ltspice_mcp/lib/raw_parser.py` —
`read_partial_raw_progress` reads the header itself and counts complete records
from the file length. When our server stops a simulation (for its time limit, a
job deadline or a cancel), it calls this before deleting the run's files and
reports the result to the caller as a `partial_progress` note in the job's
results. When the caller asked to keep the partial raw (our `keep_partial_raw`
option), `cut_partial_raw` also truncates a binary file stored in record order
after its last kept point and writes that point count into the header, so an
ordinary spicelib read loads it. An ngspice file keeps every complete point.
An LTspice file keeps at most the points its header count declares, because
LTspice's own reader stops there and LTspice XVII's records past that count
need not continue in time.

### Summary

Both simulators write the raw while they solve and fill in `No. Points` only
when a plot ends. A killed run leaves a different header in each:

- **ngspice** writes `No. Points: 0` padded with spaces, to patch in place when
  the plot ends. spicelib raises `SpiceReadException` ("No points or variables
  found") on the count of 0, so the file cannot be opened at all.
- **LTspice** rewrites the count periodically while it runs, so a killed run's
  header contains a count lower than the number of points written. spicelib
  reads exactly that many points and reports success, silently dropping every
  complete record after it.

Separately, `RunTask.run` passes the completion callback `raw_file=None` for
any nonzero exit. A killed run always exits nonzero, so the callback is never
told that a partial raw exists, even when the simulator wrote gigabytes of it.
A caller has to rebuild the path (`netlist.with_suffix(raw_extension)`).

### Reproduction

Observed 2026-09-29 on Linux with ngspice 42 and LTspice 26.1.1 under Wine,
each killed with SIGKILL part way through `.tran 10n 100m 0 10n` (ngspice) or
`.tran 0 100 0 10n` (LTspice) of an RC driven by a 1 kHz sine:

- ngspice, 2 s in: 32,850,137 bytes, header `No. Points: 0       `, 1,026,560
  complete 32-byte records (time plus three doubles). `RawRead(path,
  dialect="ngspice")` raises `SpiceReadException`.
- LTspice, 8 s in: 51,020,708 bytes, header `No. Points:      1821697`,
  1,822,135 complete 28-byte records (a double time plus five floats) and a torn
  one. Every record past the header count contains a valid, increasing time.
  `RawRead(path)` returns 1,821,697 points with no warning.

Neither simulator records progress in any other file that remains after a
kill. LTspice's log contains only its preamble until the run ends. ngspice
prints `Reference value : <time>` progress to stdout only when it has no `-o`
log file, and spicelib always passes `-o`, so neither the `.log` nor the
`.exe.log` contains it.

### Impact

A run stopped for a timeout, a deadline or a cancel can have done almost all of
its work, and no output reports how much. The ngspice file is unreadable. The
LTspice file reads as a complete, shorter run, which is the more dangerous
case, because a caller that reads it through spicelib gets a waveform that ends
early with no sign that it does.

### Proposed fix

Add an opt-in partial read on `RawRead` (for example `allow_partial=True`)
that, for a binary plot, takes the point count as `min(declared, complete
records on disk)` when the declared count is nonzero and the file is short, as
the complete records on disk when it is 0, and as the complete records on disk
when the file contains more than the declared count and no following plot
starts at the declared end. The read should report that it was partial. Separately, `RunTask` could
pass the expected raw path to the callback on a nonzero exit when the file
exists, leaving the decision to use it to the caller.

### Suggested upstream test

Truncate a finished binary raw from each dialect to a whole number of records
plus a torn one; set the ngspice copy's count to 0 and the LTspice copy's count
below the records present; assert that the partial read returns exactly the
complete records and marks itself partial, and that the default read of the
ngspice copy still raises.

### Cross-reference

Workaround: `raw_parser.py` `read_partial_raw_progress` and `cut_partial_raw`,
used by `ExperimentRunner._read_progress` in `experiment_runner.py`. Pinned by
`tests/test_raw_parser.py::TestPartialRawProgress` (the lagging LTspice count in
`test_a_declared_count_behind_the_records_does_not_cap_them`),
`tests/test_raw_parser.py::TestCutPartialRaw` (each cut file read back with
spicelib's `RawRead`) and, against a real killed ngspice run,
`tests/test_ngspice_e2e.py::test_run_timeout_reports_the_killed_runs_diagnostics_and_progress`.

## Bug 16 — `AscEditor.get_components(prefixes)` matches a character set, case-sensitively (limitation)

### Summary and affected version

In spicelib 1.4.9 and 1.5.1, `editor/asc_editor.py::AscEditor.get_components`
filters with `[k for k in self.components.keys() if k[0] in prefixes]`. The
argument is documented as "Type of prefixes to search for. Examples: 'C' for
capacitors", but it is read as a set of characters, compared with the
reference's first character as written. So `get_components("r")` returns no
`R1`, and `get_components("LX")` returns every `L…` and every `X…` rather than
the `LX…` references. SPICE element letters are case-insensitive, and LTspice
sheets contain upper-case references, so a lower-case filter silently returns
nothing.

### Reproduction

Load any sheet containing `SYMATTR InstName R1` with `AscEditor` and call
`editor.get_components("r")`. The result is `[]`. `get_components("R")`
returns `["R1"]`. `get_components("RC")` returns both resistors and
capacitors.

### Impact, proposed upstream fix and test

A caller that passes the letter it was given gets an empty component list on
a populated sheet, with no error. Compare case-insensitively
(`k[:1].upper() in prefixes.upper()`), and either document the argument as a
set of element letters or accept a sequence of prefixes and test
`k.upper().startswith(p.upper())` for each. An upstream test should cover a
lower-case letter, a multi-letter prefix, and the `'*'` default.

### Workaround and regression

Our component listing (`tools/inspect_tools.py::_do_components`) does not call
`get_components`. It reads the sheet with our own `.asc` reader, lists every
reference, and keeps those for which `montecarlo.matches_prefix` matches the
prefix. That function compares the prefix case-insensitively with the start of
the reference. Our component listing for a netlist and our listing of a
design's hierarchy use the same rule (`_check_prefix` validates the prefix and
returns it upper-cased for all three to compare against). Pinned by
`tests/test_inspect_tools.py::test_components_prefix_filter_asc_ignores_case`
and `test_components_prefix_is_a_case_insensitive_reference_prefix`. Nothing
here depends on spicelib's filter, so there is no workaround to remove when
upstream changes.

## Bug 17 — the symbol cache is keyed by file name, so one folder's symbol stands in for another's

**Status:** draft for an upstream spicelib pull request. This is a known
limitation, and we ship no workaround.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.symbol_cache` and `_asy_file_find`).
**Our workaround:** none for spicelib's own loading. Our pin geometry does not
read this cache: the session's `lib/symbol_library.py` `SymbolLibrary` looks
for a symbol beside the sheet itself and caches what it reads by file.

### Summary

`AscEditor.symbol_cache` is a class attribute shared by every editor in the
process, and `_asy_file_find` keys it by the bare `.asy` file name. The search
it caches starts in the schematic's own folder, so the first sheet to load a
name decides which file that name means for every later sheet. A second sheet
in another folder, with its own symbol of the same name beside it, is resolved
to the first sheet's file. For a hierarchical block, `AsyReader.get_schematic_file`
derives the sub-sheet from that `.asy` path, so the second sheet's block opens
the first folder's sub-sheet.

### Affected code

`spicelib/editor/asc_editor.py`:

```python
symbol_cache = {}  # This is a class variable, so it can be shared between all instances.

def _asy_file_find(self, filename) -> str | None:
    if filename in self.symbol_cache:          # keyed by bare file name
        return self.symbol_cache[filename]
    ...
    file_found = search_file_in_containers(filename,
                                           os.path.split(self.asc_file_path)[0],  # this sheet's folder
                                           ...)
    if file_found is not None:
        self.symbol_cache[filename] = file_found
```

### Reproduction

Two folders, each with a block symbol `amp.asy`, its sub-sheet `amp.asc` and a
`top.asc` placing it as `X1`. The sub-sheets differ only in `R1`'s value: 1k in
`a`, 2k in `b`. A `res.asy` is on the custom library path.

```python
a = AscEditor(root / "a" / "top.asc")
b = AscEditor(root / "b" / "top.asc")
a.get_component_value("X1:R1")   # '1k'
b.get_component_value("X1:R1")   # '1k', but b/amp.asc says 2k
AscEditor.symbol_cache           # {'amp.asy': '.../a/amp.asy', ...}

b.set_component_value("X1:R1", "5k")
b.save_netlist(root / "b" / "top.asc")
# a/amp.asc now reads "SYMATTR Value 5k"; b/amp.asc still reads 2k
```

### Impact

- A process that opens schematics from more than one folder reads a later
  sheet's same-named block through an earlier sheet's files. It gets the wrong
  sub-sheet, the wrong component values, and the wrong `SymbolType`/`Prefix`
  when spicelib decides whether an instance is a subcircuit.
- Saving the later sheet writes its sub-sheet changes into the earlier
  folder's file, because `save_netlist` saves each updated child to its own
  `asc_file_path` (see Bug 13).
- Which file is used depends on which sheet a long-lived process loaded
  first, so the same request can return different results in two sessions.

### Proposed fix

Key the cache by the folder searched as well as the name, or cache only the
library results and search the sheet's own folder on every load:

```python
key = (os.path.split(self.asc_file_path)[0], filename)
if key in self.symbol_cache:
    return self.symbol_cache[key]
```

### Suggested upstream test

```python
def test_same_named_block_beside_two_sheets_resolves_per_folder(tmp_path):
    # a/ and b/ each hold amp.asy (SymbolType BLOCK), amp.asc and top.asc,
    # with R1 = 1k in a/amp.asc and 2k in b/amp.asc.
    a = AscEditor(tmp_path / "a" / "top.asc")
    b = AscEditor(tmp_path / "b" / "top.asc")
    assert a.get_component_value("X1:R1") == "1k"
    assert b.get_component_value("X1:R1") == "2k"
```

### Cross-reference

Our pin geometry is pinned per folder by
`tests/test_circuit_asc.py::TestSheetLocalSymbols::test_same_named_symbols_beside_two_sheets_keep_their_own_pins`.
`edit_schematic` cannot reach the write in the reproduction, because it
refuses pending child edits before saving (the workaround for Bug 13), but it
can reach the wrong read. The test suite fills this cache for the fixture
library once per session (`tests/conftest.py::_asc_symbol_cache`), and tests
that load sheet-local symbols swap in a copy of it so that no later test sees
the folders they loaded.

---

## Bug 18 — `Simulator.create_from` rebinds the class it is called on instead of creating one

**Status:** draft for an upstream spicelib pull request. Recorded 2026-10-03.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`Simulator.create_from`).
**Our workaround:** `src/ltspice_mcp/lib/simulator.py` —
`bind_named_executable` creates a subclass of the family's class for each named
executable and calls `create_from` on the subclass, so the program is written
onto that subclass alone. Keep the subclass even once upstream is fixed,
because our runner cache, our limit on simultaneous simulator processes and
the process names we kill by are all looked up by class, with one class per
build. Only the comment that
explains why `create_from` must never touch the family's class can then be
removed.

### Summary

`create_from` is documented as "Creates a simulator class from a path to the
simulator executable" and returns "a class instance representing the Spice
simulator". It does not create a class. It assigns `cls.spice_exe` and
`cls.process_name` on the class it was called on and returns that same class.
Calling it on `LTspice` (or any shipped simulator class) changes the program
for every user of that class. Two builds of one simulator therefore cannot be
bound in one process through the documented method, because the second call
silently replaces the first.

### Affected code

```python
@classmethod
def create_from(cls, path_to_exe, process_name=None):
    ...
    if plib_path_to_exe is not None and (plib_path_to_exe.exists() or shutil.which(plib_path_to_exe)):
        if process_name is None:
            cls.process_name = cls.guess_process_name(exe_parts[0])
        else:
            cls.process_name = process_name
        cls.spice_exe = exe_parts
        return cls
```

### Reproduction

```python
from spicelib.simulators.ltspice_simulator import LTspice

xvii = LTspice.create_from("C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe")
lt24 = LTspice.create_from("C:/Program Files/ADI/LTspice/LTspice.exe")

assert xvii is lt24 is LTspice
xvii.spice_exe   # ['C:/Program Files/ADI/LTspice/LTspice.exe']: the XVII binding is gone
```

Any two paths that exist reproduce it.

### Impact

- A program that keeps two builds of one simulator (LTspice XVII for decks that
  depend on its cp1252 reading of `µ`, LTspice 24 for current ones) gets one
  of them, the last one bound, with no error.
- A `SimRunner` already running on the class launches its next run on the new
  program, so a rebind while runs are queued moves them to another build.
- `process_name`, which spicelib's own `kill_all_spice` reads, changes with it.

### Proposed fix

Create the class that the docstring describes, and leave the class it is called
on unchanged:

```python
@classmethod
def create_from(cls, path_to_exe, process_name=None):
    ...
    return type(cls.__name__, (cls,), {
        "spice_exe": exe_parts,
        "process_name": process_name or cls.guess_process_name(exe_parts[0]),
    })
```

Callers that relied on the in-place rebind (`LTspice.create_from(path)` and
then using `LTspice`) would need the returned class; a deprecation period that
does both would avoid breaking them.

### Suggested upstream test

```python
def test_create_from_returns_a_class_of_its_own(tmp_path):
    a = tmp_path / "a" / "LTspice.exe"
    b = tmp_path / "b" / "LTspice.exe"
    for exe in (a, b):
        exe.parent.mkdir()
        exe.write_text("")
    before = list(LTspice.spice_exe)
    first = LTspice.create_from(str(a))
    second = LTspice.create_from(str(b))
    assert first is not second
    assert first.spice_exe == [a.as_posix()] and second.spice_exe == [b.as_posix()]
    assert LTspice.spice_exe == before
```

### Cross-reference

`tests/test_named_executables.py::TestBinding::test_each_named_executable_is_a_class_launching_its_own_program`
pins that binding two named executables leaves the family's class launching
its original program, and `TestRoutedRuns` pins that each run launches the
build it named.

---

## Bug 19 — `get_default_library_paths` reports every LTspice's library, whichever program the class runs (limitation)

**Status:** known limitation. Recorded 2026-10-03.
**Affected version:** spicelib 1.5.1 (`spicelib/simulators/ltspice_simulator.py`,
`LTspice._default_lib_paths`; `spicelib/sim/simulator.py`,
`Simulator.get_default_library_paths`).
**Our workaround:** `src/ltspice_mcp/lib/simulator.py` — `generation_of`
determines which LTspice a class launches from its program's file name
(`XVIIx64.exe` is XVII, `LTspice.exe` is 24 and later), and
`simulator_library_roots` keeps only that generation's directories from
spicelib's list (`_in_generation`). On WSL, spicelib's list expands against the
Linux home and finds nothing, so `wsl.get_ltspice_lib_paths(generation)` checks
the matching Windows folder.
Remove the filter once spicelib reports the library of the program it runs.

### Summary

`LTspice._default_lib_paths` lists the library folders of every LTspice
generation in one list (`~/AppData/Local/LTspice/lib` for LTspice 24 and later,
`~/Documents/LTspiceXVII/lib/` for XVII, and older locations), and
`get_default_library_paths` returns each one that exists. It uses the
executable only to translate paths under Wine. With XVII and a later LTspice
installed side by side, a class bound to either program reports both
libraries, so a caller cannot ask which model library the program it runs
reads.

### Affected code

```python
_default_lib_paths = ["~/AppData/Local/LTspice/lib",
                      "~/Documents/LTspiceXVII/lib/",
                      "~/Documents/LTspice/lib/",
                      "~/My Documents/LTspiceXVII/lib/",
                      "~/My Documents/LTspice/lib/",
                      "~/Local Settings/Application Data/LTspice/lib"]
```

`get_default_library_paths` walks this list and keeps every directory that
exists, with no reference to which program `spice_exe` names.

### Reproduction

With both libraries present in the user profile (`~/AppData/Local/LTspice/lib`
and `~/Documents/LTspiceXVII/lib`):

```python
xvii = LTspice.create_from("C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe")
xvii.get_default_library_paths()
# ['%USERPROFILE%/AppData/Local/LTspice/lib', '%USERPROFILE%/Documents/LTspiceXVII/lib/' (expanded)]
```

### Impact

A program that trusts the running simulator's own library extends that trust
to another program's library too. We let a deck include files from the
simulator's library even when our path allow-list (`allowed_paths`) leaves the
install out, because LTspice's netlister writes such include lines itself (for
example `.lib <install>/lib/cmp/standard.mos` for every schematic with a
MOSFET). On WSL none of the listed directories exists, so XVII's library
could not be found until we searched for it ourselves.

### Proposed fix

Give each generation its own list and pick it from the executable, or let a
subclass narrow `_default_lib_paths`, for example by keying the defaults on the
program's file name:

```python
_lib_paths_by_program = {
    "xviix64.exe": ["~/Documents/LTspiceXVII/lib/", "~/My Documents/LTspiceXVII/lib/"],
    "ltspice.exe": ["~/AppData/Local/LTspice/lib", "~/Documents/LTspice/lib/", ...],
}
```

### Suggested upstream test

```python
def test_library_paths_follow_the_program(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "Documents" / "LTspiceXVII" / "lib").mkdir(parents=True)
    (tmp_path / "AppData" / "Local" / "LTspice" / "lib").mkdir(parents=True)
    exe = tmp_path / "XVIIx64.exe"
    exe.write_text("")
    xvii = type("XVII", (LTspice,), {}).create_from(str(exe))
    assert [Path(p).parts[-2] for p in xvii.get_default_library_paths()] == ["LTspiceXVII"]
```

### Cross-reference

`tests/test_named_executables.py::TestLibraryRoots` pins both routes: each
build's roots on a native profile, and XVII's under the Windows profile on WSL.

---

## Bug 20 — `LTspice.valid_switch('-ini', path)` returns a malformed flag

**Status:** draft for an upstream spicelib pull request. Verified 2026-10-02
by source inspection and function output only; the malformed argv was not
launched.
**Affected version:** spicelib 1.5.1
(`spicelib/simulators/ltspice_simulator.py`, `LTspice.ltspice_args` and
`LTspice.valid_switch`).
**Our workaround:** `lib/controlled_ltspice.py::controlled_ltspice` launches
LTspice for a job submitted with our `recoverable` option (its inputs are saved
so it can be resumed later; native Windows only). It builds the command itself,
with `-ini` and the path as separate arguments (`-Run -b <deck> -ini <path>`).
The path names a fresh writable copy of a saved settings file that LTspice
itself had written; the saved file is never modified, and each copy is checked
against it just before launch. This launch path accepts only one
LTspice executable, identified by its SHA-256, whose startup behaviour we
inspected, so correcting this upstream typo alone does not make other builds
supported.

### Summary

`LTspice.valid_switch('-ini', path)` returns `['- ini', path]`, with an
embedded space in the flag. The installed LTspice command-line documentation
specifies `-ini <path>`. The validator substitutes the path correctly but
preserves the typo in its switch table.

### Affected code

`spicelib/simulators/ltspice_simulator.py`:

```python
ltspice_args = {
    # ...
    '-ini': ['- ini', '<path>'],
}

# In valid_switch:
switches = cls.ltspice_args[switch]
switches = [switch.replace('<path>', path) for switch in switches]
return switches
```

### Reproduction

On Windows or Linux, this calls only the switch helper; it does not start
LTspice or read a settings file:

```python
from spicelib.simulators.ltspice_simulator import LTspice

print(LTspice.valid_switch('-ini', 'owned.ini'))
# Actual in 1.5.1: ['- ini', 'owned.ini']
# Expected:       ['-ini', 'owned.ini']
```

Native macOS LTspice has a separate guard that rejects these switches and is
outside this reproduction.

### Impact

Callers trusting the validator receive an argv element that differs from the
vendor's documented flag. `LTspice.run` forwards the supplied switch list
without correcting it. The malformed flag's behavior in LTspice was not
measured; this report does not claim a particular fallback, settings change,
or dialog caused by launching it.

### Proposed fix

Change the table entry to `'-ini': ['-ini', '<path>']`. Preserve the path as
a separate argument, including when it contains spaces; do not concatenate
or shell-quote it into the flag.

### Suggested upstream test

```python
@pytest.mark.parametrize('path', ['owned.ini', r'C:\probe inputs\owned.ini'])
def test_ini_switch_preserves_flag_and_separate_path(path):
    assert LTspice.valid_switch('-ini', path) == ['-ini', path]
```

Run this on Windows/Linux or account for the existing native-macOS guard.
No simulator process or settings file is needed.

### Cross-reference

`src/ltspice_mcp/lib/controlled_ltspice.py::controlled_ltspice` bypasses the
malformed expansion. `tests/test_controlled_ltspice.py::test_launch_uses_documented_flags_and_child_environment`
pins the separate flag and path arguments and the environment local to the
launch (it sets `APPDATA` to the copy's folder for the child process only).
The other tests in that file cover the saved settings file that must not
change, the check of the writable copy before launch, the settings values as
the Windows profile API reads them, and the refusal of other executables and
launch settings.

The first investigation, on 2026-10-02, came before we had any launch code for
this and did not launch the malformed flag. Later runs on native Windows
completed a numeric batch solve and exited normally using
`-Run -b <deck> -ini <profile>` with a real copy of settings that LTspice
itself had prepared. These runs do not
establish the behavior of the malformed argv, and they do not validate every
LTspice build and solver configuration.

---

## Bug 21 — simulator launches cannot accept a per-run environment (limitation)

**Status:** known API limitation; draft for an upstream enhancement.
Verified 2026-10-02 by reading the installed signatures and the code that
forwards arguments to `subprocess`.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`Simulator.run` and `run_function`; `spicelib/simulators/ngspice_simulator.py`,
`NGspiceSimulator.run`; `spicelib/sim/run_task.py`, `RunTask.run`).
**Our workaround:** `src/ltspice_mcp/lib/controlled_ngspice.py::controlled_ngspice`
launches ngspice for a job submitted with our `recoverable` option (its inputs
and startup settings are saved so it can be resumed later). It returns an
`NGspiceSimulator` subclass whose `run` builds the command from the saved
settings and calls `subprocess.run` with an environment of its own, made from
the server's environment without any `SPICE_*` variable plus the saved
`SPICE_SCRIPTS`.
Submission and cancellation go through spicelib's `SimRunner` and `RunTask`
as for any other run.

### Summary

The simulator run interface and the shared subprocess helper have no `env`
argument. `NGspiceSimulator.run` consequently launches with the parent process's
environment. A caller cannot bind a particular `SPICE_SCRIPTS` directory to one
run through that interface. Changing `os.environ` around a launch changes shared
process state and can affect concurrent runs.

This matters when a resumed job must start ngspice the same way as the original
run. Measured runs on Linux and Windows showed that `-n` suppresses user and
local initialization but not the system spinit. Each child process therefore
needs, in addition to `-n`, a `SPICE_SCRIPTS` override that points to a spinit
we write ourselves, which holds only a comment.

### Affected code

`NGspiceSimulator.run` and the abstract `Simulator.run` have this interface:

```python
def run(cls, netlist_file, cmd_line_switches=None, timeout=None,
        stdout=None, stderr=None, cwd=None, exe_log=False):
    # ... no env parameter
```

Both console-redirection branches in `NGspiceSimulator.run` call the helper
without an environment. `RunTask.run` also supplies no per-run environment.
The shared helper in `spicelib/sim/simulator.py` is:

```python
def run_function(command, timeout=None, stdout=None, stderr=None, cwd=None):
    result = subprocess.run(command, timeout=timeout, stdout=stdout,
                            stderr=stderr, cwd=cwd)
    return result.returncode
```

### Reproduction

Bind the call signature with `inspect`, without running any executable:

```python
import inspect
from spicelib.sim.simulator import run_function
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

for function, first_arg in (
    (NGspiceSimulator.run, 'bench.cir'),
    (run_function, ['ngspice', '-b', 'bench.cir']),
):
    try:
        inspect.signature(function).bind(
            first_arg, env={'SPICE_SCRIPTS': 'controlled-startup'}
        )
    except TypeError as error:
        print(function.__qualname__, error)
    # Both print: got an unexpected keyword argument 'env'
```

### Impact

Runs that need different startup directories cannot pass those environments
through spicelib's simulator interface. Startup files inherited from the ambient
environment can change simulator behavior even when the deck bytes are
unchanged. Working around this by changing the process-wide environment would
also make the result depend on how launches overlap in time.

### Proposed fix

Add an optional keyword-only `env=None` to the simulator interface and
`run_function`, forward it to `subprocess.run` in both output-redirection
branches, and pass a per-run mapping through `SimRunner`/`RunTask`. Preserve
existing inheritance behavior when `env` is `None`. Document that an explicit
mapping is the child environment, matching `subprocess.run`, and never mutate
the parent's environment to implement it.

### Suggested upstream test

Use a Python child process instead of a simulator to test the real helper. Give
two concurrent calls different values of a test environment variable, have each
child assert its own value, and assert that the parent's environment is unchanged.
Test `env=None` inheritance separately. Through `SimRunner`, `RunTask` and the
simulator's `run`, assert that each mapping reaches the helper unchanged for
both `exe_log` modes.

### Cross-reference

Our subclass copies the server's environment, removes every `SPICE_*`
variable, adds the `SPICE_SCRIPTS` value saved with the job, and adds `-n`. It
changes neither `os.environ` nor the compatibility mode of `NGspiceSimulator`
itself (it sets `_compatibility_mode` on its own subclass).
`tests/test_controlled_ngspice.py::test_concurrent_launches_keep_separate_environments`
pins separate environment mappings and an unchanged parent environment.
`::test_verification_refuses_before_spawn` pins that a failed check of the
saved inputs stops the launch before any process starts.
`::test_real_ngspice_uses_only_the_recorded_startup` runs a real ngspice process
with `SPICE_SCRIPTS` and `SPICE_USERINIT_DIR` pointing at startup files that
print a marker and quit, and checks its numeric output and that the marker is
absent.

An upstream `env` argument would let us delete our duplicate of the subprocess
call. The checks that the saved input files and startup settings are unchanged
would stay in this project's recovery code.

---

## ngspice replaces a system startup seed before electrical deck evaluation

**Status:** reproduced defect in ngspice; draft for an ngspice report. The
defect is in ngspice initialization rather than in a spicelib parser or
adapter.
**Affected version:** ngspice 42, observed with the Linux console package
`42+ds-3build1`. Matching upstream source defines `WaGauss` and calls
`frontend/trannoise/wallace.c::initw` during startup.
**Our workaround:** we never set a seed in spinit.
`src/ltspice_mcp/lib/ngspice_driver.py::seeded_commands` writes a short control
script (a driver script) that runs `setseed <seed>`, then `source <deck>`,
`run` and `write`, so the seed is set directly before ngspice reads the deck.
`pdk_native.driver_bytes` uses it for Monte Carlo samples that ngspice draws
from a PDK's own statistical models (our native statistics), and
`controlled_ngspice.prepare_seeded_driver` uses it for a job submitted with our
`recoverable` option and an explicit `simulator_seed`. An explicit
`simulator_seed` cannot be combined with native statistics, which give each
sample its own seed.

### Summary

A positive `setseed N` in system spinit executes and sets the `rndseed`
variable, but the `AGAUSS` values in the circuit deck that ngspice loads at
startup (the electrical deck) are not controlled by that seed. Later process
initialization overwrites the shared generator state. The variable still
reports the caller's seed, so checking that variable or checking a random draw
within spinit can falsely suggest reproducibility.

### Affected code and mechanism

The matching [ngspice 42 main routine](https://github.com/imr/ngspice/blob/ngspice-42/src/main.c)
first installs a default seed, then calls `ft_cpinit`. That initialization
loads system spinit and executes its commands. Before loading the initial
electrical deck, `main` calls `initw` under the `WaGauss` build definition.

The [Wallace initializer](https://github.com/imr/ngspice/blob/ngspice-42/src/frontend/trannoise/wallace.c)
unconditionally performs:

```c
srand((unsigned int) getpid());
TausSeed();
```

It then fills its Gaussian pool using the shared uniform generator. These
calls replace the state established by `setseed`, without updating `rndseed`.
The local executable's disassembly confirms the compiled `getpid`, `srand`,
and `TausSeed` sequence followed by pool allocation and filling.

The [parameter evaluator](https://github.com/imr/ngspice/blob/ngspice-42/src/frontend/numparam/xpressn.c)
implements `agauss` through `gauss1`. The
[random-number implementation](https://github.com/imr/ngspice/blob/ngspice-42/src/maths/misc/randnumb.c)
draws directly from the shared Tausworthe/LCG state; it does not restore that
state from `rndseed` before the draw.

### Reproduction

Create a new startup directory containing `spinit` with `setseed 17`, and a
new `bench.cir` containing:

```spice
* Seeded parameter evaluation
.param draw=agauss(10,1,1)
V1 n 0 {draw}
R1 n 0 1000
.op
.end
```

Run the same `bench.cir` bytes twice as separate ngspice processes, with that
directory selected through a `SPICE_SCRIPTS` variable set in each child's
environment:

```text
ngspice -n -D ngbehavior=hsa -b -o first.log -r first.raw bench.cir
ngspice -n -D ngbehavior=hsa -b -o second.log -r second.raw bench.cir
```

Use a 10-second timeout per process and compare the `v(n)` values in the two
operating-point raws. The observed seed-17 values were `9.484939411740886 V`
and `12.431703780774 V`. A diagnostic spinit reported `rndseed=17` in both
processes, and control-language `sgauss(0)` within spinit also returned the
same value in both, while the later electrical values differed. User and local
initialization was suppressed with `-n`; no user profile or installed startup
file was changed.

For comparison, we ran the project's unchanged driver generator on the same
deck bytes with the sequence `setseed -> source -> run -> write`. Two
seed-17 runs produced `9.940377579172193 V` with identical numeric binary
payloads; seed 19 produced `9.476947311890937 V`. All three exited zero.
This sequence resets the generator after Wallace initialization and directly
before electrical input evaluation.

### Impact

A run seed recorded only in spinit cannot guarantee repeatable random
parameter values in the electrical deck on the affected build. Unchanged input
files and an unchanged executable are not enough when the shared generator
state is later initialized from the process ID.

### Proposed fix

Initialize Wallace state before executing startup files, or make its setup
preserve the caller's seed and define how its pool and the shared generator
are initialized. Keep the reported seed consistent with the generator state.
Avoid claiming that resetting the shared generator alone reseeds an
already-populated Wallace pool.

### Suggested upstream test

Through the real console startup path, load a spinit written for the test with
a positive seed, and solve the operating point of a small deck that uses
`.param AGAUSS` in two separate processes. Assert identical numeric results
for the same seed and different results for a selected different seed. Cover
the `WaGauss` build, `-n`, a per-child `SPICE_SCRIPTS` override, and batch/raw
output. Test transient noise separately because it can draw from the Wallace
pool instead of the parameter evaluator's generator.

### Cross-reference

`ngspice_driver.seeded_commands` writes the command order both features share,
and `pdk_native.driver_bytes` uses it for native statistics samples. For a
recoverable job with `simulator_seed`,
`controlled_ngspice.prepare_seeded_driver` saves its own driver script, and
`verify_seeded_driver` checks the script's path, bytes and SHA-256 before each
launch.
`tests/test_recovery_seed.py::test_shared_sequence_preserves_the_audited_native_driver_bytes`
pins that native statistics keep producing the same driver bytes from the
shared sequence; `::test_real_seeded_coordinator_reproduces_electrical_results`
and `::test_seeded_retry_keeps_successes_and_reuses_recorded_seed` cover seeded
runs and retries through our normal job runner.

`native_execution.prepare_native_cases` writes each sample's deck and driver
script before the job is recorded, and `pdk_native.verify_launch` checks both
against their saved SHA-256 before the run is submitted.
`tests/test_pdk_native.py::test_windows_paths_with_spaces_use_only_generated_relative_names`
checks the generated driver. A separate check that launches console ngspice
three times confirms repeatable numeric results for the small `.param AGAUSS`
deck. This project does not claim that a seed set in a startup file controls
the results. These results do not establish reproducibility for every
stochastic function or analysis.

---

## Bug 22 — `AscEditor` cannot open a sheet whose block symbol has no sheet of its own (limitation)

**Status:** known limitation; draft for an upstream enhancement. Observed
2026-10-06 against the exports of LTspice 26.1.1 and LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist` and `AscEditor._get_subcircuit`).
**Our workaround:** `lib/schematic_ops.py::_AscEditor`, which `make_editor`
uses to open every sheet, overrides `_get_subcircuit` for a block symbol with
no library. It looks for the sheet in the same places as spicelib and, when
there is none, leaves the instance without a resolved subcircuit, as spicelib
already does for a cell symbol with no library. A sheet that exists also opens
as an `_AscEditor`, so a block deeper in the hierarchy is read the same way.
spicelib's search walks every folder it is given and is not cached, so the
override looks for each missing sheet once each time the parent is opened
instead of once per instance.

### Summary

A symbol of `SymbolType BLOCK` represents a subcircuit. LTspice netlists an
instance of one as a call to a subcircuit of the symbol's name, whether or not
a sheet of that name exists. The definition may come from a library named on
the sheet, or be missing until the deck is run. `AscEditor` instead resolves
every block symbol to its own `.asc` while it loads the parent, and raises
`FileNotFoundError` when there is none. A sheet that LTspice exports without
an error therefore cannot be opened, read or edited at all.

### Affected code

`spicelib/editor/asc_editor.py`, `_get_subcircuit` (~line 303):

```python
lib = symbol.get_library()
if lib is None and symbol.symbol_type == "BLOCK":
    asc_filename = symbol.get_schematic_file()
    ...
    if asc_path is None:
        raise FileNotFoundError(f"File {asc_filename} not found")
    answer = AscEditor(asc_path)
```

The same lookup for a `CELL` symbol with no library returns `None` and the load
continues.

### Reproduction

`probe4.asy` beside the sheet, with `SymbolType BLOCK` and `SYMATTR Prefix X`,
and no `probe4.asc`:

```
Version 4
SHEET 1 880 680
SYMBOL probe4 96 480 R0
SYMATTR InstName U1
```

```python
AscEditor("block_symbol.asc")   # FileNotFoundError: File ...probe4.asc not found
```

`LTspice.exe -netlist block_symbol.asc` exits 0 and writes
`X§U1 NC_01 NC_02 NC_03 NC_04 probe4` (LTspice XVII: `XU1 ...`). The sheet, the
symbol and both exports are recorded under
`tests/fixtures/ltspice_recorded/` as `export/block_symbol`.

### Impact

- A sheet using a block symbol whose subcircuit is defined in a library, or not
  yet drawn, cannot be opened by any tool that reads schematics through
  `AscEditor`, though LTspice itself reads and netlists it.

### Proposed fix

Treat a block symbol with no sheet the way a cell symbol with no library is
already treated: leave the instance without a resolved subcircuit instead of
failing the load, and raise only when a caller requests the subcircuit's
contents.

### Suggested upstream test

```python
def test_block_symbol_without_its_sheet_still_loads(tmp_path):
    # probe4.asy (SymbolType BLOCK) beside parent.asc, no probe4.asc
    editor = AscEditor(tmp_path / "parent.asc")
    assert "U1" in editor.get_components()
```

### Cross-reference

`tests/test_recorded_ltspice_schematics.py::TestExportedNames::test_a_block_symbol_with_no_sheet_of_its_own_opens_as_ltspice_reads_it`
pins that LTspice exports the sheet, that spicelib's own `AscEditor` still
refuses it with a message naming `probe4.asc`, and that the server's editor
opens it; `test_a_sheet_with_a_block_symbol_of_no_sheet_can_be_edited` edits it
through `edit_schematic`. Once upstream loads such a sheet, the refusal
assertion fails, and `_AscEditor` can be removed.

---

## Bug 23 — a simulator launch cannot be kept off the user's desktop (limitation)

**Status:** known API limitation; draft for an upstream enhancement. Measured
2026-10-06 on Windows 11 with LTspice 26.1.1 and LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`run_function`; `spicelib/simulators/ltspice_simulator.py`, `LTspice.run` and
`LTspice.create_netlist`).
**Our workaround:** `lib/ltspice_windows.py` provides the LTspice class that a
server on native Windows launches. It overrides `run` and `create_netlist` to
start spicelib's own command lines through `lib/hidden_desktop.py`, on a
desktop that the server creates. The code that records LTspice's output for our
test fixtures (`tests/ltspice_recorder.py`) uses the same launch.

### Summary

LTspice opens a window even for a batch run (`-Run -b`) and for a batch export
(`-netlist`), and holds the keyboard focus until it exits. `run_function`
starts it with a bare `subprocess.run`, and `Simulator.run` offers no way to
pass startup information, so every simulation takes the focus from whatever
the user is typing into, and a sweep takes it continuously.

The usual remedy does not work. Started with `STARTF_USESHOWWINDOW` and
`SW_SHOWMINNOACTIVE` or `SW_HIDE`, LTspice was still the foreground window for
about 75% of a run. Starting it with `STARTUPINFO.lpDesktop` set to a desktop
made with `CreateDesktopW` works, because its windows then exist only on that
desktop. Python's `subprocess.STARTUPINFO` has no `lpDesktop`, so this needs
`CreateProcessW` through `ctypes`.

The visible window causes a second problem. For some inputs LTspice shows a
message box and waits (XVII, given a sheet that starts with a byte order mark:
"Aborting: Unknown schematic syntax"). On the user's desktop a stray key press
dismisses it, and the launch appears to have ended by itself. On a separate
desktop the launch blocks until its timeout, which is behaviour a caller can
rely on.

### Affected code

`spicelib/sim/simulator.py`:

```python
def run_function(command, timeout=None, stdout=None, stderr=None, cwd=None):
    result = subprocess.run(command, timeout=timeout, stdout=stdout, stderr=stderr, cwd=cwd)
    return result.returncode
```

### Reproduction

Sample the foreground window's owning process every millisecond while
`LTspice.run("deck.cir")` runs a deck that takes about a second.

| launch | samples owned by LTspice |
|-|-|
| `SW_SHOWMINNOACTIVE` | 283 of 362 |
| `SW_HIDE` | 272 of 348 |
| own desktop, LTspice 26.1.1 | 0 of 356 |
| own desktop, LTspice XVII | 0 of 375 |

On its own desktop the run exits 0 and writes the same log and raw.

### Impact

- Any interactive use of a tool built on `LTspice.run` interrupts the user's
  typing for the length of every run.
- A message box the simulator raises can be dismissed by that typing, so
  whether a launch returns depends on what the user happened to press.

### Proposed fix

Let a caller control how the process is started, at least with a
`startupinfo` / `creationflags` pass-through on `Simulator.run` and
`create_netlist`, or with an optional launcher callable in place of
`run_function`. A
`desktop=` option on Windows would cover this case directly.

### Suggested upstream test

On Windows, start a simulator through the new hook on a desktop created for
the test, and assert the foreground window's owning process never becomes the
simulator's while the run completes with exit code 0.

### Cross-reference

`lib/hidden_desktop.py` implements the working launch. It also reads the text
of a message box (`HiddenDesktop.dialog`), so a build that stops to ask a
question is ended and reported instead of waiting where nobody can respond.
The recordings of `export/micro_utf8_bom` and `export/micro_utf16le_bom` on
LTspice XVII contain that text. `tests/test_ltspice_windows.py` pins that the
overriding class runs the command lines spicelib builds, and
`tests/test_ltspice_integration.py::TestWindowStaysOffTheDesktop` checks that a
real LTspice opens no window on the user's desktop. Once upstream offers a
hook, `lib/ltspice_windows.py` can pass the desktop through it, and its two
overrides can be removed.

---

## Bug 24 — `detect_encoding` gives up on an 8-bit log holding a byte none of its codecs defines

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-06
against a log written by LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/utils/detect_encoding.py`,
`detect_encoding`; reached from `spicelib/log/ltsteps.py`,
`LTSpiceLogReader.__init__`).
**Our workaround:** `lib/log_parser.py::make_log_reader` decodes the log
itself (`lib/encoding.py`, which has a character for every byte) and, when
spicelib has refused the file, passes spicelib a UTF-8 copy.

### Summary

LTspice XVII copies a deck's title line into the first line of its log, byte
for byte as the deck contains it (`Circuit: * <title>`). A title saved in a
double-byte code page, such as Japanese or Chinese, contains bytes that are
undefined in cp1252 (0x81, 0x8D, 0x8F, 0x90, 0x9D). `detect_encoding` opens
the file in each of a fixed list of codecs and returns the first that decodes
it and matches the expected pattern. For such a log none does, so it raises
`EncodingDetectError: Expected pattern ... not found`, and
`LTSpiceLogReader` cannot be built for a run that finished without error.
The message reports a missing pattern, although the `Circuit:` line is
present.

### Affected code

`spicelib/utils/detect_encoding.py`, `detect_encoding` (~line 49):

```python
for encoding in ('utf-8', 'utf-16', 'utf_16_le', 'windows-1252', 'cp1252', 'cp1250', 'shift_jis'):
    try:
        with open(file_path, encoding=encoding) as f:
            lines = f.read()
    except UnicodeDecodeError:
        continue
    ...
else:
    if expected_pattern:
        raise EncodingDetectError(f"Expected pattern \"{expected_pattern}\" not found in file:{file_path}")
```

Every codec in the list leaves some byte undefined, so no entry accepts every
8-bit file. `shift_jis` decodes most Japanese text but not the NEC and IBM
extensions that cp932 adds, and no codec in the list decodes GBK or the Korean
extended range.

### Reproduction

```python
from pathlib import Path
from spicelib.log.ltsteps import LTSpiceLogReader

log = Path("title.log")
log.write_bytes(b"Circuit: * \x81a \x80f\n\nDate: Thu Jan 15 00:00:00 2026\n")
LTSpiceLogReader(str(log))     # EncodingDetectError: Expected pattern ... not found
```

A log XVII wrote for such a deck is recorded under
`tests/fixtures/ltspice_recorded/ltspice17/` as `deck/bytes_outside_cp1252.log`.

### Impact

- `.MEAS` results, step values and Fourier blocks of a run that succeeded
  cannot be read through `LTSpiceLogReader` when the deck's title is in a
  double-byte code page and the build is LTspice XVII.
- The error reports a missing pattern, which leads the reader to look for a
  malformed log.

### Proposed fix

End the list with a codec that has a character for every byte (`latin-1`), so
an 8-bit log is always read. Every field the reader parses is ASCII, and the
title is only copied through. Report an encoding failure as an encoding
failure, separately from a missing pattern.

### Suggested upstream test

```python
def test_log_with_a_byte_no_listed_codec_defines(tmp_path):
    log = tmp_path / "title.log"
    log.write_bytes(b"Circuit: * \x81a \x80f\n\nm1: MAX(v(out))=1 FROM 0 TO 1\n")
    assert LTSpiceLogReader(str(log)).get_measure_names() == ["m1", "m1_from", "m1_to"]
```

### Cross-reference

`tests/test_recorded_ltspice_results.py::test_a_run_whose_title_holds_a_byte_cp1252_lacks_is_read`
reads the recorded log through `parse_measurements`. Once upstream reads
such a log, the last retry in `make_log_reader`, which passes spicelib the
log's text unchanged in a UTF-8 copy, can be removed.

---

## Bug 25 — `AscEditor` drops a sheet's `DATAFLAG` records on save

**Status:** draft for an upstream spicelib pull request. Found by reading the
source; reproduced 2026-10-07, including on a sheet that recordings show both
LTspice 26 and LTspice XVII read (`export/data_flags`; see *Reproduction*).
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist` ~line 272 and `AscEditor.save_netlist` ~line 81).
**Our workaround:** `tools/schematic_edit.py` reads the sheet's DATAFLAG
lines from the bytes the edit already read (`_data_flag_lines`), and
`_render_editor_text` writes them back (`_with_data_flags`).

### Summary

A `DATAFLAG` record is a data label, an expression that LTspice shows at a
point on the sheet. spicelib's own comment calls it "the placeholder to show
simulation information", and KiCad's LTspice importer reads it as
`DATAFLAG <x> <y> <expression>`. The example sheets that both LTspice builds
install contain it in that form with an empty expression
(`DATAFLAG 2128 1360 ""` in `examples/Educational/DCopPnt.asc`).
`reset_netlist` skips the line and stores nothing from it, so `save_netlist`
has no `DATAFLAG` to write. Loading a sheet and saving it, with or without a
change, therefore removes every data label on it. The 1.3.2 changelog lists
"AscEditor: Adding support to DATAFLAG"; in 1.5.1 that support consists of
skipping the line, so the line no longer stops the read as an unmodelled line
does (Bug 26).

### Affected code

`spicelib/editor/asc_editor.py`, `reset_netlist` (~line 272):

```python
elif line.startswith("DATAFLAG"):
    pass  # DATAFLAG is the placeholder to show simulation information. It is ignored by AscEditor
```

`save_netlist` writes `Version`, `SHEET`, the wires, the labels, each symbol
with its `WINDOW` and `SYMATTR` lines, the texts and the drawings, and nothing
else.

### Reproduction

```python
import io
from pathlib import Path
from spicelib.editor.asc_editor import AscEditor

sheet = Path("dataflag.asc")
sheet.write_bytes(
    b"Version 4\nSHEET 1 880 680\nWIRE 0 0 160 0\nFLAG 0 0 IN\n"
    b'DATAFLAG 80 0 ""\nDATAFLAG 160 0 "$*2"\nTEXT 0 64 Left 2 !.op\n'
)
out = io.StringIO()
AscEditor(sheet).save_netlist(out)
assert "DATAFLAG" not in out.getvalue()   # both data labels are gone
```

Before the workaround, an edit through our `edit_schematic` tool that added one
net label to such a sheet reported the edit as saved, and the file it wrote
contained no `DATAFLAG` line.

The sheet above is the smallest that shows the loss. The same loss is
reproduced on a recorded sheet,
`tests/fixtures/ltspice_recorded/inputs/export/data_flags.asc`, which contains
a resistor, a wire, a port and two data labels. LTspice 26 and LTspice XVII
each export it (`export/data_flags` in the recordings), and neither exports a
sheet containing a keyword it does not know (`export/unknown_record`), so both
read the record. The netlist contains only the resistor.

A recording cannot show where LTspice places the record when it saves a sheet,
because LTspice saves a sheet only from its window (`schematic-save-encoding`
in `tests/fixtures/ltspice_recorded/inputs/cases.toml`). The evidence for the
placement comes from the example sheets that each build installs, which
LTspice saved. These are `examples/Educational/DCopPnt.asc` and `Linkwitz.asc`
in both builds, and `examples/Applications/AD8397.asc` and `ADA4691.asc` in
LTspice 26. Each lists its `DATAFLAG` records together, after the last `FLAG`
and before the first `SYMBOL`, every one as `DATAFLAG <x> <y> ""`. None of them
contains a port (`IOPIN`) or a non-empty expression, so there is no record of
where a data label goes relative to a port's record or of how an expression is
written. An export does not answer either question. Both builds exported
sheets with the record after a symbol, between a `FLAG` and its `IOPIN`, and
with no expression at all, and each produced the same netlist as the sheet
without the record (checked 2026-10-07, not recorded).

### Impact

- Any tool that edits a sheet through `AscEditor` and saves it deletes the
  sheet's data labels without reporting it. The circuit is unchanged, so
  neither a netlist comparison nor a simulation shows the loss; the user finds
  the labels missing the next time the sheet is opened in LTspice.

### Proposed fix

Keep each `DATAFLAG` as a positioned text (coordinate and expression, the
expression verbatim) in a list of its own, and write those records in
`save_netlist` after the labels and their `IOPIN` records. A data label is
placed by its coordinate alone, so it needs no association with another
record.

### Suggested upstream test

```python
def test_data_flags_survive_a_save(tmp_path):
    p = tmp_path / "flags.asc"
    p.write_text('Version 4\nSHEET 1 880 680\nWIRE 0 0 160 0\nDATAFLAG 80 0 ""\n')
    out = io.StringIO()
    AscEditor(p).save_netlist(out)
    assert 'DATAFLAG 80 0 ""\n' in out.getvalue()
```

### Cross-reference

`tests/test_edit_schematic.py::TestDataFlagPreservation` edits a sheet
containing two data labels and a hierarchical port, reads the file back, and
checks that both data labels are present, in their order, after the net labels
and their ports. A second sheet is in cp1252 with a micro sign in its label,
which is written back in the sheet's own codec.
`tests/test_recorded_ltspice_schematics.py::TestSheetRecords` applies the same
checks to the recorded sheet. Both builds export it with no trace of the data
labels in the netlist, and an edit that sets the resistor's value saves the
sheet unchanged except for that value. The workaround places the records
together after the net labels and before the first symbol, where the example
sheets that both builds install place them (*Reproduction*). This position
keeps them out of a symbol's block of `WINDOW` and `SYMATTR` lines and out of
the space between a `FLAG` and its `IOPIN`. `TestPathsThatLeaveTheSheetAsItIs`
pins that a read, a dry run and an edit that only sets waveform plot panes
(stored in the `.plt` file beside the sheet) leave such a sheet's bytes
unchanged, because none of them rewrites the sheet. Once upstream keeps the
records, `_data_flag_lines` and `_with_data_flags` can be removed.

---

## Bug 26 — a line `AscEditor` does not model makes the whole sheet unreadable (limitation)

**Status:** known limitation; draft for an upstream enhancement. Found by
reading the source; reproduced 2026-10-07. Recordings show that LTspice 26 and
LTspice XVII both read the two kinds of sheet it refuses (`export/bus_tap`,
`export/blank_line`).
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist` ~line 274).
**Our workaround:** `lib/schematic_ops.py::make_editor` converts the
`NotImplementedError` into a `NetlistError` (`_unreadable_record`) that names
the sheet containing the line (the sheet being opened or a sheet it loads) and
gives the line's number and text. It finds that sheet in the traceback. The
innermost `reset_netlist` frame is the read that refused; its `self` gives the
file and codec, and its `line` variable gives the line.

### Summary

`reset_netlist` dispatches each line on the keyword it starts with and ends in
`raise NotImplementedError`. A record it does not model therefore makes the
whole sheet unreadable, and so does any line that does not start with a known
keyword in its first column, such as an empty or indented line. The message
quotes the line but names neither the file nor the line number, and since the
sheet of a block symbol is read while its parent loads, the line may be in a
different file from the one being opened.

In a sheet written by LTspice, the record most likely to cause this is a bus
tap. KiCad's LTspice importer reads every keyword spicelib reads and one more,
`BUSTAP <x1> <y1> <x2> <y2>`. Both LTspice builds know the keyword. Each
exports a sheet containing two taps (`export/bus_tap` in
`tests/fixtures/ltspice_recorded`), and neither exports a sheet containing a
keyword it does not know (`export/unknown_record`: LTspice 26 exits 0 without
writing anything, and XVII stops on a message box, "Unknown schematic
syntax"). Both also export a sheet containing an empty line
(`export/blank_line`). spicelib therefore refuses sheets that LTspice reads.

In the netlist a tap connects nothing. Of the recorded sheet's three
resistors, the one on the bus `D[0:3]` is on node `D[0]`, the one on a tapped
wire labelled `D0` is on `D0`, and the one on a tapped wire with no label is
on a node of its own, `N001`. A reader that keeps the record and derives no
connection from it therefore gets the same connections at the tap as LTspice.
The node that a bus's own label names is a separate question that does not
involve spicelib. Both builds put a wire labelled `D[0:3]` on the node of a
wire labelled `D[0]` (`connectivity/bus_label`), and our own grouping of a
sheet's wires into nets (`lib/connectivity.py`) does not, which
`tests/test_recorded_ltspice_schematics.py` pins as
`JOINED_BY_A_BUS_LABEL_IN_LTSPICE_ONLY`.

The field layout still comes from KiCad's importer, not from a recording. No
example sheet that either build installs contains a bus tap, and the reference
documents that LTspice 26 installs mention it only as a menu command (Place
Bus Tap). An export does not check the fields; both builds exported a sheet
whose `BUSTAP` had two of the four (checked 2026-10-07, not recorded). There
is also no record of where LTspice places the record in a sheet it saves.

### Affected code

`spicelib/editor/asc_editor.py`, `reset_netlist` (~line 274):

```python
elif line.startswith("DATAFLAG"):
    pass  # DATAFLAG is the placeholder to show simulation information. It is ignored by AscEditor
else:
    raise NotImplementedError("Primitive not supported for ASC file\n"
                              f'"{line}"')
```

### Reproduction

```python
from pathlib import Path
from spicelib.editor.asc_editor import AscEditor

Path("bustap.asc").write_bytes(
    b"Version 4\nSHEET 1 880 680\nWIRE 0 0 160 0\nFLAG 0 0 D[0:3]\n"
    b"BUSTAP 80 0 80 16\n"
)
AscEditor("bustap.asc")
# NotImplementedError: Primitive not supported for ASC file
# "BUSTAP 80 0 80 16
# "

Path("blank_line.asc").write_bytes(b"Version 4\nSHEET 1 880 680\n\nWIRE 0 0 160 0\n")
AscEditor("blank_line.asc")    # the same error, for the line "\n"
```

Before the workaround, our `edit_schematic` tool failed on the first sheet with
`Internal error in edit_schematic: NotImplementedError: ...`, and the component
and net listings of our `inspect` tool returned our generic `internal_error`
code for it.

The two sheets above are the smallest that show the error. The recorded sheets
`export/bus_tap.asc` and `export/blank_line.asc` under
`tests/fixtures/ltspice_recorded/inputs` raise it the same way, and the
recordings show both builds exporting those sheets.

### Impact

- One record spicelib does not model makes a sheet impossible to open, read or
  edit through `AscEditor`, and with it every sheet that uses that sheet as a
  block.
- LTspice reads these sheets (*Summary*), so a sheet a person drew, and that
  LTspice opens, exports and runs, cannot be opened at all.
- The exception type is "not implemented" and the message gives no file or
  line, so a caller cannot tell an unreadable sheet from a defect, or find the
  line in a hierarchy.
- Downstream, our `verify_circuit` tool reads a sheet with our own parser
  (`lib/asc_document.py::parse_asc`, drawn by
  `lib/schematic_scene.py::build_scene`), which skips a keyword it does not
  draw, so it draws and checks such a sheet without that record while the
  editing tools refuse it. It reports each such keyword as an observation (a
  note in its result), with the number of lines that contain it and the first
  such line.

### Proposed fix

The fix has two parts. First, raise a parse error that names the file and the
line number (for example a `ValueError` subclass) instead of
`NotImplementedError`. Second, keep a record that the editor does not model
verbatim, in its place among the records around it, and write it back on save,
so a sheet containing a bus tap can be opened and edited without losing it.

### Suggested upstream test

```python
def test_an_unmodelled_record_is_kept(tmp_path):
    p = tmp_path / "bus.asc"
    p.write_text("Version 4\nSHEET 1 880 680\nWIRE 0 0 160 0\nBUSTAP 80 0 80 16\n")
    out = io.StringIO()
    AscEditor(p).save_netlist(out)
    assert "BUSTAP 80 0 80 16\n" in out.getvalue()

def test_an_unreadable_line_names_its_file_and_line(tmp_path):
    p = tmp_path / "bad.asc"
    p.write_text("Version 4\nSHEET 1 880 680\nNOSUCHRECORD 1 2\n")
    with pytest.raises(ValueError, match=r"bad\.asc.*line 3"):
        AscEditor(p)
```

### Cross-reference

`tests/test_edit_schematic.py::TestRecordTheEditorCannotRead` pins that an
edit of a sheet containing a bus tap or an empty line is refused with a
`NetlistError` that names the file, line 5 and the line, and that the sheet is
left unchanged. It also pins that `inspect`'s `components` and `net` queries
return the same refusal instead of `internal_error`, and that a bus tap in a
block's sheet is reported in that sheet, not in the parent being opened.
`tests/test_verify_circuit.py::test_a_record_the_drawing_does_not_read_is_reported`
pins the observation that `verify_circuit` reports for the same record.
`tests/test_recorded_ltspice_schematics.py::TestSheetRecords` checks all of
this against the recorded sheets. It covers how each build handles a bus tap,
an empty line and an unknown keyword; the refusal the editor gives for the two
sheets LTspice reads (listed there as
`EXPORTED_BY_LTSPICE_AND_REFUSED_BY_THE_EDITOR`, with the line each refusal
names); the observation `verify_circuit` reports; and that its drawing, which
skips a tap, shows each tapped wire ending unconnected, as the netlist does.
Once upstream opens such a sheet, `_unreadable_record` can be removed. An edit
of a sheet containing a bus tap then needs the same check as Bug 25, that a
saved edit writes the record back instead of dropping it.

---

## Bug 27 — `detect_encoding` refuses a sheet behind a UTF-8 byte order mark as having no `Version` line, and reads one behind a UTF-16 mark

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-07
on the recorded inputs `export/micro_utf8_bom.asc` and
`export/micro_utf16le_bom.asc`.
**Affected version:** spicelib 1.5.1 (`spicelib/utils/detect_encoding.py`,
`detect_encoding`; reached from `AscEditor.__init__` with the pattern
`^VERSION `).
**Our workaround:** `lib/schematic_ops.py::make_editor` converts the
`EncodingDetectError` into a `NetlistError` (`_unrecognised_sheet`) that names
the UTF-8 mark; `tools/schematic_edit.py::_commit_codec` writes a sheet that
started with a UTF-16 mark back as UTF-16 LE without it; and the `quality`
check of our `verify_circuit` tool reports either mark as a finding with the
rule id `byte_order_mark`. `lib/encoding.py`
defines which marks LTspice refuses (`refused_sheet_mark`) and the note
reported for them (`refused_sheet_mark_note`).

### Summary

`detect_encoding` opens the file in each codec of a fixed list and keeps the
first whose text matches the pattern. `utf-8` decodes a UTF-8 byte order mark
to U+FEFF, which stays at the start of the text, so `^VERSION ` does not match
there or in any later codec. The error then reports that the pattern was not
found in a sheet whose first line, after the mark, is `Version 4`. A UTF-16
mark behaves the opposite way. Python's `utf-16` stream decoder consumes it, so
that sheet opens. Because `save_netlist` reopens the file in the codec it was
read in, a save writes the mark back.

Neither LTspice build reads either sheet (recorded). LTspice 26 exits 0
without writing a netlist, and XVII stops on "Aborting: Unknown schematic
syntax: ... Version 4". Refusing these sheets therefore matches LTspice. The
faults are the message and the different treatment of the two marks.

### Affected code

`spicelib/utils/detect_encoding.py`, `detect_encoding` (~line 49):

```python
for encoding in ('utf-8', 'utf-16', 'utf_16_le', 'windows-1252', 'cp1252', 'cp1250', 'shift_jis'):
    ...
    if expected_pattern:
        if not re.match(expected_pattern, lines, re_flags):
            continue
    ...
else:
    if expected_pattern:
        raise EncodingDetectError(f"Expected pattern \"{expected_pattern}\" not found in file:{file_path}")
```

### Reproduction

```python
from spicelib.editor.asc_editor import AscEditor

AscEditor("micro_utf8_bom.asc")
# EncodingDetectError: Expected pattern "^VERSION " not found in file:micro_utf8_bom.asc

editor = AscEditor("micro_utf16le_bom.asc")
editor.encoding                    # 'utf-16'
editor.save_netlist("saved.asc")   # saved.asc starts with b'\xff\xfe'
```

Both sheets are the recorded inputs named above. Before the workaround,
`inspect` and `edit_schematic` through our server failed on the first with an
internal error, and an edit of the second saved a sheet that still started with
the mark, so neither build could export the edited sheet either.

### Impact

- A caller cannot tell a sheet with a byte order mark from a file that is not
  a schematic, because the message refers to the `Version` line, which is
  present.
- A sheet with a UTF-16 mark opens and saves with the mark, so a tool that
  edits it returns a sheet that LTspice does not read, without reporting it.

### Proposed fix

Match the pattern after a byte order mark, and return a codec that names the
mark (`utf-8-sig` for UTF-8, as `utf-16` already does for UTF-16), so both
marks are read and written back the same way and a caller that wants to refuse
or drop a mark can detect it. When nothing matches, report the start of the
file instead of only reporting that the pattern was not found.

### Suggested upstream test

```python
def test_a_utf8_byte_order_mark_is_read_past_and_named(tmp_path):
    p = tmp_path / "bom.asc"
    p.write_bytes(codecs.BOM_UTF8 + b"Version 4\nSHEET 1 880 680\n")
    assert detect_encoding(p, r"^VERSION ", re.IGNORECASE) == "utf-8-sig"
```

### Cross-reference

`tests/test_recorded_ltspice_schematics.py::TestExportEncoding` checks the
server against the recordings in
`test_the_editor_names_the_utf8_byte_order_mark_it_cannot_read`,
`test_an_edit_writes_a_utf16_sheet_without_the_mark_neither_build_reads` and
`test_verify_reports_the_byte_order_mark_neither_build_reads`. If upstream
starts reading the UTF-8 sheet, it will open as the UTF-16 one does.
`_unrecognised_sheet` then no longer runs for it, and `_commit_codec` must drop
that mark as it drops the UTF-16 one.

---

## Bug 28 — `AsyReader` cannot read a symbol with a space in a pin name

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-07
against the symbol library and example sheets installed with LTspice 26.1.1.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asy_reader.py`,
`AsyReader.__init__`; reached from `spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist` and `AscEditor._get_symbol`).
**Our workaround:** none. A sheet that places such a symbol cannot be opened
by anything that reads it through `AscEditor`, which in this project is the
`edit_schematic` tool. Our `inspect` and `verify_circuit` tools read and check
it, because they read sheets with our own reader and symbols with
`lib/symbol_file.py`, which takes everything after `PINATTR PinName` as the
name.

### Summary

A pin's name may contain a space; LTspice's own library has pins named `OUT A`
and `INV B`. `AsyReader` splits a `PINATTR` line on spaces into at
most four parts and unpacks them into three names, so a name with one space
raises `ValueError: too many values to unpack (expected 3)` and the symbol is
not read. `AscEditor` reads every symbol a sheet places while it loads the
sheet, so the sheet is not read either.

### Affected code

`spicelib/editor/asy_reader.py`, `AsyReader.__init__` (~line 90):

```python
elif line.startswith("PINATTR"):
    assert pin is not None, "A PIN was already created."
    tag, attribute, value = line.split(' ', maxsplit=3)
```

`maxsplit=3` allows four parts where three are unpacked. The `SYMATTR` branch
above it splits with `maxsplit=2` and does not have this problem.

### Reproduction

`dual.asy` beside the sheet:

```
Version 4
SymbolType CELL
RECTANGLE Normal 0 0 32 32
PIN 0 16 NONE 0
PINATTR PinName OUT A
PINATTR SpiceOrder 1
```

```
Version 4
SHEET 1 880 680
SYMBOL dual 0 0 R0
SYMATTR InstName U1
```

```python
AsyReader("dual.asy")        # ValueError: too many values to unpack (expected 3)
AscEditor("with_dual.asc")   # the same error, from reset_netlist
```

### Impact

- 39 of the 6,678 symbols installed with LTspice 26.1.1 have such a pin name
  (`Comparators/LTC1442`, `FilterProducts/LTC1060` and `LTC1068`, among
  others), and 4 of its 150 largest example sheets place one of them
  (`LTC1439.asc`, `LTC1539.asc`, `ADAQ7768-1.asc`, `ADAQ7769-1.asc`). None of
  those sheets can be opened for editing.
- The error names neither the symbol nor the line, so a caller is told only
  that some unpacking failed.

### Proposed fix

Split with `maxsplit=2`, as the `SYMATTR` branch does, so that the value is
everything after the attribute's name:

```python
tag, attribute, value = line.split(' ', maxsplit=2)
```

### Suggested upstream test

```python
def test_a_pin_name_with_a_space_is_read(tmp_path):
    asy = tmp_path / "dual.asy"
    asy.write_text(
        "Version 4\nSymbolType CELL\nPIN 0 16 NONE 0\n"
        "PINATTR PinName OUT A\nPINATTR SpiceOrder 1\n"
    )
    assert AsyReader(asy).pins[0].text == "PinName=OUT A;SpiceOrder=1;"
```

### Cross-reference

`tests/test_symbol_file.py::TestAPinNameWithASpace` pins two behaviors: this
project's reader takes `OUT A` as the pin's name, and the schematic editor
cannot open a sheet that places the symbol. Once upstream reads such a symbol,
the second test (`test_the_editor_cannot_open_a_sheet_that_places_it`) fails
and can become a check that the sheet opens. If `edit_schematic` read sheets
with our own reader instead of `AscEditor`, this bug would no longer affect it.

---

## Bug 29 — `AscEditor.set_component_parameters` adds a second copy of a parameter given in `Value` or `Value2`

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-16
on a sheet placing LTspice's `Digital/diffschmtbuf` symbol.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.set_component_parameters` and `AscEditor.get_component_parameters`).
**Our workaround:** `lib/schematic_ops.py:_set_instance_params` sets a key in
whichever of `Value`, `Value2`, `SpiceLine` or `SpiceLine2` already contains
it, matching case-insensitively, and appends a key found in none of them to
`SpiceLine` as text. It does not call `set_component_parameters`.

### Summary

`set_component_parameters` looks for an existing `key=value` only inside
`SpiceLine` and `SpiceLine2`, and compares the key's spelling exactly. A part
that gives the parameter in `Value` or `Value2` (an A-device's `Vt=... Vh=...`
in `Value`, a MOSFET's `l=... w=...` in `Value2`) or spells it in another case
(`w=` against `W=`) gets a second copy appended to `SpiceLine`, and the call
gives no warning. On the sheet where this was found, LTspice's export of the
part then listed each threshold twice.

### Affected code

`spicelib/editor/asc_editor.py`, `AscEditor.get_component_parameters`
(~line 464) splits only the `LTSPICE_PARAMETERS_REDUCED` attributes
(`SpiceLine`, `SpiceLine2`) into their parameters, and
`set_component_parameters` (~line 500) looks a key up in those with `in`:

```python
for param_key in LTSPICE_PARAMETERS_REDUCED:
    if param_key in params:
        if key in params[param_key]:
```

A key found nowhere is appended to `SpiceLine`.

### Reproduction

```
Version 4
SHEET 1 880 680
SYMBOL nmos4 100 100 R0
SYMATTR InstName M1
SYMATTR Value NMOS1
SYMATTR Value2 l=1u w=20u
```

```python
editor = AscEditor("mos.asc")
editor.set_component_parameters("M1", w="10u")
editor.get_component("M1").attributes
# {'Value': 'NMOS1', 'Value2': 'l=1u w=20u', 'SpiceLine': 'w=10u'}
```

With `SYMATTR SpiceLine w=20u` in place of `Value2`, `W="10u"` gives
`SpiceLine` `w=20u W=10u`.

Adding a key also rebuilds `SpiceLine` from the parameters spicelib parsed out
of it, converted to numbers. With `SYMATTR SpiceLine m=2 ad=1e3`, `w="10u"`
gives `m=2 ad=1000.0 w=10u`, and `params: r=2k` becomes `r=2k w=10u`.

### Impact

- The part contains both `w=20u` and `w=10u`. Which copy LTspice uses has
  not been recorded, and the edit reported success.
- A comparison that keeps one value per key cannot detect the second copy. A
  netlist equivalence check passed on a sheet whose thresholds were written
  twice.
- Every other parameter on `SpiceLine` is rewritten when one is added, and
  text the parse does not match, such as a `params:` marker, is dropped.

### Proposed fix

Search every attribute that contains parameters (`Value`, `Value2`,
`SpiceLine`, `SpiceLine2`) for the key, case-insensitively, and replace each
copy found in place. Append to `SpiceLine` only when none is found, as text,
leaving the existing text unchanged.

### Suggested upstream test

```python
def test_a_parameter_in_value2_is_set_there(tmp_path):
    asc = tmp_path / "mos.asc"
    asc.write_text(
        "Version 4\nSHEET 1 880 680\nSYMBOL nmos4 100 100 R0\n"
        "SYMATTR InstName M1\nSYMATTR Value NMOS1\nSYMATTR Value2 l=1u w=20u\n"
    )
    editor = AscEditor(asc)
    editor.set_component_parameters("M1", W="10u")
    attributes = editor.get_component("M1").attributes
    assert attributes["Value2"] == "l=1u w=10u"
    assert "SpiceLine" not in attributes
```

### Cross-reference

`tests/test_circuit_asc.py::TestSetComponentValueParameterSlots` pins the
workaround through the `set_component_value` operation of our `edit_schematic`
tool. It checks that a threshold in `Value` and a width in `Value2` are
replaced in place, that a key the part does not give is appended to
`SpiceLine` after its existing text, and that copies already on a part are all
set to the same value, with our sheet check's `duplicate_parameter` finding
naming the part. `tests/test_netlist_graph.py` pins that our netlist
comparison reports a parameter given twice (`duplicate_params`) and treats it
as equal only when it is equal whichever copy is read. Once
upstream searches every slot and appends as text, the workaround can call
`set_component_parameters` again.

## Bug 30 — `AscEditor` takes every instance named `X...` for a subcircuit, whatever its symbol

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-22
on a resistor named `X9`.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist`, the `SYMATTR InstName` branch).
**Our workaround:** the sheet rule `subcircuit_value_slot`
(`lib/sheet_findings.py`) warns when a subcircuit part's `Value` would break
its netlist line (for example an opamp complexity label such as `Level.2`).
It decides whether a part is a subcircuit call by `SheetPart.element`, the
letter LTspice netlists it as (the part's own `Prefix`, else its symbol's,
else its name's first letter), read off the sheet and its symbol, and does not
read spicelib's `_SUBCKT` attribute.

### Summary

When it reads `SYMATTR InstName`, the editor gives the component a `_SUBCKT`
attribute when its name starts with `X` or its symbol is a subcircuit. The
letter that a part netlists as comes from its symbol's `Prefix` or its own,
not from its name. LTspice puts the element letter in front of a name that
does not start with it (`R§Load` for a resistor named `Load`; recorded:
`export/instance_names`, `export/instance_prefix`). A resistor named `X9` is a
resistor, but spicelib marks it as a subcircuit; it does not mark a resistor
named `x9`.

### Affected code

```python
if ref == "InstName":
    component.reference = text
    symbol = self._get_symbol(component.symbol)
    if component.reference.startswith('X') or symbol.is_subcircuit():
        component.attributes['_SUBCKT'] = self._get_subcircuit(symbol)
```

### Reproduction

```
Version 4
SHEET 1 880 680
SYMBOL res 96 144 R0
SYMATTR InstName X9
SYMATTR Value 1k
```

```python
editor = AscEditor("x9.asc")
"_SUBCKT" in editor.get_component("X9").attributes  # True
```

### Impact

Code that asks spicelib whether a part is a subcircuit gets yes for any part
named `X...` and no for one named `x...`, regardless of how LTspice netlists
the part. In this server, `edit_schematic` warned that a resistor named `X9`
with the value `Level.2` was a subcircuit given an opamp's complexity label.

### Proposed fix

Decide by the element letter. A part is a subcircuit when its own `Prefix`
attribute, or the symbol's `Prefix` if it has none, is `X` (case-insensitively),
or when its symbol is a subcircuit symbol. Drop the test on the instance's
name.

### Suggested upstream test

```python
def test_a_resistor_named_x_is_not_a_subcircuit(tmp_path):
    asc = tmp_path / "x9.asc"
    asc.write_text(
        "Version 4\nSHEET 1 880 680\nSYMBOL res 96 144 R0\n"
        "SYMATTR InstName X9\nSYMATTR Value 1k\n"
    )
    assert "_SUBCKT" not in AscEditor(asc).get_component("X9").attributes
```

### Cross-reference

`tests/test_circuit_asc.py::TestInstancePrefix::test_the_level_label_check_reads_the_parts_element_letter`
pins that the check gives no warning for a resistor named `X9` with the value
`Level.2`, and warns for the same part with its own `Prefix X`.

---

## Bug 31 — `AscEditor` keeps one component per reference, and saving deletes the others

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-29.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor._load_asc_file` and `save_netlist`).
**Our workaround:** `lib/schematic_ops.py:_refuse_lost_parts` refuses to open a
sheet when the editor holds fewer components than the file has `SYMBOL`
records, naming each shared reference and where its parts are.

### Summary

The reader stores each component as `self.components[component.reference] =
component`. A second `SYMBOL` with the same `InstName` replaces the first, and
`save_netlist` writes `self.components.values()`, so saving the sheet deletes
every part but the last of each name. Neither LTspice build netlists such a
sheet (XVII stops with "Duplicated Instance Name"; LTspice 26 writes no
netlist), but a sheet can hold one, written by hand or by a tool.

### Affected code

```python
if line.startswith("SYMBOL"):
    if component is not None:
        self.components[component.reference] = component
    component = SchematicComponent(self, line)
```

### Reproduction

```
Version 4
SHEET 1 880 680
SYMBOL res 96 96 R0
SYMATTR InstName R1
SYMATTR Value 1k
SYMBOL res 400 96 R0
SYMATTR InstName R1
SYMATTR Value 2k
```

```python
editor = AscEditor("shared.asc")
len(editor.components)  # 1
editor.save_netlist("shared.asc")  # the 1k resistor is gone
```

### Impact

Any edit saved through spicelib silently deletes a part. In this server,
adding one net label to such a sheet committed a file without the first `R1`.

### Proposed fix

Raise on a second component with a reference already read, naming both
positions, or keep components in a list and index them by reference
separately, so nothing is lost on save.

### Suggested upstream test

```python
def test_a_repeated_reference_is_not_lost(tmp_path):
    asc = tmp_path / "shared.asc"
    asc.write_text(
        "Version 4\nSHEET 1 880 680\n"
        "SYMBOL res 96 96 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\n"
        "SYMBOL res 400 96 R0\nSYMATTR InstName R1\nSYMATTR Value 2k\n"
    )
    editor = AscEditor(asc)
    editor.save_netlist(asc)
    assert asc.read_text().count("SYMBOL res") == 2
```

### Cross-reference

`tests/test_recorded_ltspice_schematics.py::test_an_edit_refuses_a_sheet_whose_parts_share_a_name_and_keeps_both`
pins the workaround against the `connectivity/shared_reference` recording.

---

## Bug 32 — Setting a component's `InstName` attribute writes its old name and its new one

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-29.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/base_editor.py`,
`set_component_attribute`; `spicelib/editor/asc_editor.py`, `save_netlist`).
**Our workaround:** `lib/schematic_ops.py:_rename_component` renames the part
itself. It re-keys `components` under the new name, in place, and sets the
component's `reference`, without an `InstName` attribute.

### Summary

`set_component_attribute(reference, "InstName", new)` stores `new` in the
component's `attributes` and leaves the component keyed, and its `reference`
set, as before. `save_netlist` writes `SYMATTR InstName <reference>` and then
every attribute, so the saved part has two `InstName` lines, the old name
first.

### Affected code

```python
asc.write(f"SYMATTR InstName {component.reference}" + END_LINE_TERM)
...
for attr, value in component.attributes.items():
    if not attr.startswith('_'):
        asc.write(f"SYMATTR {attr} {value}" + END_LINE_TERM)
```

### Reproduction

```python
editor = AscEditor("draft.asc")  # holds R1
editor.set_component_attribute("R1", "InstName", "R9")
editor.save_netlist("draft.asc")
# the file has "SYMATTR InstName R1" followed by "SYMATTR InstName R9"
```

### Impact

spicelib still finds a renamed part under its old name. What LTspice makes
of two `InstName` lines is not recorded here. In this server,
`set_component_attribute` with `InstName` committed such a file and reported
the old name as absent.

### Proposed fix

Treat `InstName` as the reference, so that setting it re-keys `components`,
sets `reference`, and stores nothing in `attributes`.

### Suggested upstream test

```python
def test_setting_instname_renames(tmp_path):
    editor = AscEditor(sheet_with_r1(tmp_path))
    editor.set_component_attribute("R1", "InstName", "R9")
    editor.save_netlist(editor.asc_file_path)
    text = editor.asc_file_path.read_text()
    assert "SYMATTR InstName R9" in text and "SYMATTR InstName R1" not in text
```

### Cross-reference

`tests/test_circuit_asc.py::TestReferences::test_setting_a_parts_name_renames_it`
pins the workaround.

---

## Bug 33 — `AscEditor` reads the sheet beside a block symbol that gives a `Prefix` as its subcircuit

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-29
in recordings of LTspice 26 and LTspice XVII.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor._get_subcircuit`).
**Our workaround:** `_AscEditor._get_subcircuit` in `lib/schematic_ops.py`
returns no subcircuit for a block symbol that gives a `Prefix`, as it does for
one with no sheet.

### Summary

For a `BLOCK` symbol with no library, `_get_subcircuit` opens the sheet of the
symbol's name, beside it or searched for, as the instance's subcircuit.
LTspice does that only for a block symbol that gives no `Prefix`, as the ones
LTspice draws for a sheet do. A block symbol that gives `Prefix X` is a call to
the subcircuit its `Value` names: LTspice exports no `.subckt` from the sheet
beside it, and a run stops on the undefined call (recorded:
`export/hier_top`, `export/hier_top_run`; the same block without the `Prefix`,
`export/hier_plain_top`, exports `.subckt hier_plain` from its sheet on both
builds).

### Affected code

```python
lib = symbol.get_library()
if lib is None and symbol.symbol_type == "BLOCK":
    asc_filename = symbol.get_schematic_file()
    ...
    answer = AscEditor(asc_path)
```

### Reproduction

`amp.asy` beside `top.asc`, and `amp.asc`, a sheet with ports `IN` and `OUT`:

```
Version 4
SymbolType BLOCK
RECTANGLE Normal -48 -32 48 32
SYMATTR Prefix X
PIN -48 0 LEFT 8
PINATTR PinName IN
PINATTR SpiceOrder 1
PIN 48 0 RIGHT 8
PINATTR PinName OUT
PINATTR SpiceOrder 2
```

```python
editor = AscEditor("top.asc")  # top.asc places amp as X1
editor.get_subcircuit("X1")    # an AscEditor of amp.asc; LTspice reads no sheet
```

### Impact

A component inside the sheet (`X1:R1`) can be read and edited through the
parent although LTspice does not netlist that sheet for the instance, so an
edit there changes nothing the simulator runs.

### Proposed fix

Open the sheet only for a `BLOCK` symbol whose attributes give no `Prefix`,
and return no subcircuit for one that gives one, as for a `CELL` symbol with
no library.

### Suggested upstream test

```python
def test_a_block_symbol_that_gives_a_prefix_has_no_sheet(tmp_path):
    (tmp_path / "amp.asy").write_text(AMP_ASY_WITH_PREFIX)
    (tmp_path / "amp.asc").write_text(AMP_SHEET)
    (tmp_path / "top.asc").write_text(TOP_PLACING_AMP_AS_X1)
    assert AscEditor(tmp_path / "top.asc").get_subcircuit("X1") is None
```

### Cross-reference

`tests/test_recorded_ltspice_schematics.py::test_the_editor_opens_a_blocks_sheet_where_ltspice_reads_it`
pins the workaround against both recordings.

---

## Bug 34 — `AscEditor` reads a comment or directive drawn on its side as upright, and saves it so

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-31.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/ltspice_utils.py`,
`TEXT_REGEX`; read by `AscEditor.reset_netlist`).
**Our workaround:** `_AscEditor.reset_netlist` in `lib/schematic_ops.py`
reads each `TEXT` line's justification again after spicelib's reader and
gives it to `asc_text_align_set` whole.

### Summary

A `TEXT` line's justification is `Left`, `Right`, `Center`, `Top`, `Bottom`
or `Invisible`, or one of those after a `V` for text drawn on its side
(`VRight`). `TEXT_REGEX` matches the `V` outside its justification group:

```python
TEXT_REGEX = re.compile(
    r"TEXT (-?\d+)\s+(-?\d+)\s+V?(Left|Right|Top|Bottom|Center|Invisible)\s(\d+)\s*(?P<type>[!;])(?P<text>.*)",
    re.IGNORECASE
)
```

so the justification `asc_text_align_set` receives never starts with `v`,
and it sets every text's `angle` to `ERotation.R0`. `asc_text_align_get`
writes the `V` back only for `ERotation.R90`, so saving the sheet writes
`TEXT 16 16 Right 3 !.op` for `TEXT 16 16 VRight 3 !.op`. LTspice keeps the
`V` when it saves the sheet (recorded on LTspice 26 and XVII:
`save/shuffled`).

### Affected code

```python
# asc_editor.py, AscEditor.reset_netlist
alignment = match.group(TEXT_REGEX_ALIGN)
text = Text(coord=coord, text=text.strip(), size=size, type=ttype)
text = asc_text_align_set(text, alignment)
```

### Reproduction

```python
from pathlib import Path
from spicelib import AscEditor

Path("turned.asc").write_text(
    "Version 4\nSHEET 1 880 680\nTEXT 16 16 VRight 3 !.op\n"
)
editor = AscEditor("turned.asc")
editor.directives[0].angle          # ERotation.R0, though the text is turned
editor.save_netlist("turned.asc")
Path("turned.asc").read_text()      # ... "TEXT 16 16 Right 3 !.op"
```

### Impact

Saving a sheet after any edit turns each of its comments and directives drawn
on its side upright, so a label of a vertical rail or a directive placed
along a part's side is drawn across what is beside it.

### Proposed fix

Capture the `V` with the justification:

```python
r"TEXT (-?\d+)\s+(-?\d+)\s+(V?(?:Left|Right|Top|Bottom|Center|Invisible))\s(\d+)\s*(?P<type>[!;])(?P<text>.*)"
```

`asc_text_align_set` already handles `vleft`, `vright` and the rest.

### Suggested upstream test

```python
def test_text_on_its_side_is_saved_on_its_side(tmp_path):
    sheet = tmp_path / "turned.asc"
    sheet.write_text("Version 4\nSHEET 1 880 680\nTEXT 16 16 VRight 3 !.op\n")
    editor = AscEditor(sheet)
    assert editor.directives[0].angle == ERotation.R90
    editor.save_netlist(sheet)
    assert "TEXT 16 16 VRight 3 !.op" in sheet.read_text().splitlines()
```

### Cross-reference

`tests/test_recorded_ltspice_sheet_save.py::TestWhatASaveChanges::test_text_on_its_side_stays_on_its_side`
pins the workaround against both recordings, and
`tests/test_sheet_findings_snapshot.py::test_an_edit_and_a_check_find_the_same_of_a_sheet`
through an edit of the recorded sheet.
