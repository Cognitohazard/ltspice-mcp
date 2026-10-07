# spicelib bug reports

Upstream [spicelib](https://github.com/nunobrum/spicelib) bugs this project has
hit. Each section is a self-contained draft for an upstream pull request, or a
known limitation we live with.

spicelib is a pinned dependency we cannot fix in place, so we work around its
bugs and delete the workaround once upstream is fixed. This file is the record
of what to delete, and each reproduction is what lets someone confirm a fix.

If you hit a new one, add a section in this format: summary, affected code and
version, reproduction, impact, proposed fix, a suggested upstream test, and a
cross-reference to our workaround and the test that pins it. Do that even when
you also ship a workaround.

---

## Bug 1 — `.MEAS` AT/WHEN crossing time dropped on whitespace-padded lines

**Status:** draft for an upstream spicelib pull request. Filed 2026-06-28.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py`). Present in the
single-line measurement regex unchanged for several releases.
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — `_RE_MEAS_AT_LINE`
plus the `at_overrides` backfill in `parse_measurements`. Remove it once
upstream is fixed. (The backfill only fills `at` when spicelib left it unset,
so it is inert against a fixed spicelib — but it is dead weight.)

### Summary

`LTSpiceLogReader` parses the single-line (stepless) `.MEAS` result format with
a regex whose `AT` clause and `FROM`/`TO` clause are written with **literal
single spaces**. When LTspice pads the result line with more than one space (or
a tab) before `AT`/`FROM`, the optional clause fails to match. The measurement
is still returned, but **the crossing time (`AT`) and window bounds
(`FROM`/`TO`) are silently dropped** — the value comes back as the trigger
level alone, with no `_at` / `_FROM` / `_TO` companion column.

For a transient WHEN measurement — the standard idiom for lock time, startup
time, propagation delay, settling time, threshold-crossing time — this means
the **answer the user actually wants, the time, is lost**, and what remains is
the trigger level they typed (e.g. `0.5`), which looks plausible and is
reported with no error.

### Affected code

`spicelib/log/ltsteps.py`, the stepless measurement regex (line ~312–315):

```python
regx = re.compile(
        r"^(?P<name>\w+)(:\s+.*)?=(?P<value>[\d(inf)E+\-\(\)dB,°(-/\w]+)( FROM (?P<from>[\d\.E+-]*) TO (?P<to>[\d\.E+-]*)|( at (?P<at>[\d\.E+-]*)))?",
        re.IGNORECASE)
```

The clause literals are `" FROM "`, `" TO "` and `" at "`, each with exactly one
leading/trailing space. `re.IGNORECASE` is set, so **case is not the problem**
(`AT` matches `at` fine). The problem is purely the **fixed single spaces**: the
`(?P<value>…)` character class has no whitespace, so the value token ends at the
first space; the optional clause must then match starting at the *next*
character, which it expects to be a single space followed by `at`. A second
space (or a tab) there breaks the match, and the whole optional group —
including the `at` capture — is skipped, because it is `?`-optional.

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

So whether the crossing time survives depends on LTspice's incidental spacing
of the result line — fragile, and silent when it fails.

### Why LTspice produces the extra space

LTspice right-pads and aligns some result fields, and the exact spacing varies
with the value's sign and width, so the same `.meas` form can emit one space on
one run and two on another. A recorded single-space sample lives at
`tests/fixtures/ltspice_sweep_meas_run0.log`; the double-space variant is what
triggers the loss.

### Impact

- Any transient `.MEAS … WHEN`/`AT` result whose line happens to be
  whitespace-padded loses its crossing time. This is the canonical idiom for
  lock, settling, delay and startup times.
- `FROM`/`TO` windowed measurements (`RMS(...) FROM a TO b`) lose their bounds
  the same way if padded.
- The failure is silent: the measurement still appears, with a
  plausible-looking value (the level), no error and no warning.

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
  class. Out of scope for this bug, but worth a look.

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
will keep passing against a fixed spicelib; once upstream lands, delete
`_RE_MEAS_AT_LINE` and the `at_overrides` block in `parse_measurements`, and
re-point those tests at the (then-correct) spicelib path.

---

## Bug 2 — `_get_text_space` places directives off-canvas on a low-content schematic

**Status:** draft for an upstream spicelib pull request. Filed 2026-07-04.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor._get_text_space`, ~line 590).
**Our workaround:** none — accepted as a known limitation. Our own directive
placement (`_append_asc_text` in `src/ltspice_mcp/tools/circuit.py`, used by the
`add_directive` schematic-edit op and by coordinate-bearing directive edits)
bypasses `_get_text_space` entirely and places at a fixed `(16, 16)` with
auto-declutter, so it is unaffected. Only the no-coordinate directive path
routes through `add_instruction` and therefore `_get_text_space`. We keep that
path on `add_instruction` deliberately, because it also performs spicelib's
unique-analysis-directive replacement (adding `.ac` after `.tran` replaces it),
which `_append_asc_text` does not — so we tolerate the off-canvas placement on
the uncommon empty-schematic case rather than lose the replacement semantics.

### Summary

`AscEditor.add_instruction` places a new directive or comment at the coordinate
returned by `_get_text_space()`. On a schematic with **no wires, labels,
directives or components**, that coordinate is far **off-canvas** — e.g.
`(880, -99976)` on the default 880 x 680 sheet — so the added text is invisible
in the LTspice view. (The netlist still exports correctly; position is
cosmetic.)

The cause: `_get_text_space` seeds its bounding box's **minimums** from the
sheet but never its **maximums**. `max_x` and `max_y` are initialized to the
sentinel `-100000` and are only updated by iterating over existing wires,
labels, directives and components. With no content to update them, `max_y`
stays `-100000`, and the method returns `(min_x, max_y + 24)` = `(880, -99976)`.

A second issue rides along: `min_x` is seeded from the sheet **width** (`880`),
not the left edge (`0`), so even the x is the far-right edge rather than the
"bottom left corner" the code comment promises.

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

Both coordinates sit about 100000 units above the 0..680 sheet, so the
directive renders far outside the visible schematic.

### Impact

- Adding a directive or comment to a freshly created (empty) `.asc` via
  `add_instruction` places it off-canvas. Subsequent adds stagger downward
  (`max_y + 24` each time) but remain off-canvas until enough on-canvas content
  exists to dominate `max_y`.
- Silent: the `.asc` and netlist are valid, so nothing errors — the text is
  just not where a human can see it.
- The `min_x = sheet width` seeding also biases placement to the right edge even
  once content exists, contradicting the "bottom left corner" intent.

### Proposed fix

Seed the bounding box from the sheet **rectangle** (both corners), so an empty
schematic degrades to a sane on-canvas bottom-left spot instead of the
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
pins that repeated no-coordinate directive adds stagger to distinct anchors
rather than stacking. It does not assert on-canvas placement, since that
depends on this upstream bug. If spicelib is fixed, that path's placement
becomes on-canvas automatically with no downstream change needed.

---

## Bug 3 — multi-plot ASCII raw hangs the parser in a CPU-bound infinite loop

**Status:** draft for an upstream spicelib pull request. Filed 2026-07-13.
**Affected version:** spicelib 1.5.1 (`spicelib/raw/plot_data.py`,
`PlotData._read_ascii_vector`, the trailing-empty-line skip at ~line 401).
**Our workaround:** `src/ltspice_mcp/lib/raw_parser.py` —
`_MultiPlotAsciiGuard`, installed over `_read_ascii_vector` by
`_install_multiplot_ascii_guard()` at import. It wraps the file object handed to
the ASCII reader and returns a one-shot empty read when the skip loop seeks back
onto a line it just read as non-empty, breaking the loop with the cursor left on
the next plot's header. It also adds a forward-progress backstop
(`_STALL_LIMIT`) that aborts any read run which stops advancing the high-water
byte offset. It keys on the read/seek pattern rather than on spicelib
internals, so it is version-independent and inert once upstream is fixed.

### Summary

`PlotData._read_ascii_vector` ends each ASCII plot by skipping trailing blank
lines. The loop reads a line; if it is **non-empty** it seeks back to the line's
start (to leave it for the caller) — but then loops and **re-reads the same
line**, because it only `break`s on an **empty** line. When an ASCII raw
contains a **second plot with no blank-line separator** before it, the first
non-empty line the loop meets is the next plot's `Title:` header, so it seeks
back and re-reads that header forever: a CPU-bound infinite loop.

ngspice writes a `.noise` result exactly this way — two plots (noise spectral
density, then integrated noise) back to back with no blank line — so a single
successful `.noise` run wedges any consumer that parses the raw. Because a
CPU-bound loop holds the GIL, offloading the parse to a thread does not help;
it starves the whole process.

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

An ASCII raw with two plots and no blank line between them (the ngspice
`.noise` shape): the second plot's `Title:`/`Plotname:` header directly follows
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
  parser. ngspice `.noise` is the canonical producer, and the failure is a full
  hang rather than an error, so a caller parsing synchronously wedges until
  killed.
- Threading the parse off the main loop does not mitigate it — the loop is
  CPU-bound and holds the GIL.

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
`test_stall_backstop_allows_a_large_forward_read` for the forward-progress
backstop. Once upstream lands, delete `_MultiPlotAsciiGuard` and
`_install_multiplot_ascii_guard()` and re-point those tests at the fixed reader.

---

## Bug 4 — windowed-transient axis `Offset:` parsed but never applied

**Status:** draft for an upstream spicelib pull request. A known limitation we
work around.
**Affected version:** spicelib 1.5.1 (`spicelib/raw/raw_read.py`
`RawRead.get_axis` ~line 545, into `spicelib/raw/plot_data.py`
`PlotData.get_axis` ~line 779).
**Our workaround:** `src/ltspice_mcp/lib/raw_parser.py` — `OffsetAwareRawRead`,
a `RawRead` subclass that reads `raw_params["Offset"]` in `__init__` and adds it
to the axis in `get_axis`, but only for transient plots with a nonzero offset.
Server-side raw loads construct this subclass, not a bare `RawRead`.

### Summary

For a **windowed** transient run — `.tran 0 <tstop> <tstart>` with a nonzero
`<tstart>` — LTspice writes the raw with the **time axis rebased to 0** and the
true start time recorded in the header's `Offset:` field. spicelib parses the
header (including `Offset:`) into `raw_params`, but `get_axis()` returns the
stored zero-based axis **without adding the offset back**. Every axis consumer
therefore works in the rebased frame: a `.tran 0 202u 196u` run — which the user
asked to observe over 196–202 us — reads back as **0–6 us**.

The failure is silent: the data is correct in shape, only the absolute time
coordinates are wrong, so anything computed against absolute time (a
`.meas ... AT`, a windowed signal-statistics start or end time, an exported
waveform's time column) is off by `<tstart>` with no error.

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
- Silent: the shape is right, only the time coordinates are wrong.
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

Downstream: `tests/test_raw_parser.py::TestOffsetAwareRawRead` pins that
`OffsetAwareRawRead` rebases a nonzero-offset transient axis to deck time and
leaves zero-offset and non-transient plots untouched. Once upstream lands,
delete `OffsetAwareRawRead` and construct a bare `RawRead` at the server-side
load sites.

---

## Bug 5 — Fourier/THD parse crashes on a `-nan`/`inf` distortion line

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py` ~lines 354/357).
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — `_RE_FOURIER_NAN` plus
`_sanitize_log_for_reader` (called by `make_log_reader`) rewrites `-nan%`/`inf%`
THD/PHD lines to `0.0%` in a temporary copy before handing the log to
`LTSpiceLogReader`, then parses that.

### Summary

When LTspice runs a `.four`/THD analysis on an identically zero or
non-oscillating signal, it writes `Total Harmonic Distortion: -nan%` (or
`inf%`). spicelib parses that line with
`float(re.search(r"\d+.\d+", line).group())` — the regex needs a
digit-dot-digit run, which `-nan%` and `inf%` lack, so `re.search` returns
`None` and `None.group()` raises `AttributeError`, **aborting the entire `.log`
parse** — not just the Fourier block, but every `.MEAS` result and error line in
the same log.

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

- A single `-nan`/`inf` THD line — routine for a zero or rail-pinned signal —
  crashes the whole log read, so all `.MEAS` scalars and diagnostics from that
  run are lost. An uncaught exception, not a skipped block.

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
upstream lands, drop `_RE_FOURIER_NAN` and `_sanitize_log_for_reader`.

---

## Bug 6 — empty `SYMATTR` value makes an `.asc` unreadable (`ValueError` on load)

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py` ~line 186).
**Our workaround:** `src/ltspice_mcp/tools/circuit.py` —
`_reject_empty_attr_value` and `_require_clearable_attr` refuse to *write* an
empty SYMATTR value (an empty value removes the attribute line instead), so we
never emit a two-token line spicelib cannot re-read.

### Summary

`AscEditor` reads each `SYMATTR` line with
`tag, ref, text = line.split(maxsplit=2)`, which assumes three tokens. A valid
LTspice-authored empty attribute is a **two-token** line — `SYMATTR Value` with
no value — so the unpack raises
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
  opened, read, edited or netlisted through spicelib — a hard failure at load,
  before any operation.

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

Downstream: `tests/test_circuit_asc.py::...::test_empty_attribute_raises` and
`tests/test_edge_cases.py::...::test_empty_attribute_rejected`. Our guard is
write-side; a fixed spicelib would also let us READ such files, which we
currently cannot.

---

## Bug 7 — failed `.MEAS` names dropped from the reader (silent absence)

**Status:** a known limitation we work around.
**Affected version:** spicelib 1.5.1 (`spicelib/log/ltsteps.py`,
`LTSpiceLogReader`).
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — `_RE_MEAS_FAILED`
re-extracts failed measurement names from the raw log text and emits them with
`value=None`, so a requested-but-failed measure is visible rather than absent.

### Summary

When a `.MEAS` fails, LTspice writes `Measurement "name" FAILed` to the log and
no result row. `LTSpiceLogReader` parses only successful result rows, so
`get_measure_names()` **omits the failed measure entirely** — a measure that was
requested and failed is indistinguishable from one that was never requested. For
a consumer asking "did my measurement pass?", the answer silently disappears.

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

- A failed `.MEAS` is a silent absence. Downstream code cannot tell "failed"
  from "not requested", and may report success by omission.

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

Downstream: the `failed_measurements` list and the `value=None` entries in
`parse_measurements` (`lib/log_parser.py`). Once upstream surfaces failed names,
drop `_RE_MEAS_FAILED`.

---

## Bug 8 — cp1252 operating-point log misdetected as utf-16 and silently garbled

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/utils/detect_encoding.py`,
`detect_encoding` called with no `expected_pattern`).
**Our workaround:** `src/ltspice_mcp/lib/log_parser.py` — decode the log with
our own BOM/UTF-16/cp1252 detector (`read_spice_text`), rewrite it to a UTF-8
temporary file, and hand *that* to `opLogReader` so its detection cannot
misfire.

### Summary

`detect_encoding()`, used by the device operating-point reader
`semi_dev_op_reader.opLogReader`, tries encodings in the order
`utf-8, utf-16, utf_16_le, windows-1252, cp1252, …` and returns the first that
decodes without raising. A cp1252 log carrying a high byte (a degree sign, micro
sign or plus-minus sign — routine in a `.options logopinfo` operating-point
dump) fails UTF-8, is then tried as **utf-16**, which decodes
ASCII-with-a-high-byte to garbage **without raising**, so utf-16 is returned and
the block is garbled. utf-16 sits **before** cp1252 in the try order, and the
only utf-16 guard (`lines[1] == '\x00'`) is gated on `encoding == 'utf-8'` — a
branch already skipped once UTF-8 raised — so the misdetection is unguarded.
The operating-point reader then finds no device parameters and wrongly reports
that there are no small-signal device parameters.

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
  No error — just wrong or empty data.

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

Downstream: `tests/test_log_parser.py::...::test_cp1252_degree_byte_in_step_value_recovered`
plus the cp1252 cases in `tests/test_encoding.py`. Once upstream fixes the
order, drop the pre-normalize-to-UTF-8 temporary-file dance around
`opLogReader`.

---

## Bug 9 — `remove_instruction` deletes the wrong directive by substring match

**Status:** draft for an upstream spicelib pull request.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`
`AscEditor.remove_instruction`, ~line 698).
**Our workaround:** `src/ltspice_mcp/tools/circuit.py` — on an `AscEditor`, match
the typed `directives` list by **exact full-text equality** and delete exactly
one, instead of routing through `remove_instruction`.

### Summary

`AscEditor.remove_instruction` finds the directive to delete with
`if instruction in self.directives[i].text` — a **substring** test — and removes
the first hit. So `remove_instruction(".tran 1")` can delete `.tran 10m`, and
removing one of several similar directives deletes whichever comes first rather
than the one intended. A silent wrong deletion.

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

- Removing a directive can silently delete a different directive that merely
  contains the target as a substring, or the wrong one of several similar
  directives.

### Proposed fix

Match on exact (whitespace-normalized) equality, not substring:

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
drop our exact-equality override.

---

## Bug 10 — `SimRunner.__del__` blocks for the whole simulation, so a dropped reference freezes the destroying thread

**Status:** draft for an upstream spicelib pull request. Filed 2026-08-07.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/sim_runner.py:333`
`__del__`, `:711` `wait_completion`). Present unchanged across the pinned range
(`>=1.4.9,<1.6`).
**Our workaround:** `src/ltspice_mcp/lib/runner_base.py` —
`_NonBlockingSimRunner` overrides `__del__` to do nothing, and
`RunnerBase._build_sim_runner` constructs that subclass instead of `SimRunner`.
Nothing in this project needs the destructor: completion arrives through the run
callback, liveness is read off the task threads, and the only other thing it
does is the name-global `kill_all_spice()` this project deliberately never uses.
`submit_netlist` still retains every runner in `self._inflight_runners` (the
kill and liveness paths need the handle) and releases it in
`_retire_finished_runners`. Drop the subclass once upstream's destructor no
longer waits.

### Summary

`SimRunner.__del__` calls `self.wait_completion(abort_all_on_timeout=True)`, and
`wait_completion` loops `while len(self.active_tasks) > 0: sleep(1)`. A
destructor therefore blocks for as long as the simulation runs, up to the
instance timeout.

**And in one shape the wait has no bound at all.** `wait_completion(timeout=None)`
recomputes its deadline every second from `_maximum_stop_time()`, which reads
`task.start_time + timeout` and **skips any task whose `start_time` is None**;
`update_completed()` likewise retires a task only `if not (is_alive() or
start_time is None)`. A task that was appended to `active_tasks` and never
started — `run()` does `active_tasks.append(t)` and only then `t.start()` — is
therefore retired by nothing and bounds nothing: the loop condition stays true,
the deadline stays `None`, and the destructor spins on `sleep(1)` forever. The
same predicate makes `RunTask.wait_results()` unbounded
(`while self.is_alive() or self.start_time is None or self.retcode == -1`).

Because the last reference is usually dropped by ordinary garbage collection,
that infinite wait lands on whichever thread happened to allocate. On
2026-09-07 a release-gate CI job lost twelve minutes and forty-eight seconds
of complete silence to it and was killed by the job timeout, reporting only
"The operation was canceled" — no traceback, no failing test, and no simulator
process left behind, because the simulation had finished and only the waiting
thread was stuck.

Under CPython's refcounting this fires at the most surprising possible moment:
the statement that submits. `run()` launches the simulation on a run-task thread
and returns the `SimRunner`; a caller that submits **for the side effect** and
does not bind the result drops the last reference right there, and the
submitting thread disappears into the destructor until the simulation finishes.
Nothing in the API says the return value is load-bearing — it is documented "For
internal use only" — so the natural way to call `run()` is the one that hangs.

The failure has no diagnostic surface. There is no exception, no log line, and
the simulation itself succeeds; only the *caller* is frozen. A stack dump shows
`__del__ -> wait_completion`, which reads like a shutdown path rather than a
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

Severe and silent for any caller that submits from a worker thread and expects
to keep working — which is the shape every async wrapper has. In this project
the coordinator submits a case via `asyncio.to_thread` and then resumes to mark
the case running and start watching for cancellation. Because the worker thread
was pinned in the destructor, the coroutine never resumed: the case never
reached the watcher, so **a cancellation could never reach the simulator**. The
run continued to completion, the job accounting recorded a case that had started
as never submitted, and the completed result was left on disk with no record
claiming it. The user-visible behavior was "cancel does nothing", and the cause
was a destructor.

A second consequence: because `abort_all_on_timeout=True`, a garbage collection
at an unlucky moment can *abort* simulations belonging to a runner the caller
merely stopped referencing.

### Proposed fix

A destructor must not block. Two options, in preference order:

0. **Give the wait a floor that always advances.** Whatever else changes,
   `_maximum_stop_time()` and `update_completed()` should treat a task that is
   not alive and has no `start_time` as finished (it can never start), or
   `wait_completion` should fall back to the instance `timeout` when no task
   offers a deadline. As written the loop has a reachable state with no exit.

1. **Drop the wait from `__del__`.** Leave `wait_completion()` as the explicit
   call it already is, and let `close()` or context-manager use handle teardown.
   Callers who want the wait ask for it; callers who do not are not punished for
   letting an object go out of scope.
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

**Status:** **Not a bug, and not spicelib's.** Documented LTspice exporter
behavior, recorded here so the signature is not re-investigated the next time it
surfaces in an exported netlist. No upstream pull request applies.
**Affected version:** observed on LTspice 26.0.2 for Windows (via WSL interop).
spicelib is uninvolved — see "Why this is not a spicelib bug" below.
**Our handling:** `src/ltspice_mcp/lib/netlist_graph.py` already strips the
marker (`_LTSPICE_INSTANCE_MARKER`, applied in `_strip_marker`) and the
foundation lexer discards the trailing `;` comment. Nothing to remove later.

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

It looks like a placeholder leak or an encoding fault. It is neither: it is
LTspice's normal output for subcircuit instances.

### Reproduction

Minimal, with **no spicelib in the loop** — LTspice.exe invoked directly.

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

### Scope — what actually triggers it

Determined by controls, not assumed:

| Symbol | `SYMATTR Prefix` | Exported line | Marker? |
|-|-|-|-|
| project-local generated `attn.asy` | `X` | `X§U1 NC_01 NC_02 NC_03 attn ;§pnba a)y)g` | yes |
| **stock** `bk_inv.asy` (ships with LTspice) | `X` | `X§U1 … bk_inv Wp=4u Wn=2u ;§pnba A)Y)VDD)VSS` | yes |
| stock `res.asy` | `R` | `R1 NC_01 NC_02 1k` | no |

So the trigger is **`Prefix X` (subcircuit) instances, stock or custom alike**.
It is *not* specific to generated or project-local symbols; a report framing it
that way is a red herring. Non-X devices are never marked. Note that the
annotation comment follows any device parameters.

### Impact

None on this project. The `§` is stripped from the reference and the `;` comment
is discarded as a comment, so the card parses exactly as intended — verified on
both reproductions above:

```
min.net    -> ref='XU1' type='X' nodes=('nc_01','nc_02','nc_03') model='attn'  params=()
stock.net  -> ref='XU1' type='X' nodes=(...x4)                   model='bk_inv' params=(('Wp','4u'),('Wn','2u'))
```

The node count is right (the comment is not absorbed as a fourth or fifth node)
and the parameters are right (the comment is not absorbed as a param). The risk
this entry guards against is a *future* parser that splits the card before
stripping comments, or that matches references literally instead of through the
reference canonicalizer.

### Why this is not a spicelib bug

`spicelib.simulators.ltspice_simulator.LTspice.create_netlist` does not
synthesize a netlist; it shells out and lets LTspice write the file:

```python
cmd_netlist = cls.spice_exe + ['-netlist'] + [circuit_file.as_posix()] + cmd_line_switches
...
netlist = circuit_file.with_suffix('.net')
```

The reproduction above bypasses spicelib entirely and produces byte-identical
output, so spicelib neither introduces nor can remove the marker.

### No workaround

Deliberately none. The marker is stripped where references are canonicalized and
the annotation is a SPICE comment; adding a scrub pass would be dead code that
hides a real change in LTspice's output format.

### Cross-reference

Pinned so the tolerance is deliberate rather than accidental:
`tests/test_netlist_graph.py::test_undefined_pdk_subckt_stays_a_black_box_leaf`
feeds the exact exported signature (both the `X§` reference and the `;§pnba`
comment, with device parameters) through `parse_netlist_graph` and asserts the
reference, node count, model and parameters.


## Bug 11 — `RawRead.get_axis()` returns an unsized 0-d array for a file with no plots, so `len()` raises `TypeError`

### Summary

For a `.raw` file that parsed but holds no plot data, `RawRead.get_axis()`
(and by the same route `get_time_axis()` / `get_wave()`) returns
`numpy.array([])` built as a 0-dimensional array rather than an empty
1-d array. `len(axis)` then raises `TypeError: len() of unsized object`
instead of answering `0`, and any caller that sizes the axis before
reading it has to special-case a Python exception that says nothing about
the file.

### Affected code

spicelib 1.5.1, `spicelib/raw/raw_read.py`, the empty-data branch of
`get_axis()` / `get_trace()`.

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
with a `TypeError` that reads like a bug in the caller. Our summary builder
used to catch `Exception` around it, which hid the file's real condition;
narrowing the catch exposed this.

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

Workaround: `src/ltspice_mcp/lib/raw_parser.py` catches `TypeError`
alongside `RuntimeError` at the axis read in `build_simulation_summary`
(the comment there names this bug). Pinned by
`tests/test_log_parser.py::test_missing_log_with_invalid_raw`. Delete the
`TypeError` arm once upstream returns a 1-d empty array.

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

The output contains both `FLAG` lines and neither `IOPIN`. Reproduced through
our public `edit_schematic` handler by adding `.param marker=1` with a matching
revision hash: it reported a complete, committed edit and the port count fell
from two to zero. The native export acceptance uses a parent/child pair to
check the interface rather than treating retained labels as retained ports.

### Impact, proposed upstream fix and test

An ordinary edit to a reusable child sheet destroys its port declarations.
Emit each port's `IOPIN` immediately after its associated `FLAG`, preserving
coordinates, direction and order. Association must use the label object,
not its text or coordinates, since distinct labels may have identical names.
Refuse an orphaned or ambiguous association instead of attaching it elsewhere.

An upstream test should round-trip multiple ports, including repeated label
names, then move/rename a label object and verify that its port follows it.
Also cover a removed label still referenced by a port.

### Workaround and regression

`tools/schematic_edit.py::_render_editor_text` restores port records alongside
the corresponding emitted flags and checks their association before staging.
`tests/test_edit_schematic.py::TestHierarchicalPortPreservation` exercises the
public edit, refused label removal, reordered/changed label facts and ambiguous
associations. Remove this workaround when the pinned dependency preserves the
same cases itself.

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
`edit_schematic(dry_run=true)` renders its candidate sheet before the dry-run
return. A parent-only transaction cannot safely commit those child changes.

### Proposed upstream fix and test

Separate rendering one sheet from saving its dependency tree. A `StringIO`
render should neither write child files nor clear their pending-update state.
Any recursive save should be explicit and expose its destination/write set.
An upstream test should modify a loaded child, render its parent to `StringIO`,
and assert unchanged child bytes and retained pending changes. Include a
portless parent and a deeper loaded descendant.

### Workaround and regression

`tools/schematic_edit.py::_refuse_pending_child_edits` checks loaded descendants
before invoking the dependency serializer, including portless sheets. It refuses
pending child updates without modifying editor flags or child files.
`TestHierarchicalPortPreservation.test_pending_child_changes_never_write_through_parent`
covers the public normal-edit and dry-run paths with real loaded child editors.
The root's existing revision guard and atomic commit remain the write boundary.

## Bug 14 — `SimRunner.run` cannot run without a timeout, though `None` is documented as "no timeout"

**Status:** draft for an upstream spicelib pull request. Filed 2026-09-29.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/sim_runner.py`, `SimRunner.run`,
the resource-wait loop after `_prepare_sim`). 1.4.9 has the same loop.
**Our workaround:** `src/ltspice_mcp/lib/runner_base.py` —
`RunnerBase._build_sim_runner` never hands spicelib `None`: an unbounded request
becomes `SUBPROCESS_TIMEOUT_CEILING_S` (4,000,000 s), and any larger value is
clamped to it. The ceiling also keeps the bound inside what Windows can wait for
(below). Remove the `None` half of the substitution once upstream runs with
`None`; the clamp stays for as long as Windows waits in 32-bit milliseconds.

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
`TypeError` and no simulation starts. The one value documented as "no timeout"
is the one value that cannot run.

A second consequence of the shared value: the only way to give the simulator
process a long bound is to give the slot wait the same one. That is harmless for
us (one fresh runner per submission, so the wait never waits), but it means the
process bound cannot be set on its own.

### Affected code

`spicelib/sim/sim_runner.py`, `SimRunner.run`: the `while clock() - t0 < timeout + 1`
loop; `run_now` has the same `t.join(timeout + 1)`. `RunTask.run` passes the same
`timeout` to `Simulator.run`, which hands it to `subprocess.run`.

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

A caller that wants the simulator bounded only by its own logic cannot say so.
Before this was found we passed a fixed 600 s instead, and that fixed value
became a hidden ceiling on every run: a case given a longer `run_timeout_s` was
killed at 600 s by `subprocess.run`, and the kill surfaced as spicelib's generic
exit code -2 rather than as the timeout it was.

The obvious substitute for `None`, a very large number, is not safe on Windows.
There `subprocess` converts the timeout to integer milliseconds and passes it
to `WaitForSingleObject`, whose timeout is a 32-bit DWORD, so nothing past
about 49.7 days can be expressed. This was read from CPython's source, not run
on Windows. Any substitute has to stay below 4,294,967 s.

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
from the file length. The experiment coordinator calls it for every case it
stops, before the partial raw is deleted, and records the result as a
`partial_progress` observation. It reads only how far the run got, never the
values, so no workaround is needed for the waveform parse itself.

### Summary

Both simulators write the raw as they solve and fill in `No. Points` only when a
plot ends, and they differ in what a killed run leaves in it:

- **ngspice** writes `No. Points: 0` padded with spaces, to patch in place when
  the plot ends. spicelib raises `SpiceReadException` ("No points or variables
  found") on the count of 0, so the file cannot be opened at all.
- **LTspice** rewrites the count now and then while it runs, so a killed run's
  header holds a count that lags the data. spicelib reads exactly that many
  points and reports success, silently dropping every complete record after it.

Separately, `RunTask.run` hands the completion callback `raw_file=None` for any
nonzero exit. A killed run always exits nonzero, so the callback is never told a
partial raw exists, even when the simulator wrote gigabytes of one; a caller
has to reconstruct the path (`netlist.with_suffix(raw_extension)`).

### Reproduction

Observed 2026-09-29 on Linux with ngspice 42 and LTspice 26.1.1 under Wine,
each killed with SIGKILL part way through `.tran 10n 100m 0 10n` (ngspice) or
`.tran 0 100 0 10n` (LTspice) of an RC driven by a 1 kHz sine:

- ngspice, 2 s in: 32,850,137 bytes, header `No. Points: 0       `, 1,026,560
  complete 32-byte records (time plus three doubles). `RawRead(path,
  dialect="ngspice")` raises `SpiceReadException`.
- LTspice, 8 s in: 51,020,708 bytes, header `No. Points:      1821697`,
  1,822,135 complete 28-byte records (a double time plus five floats) and a torn
  one. Every record past the header count holds a valid, rising time.
  `RawRead(path)` returns 1,821,697 points with no warning.

Neither simulator records progress anywhere else a killed run keeps. LTspice's
log holds only its preamble until the run ends. ngspice prints `Reference
value : <time>` progress to stdout only when it has no `-o` log file, and
spicelib always passes `-o`, so neither the `.log` nor the `.exe.log` carries it.

### Impact

A run stopped for a timeout, a deadline or a cancel can have done almost all of
its work, and nothing says how much. The ngspice file is unreadable. The
LTspice one reads as a complete, shorter run, which is the more dangerous of
the two: a caller who reaches it through spicelib gets a waveform that ends
early and no sign that it did.

### Proposed fix

An opt-in partial read on `RawRead` (for example `allow_partial=True`) that, for
a binary plot, takes the point count as `min(declared, complete records on
disk)` when the declared count is nonzero and the file is short, as the complete
records on disk when it is 0, and as the complete records on disk when the file
holds more than the declared count and no following plot starts at the declared
end. The read should report that it was partial. Separately, `RunTask` could
pass the expected raw path to the callback on a nonzero exit when the file
exists, leaving the decision to use it to the caller.

### Suggested upstream test

Truncate a finished binary raw from each dialect to a whole number of records
plus a torn one; set the ngspice copy's count to 0 and the LTspice copy's count
below the records present; assert that the partial read returns exactly the
complete records and marks itself partial, and that the default read of the
ngspice copy still raises.

### Cross-reference

Workaround: `raw_parser.py` `read_partial_raw_progress`, used by
`experiment_runner.py` `_note_partial_progress`. Pinned by
`tests/test_raw_parser.py::TestPartialRawProgress` (the lagging LTspice count in
`test_a_declared_count_behind_the_records_does_not_cap_them`) and, against a
real killed ngspice run,
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
sheets carry upper-case references, so a lower-case filter silently returns
nothing.

### Reproduction

Load any sheet holding `SYMATTR InstName R1` with `AscEditor` and call
`editor.get_components("r")`: the result is `[]`. `get_components("R")`
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

`tools/inspect_tools.py::_do_components` no longer passes a prefix to
spicelib. It lists every reference and keeps those `montecarlo.matches_prefix`
says the prefix claims, a case-insensitive start-of-reference match, the same
rule its netlist branch and the `hierarchy` query use (`_check_prefix` returns
the upper-cased prefix all three compare against). Pinned by
`tests/test_inspect_tools.py::test_components_prefix_filter_asc_ignores_case`
and `test_components_prefix_is_a_case_insensitive_reference_prefix`. Nothing
here depends on spicelib's filter, so there is no workaround to remove when
upstream changes.

## Bug 17 — the symbol cache is keyed by file name, so one folder's symbol stands in for another's

**Status:** draft for an upstream spicelib pull request. A known limitation; no
workaround ships.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.symbol_cache` and `_asy_file_find`).
**Our workaround:** none for spicelib's own loading. Our pin geometry does not
read this cache: `lib/symbol_geometry.py` `get_symbol_info(symbol, asc_path)`
looks for a symbol beside the sheet itself and caches it by file.

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
`a`, 2k in `b`. A `res.asy` sits on the custom library path.

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
  sheet's same-named block through an earlier sheet's files: the wrong
  sub-sheet, the wrong component values, and the wrong `SymbolType`/`Prefix`
  when spicelib decides whether an instance is a subcircuit.
- Saving the later sheet writes its sub-sheet changes into the earlier
  folder's file, because `save_netlist` saves each updated child to its own
  `asc_file_path` (see Bug 13).
- Which file wins depends on which sheet a long-lived process loaded first,
  so the same request can read differently in two sessions.

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
The write in the reproduction is not reachable through `edit_schematic`, which
refuses pending child edits before saving (Bug 13's workaround); the wrong read
is. The test suite warms this cache for the fixture library once per session
(`tests/conftest.py::_asc_symbol_cache`), and tests that load sheet-local
symbols swap in a copy of it so no later test inherits their folders.

---

## Bug 18 — `Simulator.create_from` rebinds the class it is called on instead of creating one

**Status:** draft for an upstream spicelib pull request. Recorded 2026-10-03.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`Simulator.create_from`).
**Our workaround:** `src/ltspice_mcp/lib/simulator.py` —
`bind_named_executable` creates a subclass of the family's class for each named
executable and calls `create_from` on the subclass, so the program is written
onto that subclass alone. Keep the subclass even once upstream is fixed: the
per-class runner, launch permits and kill names all key on a class per build.
Only the comment explaining why `create_from` must never touch the family's
class can go.

### Summary

`create_from` is documented as "Creates a simulator class from a path to the
simulator executable" and returns "a class instance representing the Spice
simulator". It creates nothing: it assigns `cls.spice_exe` and
`cls.process_name` on the class it was called on and returns that same class.
Calling it on `LTspice` (or any shipped simulator class) retargets every holder
of that class, so two builds of one simulator cannot be bound in one process
through the documented route: the second call silently replaces the first.

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

Any two existing files show it; the paths only need to exist.

### Impact

- A program that keeps two builds of one simulator (LTspice XVII for decks that
  depend on its cp1252 reading of `µ`, LTspice 24 for current ones) gets one
  of them, the last one bound, with no error.
- A `SimRunner` already running on the class launches its next run on the new
  program, so a rebind while runs are queued moves them to another build.
- `process_name`, which spicelib's own `kill_all_spice` reads, moves with it.

### Proposed fix

Create the class the docstring promises, and leave the receiver alone:

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
does both is the gentle route.

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
what it did, and `TestRoutedRuns` that each run launches the build it named.

---

## Bug 19 — `get_default_library_paths` reports every LTspice's library, whichever program the class runs (limitation)

**Status:** known limitation. Recorded 2026-10-03.
**Affected version:** spicelib 1.5.1 (`spicelib/simulators/ltspice_simulator.py`,
`LTspice._default_lib_paths`; `spicelib/sim/simulator.py`,
`Simulator.get_default_library_paths`).
**Our workaround:** `src/ltspice_mcp/lib/simulator.py` — `generation_of` reads
which LTspice a class launches off its program's file name (`XVIIx64.exe` is
XVII, `LTspice.exe` is 24 and later), and `simulator_library_roots` keeps only
that generation's directories from spicelib's list (`_in_generation`). On WSL,
where spicelib's list expands against the Linux home and finds nothing,
`wsl.get_ltspice_lib_paths(generation)` probes the matching Windows folder.
Remove the filter once spicelib reports the library of the program it runs.

### Summary

`LTspice._default_lib_paths` lists the library folders of every LTspice
generation in one list (`~/AppData/Local/LTspice/lib` for LTspice 24 and later,
`~/Documents/LTspiceXVII/lib/` for XVII, and older locations), and
`get_default_library_paths` returns each one that exists. It uses the
executable only to translate paths under Wine. With XVII and a later LTspice
installed side by side, a class bound to either program reports both
libraries, so a caller cannot ask which model library the program it runs
actually reads.

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

A program that trusts the running simulator's own library (we accept a deck's
reference into it without widening the sandbox, because LTspice's netlister
writes such references itself) extends that trust to another program's
library too. On WSL the list finds nothing at all, so XVII's library was
unreachable until we probed it ourselves.

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
**Our workaround:** `lib/controlled_ltspice.py::controlled_ltspice` constructs
separate `'-ini', path` arguments directly. It uses an immutable established
profile and a verified writable copy per attempt. Controlled recovery is
restricted to the audited Windows executable and startup policy; correcting
this upstream typo alone does not establish support for other builds.

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
pins the separate flag/path and launch-local environment. The focused tests
also cover immutable templates, initial writable-copy verification, effective
Windows profile values and unsupported execution policies.

The initial 2026-10-02 investigation had no production adapter and did not
launch the malformed flag. Subsequent native Windows evidence established a
numeric batch solve and normal process exit using
`-Run -b <deck> -ini <profile>` and an actual vendor-prepared settings copy.
It still does not establish the behavior
of malformed argv or validate every LTspice build and solver configuration.

---

## Bug 21 — simulator launches cannot accept a per-run environment (limitation)

**Status:** known API limitation; draft for an upstream enhancement.
Verified 2026-10-02 by the installed signatures and subprocess forwarding code.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`Simulator.run` and `run_function`; `spicelib/simulators/ngspice_simulator.py`,
`NGspiceSimulator.run`; `spicelib/sim/run_task.py`, `RunTask.run`).
**Our workaround:** `src/ltspice_mcp/lib/controlled_ngspice.py::controlled_ngspice`
binds recorded startup settings to a simulator subclass and passes a private
environment mapping directly to `subprocess.run`. The existing runner owns
submission and cancellation.

### Summary

The simulator run interface and shared subprocess helper expose no `env`
argument. `NGspiceSimulator.run` consequently launches with the parent process's
environment. A caller cannot bind a particular `SPICE_SCRIPTS` directory to one
run through that interface. Changing `os.environ` around a launch changes shared
process state and can affect concurrent runs.

This matters for controlled ngspice startup: measured Linux and Windows runs
showed that `-n` suppresses user/local initialization but does not suppress
system spinit. A per-child `SPICE_SCRIPTS` override pointing to captured inert
spinit is needed in addition to `-n` for our recovery contract.

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

Inspect and bind the actual call signature without invoking any executable:

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

Runs requiring different startup directories cannot express those environments
through the standard adapter API. Ambient startup can change simulator behavior
despite unchanged deck bytes. A process-wide environment workaround would also
make the outcome depend on overlapping launch timing.

### Proposed fix

Add an optional keyword-only `env=None` to the simulator interface and
`run_function`, forward it to `subprocess.run` in both output-redirection
branches, and carry a per-run mapping through `SimRunner`/`RunTask`. Preserve
existing inheritance behavior when `env` is `None`. Document that an explicit
mapping is the child environment, matching `subprocess.run`, and never mutate
the parent's environment to implement it.

### Suggested upstream test

Use a Python child rather than a simulator to exercise the real helper. Give
two concurrent calls different values of a test environment variable, have each
child assert its own value, and assert that the parent's environment is unchanged.
Test `env=None` inheritance separately. At the adapter/runner boundary, assert
that each mapping reaches the helper unchanged for both `exe_log` modes.

### Cross-reference

The downstream adapter copies the inherited environment, removes ambient
`SPICE_*` variables, applies its recorded `SPICE_SCRIPTS` override, and includes
`-n` without changing shared environment or compatibility-mode state.
`tests/test_controlled_ngspice.py::test_concurrent_launches_keep_separate_environments`
pins separate environment mappings and an unchanged parent environment.
`::test_verification_refuses_before_spawn` pins the pre-launch verification
boundary. `::test_real_ngspice_uses_only_the_recorded_startup` exercises a real
ngspice process with hostile ambient startup files and checks its numeric output
and absence of the startup marker.

An upstream environment API could remove the duplicated subprocess-forwarding
mechanism. Frozen-byte verification and the captured startup policy would still
belong in this project's recovery implementation.

---

## ngspice replaces a system startup seed before electrical deck evaluation

**Status:** reproduced simulator dependency defect; draft for an ngspice report.
This finding concerns ngspice initialization rather than a spicelib parser or
adapter defect.
**Affected version:** ngspice 42, observed with the Linux console package
`42+ds-3build1`. Matching upstream source defines `WaGauss` and calls
`frontend/trannoise/wallace.c::initw` during startup.
**Existing workaround:** `src/ltspice_mcp/lib/ngspice_driver.py::seeded_commands`
places `setseed` immediately before `source` inside an owned driver. Both
`pdk_native.driver_bytes` and `controlled_ngspice.prepare_seeded_driver` use
this sequence. Native statistical sample seeds remain distinct from an ordinary
recoverable execution's explicit `simulator_seed`; neither contract seeds spinit.

### Summary

A positive `setseed N` in system spinit executes and sets the `rndseed`
variable, but the initial electrical deck's `AGAUSS` values are not controlled
by that seed. Later process initialization overwrites the shared generator
state. The variable still reports the caller's seed, so checking that variable
or checking a random draw within spinit can falsely suggest reproducibility.

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

Create an owned startup directory containing `spinit` with `setseed 17`, and
an owned `bench.cir` containing:

```spice
* Seeded parameter evaluation
.param draw=agauss(10,1,1)
V1 n 0 {draw}
R1 n 0 1000
.op
.end
```

Launch the same electrical bytes twice as separate ngspice processes, with
the owned directory selected through a per-child `SPICE_SCRIPTS` environment:

```text
ngspice -n -D ngbehavior=hsa -b -o first.log -r first.raw bench.cir
ngspice -n -D ngbehavior=hsa -b -o second.log -r second.raw bench.cir
```

Use a 10-second timeout per process and compare the actual `v(n)` values in
the two operating-point raws. The observed seed-17 values were
`9.484939411740886 V` and `12.431703780774 V`. A diagnostic spinit reported
`rndseed=17` in both processes; control-language `sgauss(0)` within spinit
also agreed while the later electrical values differed. User/local init was
suppressed with `-n`; no user profile or installed startup file was changed.

For comparison, the unchanged project driver generator was exercised on the
same electrical bytes through `setseed -> source -> run -> write`. Two
seed-17 runs produced `9.940377579172193 V` with identical numeric binary
payloads; seed 19 produced `9.476947311890937 V`. All three exited zero.
This sequence resets the generator after Wallace initialization and directly
before electrical input evaluation.

### Impact

An execution seed recorded only in spinit cannot guarantee repeatable
electrical parameter randomness on the affected build. Frozen input bytes
and an unchanged executable are insufficient when the shared generator state
is subsequently initialized from process identity.

### Proposed fix

Initialize Wallace state before executing startup files, or make its setup
preserve the caller's seed and define how its pool and the shared generator
are initialized. Keep the reported seed consistent with actual generator
state. Avoid claiming that resetting the shared generator alone reseeds an
already-populated Wallace pool.

### Suggested upstream test

Through the real console startup path, load an owned spinit with a positive
seed and solve a tiny electrical `.param AGAUSS` operating point in two
separate processes. Assert identical numeric results for the same seed and
different results for a selected different seed. Cover the `WaGauss` build,
`-n`, a per-child `SPICE_SCRIPTS` override, and batch/raw output. Test
transient noise separately because it can consume the Wallace pool rather
than the parameter evaluator's generator.

### Cross-reference

`ngspice_driver.seeded_commands` supplies the shared deterministic ordering;
`pdk_native.driver_bytes` uses it for native samples, while
`controlled_ngspice.prepare_seeded_driver` captures a separate owned driver for
ordinary seeded recovery and `verify_seeded_driver` checks its identity and
bytes before launch. `tests/test_recovery_seed.py::test_shared_sequence_preserves_the_audited_native_driver_bytes`
pins reuse of the native command sequence;
`::test_real_seeded_coordinator_reproduces_electrical_results` and
`::test_seeded_retry_keeps_successes_and_reuses_recorded_seed` cover the ordinary
seeded coordinator and retry contract.

`native_execution.prepare_native_cases` captures electrical and driver bytes
before publication, and `pdk_native.verify_launch` verifies them before
submission. `tests/test_pdk_native.py::test_windows_paths_with_spaces_use_only_generated_relative_names`
checks the generated driver contract. The isolated three-launch console
probe verifies numeric repeatability for the tiny `.param AGAUSS` deck; no
startup-file seed contract is claimed. This evidence
does not establish reproducibility for every stochastic function or analysis.

---

## Bug 22 — `AscEditor` cannot open a sheet whose block symbol has no sheet of its own (limitation)

**Status:** known limitation; draft for an upstream enhancement. Observed
2026-10-06 against the exports of LTspice 26.1.1 and LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/editor/asc_editor.py`,
`AscEditor.reset_netlist` and `AscEditor._get_subcircuit`).
**Our workaround:** none. `lib/schematic_ops.py::make_editor` turns the
`FileNotFoundError` into a `SymbolResolutionError` whose message names the
missing file, so the caller learns which sheet is wanted.

### Summary

A symbol of `SymbolType BLOCK` stands for a subcircuit. LTspice netlists an
instance of one as a call to a subcircuit of the symbol's name, whether or not
a sheet of that name exists: the definition may come from a library named on
the sheet, or be missing until the deck is run. `AscEditor` instead resolves
every block symbol to its own `.asc` while it loads the parent, and raises
`FileNotFoundError` when there is none. A sheet LTspice exports without
complaint therefore cannot be opened, read or edited at all.

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
goes on.

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
failing the load, and raise only when something asks for the subcircuit's
contents.

### Suggested upstream test

```python
def test_block_symbol_without_its_sheet_still_loads(tmp_path):
    # probe4.asy (SymbolType BLOCK) beside parent.asc, no probe4.asc
    editor = AscEditor(tmp_path / "parent.asc")
    assert "U1" in editor.get_components()
```

### Cross-reference

`tests/test_recorded_ltspice_schematics.py::TestExportedNames::test_a_block_symbol_with_no_sheet_of_its_own_cannot_be_opened`
pins both halves: LTspice exports the sheet, and the editor refuses it with a
message naming `probe4.asc`. Once upstream loads such a sheet, that test's
second half goes and the sheet joins the ones the editor is held to.

---

## Bug 23 — a simulator launch cannot be kept off the user's desktop (limitation)

**Status:** known API limitation; draft for an upstream enhancement. Measured
2026-10-06 on Windows 11 with LTspice 26.1.1 and LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/sim/simulator.py`,
`run_function`; `spicelib/simulators/ltspice_simulator.py`, `LTspice.run` and
`LTspice.create_netlist`).
**Our workaround:** none in the server yet. The fixture recorder starts
LTspice itself, on a desktop of its own (`tests/ltspice_recorder.py`,
`HiddenDesktop`).

### Summary

LTspice opens a window even for a batch run (`-Run -b`) and for a batch export
(`-netlist`), and holds the keyboard focus until it exits. `run_function`
starts it with a bare `subprocess.run`, and `Simulator.run` offers no way to
pass startup information, so every simulation takes the focus from whatever
the user is typing into, and a sweep takes it continuously.

The usual remedy does not work: started with `STARTF_USESHOWWINDOW` and
`SW_SHOWMINNOACTIVE` or `SW_HIDE`, LTspice was still the foreground window for
about three quarters of a run. What does work is starting it with
`STARTUPINFO.lpDesktop` naming a desktop made with `CreateDesktopW`: its
windows exist only there. Python's `subprocess.STARTUPINFO` has no `lpDesktop`,
so this needs `CreateProcessW` through `ctypes`.

A second consequence of the visible window: LTspice answers some inputs with a
message box and waits (XVII, given a sheet that starts with a byte order mark:
"Aborting: Unknown schematic syntax"). On the user's desktop a stray key press
dismisses it and the launch appears to have ended by itself; off it, the
launch blocks until its timeout, which is the behaviour a caller can rely on.

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

Let a caller supply how the process is started: a `startupinfo` /
`creationflags` pass-through on `Simulator.run` and `create_netlist` at the
least, or an optional launcher callable in place of `run_function`. A
`desktop=` option on Windows would cover this case directly.

### Suggested upstream test

On Windows, start a simulator through the new hook on a desktop created for
the test, and assert the foreground window's owning process never becomes the
simulator's while the run completes with exit code 0.

### Cross-reference

`tests/ltspice_recorder.py::HiddenDesktop` is the working launch, including
reading the text of a message box (`dialog`) so that a build which stops to
ask is recorded as having done so. The recordings of
`export/micro_utf8_bom` and `export/micro_utf16le_bom` on LTspice XVII carry
that text. Giving the server the same launch is tracked separately; once
upstream offers a hook, both use it in place of their own `CreateProcessW`.

---

## Bug 24 — `detect_encoding` gives up on an 8-bit log holding a byte none of its codecs defines

**Status:** draft for an upstream spicelib pull request. Observed 2026-10-06
against a log written by LTspice XVII 17.0.37.
**Affected version:** spicelib 1.5.1 (`spicelib/utils/detect_encoding.py`,
`detect_encoding`; reached from `spicelib/log/ltsteps.py`,
`LTSpiceLogReader.__init__`).
**Our workaround:** `lib/log_parser.py::make_log_reader` decodes the log
itself (`lib/encoding.py`, which has a character for every byte) and, when
spicelib has refused the file, hands it a UTF-8 copy.

### Summary

LTspice XVII copies a deck's title line into the first line of its log as the
bytes the deck holds (`Circuit: * <title>`). A title saved in a double-byte
code page, Japanese or Chinese, holds bytes that are no character in cp1252
(0x81, 0x8D, 0x8F, 0x90, 0x9D). `detect_encoding` opens the file in each of a
fixed list of codecs and returns the first that decodes it and matches the
expected pattern. For such a log none does, so it raises
`EncodingDetectError: Expected pattern ... not found`, and
`LTSpiceLogReader` cannot be built for a run that finished without error.
The message blames the pattern; the `Circuit:` line is there.

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
8-bit file. `shift_jis` takes most Japanese text but not the NEC and IBM
extensions cp932 adds, and nothing in the list takes GBK or the Korean
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
- The error names a missing pattern, which sends the reader looking for a
  malformed log.

### Proposed fix

End the list with a codec that has a character for every byte (`latin-1`), so
an 8-bit log is always read: everything the reader parses is ASCII, and the
title is only carried along. Report an encoding failure as one, separately
from a missing pattern.

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
such a log, the last candidate in `make_log_reader` goes.
