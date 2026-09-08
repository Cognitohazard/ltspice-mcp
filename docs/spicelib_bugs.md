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
