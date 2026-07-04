"""Log file (.log) parsing: error extraction, success summaries, measurements, Fourier.

For .raw file parsing (waveform data, statistics), see raw_parser.py.
"""

from __future__ import annotations

import contextlib
import logging
import re
import tempfile
from pathlib import Path
from typing import TypedDict

from spicelib.log.ltsteps import LTSpiceLogReader
from spicelib.log.semi_dev_op_reader import opLogReader
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.spice_validator import validate_directive

logger = logging.getLogger(__name__)


class MeasErrorEntry(TypedDict):
    """One .MEAS parse failure with an optional fix suggestion."""

    directive: str
    raw_block: str
    suggestion: str | None


class LogDiagnostics(TypedDict):
    """Return shape of :func:`extract_log_diagnostics`."""

    warnings: list[str]
    errors: list[str]
    meas_errors: list[MeasErrorEntry]


class _MeasurementMetadata(TypedDict, total=False):
    """Optional metadata folded into a .MEAS result.

    Each metadata field is either a scalar (when the value is constant
    across .step iterations — e.g. a literal ``FROM=2m``) or a list of
    per-step values (when LTspice computed a different marker per step,
    e.g. TRIG/TARG times of a per-step rise time).
    """

    range_from: float | list[float | None] | None
    range_to: float | list[float | None] | None
    at: float | list[float | None] | None


class MeasurementEntry(_MeasurementMetadata):
    """One .MEAS result, with optional range/at metadata folded in.

    ``values`` is one entry per .step iteration (length 1 for unstepped runs).
    ``range_from`` / ``range_to`` carry the FROM/TO bounds for windowed measurements.
    ``at`` carries the AT/WHEN time/freq for point measurements. Missing when
    not applicable (use ``.get`` rather than ``[]`` to access). When the
    underlying value varies per .step, the field is a list aligned with
    ``values`` rather than a single scalar.
    """

    values: list[float | None]


class MeasurementsOutput(TypedDict):
    """Return shape of :func:`parse_measurements`.

    ``errors`` is populated only on the empty-measurements path (the log
    had no .MEAS results and the parser surfaced why). Always present in
    the return value — ``None`` when the measurement parse succeeded.

    ``measurements`` is keyed by .meas name; each entry is a structured
    :class:`MeasurementEntry` with ``values`` plus folded-in ``range_from``,
    ``range_to``, and ``at`` metadata. The flat ``name_from`` / ``name_to`` /
    ``name_at`` keys that spicelib emits are not surfaced separately.
    """

    measurements: dict[str, MeasurementEntry]
    step_count: int
    errors: list[str] | None
    warnings: list[str] | None
    failed_measurements: list[str]


_MAX_DIAGNOSTICS = 50

# Above this many ESTIMATED trace samples (axis points × number of non-axis
# traces) a completed result is loaded axis-only and the per-trace value scan
# (NaN/Inf/extreme-magnitude detection) is skipped, with the gap surfaced as a
# coverage observation. At or below it, all traces are loaded and scanned so the
# value facts are surfaced and no skip is reported. Budgeting on TOTAL samples,
# not axis points alone, bounds the actual load: a wide node dump with a
# moderate point count costs as much memory as a long single-probe .tran, and
# both must skip. 5M samples ≈ 40 MB (real) / 80 MB (complex AC).
_VALUE_SCAN_SAMPLE_BUDGET = 5_000_000

# Error keywords to search for in log files (case-insensitive). The trailing
# entries are LTspice convergence-abort phrases (the one-word "Timestep too
# small", the failing-node line, and the "Last Node Voltages" dump) that name
# the actual failure and land at the very END of the log.
_ERROR_KEYWORDS = [
    "error",
    "fatal",
    "failed",
    "singular matrix",
    "convergence",
    "time step too small",
    "timestep too small",
    "trouble with node",
    "last node voltages",
    "missing",
    "can't find",
]

# --- Structured log diagnostic extraction ---

# filepath(line): message — LTspice parse errors (.meas, .param, behavioral sources)
_RE_FILE_ERROR = re.compile(r"^.+\(\d+\):\s+.+")
# ^^^ caret pointer (follows a source line in a parse error block)
_RE_CARET = re.compile(r"^\s*\^+\s*$")
# "Error on line N :" — component-level errors after subcircuit expansion
_RE_LINE_ERROR = re.compile(r"^Error on line \d+", re.IGNORECASE)
# "Fatal Error:" — missing subcircuits/models
_RE_FATAL = re.compile(r"^Fatal Error:", re.IGNORECASE)
# Explicit warning prefix — LTspice "Warning:" and ngspice "Warning --".
_RE_WARNING = re.compile(r"^(?:Warning|WARNING)\s*(?:--|:)", re.IGNORECASE)
# Explicit error prefix — LTspice "ERROR: Node ... is floating", ngspice
# "Error: circuit not parsed.". Distinct from _RE_FATAL (^Fatal Error:) and
# _RE_LINE_ERROR (^Error on line N), both of which are matched earlier so the
# more specific rules win. Without this rule a hard ERROR line falls through and
# a failed-physics run (e.g. a 1e9 V floating node) reports as a clean success.
_RE_ERROR = re.compile(r"^(?:Error|ERROR)\s*:", re.IGNORECASE)
# ngspice reports convergence FAILURES under a "Warning:" / "Warning --" prefix,
# so without special handling _RE_WARNING would downgrade a run that produced no
# usable data (a floating node prints "gmin stepping failed" + "source stepping
# failed", the raw is numerical noise, and ngspice still exits 0). These phrases
# are terminal, so classify them as errors regardless of any Warning prefix.
# A transient "singular matrix" note is deliberately NOT listed —
# ngspice may still recover from it via gmin/source stepping. Matched as a
# case-insensitive substring, so any "Warning:" prefix is ignored without a
# separate strip.
#
# Gmin/source stepping are split out because they are ALSO the intermediate
# rungs of LTspice's OP-solve escalation ladder: LTspice prints
# "<method> stepping failed to find operating point" before trying the next
# method, then a success line when one converges. They are terminal only when
# no later success line exists (see the converged-check in
# extract_log_diagnostics). ngspice's genuine no-data run prints no success
# line, so those still classify as errors.
_OP_STEPPING_FAILURE_PHRASES = (
    "gmin stepping failed",
    "source stepping failed",
)
# Always-terminal convergence failures — no later success rescues these.
_CONVERGENCE_FAILURE_PHRASES = ("iteration limit reached",)
# Bare convergence / runtime messages with no prefix.
# These are matched anchored to the start of a (stripped) line so we don't
# false-positive on phrases that merely *contain* one of these substrings
# (e.g. "the singular matrix decomposition succeeded").
_BARE_ERROR_PHRASES = [
    "singular matrix",
    "time step too small",
    "no convergence",
    "questionable use of curly braces",
]
# ngspice-specific diagnostic patterns (not matched by the LTspice rules above).
_RE_NGSPICE_MEAS_BLOCKED = re.compile(r"No \.measure possible in batch mode", re.IGNORECASE)
_RE_NGSPICE_UNIMPLEMENTED = re.compile(r"unimplemented dot command '([^']+)'", re.IGNORECASE)
# ngspice skips .four/.fourier when run with -r rawfile (same batch-mode
# limitation as .meas). It prints this note and produces no Fourier block;
# without surfacing it the user asks for Fourier and silently gets nothing.
_RE_NGSPICE_FOUR_BLOCKED = re.compile(r"\.fourier line ignored", re.IGNORECASE)
# ngspice "Note:" lines carry per-component diagnostics (e.g. "v1: has
# no value, DC 0 assumed"). Exclude the preamble "Compatibility modes"
# note which is just informational noise.
_RE_NGSPICE_NOTE = re.compile(r"^Note:\s+(.+)", re.IGNORECASE)
_NGSPICE_NOTE_SKIP = {"compatibility modes selected"}
# ngspice echoes the deck title on a ``Circuit: <title>`` line. spicelib's
# LTspice-shaped log reader misparses that as a measurement named "circuit",
# scraping the first number out of the title text as its "value" — a fabricated
# .meas result on every ngspice run. It is never a real measurement name, so it
# is dropped before it can pollute measurement_stats / the observation surfacer.
_NGSPICE_TITLE_PSEUDO_MEAS = "circuit"

# LTspice writes one ``.step <name>=<value>[, ...]`` line per iteration of a
# .step parametric run — useful when the .raw lacks an axis (e.g. .step param
# RVAL + .tran) and ``RawRead.get_steps()`` returns nothing.
_RE_STEP_LINE = re.compile(r"^\.step\s+(.+)$", re.IGNORECASE)
# Value capture stops at whitespace/comma AND at the trailing degree symbol
# LTspice writes for ``.step temp=-40°``. Without the ``°`` exclusion the
# captured value is ``-40°``, which downstream value-parsing rejects.
_RE_STEP_KV = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^,\s°]+)")
# Stepped ``.op`` runs don't write ``.step name=val`` markers — LTspice
# logs one of these per iteration instead. Counting them is the only
# reliable signal that the bias point ran multiple times.
_RE_OP_ITERATION = re.compile(
    r"^\s*Direct Newton iteration succeeded in finding operating point",
    re.IGNORECASE,
)
# Terminal success line of LTspice's OP-solve escalation ladder. Any of these
# means the bias point WAS found (possibly via a fallback method), so the
# earlier "<method> stepping failed to find operating point" rungs are benign.
# Covers Direct Newton / Gmin / Source stepping ("... succeeded in finding
# operating point") and the pseudo-transient fallback ("Pseudo Transient
# succeeded at <t>", which omits "operating point").
_RE_OP_SOLVE_SUCCEEDED = re.compile(
    r"succeeded in finding (?:the )?operating point"
    r"|pseudo[- ]?transient succeeded"
    r"|stepping succeeded",
    re.IGNORECASE,
)
# Start of an OP-solve block. LTspice prints one "Direct Newton iteration ..."
# line at the top of every operating-point solve — including each step of a
# stepped .op — so the NEXT such line after a stepping-failure marks the next
# solve block. Used to scope the converged-check per block: a converged step
# must not suppress a genuinely failed later step.
_RE_OP_SOLVE_ATTEMPT = re.compile(r"^\s*Direct Newton iteration", re.IGNORECASE)

# Ambient / nominal temperature echoed in the log header. LTspice writes two
# lowercase ``key = value`` lines (``temp = 27`` / ``tnom = 27``); ngspice
# writes one combined ``Doing analysis at TEMP = 27.000000 and TNOM = ...``
# line per analysis. The two phrasings are disjoint. ``^`` anchoring on the
# LTspice patterns is load-bearing: it avoids matching ``.step temp=-40°``
# step lines (they begin ``.step``) or a B-source ``temp=x`` mid-line param.
_RE_LT_TEMP = re.compile(r"^temp\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE | re.MULTILINE)
_RE_LT_TNOM = re.compile(r"^tnom\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE | re.MULTILINE)
_RE_NG_TEMP = re.compile(
    r"analysis at TEMP\s*=\s*([-+]?\d+(?:\.\d+)?)\s+and\s+TNOM\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# Missing .MODEL — appears in log as:
#   Error on line 2 : s1 n003 n001 n002 0 sw Unable to find definition of model "sw"
# (rarer GUI-dialog phrasing "Can't find definition of model" is also tolerated).
_RE_MISSING_MODEL = re.compile(
    r'(?:Unable to find|Can\'?t find) definition of model\s+"([^"]+)"',
    re.IGNORECASE,
)
# ngspice phrasing for an unresolved model reference, e.g.:
#   Error: undefined model 2n2222
# The name is the next token, with or without surrounding quotes.
_RE_MISSING_MODEL_NGSPICE = re.compile(
    r'undefined model\s*:?\s+"?([A-Za-z0-9_.\-]+)"?',
    re.IGNORECASE,
)
# Missing .SUBCKT — appears in log as:
#   Fatal Error: Unknown subcircuit called in: xu1 n004 n001 vcc 0 lm741
# The missing subcircuit name is the LAST token of the instance line.
_RE_MISSING_SUBCKT = re.compile(
    r"Unknown subcircuit called in:\s+(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def read_log_text(log_path: Path) -> str:
    """Read a log file, returning empty string on I/O failure.

    Public so callers (e.g. ``raw_parser.build_simulation_summary``) can
    pre-read the log buffer once and pass the text to every parser that
    needs it instead of triggering one syscall per parser.
    """
    try:
        return log_path.read_text(errors="replace")
    except OSError:
        return ""


def read_device_op_points(log_path: Path) -> dict[str, float]:
    """Per-device small-signal op-point params from an LTspice ``.log``.

    LTspice writes the operating point of each semiconductor (gm, gds, vth,
    vdsat, terminal currents, junction caps) into the ``Semiconductor Device
    Operating Points:`` block of the log — but only when the deck carries
    ``.options logopinfo`` (and only for ``.op`` runs). ngspice exposes the
    same data as ``@dev[param]`` raw traces instead, so this reader is for the
    LTspice path.

    Returns the flattened block keyed in the ``@dev[param]`` form (e.g.
    ``@m1[gm]``, lowercased), so callers can fold it straight into
    ``device_op_points`` and address top-level devices by the ``m1.gm``
    shorthand. Subcircuit semiconductors keep LTspice's colon-qualified name
    (``@q:q2:1:2[gm]``); ``operating_point(device=...)`` matches those by
    instance. Non-numeric fields (the ``Model:`` row) are dropped. Empty dict
    when the block is absent or unparseable.
    """
    out: dict[str, float] = {}
    try:
        parsed = opLogReader(str(log_path))
    except (OSError, ValueError):
        return out
    for devices in parsed.values():  # categories: 'mosfet transistors', 'diodes', …
        for dev, params in devices.items():
            for pname, pval in params.items():
                if not isinstance(pval, (int, float)):
                    continue  # skip the 'Model:' string row (opLogReader yields float | str)
                out[f"@{dev.lower()}[{pname.lower()}]"] = float(pval)
    return out


def _resolve_log_text(log_path: Path | None, text: str | None) -> str:
    """Resolve ``(log_path, text)`` to a text buffer, preferring an
    explicit ``text`` over re-reading the file. Empty string on missing
    inputs / I/O failure.
    """
    if text is not None:
        return text
    if log_path is None:
        return ""
    return read_log_text(log_path)


def parse_temperatures(
    log_path: Path | None = None, *, text: str | None = None
) -> tuple[float | None, float | None]:
    """``(temp_c, tnom_c)`` ambient/nominal temperature from the log header.

    Reads the value both LTspice (``temp = 27`` / ``tnom = 27``) and ngspice
    (``Doing analysis at TEMP = 27.0 and TNOM = 27.0``) print by default;
    ``None`` for either value when the log doesn't carry it. On a stepped-
    temperature run the header value is step 0's, which is still a truthful
    fact about the header — no attempt is made to derive a per-step temp.
    """
    body = _resolve_log_text(log_path, text)
    ng = _RE_NG_TEMP.search(body)
    if ng:
        return float(ng.group(1)), float(ng.group(2))
    t = _RE_LT_TEMP.search(body)
    n = _RE_LT_TNOM.search(body)
    return (float(t.group(1)) if t else None, float(n.group(1)) if n else None)


def parse_step_iterations(
    log_path: Path | None = None, *, text: str | None = None
) -> list[dict[str, float]]:
    """Parse ``.step name=value[, ...]`` lines from an LTspice log.

    Returns one dict per step iteration, in the order they appeared.
    Empty list if the log has no .step lines (i.e. unstepped run) or
    can't be read.

    Used as a fallback by ``query_value(step_axis=…)`` when ``.step param NAME`` runs leave
    spicelib's ``RawRead.get_steps()`` empty — the parameter→step mapping
    is recorded in the log even when it's absent from the .raw header.

    Pass ``text`` to parse a pre-read log buffer (avoids a second read in
    callers that already have the contents). When both ``log_path`` and
    ``text`` are given, ``text`` wins.

    Non-numeric values (e.g. ``5k``) are silently dropped from a row;
    LTspice always writes already-evaluated floats, so this only matters
    for hand-crafted test fixtures.
    """
    iterations: list[dict[str, float]] = []
    for line in _resolve_log_text(log_path, text).splitlines():
        m = _RE_STEP_LINE.match(line.strip())
        if not m:
            continue
        params: dict[str, float] = {}
        for kv in _RE_STEP_KV.finditer(m.group(1)):
            try:
                params[kv.group(1)] = float(kv.group(2))
            except ValueError:
                continue
        if params:
            iterations.append(params)
    return iterations


def count_op_iterations(log_path: Path | None = None, *, text: str | None = None) -> int:
    """Count "Direct Newton iteration succeeded" lines in an LTspice log.

    Stepped ``.op`` runs solve the bias point once per step but do NOT
    write ``.step <name>=<value>`` lines (only ``.tran``/``.ac`` step
    iterations do). The Newton-iteration message is the only reliable
    signal that the .op ran multiple times.

    See :func:`parse_step_iterations` for the ``text`` keyword usage.
    """
    text = _resolve_log_text(log_path, text)
    return sum(1 for line in text.splitlines() if _RE_OP_ITERATION.match(line))


def scan_op_step_log(
    log_path: Path | None = None, *, text: str | None = None
) -> tuple[list[dict[str, float]], int]:
    """Single-pass equivalent of ``parse_step_iterations`` +
    ``count_op_iterations``. Returns ``(iterations, op_count)``.

    Used by stepped-``.op`` detection in :func:`build_simulation_summary`
    where both signals are needed and walking the log twice is wasteful.
    """
    iterations: list[dict[str, float]] = []
    op_count = 0
    for line in _resolve_log_text(log_path, text).splitlines():
        if _RE_OP_ITERATION.match(line):
            op_count += 1
            continue
        m = _RE_STEP_LINE.match(line.strip())
        if not m:
            continue
        params: dict[str, float] = {}
        for kv in _RE_STEP_KV.finditer(m.group(1)):
            try:
                params[kv.group(1)] = float(kv.group(2))
            except ValueError:
                continue
        if params:
            iterations.append(params)
    return iterations, op_count


def missing_refs_from_text(text: str) -> list[str]:
    """Scan log text for missing .MODEL / .SUBCKT names, de-duplicated in order."""
    seen: set[str] = set()
    refs: list[str] = []

    for regex in (_RE_MISSING_MODEL, _RE_MISSING_MODEL_NGSPICE):
        for m in regex.finditer(text):
            name = m.group(1)
            if name and name not in seen:
                seen.add(name)
                refs.append(name)

    for m in _RE_MISSING_SUBCKT.finditer(text):
        tokens = m.group(1).split()
        if not tokens:
            continue
        name = tokens[-1]
        if name and name not in seen:
            seen.add(name)
            refs.append(name)

    return refs


def extract_missing_refs(log_path: Path) -> list[str]:
    """Extract names of models/subcircuits that LTspice couldn't resolve."""
    try:
        return missing_refs_from_text(log_path.read_text(errors="replace"))
    except OSError:
        return []


def _op_block_recovered(lines: list[str], idx: int) -> bool:
    """Whether the OP-solve block holding a stepping-failure at ``idx`` converged.

    LTspice escalates within one solve (Direct Newton -> Gmin -> Source stepping
    -> pseudo-transient) and prints a success line if a fallback works. Scanning
    forward from the failure: a success before the next solve-block start means
    this block was rescued (benign rung); reaching the next block's
    "Direct Newton iteration" line first means this block never converged (the
    failure is real). Scoped per block — not whole-log — so a converged step in
    a stepped .op cannot mask a genuinely failed later step. A genuine ngspice
    no-data run has no success line at all, so it stays classified as an error.
    """
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        # Boundary check FIRST: "Direct Newton iteration succeeded ..." both
        # starts the next block AND matches the success pattern, but that success
        # belongs to the NEXT solve, not this one. Within a ladder that used
        # stepping, the rescue is a pseudo-transient / stepping "succeeded" line,
        # never a "Direct Newton iteration" line — so a Direct-Newton line here
        # is always the next block.
        if _RE_OP_SOLVE_ATTEMPT.match(line):
            return False
        if _RE_OP_SOLVE_SUCCEEDED.search(line):
            return True
    return False


def extract_log_diagnostics(log_path: Path) -> LogDiagnostics:
    """Extract structured warnings and errors from an LTspice log file.

    Detects all known LTspice diagnostic patterns:
    - filepath(line): message + source line + ^^^ caret blocks
    - "Error on line N:" component errors
    - "Fatal Error:" messages
    - "Warning:"/"WARNING:" prefixed lines
    - Bare convergence/runtime messages (e.g., "Singular matrix")

    Returns:
        {
          "warnings": [str, ...],
          "errors": [str, ...],
          "meas_errors": [{"directive": str, "raw_block": str,
                           "suggestion": str | None}, ...],
        }

    ``meas_errors`` is a structured slice of ``errors`` covering .MEAS parse
    failures specifically, with the offending directive extracted and an
    optional suggestion when the failure pattern is in the
    spice_validator blocklist (e.g. ``vdb()`` in .MEAS).
    """
    warnings: list[str] = []
    errors: list[str] = []
    meas_errors: list[MeasErrorEntry] = []

    try:
        # errors="replace": LTspice logs often carry cp1252 bytes (µ/°/±); a
        # strict read would drop all diagnostics for that log (matches the other
        # log reads in this module).
        content = log_path.read_text(errors="replace")
    except Exception:
        return {"warnings": warnings, "errors": errors, "meas_errors": meas_errors}

    # Some simulators (notably ngspice) print analysis diagnostics — e.g.
    # "doAnalyses: ..." / "analysis not run" — to stdout while exiting 0 and
    # writing only results to the ``-o`` log. When the runner captured that
    # stream (``exe_log=True``) into a sibling ``.exe.log``, fold its lines in
    # so the classifiers below surface them too. Quiet runs (LTspice batch)
    # leave it empty/absent, so this is a no-op there.
    try:
        exe_content = log_path.with_suffix(".exe.log").read_text(errors="replace")
    except OSError:
        exe_content = ""
    if exe_content.strip():
        # Fold in only lines not already in the -o log (some builds mirror a
        # diagnostic to both streams; avoid double-reporting), separated by a
        # blank line so the parse-error block lookahead below can't stitch a
        # caret/source block across the two-file boundary.
        seen = set(content.splitlines())
        extra = [ln for ln in exe_content.splitlines() if ln not in seen]
        if extra:
            content = content + "\n\n" + "\n".join(extra)

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # filepath(line): message — collect the block (source line + caret)
        if _RE_FILE_ERROR.match(line):
            block = [stripped]
            # Grab following source line + caret if present
            j = i + 1
            while j < len(lines) and j <= i + 2:
                next_stripped = lines[j].strip()
                if not next_stripped:
                    break
                block.append(next_stripped)
                if _RE_CARET.match(lines[j]):
                    j += 1
                    break
                j += 1
            full_block = "\n".join(block)
            errors.append(full_block)

            # If the source line in the block is a .MEAS directive, classify
            # it as a meas_error and attach a suggestion when the failure
            # matches a known pattern. The source line is the second entry
            # in the block (first is "file(line): message").
            if len(block) >= 2:
                source_line = block[1].lstrip()
                if source_line.lower().startswith(".meas"):
                    val_err = validate_directive(source_line, simulator="LTspice")
                    meas_errors.append(
                        {
                            "directive": source_line,
                            "raw_block": full_block,
                            "suggestion": val_err.suggestion if val_err else None,
                        }
                    )
            i = j
            continue

        # Fatal Error:
        if _RE_FATAL.match(stripped):
            errors.append(stripped)
            i += 1
            continue

        # Error on line N:
        if _RE_LINE_ERROR.match(stripped):
            errors.append(stripped)
            i += 1
            continue

        # Bare "Error:" / "ERROR:" prefix (floating node, circuit-not-parsed, ...)
        if _RE_ERROR.match(stripped):
            errors.append(stripped)
            i += 1
            continue

        # ngspice convergence FAILURES arrive "Warning:"-prefixed; classify the
        # terminal ones as errors (substring match, so the prefix is ignored)
        # before the generic warning rule below would downgrade a no-data run.
        stripped_lower = stripped.lower()
        # Gmin/source stepping "failed" is an intermediate escalation rung on
        # LTspice — drop it when THIS solve block later converged; keep it as an
        # error otherwise (ngspice no-data run, or a step that never converged).
        if any(phrase in stripped_lower for phrase in _OP_STEPPING_FAILURE_PHRASES):
            if not _op_block_recovered(lines, i):
                errors.append(stripped)
            i += 1
            continue
        if any(phrase in stripped_lower for phrase in _CONVERGENCE_FAILURE_PHRASES):
            errors.append(stripped)
            i += 1
            continue

        # Warning: / WARNING: / Warning --
        if _RE_WARNING.match(stripped):
            warnings.append(stripped)
            i += 1
            continue

        # Bare convergence / runtime phrases — anchored at start of line to
        # avoid false positives on lines that merely contain the phrase as a
        # substring (e.g., "the singular matrix decomposition succeeded").
        if any(stripped_lower.startswith(phrase) for phrase in _BARE_ERROR_PHRASES):
            errors.append(stripped)
            i += 1
            continue

        # ngspice-specific diagnostics
        if _RE_NGSPICE_MEAS_BLOCKED.search(stripped):
            warnings.append(stripped + " Use signal_stats or query_value for post-processing.")
            i += 1
            continue
        if _RE_NGSPICE_FOUR_BLOCKED.search(stripped):
            warnings.append(
                stripped + " ngspice skips .four when writing a rawfile, so Fourier/THD "
                "is unavailable for this run."
            )
            i += 1
            continue
        if _RE_NGSPICE_UNIMPLEMENTED.search(stripped):
            errors.append(stripped)
            i += 1
            continue
        m_note = _RE_NGSPICE_NOTE.match(stripped)
        if m_note:
            note_body = m_note.group(1).strip()
            if not any(skip in note_body.lower() for skip in _NGSPICE_NOTE_SKIP):
                warnings.append(stripped)
            i += 1
            continue

        i += 1

    return {"warnings": warnings, "errors": errors, "meas_errors": meas_errors}


def extract_error_context(log_file: Path, max_lines: int = 20) -> str:
    """Extract relevant error context from simulation log file.

    Searches for error keywords and returns surrounding lines.
    If no errors found, returns the last N lines of the log.

    Args:
        log_file: Path to simulation log file
        max_lines: Maximum number of lines to return

    Returns:
        String containing relevant log excerpt with context around errors,
        or "(Log file not found)" if the file doesn't exist
    """
    if not log_file.exists():
        return "(Log file not found)"

    try:
        # errors="replace": tolerate cp1252 bytes in the log (see above).
        content = log_file.read_text(errors="replace")
        lines = [line.rstrip() for line in content.splitlines()]

        if not lines:
            return "(Empty log file)"

        error_indices = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in _ERROR_KEYWORDS):
                error_indices.append(i)

        if not error_indices:
            start_line = max(0, len(lines) - max_lines)
            excerpt = lines[start_line:]
            if start_line > 0:
                excerpt.insert(0, "...")
            return "\n".join(excerpt)

        # Anchor a [center-3, center+7] context window on both the FIRST and
        # the LAST keyword hit. The first hit captures a parse error's caret
        # block; the last captures the convergence-abort tail (failing node,
        # "Last Node Voltages") an LTspice run writes at the very end — which a
        # first-only anchor dropped whenever an earlier benign keyword line
        # (e.g. "Missing parameter") came first.
        def _window(center: int) -> tuple[int, int]:
            return max(0, center - 3), min(len(lines), center + 7)

        head_start, head_end = _window(error_indices[0])
        tail_start, tail_end = _window(error_indices[-1])

        if tail_start <= head_end:
            # First and last windows touch/overlap — render one contiguous
            # slice spanning both, capped at max_lines.
            start = head_start
            end = max(head_end, tail_end)
            if end - start > max_lines:
                end = start + max_lines
            excerpt = lines[start:end]
            if start > 0:
                excerpt.insert(0, "...")
            if end < len(lines):
                excerpt.append("...")
            return "\n".join(excerpt)

        # Disjoint head and tail — show both, joined by an ellipsis, splitting
        # the line budget so neither dominates.
        head_budget = max_lines // 2
        head_slice = lines[head_start : min(head_end, head_start + head_budget)]
        tail_slice = lines[max(tail_start, tail_end - (max_lines - head_budget)) : tail_end]

        excerpt = []
        if head_start > 0:
            excerpt.append("...")
        excerpt.extend(head_slice)
        excerpt.append("...")
        excerpt.extend(tail_slice)
        if tail_end < len(lines):
            excerpt.append("...")
        return "\n".join(excerpt)

    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return f"(Error reading log file: {e})"


def parse_success_summary(
    raw_file: Path,
    log_file: Path,
    duration: float,
    *,
    dialect: str | None = None,
    netlist: Path | None = None,
) -> dict:
    """Build the success-path summary that ``run_simulation`` returns.

    Delegates to ``raw_parser.build_simulation_summary`` so the canonical
    summary fields (``range``, ``measurements``, ``fourier``, ``meas_errors``)
    are included alongside the legacy ``sim_type``/``step_count``/``signals``
    fields. Adds ``raw_file``/``log_file`` for downstream tool chains that
    feed these back into ``simulation_summary``, ``measurement_stats``, etc.

    ``netlist`` (when supplied and readable) enables the requested-vs-produced
    reconciliation in the observation surfacer.

    Value surfacing respects the bounded-load contract: single-point results
    (operating points) load full traces and are scanned for NaN/extremes;
    multi-point results stay axis-only and record the skipped scan as a coverage
    observation rather than materialising every trace on every completion.

    Truncates ``warnings``/``errors`` to ``_MAX_DIAGNOSTICS`` entries with
    the ``*_truncated`` sibling preserved from the prior implementation.

    Returns partial data on parse errors (graceful degradation): an
    unparseable raw still yields a dict carrying the paths and duration.
    """
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from ltspice_mcp.lib.result_observations import parse_requested_outputs

    requested: dict[str, list[str]] | None = None
    if netlist is not None:
        try:
            requested = parse_requested_outputs(netlist.read_text(errors="replace"))
        except OSError:
            requested = None

    result: dict = {
        "sim_type": "Unknown",
        "duration": duration,
        "step_count": 1,
        "warnings": [],
        "signals": [],
        "raw_file": str(raw_file),
        "log_file": str(log_file),
    }

    try:
        # Two-step load to keep the success-path bounded: read the header
        # (no trace data) to discover the axis trace name, then re-open
        # loading ONLY that single trace. ``build_simulation_summary``
        # needs the axis to populate ``range`` and ``point_count``; it
        # does NOT need V(*)/I(*) trace data. Loading "*" would
        # materialise every signal on every completion — fine for a
        # short .op, unbounded for a long .tran (Codex M3).
        header = RawRead(str(raw_file), traces_to_read=None, dialect=dialect)
        trace_names = header.get_trace_names()
        # Decide value-scan coverage by the ESTIMATED total sample count (axis
        # points × number of non-axis traces) — the real memory/time cost of
        # loading "*", not the axis length alone. At or below the budget, load
        # all traces and scan (every normal interactive run, including a
        # single-point operating point and the floating-node / 1e30 case worth
        # scanning); above it (a long .tran OR a wide node dump) stay axis-only
        # and let the surfacer record the skipped scan, bounding the worst-case
        # load. Reading the header counts avoids a throwaway open to size it.
        try:
            point_count = header.nPoints
        except Exception:
            point_count = 1
        trace_count = max(0, len(trace_names) - 1)  # exclude the axis
        if point_count * trace_count <= _VALUE_SCAN_SAMPLE_BUDGET:
            raw_read = RawRead(str(raw_file), traces_to_read="*", dialect=dialect)
            value_scan = "scan"
        else:
            axis_only = [trace_names[0]] if trace_names else None
            raw_read = RawRead(str(raw_file), traces_to_read=axis_only, dialect=dialect)
            value_scan = "skipped_large"
    except Exception as e:
        logger.warning(f"Could not parse raw file {raw_file}: {e}")
        # No raw — surface what diagnostics we can from the log alone.
        if log_file.exists():
            try:
                diagnostics = extract_log_diagnostics(log_file)
                result["warnings"] = diagnostics["warnings"]
                if diagnostics["errors"]:
                    result["errors"] = diagnostics["errors"]
            except Exception as log_e:
                logger.warning(f"Could not parse log file {log_file}: {log_e}")
        return result

    log_path = log_file if log_file.exists() else None
    summary = build_simulation_summary(
        raw_read, log_path, duration, requested=requested, value_scan=value_scan
    )
    # ``summary`` doesn't carry ``raw_file``/``log_file``; they're already
    # set in ``result`` above and survive ``update``.
    result.update(summary)

    # Diagnostics truncation (preserved from the legacy contract).
    for key in ("warnings", "errors"):
        items = result.get(key) or []
        if len(items) > _MAX_DIAGNOSTICS:
            result[f"{key}_truncated"] = len(items)
            result[key] = items[:_MAX_DIAGNOSTICS]

    return result


# Suffixes that spicelib's LTSpiceLogReader peels off into separate flat
# measurements alongside the parent .MEAS name (e.g. ``v_rms`` →
# ``v_rms_from`` and ``v_rms_to`` for FROM/TO window bounds, ``v_rms_at``
# for AT/WHEN). These are metadata, not measurements — we fold them into
# the parent's :class:`MeasurementEntry` and never surface them as their
# own entries.
_MEAS_METADATA_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_from", "range_from"),
    ("_to", "range_to"),
    ("_at", "at"),
)

# Patterns we sanitize out of LTspice .FOUR blocks before handing them to
# spicelib's LTSpiceLogReader. Upstream regex (``r"\d+.\d+"`` in
# ``ltsteps.py:354``) does not match ``-nan`` / ``inf`` and crashes with
# ``'NoneType' object has no attribute 'group'``. Replacing with ``0.0``
# preserves the rest of the log so .MEAS results, errors, and any other
# Fourier blocks survive.
# Pattern for FAIL'ed .MEAS results. Spicelib's LTSpiceLogReader filters
# these out of get_measure_names() — we re-extract them from the raw log
# text so callers can distinguish "did not trigger" from "did not parse"
# rather than seeing a silent absence.
_RE_MEAS_FAILED = re.compile(r"Measurement\s+\"([^\"]+)\"\s+FAIL[''`]?ed", re.IGNORECASE)

# AT/WHEN crossing line, e.g. ``tcross: V(out)=0.5  AT 0.000693147672285``.
# spicelib's LTSpiceLogReader matches the AT clause with a literal single-space
# `` at `` pattern (ltsteps.py — case-insensitive, but the spaces are literal),
# so a WHEN line padded with extra whitespace (a double space, or a tab) loses
# its crossing time: the value comes back as the trigger level with no ``at``.
# We re-extract it here, whitespace-tolerant, and fold it in when spicelib left
# ``at`` unset. ``.*?`` stops at the first ``=`` so the level is the value and
# the trailing number after AT is the crossing point.
_RE_MEAS_AT_LINE = re.compile(
    r"^\s*(?P<name>\w+):\s+.*?=\s*\S+\s+at\s+(?P<at>[-+]?[\d.][\d.eE+-]*)",
    re.IGNORECASE | re.MULTILINE,
)


_RE_FOURIER_NAN = re.compile(
    r"^(Total Harmonic Distortion|Partial Harmonic Distortion):\s*[-+]?(?:nan|inf)%?",
    re.IGNORECASE | re.MULTILINE,
)


def _sanitize_log_for_reader(content: str) -> str:
    """Replace ``-nan``/``inf`` THD/PHD lines with ``0.0%`` so spicelib parses."""

    def _sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}: 0.0%"

    return _RE_FOURIER_NAN.sub(_sub, content)


def _preprocess_ngspice_log(content: str) -> str | None:
    """Strip ngspice preamble so ``LTSpiceLogReader``'s regex can match.

    ngspice prepends ``Note: Compatibility modes selected: ...`` and
    possibly blank lines before the ``Circuit:`` line that spicelib
    expects near the top. Returns the trimmed content if a ``Circuit:``
    line was found, else ``None``.
    """
    idx = content.find("\nCircuit:")
    if idx == -1:
        idx = content.find("Circuit:")
        if idx == 0:
            return None  # already at the start — no preprocessing needed
        if idx == -1:
            return None
    return content[idx:].lstrip("\n")


def make_log_reader(log_path: Path) -> LTSpiceLogReader:
    """Build an LTSpiceLogReader, sanitizing known crash patterns on retry.

    Spicelib's Fourier-block parser crashes on ``-nan``/``inf`` THD values
    — common when the analysed signal is identically zero. We retry once
    with a sanitized copy in a temp file before giving up.

    Also preprocesses ngspice logs whose preamble trips up spicelib's
    start-of-file regex.
    """
    try:
        return LTSpiceLogReader(str(log_path))
    except ResultError:
        raise
    except Exception as first_err:
        try:
            content = log_path.read_text(errors="replace")
        except OSError as e:
            raise ResultError(f"Could not parse log file: {first_err}") from e

        # Try ngspice preamble stripping, then -nan sanitization, then both.
        candidates: list[str] = []
        preprocessed = _preprocess_ngspice_log(content)
        sanitized = _sanitize_log_for_reader(content)
        if preprocessed is not None:
            candidates.append(preprocessed)
            candidates.append(_sanitize_log_for_reader(preprocessed))
        if sanitized != content:
            candidates.append(sanitized)
        if not candidates:
            raise ResultError(f"Could not parse log file: {first_err}") from first_err

        for candidate in candidates:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=log_path.suffix,
                prefix=f"{log_path.stem}.sanitized.",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(candidate)
                tmp_path = Path(tmp.name)
            try:
                return LTSpiceLogReader(str(tmp_path))
            except Exception:
                continue
            finally:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
        raise ResultError(f"Could not parse log file: {first_err}") from first_err


def parse_measurements(
    log_path: Path, reader: LTSpiceLogReader | None = None
) -> MeasurementsOutput:
    """Parse .MEAS measurement results from simulation log file.

    The flat ``name_from`` / ``name_to`` / ``name_at`` keys that spicelib
    emits as side-effects of ``FROM``/``TO``/``AT`` arguments are folded
    into the parent measurement's :class:`MeasurementEntry` rather than
    surfaced as standalone entries.

    Args:
        log_path: Path to .log file
        reader: Optional pre-built LTSpiceLogReader (avoids re-parsing)

    Returns:
        :class:`MeasurementsOutput` with structured measurements.

    Raises:
        ResultError: If log file cannot be parsed
    """
    if reader is None:
        reader = make_log_reader(log_path)

    # Pull FAIL'ed measurements from the raw log text. Spicelib's reader
    # filters them out, so without this pass they appear as a silent
    # absence — indistinguishable from a measurement that wasn't parsed.
    failed_names: list[str] = []
    log_text = ""
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        # Preserve discovery order, drop duplicates (spicelib reports
        # FAIL'ed once per step, but the name is the same).
        seen: set[str] = set()
        for match in _RE_MEAS_FAILED.finditer(log_text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                failed_names.append(name)
    except OSError:
        pass

    # Drop ngspice's title-echo pseudo-measurement (see _NGSPICE_TITLE_PSEUDO_MEAS).
    measure_names = [
        n for n in reader.get_measure_names() if n.lower() != _NGSPICE_TITLE_PSEUDO_MEAS
    ]
    if not measure_names:
        # Surface any errors AND warnings from the log that explain why
        # measurements are missing. ngspice's "No .measure possible in batch
        # mode" is a *warning*-class diagnostic; without carrying it, callers
        # (e.g. measurement_stats) report "no diagnostics" while every other
        # tool surfaces the reason.
        diagnostics = extract_log_diagnostics(log_path)
        errors_list = diagnostics["errors"] or None
        warnings_list = diagnostics["warnings"] or None
        # Even when spicelib reports no measurements, FAIL'ed names still
        # need to show up in ``measurements`` (value=None) so consumers
        # don't see a silent absence.
        measurements: dict[str, MeasurementEntry] = {
            name: {"values": [None]} for name in failed_names
        }
        return {
            "measurements": measurements,
            "step_count": 0,
            "errors": errors_list,
            "warnings": warnings_list,
            "failed_measurements": failed_names,
        }

    def _coerce(values: list) -> list[float | None]:
        out: list[float | None] = []
        for val in values:
            if val is None or (isinstance(val, str) and val.upper() == "FAILED"):
                out.append(None)
            elif isinstance(val, complex):
                out.append(float(abs(val)))
            elif hasattr(val, "item") and not isinstance(val, str):
                out.append(float(val.item()))  # numpy scalar
            else:
                try:
                    out.append(float(val))
                except (TypeError, ValueError):
                    logger.warning("Measurement value %r is non-numeric; recording as None", val)
                    out.append(None)
        return out

    # Pass 1: split flat names into parent + metadata-suffix.
    measure_name_set = set(measure_names)
    parents: dict[str, list[float | None]] = {}
    metadata: dict[str, dict[str, float | list[float | None] | None]] = {}
    for name in measure_names:
        coerced = _coerce(reader.dataset.get(name.lower(), []))
        suffix_match = next(
            ((suffix, key) for suffix, key in _MEAS_METADATA_SUFFIXES if name.endswith(suffix)),
            None,
        )
        if suffix_match:
            suffix, meta_key = suffix_match
            parent_name = name[: -len(suffix)]
            if parent_name in measure_name_set:
                # Literal FROM/TO bounds are constants → collapse to a scalar.
                # TRIG/TARG marker times vary per step → keep the per-step list
                # so the user can correlate each rise time with the marker
                # times that produced it.
                non_none = [v for v in coerced if v is not None]
                if not non_none:
                    folded: float | list[float | None] | None = None
                elif all(v == non_none[0] for v in non_none) and len(coerced) == len(non_none):
                    folded = non_none[0]
                else:
                    folded = coerced
                metadata.setdefault(parent_name, {})[meta_key] = folded
                continue
        parents[name] = coerced

    # AT/WHEN crossing times spicelib's single-space ` at ` pattern missed on
    # whitespace-padded lines (see _RE_MEAS_AT_LINE): name (lowercased) → times,
    # folded into ``at`` below when spicelib left it unset. Scanned here — past
    # the no-measurement early return — so an empty run never pays for it.
    # ``None`` in the type matches the invariant MeasurementEntry["at"] list.
    at_overrides: dict[str, list[float | None]] = {}
    for at_match in _RE_MEAS_AT_LINE.finditer(log_text):
        try:
            at_overrides.setdefault(at_match.group("name").lower(), []).append(
                float(at_match.group("at"))
            )
        except ValueError:
            continue

    measurements: dict[str, MeasurementEntry] = {}
    for name, values in parents.items():
        entry: MeasurementEntry = {"values": values}
        for meta_key, meta_val in metadata.get(name, {}).items():
            entry[meta_key] = meta_val  # type: ignore[literal-required]
        # Backfill an AT/WHEN crossing time spicelib dropped (extra whitespace
        # around ``AT``). Constant across steps → scalar, else list, matching
        # the FROM/TO/at fold above.
        if "at" not in entry:
            override = at_overrides.get(name.lower())
            if override:
                entry["at"] = override[0] if all(v == override[0] for v in override) else override
        measurements[name] = entry

    step_count = len(next(iter(parents.values()))) if parents else 0

    # Merge FAIL'ed names that aren't already in measurements. Use
    # ``values: [None] * step_count`` so consumers can branch on it the
    # same way they do for any other None entry, and the per-step shape
    # matches successful measurements.
    for fname in failed_names:
        if fname in measurements:
            continue
        measurements[fname] = {"values": [None] * max(step_count, 1)}

    return {
        "measurements": measurements,
        "step_count": step_count,
        "errors": None,
        "warnings": None,
        "failed_measurements": failed_names,
    }


def parse_fourier_data(log_path: Path, reader: LTSpiceLogReader | None = None) -> list[dict]:
    """Extract Fourier analysis (.FOUR) results from log file.

    Args:
        log_path: Path to .log file
        reader: Optional pre-built LTSpiceLogReader (avoids re-parsing)

    Returns:
        List of dicts, each containing signal name, THD, fundamental frequency,
        and list of harmonics (number, frequency, magnitude, phase).
        All values are Python float.
    """
    if reader is None:
        try:
            reader = make_log_reader(log_path)
        except Exception:
            # If log parsing fails, return empty (graceful degradation)
            return []

    if not hasattr(reader, "fourier") or not reader.fourier:
        return []

    # ``reader.fourier`` is ``dict[signal, list[FourierData]]`` — one
    # FourierData entry per .step iteration (length 1 for unstepped runs).
    # We flatten and emit one result dict per FourierData.
    results = []
    try:
        for signal_name, fourier_list in reader.fourier.items():
            for fourier_data in fourier_list:
                thd = (
                    float(fourier_data.thd)
                    if getattr(fourier_data, "thd", None) is not None
                    else None
                )
                # Spicelib's FourierData doesn't have a `.fundamental_frequency`
                # field; the fundamental is the first harmonic's frequency.
                fundamental_freq = None
                harmonics_raw = getattr(fourier_data, "harmonics", None) or []
                if harmonics_raw:
                    first = harmonics_raw[0]
                    if hasattr(first, "frequency"):
                        fundamental_freq = float(first.frequency)

                harmonics = []
                for h in harmonics_raw:
                    harmonics.append(
                        {
                            "number": int(h.harmonic_number)
                            if hasattr(h, "harmonic_number")
                            else None,
                            "frequency": float(h.frequency) if hasattr(h, "frequency") else None,
                            "magnitude": (
                                float(h.fourier_component)
                                if hasattr(h, "fourier_component")
                                else None
                            ),
                            "phase": float(h.phase) if hasattr(h, "phase") else None,
                        }
                    )

                results.append(
                    {
                        "signal": signal_name,
                        "thd": thd,
                        "thd_unit": "%",  # thd is a percentage (LTspice "23.95%"), not a ratio
                        "fundamental_frequency": fundamental_freq,
                        "harmonics": harmonics,
                    }
                )
    except Exception:
        # Graceful degradation - return partial data if format is unexpected
        pass

    return results
