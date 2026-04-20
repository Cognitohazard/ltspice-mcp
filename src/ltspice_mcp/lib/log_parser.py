"""Log file (.log) parsing: error extraction, success summaries, measurements, Fourier.

For .raw file parsing (waveform data, statistics), see raw_parser.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

from spicelib.log.ltsteps import LTSpiceLogReader
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import ResultError

logger = logging.getLogger(__name__)


class MeasurementsOutput(TypedDict):
    """Return shape of :func:`parse_measurements`.

    ``errors`` is populated only on the empty-measurements path (the log
    had no .MEAS results and the parser surfaced why). Always present in
    the return value — ``None`` when the measurement parse succeeded.
    """

    measurements: dict[str, list[float | None]]
    step_count: int
    errors: list[str] | None

_MAX_DIAGNOSTICS = 50

# Error keywords to search for in log files (case-insensitive)
_ERROR_KEYWORDS = [
    "error",
    "fatal",
    "failed",
    "singular matrix",
    "convergence",
    "time step too small",
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
# Explicit warning prefix (LTspice uses both casings)
_RE_WARNING = re.compile(r"^(?:Warning|WARNING):", re.IGNORECASE)
# Bare convergence / runtime messages with no prefix.
# These are matched anchored to the start of a (stripped) line so we don't
# false-positive on phrases that merely *contain* one of these substrings
# (e.g. "the singular matrix decomposition succeeded").
_BARE_ERROR_PHRASES = [
    "singular matrix",
    "time step too small",
    "no convergence",
    "gmin/source stepping failed",
    "questionable use of curly braces",
]

# Missing .MODEL — appears in log as:
#   Error on line 2 : s1 n003 n001 n002 0 sw Unable to find definition of model "sw"
# (rarer GUI-dialog phrasing "Can't find definition of model" is also tolerated).
_RE_MISSING_MODEL = re.compile(
    r'(?:Unable to find|Can\'?t find) definition of model\s+"([^"]+)"',
    re.IGNORECASE,
)
# Missing .SUBCKT — appears in log as:
#   Fatal Error: Unknown subcircuit called in: xu1 n004 n001 vcc 0 lm741
# The missing subcircuit name is the LAST token of the instance line.
_RE_MISSING_SUBCKT = re.compile(
    r"Unknown subcircuit called in:\s+(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def missing_refs_from_text(text: str) -> list[str]:
    """Scan log text for missing .MODEL / .SUBCKT names, de-duplicated in order."""
    seen: set[str] = set()
    refs: list[str] = []

    for m in _RE_MISSING_MODEL.finditer(text):
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


def extract_log_diagnostics(log_path: Path) -> dict[str, list[str]]:
    """Extract structured warnings and errors from an LTspice log file.

    Detects all known LTspice diagnostic patterns:
    - filepath(line): message + source line + ^^^ caret blocks
    - "Error on line N:" component errors
    - "Fatal Error:" messages
    - "Warning:"/"WARNING:" prefixed lines
    - Bare convergence/runtime messages (e.g., "Singular matrix")

    Returns:
        {"warnings": [...], "errors": [...]} with human-readable strings.
    """
    warnings: list[str] = []
    errors: list[str] = []

    try:
        content = log_path.read_text()
    except Exception:
        return {"warnings": warnings, "errors": errors}

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
            errors.append("\n".join(block))
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

        # Warning: / WARNING:
        if _RE_WARNING.match(stripped):
            warnings.append(stripped)
            i += 1
            continue

        # Bare convergence / runtime phrases — anchored at start of line to
        # avoid false positives on lines that merely contain the phrase as a
        # substring (e.g., "the singular matrix decomposition succeeded").
        stripped_lower = stripped.lower()
        if any(stripped_lower.startswith(phrase) for phrase in _BARE_ERROR_PHRASES):
            errors.append(stripped)
            i += 1
            continue

        i += 1

    return {"warnings": warnings, "errors": errors}


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
        content = log_file.read_text()
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

        first_error = error_indices[0]
        start = max(0, first_error - 3)
        end = min(len(lines), first_error + 7)

        next_error = None
        for idx in error_indices[1:]:
            if idx > first_error + 7:
                next_error = idx
                break

        if next_error is not None:
            end = min(end, next_error)

        if end - start > max_lines:
            end = start + max_lines

        excerpt = lines[start:end]

        if start > 0:
            excerpt.insert(0, "...")
        if end < len(lines):
            excerpt.append("...")

        return "\n".join(excerpt)

    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return f"(Error reading log file: {e})"


def parse_success_summary(raw_file: Path, log_file: Path, duration: float) -> dict:
    """Parse simulation success summary from raw and log files.

    Extracts metadata about the simulation including type, trace names,
    step count, and warnings.

    Args:
        raw_file: Path to .raw simulation output file
        log_file: Path to .log simulation log file
        duration: Simulation duration in seconds

    Returns:
        Dictionary with keys:
        - sim_type: Simulation type (e.g., "Transient Analysis")
        - duration: Duration in seconds
        - step_count: Number of parameter steps (1 for non-stepped)
        - warnings: List of warning messages (first 5 only)
        - trace_names: List of available signal/trace names
        - raw_file: Path to raw file (as string)
        - log_file: Path to log file (as string)

        Returns partial data on parse errors (graceful degradation).
    """
    result = {
        "sim_type": "Unknown",
        "duration": duration,
        "step_count": 1,
        "warnings": [],
        "trace_names": [],
        "raw_file": str(raw_file),
        "log_file": str(log_file),
    }

    try:
        raw_read = RawRead(str(raw_file), traces_to_read=None)

        result["trace_names"] = raw_read.get_trace_names()

        try:
            sim_type = raw_read.get_raw_property("Plotname")
            if sim_type:
                result["sim_type"] = sim_type
        except Exception:
            pass

        try:
            n_steps = raw_read.get_steps()
            if n_steps is not None:
                result["step_count"] = len(n_steps)
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Could not parse raw file {raw_file}: {e}")

    if log_file.exists():
        try:
            diagnostics = extract_log_diagnostics(log_file)
            warnings = diagnostics["warnings"]
            if len(warnings) > _MAX_DIAGNOSTICS:
                result["warnings"] = warnings[:_MAX_DIAGNOSTICS]
                result["warnings_truncated"] = len(warnings)
            else:
                result["warnings"] = warnings
            errors = diagnostics["errors"]
            if errors:
                if len(errors) > _MAX_DIAGNOSTICS:
                    result["errors"] = errors[:_MAX_DIAGNOSTICS]
                    result["errors_truncated"] = len(errors)
                else:
                    result["errors"] = errors
        except Exception as e:
            logger.warning(f"Could not parse log file {log_file}: {e}")

    return result


def parse_measurements(
    log_path: Path, reader: LTSpiceLogReader | None = None
) -> MeasurementsOutput:
    """Parse .MEAS measurement results from simulation log file.

    Args:
        log_path: Path to .log file
        reader: Optional pre-built LTSpiceLogReader (avoids re-parsing)

    Returns:
        Dictionary with measurements dict (name -> list of values) and step_count.
        Values are Python float (or None for failed measurements).

    Raises:
        ResultError: If log file cannot be parsed
    """
    if reader is None:
        try:
            reader = LTSpiceLogReader(str(log_path))
        except Exception as e:
            raise ResultError(f"Could not parse log file: {e}") from e

    measure_names = reader.get_measure_names()
    if not measure_names:
        # Surface any errors from the log that explain why measurements are missing
        diagnostics = extract_log_diagnostics(log_path)
        errors_list = diagnostics["errors"] or None
        return {"measurements": {}, "step_count": 0, "errors": errors_list}

    measurements: dict[str, list[float | None]] = {}
    for name in measure_names:
        values = reader.dataset.get(name.lower(), [])  # dataset uses lowercase keys
        python_values = []
        for val in values:
            if val is None or (isinstance(val, str) and val.upper() == "FAILED"):
                python_values.append(None)
            elif isinstance(val, complex):
                python_values.append(float(abs(val)))
            elif hasattr(val, "item") and not isinstance(val, str):
                python_values.append(float(val.item()))  # numpy scalar
            else:
                # Try to coerce to float; on failure record as None (failed
                # measurement) instead of crashing the whole call.
                try:
                    python_values.append(float(val))
                except (TypeError, ValueError):
                    logger.warning(
                        f"Measurement '{name}' has non-numeric value {val!r}; recording as None"
                    )
                    python_values.append(None)
        measurements[name] = python_values

    step_count = len(measurements[measure_names[0]]) if measurements else 0

    return {"measurements": measurements, "step_count": step_count, "errors": None}


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
            reader = LTSpiceLogReader(str(log_path))
        except Exception:
            # If log parsing fails, return empty (graceful degradation)
            return []

    if not hasattr(reader, "fourier") or not reader.fourier:
        return []

    results = []
    try:
        for signal_name, fourier_data in reader.fourier.items():
            thd = None
            if hasattr(fourier_data, "thd"):
                thd = float(fourier_data.thd) if fourier_data.thd is not None else None

            fundamental_freq = None
            if hasattr(fourier_data, "fundamental_frequency"):
                fundamental_freq = (
                    float(fourier_data.fundamental_frequency)
                    if fourier_data.fundamental_frequency is not None
                    else None
                )

            harmonics = []
            if hasattr(fourier_data, "harmonics") and fourier_data.harmonics:
                for harmonic in fourier_data.harmonics:
                    harm_dict = {
                        "number": int(harmonic.number) if hasattr(harmonic, "number") else None,
                        "frequency": (
                            float(harmonic.frequency) if hasattr(harmonic, "frequency") else None
                        ),
                        "magnitude": (
                            float(harmonic.magnitude) if hasattr(harmonic, "magnitude") else None
                        ),
                        "phase": float(harmonic.phase) if hasattr(harmonic, "phase") else None,
                    }
                    harmonics.append(harm_dict)

            results.append(
                {
                    "signal": signal_name,
                    "thd": thd,
                    "fundamental_frequency": fundamental_freq,
                    "harmonics": harmonics,
                }
            )
    except Exception:
        # Graceful degradation - return partial data if format is unexpected
        pass

    return results
