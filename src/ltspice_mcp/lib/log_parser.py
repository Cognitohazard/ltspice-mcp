"""Log file parsing and error context extraction."""

import logging
from pathlib import Path

from spicelib.raw.raw_read import RawRead

logger = logging.getLogger(__name__)

# Error keywords to search for in log files (case-insensitive)
ERROR_KEYWORDS = [
    "error",
    "fatal",
    "failed",
    "singular matrix",
    "convergence",
    "time step too small",
    "missing",
    "can't find",
]


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
        lines = [line.rstrip() for line in content.splitlines()]  # Strip trailing whitespace

        if not lines:
            return "(Empty log file)"

        # Search for error indicators (case-insensitive)
        error_indices = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ERROR_KEYWORDS):
                error_indices.append(i)

        if not error_indices:
            # No specific errors found - return last N lines
            start_line = max(0, len(lines) - max_lines)
            excerpt = lines[start_line:]
            if start_line > 0:
                excerpt.insert(0, "...")
            return "\n".join(excerpt)

        # Return context around first error (3 lines before, 7 lines after)
        first_error = error_indices[0]
        start = max(0, first_error - 3)
        end = min(len(lines), first_error + 7)

        # Find next error region to avoid including too much
        next_error = None
        for idx in error_indices[1:]:
            if idx > first_error + 7:
                next_error = idx
                break

        if next_error is not None:
            end = min(end, next_error)

        # Cap at max_lines
        if end - start > max_lines:
            end = start + max_lines

        excerpt = lines[start:end]

        # Add continuation indicators
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

    # Parse raw file for metadata (headers only - no waveform data)
    try:
        raw_read = RawRead(str(raw_file), traces_to_read=None)

        # Extract trace names
        result["trace_names"] = raw_read.get_trace_names()

        # Extract simulation type
        try:
            sim_type = raw_read.get_raw_property("Plotname")
            if sim_type:
                result["sim_type"] = sim_type
        except Exception:
            # Fallback: try get_plot_name() method if available
            try:
                plot_name = getattr(raw_read, "get_plot_name", lambda: None)()
                if plot_name:
                    result["sim_type"] = plot_name
            except Exception:
                pass

        # Get step count
        try:
            n_steps = raw_read.get_steps()
            if n_steps is not None:
                result["step_count"] = n_steps
        except Exception:
            # Default to 1 if step count unavailable
            pass

    except Exception as e:
        logger.warning(f"Could not parse raw file {raw_file}: {e}")
        # Continue with partial data

    # Parse log file for warnings
    if log_file.exists():
        try:
            log_content = log_file.read_text()
            warnings = []
            for line in log_content.splitlines():
                if "warning" in line.lower():
                    warnings.append(line.strip())
                    if len(warnings) >= 5:  # Limit to first 5 warnings
                        break
            result["warnings"] = warnings
        except Exception as e:
            logger.warning(f"Could not parse log file {log_file}: {e}")
            # Continue with empty warnings list

    return result
