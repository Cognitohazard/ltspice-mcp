"""Error hierarchy for ltspice-mcp server."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

_MAX_VALIDATION_ERRORS = 6


def compact_validation_error(
    exc: ValidationError | ValueError,
    *,
    field_owners: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Render validation failures without input echoes or documentation URLs."""
    if not isinstance(exc, ValidationError):
        return str(exc)

    entries: list[tuple[tuple[object, ...], str, str]] = []
    seen: set[tuple[tuple[object, ...], str, str]] = set()
    referral_fields: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        loc = tuple(error["loc"])
        error_type = error["type"]
        message = error["msg"]
        key = (loc, error_type, message)
        if key in seen:
            continue
        seen.add(key)
        entries.append(key)
        if (
            field_owners is not None
            and error_type == "extra_forbidden"
            and len(loc) == 1
            and isinstance(loc[0], str)
            and loc[0] in field_owners
            and loc[0] not in referral_fields
        ):
            referral_fields.append(loc[0])

    rendered = [
        f"{'.'.join(str(part) for part in loc) or '<root>'}: {message}"
        for loc, _, message in entries[:_MAX_VALIDATION_ERRORS]
    ]
    remaining = len(entries) - _MAX_VALIDATION_ERRORS
    if remaining > 0:
        rendered.append(f"… and {remaining} more")
    text = "; ".join(rendered) or "Validation failed"

    for field in referral_fields:
        owners = tuple(dict.fromkeys(field_owners[field]))
        if owners:
            text += f" Field {field!r} is accepted by {', '.join(owners)}."
    return text


class LTSpiceMCPError(Exception):
    """Base exception for all ltspice-mcp errors.

    ``suggestions`` — optional ranked candidate dicts surfaced by the MCP
    dispatch layer as ``structuredContent`` on the error response.

    ``show_hint`` — when False, the dispatch layer does NOT append the generic
    per-error-type hint. Set it on errors that already carry precise, actionable
    guidance (e.g. "use operating_point for .OP raws"), so the generic
    "verify with check_job / simulation_summary" hint doesn't misdirect.
    """

    def __init__(
        self,
        *args: object,
        suggestions: list[dict[str, Any]] | None = None,
        show_hint: bool = True,
    ) -> None:
        super().__init__(*args)
        self.suggestions: list[dict[str, Any]] = suggestions or []
        self.show_hint: bool = show_hint


class PathSecurityError(LTSpiceMCPError):
    """Path resolves outside allowed directories."""


class NetlistError(LTSpiceMCPError):
    """Invalid netlist or component reference."""


class SimulationError(LTSpiceMCPError):
    """Simulation execution failed."""


class ConvergenceError(SimulationError):
    """Time step too small / failed to converge."""


class SingularMatrixError(SimulationError):
    """Singular matrix — floating node or short circuit."""


class MissingModelError(SimulationError):
    """Referenced subcircuit or model not found."""


class ResultError(LTSpiceMCPError):
    """Error reading simulation results."""


class JobNotFoundError(ResultError):
    """No job with the requested id exists in the job store."""


class LibraryError(LTSpiceMCPError):
    """Component library error (load, parse, or lookup failure)."""


class BatchJobError(LTSpiceMCPError):
    """Batch job error (config not found, job not found, invalid config, etc.)."""
