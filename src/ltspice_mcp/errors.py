"""Error hierarchy for ltspice-mcp server."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

_MAX_VALIDATION_ERRORS = 6


def compact_validation_error(
    exc: ValidationError | ValueError | TypeError,
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

    if field_owners is not None:
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

    ``code`` — the wire code for this class of failure. Every subclass declares
    its own, and it is the DEFAULT, not the last word: a handler that knows
    which stage failed may name the failure by that stage instead, because the
    stage is often the more useful fact (a result read that fails while a deck
    is being staged is a submission failure, not a result failure). More
    specific still is a code set on the INSTANCE at the raise site, which names
    that one failure — see :func:`raise_site_code`, the lookup handlers use so a
    class default cannot quietly outrank the stage they are reporting.

    The set of codes is public: adding one is fine, renaming or removing one is
    a client-visible change. ``tests/test_error_codes.py`` pins the vocabulary.
    """

    code: str = "internal_error"

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

    code = "path_denied"


class NetlistError(LTSpiceMCPError):
    """Invalid netlist or component reference."""

    code = "netlist_invalid"


class SymbolResolutionError(NetlistError):
    """A schematic was found but a file it refers to was not.

    Its own type because the discriminant is structural, not textual: the
    schematic itself opened, so the missing file is one of its dependencies —
    a symbol (``.asy``), a hierarchical sub-sheet, or a model library. Asking
    whether the underlying message mentioned ``.asy`` blamed the schematic for
    every dependency the third-party editor named some other way.
    """

    code = "symbol_unresolved"


class SimulationError(LTSpiceMCPError):
    """Simulation execution failed."""

    code = "simulation_failed"


class ResultError(LTSpiceMCPError):
    """Error reading simulation results."""

    code = "result_unreadable"


class JobNotFoundError(ResultError):
    """No job with the requested id exists in the job store."""

    code = "job_not_found"


class AnalysisDeadlineExceeded(ResultError):
    """A read of a result artifact ran past the deadline it was given.

    Its own type, not its wording, is what marks a result read as having run
    out of time: callers classify the failure with ``except
    AnalysisDeadlineExceeded``. Matching on the message instead would catch any
    other error that happens to say "exceeded" — a size cap, a case cap, or a
    path that merely contains the word.
    """

    code = "analysis_deadline"


class NoAxisError(ResultError):
    """The result has no sweep axis, so there is no point to query at.

    An operating-point raw stores one bias solution per step and no
    time/frequency column. The parser turns the third-party reader's several
    ways of saying that into this one type, so callers stop matching on the
    sentence "This RAW file does not have an axis".
    """

    code = "no_axis"


class LibraryError(LTSpiceMCPError):
    """Component library error (load, parse, or lookup failure)."""

    code = "library_error"


class BatchJobError(LTSpiceMCPError):
    """Batch job error (config not found, job not found, invalid config, etc.)."""

    code = "batch_job_error"


def raise_site_code(exc: BaseException) -> str | None:
    """Return the code an exception named at its own raise site, if any.

    Only an INSTANCE attribute counts. A class-level ``code`` (see
    :attr:`LTSpiceMCPError.code`) is the default for a type, and a handler that
    reports a failure by the stage it happened in must keep that name rather
    than inherit the type's; only a code chosen at the raise site is specific
    enough to outrank it.
    """
    code = exc.__dict__.get("code")
    return code if isinstance(code, str) else None
