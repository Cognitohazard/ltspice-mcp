"""Error hierarchy for ltspice-mcp server."""

import difflib
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar

from pydantic import ValidationError

_MAX_VALIDATION_ERRORS = 6


def compact_validation_error(
    exc: ValidationError | ValueError | TypeError,
    *,
    field_owners: Mapping[str, Sequence[str]] | None = None,
    nested: Mapping[str, Sequence[str]] | None = None,
    renamed: Mapping[str, str] | None = None,
    removed: Mapping[str, str] | None = None,
    accepted: Mapping[str, Sequence[str]] | None = None,
    branches: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> str:
    """Render validation failures without input echoes or documentation URLs.

    An unknown top-level field is referred to where it is taken: ``nested``
    maps a field to its place inside this tool's own arguments (``path`` to
    ``circuits[].path``), named first, and ``field_owners`` to the tools
    that take it at their top level. An unknown field at a place where this
    tool retired its name is referred to the one that replaced it, by
    ``renamed``, keyed by that place as a client writes it (``raw_file``,
    ``sources[].raw_path``; a union's branch name is no part of it, since a
    client never writes one), or told why it was removed, by ``removed``,
    keyed the same way. An unknown field with no such referral is
    answered with the
    names its place takes, from ``accepted`` (the place spelled as the
    reference table spells it: ``""`` for the tool's own arguments,
    ``circuits[]`` inside each circuit), the closest one first.

    ``branches`` maps each field that chooses a branch (``action``, ``op``) to
    each branch's fields. A missing one is answered with the values it takes,
    and the item's fields that no branch takes, which validation never
    reaches without a branch, are refused beside it.
    """
    if not isinstance(exc, ValidationError):
        return str(exc)
    renamed = renamed or {}
    removed = removed or {}
    nested = nested or {}
    field_owners = field_owners or {}
    accepted = accepted or {}
    branches = branches or {}

    #: The names of every union branch, which a location carries and a place does not.
    tags = frozenset(name for fields in branches.values() for name in fields)

    def referral(field: str, replaced_by: str | None = None, why: str | None = None) -> str:
        # A client that checks a call against its own copy of the tool's
        # arguments keeps sending a retired name while that copy lists it.
        stale = (
            f" If your copy of this tool's arguments still lists {field!r}, it is "
            "out of date: reload the tool's definition or reconnect this server."
        )
        if replaced_by is not None:
            return f" {field!r} is now {replaced_by.rpartition('.')[2]!r}.{stale}"
        if why is not None:
            return f" {field!r} was removed: {why}.{stale}"
        text = ""
        if nested.get(field):
            text += f" This tool takes {field!r} as {' or '.join(nested[field])}."
        owners = tuple(dict.fromkeys(field_owners.get(field, ())))
        if owners:
            text += f" Field {field!r} is accepted by {', '.join(owners)}."
        return text

    # The item's input is read for its field names only, never echoed.
    errors: list[tuple[tuple[object, ...], str, str]] = []
    #: The names an untagged item's place takes, by the location of each field
    #: it carries that no branch does.
    taken_at: dict[tuple[object, ...], tuple[str, ...]] = {}
    for error in exc.errors(include_url=False):
        loc = tuple(error["loc"])
        error_type = error["type"]
        message = error["msg"]
        ctx = error.get("ctx") or {}
        if error_type == "union_tag_invalid":
            loc, message = _tag_mismatch(loc, ctx, message)
        elif error_type == "union_tag_not_found" and _discriminator(ctx):
            discriminator = _discriminator(ctx)
            fields = branches.get(discriminator, {})
            errors.append((*_tag_missing(loc, discriminator, fields), error_type))
            if fields and isinstance(error.get("input"), Mapping):
                taken = (discriminator, *dict.fromkeys(n for f in fields.values() for n in f))
                for name in error["input"]:
                    if isinstance(name, str) and name not in taken:
                        taken_at[(*loc, name)] = taken
                        errors.append(
                            ((*loc, name), "Extra inputs are not permitted", "extra_forbidden")
                        )
            continue
        errors.append((loc, message, error_type))

    entries: list[tuple[tuple[object, ...], str]] = []
    seen: set[tuple[tuple[object, ...], str, str]] = set()
    referrals: list[str] = []
    for loc, message, error_type in errors:
        if (loc, message, error_type) in seen:
            continue
        seen.add((loc, message, error_type))
        entries.append((loc, message))
        field = loc[-1] if loc else None
        if error_type != "extra_forbidden" or not isinstance(field, str):
            continue
        written = _place(loc, tags)
        if written in renamed:
            sentence = referral(field, renamed[written])
        elif written in removed:
            sentence = referral(field, why=removed[written])
        elif len(loc) == 1 and (field in nested or field in field_owners):
            sentence = referral(field)
        else:
            sentence = _accepted_here(field, loc[:-1], accepted, taken_at.get(loc, ()))
        if sentence and sentence not in referrals:
            referrals.append(sentence)

    rendered = [
        f"{'.'.join(str(part) for part in loc) or '<root>'}: {message}"
        for loc, message in entries[:_MAX_VALIDATION_ERRORS]
    ]
    remaining = len(entries) - _MAX_VALIDATION_ERRORS
    if remaining > 0:
        rendered.append(f"… and {remaining} more")
    text = "; ".join(rendered) or "Validation failed"
    return text + ("." if referrals else "") + "".join(referrals)


def _place(loc: Sequence[object], tags: Collection[str] = ()) -> str:
    """A location as the reference table spells a field's place: ``("circuits",
    0)`` is ``circuits[]``, the tool's own arguments are ``""``.

    A part in ``tags`` that starts the location or follows a list index is the
    branch a union chose (``("ops", 0, "move", "dx")``), which a client does
    not write, so it is left out: ``ops[].dx``.
    """
    place = ""
    for index, part in enumerate(loc):
        if isinstance(part, int):
            place += "[]"
        elif part in tags and (index == 0 or isinstance(loc[index - 1], int)):
            continue
        else:
            place += f".{part}" if place else str(part)
    return place


def _accepted_here(
    field: str,
    loc: Sequence[object],
    accepted: Mapping[str, Sequence[str]],
    taken: Sequence[str] = (),
) -> str:
    """The names the place of an unknown ``field`` takes, the closest first.

    ``taken`` stands in where the reference table lists no place: the names
    the branches of an untagged item take, offered as the closest only.
    """
    place = _place(loc)
    names = list(accepted.get(place, ()))
    close = difflib.get_close_matches(field, names or list(taken), n=1)
    guess = f" Did you mean {close[0]!r} for {field!r}?" if close else ""
    if not names:
        return guess
    where = "This tool takes" if not place else f"{place} takes"
    return f"{guess} {where} {', '.join(names)}."


def _discriminator(ctx: Mapping[str, Any]) -> str:
    """The field that chooses a branch, as a union error's context names it."""
    return str(ctx.get("discriminator", "")).strip("'")


def _tag_missing(
    loc: tuple[object, ...], discriminator: str, fields: Mapping[str, Sequence[str]]
) -> tuple[tuple[object, ...], str]:
    """A missing field that chooses a branch, said at that field with the
    values it takes.

    pydantic says it as "Unable to extract tag using discriminator '…'",
    located at the union rather than at the missing field.
    """
    if not fields:
        return (*loc, discriminator), "Field required"
    return (*loc, discriminator), "Field required, one of " + ", ".join(map(repr, fields))


#: How much of a rejected branch name the rewritten message quotes.
_TAG_QUOTE_CHARS = 60


def _tag_mismatch(
    loc: tuple[object, ...], ctx: Mapping[str, Any], message: str
) -> tuple[tuple[object, ...], str]:
    """A branch name no branch has, said at the field that names the branch.

    pydantic says it as "Input tag '…' found using 'action' does not match any
    of the expected tags: …", located at the union rather than at the field,
    and quoting whatever the field held, an object included.
    """
    discriminator = _discriminator(ctx)
    expected = ctx.get("expected_tags")
    if not discriminator or not expected:
        return loc, message
    given = str(ctx.get("tag", ""))
    if len(given) > _TAG_QUOTE_CHARS:
        given = given[:_TAG_QUOTE_CHARS] + "…"
    return (*loc, discriminator), f"must be one of {expected}, not {given}"


class LTSpiceMCPError(Exception):
    """Base exception for all ltspice-mcp errors.

    ``suggestions`` — optional ranked candidate dicts surfaced by the MCP
    dispatch layer as ``structuredContent`` on the error response.

    ``show_hint`` — when False, the dispatch layer does not append the generic
    per-error-type hint. Set it on errors that already carry precise, actionable
    guidance (e.g. "use operating_point for .OP raws"), so the generic
    "verify with check_job / simulation_summary" hint doesn't misdirect.

    ``python_route`` — the message ends in a Python snippet written against the
    engine object ``api``. Where that snippet runs depends on which tools the
    session serves, which the raise site cannot see, so the MCP surfaces follow
    the message with :func:`python_route_text`. A Python API caller reads the
    message as is.

    ``code`` — the wire code for this class of failure. Every subclass declares
    its own, and it is the default, not the last word: a handler that knows
    which stage failed may name the failure by that stage instead, because the
    stage is often the more useful fact (a result read that fails while a deck
    is being staged is a submission failure, not a result failure). More
    specific still is a code set on the instance at the raise site, which names
    that one failure — see :func:`raise_site_code`, the lookup handlers use so a
    class default cannot quietly outrank the stage they are reporting.

    The set of codes is public: adding one is fine, renaming or removing one is
    a client-visible change. ``tests/test_error_codes.py`` pins the vocabulary.

    ``takes_family_hint`` — False on a type whose family's recovery hint would
    send the caller the wrong way: it says its own recovery in its message, or
    it is not the failure its family's hint is about. :func:`hint_for_type`
    stops at such a type.
    """

    code: str = "internal_error"
    takes_family_hint: ClassVar[bool] = True

    def __init__(
        self,
        *args: object,
        suggestions: list[dict[str, Any]] | None = None,
        show_hint: bool = True,
        python_route: bool = False,
    ) -> None:
        super().__init__(*args)
        self.suggestions: list[dict[str, Any]] = suggestions or []
        self.show_hint: bool = show_hint
        self.python_route: bool = python_route


class PathSecurityError(LTSpiceMCPError):
    """A path was refused: it lies outside the sandbox (``allowed_paths``),
    when one is set, or it cannot be resolved as written."""

    code = "path_denied"


class NetlistError(LTSpiceMCPError):
    """Invalid netlist or component reference."""

    code = "netlist_invalid"


class MultiplierRewriteStop(NetlistError):
    """ngspice stops on a deck: the ``m`` it appends to an ``X`` line in a
    subcircuit it rewrote reaches a subcircuit that takes no parameters
    (``hierarchy_expr.card_copies``)."""

    code = "multiplier_rewrite_stops"


class UnsupportedFileError(NetlistError):
    """A file of a kind the call does not take: one that is no circuit where a
    circuit is needed."""

    code = "unsupported_file"
    takes_family_hint = False


class SymbolResolutionError(NetlistError):
    """A schematic was found but a file it refers to was not.

    Its own type because the discriminant is structural, not textual: the
    schematic itself opened, so the missing file is one of its dependencies —
    a symbol (``.asy``) or a model library. Asking whether the underlying
    message mentioned ``.asy`` blamed the schematic for every dependency the
    third-party editor named some other way.
    """

    code = "symbol_unresolved"
    takes_family_hint = False


class SimulationError(LTSpiceMCPError):
    """Simulation execution failed."""

    code = "simulation_failed"


class ResultError(LTSpiceMCPError):
    """Error reading simulation results."""

    code = "result_unreadable"


class JobNotFoundError(ResultError):
    """No job with the requested id exists in the job store."""

    code = "job_not_found"


class OtherRawNotWritten(ResultError):
    """A read asked for an experiment case's other raw (an FRA's results, a
    transient's operating point) of a kind its run did not write."""

    code = "other_raw_not_written"
    takes_family_hint = False


class AnalysisDeadlineExceeded(ResultError):
    """A read of a result artifact ran past the deadline it was given.

    Its own type, not its wording, is what marks a result read as having run
    out of time: callers classify the failure with ``except
    AnalysisDeadlineExceeded``. Matching on the message instead would catch any
    other error that happens to say "exceeded" — a size cap, a case cap, or a
    path that merely contains the word.
    """

    code = "analysis_deadline"
    takes_family_hint = False


class NoAxisError(ResultError):
    """The result has no sweep axis, so there is no point to query at.

    An operating-point raw stores one bias solution per step and no
    time/frequency column. The parser turns the third-party reader's several
    ways of saying that into this one type, so callers stop matching on the
    sentence "This RAW file does not have an axis".
    """

    code = "no_axis"
    takes_family_hint = False


class LibraryError(LTSpiceMCPError):
    """Component library error (load, parse, or lookup failure)."""

    code = "library_error"


class BatchJobError(LTSpiceMCPError):
    """Batch job error (config not found, job not found, invalid config, etc.)."""

    code = "batch_job_error"


class RecipeInvalidError(LTSpiceMCPError):
    """An analysis recipe, or the request around it, that is not a valid
    ``analyze_results`` request.

    Raised where the whole call is refused for it: an analysis attached to
    ``run_experiments`` is checked at submission, before anything runs.
    ``analyze_results`` itself reports an invalid recipe as that recipe's
    failure, under the same code, and runs the others. Resending the same
    request fails the same way, so it is never retryable.
    """

    code = "recipe_invalid"


class InvalidCursorError(LTSpiceMCPError):
    """A page cursor did not decode, failed its integrity check, or was minted
    for another listing, view, request or result set, or for a file that has
    changed since.

    One type for every tool's cursors, so the same mistake carries the same
    code on every tool, and its recovery is the one that applies: send back
    the cursor that tool's own previous page returned, not check a job or
    lint a netlist.
    """

    code = "invalid_cursor"


# Error type → the recovery step every surface that reports it gives. Hints name
# only tools the consolidated surface exposes (the only profile since 0.6.0).
# PathSecurityError is handled separately: its hint is the session's sandbox
# guidance, which reads the sandbox as it stands at the call.
ERROR_HINTS: dict[type[LTSpiceMCPError], str] = {
    SimulationError: ("Use inspect with a capabilities query to verify simulator availability."),
    NetlistError: (
        "Use verify_circuit to lint the file, or inspect its components — "
        "or read the netlist directly."
    ),
    JobNotFoundError: (
        'Use jobs with action:"list" to see known jobs — the id may be '
        "mistyped, evicted, or from a previous server session."
    ),
    ResultError: (
        'Verify the run reached a terminal state with jobs (action:"status"), '
        "and read signals with analyze_results."
    ),
    LibraryError: (
        'Use inspect with a model query (mode:"search") to find the part in the '
        "simulator's own libraries, or add .lib/.include directives to the netlist."
    ),
    RecipeInvalidError: (
        "Correct the recipe and send the call again; inspect(kind='reference', "
        "query='<metric>') lists each recipe's fields."
    ),
    InvalidCursorError: (
        "Send back, unchanged, the cursor this tool's own previous page returned "
        "(next_cursor; for analyze_results also the next block's cursor), or leave "
        "it out to start from the first page."
    ),
}


def hint_for_type(err_type: type[LTSpiceMCPError]) -> str | None:
    """The recovery hint for an error type: its own entry in ``ERROR_HINTS``,
    else its nearest ancestor's.

    A subclass added to a family gets that family's recovery unless the
    subclass, or an ancestor below the one with the entry, sets
    ``takes_family_hint = False``.
    """
    for cls in err_type.__mro__:
        if cls.__dict__.get("takes_family_hint") is False:
            return None
        hint = ERROR_HINTS.get(cls)
        if hint is not None:
            return hint
    return None


def error_hint(exc: BaseException) -> str | None:
    """The recovery step for ``exc``, by :func:`hint_for_type`.

    One answer for every surface that reports a failure, whether it raises it
    (``server._invoke``) or carries it in its own envelope (a coverage row, a
    batch item, a jobs error), so one kind of failure gets one hint everywhere.
    None for a type with none, and for an error that carries precise guidance
    of its own (``show_hint=False``).
    """
    if not isinstance(exc, LTSpiceMCPError) or not exc.show_hint:
        return None
    return hint_for_type(type(exc))


def raise_site_code(exc: BaseException) -> str | None:
    """Return the code an exception named at its own raise site, if any.

    Only an instance attribute counts. A class-level ``code`` (see
    :attr:`LTSpiceMCPError.code`) is the default for a type, and a handler that
    reports a failure by the stage it happened in must keep that name rather
    than inherit the type's; only a code chosen at the raise site is specific
    enough to outrank it.
    """
    code = exc.__dict__.get("code")
    return code if isinstance(code, str) else None


def python_route_text(served: Collection[str]) -> str:
    """Where a Python snippet written against ``api`` runs, for a session
    serving the tools ``served``: ``run_code`` when it is served, the
    ``ltspice_mcp.api`` library when the operator turned it off, so the text
    never names a tool the client cannot see."""
    if "run_code" in served:
        return "run_code runs it with api in scope."
    return (
        "Run it in your own Python: from ltspice_mcp.api import Api; api = Api(working_dir=...)."
    )


def caller_message(exc: BaseException, served: Collection[str]) -> str:
    """``exc``'s message as an MCP caller reads it: followed by
    :func:`python_route_text` when it carries an ``api`` snippet."""
    message = str(exc)
    if getattr(exc, "python_route", False):
        return f"{message} {python_route_text(served)}"
    return message
