"""Shared utilities for tool handlers."""

import base64
import contextlib
import contextvars
import hashlib
import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict, get_args, get_origin

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import PathSecurityError, SimulationError
from ltspice_mcp.lib import atomic_write_bytes, response_budget

# Re-exported façade names: the strict Pydantic base lives in ``lib`` (models
# below the tool layer declare models too), and the tool modules reach it here.
from ltspice_mcp.lib.models import StrictModel as StrictModel
from ltspice_mcp.lib.netlist_graph import IncludeResolver
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.raster import DEFAULT_SCALE, RenderedImage, render_image

# Re-exported, not defined here: both netlist injections live in the runner
# layer so the experiment coordinator (lib/) can call them without importing
# tools/, and the tool handlers keep reaching them through this module as before.
from ltspice_mcp.lib.runner_base import inject_logopinfo as inject_logopinfo
from ltspice_mcp.lib.runner_base import (
    inject_ngspice_control_write as inject_ngspice_control_write,
)
from ltspice_mcp.lib.schematic_renderer import render_svg
from ltspice_mcp.lib.schematic_scene import Scene, SymbolResolver, default_stock_paths
from ltspice_mcp.lib.simulator import no_simulator_message, simulator_library_roots
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._schema import (
    ToolInput,
    build_input_schema,
    schema_from_typeddict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response helpers — standardize tool output format
#
# All helpers return types.CallToolResult, the MCP protocol's canonical
# response type.  text_response() returns text-only (for confirmations).
# format_response() returns both human-readable text content AND structured
# data via structuredContent (for data-returning tools).
# ---------------------------------------------------------------------------


def text_response(text: str) -> types.CallToolResult:
    """Return a text-only CallToolResult (confirmations, simple messages)."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
    )


def result_text(result: types.CallToolResult, *, joined: bool = False) -> str:
    """The text channel of a ``CallToolResult``, for relaying a sub-handler's
    message out of a dispatcher.

    Returns the first text block verbatim. ``joined`` merges every text block
    onto one stripped line instead, for callers that fold the text into a
    message of their own rather than re-rendering it.
    """
    blocks = [block.text for block in result.content if isinstance(block, types.TextContent)]
    if joined:
        return " ".join(blocks).strip()
    return blocks[0] if blocks else ""


# Most named key paths one scrub warning lists — a fully-NaN trace array
# should produce one warning naming a few paths plus a count, not thousands.
_SCRUB_NAMED_PATHS = 10


def _scrub_non_finite(obj: Any, path: str, stats: dict[str, Any]) -> Any:
    """Replace non-finite floats with None, copy-on-write, recording key paths.

    Returns ``obj`` itself when nothing changed so unchanged payloads cost no
    copies. ``stats`` accumulates ``total`` hits and the first few ``named``
    paths.
    """
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        stats["total"] += 1
        if len(stats["named"]) < _SCRUB_NAMED_PATHS:
            stats["named"].append(path or "<root>")
        return None
    if isinstance(obj, dict):
        out: dict | None = None
        for k, v in obj.items():
            nv = _scrub_non_finite(v, f"{path}.{k}" if path else str(k), stats)
            if nv is not v:
                if out is None:
                    out = dict(obj)
                out[k] = nv
        return out if out is not None else obj
    if isinstance(obj, list):
        out_l: list | None = None
        for i, v in enumerate(obj):
            nv = _scrub_non_finite(v, f"{path}[{i}]", stats)
            if nv is not v:
                if out_l is None:
                    out_l = list(obj)
                out_l[i] = nv
        return out_l if out_l is not None else obj
    return obj


def sanitize_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Null out non-finite floats and surface the substitution as a warning.

    NaN/Inf are not JSON: pydantic silently serializes them as null on the
    wire while the text channel prints "nan" — the two channels contradict
    each other exactly on degenerate results. Per the emit-a-null-over-a-
    meaningless-number rule (lib/result_observations.py), substitute null
    OURSELVES and say so, naming the affected keys, so the substitution is a
    surfaced fact instead of a serializer accident.
    """
    stats: dict[str, Any] = {"total": 0, "named": []}
    scrubbed = _scrub_non_finite(data, "", stats)
    if not stats["total"]:
        return data
    unnamed = stats["total"] - len(stats["named"])
    note = (
        f"Non-finite values (NaN/Inf) at {', '.join(stats['named'])}"
        + (f" and {unnamed} more" if unnamed else "")
        + " were replaced with null; the underlying samples are not finite."
    )
    # Any hit already forced a fresh top-level dict via copy-on-write.
    existing = scrubbed.get("warnings")
    scrubbed["warnings"] = [*existing, note] if isinstance(existing, list) else [note]
    return scrubbed


def json_response(data: Any) -> types.CallToolResult:
    """Return data as JSON text + structuredContent."""
    if isinstance(data, dict):
        data = sanitize_payload(data)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=2, allow_nan=False))],
        structured_content=data,
    )


def format_response(
    text: str,
    data: Mapping[str, Any],
    fmt: str | None = None,
) -> types.CallToolResult:
    """Return a CallToolResult with text content and structuredContent.

    Always populates structuredContent for programmatic access. Accepts any
    mapping so TypedDict return values from lib functions pass through
    without cast/copy. The format param controls the text representation:
    - "json": text is JSON-formatted (for clients that parse text)
    - "text" or None: text is human-readable (default)

    Self-sufficiency contract: structured-aware clients (Claude Code included)
    render ONLY structuredContent when it is present and drop the text channel
    entirely, so ``data`` must carry everything the caller needs to act on.
    Any caller-guidance composed into ``text`` (hints, referrals, recovery
    steps, caveats) must be mirrored into ``data`` — conventionally an
    optional ``hint`` key declared in the tool's output_schema. The text
    channel is presentation only.
    """
    # MCP SDK's CallToolResult wants a plain dict for structuredContent;
    # TypedDicts ARE plain dicts at runtime, but wrap defensively so
    # non-dict mappings (rare, but cheap to support) also work.
    payload: dict[str, Any] = dict(data) if not isinstance(data, dict) else data
    if fmt == "json":
        return json_response(payload)  # sanitizes internally — don't walk twice
    payload = sanitize_payload(payload)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=payload,
    )


def image_response(
    image: RenderedImage,
    text: str,
    data: Mapping[str, Any] | None = None,
) -> types.CallToolResult:
    """Return a rendered image plus the metadata describing it.

    A raster is carried as an MCP image block, which is what a model actually
    looks at. Vector output is carried as text instead: SVG is markup, and
    ``image/svg+xml`` is not reliably rendered by clients, so sending it as an
    image block would produce a blank space where a picture should be.

    Self-sufficiency contract (as for :func:`format_response`): a client that
    renders only ``structuredContent`` must still be able to act, so the image's
    own description — format, scale actually applied, byte size, and any note
    explaining a degraded result — is always mirrored there. That is what tells
    a caller it asked for a PNG and received SVG, without decoding anything.
    """
    payload: dict[str, Any] = dict(data or {})
    if "image" in payload:
        raise ValueError(
            "image_response owns the 'image' key in structuredContent; the "
            "caller's data must not set it (rename the caller's key)"
        )
    payload["image"] = image.to_dict()
    payload = sanitize_payload(payload)

    content: list[Any] = []
    if image.is_raster:
        content.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(image.data).decode("ascii"),
                mime_type=image.mime_type,
            )
        )
    else:
        content.append(types.TextContent(type="text", text=image.data.decode("utf-8")))
    content.append(types.TextContent(type="text", text=text))

    return types.CallToolResult(content=content, structured_content=payload)


# ---------------------------------------------------------------------------
# Shared tool schema fragments and annotations
# ---------------------------------------------------------------------------

# Client-visible description of the shared ``format`` input param — one string,
# reused by every tool input model that exposes the param.
FORMAT_DESCRIPTION = (
    "Response format: 'json' for structured data, 'text' for human-readable "
    "(default; both carry the same structured content)"
)

# Output-schema fragment for the optional ``hint`` key: caller guidance
# mirrored from the text channel (see format_response's self-sufficiency
# contract). Sites needing a custom description inline their own dict.
HINT_SCHEMA: dict[str, str] = {"type": "string"}

# Free-text measurement caveats (see the observations-vs-warnings rule in
# lib/result_observations.py).
WARNINGS_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

# Fuzzy library matches for unresolved model/subcircuit references, keyed by
# the missing ref: ``{ref: [{name, score, source_path}, ...]}`` (produced by
# services.suggestions_from_errors / attach_suggestions_to_failure).
SUGGESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
                "source_path": {"type": "string"},
            },
        },
    },
}

PIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "dir": {"type": "string"},
        "order": {"type": "integer"},
    },
}

BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
}

# Structured advisories emitted by mutating .asc handlers after a successful
# op. ``message`` is always present and human-readable; the other keys
# depend on ``kind``. New kinds extend ``VALIDATION_WARNING_KINDS`` and the
# schema enum together so producers and consumers stay in lockstep.
VALIDATION_WARNING_KINDS: tuple[str, ...] = (
    "floating_pin",
    "duplicate_wire",
    "dangling_label",
    "label_over_component",
)

VALIDATION_WARNINGS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(VALIDATION_WARNING_KINDS)},
            "message": {"type": "string"},
            "ref": {"type": "string"},
            "pin": {"type": "string"},
            "label": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "from": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "to": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "count": {"type": "integer"},
        },
        "required": ["kind", "message"],
    },
}

# Structured .MEAS parse errors surfaced from extract_log_diagnostics.
# Each entry is the offending directive plus an optional fix suggestion
# pulled from the spice_validator blocklist.
MEAS_ERRORS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "directive": {"type": "string"},
            "raw_block": {"type": "string"},
            "suggestion": {"type": ["string", "null"]},
        },
    },
}

# Surfaced result observations — see lib/result_observations.py and the
# "Result trust: report facts, do not rate them" in CLAUDE.md. A surfacing
# layer: facts lifted into view for the consuming agent to judge, never a trust
# verdict. ``severity`` is present only on ``relay`` items (the simulator's own
# classification); ``value``/``reconciliation``/``coverage`` items omit it.
OBSERVATIONS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": ["relay", "reconciliation", "value", "coverage"],
            },
            "detail": {"type": "string"},
            "severity": {"type": "string"},
            "evidence": {"type": "object"},
        },
        "required": ["code", "kind", "detail"],
    },
}

# Parsed .MEAS results, keyed by measurement name. Shared by run_simulation,
# check_job, and simulation_summary so the WHEN/AT field semantics are
# described identically everywhere. For a plain value measurement ``values``
# holds the scalar(s); for a WHEN/AT point measurement ``values`` is the
# constant trigger LEVEL and the crossing time/frequency lives in ``at`` —
# the descriptions below are the only place that distinction is self-evident
# to a client reading the schema.
MEASUREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": ["number", "null"]},
                "description": (
                    "Per-.step scalar(s). For a WHEN/AT measurement this is the "
                    "constant trigger level, not the crossing point; read 'at' "
                    "for the crossing time/frequency."
                ),
            },
            # Scalar when the bound is constant across .step iterations; list
            # (one entry per step) when it varies (e.g. TRIG/TARG marker times).
            "range_from": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
            },
            "range_to": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
            },
            "at": {
                "type": ["number", "array", "null"],
                "items": {"type": ["number", "null"]},
                "description": (
                    "Crossing time (.tran) or frequency (.ac) for a WHEN/AT point "
                    "measurement. This is the answer for a WHEN rise-time/crossing "
                    "query; 'values' holds the constant level. Null for plain "
                    "value measurements."
                ),
            },
        },
        "required": ["values"],
    },
}


def format_meas_errors(meas_errors: list[dict[str, Any]]) -> list[str]:
    """Render structured .MEAS errors for the text-format response.

    Returns the lines (no trailing blank); callers append to their own
    line list. Empty input returns an empty list so callers don't need
    to guard.
    """
    if not meas_errors:
        return []
    lines = [f".MEAS errors ({len(meas_errors)}):"]
    for me in meas_errors:
        lines.append(f"  Directive: {me['directive']}")
        if me.get("suggestion"):
            lines.append(f"    Suggestion: {me['suggestion']}")
    return lines


def format_observations(observations: list[dict[str, Any]]) -> list[str]:
    """Render surfaced result observations for a text-format response.

    Relay observations are omitted — they already print in the Errors section —
    so this shows the new facts (reconciliation/value/coverage). Returns the
    lines (no trailing blank); empty / all-relay input returns ``[]``. The full
    list (including relay) always rides in structuredContent.
    """
    surfaced = [o for o in observations if o.get("kind") != "relay"]
    if not surfaced:
        return []
    lines = ["Observations (facts to weigh, not a verdict):"]
    lines.extend(f"  [{o.get('kind')}] {o.get('detail')}" for o in surfaced)
    return lines


# ---------------------------------------------------------------------------
# The shared response envelope
#
# Every consolidated tool's payload carries these five keys, and they mean the
# same thing on all of them. Defined once here so agreement is a property of
# the construction rather than of a test that compares six hand-written
# schemas.
# ---------------------------------------------------------------------------

#: The ratified call-level outcome vocabulary, in escalating order.
CONTRACT_OUTCOMES: tuple[str, ...] = ("complete", "partial", "failed", "in_progress")

CallOutcome = Literal["complete", "partial", "failed", "in_progress"]


class Envelope(TypedDict, total=False):
    """The five keys shared by every consolidated tool's payload.

    ``outcome`` — the call-level verdict, from ``CONTRACT_OUTCOMES``, decided
    by ``outcome_of``.

    ``failures`` — what did not work. One channel, but deliberately NOT one row
    shape: a verify failure names the stage that failed, and a receipt failure
    names the case that failed. Each tool declares its own row schema; what is
    shared is that the channel exists, is an array, and is separate from the
    two below.

    ``observations`` — is the data trustworthy: relayed simulator errors,
    coverage gaps, provenance facts. Structured on the tools that have a
    structured vocabulary (``OBSERVATIONS_SCHEMA``), free text on the rest.

    ``warnings`` — did this measurement assume something: a clamped window, an
    unparseable deck diffed as empty. Free text, always actionable.

    ``hint`` — the one next step, mirrored from the text channel because a
    structured-aware client renders only ``structuredContent``.

    ``observations`` and ``warnings`` are never merged; see the
    surface-don't-judge rules in ``lib/result_observations.py``.
    """

    outcome: CallOutcome
    failures: list[Any]
    observations: list[Any]
    warnings: list[str]
    hint: str


#: The envelope's keys, in declaration order. Derived from the type so the
#: contract battery reads one declaration instead of restating the list.
ENVELOPE_KEYS: tuple[str, ...] = tuple(Envelope.__annotations__)

#: The three fact channels, which stay separate on every tool that has them.
ENVELOPE_CHANNELS: tuple[str, ...] = ("failures", "observations", "warnings")


def outcome_of(
    failures: Any,
    *,
    partial: bool = False,
    in_progress: bool = False,
    delivered: bool = True,
) -> CallOutcome:
    """The one call-level outcome rule.

    ``failures`` is anything truthy-when-non-empty (a list, a count, a bool).

    * ``in_progress`` — the work has not reached a terminal state, so nothing
      else is decided yet. It outranks every other signal.
    * ``failed`` — something failed AND nothing came back with it. Pass
      ``delivered=False`` from a surface where a failure means the whole call
      produced nothing; the default is the read/batch case, where a failed item
      sits beside items that answered.
    * ``partial`` — something failed, or the caller-visible shortfall in
      ``partial`` was recorded, but results came back too.
    * ``complete`` — nothing failed and nothing fell short.

    The shortfalls each tool counts as ``partial`` are its own — a finding of
    error severity, a comparison mismatch, a truncated page, a run that never
    produced — so they arrive as one already-decided flag rather than as a
    growing pile of tool-specific branches in here.
    """
    if in_progress:
        return "in_progress"
    has_failures = bool(failures)
    if has_failures and not delivered:
        return "failed"
    if has_failures or partial:
        return "partial"
    return "complete"


def comparison_mismatch(comparison: Mapping[str, Any] | None) -> bool:
    """Anything short of a positive match — a real difference OR no verdict at all.

    Deliberately not ``not equivalent``: only ``True`` is a clean result, so a null
    verdict (the compared side could not be exported or parsed) keeps the outcome
    off ``complete`` instead of falling through it. Shared, because
    ``verify_circuit`` and ``edit_schematic`` both compare against a reference and
    a caller cannot be told the same mismatch is a shortfall on one and a clean
    result on the other. ``None`` means no comparison was asked for, which is no
    shortfall.
    """
    if comparison is None:
        return False
    return comparison.get("equivalent") is not True


def outcome_schema(*outcomes: str) -> dict[str, Any]:
    """The ``outcome`` property, restricted to the outcomes a tool can reach.

    A read that cannot fail the whole call never returns ``failed``, and a tool
    with no durable job never returns ``in_progress``; declaring the values a
    tool can actually produce is what makes the enum worth reading. Every value
    must come from ``CONTRACT_OUTCOMES``.
    """
    chosen = outcomes or CONTRACT_OUTCOMES
    stray = [value for value in chosen if value not in CONTRACT_OUTCOMES]
    if stray:
        raise ValueError(f"outcome(s) {stray} are not in the ratified vocabulary")
    return {"type": "string", "enum": list(chosen)}


OUTCOME_SCHEMA: dict[str, Any] = outcome_schema()


def failures_schema(row: dict[str, Any]) -> dict[str, Any]:
    """The ``failures`` channel over one tool's failure-row shape."""
    return {"type": "array", "items": row}


# One lint/check finding, wherever findings are reported. ``at`` locates it as
# precisely as the checked artifact allows: a netlist has a file and a line, a
# schematic also has coordinates, and only ``file`` is guaranteed.
FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_id": {"type": "string"},
        "severity": {"type": "string"},
        "ok": {"type": "boolean"},
        "evidence": {},
        "at": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "line": {"type": "integer"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["file"],
        },
        "subject": {"type": "string"},
    },
    "required": ["rule_id", "severity", "ok", "evidence", "at", "subject"],
}

#: The keys every offset page declares — see ``lib/pagination.page``, which
#: builds them.
PAGE_REQUIRED: list[str] = ["items", "total", "returned", "truncated", "next_cursor"]


def page_schema(
    items: dict[str, Any] | None = None,
    **extra_properties: dict[str, Any],
) -> dict[str, Any]:
    """The offset-page object: the five shared keys plus a tool's own additions.

    ``items`` overrides the item-array schema (a tool with a narrower row
    passes its own fragment). ``extra_properties`` are declared but not
    required, so a reader of any page can rely on the five without knowing
    which tool produced it.
    """
    return {
        "type": "object",
        "properties": {
            "items": items
            if items is not None
            else {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "returned": {"type": "integer"},
            "truncated": {"type": "boolean"},
            **extra_properties,
            "next_cursor": {"type": ["string", "null"]},
        },
        "required": list(PAGE_REQUIRED),
    }


# ---------------------------------------------------------------------------
# Shared argument models
#
# "Draw this sheet" and "compare it against that netlist" are the same two
# requests wherever they are asked, so both tools that ask them take the same
# two models. A tool that can do more than the shared model describes
# SUBCLASSES it (verify's render also chooses a delivery and whether to skip
# the checks) rather than growing a parallel spelling — so every field on the
# base means the same thing on every tool, and a field a tool cannot honour is
# not advertised there at all.
# ---------------------------------------------------------------------------


class RenderPolicy(StrictModel):
    """How to draw a schematic: the choices any renderer here can honour."""

    format: Literal["png", "svg"] = Field(
        default="png",
        description=(
            "PNG (lossless, what a model looks at) needs the optional 'raster' "
            "extra; without it the render degrades to SVG and says so."
        ),
    )
    scale: float = Field(
        default=DEFAULT_SCALE,
        ge=0.5,
        le=4.0,
        description=(
            "Render scale. Image token cost tracks pixel area, so halving the "
            "scale costs about a quarter as much."
        ),
    )
    max_pixels: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Cap the rendered pixel area. A PNG larger than this is re-rendered at "
            "a reduced scale that fits, and 'downscaled' is set. Bounds inline cost."
        ),
    )


class CompareSpec(StrictModel):
    """What to compare a circuit against, and how closely."""

    reference: str = Field(
        description=(
            "Reference netlist: a file path, or literal netlist text (anything "
            "containing a newline is read as text)."
        ),
    )
    anchors: list[str] | None = Field(
        default=None,
        description=(
            "Named nets that must map by name between the reference and this "
            "circuit — ports, rails, outputs. A design that is isomorphic but "
            "puts 'vout' in the wrong place fails on these. Ground is implicit."
        ),
    )
    rtol: float = Field(
        default=1e-6,
        description="Relative tolerance when comparing numeric values and parameters.",
    )


def render_spellings(policy: type[RenderPolicy]) -> str:
    """The refusal text for a bad ``render`` argument, listing what does work.

    Read off the policy's own fields — names, and for a closed choice its
    values — so a tool that adds a field (or has fewer) cannot advertise a
    spelling its model would reject, and a renamed value cannot leave the
    refusal naming the old one.
    """
    parts: list[str] = []
    for name in sorted(policy.model_fields):
        annotation = policy.model_fields[name].annotation
        choices = get_args(annotation) if get_origin(annotation) is Literal else ()
        parts.append(f"{name} {'|'.join(repr(c) for c in choices)}" if choices else name)
    return (
        "render takes true (draw with the default policy), false or omitted (do "
        f"not draw), or an object with any of: {', '.join(parts)}"
    )


def coerce_render_policy(value: Any, *, policy: type[RenderPolicy] = RenderPolicy) -> Any:
    """Accept the bare-boolean spellings of "just draw it" / "do not draw".

    Bind it into a tool's ``render`` field with ``BeforeValidator`` and a
    ``json_schema_input_type`` naming that tool's policy class; see the
    ``RenderArgument`` alias in verify.py and schematic_edit.py.

    ``render=True`` is what a caller reaches for first, and rejecting it used to
    name the policy class — a type the message gave no way to reach — instead of
    the keys and values that actually work. The boolean is coerced here so the
    policy object stays the single source of truth for the defaults, and the
    refusal for anything else enumerates the accepted spellings inline.
    """
    if value is True:
        return {}
    if value is False:
        return None
    if value is None or isinstance(value, (Mapping, RenderPolicy)):
        return value
    raise ValueError(render_spellings(policy))


# The behaviour hints a client shows a person before it approves a call. Each
# constant names what the four flags MEAN together, so a tool declares the
# claim it is making rather than four booleans a reader has to re-derive.

#: Reads and reports; changes nothing, and the same call answers the same way.
RO_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

#: Starts new work, leaving new artifacts beside whatever was already there.
#: Calling it again does the work again rather than returning the first answer,
#: and it reaches a simulator and a filesystem outside the server's own state.
NEW_WORK_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)

#: Changes something that is already there — a running job, an exported file —
#: within the paths and records this server owns. Repeating the call settles on
#: the same state instead of changing more.
REPEATABLE_CHANGE_ANNOTATIONS = types.ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)


@dataclass(frozen=True)
class RegisteredTool:
    """Tool registration metadata used by the dispatch layer."""

    definition: types.Tool
    handler: Callable
    input_model: type[ToolInput] | None


def _declare_warnings_key(schema: dict[str, Any]) -> dict[str, Any]:
    """Declare the ``warnings`` key any payload can grow.

    ``sanitize_payload`` injects a top-level ``warnings`` list into ANY
    tool's payload the moment a float in it is non-finite — reachable
    through raw waveform samples on a diverged run. A schema that closes
    itself with ``additionalProperties: false`` and does not declare the key
    therefore rejects its own response exactly when a run went wrong, and a
    strict client rejects the whole ``tools/list`` over one such schema,
    disabling every tool on the server.

    Declared here because this is the schema choke point, mirroring the
    response choke point that adds the key: per-tool declarations put the
    two in different places and let each new tool omit it silently. Both
    schema entry points pass through it: ``@registry.tool`` and
    ``declare_output_schema``.

    Edits the schema in place and returns it, so a module's exported
    ``*_OUTPUT_SCHEMA`` constant IS the schema clients are served — a copy
    would leave the two able to disagree, which is the drift this exists to
    remove. A schema that already declares ``warnings`` is left alone.
    """
    properties = schema.setdefault("properties", {})
    properties.setdefault("warnings", WARNINGS_SCHEMA)
    return schema


def _stamp_output_schema(fn: Callable, schema: dict[str, Any]) -> None:
    """Stamp a handler's structuredContent contract onto the handler itself.

    The single choke point for the "contract belongs to the handler" rule:
    both ``@registry.tool`` and ``declare_output_schema`` stamp through here.
    Written via ``__dict__`` because a plain attribute assignment on a function
    is a pyright error, and ruff auto-rewrites ``setattr()`` back into one.
    """
    fn.__dict__["__output_schema__"] = schema


class ToolRegistry:
    """Registry for tool definitions and handlers."""

    def __init__(self) -> None:
        self._registered: list[RegisteredTool] = []

    def tool(
        self,
        *,
        name: str,
        title: str,
        description: str,
        input_model: type[ToolInput] | None,
        annotations: types.ToolAnnotations,
        output_schema: dict[str, Any] | None = None,
        output_model: type | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Register a tool and derive its schema from the input model.

        ``title`` is the short human-readable label a client shows in place of
        the wire name — a few words, no punctuation, readable by someone who
        does not know the tool. The description stays the contract; this is
        only what a person sees in a list.

        ``output_model`` (a TypedDict) is preferred over ``output_schema``
        (a hand-written dict): the schema is generated once at registration
        time from the same type the lib already returns, so the two can't
        drift. Only one of the two should be supplied.

        ``meta`` populates the tool definition's ``_meta`` object (e.g.
        ``{"ui": {"resourceUri": ...}}`` to declare an MCP Apps UI resource).
        """
        if output_model is not None and output_schema is not None:
            raise ValueError(
                f"Tool {name!r}: supply either output_model or output_schema, not both"
            )

        def decorator(handler: Callable) -> Callable:
            if any(rt.definition.name == name for rt in self._registered):
                raise ValueError(f"Tool already registered: {name}")

            @wraps(handler)
            async def wrapped(arguments: Any, state: SessionState) -> types.CallToolResult:
                if input_model is not None and not isinstance(arguments, input_model):
                    arguments = input_model.model_validate(arguments or {})
                return await handler(arguments, state)

            definition_kwargs: dict[str, Any] = {
                "name": name,
                "title": title,
                "description": description,
                "input_schema": (
                    build_input_schema(input_model)
                    if input_model is not None
                    else {"type": "object", "properties": {}, "additionalProperties": False}
                ),
                "annotations": annotations,
            }
            if output_model is not None:
                definition_kwargs["output_schema"] = _declare_warnings_key(
                    schema_from_typeddict(output_model)
                )
            elif output_schema is not None:
                definition_kwargs["output_schema"] = _declare_warnings_key(output_schema)

            definition = types.Tool(**definition_kwargs)
            if meta is not None:
                # ``meta`` is the field name; serializes by alias to ``_meta``.
                definition.meta = meta

            # The structuredContent contract belongs to the handler, not to
            # its registration: stamp it on both the original handler (whose
            # code object is the emitting stack frame) and the returned
            # wrapper (the module-visible name), so the test suite's
            # conformance hook attributes emissions uniformly for registered
            # tools and unregistered adapters alike (see
            # declare_output_schema).
            if definition.output_schema is not None:
                _stamp_output_schema(handler, definition.output_schema)
                _stamp_output_schema(wrapped, definition.output_schema)

            self._registered.append(
                RegisteredTool(
                    definition=definition,
                    handler=wrapped,
                    input_model=input_model,
                )
            )
            return wrapped

        return decorator

    def field_owners(self) -> dict[str, tuple[str, ...]]:
        """Map each advertised top-level wire field to the tools that take it."""
        owners: dict[str, list[str]] = {}
        for registered in self._registered:
            properties = registered.definition.input_schema.get("properties", {})
            for field in properties:
                owners.setdefault(field, []).append(registered.definition.name)
        return {field: tuple(sorted(set(names))) for field, names in owners.items()}

    def get_tools(self) -> tuple[list[types.Tool], dict[str, RegisteredTool]]:
        """Return the advertised tool list and the dispatch map behind it."""
        tool_defs: list[types.Tool] = []
        tool_dispatch: dict[str, RegisteredTool] = {}
        for registered in self._registered:
            # The ADVERTISED definition is the registered one with its
            # outputSchema dropped, and nothing else changed. Every
            # description a model declares — the tool's own and each field's
            # — reaches the client verbatim, so reading the source tells you
            # exactly what a client is shown; the surface is kept small by
            # writing each description short, which the size pins in
            # tests/test_consolidated_contracts.py hold to. The outputSchema
            # is the one exception: it was the single largest schema block
            # (84% of `jobs`, -35% across the surface) and a response teaches
            # its own shape, so it stays on the registered definition — the
            # dispatch side, which the doc gates scan and the conformance hook
            # validates emissions against.
            definition = registered.definition.model_copy(update={"output_schema": None})
            tool_defs.append(definition)
            tool_dispatch[registered.definition.name] = registered
        if not tool_defs:
            # Zero tools would complete the MCP handshake while advertising
            # nothing — a working connection to an empty server, which every
            # client reads as "no capabilities" rather than "misconfigured".
            # Fail loudly instead.
            raise RuntimeError(
                "The tool registry resolved to zero tools — a module that "
                "registers one is no longer imported"
            )
        return tool_defs, tool_dispatch


def declare_output_schema(
    output_schema: dict[str, Any] | None = None,
    *,
    output_model: type | None = None,
) -> Callable[[Callable], Callable]:
    """Declare the structuredContent contract of an unregistered handler.

    ``@registry.tool`` stamps ``__output_schema__`` on the handler it
    registers; this decorator stamps the same attribute on an internal
    adapter — a de-registered tool handler that a consolidated tool delegates
    to. The test suite's conformance hook validates every structuredContent
    emission at the first stack frame carrying the attribute, so a delegated
    emission is checked against the ADAPTER's contract instead of falling
    through to the delegating tool's envelope. Re-exposing an adapter as an
    MCP tool is a decorator swap back to ``@registry.tool(...)``.

    Accepts either a hand-written schema dict or a TypedDict ``output_model``
    (mutually exclusive), resolved exactly as ``registry.tool`` resolves them,
    including the shared ``warnings``-key declaration.
    """
    if (output_schema is None) == (output_model is None):
        raise ValueError("supply exactly one of output_schema or output_model")
    schema = schema_from_typeddict(output_model) if output_model is not None else output_schema
    assert schema is not None
    resolved = _declare_warnings_key(schema)

    def decorator(handler: Callable) -> Callable:
        _stamp_output_schema(handler, resolved)
        return handler

    return decorator


registry = ToolRegistry()


class ResponseBudget(NamedTuple):
    """The budget one call negotiates against, and how far its ladder may go."""

    tokens: int | None
    max_rung: int = response_budget.RUNG_SHRINK


_AUTOMATIC_DOOR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "automatic_door", default=False
)


@contextlib.contextmanager
def automatic_door() -> Iterator[None]:
    """Mark the handler calls inside as arriving through the API's automatic mode.

    Entered from inside the coroutine the engine loop runs, so the flag lives in
    that one task's context: a co-resident MCP server sharing the process (and
    the same :class:`ServerConfig`) cannot see it, which a config field or a
    session attribute could not promise.
    """
    token = _AUTOMATIC_DOOR.set(True)
    try:
        yield
    finally:
        _AUTOMATIC_DOOR.reset(token)


def resolve_response_budget(explicit: int | None, state: SessionState) -> ResponseBudget:
    """Resolve one call's budget: the caller's, else the server's default.

    An explicit budget is the caller's decision and runs the full ladder. Absent
    one, the server's ``[analysis] default_budget`` applies at the trim rung
    ONLY. That asymmetry is the whole safety argument: rung 0 removes empty
    presentation blocks and the identity echo and nothing else, so it can cut no
    fact and revoke no detail the caller asked for, while rung 1 revokes opt-ins
    — doing that unasked would silently answer a different question.

    Only the four consolidated tools that advertise ``budget`` consult this, so
    the default reaches exactly the surface it was designed for; ``0`` disables it
    and restores the fully undegraded default response.

    The API's automatic mode gets no default at all. That mode promises complete
    results and refuses ``budget`` outright, so a presentation ladder there would
    both contradict the promise and leave the caller no way to lift it — the
    ladder's own route text would send them at the field the interface rejects.
    """
    if explicit is not None:
        return ResponseBudget(explicit)
    if _AUTOMATIC_DOOR.get():
        return ResponseBudget(None)
    default = state.config.default_budget
    if default <= 0:
        return ResponseBudget(None)
    return ResponseBudget(default, response_budget.RUNG_TRIM)


# ---------------------------------------------------------------------------
# Simulation helpers — shared pre-checks
# ---------------------------------------------------------------------------


def require_simulator(state: SessionState) -> None:
    """Raise SimulationError if no simulator is available."""
    if state.default_simulator is None:
        raise SimulationError(no_simulator_message())


def resolve_run_simulator(requested: str | None, state: SessionState) -> type:
    """Resolve a per-run ``simulator=`` override to its simulator class, or fall
    back to the session default when ``requested`` is None.

    Shared by run_simulation, run_sweep, and run_montecarlo so a caller can pick
    which detected simulator a run/batch executes on. Raises SimulationError if
    the requested name is not among the detected simulators, or (via
    ``require_simulator``) if none is available at all.
    """
    if requested is not None:
        sim_cls = state.available_simulators.get(requested.lower())
        if sim_cls is None:
            raise SimulationError(
                f"Simulator '{requested}' is not available on this server "
                f"(detected: {list(state.available_simulators)}). "
                "inspect(kind='capabilities') lists the detected simulators.",
                show_hint=False,
            )
        return sim_cls
    require_simulator(state)
    assert state.default_simulator is not None  # guaranteed by require_simulator
    return state.default_simulator


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def safe_path(user_path: str, state: SessionState) -> Path:
    """Resolve and validate a user-provided path within security sandbox.

    This is a convenience wrapper around resolve_safe_path that uses
    the allowed_paths from the session state configuration.

    Args:
        user_path: Path string from user (relative or absolute)
        state: Current session state containing security configuration

    Returns:
        Resolved absolute path within sandbox

    Raises:
        PathSecurityError: If path violates security constraints
    """
    return resolve_safe_path(user_path, state.config.allowed_paths)


# ---------------------------------------------------------------------------
# Concurrency contract
#
# The MCP SDK dispatches EVERY incoming request as its own asyncio task on
# one shared event loop (mcp.shared.jsonrpc_dispatcher.JSONRPCDispatcher.run
# start_soons a task per message; only `initialize` is served inline, so that
# a client pipelining it with the next request still sees an initialized
# connection), so tool handlers run concurrently. Anything that blocks the
# loop stalls every in-flight request — including cancel_job — and the
# transport's receive loop itself. Where work runs:
#
#   * Event loop (handler coroutines): argument validation, JobRegistry /
#     SessionState mutation, and ALL response building. format_response /
#     json_response / CallToolResult construction must stay in the handler
#     frame — the test suite's output-schema conformance hook attributes
#     emissions by walking the current thread's stack, so an emission from a
#     worker thread would silently skip schema validation. Offload
#     boundaries return plain data.
#   * asyncio.to_thread: heavy or potentially-slow filesystem work that is
#     read-only or atomic. The categories, one example each: result parsing
#     (services.load_raw), batch result/log loops (compute_batch_stats),
#     cross-process index writes (the recent-circuits touch: filelock poll
#     router behind server.read_resource). Offloaded functions must stay
#     effect-free or atomic under cancellation: the awaiting task sees
#     CancelledError, but a worker thread that has started runs to
#     completion (cancellation cannot interrupt it mid-write); work
#     cancelled before the executor picks it up never begins. Either way
#     shared state stays consistent.
#   * Runner threads: long-lived simulator processes are owned by the
#     runner layer (sim_runner, sweep_runner, montecarlo_runner).
#
# Submit ordering: no suspension point between job registration/persistence
# (state.add_job / add_batch_job) and the asyncio.create_task that advances
# the job — a request cancelled at such an await would orphan a persisted
# "running"/"queued" job with no task behind it. Acquire runners (which
# await output-folder resolution) BEFORE registering the job.
#
# Intentionally inline on the loop, with bounds: netlist/schematic editor
# parses, mutations, and their cache invalidation (cached editor instances
# are MUTABLE and entangled with per-session snapshots — concurrent edits
# would be last-writer-wins data loss; worst case is a cold .asc parse over
# /mnt/c, ~1 s), job sidecar JSON loads (small per-circuit files), config
# saves (durable=False), log-file reads (KB scale), and library .lib parses
# (LibraryManager sessions are loop-owned mutable state; worst case ~1 s for
# a multi-MB vendor library).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schematic include-resolver + symbol-resolver + scene render (shared)
# ---------------------------------------------------------------------------


def make_include_resolver(state: SessionState) -> IncludeResolver:
    """An include resolver that routes every include/lib open through safe_path.

    The graph engine calls this before opening any include, so an in-deck include
    that escapes the allowed roots is denied and never read.

    The detected simulator's own library directories are a second allowed set,
    the same trust class ``deck_staging.stage_deck`` accepts as
    ``simulator_roots``: LTspice's ``.asc`` netlister appends a ``.lib`` into the
    install's model library on every sheet carrying a MOSFET, so with only
    ``allowed_paths`` this resolver denies a file the run path just staged —
    and ``verify_circuit`` reports the schematic's own library as an unusable
    include. The MCP and the Python API must answer "may I read this referenced file?" the
    same way, or the answer depends on which one you asked. The roots are
    resolved once per resolver rather than per include, and a deck still may
    not RUN from one — staging checks the authored file against
    ``allowed_paths`` alone.
    """
    simulator_roots = simulator_library_roots(state.default_simulator)

    def resolver(candidate: Path) -> Path | None:
        try:
            return safe_path(str(candidate), state)
        except PathSecurityError:
            pass
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        if any(resolved.is_relative_to(root) for root in simulator_roots):
            return resolved
        return None

    return resolver


def symbol_resolver_for(
    asc_path: Path | None, state: SessionState | None = None
) -> SymbolResolver:
    """Resolver with the sheet's own dir first, then configured/stock libraries.

    Mirrors the precedence the compiler and LTspice's own export use so a
    schematic that resolves for them resolves here too. When ``state`` is given,
    its configured ``symbol_paths`` take precedence over the stock libraries.
    ``asc_path`` may be ``None`` for a vocabulary lookup with no schematic in
    hand (``inspect(symbols)`` without a ``path``): the local dir is then simply
    absent from the precedence and only the configured/stock libraries apply.
    """
    from spicelib import AscEditor

    project: list[Path] = []
    if state is not None:
        project += [Path(p) for p in state.config.symbol_paths]
    project += [Path(p) for p in (AscEditor.custom_lib_paths or [])]
    project += [Path(p) for p in (getattr(AscEditor, "simulator_lib_paths", None) or [])]
    return SymbolResolver(
        local_dir=asc_path.parent if asc_path is not None else None,
        project_paths=project,
        stock_paths=default_stock_paths(),
    )


def render_scene_artifact(
    scene: Scene,
    out_dir: Path,
    *,
    image_format: Literal["png", "svg"],
    scale: float,
    max_pixels: int | None = None,
) -> tuple[RenderedImage, Path, bool]:
    """Render a scene, bound its pixels, and write a content-hashed artifact.

    Rasterizes to ``image_format`` at ``scale``; when ``max_pixels`` is set and a
    raster exceeds it, re-renders at the largest scale that fits and flags
    ``downscaled``. Writes ``<source-stem>.<sha8>.<suffix>`` into ``out_dir``
    (created on demand). Returns ``(image, out_path, downscaled)``.
    """
    svg = render_svg(scene)
    image = render_image(svg, image_format=image_format, scale=scale)
    downscaled = False
    if (
        image.is_raster
        and max_pixels is not None
        and image.width
        and image.height
        and image.width * image.height > max_pixels
    ):
        factor = math.sqrt(max_pixels / (image.width * image.height))
        reduced = max(0.1, round(scale * factor, 3))
        if reduced < scale:
            image = render_image(svg, image_format=image_format, scale=reduced)
            downscaled = True

    suffix = "png" if image.is_raster else "svg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{scene.source.stem}.{hashlib.sha256(image.data).hexdigest()[:8]}.{suffix}"
    )
    atomic_write_bytes(out_path, image.data, durable=False)
    return image, out_path, downscaled
