"""Shared utilities for tool handlers."""

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import re
import types as _stdlib_types
import typing
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from mcp import types
from pydantic import BaseModel, ConfigDict

from ltspice_mcp.config import VALID_PROFILES
from ltspice_mcp.errors import NetlistError, PathSecurityError, SimulationError
from ltspice_mcp.lib import atomic_write_bytes
from ltspice_mcp.lib.filelock import DEFAULT_TIMEOUT, file_lock
from ltspice_mcp.lib.job_store import SIDECAR_DIRNAME
from ltspice_mcp.lib.netlist_graph import IncludeResolver
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.raster import RenderedImage, render_image
from ltspice_mcp.lib.runner_base import NGSPICE_CONTROL_WRITE_MARKER

# Re-exported, not defined here: the logopinfo injection lives in the runner
# layer so the experiment coordinator (lib/) can call it without importing
# tools/, and the tool handlers keep reaching it through this module as before.
from ltspice_mcp.lib.runner_base import inject_logopinfo as inject_logopinfo
from ltspice_mcp.lib.schematic_renderer import render_svg
from ltspice_mcp.lib.schematic_scene import Scene, SymbolResolver, default_stock_paths
from ltspice_mcp.lib.simulator import no_simulator_message
from ltspice_mcp.state import SessionState

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
    meaningless-number doctrine (lib/result_observations.py), substitute null
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
        structuredContent=data,
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
        structuredContent=payload,
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
                mimeType=image.mime_type,
            )
        )
    else:
        content.append(types.TextContent(type="text", text=image.data.decode("utf-8")))
    content.append(types.TextContent(type="text", text=text))

    return types.CallToolResult(content=content, structuredContent=payload)


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

# Free-text measurement caveats (see the observations-vs-warnings doctrine in
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

PAGINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total": {"type": "integer"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
        "has_more": {"type": "boolean"},
        "next_offset": {"type": ["integer", "null"]},
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
# "Result-trust: surface, don't judge" in CLAUDE.md. A "surfacer"
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
                    "constant trigger LEVEL, not the crossing point — read 'at' "
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
                    "measurement — THIS is the answer for a WHEN rise-time/crossing "
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


RO_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class StrictModel(BaseModel):
    """Shared Pydantic config for all strict models (tool inputs and nested schemas)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ToolInput(StrictModel):
    """Base for top-level tool input models registered via @registry.tool(input_model=...)."""

    pass


@dataclass(frozen=True)
class RegisteredTool:
    """Tool registration metadata used by the dispatch layer."""

    definition: types.Tool
    handler: Callable
    input_model: type[ToolInput] | None
    profiles: frozenset[str]
    # Deprecated former names that still dispatch to this tool but are NOT
    # advertised in the tool list (keeps a rename back-compatible without
    # growing the tool surface).
    aliases: frozenset[str] = frozenset()


def _strip_titles(node: Any) -> Any:
    """Recursively remove Pydantic title metadata from a JSON schema node."""
    if isinstance(node, dict):
        node = {k: _strip_titles(v) for k, v in node.items() if k != "title"}
        return node
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


# Keywords that make a schema branch more than a plain type constraint, so it
# cannot be merged into a multi-type ``type`` array without changing meaning
# (``const``/``enum`` would also constrain the ``null`` member; the composition
# and reference keywords have no per-type semantics to inherit).
_UNFOLDABLE_KEYWORDS = frozenset({"$ref", "allOf", "anyOf", "const", "enum", "not", "oneOf"})

# Cheap pre-filter: a fragment shorter than the ``$ref`` that would replace it
# can never pay, so it is not worth hashing. The gain test below is the real
# gate.
_HOIST_MIN_CHARS = 28

# A shared definition has to earn its indirection: a reader now has to look the
# name up. Roughly twenty-five tokens of net saving is the line.
_HOIST_MIN_GAIN = 100


_JSON_TYPE_OF: dict[type, str] = {
    bool: "boolean",
    str: "string",
    int: "integer",
    float: "number",
}


def _compact_type_keywords(node: Any) -> Any:
    """Collapse type keywords that say something the schema already said.

    Pydantic spells every ``X | None`` as a two-branch ``anyOf``. Where the
    branches are plain type constraints, the equivalent ``{"type": [...]}``
    array says the same thing in far fewer characters: type-specific keywords
    (``items``, ``minLength``, ``minimum``) are no-ops for the other members,
    so ``{"items": ..., "type": ["array", "null"]}`` accepts exactly what the
    two-branch form did. Branches carrying ``const``/``enum``/``$ref`` are left
    alone — those constrain the *value*, not just its type.

    A ``type`` sitting beside a ``const`` of that same type is dropped for the
    same reason: the literal already pins the value, so the keyword narrows
    nothing. The union discriminators (``metric``, ``op``, ``kind``) are all
    of that shape, one per member of each tagged union.
    """
    if isinstance(node, list):
        return [_compact_type_keywords(item) for item in node]
    if not isinstance(node, dict):
        return node

    folded = {key: _compact_type_keywords(value) for key, value in node.items()}
    if "const" in folded and _JSON_TYPE_OF.get(type(folded["const"])) == folded.get("type"):
        folded.pop("type")
    branches = folded.get("anyOf")
    if not isinstance(branches, list) or len(branches) < 2:
        return folded
    if not all(
        isinstance(b, dict)
        and isinstance(b.get("type"), str)
        and not (_UNFOLDABLE_KEYWORDS & b.keys())
        for b in branches
    ):
        return folded

    # Only one branch may carry type-specific keywords; merging two constrained
    # branches would apply each one's keywords to the other's type.
    constrained = [b for b in branches if b.keys() != {"type"}]
    if len(constrained) > 1:
        return folded

    types = list(dict.fromkeys(b["type"] for b in branches))
    if len(types) != len(branches):
        return folded

    siblings = {k: v for k, v in folded.items() if k != "anyOf"}
    inner = dict(constrained[0]) if constrained else {}
    if inner.keys() & (siblings.keys() - {"type"}):
        return folded
    return {**inner, **siblings, "type": types}


def _shape_name(node: dict[str, Any]) -> str | None:
    """Name a fragment after the shape it describes, e.g. ``StringListOrNull``.

    Used when one fragment is shared by properties with different names, where
    naming it after any one of them would misdescribe the others. Only a plain
    type constraint gets one: a shape name that hid a real default (a
    ``Boolean`` that is secretly false unless set) would cost a reader more
    than the characters it saved.
    """
    if node.keys() - {"type", "items", "default"} or node.get("default") is not None:
        return None
    raw = node.get("type")
    types = [raw] if isinstance(raw, str) else list(raw or [])
    core = [t for t in types if t != "null"]
    if len(core) != 1:
        return None
    if core[0] == "array":
        items = node.get("items")
        if not isinstance(items, dict) or not isinstance(items.get("type"), str):
            return None
        base = f"{items['type'].capitalize()}List"
    else:
        base = core[0].capitalize()
    return base + ("OrNull" if "null" in types else "")


def _defs_name(hint: str, taken: set[str]) -> str:
    """Turn a naming hint into a ``$defs`` key that is free to use."""
    base = "".join(part[:1].upper() + part[1:] for part in hint.split("_") if part)
    base = re.sub(r"[^0-9A-Za-z]", "", base) or "Shared"
    name = base
    suffix = 0
    while name in taken:
        suffix += 1
        name = f"{base}Arg" if suffix == 1 else f"{base}Arg{suffix}"
    return name


def _hoist_shared_fragments(schema: dict[str, Any]) -> dict[str, Any]:
    """Move sub-schemas repeated across the document into shared ``$defs``.

    Pydantic re-emits a field's schema at every model that declares it, so a
    mixin field (``sources``, ``step``, ``spec`` on the analysis recipes) is
    serialized once per recipe. A single ``$defs`` entry with ``$ref`` use
    sites says the same thing once.

    Only nested property values are hoisted, and only when every use site sits
    under the same property name — the name it lends the ``$defs`` entry has to
    describe every reference, or the indirection costs a reader more than the
    characters it saves. The tools' own top-level properties are never hoisted:
    their inline ``description`` is the only documentation a caller gets.
    """
    counts: dict[str, int] = {}
    hints: dict[str, set[str]] = {}

    def survey(node: Any, key_hint: str | None, depth: int) -> None:
        if isinstance(node, list):
            for item in node:
                survey(item, key_hint, depth)
            return
        if not isinstance(node, dict):
            return
        if key_hint is not None:
            blob = json.dumps(node, separators=(",", ":"), sort_keys=True)
            if len(blob) >= _HOIST_MIN_CHARS:
                counts[blob] = counts.get(blob, 0) + 1
                hints.setdefault(blob, set()).add(key_hint)
        for name, value in node.items():
            if name == "properties" and isinstance(value, dict):
                # Depth 0 is the tool's own argument list — leave it inline.
                for prop, sub in value.items():
                    survey(sub, prop if depth else None, depth + 1)
            elif name == "$defs" and isinstance(value, dict):
                for sub in value.values():
                    survey(sub, None, max(depth, 1))
            else:
                survey(value, None, depth)

    survey(schema, None, 0)

    defs: dict[str, Any] = dict(schema.get("$defs") or {})
    taken = set(defs)
    replacements: dict[str, str] = {}
    for blob, count in sorted(counts.items()):
        if count < 2:
            continue
        fragment = json.loads(blob)
        names = hints[blob]
        hint = next(iter(names)) if len(names) == 1 else _shape_name(fragment)
        if hint is None:
            continue
        name = _defs_name(hint, taken)
        ref_cost = len(f'{{"$ref":"#/$defs/{name}"}}')
        # Net saving: every use site shrinks, minus the one def entry we add.
        gain = count * (len(blob) - ref_cost) - (len(name) + 3 + len(blob))
        if gain < _HOIST_MIN_GAIN:
            continue
        taken.add(name)
        defs[name] = fragment
        replacements[blob] = name

    if not replacements:
        return schema

    def rewrite(node: Any, key_hint: str | None, depth: int) -> Any:
        if isinstance(node, list):
            return [rewrite(item, key_hint, depth) for item in node]
        if not isinstance(node, dict):
            return node
        if key_hint is not None:
            blob = json.dumps(node, separators=(",", ":"), sort_keys=True)
            target = replacements.get(blob)
            if target is not None:
                return {"$ref": f"#/$defs/{target}"}
        out: dict[str, Any] = {}
        for name, value in node.items():
            if name == "properties" and isinstance(value, dict):
                out[name] = {
                    prop: rewrite(sub, prop if depth else None, depth + 1)
                    for prop, sub in value.items()
                }
            elif name == "$defs" and isinstance(value, dict):
                out[name] = {
                    def_name: rewrite(sub, None, max(depth, 1)) for def_name, sub in value.items()
                }
            else:
                out[name] = rewrite(value, None, depth)
        return out

    rewritten = rewrite({k: v for k, v in schema.items() if k != "$defs"}, None, 0)
    # Def bodies are rewritten with no key hint at their own root, so a hoisted
    # fragment can never be rewritten into a reference to itself; a fragment
    # nested inside another one is strictly shorter, so the refs cannot cycle.
    rewritten["$defs"] = {name: rewrite(body, None, 1) for name, body in defs.items()}
    return rewritten


def _build_input_schema(input_model: type[ToolInput]) -> dict[str, Any]:
    """Generate a cleaned MCP-ready JSON schema from a Pydantic model.

    ``$defs`` are kept as Pydantic emits them, not inlined: a shared submodel
    appears once and every use site is a ``$ref``, which measured 21% smaller
    on the consolidated surface (followups item 30). Every ref is internal to
    the one schema document, so any conformant client resolves it locally.

    Two further passes shrink the *advertised* shape only — the Pydantic model
    stays the validator and accepts exactly what it did before.
    ``_compact_type_keywords`` drops the type keywords a schema already implies
    (a nullable branch becomes a multi-type ``type`` array; the ``type`` beside
    a ``const`` goes), and ``_hoist_shared_fragments`` gives a sub-schema
    repeated across models one ``$defs`` entry instead of a copy per use site.
    Together they measured a tenth off the consolidated surface, most of it on
    ``analyze_results``, whose twenty-odd recipe models each restated the same
    seven shared fields. ``tests/test_consolidated_contracts.py`` pins the
    resulting size per tool.
    """
    schema = _strip_titles(input_model.model_json_schema())
    return _hoist_shared_fragments(_compact_type_keywords(schema))


# ---------------------------------------------------------------------------
# TypedDict → JSON Schema generator
# ---------------------------------------------------------------------------


_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and typing.is_typeddict(tp)


def _jsontype_from_union(args: tuple[Any, ...]) -> dict[str, Any]:
    """Handle ``X | None`` and ``X | Y | None`` unions.

    ``X | None`` becomes ``{"type": ["X", "null"]}`` when X is a single
    primitive — the common case for ``float | None`` fields. Mixed unions
    with complex members fall back to ``anyOf``.
    """
    non_none = [a for a in args if a is not type(None)]
    has_none = len(non_none) != len(args)
    if len(non_none) == 1:
        inner = _schema_for_type(non_none[0])
        if has_none and "type" in inner and isinstance(inner["type"], str):
            type_val = inner["type"]
            return {**inner, "type": [type_val, "null"]}
        if has_none:
            # Complex inner (nested object/array) — use anyOf with null.
            return {"anyOf": [inner, {"type": "null"}]}
        return inner
    variants = [_schema_for_type(a) for a in non_none]
    if has_none:
        variants.append({"type": "null"})
    return {"anyOf": variants}


def _is_union(tp: Any) -> bool:
    """True for both ``typing.Union[X, Y]`` and ``X | Y`` syntax."""
    if get_origin(tp) is Union:
        return True
    # Python 3.10+: `X | Y` has origin == types.UnionType (the class).
    return get_origin(tp) is _stdlib_types.UnionType


def _schema_for_type(tp: Any) -> dict[str, Any]:
    """Return a JSON Schema fragment for a type annotation."""
    if tp is Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    if tp in _PRIMITIVE_MAP:
        return {"type": _PRIMITIVE_MAP[tp]}
    if _is_typeddict(tp):
        return schema_from_typeddict(tp)

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        return {"enum": list(args)}
    if _is_union(tp):
        return _jsontype_from_union(args)
    if origin in (list, tuple):
        # A fixed, heterogeneous tuple (``tuple[int, str]``) has no single
        # ``items`` schema; rendering ``args[0]`` only would SILENTLY drop the
        # rest, so refuse it loudly (use a TypedDict or list[...], or add
        # prefixItems support) rather than emit a schema that lies about the
        # shape. A list, a ``tuple[X, ...]``, and a single-type/empty tuple all
        # map faithfully to an array of one item type.
        if origin is tuple and len(args) > 1 and args[1] is not Ellipsis:
            raise TypeError(
                f"Fixed heterogeneous tuple {tp!r} has no faithful single-`items` "
                "JSON Schema. Use a TypedDict (named fields) or list[...] for the "
                "output model, or add prefixItems support to _schema_for_type."
            )
        item_type = args[0] if args else Any
        return {"type": "array", "items": _schema_for_type(item_type)}
    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": _schema_for_type(value_type),
        }

    raise TypeError(
        f"Unsupported type annotation for schema generation: {tp!r}. "
        "Extend _schema_for_type in tools/_base.py if this construct is "
        "now used in the repo."
    )


@cache
def schema_from_typeddict(td: type) -> dict[str, Any]:
    """Generate a JSON Schema (``{"type": "object", ...}``) from a TypedDict.

    Every field is emitted under ``properties``. ``required`` reflects the
    two DISTINCT ways a field can be optional, both of which exist in the
    wire format: a ``NotRequired``/``total=False`` field may be ABSENT
    (omit-when-empty convention; e.g. ``GainAtPoint.phase_deg_unwrapped``),
    while an ``X | None`` field is always present but may be null. Marking
    an omitted key as required makes schema-validating MCP clients reject
    responses that follow the documented omit-when-empty behavior.
    """
    if not _is_typeddict(td):
        raise TypeError(f"Expected TypedDict, got {td!r}")

    hints = get_type_hints(td)
    # ``__required_keys__`` is computed at class-creation time and is UNRELIABLE
    # for ``NotRequired`` fields when the defining module uses ``from __future__
    # import annotations``: the wrapper is stringized, so the TypedDict metaclass
    # can't see it and wrongly counts the field as required (verified on 3.13).
    # ``get_type_hints(..., include_extras=True)`` EVALUATES the annotation,
    # recovering the ``NotRequired`` wrapper, so it detects optionality regardless
    # of stringization. (``Required`` in a ``total=False`` class is the symmetric
    # case but isn't used in this repo, so it's not special-cased.)
    extra_hints = get_type_hints(td, include_extras=True)
    structurally_required = getattr(td, "__required_keys__", frozenset(hints))
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name, field_type in hints.items():
        properties[field_name] = _schema_for_type(field_type)
        # A field is required unless the key may be absent entirely
        # (NotRequired / total=False) or its type admits None.
        admits_none = _is_union(field_type) and type(None) in get_args(field_type)
        not_required = get_origin(extra_hints.get(field_name)) is typing.NotRequired
        if field_name in structurally_required and not admits_none and not not_required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry for tool definitions and handlers."""

    def __init__(self) -> None:
        self._registered: list[RegisteredTool] = []

    @staticmethod
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
        two in different places and let each new tool omit it silently.

        Edits the schema in place and returns it, so a module's exported
        ``*_OUTPUT_SCHEMA`` constant IS the schema clients are served — a copy
        would leave the two able to disagree, which is the drift this exists to
        remove. A schema that already declares ``warnings`` is left alone.
        """
        properties = schema.setdefault("properties", {})
        properties.setdefault("warnings", WARNINGS_SCHEMA)
        return schema

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_model: type[ToolInput] | None,
        annotations: types.ToolAnnotations,
        profiles: tuple[str, ...] = ("full",),
        output_schema: dict[str, Any] | None = None,
        output_model: type | None = None,
        meta: dict[str, Any] | None = None,
        aliases: tuple[str, ...] = (),
    ) -> Callable[[Callable], Callable]:
        """Register a tool and derive its schema from the input model.

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
                "description": description,
                "inputSchema": (
                    _build_input_schema(input_model)
                    if input_model is not None
                    else {"type": "object", "properties": {}, "additionalProperties": False}
                ),
                "annotations": annotations,
            }
            if output_model is not None:
                definition_kwargs["outputSchema"] = self._declare_warnings_key(
                    schema_from_typeddict(output_model)
                )
            elif output_schema is not None:
                definition_kwargs["outputSchema"] = self._declare_warnings_key(output_schema)

            definition = types.Tool(**definition_kwargs)
            if meta is not None:
                # ``meta`` is the field name; serializes by alias to ``_meta``.
                definition.meta = meta

            self._registered.append(
                RegisteredTool(
                    definition=definition,
                    handler=wrapped,
                    input_model=input_model,
                    profiles=frozenset(profiles),
                    aliases=frozenset(aliases),
                )
            )
            return wrapped

        return decorator

    def known_names(self) -> set[str]:
        """All registered tool names, across every profile (for diagnostics).

        Lets the dispatcher tell a profile-filtered tool (exists, hidden by the
        active profile) apart from a genuinely unknown name.
        """
        return {rt.definition.name for rt in self._registered} | {
            a for rt in self._registered for a in rt.aliases
        }

    def field_owners_for_profile(self, profile: str) -> dict[str, tuple[str, ...]]:
        """Map advertised top-level wire fields to tools in one profile."""
        effective_profile = profile if profile in VALID_PROFILES else "full"
        owners: dict[str, list[str]] = {}
        for registered in self._registered:
            if effective_profile not in registered.profiles:
                continue
            properties = registered.definition.inputSchema.get("properties", {})
            for field in properties:
                owners.setdefault(field, []).append(registered.definition.name)
        return {field: tuple(sorted(set(names))) for field, names in owners.items()}

    def get_for_profile(self, profile: str) -> tuple[list[types.Tool], dict[str, RegisteredTool]]:
        """Return the tool list and dispatch map for a profile."""
        effective_profile = profile if profile in VALID_PROFILES else "full"
        tool_defs: list[types.Tool] = []
        tool_dispatch: dict[str, RegisteredTool] = {}
        for registered in self._registered:
            if effective_profile in registered.profiles:
                definition = registered.definition
                if definition.outputSchema is not None:
                    # The advertised tool list drops outputSchema — it was the
                    # single largest schema block (84% of `jobs`, -35% across
                    # the consolidated surface; followups item 30). Return
                    # shapes are learned from responses instead. The
                    # dispatch-side definition keeps the schema: the test
                    # suite's conformance hook validates every emission
                    # against it, so the declared shape is still enforced.
                    definition = definition.model_copy(update={"outputSchema": None})
                tool_defs.append(definition)
                tool_dispatch[registered.definition.name] = registered
        # Second pass: deprecated aliases dispatch to their tool but are not
        # listed in tool_defs. Done AFTER all definition names so a real tool
        # name always wins; a collision with a real name or another tool's
        # alias is a registration bug we surface loudly, never silently drop.
        for registered in self._registered:
            if effective_profile not in registered.profiles:
                continue
            for alias in registered.aliases:
                existing = tool_dispatch.get(alias)
                if existing is not None and existing is not registered:
                    raise ValueError(
                        f"Alias {alias!r} of tool {registered.definition.name!r} collides "
                        f"with an existing tool name or alias ({existing.definition.name!r})"
                    )
                tool_dispatch[alias] = registered
        return tool_defs, tool_dispatch


registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------


DEFAULT_PAGE_CAP = 50
"""Server-side ceiling on list-endpoint page size — the "caps at 50" the
``limit`` field descriptions document."""


def paginate(
    items: list, arguments: Any, cap: int = DEFAULT_PAGE_CAP
) -> tuple[list, int, int, int]:
    """Slice a list according to offset/limit from tool arguments.

    Returns:
        (page, total, offset, limit) tuple
    """
    total = len(items)
    offset = max(0, min(int(getattr(arguments, "offset", 0)), total))
    # Floor as well as cap: limit=0 would otherwise report has_more=true with
    # next_offset == offset — a pagination loop that never advances — and a
    # negative limit would mis-slice. Server-side clamping (not rejection) is
    # the documented contract for out-of-range limits.
    limit = max(1, min(int(getattr(arguments, "limit", cap)), cap))
    return items[offset : offset + limit], total, offset, limit


def pagination_metadata(total: int, offset: int, limit: int) -> dict[str, Any]:
    """Build structured pagination metadata for JSON responses."""
    has_more = offset + limit < total
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": offset + limit if has_more else None,
    }


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
                f"(detected: {list(state.available_simulators)}). server_status "
                "lists the detected simulators.",
                show_hint=False,
            )
        return sim_cls
    require_simulator(state)
    assert state.default_simulator is not None  # guaranteed by require_simulator
    return state.default_simulator


def resolve_netlist_path(netlist_str: str, state: SessionState) -> Path:
    """Resolve and validate a netlist path. Raises SimulationError on failure.

    PathSecurityError propagates unchanged: the dispatch layer has a dedicated
    branch that appends the sandbox-widening guidance (allowed paths, the TOML
    knob, the restart requirement) — re-wrapping it as SimulationError would
    replace that guidance with a misdirecting simulator hint.
    """
    try:
        netlist_path = safe_path(netlist_str, state)
    except PathSecurityError:
        raise
    except Exception as e:
        raise SimulationError(f"Invalid netlist path: {e}") from e
    if not netlist_path.exists():
        raise SimulationError(f"Netlist file not found: {netlist_path}")
    return netlist_path


def path_lock(registry: dict[Path, asyncio.Lock], path: Path, cap: int = 64) -> asyncio.Lock:
    """Get or create a per-path lock in ``registry``, LRU-bounded at ``cap``.

    Shared mechanism behind every per-file lock registry (schematic edits,
    ``.asc`` exports): refresh recency on hit; at capacity evict the oldest
    *unheld* lock — if all are held, overshoot temporarily rather than break
    mutual exclusion by evicting a lock someone is inside.
    """
    if path in registry:
        registry[path] = registry.pop(path)
        return registry[path]
    if len(registry) >= cap:
        for candidate in list(registry):
            if not registry[candidate].locked():
                del registry[candidate]
                break
    registry[path] = asyncio.Lock()
    return registry[path]


def circuit_lock_target(path: Path) -> Path:
    """Anchor for the cross-process lock on one circuit file.

    Lives under the circuit's ``.ltspice-mcp/locks/`` sidecar directory
    (``file_lock`` appends ``.lock``) so user directories aren't littered
    with lock files next to their circuits.
    """
    return path.parent / SIDECAR_DIRNAME / "locks" / path.name


@contextlib.asynccontextmanager
async def circuit_file_lock(path: Path) -> AsyncIterator[None]:
    """Cross-process lock for mutations/exports of one circuit file.

    Parallel MCP server processes editing the same circuit serialize here —
    without it, the whole-file read-modify-write saves are last-writer-wins
    and a concurrent session's edit is silently lost. Acquisition polls in a
    worker thread (per filelock's contract, so a contended lock never stalls
    the event loop); release is two fast syscalls, done inline.

    Acquire this BEFORE fetching a cached editor: the editor cache re-stats
    the file on every fetch, so taking the lock first guarantees the stat
    sees a concurrent writer's completed save rather than a mid-edit state.
    (Residual: on coarse-mtime filesystems like WSL's /mnt/c a same-size
    rewrite within one mtime tick can still go undetected — see FileCache.)
    """
    # Acquire INSIDE the try so stack.close() always runs: a cancel (cancel_job
    # / shutdown) landing at the await boundary right after the worker thread
    # took the flock would otherwise leak it until process exit. (Residual: if
    # the cancel lands while the worker is still blocked acquiring, the thread
    # can register the lock after close() already ran — inherent to to_thread,
    # not fixable without a cancel-aware lock; the narrow window is cancel-only.)
    stack = contextlib.ExitStack()
    try:
        try:
            await asyncio.to_thread(stack.enter_context, file_lock(circuit_lock_target(path)))
        except TimeoutError as e:
            raise NetlistError(
                f"{path.name} is locked by another ltspice-mcp process "
                f"(waited {DEFAULT_TIMEOUT:.0f}s). Retry once its edit finishes."
            ) from e
        yield
    finally:
        stack.close()


# LTspice's ``create_netlist`` always writes the sidecar ``<name>.net`` next to
# the ``.asc``, so two concurrent exports of the same schematic would race on
# one output file (torn/partial reads of the deck). Serialize per resolved
# ``.asc`` path; distinct schematics still export in parallel.
_asc_export_locks: dict[Path, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def asc_export_lock(asc_path: Path) -> AsyncIterator[None]:
    """Serialize LTspice netlist exports of one schematic.

    In-process: a per-``.asc`` asyncio lock. Cross-process: the shared
    circuit file locks on BOTH the schematic and the sidecar ``.net`` —
    LTspice reads the ``.asc`` and overwrites the ``.net``, and a parallel
    session may be editing the ``.net`` itself under its own file lock.
    Fixed acquisition order (``.asc`` then ``.net``); edit paths take exactly
    one file lock, so no cycle is possible.
    """
    async with (
        path_lock(_asc_export_locks, asc_path),
        circuit_file_lock(asc_path),
        circuit_file_lock(asc_path.with_suffix(".net")),
    ):
        yield


def _sanitize_export_for_ngspice(net_path: Path) -> Path:
    """Write an ngspice-runnable twin of an LTspice-exported netlist.

    LTspice's exporter appends ``.backanno`` — an LTspice-only dot command
    ngspice aborts on ("unimplemented dot command") — and can emit its
    private ``§`` name-prefix character and ``µ`` unit suffix, neither of
    which ngspice's parser accepts. The scrub goes to its own
    ``{stem}.ngspice.net`` sidecar rather than rewriting the shared ``.net``
    in place: a concurrent LTspice-target run of the same schematic
    regenerates ``.net`` after the export lock releases, and an in-place
    rewrite would hand one of the two runs the other simulator's deck.
    """
    from ltspice_mcp.lib import atomic_write_text
    from ltspice_mcp.lib.encoding import read_spice_text

    text = read_spice_text(net_path)
    lines = [ln for ln in text.splitlines() if ln.strip().lower() != ".backanno"]
    cleaned = "\n".join(lines).replace("§", "").replace("µ", "u").replace("μ", "u")
    out_path = net_path.with_name(net_path.stem + ".ngspice.net")
    atomic_write_text(out_path, cleaned + "\n", durable=False)
    return out_path


async def resolve_runnable_netlist(
    netlist_str: str, state: SessionState, simulator: type | None = None
) -> Path:
    """Resolve a path AND auto-export ``.asc`` → ``.net`` if needed.

    spicelib's ``SpiceEditor`` (used by the sweep / Monte Carlo runners)
    rejects ``.asc`` schematics — it expects the ``^*`` netlist comment
    header and otherwise fails with a cryptic ``Expected pattern "^\\*"
    not found``. This helper detects ``.asc`` and runs the LTspice
    ``create_netlist`` exporter to produce a sidecar ``.net``, so
    callers (sweep / MC config) can store the runnable path up front.

    ``simulator`` is the class the run will execute on (defaults to the
    session default): when it is ngspice, the LTspice export is sanitized
    for it (see ``_sanitize_export_for_ngspice``) — without that, every
    schematic run on ngspice dies on the exporter's ``.backanno``.

    The cheap safe_path/exists checks run inline, but the export launches the
    LTspice binary and blocks until it exits — heavy work that would stall the
    shared event loop, so it is offloaded via ``asyncio.to_thread``. It touches
    no cached editors, so the offload is safe under the concurrency contract.
    """
    netlist_path = resolve_netlist_path(netlist_str, state)
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path

    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        # Don't recommend export_netlist here — it ALSO needs LTspice, so that
        # advice dead-ends when only ngspice/etc. is available.
        raise SimulationError(
            f"{netlist_path.name} is an .asc schematic, which only LTspice can "
            "convert to a netlist, and LTspice is not available "
            f"(simulators: {list(state.available_simulators.keys())}). Supply a "
            "hand-written .cir/.net to simulate with the current simulator, or "
            "point the server at an LTspice executable ([simulator] path in the "
            "config file or LTSPICE_MCP_SIMULATOR_EXE) and restart. (The .asc's "
            "embedded .model/.lib/analysis directives can be reused in a .cir.)",
            show_hint=False,
        )
    async with asc_export_lock(netlist_path):
        try:
            # Bound the export: create_netlist launches LTspice, which can hang
            # indefinitely on a Windows-side modal dialog. The export lock is
            # held across this call, so an unbounded hang wedges every later run
            # of this schematic — cap it at the sim timeout so it fails loudly.
            net_path = Path(
                await asyncio.to_thread(
                    ltspice_cls.create_netlist,
                    str(netlist_path),
                    timeout=state.config.default_timeout,
                )
            )
        except Exception as e:
            raise SimulationError(
                f"Auto-exporting {netlist_path.name} to a netlist failed: {e}"
            ) from e
        if not await asyncio.to_thread(net_path.exists):
            raise SimulationError(f"Auto-export of {netlist_path.name} produced no .net file")
        from ltspice_mcp.lib.simulator import is_ngspice

        if is_ngspice(simulator or state.default_simulator):
            net_path = await asyncio.to_thread(_sanitize_export_for_ngspice, net_path)

        # Snapshot the fresh deck INSIDE the lock and return THAT: create_netlist
        # writes a shared <stem>.net, so a parallel session re-exporting this .asc
        # overwrites it — and a caller that stored the shared path (sweep/MC
        # config, or a run staged moments later) would then read the peer's deck.
        # The snapshot stays in the same directory so a relative .include/.lib in
        # the deck still resolves against it.
        return await asyncio.to_thread(_stage_deck_snapshot, net_path)


def _stage_deck_snapshot(net_path: Path) -> Path:
    """Copy the exported deck to a content-addressed sibling and return it.

    Named by a hash of its bytes so repeat exports of the same .asc reuse one
    file — the snapshots stay bounded to one per distinct deck content, not one
    per run (a plain per-call unique name accumulates unbounded). Written
    atomically so a concurrent reader sees a whole file, never a torn copy.
    """
    from ltspice_mcp.lib import atomic_write_bytes

    data = net_path.read_bytes()
    digest = hashlib.sha1(data).hexdigest()[:12]
    snapshot = net_path.with_name(f"{net_path.stem}.run-{digest}{net_path.suffix}")
    if not snapshot.exists():
        atomic_write_bytes(snapshot, data, durable=False)
    return snapshot


# A ``.control``...``.endc`` block, case-insensitive. Group 1 is the body —
# everything between the ``.control`` line and the ``.endc`` line — so
# ``match.end(1)`` is exactly where the ``.endc`` line begins (the fallback
# insertion point when the block has no ``quit``/``exit``).
_RE_CONTROL_BLOCK = re.compile(rb"(?ims)^[ \t]*\.control\b[^\n]*\n(.*?)^[ \t]*\.endc\b[^\n]*$")
# A ``write``/``wrdata`` command starting a line, anywhere in the deck — not
# just inside the block, since a script could call either from a subckt or a
# second block this pass doesn't otherwise recognize.
_RE_EXISTING_WRITE = re.compile(rb"(?im)^[ \t]*(?:write|wrdata)\b")
# ``quit``/``exit`` end control-script execution; a command placed after one
# would never run, so the injected ``write`` must land before the LAST one.
_RE_QUIT_EXIT = re.compile(rb"(?im)^[ \t]*(?:quit|exit)\b.*$")
# A tail (from just after a quit/exit line to the block's .endc) that is only
# blank lines and ``*`` comments — i.e. the quit/exit was the block's LAST
# statement. Used to tell a script-ending trailing quit from one nested in an
# if/while (which must NOT anchor the injected write, or it lands inside that
# conditional and never runs on the success path).
_RE_TRIVIAL_TAIL = re.compile(rb"(?m)\A(?:[ \t]*(?:\*.*)?(?:\n|\Z))*\Z")


def inject_ngspice_control_write(
    netlist_path: Path, simulator: type, job_id: str, output_folder: Path
) -> Path:
    """Return a runnable netlist with a ``write`` injected into its
    ``.control`` block, for ngspice decks that drive their own analyses via
    scripting.

    This is ngspice runtime behavior, not a spicelib bug: a ``.control``
    block replaces the raw ngspice would otherwise write from the ``-r
    <rawfile>`` switch spicelib always passes — the script runs instead, and
    unless it calls ``write``/``wrdata`` itself, no raw file is ever
    produced. ``collect_run_outcome`` already classifies that as a clean
    log-only completion (not a failure), but nothing then exists for
    get_waveform/signal_stats/etc. to read. Injecting a canonical ``write
    <rawpath>`` gives the deck a raw at the exact path the runner expects for
    this job, so the existing raw>0 code path (raw_parser + every analysis
    tool) picks it up unchanged — no new parser, no new tool.

    Limitation: a bare ``write`` captures ngspice's current/last plot only. A
    script that runs multiple analyses, or writes per Monte-Carlo iteration
    inside a loop, needs its own explicit writes to capture each one — guard
    (c) below leaves any deck that already writes its own output alone
    rather than duplicating or fighting it. A second, unrelated limitation:
    ngspice's ``write`` parser cannot handle a target containing whitespace
    at all — neither quoting nor backslash-escaping works, both fail with
    "No such file or directory" (verified empirically). So a run whose
    output folder path contains a space can't get an auto-injected write
    either (guard (d)) — that run just stays log-only, same as today.

    Guards (all required, or the original path is returned unchanged):
    (a) ngspice only (LTspice has no ``.control``; ``inject_logopinfo``
        covers its own op-point injection separately).
    (b) exactly one ``.control``...``.endc`` block (ambiguous otherwise —
        e.g. which block's last analysis is "the" result).
    (c) no existing ``write``/``wrdata`` anywhere in the deck — never
        override a user who already captures their own output.
    (d) the write target has no whitespace (see the limitation above).

    The ``write`` target is the ABSOLUTE path ``{output_folder}/{job_id}.raw``
    — the same path spicelib's own (suppressed) ``-r`` would use, since it
    derives the rawfile from the staged netlist's own path via
    ``.with_suffix('.raw')``. It must be absolute: the runner's SimRunner
    passes no ``cwd``, so ngspice inherits the MCP server's own working
    directory, not the output folder — a relative ``write`` target would land
    there instead. Written UNQUOTED — see the whitespace limitation above.
    Inserted before the block's LAST ``quit``/``exit`` (if any) so it
    actually runs — those commands end script execution, so a ``write``
    placed after one would never fire; otherwise inserted just before
    ``.endc``.

    Same per-job sibling-file technique as ``runner_base.inject_logopinfo``
    (see its docstring): append-only into a leading-dot, ``job_id``-stamped copy so
    the user's deck is never touched and relative ``.include``/``.lib``
    paths still resolve. Returns the original path when injection doesn't
    apply or the sibling can't be written.

    Scope: single runs only. A sweep/Monte-Carlo batch's per-sub-run raw
    naming isn't static the way a one-shot job's is, so this is not wired
    into those batch paths.
    """
    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    if not (isinstance(simulator, type) and issubclass(simulator, NGspiceSimulator)):
        return netlist_path
    if netlist_path.suffix.lower() not in (".cir", ".net", ".sp"):
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    if _RE_EXISTING_WRITE.search(data):
        return netlist_path
    blocks = list(_RE_CONTROL_BLOCK.finditer(data))
    if len(blocks) != 1:
        return netlist_path
    block = blocks[0]
    body_start, body_end = block.start(1), block.end(1)

    raw_path = (output_folder / f"{job_id}.raw").as_posix()
    # ngspice's `write` parser cannot handle a spaced target at all — not
    # quoted, not escaped (verified empirically) — so a spaced output folder
    # can't get an auto-injected write; that run just stays log-only.
    if any(c.isspace() for c in raw_path):
        return netlist_path
    write_line = f"write {raw_path}\n".encode()

    # Insert before .endc, UNLESS the block's last statement is an
    # unconditional trailing quit/exit — a write after that would never run. A
    # quit/exit nested in an if/while is not the last statement (an ``end`` and
    # possibly more follow it), so anchoring on it is skipped: the write goes
    # before .endc and runs on the normal path.
    insert_at = body_end
    quit_matches = list(_RE_QUIT_EXIT.finditer(data, body_start, body_end))
    if quit_matches and _RE_TRIVIAL_TAIL.match(data[quit_matches[-1].end() : body_end]):
        insert_at = quit_matches[-1].start()
    augmented = data[:insert_at] + write_line + data[insert_at:]

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{NGSPICE_CONTROL_WRITE_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


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
# one shared event loop (mcp.server.lowlevel.Server.run start_soons a task
# per message), so tool handlers run concurrently. Anything that blocks the
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
#     plus a durable fsync write), WSL interop (the first-call cmd.exe spawn
#     inside resolve_output_folder), and resource reads (the whole resource
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


# .include / .inc / .lib / .libfile <path> [extra]
_INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*\.(?:include|inc|lib|libfile)\b\s+(.+)$", re.IGNORECASE)


def _first_path_token(rest: str) -> str:
    """First (possibly quoted) path token of an include/lib directive's args."""
    rest = rest.strip()
    if rest[:1] in ("'", '"'):
        end = rest.find(rest[0], 1)
        if end != -1:
            return rest[1:end]
    parts = rest.split()
    return parts[0] if parts else ""


def _netlist_has_local_dependency(netlist_path: Path) -> bool:
    """True if the netlist pulls in a sibling file via a *relative* .include/.lib.

    Such a netlist can't be relocated to the run sidecar: a simulator resolves a
    relative include against the (now-moved) netlist's own directory, so the
    dependency would no longer be found. Bare library NAMES resolved via the
    simulator's own lib path (no matching local file) and absolute paths both
    survive relocation and don't count.
    """
    from ltspice_mcp.lib.encoding import read_spice_text

    try:
        text = read_spice_text(netlist_path)
    except OSError:
        return True  # unreadable — be conservative, keep it in place
    base = netlist_path.parent
    for line in text.splitlines():
        m = _INCLUDE_DIRECTIVE_RE.match(line)
        if not m:
            continue
        tok = _first_path_token(m.group(1))
        if not tok:
            continue
        # Absolute (POSIX, Windows drive, or UNC) paths survive relocation.
        if Path(tok).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", tok) or tok.startswith("\\\\"):
            continue
        if (base / tok).exists():
            return True
    return False


async def resolve_output_folder(
    state: SessionState,
    netlist_path: Path | None = None,
    simulator: type | None = None,
) -> Path:
    """Determine the output folder for the simulation runner.

    Kept **stable** — one ``{working_dir}/.ltspice-mcp/runs`` sidecar — so the
    single cached runner, ``cancel_job``, and the global ``max_parallel`` cap stay
    valid across runs. A per-deck output dir would change the folder on every run
    in a different directory, and ``RunnerManager`` invalidates the whole runner
    cache when the folder changes (losing in-flight process handles and splitting
    the concurrency semaphore per directory). Each run's artifacts are uniquely
    named (``{job_id}.*``), so they stay isolated within this shared folder; a
    caller finds them through the result path ``check_job`` reports.

    Two overrides:

    - **Relative ``.include``/``.lib`` deck:** the deck's own dir — the simulator
      resolves the relative path against the staged netlist's directory, so it
      can't be relocated (applies to single runs and sweeps/MC alike).
    - **WSL + LTspice + Linux-fs source:** a Windows-native temp dir. LTspice (a
      Windows process reaching the Linux fs over a ``wsl.localhost`` UNC share)
      can't write the SQLite ``.db`` behind ``.MEAS`` over UNC.

    Adds the chosen dir to allowed_paths so analysis tools can read results via
    safe_path(). The Windows temp-dir resolution spawns a cmd.exe interop
    subprocess on first call (memoized), so it runs via ``asyncio.to_thread`` — a
    wedged interop must not freeze the loop; the allowed_paths mutation stays on
    the loop after the await.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    from ltspice_mcp.lib.wsl import get_windows_output_dir, is_windows_native_path, is_wsl

    source_dir = netlist_path.parent if netlist_path is not None else state.working_dir
    has_local_dep = netlist_path is not None and _netlist_has_local_dependency(netlist_path)

    # Override: WSL + LTspice + Linux-fs source → Windows temp (UNC .db failure).
    if is_wsl() and not is_windows_native_path(source_dir) and not has_local_dep:
        sim_cls = simulator or state.default_simulator
        if sim_cls is not None and issubclass(sim_cls, LTspice):
            out = await asyncio.to_thread(get_windows_output_dir)
            if out is not None:
                if out not in state.config.allowed_paths:
                    logger.info(
                        f"WSL: routing LTspice output to {out} (source dir "
                        f"{source_dir} is on the Linux filesystem; .db/.MEAS "
                        "cannot write over UNC)"
                    )
                    state.config.allowed_paths.append(out)
                return out

    # Override: relative-include deck runs in its own dir so the include resolves.
    if has_local_dep:
        return source_dir

    # Default: one stable sidecar; per-job {job_id} naming isolates each run.
    runs = state.working_dir / ".ltspice-mcp" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if runs not in state.config.allowed_paths:
        state.config.allowed_paths.append(runs)
    return runs


# ---------------------------------------------------------------------------
# Schematic include-resolver + symbol-resolver + scene render (shared)
# ---------------------------------------------------------------------------


def make_include_resolver(state: SessionState) -> IncludeResolver:
    """An include resolver that routes every include/lib open through safe_path.

    The graph engine calls this before opening any include, so an in-deck include
    that escapes the allowed roots is denied and never read.
    """

    def resolver(candidate: Path) -> Path | None:
        try:
            return safe_path(str(candidate), state)
        except PathSecurityError:
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


# --- dotted field projection (shared by analyze_results and run_experiments) ---
#
# A keep-plan mirrors the row's own nesting: ``None`` keeps a whole subtree, a
# nested plan keeps only the named keys inside it. Projection therefore returns
# the SAME shape with fewer keys, never a flattened one — caller code that reads
# ``row["value"]["phase_margin_deg"]`` reads it identically either way.
KeepPlan = dict[str, "KeepPlan | None"]

ABSENT = object()


def split_field_path(path: str) -> list[str]:
    r"""Segments of a dotted path, where ``\.`` is a dot INSIDE a key.

    A row key is whatever the simulator called the thing, and plenty of them
    carry dots of their own — ngspice spells a subcircuit node ``v(x1.out)``
    and a subcircuit device parameter ``@m.x1.m1[gm]``. Splitting those on
    every dot addresses a nesting that does not exist, so the escape is what
    makes the most ordinary op-point key projectable at all.
    """
    segments: list[str] = []
    current: list[str] = []
    escaped = False
    for char in path:
        if escaped:
            # Only the dot is escapable; anything else keeps its backslash, so
            # a path that never meant to escape reads back unchanged.
            current.append(char if char == "." else "\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            segments.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    segments.append("".join(current))
    return segments


def escape_field_segment(segment: str) -> str:
    r"""Spell one row key as the path segment that addresses it.

    Only the dot is escaped, mirroring the split: a backslash means nothing on
    its own there, so doubling one here would put a character in the path that
    reading it back would not remove.
    """
    return segment.replace(".", "\\.")


def keep_plan(paths: list[str]) -> KeepPlan:
    """Group dotted ``paths`` into a nested keep-plan, in first-named order."""
    return _plan_for([split_field_path(path) for path in paths])


def _plan_for(paths: list[list[str]]) -> KeepPlan:
    order: list[str] = []
    nested: dict[str, list[list[str]]] = {}
    whole: set[str] = set()
    for segments in paths:
        root, rest = segments[0], segments[1:]
        if root not in order:
            order.append(root)
        if rest:
            nested.setdefault(root, []).append(rest)
        else:
            # A bare key wins over any dotted sibling: asking for the subtree and
            # a leaf inside it means the subtree.
            whole.add(root)
    return {root: None if root in whole else _plan_for(nested[root]) for root in order}


def project_row(row: dict[str, Any], plan: KeepPlan) -> dict[str, Any]:
    """A NEW row carrying only the planned keys.

    Never mutates ``row``: the full record is still read after this call — spec
    attribution, reductions and the analyzed-identity accounting all index keys
    a projection drops — so this is a view built for emission, not an edit.
    """
    kept: dict[str, Any] = {}
    for key, sub in plan.items():
        value = row.get(key, ABSENT)
        if value is ABSENT:
            continue
        if sub is None:
            kept[key] = value
        elif isinstance(value, dict):
            nested = project_row(value, sub)
            if nested:
                kept[key] = nested
    return kept
