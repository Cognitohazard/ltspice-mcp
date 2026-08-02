"""verify_circuit — the non-mutating AUTHOR gate over any circuit file.

One tool checks a ``.asc`` schematic or a ``.cir``/``.net``/``.sp`` netlist and,
optionally, renders it and compares it to a reference. Every checkable-and-
fixable condition is reported in the shared findings shape
(``{rule_id, severity, ok, evidence, at, subject}``); the reference comparison
keeps its own rich, per-mode structure in ``comparison``; the render metadata is
its own ``render`` object.

Checks by file kind:

* ``.asc`` — ``symbols`` (resolution against the configured/stock ``.asy``
  paths), ``export`` (the authoritative LTspice netlist export, plus the wires
  LTspice silently drops), ``layout`` (geometric placement facts), ``quality``
  (label-island and text-in-body hygiene), and ``compare``.
* netlist — ``syntax`` (directive + element arity) and ``compare``. No layout,
  symbol, or export claim is made on a text deck: it carries no geometry and no
  symbol library.

``compare`` runs in one of two modes: ``equivalence`` graph-compares through the
connectivity engine (every include/lib open gated by ``safe_path`` so an in-deck
include that escapes the allowed roots is denied and never read); ``structural_diff``
reuses the shipped ``diff_circuit`` internals for an added/removed/changed delta.

The three channels stay separate. ``observations`` are facts about what the checks
could see — an unresolved symbol drawn as a placeholder, a finding list truncated
at the cap, the layout scan's stated blind spot — each weighed by the caller, none
with a single required action. ``warnings`` are the assumptions a check had to make
for its result to exist: today, a deck in a comparison that could not be parsed and
was therefore diffed as an empty circuit, which makes the other side's whole content
read as added or removed. Those warnings have one home, the top-level channel:
nested inside ``comparison`` they were invisible to the schema, so a structured-only
client read the bogus delta as fact.

An unparsed deck sits in ``warnings`` rather than ``observations`` deliberately. The
doctrine in ``lib/result_observations.py`` routes a *run-level solve failure* to
``observations`` where a tool has that channel, but that rule is about a fact the
simulator produced about a solve; verify_circuit never solves. What it reports here
is its own substitution — it could not read a deck, so it stood an empty circuit in
its place and carried on — which is the ``warnings`` question verbatim ("did this
have to make assumptions?"), free-text with one thing to fix. It also keeps the
channel consistent with ``diff_circuit``, which has surfaced these same messages,
from this same helper, in ``warnings`` all along.

``export_to`` is ``managed`` by default — a non-destructive export into a staged
scratch directory that leaves the caller's files untouched. ``sidecar`` overwrites
the deck's conventional ``<name>.net`` next to the schematic under the export lock,
which is what makes that mode destructive.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.netlist_graph import (
    IncludeResolver,
    NetlistGraph,
    NetlistGraphError,
    compare_graphs,
    parse_netlist_graph,
)
from ltspice_mcp.lib.raster import DEFAULT_SCALE, RenderedImage
from ltspice_mcp.lib.schematic_scene import (
    LayoutIssue,
    NetFlag,
    Scene,
    build_scene,
    layout_issues,
)
from ltspice_mcp.lib.schematic_scene import _point_on_segment as point_on_segment
from ltspice_mcp.lib.spice_lex import SpiceLexError, lex
from ltspice_mcp.lib.spice_validator import (
    drop_title_card,
    validate_directive,
    validate_netlist_arity,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    WARNINGS_SCHEMA,
    StrictModel,
    ToolInput,
    asc_export_lock,
    circuit_file_lock,
    format_response,
    make_include_resolver,
    registry,
    render_scene_artifact,
    safe_path,
    symbol_resolver_for,
)
from ltspice_mcp.tools.circuit import (
    STRUCTURAL_DELTA_PROPS,
    _components_and_directives,
    _norm_micro,
    _same_instance_dropped_segments,
    parse_failure_warnings,
)

# The apply_schematic_ops geometry helpers and diff internals are reused verbatim
# from tools/circuit.py rather than duplicated, so this module imports those
# module-private names by design.
# pyright: reportPrivateUsage=false

NETLIST_SUFFIXES = frozenset({".cir", ".net", ".sp"})

CHECK_ORDER = ("syntax", "symbols", "export", "layout", "quality", "compare")

# Applicable checks per file kind. A netlist carries no geometry and no symbol
# library, so it gets only the text-deck checks — claiming a layout/symbol/export
# result for it would be manufacturing a finding out of an absent capability.
_ASC_CHECKS = frozenset({"symbols", "export", "layout", "quality", "compare"})
_NETLIST_CHECKS = frozenset({"syntax", "compare"})

# Project-local dependencies staged alongside a schematic for a managed export.
# Symbol resolution privileges the schematic's own directory, so exporting a lone
# copy of the .asc strands every project-local symbol and manufactures missing-
# symbol findings for a schematic that is entirely fine.
STAGED_SUFFIXES = frozenset({".asy", ".lib", ".sub", ".inc", ".mod"})
STAGE_FILE_CAP = 200

# Bounded arrays: retained sample per finding rule, and the true count travels
# in the accompanying observation when a rule is truncated.
FINDING_RULE_CAP = 25

# Scene-issue kinds routed to the layout check (geometric placement facts) and
# to the quality check (the text-in-body hygiene fact). label-island is computed
# separately, from the flags-vs-wires geometry.
_LAYOUT_ISSUE_KINDS = (
    "symbol_overlap",
    "wire_through_symbol",
    "floating_pin",
    "dangling_wire_end",
)
_QUALITY_ISSUE_KINDS = ("text_in_symbol_body",)

# Stated inline so an empty findings list is not read as a clean drawing. Kept
# to the one fact that changes what a caller concludes — the scan's blind spot,
# not how bounding boxes are built.
LAYOUT_COVERAGE = "Layout scan excludes WINDOW attribute text."


# ---------------------------------------------------------------------------
# Response fragments
# ---------------------------------------------------------------------------

_FINDING_SCHEMA: dict[str, Any] = {
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

_CHECK_SKIPPED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"check": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["check", "reason"],
}

_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage": {"type": "string"},
        "error": {"type": "string"},
        "where": {"type": ["string", "null"]},
        "remedy": {"type": ["string", "null"]},
    },
    "required": ["stage", "error"],
}

_EXPORT_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "ok": {"type": "boolean"},
        "netlist": {"type": ["string", "null"]},
        "sha256": {"type": ["string", "null"]},
        "components": {"type": ["integer", "null"]},
        "nets": {"type": ["integer", "null"]},
        "destination": {"type": "string"},
        "diff_vs_prior": {
            "type": ["object", "null"],
            "properties": dict(STRUCTURAL_DELTA_PROPS),
            "required": list(STRUCTURAL_DELTA_PROPS),
            "description": (
                "sidecar mode only: the structural delta between the .net that was "
                "already on disk and the one just exported. Null when there was no "
                "prior file or the export was managed."
            ),
        },
    },
    "required": ["ok", "destination"],
}

_RENDER_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "path": {"type": ["string", "null"]},
        "sha256": {"type": ["string", "null"]},
        "width": {"type": ["integer", "null"]},
        "height": {"type": ["integer", "null"]},
        "downscaled": {"type": "boolean"},
        "image_format": {"type": "string"},
        "scale": {"type": ["number", "null"]},
        "bytes": {"type": "integer"},
        "estimated_tokens": {"type": ["integer", "null"]},
        "returned_inline": {"type": "boolean"},
        "delivery": {"type": "string"},
        "note": {"type": ["string", "null"]},
    },
    "required": ["path", "sha256", "width", "height", "downscaled"],
}

_SCENE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "symbols": {"type": "integer"},
        "wires": {"type": "integer"},
        "flags": {"type": "integer"},
        "directives": {"type": "integer"},
        "bbox": {"type": ["array", "null"], "items": {"type": "integer"}},
    },
}

# Equivalence-mode difference records — the JSON projection of the matching
# lib.netlist_graph dataclasses (see GraphComparison.as_dict).
_REF: dict[str, Any] = {
    "type": "string",
    "description": "Component reference, hierarchical for a subcircuit leaf ('X1.M2').",
}

_COMPONENT_DELTA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": _REF,
        "type_letter": {"type": "string", "description": "SPICE element letter (R, C, M, X, …)."},
        "detail": {"type": "string", "description": "What the component is, in words."},
    },
    "required": ["ref", "type_letter", "detail"],
}

_RETYPE_DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": _REF,
        "reference_type": {
            "type": "string",
            "description": "Element type / model name on the reference side.",
        },
        "candidate_type": {
            "type": "string",
            "description": "Element type / model name on this circuit's side.",
        },
    },
    "required": ["ref", "reference_type", "candidate_type"],
}

_VALUE_DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": _REF,
        "reference_value": {
            "type": ["string", "null"],
            "description": "Primary value on the reference side.",
        },
        "candidate_value": {
            "type": ["string", "null"],
            "description": "Primary value on this circuit's side.",
        },
    },
    "required": ["ref", "reference_value", "candidate_value"],
}

_PARAM_DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": _REF,
        "key": {"type": "string", "description": "The key=value parameter name that differs."},
        "reference_value": {
            "type": ["string", "null"],
            "description": "Its value on the reference side.",
        },
        "candidate_value": {
            "type": ["string", "null"],
            "description": "Its value on this circuit's side.",
        },
    },
    "required": ["ref", "key", "reference_value", "candidate_value"],
}

_NODE_PARTITION_DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "One connectivity collision: one reference net forced onto several candidate "
        "nets (a fan-out) or several reference nets merged onto one (a short). Both "
        "sides are named, so the other end never has to be reconstructed."
    ),
    "properties": {
        "reference_nets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The reference-side nets in the collision.",
        },
        "candidate_nets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The candidate-side nets they could not be reconciled with.",
        },
        "involved": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Device pins ('R1.2') that witness the constraint.",
        },
    },
    "required": ["reference_nets", "candidate_nets", "involved"],
}

_ANCHOR_VIOLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor": {"type": "string", "description": "The anchor net name that is misplaced."},
        "detail": {"type": "string", "description": "How its structural position differs."},
    },
    "required": ["anchor", "detail"],
}

_ARITY_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": _REF,
        "reference_arity": {
            "type": "integer",
            "description": "Terminal count on the reference side.",
        },
        "candidate_arity": {
            "type": "integer",
            "description": "Terminal count on this circuit's side.",
        },
    },
    "required": ["ref", "reference_arity", "candidate_arity"],
}

_UNRESOLVED_SUBCKT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "An X instance whose .subckt definition was found nowhere, left as a black "
        "box. A fact, not a difference — it does not flip 'equivalent'."
    ),
    "properties": {
        "name": {"type": "string", "description": "The subcircuit name that was asked for."},
        "side": {
            "type": "string",
            "description": "Which deck asked for it: 'reference' or 'candidate'.",
        },
        "missing_includes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "That side's unusable include targets — empty means the library was "
                "never referenced, non-empty means the definition file is missing."
            ),
        },
    },
    "required": ["name", "side", "missing_includes"],
}

_DUPLICATE_SUBCKT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "A subcircuit defined more than once; SPICE keeps the first in textual "
        "order. A fact, not a difference — it does not flip 'equivalent'."
    ),
    "properties": {
        "name": {"type": "string", "description": "The duplicated subcircuit name."},
        "used": {"type": "string", "description": "The source whose definition won."},
        "ignored": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The sources whose definitions were discarded.",
        },
    },
    "required": ["name", "used", "ignored"],
}

# One flat object over both compare modes rather than a nested oneOf: strict MCP
# clients handle a plain properties map everywhere, and 'mode' says which subset
# is populated. Only 'mode' and 'equivalent' are emitted by both modes.
_COMPARISON_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "description": (
        "The reference comparison, null when no reference was supplied or compare "
        "did not run. 'mode' selects which keys are present."
    ),
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["equivalence", "structural_diff"],
            "description": (
                "'equivalence' populates the graph-comparison keys "
                "(structurally_equivalent, added/removed/retyped, the mismatch lists, "
                "the subckt facts); 'structural_diff' populates the components_*/"
                "directives_* delta."
            ),
        },
        "equivalent": {
            "type": ["boolean", "null"],
            "description": (
                "The single-glance verdict. In equivalence mode: structurally "
                "isomorphic AND no component, value, parameter, anchor or arity "
                "difference. In structural_diff mode: the delta is empty."
            ),
        },
        "structurally_equivalent": {
            "type": ["boolean", "null"],
            "description": (
                "equivalence mode only: the wiring-only verdict, ignoring values and "
                "anchor placement. Can be true while 'equivalent' is false."
            ),
        },
        "added": {
            "type": "array",
            "items": _COMPONENT_DELTA_SCHEMA,
            "description": "equivalence: components present only in this circuit.",
        },
        "removed": {
            "type": "array",
            "items": _COMPONENT_DELTA_SCHEMA,
            "description": "equivalence: components present only in the reference.",
        },
        "retyped": {
            "type": "array",
            "items": _RETYPE_DIFF_SCHEMA,
            "description": "equivalence: matched components whose type or model changed.",
        },
        "value_mismatches": {
            "type": "array",
            "items": _VALUE_DIFF_SCHEMA,
            "description": "equivalence: matched components whose value differs beyond rtol.",
        },
        "param_mismatches": {
            "type": "array",
            "items": _PARAM_DIFF_SCHEMA,
            "description": "equivalence: matched components whose key=value parameter differs.",
        },
        "node_partition_mismatches": {
            "type": "array",
            "items": _NODE_PARTITION_DIFF_SCHEMA,
            "description": "equivalence: the wiring collisions — where a miswire shows up.",
        },
        "anchor_violations": {
            "type": "array",
            "items": _ANCHOR_VIOLATION_SCHEMA,
            "description": "equivalence: anchor nets not in corresponding positions.",
        },
        "arity_errors": {
            "type": "array",
            "items": _ARITY_ERROR_SCHEMA,
            "description": "equivalence: matched components whose terminal count differs.",
        },
        "unresolved_subckts": {
            "type": "array",
            "items": _UNRESOLVED_SUBCKT_SCHEMA,
            "description": "equivalence: subcircuits left as black boxes on either side.",
        },
        "duplicate_subckts": {
            "type": "array",
            "items": _DUPLICATE_SUBCKT_SCHEMA,
            "description": "equivalence: subcircuits defined more than once.",
        },
        **STRUCTURAL_DELTA_PROPS,
    },
    "required": ["mode", "equivalent"],
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "kind": {"type": "string"},
        "outcome": {"type": "string", "enum": ["complete", "partial", "failed"]},
        "checks_run": {"type": "array", "items": {"type": "string"}},
        "checks_skipped": {"type": "array", "items": _CHECK_SKIPPED_SCHEMA},
        "findings": {"type": "array", "items": _FINDING_SCHEMA},
        "comparison": _COMPARISON_SCHEMA,
        "export": _EXPORT_SCHEMA,
        "render": _RENDER_SCHEMA,
        "scene": _SCENE_SCHEMA,
        "observations": {
            **WARNINGS_SCHEMA,
            "description": (
                "What the checks could and could not see: a symbol that resolved "
                "only to a placeholder, a finding list truncated at the cap, the "
                "layout scan's stated blind spot, a render that was skipped. Facts "
                "to weigh — none carries a single required action."
            ),
        },
        "warnings": {
            **WARNINGS_SCHEMA,
            "description": (
                "Assumptions a check had to make for its result to exist at all — "
                "today, a deck in a comparison that could not be parsed and was "
                "diffed as an empty circuit. Each names one thing to fix, so read "
                "these before trusting 'comparison' or 'export.diff_vs_prior'."
            ),
        },
        "failures": {"type": "array", "items": _FAILURE_SCHEMA},
        "hint": {"type": "string"},
    },
    "required": ["path", "kind", "outcome", "checks_run", "findings", "failures"],
}


def _finding(
    *,
    rule_id: str,
    severity: str,
    at: dict[str, Any],
    subject: str,
    evidence: Any,
    ok: bool = False,
) -> dict[str, Any]:
    """One finding in the shared shape. ``at`` + ``subject`` are always present.

    ``severity`` names an input condition — ``error`` for a fault that breaks
    the deck, ``warning`` for a real discrepancy, ``observation`` for a fact the
    model weighs — never a trust verdict.
    """
    return {
        "rule_id": rule_id,
        "severity": severity,
        "ok": ok,
        "evidence": evidence,
        "at": at,
        "subject": subject,
    }


def _failure(
    stage: str, error: str, *, where: str | None = None, remedy: str | None = None
) -> dict[str, Any]:
    """One requested stage that did not produce its result."""
    return {"stage": stage, "error": error, "where": where, "remedy": remedy}


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

_ANCHORS_DESCRIPTION = (
    "Named nets that must map BY NAME between the reference and this circuit — "
    "ports, rails, outputs, measurement nets. A design that is structurally "
    "isomorphic but puts 'vout' in the wrong place fails on these. Ground is "
    "always an implicit anchor. Only meaningful with 'reference' in equivalence mode."
)


class RenderPolicy(StrictModel):
    """How (and whether) to draw the schematic alongside the checks."""

    mode: Literal["with_checks", "only"] = Field(
        default="with_checks",
        description=(
            "'with_checks' renders the drawing in addition to running the checks; "
            "'only' renders and skips every check (a fast look with no analysis)."
        ),
    )
    delivery: Literal["artifact", "inline", "both"] = Field(
        default="artifact",
        description=(
            "'artifact' writes the image to disk and returns only its handle; "
            "'inline' also returns the image as image content in the response; "
            "'both' does both. Inline delivery applies to PNG only — SVG is markup "
            "clients do not render as a picture."
        ),
    )
    format: Literal["png", "svg"] = Field(
        default="png",
        description=(
            "PNG (lossless, what a model looks at) needs the optional 'raster' "
            "extra; without it the render degrades to SVG and a per-item failure "
            "names the missing extra. SVG always works and writes the vector artifact."
        ),
    )
    scale: float = Field(
        default=DEFAULT_SCALE,
        ge=0.5,
        le=4.0,
        description=(
            "Render scale — the cost dial. Image token cost tracks pixel area, so "
            "halving the scale costs about a quarter as much. Raise it only when "
            "detail is genuinely unreadable."
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


class VerifyCircuitInput(ToolInput):
    path: str = Field(description="Circuit to check: .asc, .cir, .net or .sp.")
    checks: list[Literal["syntax", "symbols", "export", "layout", "quality", "compare"]] | None = (
        Field(
            default=None,
            description=(
                "Narrow the checks. Default is every check applicable to this file "
                "type: syntax for netlists; symbols, export, layout and quality for "
                "schematics; compare whenever 'reference' is given. Use this only to "
                "skip something expensive — 'export' runs LTspice."
            ),
        )
    )
    reference: str | None = Field(
        default=None,
        description=(
            "Reference netlist to compare against: a file path, or literal netlist "
            "text (anything containing a newline is read as text). Omit when there "
            "is no reference — the syntax, symbol, export, layout and quality checks "
            "are self-checks and stand on their own."
        ),
    )
    compare_mode: Literal["equivalence", "structural_diff"] = Field(
        default="equivalence",
        description=(
            "'equivalence' graph-compares connectivity (component set, values, "
            "normalized parameters, node partitions by canonical labeling, arity, "
            "and 'anchors'); a deck it cannot read yields a compare failure and no "
            "'comparison' at all, because isomorphism is undefined without both "
            "graphs. 'structural_diff' reports the added/removed/changed component "
            "and directive delta between the two decks; a deck it cannot read is "
            "diffed as empty, so the delta still comes back but 'equivalent' is null "
            "and a warning names the deck that failed. Either way the outcome is "
            "'partial' and the reason is in the response."
        ),
    )
    anchors: list[str] | None = Field(default=None, description=_ANCHORS_DESCRIPTION)
    rtol: float = Field(
        default=1e-6,
        description="Relative tolerance when comparing numeric values and parameters (equivalence).",
    )
    render: RenderPolicy | None = Field(
        default=None,
        description="Draw the .asc alongside (or instead of) the checks. Omit to skip rendering.",
    )
    export_to: Literal["managed", "sidecar"] = Field(
        default="managed",
        description=(
            "Where the exported netlist goes. 'managed' writes into the server's "
            "scratch directory and leaves your files untouched. 'sidecar' writes the "
            "conventional <name>.net next to the schematic, OVERWRITING any existing "
            "one — use it only when you want that file on disk."
        ),
    )


VERIFY_DESCRIPTION = (
    "Check a circuit file without changing it, and optionally render it. For a "
    ".cir/.net/.sp: SPICE syntax, directive and element arity. For an .asc: symbol "
    "and pin resolution, the authoritative LTspice netlist export (which silently "
    "drops wires the file appears to contain), geometric layout facts (overlapping "
    "bodies, wires through a body, floating pins, dangling wire ends), and quality "
    "facts (net connected only by label stubs with no drawn wire; text anchored "
    "inside a symbol). Supply 'reference' to graph-compare against a known-good "
    "netlist (equivalence) or take an added/removed/changed delta (structural_diff). "
    "Every fixable finding carries its location and subject. By default nothing is "
    "written outside the server's scratch directory."
)


# ---------------------------------------------------------------------------
# Scene + resolver helpers
# ---------------------------------------------------------------------------


def _analyze_scene(
    asc_path: Path, state: SessionState, *, compute_issues: bool
) -> tuple[Scene, list[LayoutIssue]]:
    """Build the scene and, only when needed, compute its layout issues.

    ``layout_issues`` is an O(n²) geometric scan; skip it unless a check that
    consumes it (layout or quality) is going to run, so a pure render does not
    pay for it.
    """
    scene = build_scene(asc_path, resolver=symbol_resolver_for(asc_path, state))
    issues = layout_issues(scene) if compute_issues else []
    return scene, issues


def _resolve_reference(reference: str, state: SessionState) -> str | Path:
    """A reference is literal netlist text when it spans lines, else a safe path."""
    if "\n" in reference:
        return reference
    return safe_path(reference, state)


def _reference_to_path(reference: str | Path, state: SessionState) -> Path:
    """A filesystem path for the reference — text references are staged to scratch.

    ``structural_diff`` reuses the editor-based diff internals, which read a file,
    so literal reference text is materialized under the managed scratch directory.
    """
    if isinstance(reference, Path):
        return reference
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()[:12]
    dest = _scratch_dir(state, "reference") / f"{digest}.cir"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(reference, encoding="utf-8")
    return dest


def _scratch_dir(state: SessionState, name: str) -> Path:
    """A server-owned scratch subdirectory (created on demand)."""
    path = state.working_dir / ".ltspice-mcp" / "verify" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# syntax
# ---------------------------------------------------------------------------


def _syntax_findings(text: str, path: Path) -> list[dict[str, Any]]:
    """Directive and element-arity findings in a netlist, in the shared shape."""
    findings: list[dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("."):
            continue
        err = validate_directive(line)
        if err is None:
            continue
        detail = err.message + (f" {err.suggestion}" if err.suggestion else "")
        findings.append(
            _finding(
                rule_id="directive_syntax",
                severity="error",
                at={"file": str(path), "line": lineno},
                subject=line,
                evidence={"detail": detail, "card": line},
            )
        )

    try:
        cards = lex(text).cards
    except SpiceLexError as exc:
        findings.append(
            _finding(
                rule_id="lex_error",
                severity="error",
                at={"file": str(path)},
                subject=str(path.name),
                evidence={"detail": str(exc)},
            )
        )
        cards = []
    # Line 1 is a free-text title by SPICE convention; the lexer has no title
    # concept, so a title beginning with an element letter parses as an element.
    for issue in validate_netlist_arity(drop_title_card(cards)):
        detail = str(issue.get("message", ""))
        suggestion = issue.get("suggestion")
        if suggestion:
            detail = f"{detail} {suggestion}"
        card = str(issue.get("directive", ""))
        at: dict[str, Any] = {"file": str(path)}
        line_no = issue.get("line")
        if isinstance(line_no, int):
            at["line"] = line_no
        findings.append(
            _finding(
                rule_id="element_arity",
                severity="error",
                at=at,
                subject=card or str(path.name),
                evidence={"detail": detail, "card": card},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# symbols / layout / quality (scene-derived)
# ---------------------------------------------------------------------------


def _symbol_findings(scene: Scene, path: Path) -> list[dict[str, Any]]:
    """Unresolved-symbol findings — each drawn as a placeholder box."""
    grouped: dict[str, list[str]] = {}
    for sym in scene.symbols:
        if sym.missing:
            grouped.setdefault(sym.symbol, []).append(sym.reference)
    findings: list[dict[str, Any]] = []
    for name, refs in sorted(grouped.items()):
        findings.append(
            _finding(
                rule_id="unresolved_symbol",
                severity="error",
                at={"file": str(path)},
                subject=name,
                evidence={
                    "symbol": name,
                    "instances": refs,
                    "detail": (
                        "drawn as a placeholder box; searched the schematic directory, "
                        "the configured symbol paths, and the stock library"
                    ),
                },
            )
        )
    return findings


def _issue_finding(issue: LayoutIssue, path: Path, severity: str) -> dict[str, Any]:
    """Project a scene LayoutIssue into the shared finding shape."""
    at: dict[str, Any] = {"file": str(path)}
    coord = issue.coords[0] if issue.coords else None
    if coord is not None:
        at["x"], at["y"] = int(coord[0]), int(coord[1])
    if issue.refs:
        subject = " & ".join(issue.refs)
    elif coord is not None:
        subject = f"({coord[0]},{coord[1]})"
    else:
        subject = issue.kind
    return _finding(
        rule_id=issue.kind,
        severity=severity,
        at=at,
        subject=subject,
        evidence={
            "detail": issue.detail,
            "refs": list(issue.refs),
            "coords": [list(c) for c in issue.coords],
        },
    )


def _issue_findings(
    issues: list[LayoutIssue], path: Path, kinds: tuple[str, ...], severity: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return every finding for the requested issue kinds and counts by rule."""
    findings: list[dict[str, Any]] = []
    total: dict[str, int] = {}
    for issue in issues:
        if issue.kind not in kinds:
            continue
        total[issue.kind] = total.get(issue.kind, 0) + 1
        findings.append(_issue_finding(issue, path, severity))
    return findings, total


def _label_island_findings(scene: Scene, path: Path) -> list[dict[str, Any]]:
    """Nets connected only by label stubs with zero drawn wires.

    A schematic's signal nets should be joined by drawn wires; a net whose only
    connection is two or more identically-named net-label stubs, with no wire on
    any of them, is a "label island" — electrically valid but a netlist wearing
    symbols. Ground (and any flag LTspice treats as ground) is exempt: connecting
    ground by flag is standard practice. Surfaced as observation-severity facts;
    whether a given rail is acceptable that way is the model's call.
    """
    groups: dict[str, list[NetFlag]] = {}
    for flag in scene.flags:
        if flag.is_ground:
            continue
        name = flag.text.strip()
        if not name:
            continue
        groups.setdefault(name, []).append(flag)

    segments = [((w.x1, w.y1), (w.x2, w.y2)) for w in scene.wires if (w.x1, w.y1) != (w.x2, w.y2)]
    findings: list[dict[str, Any]] = []
    for name, flags in sorted(groups.items()):
        if len(flags) < 2:
            continue  # a lone label is not a by-name connection replacing a wire
        wired = any(
            point_on_segment((flag.x, flag.y), a, b) for flag in flags for a, b in segments
        )
        if wired:
            continue
        coords = [[flag.x, flag.y] for flag in flags]
        findings.append(
            _finding(
                rule_id="label_island",
                severity="observation",
                at={"file": str(path), "x": flags[0].x, "y": flags[0].y},
                subject=name,
                evidence={
                    "net": name,
                    "stub_count": len(flags),
                    "coords": coords,
                    "detail": (
                        f"net '{name}' is connected by {len(flags)} net-label stubs and "
                        "no drawn wire segment"
                    ),
                },
            )
        )
    return findings


def _dropped_wire_findings(scene: Scene, path: Path) -> list[dict[str, Any]]:
    """Wires present in the drawing but absent from the exported netlist.

    LTspice drops a run whose two ends both land on pins of the SAME instance:
    the pins stay on separate nodes, so the ``.asc`` shows a tie the netlist does
    not have. The rule is the schematic editor's LTspice-verified one.
    """
    owners: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for sym in scene.symbols:
        for pin in sym.pins:
            owners.setdefault((pin.x, pin.y), []).append((sym.reference, ""))
    segments = [(w.x1, w.y1, w.x2, w.y2) for w in scene.wires]

    findings: list[dict[str, Any]] = []
    for drop in _same_instance_dropped_segments(owners, segments):
        x1, y1, x2, y2 = drop["segment"]
        findings.append(
            _finding(
                rule_id="dropped_wire",
                severity="warning",
                at={"file": str(path), "x": int(x1), "y": int(y1)},
                subject=drop["ref"],
                evidence={
                    "detail": (
                        f"wire joins two pins of the same instance {drop['ref']} and is "
                        "not exported"
                    ),
                    "refs": [drop["ref"]],
                    "coords": [[int(x1), int(y1)], [int(x2), int(y2)]],
                },
            )
        )
    return findings


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _stage_export_inputs(asc_path: Path, dest: Path) -> tuple[Path, int, bool]:
    """Copy a schematic and its project-local dependencies into ``dest``.

    Returns the staged ``.asc``, the dependency count, and whether the file cap
    was hit. Relative structure is preserved so a symbol referenced as
    ``sub/thing`` still resolves; dot-directories are skipped.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    staged_asc = dest / asc_path.name
    shutil.copy2(asc_path, staged_asc)

    copied = 0
    truncated = False
    root = asc_path.parent
    for src in sorted(root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in STAGED_SUFFIXES:
            continue
        rel = src.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if copied >= STAGE_FILE_CAP:
            truncated = True
            break
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied += 1
    return staged_asc, copied, truncated


def _create_netlist(simulator_cls: Any, asc_path: Path, timeout: float) -> Path:
    """Drive LTspice to export ``asc_path`` to a sibling ``.net``."""
    return Path(simulator_cls.create_netlist(str(asc_path), timeout=timeout))


def _netlist_counts(net_path: Path) -> tuple[int | None, int | None]:
    """Component and net counts of an exported netlist, or ``(None, None)``."""
    try:
        graph = parse_netlist_graph(net_path)
    except (NetlistGraphError, OSError, SpiceLexError):
        return (None, None)
    nets = {node for comp in graph.components for node in comp.nodes}
    return (len(graph.components), len(nets))


def _file_digest(path: Path, length: int | None = None) -> str | None:
    """Hex SHA-256 of ``path``'s bytes, truncated to ``length`` characters.

    Returns ``None`` when the file can't be read and no ``length`` is requested —
    a full content hash has no meaningful fallback. When a truncated digest is
    requested (a scratch-name stamp), an unreadable file falls back to hashing the
    path string so a stable name is always available.
    """
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        if length is None:
            return None
        digest = hashlib.sha256(str(path).encode()).hexdigest()
    return digest[:length] if length is not None else digest


async def _run_export(
    asc_path: Path, state: SessionState, export_to: str, simulator_cls: Any
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str], list[str]]:
    """Export a schematic to a netlist, managed (scratch copy) or sidecar (in place).

    Returns ``(export_payload, failure_or_none, observations, warnings)``. The
    ``sidecar`` mode runs under the export lock (it overwrites ``<name>.net``) and
    records a structural ``diff_vs_prior``; ``managed`` stages the schematic with its
    project-local assets and exports there, touching none of the caller's files.
    """
    observations: list[str] = []
    warnings: list[str] = []
    timeout = state.config.default_timeout
    payload: dict[str, Any] = {
        "ok": False,
        "netlist": None,
        "sha256": None,
        "components": None,
        "nets": None,
        "destination": export_to,
        "diff_vs_prior": None,
    }

    try:
        if export_to == "sidecar":
            net_path = asc_path.with_suffix(".net")
            async with asc_export_lock(asc_path):
                prior = net_path.read_bytes() if net_path.exists() else None
                new_path = await asyncio.to_thread(
                    _create_netlist, simulator_cls, asc_path, timeout
                )
                if prior is not None and new_path.exists():
                    payload["diff_vs_prior"], diff_warnings = await asyncio.to_thread(
                        _diff_vs_prior, prior, new_path, state
                    )
                    warnings.extend(diff_warnings)
                net_path = new_path
        else:
            scratch = (
                _scratch_dir(state, "export") / f"{asc_path.stem}.{_file_digest(asc_path, 8)}"
            )
            async with circuit_file_lock(asc_path):
                staged_asc, _staged, truncated = await asyncio.to_thread(
                    _stage_export_inputs, asc_path, scratch
                )
            if truncated:
                observations.append(
                    f"only the first {STAGE_FILE_CAP} project-local symbol/library files were "
                    "staged for the export, so a symbol beyond that may not resolve"
                )
            async with asc_export_lock(staged_asc):
                net_path = await asyncio.to_thread(
                    _create_netlist, simulator_cls, staged_asc, timeout
                )
    except Exception as exc:  # the simulator is a subprocess; any failure is data
        return (
            payload,
            _failure(
                "export",
                f"LTspice netlist export failed: {exc}",
                where=str(asc_path),
                remedy="drop 'export' from checks to run the offline checks only",
            ),
            observations,
            warnings,
        )

    if not net_path.exists():
        return (
            payload,
            _failure(
                "export",
                "LTspice exited without an error but produced no .net file",
                where=str(net_path),
            ),
            observations,
            warnings,
        )

    components, nets = await asyncio.to_thread(_netlist_counts, net_path)
    payload.update(
        {
            "ok": True,
            "netlist": str(net_path),
            "sha256": _file_digest(net_path),
            "components": components,
            "nets": nets,
        }
    )
    return payload, None, observations, warnings


def _diff_vs_prior(
    prior_bytes: bytes, new_path: Path, state: SessionState
) -> tuple[dict[str, Any], list[str]]:
    """Structural delta between the sidecar .net's prior content and the fresh one.

    The staged copy of the prior content is named for what it is, not for its
    digest alone, so a parse warning naming that file still identifies it.
    """
    scratch = _scratch_dir(state, "prior")
    stamp = hashlib.sha256(prior_bytes).hexdigest()[:12]
    prior_path = scratch / f"prior-{new_path.stem}.{stamp}.net"
    prior_path.write_bytes(prior_bytes)
    try:
        delta, warnings, _both_parsed = _structural_diff(prior_path, new_path)
        return delta, warnings
    finally:
        with contextlib.suppress(OSError):
            prior_path.unlink()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _denied_include_findings(graph: NetlistGraph, source: Path) -> list[dict[str, Any]]:
    """path_denied findings for includes the resolver refused (never read)."""
    findings: list[dict[str, Any]] = []
    for missing in graph.missing_includes:
        if "denied" not in missing.reason.lower():
            continue
        findings.append(
            _finding(
                rule_id="path_denied",
                severity="error",
                at={"file": str(source)},
                subject=missing.target,
                evidence={
                    "target": missing.target,
                    "reason": missing.reason,
                    "detail": (
                        f"include '{missing.target}' resolves outside the allowed roots; "
                        "it was denied and never read"
                    ),
                },
            )
        )
    return findings


def _parse_graph_or_fail(
    source: str | Path,
    label: str,
    where: Path,
    resolver: IncludeResolver,
    *,
    base_dir: Path | None = None,
) -> tuple[NetlistGraph | None, dict[str, Any] | None]:
    """Parse a netlist for comparison, or produce a ``compare`` failure.

    Returns ``(graph, None)`` on success and ``(None, failure)`` on a parse error,
    naming ``label`` (e.g. "reference netlist") and locating it at ``where``.
    """
    try:
        graph = parse_netlist_graph(source, base_dir=base_dir, include_resolver=resolver)
    except (NetlistGraphError, SpiceLexError, OSError) as exc:
        return None, _failure(
            "compare", f"the {label} could not be parsed: {exc}", where=str(where)
        )
    return graph, None


# Both compare modes return this, in this order, so the call site destructures
# one shape regardless of mode: (comparison, findings, failure, warnings). Each
# mode leaves the channel it does not use empty — equivalence fails hard on a
# parse error rather than warning, structural_diff routes everything through
# warnings rather than findings.
CompareResult = tuple[
    dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None, list[str]
]


def _compare_equivalence(
    reference: str | Path,
    candidate: str | Path,
    ref_source: Path,
    cand_source: Path,
    anchors: list[str] | None,
    rtol: float,
    resolver: IncludeResolver,
) -> CompareResult:
    """Graph-compare candidate against reference through the safe_path resolver.

    ``candidate`` may be the already-read netlist text (parsed with ``cand_source``
    as the include base dir) or a path; ``cand_source`` is the location reported in
    findings and failures either way.
    """
    findings: list[dict[str, Any]] = []
    ref_graph, failure = _parse_graph_or_fail(reference, "reference netlist", ref_source, resolver)
    if failure is not None:
        return None, findings, failure, []
    cand_graph, failure = _parse_graph_or_fail(
        candidate, "netlist under test", cand_source, resolver, base_dir=cand_source.parent
    )
    if failure is not None:
        return None, findings, failure, []
    assert ref_graph is not None and cand_graph is not None  # failure is None ⇒ both parsed
    findings.extend(_denied_include_findings(ref_graph, ref_source))
    findings.extend(_denied_include_findings(cand_graph, cand_source))
    comparison = compare_graphs(ref_graph, cand_graph, anchors=anchors, rtol=rtol)
    payload: dict[str, Any] = {"mode": "equivalence", **comparison.as_dict()}
    return payload, findings, None, []


def _by_directive_key(directives: set[str]) -> dict[str, list[str]]:
    """Group directives by a case- and micro-insensitive key, dropping ``.end``."""
    by_key: dict[str, list[str]] = {}
    for d in directives:
        if d.strip().lower() == ".end":
            continue
        by_key.setdefault(_norm_micro(d.lower()), []).append(d)
    return by_key


def _structural_diff(ref_path: Path, cand_path: Path) -> tuple[dict[str, Any], list[str], bool]:
    """Added/removed/changed component and directive delta (diff_circuit internals).

    Returns ``(delta, warnings, both_parsed)``. A deck that could not be parsed is
    diffed as an empty circuit, which makes every component of the other side look
    added or removed — so the caveat rides out as a warning rather than inside the
    delta, where the caller browsing the difference lists would never look for it.

    ``both_parsed`` is reported separately rather than inferred from an empty
    ``warnings`` list, so a future warning of some other kind cannot silently be
    read as a parse failure.
    """
    a, da, err_a = _components_and_directives(ref_path)
    b, db, err_b = _components_and_directives(cand_path)
    warnings = parse_failure_warnings([(ref_path.name, err_a), (cand_path.name, err_b)])

    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed: list[dict[str, str]] = [
        {"reference": ref, "before": a[ref], "after": b[ref]}
        for ref in sorted(set(a) & set(b))
        if _norm_micro(a[ref]) != _norm_micro(b[ref])
    ]

    da_by = _by_directive_key(da)
    db_by = _by_directive_key(db)
    directives_added = sorted(d for k in db_by.keys() - da_by.keys() for d in db_by[k])
    directives_removed = sorted(d for k in da_by.keys() - db_by.keys() for d in da_by[k])

    delta = {
        "components_added": added,
        "components_removed": removed,
        "components_changed": changed,
        "directives_added": directives_added,
        "directives_removed": directives_removed,
    }
    return delta, warnings, err_a is None and err_b is None


def _compare_structural(ref_path: Path, candidate: Path) -> CompareResult:
    """structural_diff mode: reuse the shipped diff internals."""
    try:
        diff, warnings, both_parsed = _structural_diff(ref_path, candidate)
    except (OSError, ValueError) as exc:
        return (
            None,
            [],
            _failure("compare", f"structural diff failed: {exc}", where=str(candidate)),
            [],
        )
    # An unparsed deck is diffed as an empty circuit, so the delta describes a
    # circuit nothing read. Two decks that both failed then produce an EMPTY
    # delta, and an empty delta otherwise means "these match" — success reported
    # for a comparison that compared nothing. The verdict is not derivable from a
    # fabricated side in either direction, so it is null and the warning says why.
    #
    # Keyed on the delta's own difference lists rather than ``any(diff.values())``:
    # a metadata key added to the delta later must not read as a difference.
    equivalent = not any(diff[key] for key in STRUCTURAL_DELTA_PROPS) if both_parsed else None
    return {"mode": "structural_diff", "equivalent": equivalent, **diff}, [], None, warnings


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _render_scene(
    scene: Scene,
    state: SessionState,
    *,
    image_format: Literal["png", "svg"],
    scale: float,
    max_pixels: int | None,
) -> tuple[RenderedImage, Path, bool]:
    """Render a scene into the verify scratch dir. Returns (image, path, downscaled)."""
    return render_scene_artifact(
        scene,
        _scratch_dir(state, "renders"),
        image_format=image_format,
        scale=scale,
        max_pixels=max_pixels,
    )


async def _do_render(
    scene: Scene | None,
    kind: str,
    policy: RenderPolicy,
    path: Path,
    state: SessionState,
) -> tuple[dict[str, Any] | None, RenderedImage | None, list[dict[str, Any]], list[str]]:
    """The render check, factored like the other checks.

    Returns ``(render_payload, inline_image, failures, observations)``. Only .asc
    schematics render; a missing scene is a per-item failure.
    """
    if kind != "asc":
        return None, None, [], ["render skipped: only .asc schematics can be rendered"]
    if scene is None:
        return (
            None,
            None,
            [_failure("render", "the schematic could not be parsed", where=str(path))],
            [],
        )
    try:
        image, out_path, downscaled = await asyncio.to_thread(
            _render_scene,
            scene,
            state,
            image_format=policy.format,
            scale=policy.scale,
            max_pixels=policy.max_pixels,
        )
    except (OSError, ValueError) as exc:
        return None, None, [_failure("render", str(exc), where=str(path))], []

    failures: list[dict[str, Any]] = []
    if policy.format == "png" and not image.is_raster:
        failures.append(
            _failure(
                "render",
                "PNG was requested but the optional 'raster' extra is not "
                "installed; returned SVG instead",
                remedy="install the 'raster' extra (pip install 'ltspice-mcp[raster]')",
            )
        )
    want_inline = policy.delivery in ("inline", "both") and image.is_raster
    payload = _render_payload(
        image,
        out_path,
        downscaled=downscaled,
        delivery=policy.delivery,
        returned_inline=want_inline,
    )
    return payload, (image if want_inline else None), failures, []


def _render_payload(
    image: RenderedImage, path: Path, *, downscaled: bool, delivery: str, returned_inline: bool
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(image.data).hexdigest(),
        "width": image.width,
        "height": image.height,
        "downscaled": downscaled,
        "image_format": image.image_format,
        "scale": image.scale,
        "bytes": len(image.data),
        "estimated_tokens": image.estimated_tokens,
        "returned_inline": returned_inline,
        "delivery": delivery,
        "note": image.note,
    }


def _image_content(image: RenderedImage) -> types.ImageContent:
    return types.ImageContent(
        type="image",
        data=base64.b64encode(image.data).decode("ascii"),
        mimeType=image.mime_type,
    )


# ---------------------------------------------------------------------------
# outcome + hint
# ---------------------------------------------------------------------------


def _comparison_unverified(comparison: dict[str, Any] | None) -> bool:
    """The comparison ran but could not reach a verdict (a side was unparseable)."""
    return comparison is not None and comparison.get("equivalent") is None


def _comparison_mismatch(comparison: dict[str, Any] | None) -> bool:
    """Anything short of a positive match — a real difference OR no verdict at all.

    Deliberately not ``not equivalent``: only ``True`` is a clean result, so a null
    verdict keeps the outcome off ``complete`` instead of falling through it.
    """
    if comparison is None:
        return False
    return comparison.get("equivalent") is not True


def _outcome(
    findings: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> str:
    if failures:
        return "partial"
    if any(f["severity"] in ("error", "warning") for f in findings):
        return "partial"
    if _comparison_mismatch(comparison):
        return "partial"
    return "complete"


def _hint(data: dict[str, Any]) -> str:
    """One concrete thing the caller most needs to know."""
    failures = data["failures"]
    if failures:
        first = failures[0]
        remedy = first.get("remedy")
        return f"{first['stage']} failed: {first['error']}" + (f" — {remedy}" if remedy else "")
    parts: list[str] = []
    # A warning says a result exists only because something was assumed, so it
    # leads: without it "no problems found" reads as a clean bill of health over a
    # comparison that was built on a deck nothing could parse.
    warnings_channel = data.get("warnings") or []
    if warnings_channel:
        parts.append(warnings_channel[0].rstrip("."))
    errors = [f for f in data["findings"] if f["severity"] == "error"]
    # Findings of warning severity — a different thing from the top-level
    # warnings channel read above, which is why neither is just "warnings".
    warning_findings = [f for f in data["findings"] if f["severity"] == "warning"]
    if errors:
        parts.append(
            f"{len(errors)} error finding(s): " + ", ".join(sorted({f["rule_id"] for f in errors}))
        )
    if warning_findings:
        parts.append(
            f"{len(warning_findings)} warning(s): "
            + ", ".join(sorted({f["rule_id"] for f in warning_findings}))
        )
    comparison = data.get("comparison")
    if _comparison_unverified(comparison):
        # Not the same news as a mismatch: nothing was compared, so telling the
        # caller the decks "did not match" would invent a difference.
        parts.append("reference comparison reached no verdict — see warnings")
    elif _comparison_mismatch(comparison):
        parts.append("reference comparison did not match — see comparison")
    observations = [f for f in data["findings"] if f["severity"] == "observation"]
    if observations and not parts:
        parts.append(
            f"{len(observations)} layout/quality observation(s): "
            + ", ".join(sorted({f["rule_id"] for f in observations}))
            + " (facts, not a verdict)"
        )
    if not parts:
        skipped = data.get("checks_skipped") or []
        base = "No problems found in the checks that ran."
        if skipped:
            base += " Not run: " + ", ".join(f"{s['check']} ({s['reason']})" for s in skipped)
        return base
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FindingCapSummary:
    """A presentation note inserted where one capped rule family was evaluated."""

    totals: dict[str, int]


@dataclass(frozen=True)
class VerifyCircuitEvaluation:
    """Neutral verify result before the MCP finding cap is applied.

    ``data["findings"]`` contains every evaluated finding.  The remaining
    fields are the same evaluator facts used by the wire renderer; the inline
    image is held separately because it is MCP content rather than structured
    data.
    """

    data: dict[str, Any]
    inline_image: RenderedImage | None = None
    is_error: bool = False
    capped_rules: frozenset[str] = frozenset()
    observation_events: tuple[str | _FindingCapSummary, ...] = ()


def _base_data(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "unknown",
        "outcome": "failed",
        "checks_run": [],
        "checks_skipped": [],
        "findings": [],
        "comparison": None,
        "export": None,
        "render": None,
        "scene": None,
        "observations": [],
        "warnings": [],
        "failures": [],
    }


def _error_evaluation(data: dict[str, Any], hint: str) -> VerifyCircuitEvaluation:
    data["hint"] = hint
    return VerifyCircuitEvaluation(data=data, is_error=True)


async def evaluate_verify_circuit(
    args: VerifyCircuitInput, state: SessionState
) -> VerifyCircuitEvaluation:
    """Evaluate every requested check without applying MCP finding caps."""
    data = _base_data(args.path)

    try:
        path = safe_path(args.path, state)
        reference = _resolve_reference(args.reference, state) if args.reference else None
    except PathSecurityError as exc:
        data["findings"] = [
            _finding(
                rule_id="path_denied",
                severity="error",
                at={"file": args.path},
                subject=args.path,
                evidence={"detail": str(exc)},
            )
        ]
        return _error_evaluation(data, str(exc))

    data["path"] = str(path)
    if not path.is_file():
        return _error_evaluation(data, f"'{path}' does not exist or is not a file")

    suffix = path.suffix.lower()
    if suffix != ".asc" and suffix not in NETLIST_SUFFIXES:
        return _error_evaluation(
            data,
            f"'{suffix}' is not a circuit file this tool can check; pass a .asc "
            "schematic or a .cir / .net / .sp netlist",
        )

    kind = "asc" if suffix == ".asc" else "netlist"
    kind_label = ".asc" if kind == "asc" else "netlist"
    data["kind"] = kind
    applicable = _ASC_CHECKS if kind == "asc" else _NETLIST_CHECKS
    requested = set(args.checks) if args.checks is not None else None
    render_only = args.render is not None and args.render.mode == "only"

    findings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    checks_run: list[str] = []
    skipped: list[dict[str, str]] = []
    capped_rules: set[str] = set()
    observation_events: list[str | _FindingCapSummary] = []

    def skip(check: str, reason: str) -> None:
        skipped.append({"check": check, "reason": reason})

    wanted: dict[str, bool] = {}
    for check in CHECK_ORDER:
        if render_only:
            skip(check, "render.mode='only'")
            wanted[check] = False
            continue
        if check == "compare" and reference is None:
            skip(check, "no reference supplied")
            wanted[check] = False
            continue
        if check not in applicable:
            skip(check, f"not applicable to a {kind_label} file")
            wanted[check] = False
            continue
        if requested is not None and check not in requested:
            skip(check, "not requested")
            wanted[check] = False
            continue
        wanted[check] = True

    # --- netlist text (syntax) ---------------------------------------------
    text: str | None = None
    if kind == "netlist" and (wanted.get("syntax") or wanted.get("compare")):
        try:
            text = await asyncio.to_thread(read_spice_text, path)
        except OSError as exc:
            failures.append(_failure("read", str(exc), where=str(path)))

    if wanted.get("syntax") and text is not None:
        findings.extend(await asyncio.to_thread(_syntax_findings, text, path))
        checks_run.append("syntax")

    # --- scene-derived checks (symbols, layout, quality, dropped wires) -----
    scene: Scene | None = None
    scene_issues: list[LayoutIssue] = []
    needs_scene = kind == "asc" and (
        wanted.get("symbols")
        or wanted.get("layout")
        or wanted.get("quality")
        or wanted.get("export")
        or args.render is not None
    )
    want_issues = bool(wanted.get("layout") or wanted.get("quality"))
    if needs_scene:
        try:
            scene, scene_issues = await asyncio.to_thread(
                _analyze_scene, path, state, compute_issues=want_issues
            )
        except (OSError, ValueError) as exc:
            failures.append(_failure("scene", str(exc), where=str(path)))

    if scene is not None:
        observation_events.extend(scene.diagnostics)
        bbox = scene.content_bbox()
        data["scene"] = {
            "symbols": len(scene.symbols),
            "wires": len(scene.wires),
            "flags": len(scene.flags),
            "directives": len(scene.directives),
            "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2] if bbox is not None else None,
        }
        if wanted.get("symbols"):
            findings.extend(_symbol_findings(scene, path))
            checks_run.append("symbols")
        if wanted.get("layout"):
            layout_findings, totals = _issue_findings(
                scene_issues, path, _LAYOUT_ISSUE_KINDS, "observation"
            )
            findings.extend(layout_findings)
            capped_rules.update(totals)
            observation_events.append(_FindingCapSummary(totals))
            observation_events.append(LAYOUT_COVERAGE)
            checks_run.append("layout")
        if wanted.get("quality"):
            quality_findings, totals = _issue_findings(
                scene_issues, path, _QUALITY_ISSUE_KINDS, "observation"
            )
            quality_findings.extend(_label_island_findings(scene, path))
            findings.extend(quality_findings)
            capped_rules.update(totals)
            observation_events.append(_FindingCapSummary(totals))
            checks_run.append("quality")

    # --- export -------------------------------------------------------------
    candidate: Path | None = path if kind == "netlist" else None
    if wanted.get("export"):
        simulator_cls = state.available_simulators.get("ltspice")
        if simulator_cls is None:
            skip("export", "LTspice not detected")
        else:
            export_payload, export_failure, export_obs, export_warnings = await _run_export(
                path, state, args.export_to, simulator_cls
            )
            observation_events.extend(export_obs)
            warnings.extend(export_warnings)
            if scene is not None:
                dropped = _dropped_wire_findings(scene, path)
                findings.extend(dropped)
                if dropped:
                    capped_rules.add("dropped_wire")
            data["export"] = export_payload
            checks_run.append("export")
            if export_failure is not None:
                failures.append(export_failure)
            elif export_payload.get("netlist"):
                candidate = Path(export_payload["netlist"])

    # --- compare ------------------------------------------------------------
    if wanted.get("compare") and reference is not None:
        if candidate is None:
            skip("compare", "the exported netlist is required and the export did not run")
        else:
            ref_source = reference if isinstance(reference, Path) else path
            compared: CompareResult
            if args.compare_mode == "equivalence":
                # Reuse the netlist text already read for syntax, so the candidate
                # is not read+lexed a second time; the export path has no such text.
                cand_input: str | Path = (
                    text if kind == "netlist" and text is not None else candidate
                )
                compared = await asyncio.to_thread(
                    _compare_equivalence,
                    reference,
                    cand_input,
                    ref_source,
                    candidate,
                    args.anchors,
                    args.rtol,
                    make_include_resolver(state),
                )
            else:
                ref_path = _reference_to_path(reference, state)
                compared = await asyncio.to_thread(_compare_structural, ref_path, candidate)
            comparison, cmp_findings, cmp_failure, cmp_warnings = compared
            findings.extend(cmp_findings)
            warnings.extend(cmp_warnings)
            if cmp_failure is not None:
                failures.append(cmp_failure)
            else:
                data["comparison"] = comparison
                checks_run.append("compare")

    # --- render -------------------------------------------------------------
    inline_image: RenderedImage | None = None
    if args.render is not None:
        render_payload, inline_image, render_failures, render_obs = await _do_render(
            scene, kind, args.render, path, state
        )
        if render_payload is not None:
            data["render"] = render_payload
        failures.extend(render_failures)
        observation_events.extend(render_obs)

    data.update(
        {
            "checks_run": checks_run,
            "checks_skipped": skipped,
            "findings": findings,
            "observations": [event for event in observation_events if isinstance(event, str)],
            "warnings": warnings,
            "failures": failures,
        }
    )
    data["outcome"] = _outcome(findings, failures, data.get("comparison"))
    data["hint"] = _hint(data)
    return VerifyCircuitEvaluation(
        data=data,
        inline_image=inline_image,
        capped_rules=frozenset(capped_rules),
        observation_events=tuple(observation_events),
    )


def render_verify_circuit(evaluation: VerifyCircuitEvaluation) -> types.CallToolResult:
    """Apply the existing MCP cap and observation presentation to an evaluation.

    Selects findings by reference before building the envelope, so the uncapped
    evaluation is never copied wholesale; presentation replaces keys rather than
    writing through any value it shares with the evaluation.
    """
    shown: dict[str, int] = {}
    presented: list[dict[str, Any]] = []
    for finding in evaluation.data["findings"]:
        rule = str(finding["rule_id"])
        if rule in evaluation.capped_rules:
            count = shown.get(rule, 0)
            if count >= FINDING_RULE_CAP:
                continue
            shown[rule] = count + 1
        presented.append(finding)
    data = dict(evaluation.data)
    data["findings"] = presented

    observations: list[str] = []
    for event in evaluation.observation_events:
        if isinstance(event, str):
            observations.append(event)
            continue
        observations.extend(
            f"{kind}: showing {FINDING_RULE_CAP} of {count} findings"
            for kind, count in sorted(event.totals.items())
            if count > FINDING_RULE_CAP
        )
    data["observations"] = observations
    if not evaluation.is_error:
        data["outcome"] = _outcome(data["findings"], data["failures"], data.get("comparison"))
        data["hint"] = _hint(data)

    result = format_response(data["hint"], data)
    if evaluation.is_error:
        result.isError = True
    if evaluation.inline_image is not None:
        result.content.insert(0, _image_content(evaluation.inline_image))
    return result


@registry.tool(
    name="verify_circuit",
    description=VERIFY_DESCRIPTION,
    input_model=VerifyCircuitInput,
    # Not read-only: export writes a file on every path (managed scratch by
    # default), and export_to:sidecar overwrites the deck's .net. The annotation
    # states the worst case; the description carries the conditional nuance.
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("consolidated",),
    output_schema=_OUTPUT_SCHEMA,
)
async def handle_verify_circuit(
    args: VerifyCircuitInput, state: SessionState
) -> types.CallToolResult:
    """Check (and optionally render/compare) a circuit file without changing it."""
    return render_verify_circuit(await evaluate_verify_circuit(args, state))
