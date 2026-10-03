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
  LTspice silently drops and the value suffixes the exported netlist spells
  outside ASCII), ``layout`` (geometric placement facts), ``quality``
  (label-island and text-in-body hygiene), and ``compare``.
* netlist — ``syntax`` (directive + element arity, and a non-ASCII character
  where a value's scale suffix goes), ``quality`` (nodes wired to
  a single terminal, directives naming something no element declares, nets with
  no DC path to ground), and ``compare``. No layout, symbol, or export claim is
  made on a text deck: it carries no geometry and no symbol library.

``compare`` runs in one of two modes: ``equivalence`` graph-compares through the
connectivity engine (every include/lib open gated by ``safe_path`` so an in-deck
include that escapes the allowed roots is denied and never read); ``structural_diff``
lexes both decks and reports an added/removed/changed delta over the parsed cards,
leaving out the boilerplate LTspice's netlister adds to every export. Both modes
compare netlists: an ``.asc`` under test is compared as its export, so an ``.asc``
reference is exported the same way, and with no exporter the compare fails rather
than diff a schematic's attributes against a netlist's cards.

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
rules in ``lib/result_observations.py`` route a *run-level solve failure* to
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
import contextlib
import functools
import hashlib
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypeAlias

from mcp import types
from pydantic import BeforeValidator, Field

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.deck_prep import asc_export_lock
from ltspice_mcp.lib.encoding import read_spice_text_with_encoding
from ltspice_mcp.lib.filelock import circuit_file_lock
from ltspice_mcp.lib.lint_rules import deck_generator, value_suffix_evidence
from ltspice_mcp.lib.netlist_diff import Deck, read_deck, structural_delta
from ltspice_mcp.lib.netlist_graph import (
    IncludeResolver,
    NetlistGraph,
    NetlistGraphError,
    compare_graphs,
    parse_netlist_graph,
)
from ltspice_mcp.lib.raster import RenderedImage
from ltspice_mcp.lib.schematic_ops import (
    is_asc,
    same_instance_dropped_segments,
)
from ltspice_mcp.lib.schematic_scene import (
    LayoutIssue,
    NetFlag,
    Scene,
    build_scene,
    layout_issues,
)
from ltspice_mcp.lib.schematic_scene import point_on_segment as point_on_segment
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, lex
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.spice_validator import (
    drop_title_card,
    validate_directive,
    validate_netlist_arity,
    validate_netlist_bias_topology,
    validate_netlist_dangling_nodes,
    validate_netlist_directive_refs,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FINDING_SCHEMA,
    HINT_SCHEMA,
    REPEATABLE_CHANGE_ANNOTATIONS,
    WARNINGS_SCHEMA,
    CompareSpec,
    RenderPolicy,
    ToolInput,
    coerce_render_policy,
    comparison_mismatch,
    failures_schema,
    format_response,
    image_content,
    make_include_resolver,
    outcome_of,
    outcome_schema,
    path_denied_text,
    registry,
    render_scene_artifact,
    resolve_reference,
    safe_path,
    symbol_resolver_for,
)

# The wiring geometry comes from lib/schematic_ops.py rather than a second copy,
# so a refusal the editor enforces and a finding this tool reports cannot drift.


# The added/removed/changed delta both structural comparisons in this codebase
# produce from ``netlist_diff.structural_delta``: verify_circuit's structural_diff
# (which edit_schematic's reference stage runs too) and the sidecar export's
# diff_vs_prior. One payload, one schema, declared here where it is published.
#
# "baseline" is the first deck given (verify's reference, the prior sidecar);
# "compared" is the second (the circuit under test, the fresh export).
STRUCTURAL_DELTA_PROPS: dict[str, Any] = {
    "components_added": {
        "type": "array",
        "items": {"type": "string"},
        "description": "References present in the compared deck but absent from the baseline.",
    },
    "components_removed": {
        "type": "array",
        "items": {"type": "string"},
        "description": "References present in the baseline but absent from the compared deck.",
    },
    "components_changed": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "reference": {"type": "string", "description": "Component reference, e.g. 'R1'."},
                "before": {"type": "string", "description": "Its signature in the baseline."},
                "after": {"type": "string", "description": "Its signature in the compared deck."},
            },
            "required": ["reference", "before", "after"],
        },
        "description": (
            "References in both decks whose model/value or instance parameters differ."
        ),
    },
    "directives_added": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "SPICE directives in the compared deck and not the baseline, each "
            "with its continuation lines joined."
        ),
    },
    "directives_removed": {
        "type": "array",
        "items": {"type": "string"},
        "description": "SPICE directives in the baseline and not the compared deck.",
    },
}


def parse_failure_warnings(pairs: Sequence[tuple[str, str | None]]) -> list[str]:
    """Structured warnings for the decks a structural diff could not parse.

    ``pairs`` is ``(display name, parse error or None)`` per side. An unparsed
    deck is diffed as an EMPTY circuit, so the other side's whole content reads
    as added or removed. That interpretation has to ride in the structured
    channel, not only the text one: structured-aware clients never see the text
    caveat, and the bogus added/removed lists look trustworthy without it.

    Returns the interpretation first, then one message per unparsed deck.
    """
    errors = [err for _name, err in pairs if err]
    if not errors:
        return []
    unparsed = " and ".join(name for name, err in pairs if err)
    return [
        f"{unparsed} could not be parsed; the diff treats it as empty, so its "
        "components/directives appear as added/removed. Fix the file before "
        "trusting this comparison.",
        *errors,
    ]


NETLIST_SUFFIXES = frozenset({".cir", ".net", ".sp"})

CHECK_ORDER = ("syntax", "symbols", "export", "layout", "quality", "compare")

# Applicable checks per file kind. A netlist carries no geometry and no symbol
# library, so it gets only the text-deck checks — claiming a layout/symbol/export
# result for it would be manufacturing a finding out of an absent capability.
# ``quality`` means different things on the two kinds because the two files hold
# different evidence: geometry hygiene on a sheet, connectivity on a deck.
_ASC_CHECKS = frozenset({"symbols", "export", "layout", "quality", "compare"})
_NETLIST_CHECKS = frozenset({"syntax", "quality", "compare"})

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

_CHECK_SKIPPED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"check": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["check", "reason"],
}

#: A verify failure names the stage that failed, not a case (see Envelope).
_STAGE_FAILURE_SCHEMA: dict[str, Any] = {
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
        "source_sha256": {"type": ["string", "null"]},
        "width": {"type": ["integer", "null"]},
        "height": {"type": ["integer", "null"]},
        "downscaled": {"type": "boolean"},
        "image_format": {"type": "string"},
        "scale": {"type": ["number", "null"]},
        "bytes": {"type": "integer"},
        "estimated_tokens": {"type": ["integer", "null"]},
        "returned_inline": {"type": "boolean"},
        "delivery": {"type": "string"},
        "inline_skipped": {
            "type": ["string", "null"],
            "enum": ["svg_requested", "png_unavailable", None],
            "description": (
                "Why an inline or both delivery returned no image: inline is PNG "
                "only, and either format 'svg' was asked for or no PNG could be "
                "made. Null when the image was inlined or inline was not asked for."
            ),
        },
        "note": {
            "type": ["string", "null"],
            "description": (
                "What was delivered instead of what was asked for, and why: a PNG "
                "that fell back to SVG, an image that was not inlined, and where "
                "the file is."
            ),
        },
    },
    "required": ["path", "sha256", "source_sha256", "width", "height", "downscaled"],
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
        "box. A fact, not a difference: it does not change 'equivalent'."
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
        "order. A fact, not a difference: it does not change 'equivalent'."
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
COMPARISON_SCHEMA: dict[str, Any] = {
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
                "The overall verdict. In equivalence mode: structurally isomorphic "
                "and no component, value, parameter, anchor or arity difference. In "
                "structural_diff mode: the delta is empty."
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
        "outcome": outcome_schema("complete", "partial", "failed"),
        "checks_run": {"type": "array", "items": {"type": "string"}},
        "checks_skipped": {"type": "array", "items": _CHECK_SKIPPED_SCHEMA},
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        "comparison": COMPARISON_SCHEMA,
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
        "failures": failures_schema(_STAGE_FAILURE_SCHEMA),
        "hint": HINT_SCHEMA,
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


# ``mode`` and ``delivery`` are here rather than on ``RenderPolicy`` because
# they are decisions only this tool has to make: which checks to skip, and
# which channel the image goes out on. The base stays what any renderer could
# honour, so a second one would not inherit choices it cannot make.
class VerifyRenderPolicy(RenderPolicy):
    """The shared render policy, plus what only this tool can decide."""

    mode: Literal["with_checks", "only"] = Field(
        default="with_checks",
        description=(
            "'with_checks' renders the drawing in addition to running the checks; "
            "'only' renders and skips every check, which is faster."
        ),
    )
    delivery: Literal["artifact", "inline", "both"] = Field(
        default="artifact",
        description=(
            "'artifact' (default) writes the image and returns its path; "
            "'inline' and 'both' also return the PNG as image content. An "
            "inline image costs about 3k tokens at the default scale and stays "
            "in context for every later turn. PNG only."
        ),
    )


class VerifyCompareSpec(CompareSpec):
    """The shared comparison spec, plus the two comparisons only this tool runs."""

    mode: Literal["equivalence", "structural_diff"] = Field(
        default="equivalence",
        description=(
            "'equivalence' graph-compares connectivity (components, values, "
            "parameters, node partitions, arity, 'anchors') and returns no "
            "'comparison' if either deck cannot be read; 'structural_diff' "
            "reports the added/removed/changed delta and still returns one, "
            "with 'equivalent' null."
        ),
    )


#: The checks that are asked for by filling in an argument object of their own,
#: and the object each one's arguments live on. A check is otherwise a name in
#: the ``CHECK_ORDER`` list and has no fields, so the reference index reads this
#: instead of special-casing one name — a new check with its own object is
#: declared here, in the module that owns checks.
CHECK_ARGUMENT_MODELS: dict[str, type[CompareSpec]] = {"compare": VerifyCompareSpec}


RenderArgument: TypeAlias = Annotated[
    VerifyRenderPolicy | None,
    BeforeValidator(
        functools.partial(coerce_render_policy, policy=VerifyRenderPolicy),
        json_schema_input_type=VerifyRenderPolicy | bool | None,
    ),
]
"""``VerifyRenderPolicy | None`` that also takes ``True``/``False``."""


class VerifyCircuitInput(ToolInput):
    path: str = Field(description="Circuit to check: .asc, .cir, .net or .sp.")
    checks: list[Literal["syntax", "symbols", "export", "layout", "quality", "compare"]] | None = (
        Field(
            default=None,
            description=(
                "Narrow the checks. Default is every check applicable to the "
                "file type, plus compare whenever 'compare' is given. Use it "
                "to skip something expensive — 'export' runs LTspice."
            ),
        )
    )
    compare: VerifyCompareSpec | None = Field(
        default=None,
        description=(
            "Compare this circuit against a reference netlist. Omit when there is "
            "no reference: the syntax, symbol, export, layout and quality checks "
            "need no reference to run."
        ),
    )
    render: RenderArgument = Field(
        default=None,
        description=(
            "Draw the .asc alongside (or instead of) the checks. true takes the "
            "defaults below; omitted or false draws nothing."
        ),
    )

    export_to: Literal["managed", "sidecar"] = Field(
        default="managed",
        description=(
            "Where the exported netlist goes: 'managed' writes into the "
            "server's scratch directory and leaves your files alone; 'sidecar' "
            "writes <name>.net next to the schematic, overwriting any existing "
            "one."
        ),
    )


VERIFY_DESCRIPTION = (
    "Check a circuit file, and optionally render it. It does not change the file "
    "it checks; with export_to='sidecar' the export check rewrites the .net next "
    "to an .asc. For a "
    ".cir/.net/.sp: SPICE syntax, directive and element arity, non-ASCII value "
    "suffixes such as µ, plus connectivity "
    "facts — nodes wired to one terminal, V()/I() naming something no element "
    "declares, nets with no DC path to ground. For an .asc: symbol "
    "and pin resolution, the authoritative LTspice netlist export (which silently "
    "drops wires the file appears to contain), geometric layout facts (overlapping "
    "bodies, wires through a body, floating pins, dangling wire ends), and quality "
    "facts (net connected only by label stubs with no drawn wire; text anchored "
    "inside a symbol). Supply 'compare' to graph-compare against a known-good "
    "netlist (equivalence) or take an added/removed/changed delta (structural_diff). "
    "Every fixable finding carries its location and subject."
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


def _scratch_dir(state: SessionState, name: str) -> Path:
    """A server-owned scratch subdirectory (created on demand)."""
    path = state.store.verify_artifact(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_schematic_text(text: str) -> bool:
    """Whether reference text is an ``.asc`` schematic rather than a netlist.

    An ``.asc`` opens with its ``Version`` line and the ``SHEET`` line after it;
    a netlist's first line is a free-text title, so both are required.
    """
    lines = text.lstrip().split("\n", 3)[:3]
    return re.match(r"Version\s+\d", lines[0]) is not None and any(
        line.startswith("SHEET ") for line in lines[1:]
    )


@dataclass(frozen=True)
class ReferenceNetlist:
    """A compare reference in the form both comparison modes read.

    ``source`` is netlist text or a netlist path — for an ``.asc`` reference, its
    LTspice export. It is None when the reference could not be made a netlist,
    and ``error`` (with ``remedy`` when there is one) says why. ``base_dir`` is
    where its relative includes resolve when that is not ``source``'s own
    directory: for an export, the schematic's folder, not the scratch copy's.
    """

    source: str | Path | None
    error: str = ""
    remedy: str | None = None
    observations: tuple[str, ...] = ()
    base_dir: Path | None = None


def reference_as_given(reference: str | Path) -> ReferenceNetlist | None:
    """The reference when it already is a netlist; None for an ``.asc`` path.

    Both tools compare the circuit under test as its LTspice export, so an
    ``.asc`` reference has to reach the comparison the same way: read through the
    schematic editor, its SYMATTRs and TEXT blocks are a different representation
    of the same circuit, and every difference in representation reads as a
    change. None tells the caller to export it with the exporter its own circuit
    under test went through. Schematic *text* has no path to export from, so it
    is refused.
    """
    if isinstance(reference, Path):
        return None if is_asc(reference) else ReferenceNetlist(reference)
    if _is_schematic_text(reference):
        return ReferenceNetlist(
            None,
            "compare.reference is .asc schematic text, which has no netlist to compare "
            "until LTspice exports it; pass the schematic's path instead",
        )
    return ReferenceNetlist(reference)


@contextlib.asynccontextmanager
async def reference_netlist(
    reference: str | Path, state: SessionState
) -> AsyncIterator[ReferenceNetlist]:
    """The reference as a netlist; an ``.asc`` is exported the way a managed export is.

    The export runs on a staged copy (with the project-local files a managed
    export stages), so nothing is written beside the caller's file. The staging
    directory is this call's alone and is removed on exit.
    """
    given = reference_as_given(reference)
    if given is not None:
        yield given
        return
    assert isinstance(reference, Path)  # only an .asc path needs exporting
    staging = state.store.verify_artifact("reference-export") / (
        f"{reference.stem}.{uuid.uuid4().hex[:12]}"
    )
    try:
        yield await _export_reference(reference, staging, state)
    finally:
        await asyncio.to_thread(shutil.rmtree, staging, True)


async def _export_reference(
    reference: Path, staging: Path, state: SessionState
) -> ReferenceNetlist:
    """Stage an ``.asc`` reference in ``staging`` and export it with LTspice."""
    simulator_cls = state.available_simulators.get("ltspice")
    if simulator_cls is None:
        return ReferenceNetlist(
            None,
            f"the reference {reference.name} is an .asc schematic, which is compared "
            "as its LTspice netlist export, and LTspice is not detected",
            remedy="pass the reference's exported netlist (.net) or its netlist text",
        )
    staged_asc, note = await _stage_for_export(reference, staging)
    observations = (note,) if note else ()
    try:
        async with asc_export_lock(staged_asc):
            net_path = await asyncio.to_thread(
                _export_if_written, simulator_cls, staged_asc, state.config.default_timeout
            )
    except Exception as exc:  # the simulator is a subprocess; any failure is data
        error = f"LTspice netlist export of the reference {reference.name} failed: {exc}"
        return ReferenceNetlist(None, error, observations=observations)
    if net_path is None:
        error = f"LTspice exported the reference {reference.name} but produced no .net file"
        return ReferenceNetlist(None, error, observations=observations)
    return ReferenceNetlist(net_path, observations=observations, base_dir=reference.parent)


# ---------------------------------------------------------------------------
# syntax
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LexedDeck:
    """One lex of a netlist, shared by the syntax and quality checks.

    ``cards`` already has the title card dropped: line 1 is free text by SPICE
    convention and the lexer has no title concept, so a title beginning with an
    element letter parses as an element whose words the instance-level rules
    would otherwise count as circuit nodes.

    ``notes`` are the lexer's own remarks about what it had to guess — an
    unclosed ``.SUBCKT``, an ``.ENDS`` that matches nothing, a continuation with
    no card to continue. They are facts about the read, not rule violations, and
    the lexer assigns them no severity, so they are relayed as observations
    rather than promoted to findings — but relayed they must be, since the deck
    they describe parsed into cards that mean something other than what the file
    says. ``error`` is set when the deck did not lex at all.
    """

    cards: list[SpiceCard]
    notes: list[str]
    error: str | None = None


def _lex_deck(text: str) -> _LexedDeck:
    """Lex a netlist once for every check that reads its cards."""
    try:
        result = lex(text)
    except SpiceLexError as exc:
        return _LexedDeck(cards=[], notes=[], error=str(exc))
    return _LexedDeck(
        cards=drop_title_card(result.cards),
        notes=[f"netlist lexer: {note}" for note in result.warnings],
    )


def _rule_finding(
    issue: dict[str, object], path: Path, *, rule_id: str, severity: str
) -> dict[str, Any]:
    """A netlist-validator issue in the shared finding shape.

    Every netlist rule reports ``{line, directive, message, suggestion}``, so the
    conversion is one function rather than one per rule.
    """
    detail = str(issue.get("message", ""))
    suggestion = issue.get("suggestion")
    if suggestion:
        detail = f"{detail} {suggestion}"
    card = str(issue.get("directive", ""))
    at: dict[str, Any] = {"file": str(path)}
    line_no = issue.get("line")
    if isinstance(line_no, int):
        at["line"] = line_no
    return _finding(
        rule_id=rule_id,
        severity=severity,
        at=at,
        subject=card or str(path.name),
        evidence={"detail": detail, "card": card},
    )


def _value_suffix_findings(
    cards: list[SpiceCard], path: Path, text: str, encoding: str
) -> list[dict[str, Any]]:
    """Numbers whose scale-suffix position holds a non-ASCII character.

    One scan and one evidence builder with the ``run_experiments`` linter's
    ``value-suffix-*`` rules, so both surfaces say the same thing. A micro sign
    is a warning: it is micro to a reader that decodes the file in the encoding
    it was written in, which LTspice XVII does not do for a UTF-8 file. A
    suffix that shows the file was decoded in an encoding it was not written
    in (``Âµ``) is an error: the deck runs at the bare number, and a micro sign
    lost that way is a factor of 1e6. Any other symbol (``10Ω``) is a warning:
    it is read as the bare number, which is usually what it means.
    ``encoding`` is the codec the file decoded as, which decides which reader
    misreads it. ``cards`` has its title card dropped already.
    """
    sites = value_suffix_sites(cards)
    generated_by = deck_generator(text) if sites else None
    findings: list[dict[str, Any]] = []
    for site in sites:
        evidence = value_suffix_evidence(site, generated_by=generated_by)
        evidence["card"] = site.card.body
        evidence["encoding"] = encoding
        if site.micro:
            rule_id, severity = "value_suffix_micro_sign", "warning"
        elif site.mojibake:
            rule_id, severity = "value_suffix_mojibake", "error"
        else:
            rule_id, severity = "value_suffix_nonascii", "warning"
        findings.append(
            _finding(
                rule_id=rule_id,
                severity=severity,
                at={"file": str(path), "line": site.line},
                subject=site.token,
                evidence=evidence,
            )
        )
    return findings


def _syntax_findings(
    text: str, path: Path, deck: _LexedDeck, encoding: str, simulator: str
) -> list[dict[str, Any]]:
    """Directive, lex, element-arity and value-suffix findings in a netlist.

    Everything here changes what the simulator reads. A finding is an error
    unless its rule says the deck still runs as meant: an element-arity issue
    carries the severity its validator check declares, and a suffix finding
    is a warning except for a mis-decoded file (see
    ``_value_suffix_findings``). ``simulator`` is the one the session runs
    decks on, ``"LTspice"`` or ``"ngspice"``: some directive and element forms
    are a fault for one and valid for the other. The facts that are
    legal-but-notable live in the quality check.
    """
    findings: list[dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("."):
            continue
        err = validate_directive(line, simulator)
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

    if deck.error is not None:
        findings.append(
            _finding(
                rule_id="lex_error",
                severity="error",
                at={"file": str(path)},
                subject=str(path.name),
                evidence={"detail": deck.error},
            )
        )
    findings.extend(
        _rule_finding(issue, path, rule_id="element_arity", severity=str(issue["severity"]))
        for issue in validate_netlist_arity(deck.cards, simulator=simulator)
    )
    findings.extend(_value_suffix_findings(deck.cards, path, text, encoding))
    return findings


# The connectivity rules the netlist quality check runs, with the severity this
# tool attaches to each. Severity names an input condition, never a verdict:
#
# ``dangling_node`` is legal SPICE — bias fragments and deliberately
# unterminated test decks leave nodes open on purpose — so it is a fact the
# model weighs, matching the ``floating_pin`` observation its .asc sibling
# reports for the same geometry.
#
# The other two are real discrepancies between what the deck says and what it
# can compute: a directive naming something no element declares resolves to
# nothing, and a net with no DC path to ground has no defined operating point.
_NETLIST_QUALITY_RULES: tuple[tuple[str, str, Any], ...] = (
    ("dangling_node", "observation", validate_netlist_dangling_nodes),
    ("undefined_reference", "warning", validate_netlist_directive_refs),
    ("floating_net", "warning", validate_netlist_bias_topology),
)


def _netlist_quality_findings(path: Path, deck: _LexedDeck) -> list[dict[str, Any]]:
    """Connectivity findings over a lexed netlist: nodes wired to one terminal,
    directives naming something that exists nowhere, and nets with no DC path
    to ground."""
    findings: list[dict[str, Any]] = []
    for rule_id, severity, validate in _NETLIST_QUALITY_RULES:
        findings.extend(
            _rule_finding(issue, path, rule_id=rule_id, severity=severity)
            for issue in validate(deck.cards)
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
    for drop in same_instance_dropped_segments(owners, segments):
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


async def _stage_for_export(asc_path: Path, dest: Path) -> tuple[Path, str | None]:
    """Stage ``asc_path`` in ``dest`` under its circuit lock, for a managed export.

    Returns the staged ``.asc`` and, when the file cap was hit, the observation
    saying so.
    """
    async with circuit_file_lock(asc_path):
        staged_asc, _staged, truncated = await asyncio.to_thread(
            _stage_export_inputs, asc_path, dest
        )
    note = (
        f"only the first {STAGE_FILE_CAP} project-local symbol/library files were staged "
        f"for the export of {asc_path.name}, so a symbol beyond that may not resolve"
    )
    return staged_asc, note if truncated else None


def _create_netlist(simulator_cls: Any, asc_path: Path, timeout: float) -> Path:
    """Drive LTspice to export ``asc_path`` to a sibling ``.net``."""
    return Path(simulator_cls.create_netlist(str(asc_path), timeout=timeout))


def _export_if_written(simulator_cls: Any, asc_path: Path, timeout: float) -> Path | None:
    """``_create_netlist``, or None when LTspice exited without writing the ``.net``."""
    net_path = _create_netlist(simulator_cls, asc_path, timeout)
    return net_path if net_path.exists() else None


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


def _exported_value_suffix_findings(net_path: Path) -> list[dict[str, Any]]:
    """Value-suffix findings over an exported netlist.

    LTspice 24 and later write the export as UTF-8, a micro sign as C2 B5, and
    the exported ``.net`` is what gets handed to another LTspice.
    """
    try:
        text, encoding = read_spice_text_with_encoding(net_path)
    except OSError:
        return []
    return _value_suffix_findings(_lex_deck(text).cards, net_path, text, encoding)


class _MeasuredNetlist(NamedTuple):
    """What ``_measure_netlist`` read off a just-exported netlist."""

    components: int | None
    nets: int | None
    sha256: str | None
    findings: list[dict[str, Any]]


def _measure_netlist(net_path: Path) -> _MeasuredNetlist | None:
    """Component count, net count, digest and value-suffix findings of a
    just-exported netlist.

    Returns ``None`` when the exporter wrote no file at all.

    Call this while the export lock that produced ``net_path`` is still held.
    ``create_netlist`` writes a shared ``<stem>.net``, so the moment the lock is
    released a second export of the same schematic reopens that path in write
    mode and truncates it. Counts and a digest read after that describe a
    netlist no export ever produced — and they would be reported as a success,
    with the SHA-256 of an empty file standing in as this export's provenance.
    """
    if not net_path.exists():
        return None
    components, nets = _netlist_counts(net_path)
    return _MeasuredNetlist(
        components, nets, _file_digest(net_path), _exported_value_suffix_findings(net_path)
    )


class _ExportOutcome(NamedTuple):
    """What the export stage hands back to the evaluator."""

    payload: dict[str, Any]
    failure: dict[str, Any] | None
    observations: list[str]
    warnings: list[str]
    # About the exported netlist's text: value suffixes spelled outside ASCII.
    findings: Sequence[dict[str, Any]] = ()


async def _run_export(
    asc_path: Path, state: SessionState, export_to: str, simulator_cls: Any
) -> _ExportOutcome:
    """Export a schematic to a netlist, managed (scratch copy) or sidecar (in place).

    The
    ``sidecar`` mode runs under the export lock (it overwrites ``<name>.net``) and
    records a structural ``diff_vs_prior``; ``managed`` stages the schematic with its
    project-local assets and exports there, touching none of the caller's files.

    Both modes measure the netlist they just wrote *inside* their own export
    lock (see ``_measure_netlist``): the file is a shared path a peer export
    reopens for writing, so a digest taken after the lock is released can
    describe the peer's truncation instead of this export's output.
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

    measured: _MeasuredNetlist | None = None
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
                measured = await asyncio.to_thread(_measure_netlist, net_path)
        else:
            scratch = (
                _scratch_dir(state, "export") / f"{asc_path.stem}.{_file_digest(asc_path, 8)}"
            )
            staged_asc, note = await _stage_for_export(asc_path, scratch)
            if note:
                observations.append(note)
            async with asc_export_lock(staged_asc):
                net_path = await asyncio.to_thread(
                    _create_netlist, simulator_cls, staged_asc, timeout
                )
                measured = await asyncio.to_thread(_measure_netlist, net_path)
    except Exception as exc:  # the simulator is a subprocess; any failure is data
        return _ExportOutcome(
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

    if measured is None:
        return _ExportOutcome(
            payload,
            _failure(
                "export",
                "LTspice exited without an error but produced no .net file",
                where=str(net_path),
            ),
            observations,
            warnings,
        )

    payload.update(
        {
            "ok": True,
            "netlist": str(net_path),
            "sha256": measured.sha256,
            "components": measured.components,
            "nets": measured.nets,
        }
    )
    return _ExportOutcome(payload, None, observations, warnings, measured.findings)


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


def _denied_include_findings(graph: NetlistGraph, source: Path, hint: str) -> list[dict[str, Any]]:
    """path_denied findings for includes the resolver refused (never read).

    ``hint`` is the sandbox guidance: the refusal is the sandbox's, so its
    remedy is the same config line as any other refused path.
    """
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
                    "hint": hint,
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


def compare_equivalence(
    reference: str | Path,
    candidate: str | Path,
    ref_source: Path,
    cand_source: Path,
    anchors: list[str] | None,
    rtol: float,
    resolver: IncludeResolver,
    *,
    denied_hint: str,
    ref_base_dir: Path | None,
) -> CompareResult:
    """Graph-compare candidate against reference through the safe_path resolver.

    ``candidate`` may be the already-read netlist text or a path. ``cand_source``
    is the file the caller named: findings and failures point at it, and the
    candidate's relative includes resolve in its directory, not in the store
    scratch an export was made in. ``ref_base_dir`` is the reference's
    (``ReferenceNetlist.base_dir``). ``denied_hint`` is the sandbox guidance
    each refused include's finding carries.
    """
    findings: list[dict[str, Any]] = []
    ref_graph, failure = _parse_graph_or_fail(
        reference, "reference netlist", ref_source, resolver, base_dir=ref_base_dir
    )
    if failure is not None:
        return None, findings, failure, []
    cand_graph, failure = _parse_graph_or_fail(
        candidate, "netlist under test", cand_source, resolver, base_dir=cand_source.parent
    )
    if failure is not None:
        return None, findings, failure, []
    assert ref_graph is not None and cand_graph is not None  # failure is None ⇒ both parsed
    findings.extend(_denied_include_findings(ref_graph, ref_source, denied_hint))
    findings.extend(_denied_include_findings(cand_graph, cand_source, denied_hint))
    comparison = compare_graphs(ref_graph, cand_graph, anchors=anchors, rtol=rtol)
    payload: dict[str, Any] = {"mode": "equivalence", **comparison.as_dict()}
    return payload, findings, None, []


def _read_side(source: str | Path, label: str) -> tuple[str, Deck, str | None]:
    """``(name, deck, parse_error)`` for one side of a structural diff.

    ``label`` names a side given as text. An unreadable side is an empty deck and
    a short message, so the diff can flag it rather than treat it as an empty
    circuit (which would report every component of the other side as a removal).
    """
    name = source.name if isinstance(source, Path) else label
    try:
        return name, read_deck(source), None
    except Exception as e:  # unreadable for any reason: reported, never raised
        return name, Deck(), f"{name} could not be parsed ({e})"


def _structural_diff(
    reference: str | Path, candidate: str | Path
) -> tuple[dict[str, Any], list[str], bool]:
    """Added/removed/changed component and directive delta between two netlists.

    Returns ``(delta, warnings, both_parsed)``. A deck that could not be parsed is
    diffed as an empty circuit, which makes every component of the other side look
    added or removed — so the caveat rides out as a warning rather than inside the
    delta, where the caller browsing the difference lists would never look for it.

    ``both_parsed`` is reported separately rather than inferred from an empty
    ``warnings`` list, so a future warning of some other kind cannot silently be
    read as a parse failure.
    """
    name_a, ref, err_a = _read_side(reference, "the reference netlist text")
    name_b, cand, err_b = _read_side(candidate, "the netlist under test")
    warnings = parse_failure_warnings([(name_a, err_a), (name_b, err_b)])
    return structural_delta(ref, cand), warnings, err_a is None and err_b is None


def compare_structural(reference: str | Path, candidate: str | Path) -> CompareResult:
    """structural_diff mode over two netlists, each given as a path or as text.

    A schematic path is refused rather than read: its attributes and TEXT blocks
    are not the cards its export holds, so diffing one against a netlist reports
    the difference in representation as a difference in circuit. The tools export
    an ``.asc`` before they get here.
    """
    for side in (reference, candidate):
        if isinstance(side, Path) and is_asc(side):
            error = (
                f"{side.name} is an .asc schematic; a structural diff compares "
                "netlists, so compare its LTspice export"
            )
            return None, [], _failure("compare", error, where=str(side)), []
    try:
        diff, warnings, both_parsed = _structural_diff(reference, candidate)
    except (OSError, ValueError) as exc:
        where = str(candidate) if isinstance(candidate, Path) else None
        return None, [], _failure("compare", f"structural diff failed: {exc}", where=where), []
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


def compare_netlists(
    spec: VerifyCompareSpec,
    reference: str | Path,
    candidate: str | Path,
    ref_source: Path,
    cand_source: Path,
    state: SessionState,
    *,
    ref_base_dir: Path | None,
) -> CompareResult:
    """Compare two netlists the way ``spec`` asks (blocking CPU/IO).

    ``reference`` and ``candidate`` are netlist text or paths; ``ref_source`` and
    ``cand_source`` are where a finding about each side points (the caller's own
    file, even when what was compared is its export). ``cand_source``'s
    directory and ``ref_base_dir`` are where each side's relative includes
    resolve; a structural diff opens no includes.
    """
    if spec.mode == "structural_diff":
        return compare_structural(reference, candidate)
    return compare_equivalence(
        reference,
        candidate,
        ref_source,
        cand_source,
        spec.anchors,
        spec.rtol,
        make_include_resolver(state),
        denied_hint=state.sandbox_guidance(),
        ref_base_dir=ref_base_dir,
    )


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


#: Why an inline delivery returned no image. Inline is PNG only; the reasons
#: are in the verify_circuit section of docs/design/mcp_surface.md.
InlineSkipped: TypeAlias = Literal["svg_requested", "png_unavailable"]


#: What the render check produces: ``(payload, inline image, failures,
#: observations)``.
_RenderResult = tuple[
    dict[str, Any] | None, "RenderedImage | None", list[dict[str, Any]], list[str]
]


async def _do_render(
    scene: Scene | None,
    kind: str,
    policy: VerifyRenderPolicy,
    path: Path,
    state: SessionState,
) -> _RenderResult:
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
    unavailable = image.png_unavailable
    if unavailable is not None:
        failures.append(
            _failure(
                "render",
                f"PNG was requested and SVG returned instead: {unavailable.reason}",
                remedy=unavailable.remedy,
            )
        )
    inline_asked = policy.delivery in ("inline", "both")
    want_inline = inline_asked and image.is_raster
    inline_skipped: InlineSkipped | None = None
    if inline_asked and not image.is_raster:
        inline_skipped = "svg_requested" if policy.format == "svg" else "png_unavailable"
    payload = _render_payload(
        image,
        out_path,
        downscaled=downscaled,
        delivery=policy.delivery,
        returned_inline=want_inline,
        inline_skipped=inline_skipped,
        source_sha256=scene.source_sha256,
    )
    return payload, (image if want_inline else None), failures, []


def _render_note(
    image: RenderedImage, path: Path, inline_skipped: InlineSkipped | None
) -> str | None:
    """The render's own explanation: what was delivered in place of the request."""
    if inline_skipped is None:
        return image.note
    if inline_skipped == "svg_requested":
        return (
            "Render not returned inline: inline delivery is PNG only and format "
            f"'svg' was requested; the SVG is at {path}. Request format 'png' "
            "to receive the drawing inline."
        )
    return f"{image.note}. Not returned inline: inline delivery is PNG only; the SVG is at {path}."


def _render_payload(
    image: RenderedImage,
    path: Path,
    *,
    downscaled: bool,
    delivery: str,
    returned_inline: bool,
    inline_skipped: InlineSkipped | None,
    source_sha256: str | None,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(image.data).hexdigest(),
        # The digest of the SHEET this drew, not of the image and not of any
        # exported netlist — read off the scene, so it is the hash of the bytes
        # that were actually parsed rather than of whatever a later re-read
        # would find. Comparing it with the sha256 edit_schematic returned is
        # how a caller knows the picture is of the revision it wrote, and a
        # peer committing mid-call shows up as a mismatch instead of hiding
        # behind a fresh hash of the new file.
        "source_sha256": source_sha256,
        "width": image.width,
        "height": image.height,
        "downscaled": downscaled,
        "image_format": image.image_format,
        "scale": image.scale,
        "bytes": len(image.data),
        "estimated_tokens": image.estimated_tokens,
        "returned_inline": returned_inline,
        "delivery": delivery,
        "inline_skipped": inline_skipped,
        "note": _render_note(image, path, inline_skipped),
    }


# ---------------------------------------------------------------------------
# outcome + hint
# ---------------------------------------------------------------------------


def _comparison_unverified(comparison: dict[str, Any] | None) -> bool:
    """The comparison ran but could not reach a verdict (a side was unparseable)."""
    return comparison is not None and comparison.get("equivalent") is None


def _outcome(
    findings: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> str:
    """This tool's shortfalls, handed to the shared outcome rule.

    A check that failed to run, a finding at error or warning severity, and a
    comparison that came back non-equivalent are each a reason the caller did
    not get the clean answer they asked for. None of them fails the whole call
    — the checks that did run still reported — so this surface never returns
    ``failed`` from here.
    """
    return outcome_of(
        failures,
        partial=any(f["severity"] in ("error", "warning") for f in findings)
        or comparison_mismatch(comparison),
    )


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
    elif comparison_mismatch(comparison):
        parts.append("reference comparison did not match — see comparison")
    observations = [f for f in data["findings"] if f["severity"] == "observation"]
    if observations and not parts:
        parts.append(
            f"{len(observations)} layout/quality observation(s): "
            + ", ".join(sorted({f["rule_id"] for f in observations}))
            + " (facts, not a verdict)"
        )
    # A caller that asked for the picture inline and chose SVG has nothing to
    # fix, but is still not holding what it asked for, so the headline says so.
    render = data.get("render") or {}
    delivery_note = render.get("note") if render.get("inline_skipped") == "svg_requested" else None
    if parts:
        headline = "; ".join(parts) + "."
    else:
        skipped = data.get("checks_skipped") or []
        headline = "No problems found in the checks that ran."
        if skipped:
            headline += " Not run: " + ", ".join(f"{s['check']} ({s['reason']})" for s in skipped)
    return f"{headline.rstrip('.')}. {delivery_note}" if delivery_note else headline


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
    """The payload every path starts from.

    Its outcome is the verdict for a call that refuses before any check runs —
    something is wrong and nothing came back with it — and ``_outcome`` decides
    it again the moment the checks report.
    """
    return {
        "path": path,
        "kind": "unknown",
        "outcome": outcome_of(True, delivered=False),
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
    compare = args.compare
    render = args.render

    try:
        path = safe_path(args.path, state)
        reference = resolve_reference(compare.reference, state) if compare else None
    except PathSecurityError as exc:
        data["findings"] = [
            _finding(
                rule_id="path_denied",
                severity="error",
                at={"file": args.path},
                subject=args.path,
                evidence={"detail": str(exc), "hint": state.sandbox_guidance()},
            )
        ]
        return _error_evaluation(data, path_denied_text(exc, state))

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
    render_only = render is not None and render.mode == "only"

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

    # --- netlist text (syntax, quality) -------------------------------------
    # The text and the codec it decoded as, which the syntax check reports.
    decoded: tuple[str, str] | None = None
    if kind == "netlist" and (
        wanted.get("syntax") or wanted.get("quality") or wanted.get("compare")
    ):
        try:
            decoded = await asyncio.to_thread(read_spice_text_with_encoding, path)
        except OSError as exc:
            failures.append(_failure("read", str(exc), where=str(path)))
    text = decoded[0] if decoded is not None else None

    # One lex serves both text-deck checks, and its notes are relayed once
    # whichever of the two asked for it.
    deck: _LexedDeck | None = None
    if text is not None and (wanted.get("syntax") or wanted.get("quality")):
        deck = await asyncio.to_thread(_lex_deck, text)
        observation_events.extend(deck.notes)

    if wanted.get("syntax") and decoded is not None and deck is not None:
        text, encoding = decoded
        # The simulator a run_experiments call with no override would use.
        simulator = "ngspice" if state.raw_dialect == "ngspice" else "LTspice"
        findings.extend(
            await asyncio.to_thread(_syntax_findings, text, path, deck, encoding, simulator)
        )
        checks_run.append("syntax")

    if wanted.get("quality") and deck is not None and kind == "netlist":
        findings.extend(await asyncio.to_thread(_netlist_quality_findings, path, deck))
        checks_run.append("quality")

    # --- scene-derived checks (symbols, layout, quality, dropped wires) -----
    scene: Scene | None = None
    scene_issues: list[LayoutIssue] = []
    needs_scene = kind == "asc" and (
        wanted.get("symbols")
        or wanted.get("layout")
        or wanted.get("quality")
        or wanted.get("export")
        or render is not None
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

    # --- render (started here, collected below) -----------------------------
    # It reads the scene that is already parsed, so nothing it needs comes from
    # the export — and the export shells out to LTspice for seconds. Only
    # `compare` consumes the export's output, so the picture is drawn in
    # parallel and costs no wall time of its own.
    render_task: asyncio.Task[_RenderResult] | None = None
    if render is not None:
        render_task = asyncio.create_task(_do_render(scene, kind, render, path, state))

    try:
        # --- export ---------------------------------------------------------
        candidate: Path | None = path if kind == "netlist" else None
        if wanted.get("export"):
            simulator_cls = state.available_simulators.get("ltspice")
            if simulator_cls is None:
                skip("export", "LTspice not detected")
            else:
                export = await _run_export(path, state, args.export_to, simulator_cls)
                observation_events.extend(export.observations)
                warnings.extend(export.warnings)
                findings.extend(export.findings)
                if scene is not None:
                    dropped = _dropped_wire_findings(scene, path)
                    findings.extend(dropped)
                    if dropped:
                        capped_rules.add("dropped_wire")
                data["export"] = export.payload
                checks_run.append("export")
                if export.failure is not None:
                    failures.append(export.failure)
                elif export.payload.get("netlist"):
                    candidate = Path(export.payload["netlist"])

        # --- compare ------------------------------------------------------------
        if wanted.get("compare") and reference is not None:
            if candidate is None:
                skip("compare", "the exported netlist is required and the export did not run")
            else:
                assert compare is not None  # guarded by `reference is not None`
                ref_source = reference if isinstance(reference, Path) else path
                # Reuse the netlist text already read for syntax, so the candidate is
                # not read+lexed a second time; the export path has no such text.
                cand_input: str | Path = (
                    text if kind == "netlist" and text is not None else candidate
                )
                compared: CompareResult
                async with reference_netlist(reference, state) as ref_netlist:
                    observation_events.extend(ref_netlist.observations)
                    if ref_netlist.source is None:
                        failure = _failure(
                            "compare",
                            ref_netlist.error,
                            where=str(ref_source),
                            remedy=ref_netlist.remedy,
                        )
                        compared = None, [], failure, []
                    else:
                        compared = await asyncio.to_thread(
                            compare_netlists,
                            compare,
                            ref_netlist.source,
                            cand_input,
                            ref_source,
                            path,
                            state,
                            ref_base_dir=ref_netlist.base_dir,
                        )
                comparison, cmp_findings, cmp_failure, cmp_warnings = compared
                findings.extend(cmp_findings)
                warnings.extend(cmp_warnings)
                if cmp_failure is not None:
                    failures.append(cmp_failure)
                else:
                    data["comparison"] = comparison
                    checks_run.append("compare")

    except BaseException:
        # The picture is worthless once the call is failing, and an abandoned
        # task would report its own failure to nobody.
        if render_task is not None:
            render_task.cancel()
        raise

    # --- render -------------------------------------------------------------
    inline_image: RenderedImage | None = None
    if render_task is not None:
        render_payload, inline_image, render_failures, render_obs = await render_task
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
        result.is_error = True
    if evaluation.inline_image is not None:
        result.content.insert(0, image_content(evaluation.inline_image))
    return result


@registry.tool(
    name="verify_circuit",
    title="Check Circuit",
    description=VERIFY_DESCRIPTION,
    input_model=VerifyCircuitInput,
    # Not read-only: export writes a file on every path (managed scratch by
    # default), and export_to:sidecar overwrites the deck's .net. The annotation
    # states the worst case; the description carries the conditional nuance.
    annotations=REPEATABLE_CHANGE_ANNOTATIONS,
    output_schema=_OUTPUT_SCHEMA,
)
async def handle_verify_circuit(
    args: VerifyCircuitInput, state: SessionState
) -> types.CallToolResult:
    """Check (and optionally render/compare) a circuit file without changing it."""
    return render_verify_circuit(await evaluate_verify_circuit(args, state))
