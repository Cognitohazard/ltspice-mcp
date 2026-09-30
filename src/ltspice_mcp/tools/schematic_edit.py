"""edit_schematic — transactional, revision-guarded ``.asc`` mutation (AUTHOR).

One input language: a typed op batch applied to a sheet that is either blank
(the whole-plan compile) or an existing file (deltas preserving untouched
content). The whole transaction — revision check, editor load, geometry
resolution, mutation, staged write, and the atomic rename onto the target — runs
inside the shared per-file edit guard, so a parallel session that committed
first turns a stale ``expected_sha256`` into a ``revision_conflict`` with nothing
written.

The op models and their in-place applier are reused verbatim from
``lib/schematic_ops.py``. Post-commit, an optional compare stage exports the
committed sheet on a COPY and compares it to a reference — a netlist, or an
``.asc`` exported the same way — the way verify_circuit does; a mismatch or an
export failure there is reported but never un-commits the sheet.
"""

# The op models, the in-place applier and the net-partition helpers are the
# shared engine in lib/schematic_ops.py, imported rather than duplicated.
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, cast, get_args

from mcp import types
from pydantic import Field
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import O_BINARY, atomic_write_bytes, fsync_dir, fsync_fd, replace_file
from ltspice_mcp.lib.cursor_codec import canonical_json
from ltspice_mcp.lib.deck_prep import resolve_runnable_netlist
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.pin_legend import (
    PageCursorError,
    build_pin_legend,
    decode_page_cursor,
    encode_page_cursor,
    find_label_only_pins,
    paginate_view,
)
from ltspice_mcp.lib.schematic_ops import (
    COORDINATE_DESCRIPTION,
    OpAddComponent,
    OpAddDirective,
    OpAddNetLabel,
    OpMoveComponent,
    OpRemoveComponent,
    OpRemoveDirective,
    OpRemoveNetLabel,
    OpRemoveWire,
    OpSetComponentAttribute,
    OpSetComponentValue,
    OpWirePins,
    blank_sheet,
    build_on_wire_predicate,
    collapse_result_warnings,
    collect_component_geometry,
    edit_guard,
    get_asc_editor,
    make_editor,
    post_op_warnings,
    require_asc,
    run_op_batch,
    trace_nets,
    wiring_profile,
)
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    OUTCOME_SCHEMA,
    VALIDATION_WARNINGS_SCHEMA,
    StrictModel,
    ToolInput,
    comparison_mismatch,
    format_response,
    outcome_of,
    page_schema,
    registry,
    resolve_reference,
    safe_path,
)
from ltspice_mcp.tools.verify import (
    COMPARISON_SCHEMA,
    ReferenceNetlist,
    VerifyCompareSpec,
    compare_netlists,
    reference_as_given,
)

# The blank-sheet template — identical to what ``create_schematic`` writes.
_BLANK_TEMPLATE = blank_sheet()

_DEFAULT_VIEW_LIMIT = 100


# The op union for this surface.
#
# Discriminated on ``op``. Without the discriminator pydantic tries every branch
# and reports each one's complaint, so a single mistyped op produced 30-odd
# errors that the compact renderer cut off at "… and 23 more" — the caller
# learned neither which kinds exist nor what their own payload was missing. With
# it, an unknown op is one error naming every accepted kind, and a known op with
# a bad field reports against that kind alone.
ConsolidatedOp = Annotated[
    OpAddComponent
    | OpSetComponentValue
    | OpSetComponentAttribute
    | OpRemoveComponent
    | OpMoveComponent
    | OpAddNetLabel
    | OpRemoveNetLabel
    | OpRemoveWire
    | OpWirePins
    | OpAddDirective
    | OpRemoveDirective,
    Field(discriminator="op"),
]


# The views return_views can name, each resumable through its own field of
# EditViewCursors.
ViewName = Literal["touched", "pin_legend", "preexisting"]
_VIEWS: tuple[str, ...] = get_args(ViewName)


class EditViewCursors(StrictModel):
    """Resumption cursors for the paginated views, each a page's ``next_cursor``."""

    label_only_pins: str | None = Field(
        default=None, description="next_cursor from a previous wiring.label_only_pins page."
    )
    pin_legend: str | None = Field(
        default=None, description="next_cursor from a previous views.pin_legend page."
    )
    touched: str | None = Field(
        default=None, description="next_cursor from a previous views.touched page."
    )
    preexisting: str | None = Field(
        default=None,
        description="preexisting.cursor, or a views.preexisting page's next_cursor.",
    )


class EditSchematicInput(ToolInput):
    target: str = Field(description="Path to the .asc schematic (created if absent).")
    base: Literal["existing", "blank"] = Field(
        default="existing",
        description=(
            "'existing' (default) applies the ops as deltas onto the current "
            "file; 'blank' treats the sheet as empty first (a whole-circuit "
            "one-call build)."
        ),
    )
    expected_sha256: str | None = Field(
        default=None,
        description=(
            "Required when the target exists: the SHA-256 of the file you "
            "edited against, reported as 'sha256' by inspect and by every "
            "commit. A mismatch returns revision_conflict, writing nothing."
        ),
    )
    ops: list[ConsolidatedOp] = Field(
        description=(
            "Typed edit ops, each tagged by its 'op' field, applied in order "
            "and committed atomically: the first failure aborts the batch and "
            "nothing is written. " + COORDINATE_DESCRIPTION
        )
    )
    compare: VerifyCompareSpec | None = Field(
        default=None,
        description=(
            "Verify the committed sheet against a reference by exporting a "
            "copy and comparing it the way verify_circuit does "
            "(same modes, anchors and tolerance). It runs after the commit, so "
            "a mismatch is reported but not undone."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Resolve, validate, and compute geometry without writing. Every op "
            "is attempted, so all problems surface at once."
        ),
    )
    return_views: list[ViewName] = Field(
        default_factory=lambda: ["touched"],
        description=(
            "Which pin/net table to return: 'touched' (default) covers the "
            "components this batch named, 'pin_legend' the whole sheet. "
            "'preexisting' lists the findings and label-only pins counted "
            "under preexisting. To draw the sheet, call verify_circuit."
        ),
    )
    view_cursors: EditViewCursors | None = Field(
        default=None,
        description=(
            "Resume a paginated view by echoing back that page's next_cursor "
            "unmodified; it returns that view even if return_views omits it. "
            "Checked before any work runs, so a bad one cannot surface after "
            "the sheet is committed."
        ),
    )
    view_limit: int = Field(
        default=_DEFAULT_VIEW_LIMIT,
        description="Page size for every views page and for wiring.label_only_pins.",
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_PAGE_SCHEMA: dict[str, Any] = page_schema(
    primary_truncated={
        "type": "boolean",
        "description": (
            "Whether this collection has more rows. Equal to 'truncated' here; the "
            "two differ only on a page that carries a second collection."
        ),
    },
)

# One wiring.label_only_pins row, tagged with its kind where it shares a page
# with the validation pass's findings.
_LABEL_ONLY_ROW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"const": "label_only_pin"},
        "ref": {"type": "string"},
        "pin": {"type": "string"},
        "net": {"type": ["string", "null"]},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
    },
    "required": ["kind", "ref", "pin", "net", "x", "y"],
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": OUTCOME_SCHEMA,
        # Edit-specific: the write state of the target sheet, kept at the top
        # level (and mirrored into error.commit_state on failure envelopes).
        "commit_state": {"type": "string", "enum": ["committed", "not_committed"]},
        "target": {"type": "string"},
        "sha256": {"type": ["string", "null"]},
        "build_id": {"type": "string"},
        "base": {"type": "string"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "stage": {"type": "string"},
                "retryable": {"type": "boolean"},
                "commit_state": {
                    "type": "string",
                    "enum": ["not_started", "committed", "unknown"],
                },
            },
            "required": ["code", "message", "stage", "retryable", "commit_state"],
        },
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "error": {"type": ["string", "null"]},
                },
                "required": ["stage", "ok"],
            },
        },
        "netlist": {
            "type": ["string", "null"],
            "description": (
                "The committed sheet's exported netlist text. Present only when a "
                "reference comparison did not confirm equivalence: a mismatch, a "
                "compare error, or no verdict."
            ),
        },
        "verification": {
            "type": "object",
            "description": (
                "Reference-compare result: equivalent is the overall verdict, and "
                "the difference lists explain a false one. Present only when a reference "
                "was supplied. 'export_error' means the sheet could not be exported."
            ),
            "properties": {
                "reference": {"type": "string"},
                "equivalent": {"type": ["boolean", "null"]},
                "structurally_equivalent": {"type": ["boolean", "null"]},
                "export_error": {"type": ["string", "null"]},
                "compare_error": {"type": ["string", "null"]},
                "comparison": COMPARISON_SCHEMA,
            },
        },
        "wiring": {
            "type": "object",
            "description": (
                "Connectivity counts over components with resolvable geometry: of "
                "pins_total pins, pins_wired have a wire through them and "
                "pins_label_only carry a net-label but no wire. The counts are "
                "whole-sheet; label_only_pins pages the label-only pins this batch "
                "introduced or named, and preexisting.label_only_pins counts the "
                "rest, so the two sum to pins_label_only."
            ),
            "properties": {
                "wire_segments": {"type": "integer"},
                "pins_total": {"type": "integer"},
                "pins_wired": {"type": "integer"},
                "pins_label_only": {"type": "integer"},
                "label_only_pins": _PAGE_SCHEMA,
            },
            "required": ["pins_total", "pins_wired", "pins_label_only", "label_only_pins"],
        },
        "preexisting": {
            "type": "object",
            "description": (
                "What the sheet already reported before this batch and that involves "
                "no reference or coordinate the batch named: counted here, left out "
                "of 'warnings' and wiring.label_only_pins. count = findings + "
                "label_only_pins. Echo cursor as view_cursors.preexisting, or ask "
                "return_views ['preexisting'], to list them."
            ),
            "properties": {
                "count": {"type": "integer"},
                "findings": {
                    "type": "integer",
                    "description": "Sheet findings (floating pins, dangling labels, ...).",
                },
                "label_only_pins": {"type": "integer"},
                "cursor": {
                    "type": ["string", "null"],
                    "description": "Starts the views.preexisting page; null when count is 0.",
                },
            },
            "required": ["count", "findings", "label_only_pins", "cursor"],
        },
        "views": {
            "type": "object",
            "properties": {
                "touched": _PAGE_SCHEMA,
                "pin_legend": _PAGE_SCHEMA,
                "preexisting": page_schema(
                    items={
                        "type": "array",
                        "items": {
                            "anyOf": [
                                VALIDATION_WARNINGS_SCHEMA["items"],
                                _LABEL_ONLY_ROW_SCHEMA,
                            ]
                        },
                        "description": (
                            "Sheet findings as the validation pass reports them, then "
                            "label-only pins with kind 'label_only_pin'."
                        ),
                    },
                    primary_truncated=_PAGE_SCHEMA["properties"]["primary_truncated"],
                ),
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "failures": {"type": "array", "items": {"type": "object"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "hint": {"type": "string"},
    },
    "required": ["outcome", "commit_state", "target", "build_id", "stages"],
}


# ---------------------------------------------------------------------------
# Commit protocol helpers (seams for crash-injection tests)
# ---------------------------------------------------------------------------


def _target_mode(target: Path) -> int:
    """The mode bits to stage with: the target's own when it pre-exists, else 0o644.

    Preserving an existing file's permissions keeps a committed edit from silently
    widening/narrowing access that a caller set on the sheet.
    """
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return 0o644


def _stage_asc(text: str, target: Path, build_id: str, encoding: str) -> Path:
    """Write ``text`` to a fsync'd temp sibling of ``target`` (pre-rename)."""
    tmp = target.with_name(f"{target.name}.staging-{build_id}")
    data = text.encode(encoding)
    # Binary, or Windows turns every "\n" into "\r\n" and the bytes on disk stop
    # matching the sha the reply reports: every later expected_sha256 conflicts.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_BINARY, _target_mode(target))
    try:
        os.write(fd, data)
        fsync_fd(fd)
    finally:
        os.close(fd)
    return tmp


def _commit_rename(tmp: Path, target: Path) -> None:
    """Atomically move the staged file onto the target — the LAST commit step."""
    replace_file(tmp, target)
    with contextlib.suppress(OSError):
        fsync_dir(target.parent)


class _CommitOutcome(NamedTuple):
    """Per-phase result of the offloaded commit-write function."""

    staged: bool
    renamed: bool
    error: str | None


def _commit_asc(text: str, target: Path, build_id: str, encoding: str) -> _CommitOutcome:
    """The whole commit write path (stage + fsync + atomic rename + dir fsync).

    Run as one blocking unit off the event loop and under ``asyncio.shield`` so a
    transport cancel can't abandon a half-commit. Returns per-phase outcomes so
    the caller keeps its two-stage bookkeeping; on a rename failure it removes the
    orphaned staging temp itself (the target is left untouched either way).
    """
    try:
        tmp = _stage_asc(text, target, build_id, encoding)
    except Exception as exc:  # broad by design — pre-rename failure; target untouched
        return _CommitOutcome(staged=False, renamed=False, error=str(exc))
    try:
        _commit_rename(tmp, target)
    except Exception as exc:  # broad by design — pre-commit rename failure; target untouched
        with contextlib.suppress(OSError):
            tmp.unlink()
        return _CommitOutcome(staged=True, renamed=False, error=str(exc))
    return _CommitOutcome(staged=True, renamed=True, error=None)


async def _export_asc_to_netlist(asc_copy: Path, state: SessionState) -> str:
    """Export a committed-sheet COPY to a netlist and return its text (seam).

    Uses ``resolve_runnable_netlist`` — which takes its own asc_export_lock on
    the copy's path, so there is no reentrancy with the guard-held target lock.
    Isolated as its own function so tests can substitute a netlist without a
    real LTspice binary.
    """
    from ltspice_mcp.lib.encoding import read_spice_text

    net_path = await resolve_runnable_netlist(str(asc_copy), state)
    return await asyncio.to_thread(read_spice_text, net_path)


# ---------------------------------------------------------------------------
# Geometry / view builders
# ---------------------------------------------------------------------------


def _pin_tables(editor, *, include_legend: bool) -> tuple[list[dict], list[dict]]:
    """The pin legend (only when ``include_legend``) and the label-only pins of a placed editor.

    A label-only pin has no wire through it, so its net holds nothing but what
    sits at its own coordinate, and its net name is the labels there — read
    without tracing the sheet. The whole-sheet trace (``trace_nets``, the
    union-find over every segment and point) runs only for the legend, which
    names the net of every pin. The plain results go to the pure
    ``lib/pin_legend`` builders.
    """
    geometry = collect_component_geometry(editor)
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    labels_at: dict[tuple[int, int], set[str]] = {}
    for lbl in editor.labels:
        labels_at.setdefault((int(lbl.coord.X), int(lbl.coord.Y)), set()).add(lbl.text)

    def net_name(labels: Iterable[str]) -> str | None:
        return "/".join(sorted(labels)) or None

    label_only = find_label_only_pins(
        geometry,
        is_wired=build_on_wire_predicate(segments),
        is_labeled=lambda c: c in labels_at,
        net_name_of=lambda c: net_name(labels_at.get(c, ())),
    )
    if not include_legend:
        return [], label_only
    nets = trace_nets(editor)
    return build_pin_legend(geometry, lambda c: net_name(nets.get(c, ()))), label_only


# The ``kind`` a label-only pin carries in the preexisting view, beside the
# validation pass's own finding kinds.
_LABEL_ONLY_KIND = "label_only_pin"


def _sheet_report(editor) -> tuple[list[dict], list[dict]]:
    """``(findings, label_only_pins)`` of a placed editor: what the sheet says about itself."""
    return post_op_warnings(editor), _pin_tables(editor, include_legend=False)[1]


def _names_ref(row: Mapping[str, Any], refs: set[str]) -> bool:
    """Whether a row's ``ref`` is one of ``refs`` (casefolded references)."""
    ref = row.get("ref")
    return isinstance(ref, str) and ref.casefold() in refs


def _row_coords(row: dict[str, Any]) -> list[tuple[int, int]]:
    """The coordinates a row names: its anchor, or a wire's two ends."""
    coords = []
    if isinstance(row.get("x"), int) and isinstance(row.get("y"), int):
        coords.append((row["x"], row["y"]))
    for end in ("from", "to"):
        point = row.get(end)
        if isinstance(point, dict):
            coords.append((point["x"], point["y"]))
    return coords


def _split_by_edit(
    after: Sequence[dict[str, Any]],
    before: Sequence[dict[str, Any]],
    refs: set[str],
    coords: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(reported, preexisting)``: the rows of ``after`` this batch accounts for, and the rest.

    A row is reported when the sheet did not have it before the batch — any
    change to it counts, since its identity is every field — or when it names a
    reference in ``refs`` (casefolded) or a coordinate in ``coords``. Every
    other row was already there and involves nothing the batch named. Identity
    is counted, not merely matched, so a second copy of a row that was there
    once is new.
    """
    remaining = Counter(map(canonical_json, before))
    reported: list[dict[str, Any]] = []
    preexisting: list[dict[str, Any]] = []
    for row in after:
        key = canonical_json(row)
        was_there = remaining[key] > 0
        if was_there:
            remaining[key] -= 1
        named = _names_ref(row, refs) or any(coord in coords for coord in _row_coords(row))
        (preexisting if was_there and not named else reported).append(row)
    return reported, preexisting


@dataclass(frozen=True)
class EditSchematicViews:
    """Complete pin/net views tied to the bytes produced by one transaction.

    ``label_only_pins`` holds the label-only pins this batch introduced or
    named; ``preexisting`` holds the sheet findings and label-only pins it left
    out (see ``_split_by_edit``), each row tagged with its ``kind``.

    Drawing the sheet is not among them: ``verify_circuit`` owns rendering, and
    its policy is the more capable one (a pixel cap, inline delivery, and
    render-only mode), so an edit that also wants a picture is one call away
    from it.
    """

    sha256: str
    wiring_profile: dict[str, int]
    pin_legend: tuple[dict[str, Any], ...]
    label_only_pins: tuple[dict[str, Any], ...]
    preexisting: tuple[dict[str, Any], ...]


def _build_edit_views(
    profile: dict[str, int],
    legend: list[dict],
    label_only: list[dict],
    preexisting: list[dict],
    sheet_sha256: str,
) -> EditSchematicViews:
    """Package the transaction's own views.

    ``sheet_sha256`` is the digest of the same bytes the commit protocol writes;
    the caller already has it, so the sheet is encoded and hashed once per edit.
    """
    return EditSchematicViews(
        sha256=sheet_sha256,
        wiring_profile=profile,
        pin_legend=tuple(legend),
        label_only_pins=tuple(label_only),
        preexisting=tuple(preexisting),
    )


def _preexisting_block(findings: int, label_only_pins: int) -> dict[str, Any]:
    """The count of what was left out, by collection, and where listing it starts."""
    count = findings + label_only_pins
    return {
        "count": count,
        "findings": findings,
        "label_only_pins": label_only_pins,
        "cursor": encode_page_cursor("preexisting", 0) if count else None,
    }


def _preexisting_hint(block: dict[str, Any], *, listed: bool) -> str | None:
    """One sentence on what was left out, or None when nothing was.

    ``listed`` is whether this reply already carries the preexisting view.
    """
    if not block["count"]:
        return None
    where = (
        "listed in views.preexisting"
        if listed
        else "counted in preexisting, not listed; add 'preexisting' to return_views, "
        "or echo preexisting.cursor as view_cursors.preexisting, to list them"
    )
    return (
        f"{block['findings']} sheet finding(s) and {block['label_only_pins']} label-only "
        f"pin(s) predate this batch and involve nothing it named, so they are {where}."
    )


def touched_refs(ops: list[ConsolidatedOp]) -> set[str]:
    """Component references this op batch named, casefolded.

    Reads the ops' own addressing rather than diffing the sheet: an op names
    the component it acts on, and a pin endpoint (``M1.D``) names one too. A
    ``net:NAME`` endpoint and the coordinate forms name no component and
    contribute nothing.
    """
    refs: set[str] = set()

    def add_pin(pin: str | None) -> None:
        if pin and not pin.lower().startswith("net:") and "." in pin:
            refs.add(pin.split(".", 1)[0].casefold())

    for op in ops:
        reference = getattr(op, "reference", None)
        if isinstance(reference, str) and reference:
            refs.add(reference.casefold())
        add_pin(getattr(op, "from_pin", None))
        add_pin(getattr(op, "to_pin", None))
        add_pin(getattr(op, "pin", None))
    return refs


def touched_coords(ops: list[ConsolidatedOp]) -> set[tuple[int, int]]:
    """Sheet coordinates this op batch named.

    An op's own ``x``/``y`` (a label, a directive anchor, a component origin,
    a wire's incident point), both ends of an exact wire segment, and every
    routing waypoint. A pin endpoint names a component, not a coordinate; see
    ``touched_refs``.
    """
    coords: set[tuple[int, int]] = set()
    for op in ops:
        for xname, yname in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
            x, y = getattr(op, xname, None), getattr(op, yname, None)
            if isinstance(x, int) and isinstance(y, int):
                coords.add((x, y))
        for point in getattr(op, "waypoints", None) or ():
            coords.add((point.x, point.y))
    return coords


def _requested_views(args: EditSchematicInput) -> list[str]:
    """The views to return: those in return_views, then any whose cursor was echoed.

    Echoing a page's cursor is itself the request for that page, so it never
    needs a second spelling in return_views to take effect.
    """
    cursors = args.view_cursors or EditViewCursors()
    echoed = [view for view in _VIEWS if getattr(cursors, view)]
    return list(dict.fromkeys([*args.return_views, *echoed]))


def _present_edit_views(
    args: EditSchematicInput,
    neutral: EditSchematicViews,
    *,
    complete: bool = False,
) -> tuple[dict, dict]:
    """Apply MCP paging, or build the same page shapes without omissions."""
    cursors = args.view_cursors or EditViewCursors()

    def page(rows: Sequence[dict[str, Any]], kind: str) -> dict[str, Any]:
        if complete:
            return paginate_view(rows, kind, limit=max(len(rows), 1))
        return paginate_view(rows, kind, cursor=getattr(cursors, kind), limit=args.view_limit)

    views: dict[str, Any] = {}
    for view in _requested_views(args):
        rows = neutral.preexisting if view == "preexisting" else neutral.pin_legend
        if view == "touched":
            wanted = touched_refs(args.ops)
            rows = [row for row in rows if _names_ref(row, wanted)]
        views[view] = page(rows, view)
    return page(neutral.label_only_pins, "label_only_pins"), views


def _paged_edit_views(
    args: EditSchematicInput,
    profile: dict[str, int],
    neutral: EditSchematicViews,
    *,
    present_mcp_views: bool,
) -> tuple[dict | None, dict | None]:
    """The wiring block and view pages an MCP response carries, or both absent."""
    if not present_mcp_views:
        return None, None
    label_only_page, views = _present_edit_views(args, neutral)
    return _wiring_dict(profile, label_only_page), views


# ---------------------------------------------------------------------------
# Reference stage (post-commit)
# ---------------------------------------------------------------------------


def _write_export_copy(
    export_root: Path, copy_asc: Path, committed_text: str, encoding: str
) -> None:
    """Materialize the committed-sheet copy under a fresh export dir (blocking)."""
    export_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(copy_asc, committed_text.encode(encoding), durable=False)


async def _exported_reference(
    ref: Path, export_root: Path, state: SessionState
) -> ReferenceNetlist:
    """An ``.asc`` reference exported the way the committed sheet is: a copy
    beside the committed-sheet copy, through ``_export_asc_to_netlist``."""
    ref_copy = export_root / "reference.asc"
    await asyncio.to_thread(shutil.copyfile, ref, ref_copy)
    try:
        return ReferenceNetlist(await _export_asc_to_netlist(ref_copy, state))
    except Exception as exc:  # broad by design — export failure is a reported fact
        return ReferenceNetlist(
            None, f"the reference {ref.name} could not be exported to a netlist: {exc}"
        )


async def _run_reference_stage(
    committed_text: str,
    encoding: str,
    ref: str | Path,
    spec: VerifyCompareSpec,
    build_id: str,
    state: SessionState,
    target: Path,
) -> dict:
    """Export a copy of the committed sheet and compare it to ``ref`` (a path or netlist text).

    Runs entirely on a COPY under a managed temp dir, so the exporter's own
    asc_export_lock keys on the copy's path (no reentrancy with the guard-held
    target lock). The reference path is already resolved — the handler validates
    it in pre-flight, so no path rejection can land here, after the commit. An
    export failure is captured into the returned dict; anything else escapes to
    the handler, which reports it on a committed envelope. Either way the sheet
    is already committed and stays so.
    """
    verification: dict[str, Any] = {
        "reference": str(ref) if isinstance(ref, Path) else "inline netlist"
    }
    export_root = state.store.edit_export(build_id)
    copy_asc = export_root / "committed.asc"
    try:
        await asyncio.to_thread(
            _write_export_copy, export_root, copy_asc, committed_text, encoding
        )
        try:
            netlist_text = await _export_asc_to_netlist(copy_asc, state)
        except Exception as exc:  # broad by design — export failure is a reported fact
            verification["export_error"] = str(exc)
            verification["equivalent"] = None
            return verification
        verification["_netlist"] = netlist_text
        ref_netlist = reference_as_given(ref)
        if ref_netlist is None:
            assert isinstance(ref, Path)  # only an .asc path needs exporting
            ref_netlist = await _exported_reference(ref, export_root, state)
        if ref_netlist.source is None:
            verification["compare_error"] = ref_netlist.error
            verification["equivalent"] = None
            return verification
        ref_source = ref if isinstance(ref, Path) else target
        payload, findings, failure, cmp_warnings = await asyncio.to_thread(
            compare_netlists,
            spec,
            ref_netlist.source,
            netlist_text,
            ref_source,
            copy_asc.with_suffix(".net"),
            state,
        )
        verification["_warnings"] = cmp_warnings + [
            f"{f.get('rule_id')}: {(f.get('evidence') or {}).get('detail') or f.get('subject')}"
            for f in findings
        ]
        if failure is not None:
            verification["compare_error"] = failure["error"]
            verification["equivalent"] = None
            return verification
        assert payload is not None  # a compare without a failure carries its payload
        verification["equivalent"] = payload.get("equivalent")
        verification["structurally_equivalent"] = payload.get("structurally_equivalent")
        verification["comparison"] = payload
        return verification
    finally:
        await asyncio.to_thread(shutil.rmtree, export_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _build_editor(target: Path, use_template: bool, state: SessionState) -> AscEditor:
    """Return the editor to mutate: a fresh blank-template one, or the cached target."""
    if not use_template:
        return get_asc_editor(target, state)
    # Construct an AscEditor from a throwaway blank template; it parses on init,
    # so the temp file can go away immediately and all mutation is in-memory.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltspice-blank-"))
    try:
        tmp_asc = tmp_dir / "blank.asc"
        tmp_asc.write_text(_BLANK_TEMPLATE, encoding="utf-8")
        # The template is .asc, so make_editor always yields an AscEditor here.
        return cast(AscEditor, make_editor(tmp_asc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _apply_ops(
    editor, ops, target: Path, dry_run: bool
) -> tuple[list[dict], list[dict], str | None]:
    """Apply every op in order. Returns (results, failures, abort_reason).

    Delegates the loop to the shared ``run_op_batch`` runner (abort-on-first-
    failure unless ``dry_run``), then splits its unified entries into this
    surface's separate success/failure lists. Identical advisories across the
    batch are collapsed on the way out — see ``_op_warnings``.
    """
    entries, abort_reason = run_op_batch(editor, ops, target, stop_on_error=not dry_run)
    collapse_result_warnings(entries)
    results = [e for e in entries if e["ok"]]
    failures = [
        {"index": e["index"], "op": e["op"], "error": e["error"]} for e in entries if not e["ok"]
    ]
    return results, failures, abort_reason


def _op_warnings(results: list[dict]) -> list[str]:
    """The batch's per-op advisories, attributed to the op that raised them.

    An op can succeed and still have something to say — a duplicate net label,
    a bbox-crossing wire, orphaned wires left behind by a removal. The envelope
    carries one flat ``warnings`` list, so each is prefixed with its op; the
    repeats are already collapsed by ``collapse_result_warnings``, which keeps
    the first occurrence and annotates it with how many ops it covers (the
    documented per-pin-label style repeats one advisory on every label op, and
    a converter-scale batch would otherwise spend hundreds of lines on it).
    """
    return [
        f"op {entry['index']} ({entry['op']}): {message}"
        for entry in results
        for message in entry.get("warnings", ())
    ]


def _mirror_commit_state(commit_state: str) -> Literal["not_started", "committed", "unknown"]:
    """Project the top-level commit_state onto the error envelope's enum.

    The top-level field uses ``committed``/``not_committed``; the shared error
    envelope uses ``not_started``/``committed``/``unknown``. A not-yet-written
    sheet mirrors to ``not_started``.
    """
    return "committed" if commit_state == "committed" else "not_started"


def _envelope(
    *,
    delivered: bool,
    commit_state: str,
    target: Path,
    build_id: str,
    base: str,
    stages: list[dict],
    partial: bool = False,
    sha256: str | None = None,
    error: dict | None = None,
    wiring: dict | None = None,
    views: dict | None = None,
    preexisting: dict | None = None,
    verification: dict | None = None,
    netlist: str | None = None,
    warnings: list[str] | None = None,
    failures: list[dict] | None = None,
    observations: list[str] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Build one response payload; the shared rule decides its outcome.

    Every exit from this tool comes through here, so no call site names an
    outcome. A call site states what happened — whether anything usable came
    back (``delivered``) and whether the call fell short of what was asked for
    (``partial``) — and the one rule in ``_base`` turns that into the verdict.
    A failure lives in ``failures`` or in ``error`` depending on the path;
    either one is the same signal here.
    """
    data: dict[str, Any] = {
        "outcome": outcome_of(failures or error, partial=partial, delivered=delivered),
        "commit_state": commit_state,
        "target": str(target),
        "build_id": build_id,
        "base": base,
        "stages": stages,
        "sha256": sha256,
        "warnings": warnings or [],
        "failures": failures or [],
        "observations": observations or [],
    }
    if error is not None:
        # Mirror the top-level commit_state into the error object so a failure
        # envelope is self-describing without cross-referencing the top level.
        error.setdefault("commit_state", _mirror_commit_state(commit_state))
        data["error"] = error
    if wiring is not None:
        data["wiring"] = wiring
    if preexisting is not None:
        data["preexisting"] = preexisting
    if views:
        data["views"] = views
    if verification is not None:
        data["verification"] = verification
    if netlist is not None:
        data["netlist"] = netlist
    if hint is not None:
        data["hint"] = hint
    return data


@dataclass(frozen=True)
class EditSchematicEvaluation:
    """One mutation result plus its complete transaction-bound in-memory views."""

    data: dict[str, Any]
    text: str
    views: EditSchematicViews | None = None
    mcp_result: types.CallToolResult | None = None


def _finish_edit_evaluation(
    evaluation: EditSchematicEvaluation,
    *,
    present_mcp_views: bool,
) -> EditSchematicEvaluation:
    """Optionally format inside the transaction's guarded error boundary."""
    if not present_mcp_views:
        return evaluation
    return replace(
        evaluation,
        mcp_result=format_response(evaluation.text, evaluation.data),
    )


async def _evaluate_edit_schematic(
    args: EditSchematicInput,
    state: SessionState,
    *,
    present_mcp_views: bool = False,
) -> EditSchematicEvaluation:
    """Implementation shared by the neutral seam and guarded MCP presentation."""
    target = safe_path(args.target, state)
    require_asc(target)
    views = _requested_views(args)
    if not args.ops and not views:
        raise NetlistError(
            "ops list is empty — pass at least one op, or name return_views to "
            "read the sheet's pin table without changing it."
        )
    # An op-less batch is a read: nothing to commit, so it runs as a dry run.
    dry_run = args.dry_run or not args.ops
    _validate_view_cursors(args.view_cursors)
    # Resolve the optional reference here, with the rest of the argument checks:
    # a path outside allowed_paths is an argument fault the caller fixes by
    # resending, not a property of the sheet. Resolving it in the post-commit
    # stage instead would reject the call after the target was already written.
    compare = args.compare
    reference_path = resolve_reference(compare.reference, state) if compare is not None else None

    build_id = generate_id("build")
    stages: list[dict] = []

    def finish(evaluation: EditSchematicEvaluation) -> EditSchematicEvaluation:
        return _finish_edit_evaluation(
            evaluation,
            present_mcp_views=present_mcp_views,
        )

    def _stage(name: str, ok: bool = True, error: str | None = None) -> None:
        """Append one commit-protocol stage entry (the ok path omits ``error``).

        Recording an outcome also ends that stage: ``post_commit_stage`` drops
        back to "response", so a later failure cannot be appended as a second,
        contradictory entry for a stage that already reported ok. A stage names
        itself right before it runs; nothing has to remember to un-name it.
        """
        nonlocal post_commit_stage
        entry: dict[str, Any] = {"stage": name, "ok": ok}
        if error is not None:
            entry["error"] = error
        stages.append(entry)
        post_commit_stage = "response"

    async with edit_guard(target):
        # --- revision guard (inside the guard so a peer's committed write is seen)
        exists = target.exists()
        expected = args.expected_sha256.lower() if args.expected_sha256 else None
        if exists:
            current = sha256_file(target)
            if expected is None:
                # The guard stands — nothing is written without the token — but
                # the refusal hands the token over rather than sending the
                # caller off to fetch it. A digest read here is a fact about
                # the file as it is right now, under the same edit guard the
                # write would take, so a retry that quotes it is exactly as
                # safe as one quoting a prior read: a peer's write between the
                # two still loses the race and comes back as revision_conflict.
                _stage("revision_check", False, "expected_sha256 missing")
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            delivered=False,
                            commit_state="not_committed",
                            target=target,
                            build_id=build_id,
                            base=args.base,
                            stages=stages,
                            sha256=current,
                            error={
                                "code": "expected_sha256_required",
                                "message": (
                                    f"{target.name} already exists, so expected_sha256 is "
                                    "required — it is what keeps a concurrent edit from "
                                    f"being lost. Its current sha256 is {current}; resubmit "
                                    "with that if it is the revision you edited against."
                                ),
                                "stage": "revision_check",
                                "retryable": True,
                            },
                            hint=(
                                "Nothing was written. Resubmit the same ops with "
                                f"expected_sha256={current}."
                            ),
                        ),
                        text=(
                            f"edit_schematic: {target.name} exists and needs "
                            f"expected_sha256; its current sha256 is {current}. "
                            "Nothing was written."
                        ),
                    )
                )
            if current != expected:
                _stage("revision_check", False, "sha mismatch")
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            delivered=False,
                            commit_state="not_committed",
                            target=target,
                            build_id=build_id,
                            base=args.base,
                            stages=stages,
                            sha256=current,
                            error={
                                "code": "revision_conflict",
                                "message": (
                                    f"expected_sha256 {expected} does not match the current "
                                    f"file ({current}); nothing was written."
                                ),
                                "stage": "revision_check",
                                "retryable": True,
                            },
                            hint="Re-read the target, then resubmit with its current sha256.",
                        ),
                        text=(
                            f"edit_schematic: revision_conflict on {target.name} — the file "
                            "changed since you read it. Re-read it and resubmit with the "
                            "current sha256."
                        ),
                    )
                )
        _stage("revision_check")

        use_template = args.base == "blank" or not exists
        editor = _build_editor(target, use_template, state)
        # Set once the atomic rename lands. From that point every escape must be
        # reported on a committed envelope instead of re-raised (see the except
        # clauses below); post_commit_stage names the stage that was in flight,
        # and "response" means none is — the failure is in assembling the reply.
        committed_sha: str | None = None
        post_commit_stage = "response"
        try:
            # What the sheet already said about itself, read before the ops run,
            # so the reply can tell what this batch introduced from what it found.
            # A blank base starts from nothing; an op-less read changes nothing,
            # so its "before" is its "after" (None here) and is not read twice.
            before: tuple[list[dict], list[dict]] | None = None
            if use_template:
                before = ([], [])
            elif args.ops:
                before = _sheet_report(editor)
            results, failures, abort_reason = _apply_ops(editor, args.ops, target, dry_run)

            # --- op failure → transactional abort (nothing written)
            if abort_reason is not None:
                state.editors.invalidate(target)
                _stage("apply_ops", False, abort_reason)
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            delivered=False,
                            commit_state="not_committed",
                            target=target,
                            build_id=build_id,
                            base=args.base,
                            stages=stages,
                            failures=failures,
                            error={
                                "code": "op_failed",
                                "message": abort_reason,
                                "stage": "apply_ops",
                                "retryable": False,
                            },
                            hint="Fix the failing op and resubmit with the same expected_sha256.",
                        ),
                        text=(
                            f"edit_schematic: transaction aborted — {abort_reason}. "
                            "No changes saved."
                        ),
                    )
                )
            _stage("apply_ops")

            profile = wiring_profile(editor)
            legend, label_only = _pin_tables(
                editor, include_legend=bool({"pin_legend", "touched"} & set(views))
            )
            findings = post_op_warnings(editor)
            before_findings, before_pins = before or (findings, label_only)
            refs, coords = touched_refs(args.ops), touched_coords(args.ops)
            findings_reported, findings_left_out = _split_by_edit(
                findings, before_findings, refs, coords
            )
            pins_reported, pins_left_out = _split_by_edit(label_only, before_pins, refs, coords)
            preexisting_rows = findings_left_out + [
                {"kind": _LABEL_ONLY_KIND, **row} for row in pins_left_out
            ]
            preexisting = _preexisting_block(len(findings_left_out), len(pins_left_out))
            left_out_hint = _preexisting_hint(preexisting, listed="preexisting" in views)
            # Two sources, one channel: what the ops themselves reported, then
            # what the finished sheet reports about itself that this batch
            # accounts for. The rest is counted under preexisting.
            warnings = _op_warnings(results) + [w["message"] for w in findings_reported]
            encoding = getattr(editor, "encoding", "utf-8") or "utf-8"
            committed_text = _render_editor_text(editor)

            # --- dry run: validate-only, nothing written, target dir untouched
            if dry_run:
                state.editors.invalidate(target)
                neutral_views = _build_edit_views(
                    profile,
                    legend,
                    pins_reported,
                    preexisting_rows,
                    hashlib.sha256(committed_text.encode(encoding)).hexdigest(),
                )
                wiring, presented_views = _paged_edit_views(
                    args,
                    profile,
                    neutral_views,
                    present_mcp_views=present_mcp_views,
                )
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            delivered=True,
                            commit_state="not_committed",
                            target=target,
                            build_id=build_id,
                            base=args.base,
                            stages=stages,
                            wiring=wiring,
                            views=presented_views,
                            preexisting=preexisting,
                            warnings=warnings,
                            failures=failures,
                            hint=" ".join(
                                filter(
                                    None,
                                    (
                                        "Dry run — resubmit without dry_run to commit.",
                                        left_out_hint,
                                    ),
                                )
                            ),
                        ),
                        text=(
                            f"edit_schematic (dry run) on {target.name}: {len(results)} ops "
                            "validated; nothing saved."
                        ),
                        views=neutral_views,
                    )
                )

            # --- commit protocol: assets (none today) → stage → rename LAST.
            # The whole write path runs off the loop as one shielded unit so a
            # transport cancel can't abandon a half-commit while the guard releases.
            _stage("stage_assets")
            outcome = await asyncio.shield(
                asyncio.to_thread(_commit_asc, committed_text, target, build_id, encoding)
            )
            if not outcome.staged:
                state.editors.invalidate(target)
                _stage("stage_asc", False, outcome.error)
                return _commit_failure_response(
                    args,
                    target,
                    build_id,
                    stages,
                    outcome.error or "",
                    present_mcp_views=present_mcp_views,
                )
            _stage("stage_asc")
            if not outcome.renamed:
                state.editors.invalidate(target)
                _stage("rename", False, outcome.error)
                return _commit_failure_response(
                    args,
                    target,
                    build_id,
                    stages,
                    outcome.error or "",
                    present_mcp_views=present_mcp_views,
                )
            _stage("rename")
            state.editors.invalidate(target)
            # The bytes we just staged and renamed ARE the file — hash them in
            # memory instead of re-reading the target back off disk.
            committed_sha = hashlib.sha256(committed_text.encode(encoding)).hexdigest()

            # --- post-commit: everything below keeps commit_state='committed'
            # Views report no stage entry of their own, so they open and close
            # their name by hand; a stage that calls _stage() only opens it.
            post_commit_stage = "views"
            neutral_views = _build_edit_views(
                profile, legend, pins_reported, preexisting_rows, committed_sha
            )
            wiring, presented_views = _paged_edit_views(
                args,
                profile,
                neutral_views,
                present_mcp_views=present_mcp_views,
            )
            post_commit_stage = "response"

            verification = None
            netlist = None
            if reference_path is not None:
                post_commit_stage = "reference"
                assert compare is not None  # reference_path implies it
                verification = await _run_reference_stage(
                    committed_text, encoding, reference_path, compare, build_id, state, target
                )
                exported = verification.pop("_netlist", None)
                warnings.extend(verification.pop("_warnings", []))
                mismatch = comparison_mismatch(verification)
                _stage("reference", not mismatch)
                # A confirmed match is the answer, and the exported deck only
                # restates the reference the caller supplied. A mismatch, a
                # compare error or no verdict keeps it: it is the sheet's side
                # of a comparison the caller now has to diagnose.
                if mismatch:
                    netlist = exported

            hint = _commit_hint(profile, verification, left_out_hint)
            return finish(
                EditSchematicEvaluation(
                    data=_envelope(
                        delivered=True,
                        partial=comparison_mismatch(verification),
                        commit_state="committed",
                        target=target,
                        build_id=build_id,
                        base=args.base,
                        stages=stages,
                        sha256=committed_sha,
                        wiring=wiring,
                        views=presented_views,
                        preexisting=preexisting,
                        verification=verification,
                        netlist=netlist,
                        warnings=warnings,
                        hint=hint,
                    ),
                    text=f"edit_schematic committed {target.name} (build {build_id}).",
                    views=neutral_views,
                )
            )
        except Exception as exc:
            # Any escape leaves the cached editor dirty — evict so the next read
            # re-parses from disk (the on-disk file is intact or already renamed).
            state.editors.invalidate(target)
            if committed_sha is None:
                raise
            # Read the name before recording it: _stage ends the stage it
            # records, so post_commit_stage is "response" by the time it returns.
            failed_stage = post_commit_stage
            _stage(failed_stage, False, str(exc))
            return _post_commit_failure_response(
                args,
                target,
                build_id,
                stages,
                committed_sha,
                failed_stage,
                str(exc),
                present_mcp_views=present_mcp_views,
            )
        except BaseException:
            # Cancellation (and any other non-Exception escape) still propagates
            # unchanged — it is not ours to convert into a response.
            state.editors.invalidate(target)
            raise


async def evaluate_edit_schematic(
    args: EditSchematicInput,
    state: SessionState,
) -> EditSchematicEvaluation:
    """Execute one transaction and retain its full views without replaying it."""
    return await _evaluate_edit_schematic(args, state, present_mcp_views=False)


def complete_edit_schematic_data(
    evaluation: EditSchematicEvaluation,
    args: EditSchematicInput,
) -> dict[str, Any]:
    """Return the existing response shape with every in-memory view row included.

    The envelope was built for this evaluation and nothing else reads it, so the
    complete views replace keys on a shallow copy of it.
    """
    data = dict(evaluation.data)
    if evaluation.views is None:
        return data
    label_only_page, views = _present_edit_views(
        args,
        evaluation.views,
        complete=True,
    )
    data["wiring"] = _wiring_dict(evaluation.views.wiring_profile, label_only_page)
    if views:
        data["views"] = views
    return data


@registry.tool(
    name="edit_schematic",
    title="Edit Schematic",
    description=(
        "Apply a typed op batch to an LTspice .asc schematic in one revision-"
        "guarded, transactional call. base='blank' builds a whole circuit from an "
        "empty sheet; base='existing' applies deltas. Pass expected_sha256 of the "
        "file you edited against (required when the target exists) — a peer that "
        "committed first yields revision_conflict with nothing written. Returns "
        "geometry facts (pin/net table, wiring metric) and an optional "
        "post-commit netlist verification against a reference. To draw the "
        "sheet, call verify_circuit with a render policy."
    ),
    input_model=EditSchematicInput,
    annotations=types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    output_schema=_OUTPUT_SCHEMA,
)
async def handle_edit_schematic(
    args: EditSchematicInput, state: SessionState
) -> types.CallToolResult:
    """Transactional .asc mutation with revision guard, commit protocol, and views."""
    evaluation = await _evaluate_edit_schematic(args, state, present_mcp_views=True)
    if evaluation.mcp_result is None:  # pragma: no cover - enforced by the call above
        raise RuntimeError("edit_schematic MCP presentation was not produced")
    return evaluation.mcp_result


def _validate_view_cursors(cursors: EditViewCursors | None) -> None:
    """Reject a malformed / cross-view resumption cursor before any work runs.

    Decoding up front (outside the edit guard) keeps a bad cursor from surfacing
    after the sheet is already committed.
    """
    if cursors is None:
        return
    for kind, cursor in (
        ("label_only_pins", cursors.label_only_pins),
        ("pin_legend", cursors.pin_legend),
        ("touched", cursors.touched),
        ("preexisting", cursors.preexisting),
    ):
        if cursor is None:
            continue
        try:
            decode_page_cursor(cursor, kind)
        except PageCursorError as exc:
            raise NetlistError(f"invalid view_cursors.{kind}: {exc}") from exc


def _render_editor_text(editor: AscEditor) -> str:
    """Render this sheet without losing ports or saving loaded child sheets."""
    _refuse_pending_child_edits(editor)
    label_counts = Counter(id(label) for label in editor.labels)
    port_directions: dict[int, str] = {}
    for port in editor.ports:
        label_id = id(port.text)
        if label_counts[label_id] != 1 or label_id in port_directions:
            raise NetlistError("Each hierarchical port must belong to exactly one unique label.")
        port_directions[label_id] = port.direction

    buf = io.StringIO()
    editor.save_netlist(buf)
    rendered = buf.getvalue()
    if not port_directions:
        return rendered

    # spicelib 1.5.1 emits FLAGs in label order but omits their IOPIN records.
    # Match occurrences, since different label objects may have identical text.
    labels = iter(editor.labels)
    lines: list[str] = []
    for line in rendered.splitlines(keepends=True):
        lines.append(line)
        if not line.startswith("FLAG "):
            continue
        label = next(labels, None)
        if label is None or line.rstrip("\r\n") != (
            f"FLAG {label.coord.X} {label.coord.Y} {label.text}"
        ):
            raise NetlistError("Cannot preserve hierarchical ports: serialized labels changed.")
        direction = port_directions.get(id(label))
        if direction is not None:
            ending = line[len(line.rstrip("\r\n")) :]
            lines.append(f"IOPIN {label.coord.X} {label.coord.Y} {direction}{ending}")
    if next(labels, None) is not None:
        raise NetlistError("Cannot preserve hierarchical ports: serialized labels are missing.")
    return "".join(lines)


def _refuse_pending_child_edits(editor: AscEditor) -> None:
    """StringIO does not stop spicelib from writing modified descendants."""
    pending = [editor]
    visited: set[int] = set()
    while pending:
        sheet = pending.pop()
        if id(sheet) in visited:
            continue
        visited.add(id(sheet))
        for component in sheet.components.values():
            child = component.attributes.get("_SUBCKT")
            if child is None:
                continue
            if getattr(child, "updated", False):
                raise NetlistError(
                    "Cannot save a parent sheet with pending child edits; "
                    "save or discard those child edits separately."
                )
            if isinstance(child, AscEditor):
                pending.append(child)


def _wiring_dict(profile: dict[str, int], label_only_page: dict) -> dict:
    return {
        "wire_segments": profile["wire_segments"],
        "pins_total": profile["pins_total"],
        "pins_wired": profile["pins_wired"],
        "pins_label_only": profile["pins_label_only"],
        "label_only_pins": label_only_page,
    }


def _commit_failure_response(
    args: EditSchematicInput,
    target: Path,
    build_id: str,
    stages: list[dict],
    error: str,
    *,
    present_mcp_views: bool,
) -> EditSchematicEvaluation:
    """Envelope for a commit failure before the rename; the target is untouched.

    Nothing is quarantined. The batch that failed is the caller's own ops plus
    the sheet it named, both of which it still holds, and the response says
    which stage failed and why — so a copy of the would-be bytes on disk beside
    the target added a file to clean up rather than a fact to act on.
    """
    return _finish_edit_evaluation(
        EditSchematicEvaluation(
            data=_envelope(
                delivered=False,
                commit_state="not_committed",
                target=target,
                build_id=build_id,
                base=args.base,
                stages=stages,
                error={
                    "code": "commit_failed",
                    "message": error,
                    # The failed commit phase is the last stage recorded before the abort.
                    "stage": stages[-1]["stage"] if stages else "commit",
                    "retryable": True,
                },
                hint="The sheet was not modified; retry with the same expected_sha256.",
            ),
            text=f"edit_schematic: commit failed before rename ({error}); target unchanged.",
        ),
        present_mcp_views=present_mcp_views,
    )


def _post_commit_failure_response(
    args: EditSchematicInput,
    target: Path,
    build_id: str,
    stages: list[dict],
    committed_sha: str,
    stage: str,
    error: str,
    *,
    present_mcp_views: bool,
) -> EditSchematicEvaluation:
    """Envelope for a failure AFTER the atomic rename — the sheet stays committed.

    The rename is the irreversible step: once it lands the caller MUST learn the
    new sha256, or its next edit sends the stale expected_sha256 and gets a
    revision_conflict on a file it just wrote successfully. So a post-commit
    escape is reported as a partial outcome carrying commit_state and the real
    sha, never re-raised — a raise returns no structuredContent at all, which is
    exactly the state the caller cannot recover from.
    """
    return _finish_edit_evaluation(
        EditSchematicEvaluation(
            data=_envelope(
                delivered=True,
                commit_state="committed",
                target=target,
                build_id=build_id,
                base=args.base,
                stages=stages,
                sha256=committed_sha,
                error={
                    "code": "post_commit_failed",
                    "message": error,
                    "stage": stage,
                    "retryable": False,
                },
                hint=(
                    f"The edit committed: {target.name} is now sha256 {committed_sha} — use that "
                    f"as expected_sha256 for your next edit. Only the post-commit {stage} stage "
                    "failed; re-run it separately if you need it."
                ),
            ),
            text=(
                f"edit_schematic committed {target.name} (build {build_id}), then the "
                f"post-commit {stage} stage failed: {error}. The sheet IS written."
            ),
        ),
        present_mcp_views=present_mcp_views,
    )


def _commit_hint(profile: dict[str, int], verification: dict | None, left_out: str | None) -> str:
    parts = [
        f"Committed. Of {profile['pins_total']} pins, {profile['pins_wired']} are on wires and "
        f"{profile['pins_label_only']} carry a net-label only."
    ]
    if left_out:
        parts.append(left_out)
    if verification is not None:
        if verification.get("export_error"):
            parts.append(f"Reference check could not export: {verification['export_error']}.")
        elif verification.get("compare_error"):
            parts.append(f"Reference check could not compare: {verification['compare_error']}.")
        elif verification.get("equivalent"):
            parts.append("Reference netlist verified: equivalent.")
        else:
            parts.append("Reference netlist differs — see verification.comparison.")
    return " ".join(parts)
