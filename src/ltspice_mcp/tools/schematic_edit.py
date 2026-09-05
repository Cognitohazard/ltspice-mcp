"""edit_schematic — transactional, revision-guarded ``.asc`` mutation (AUTHOR).

One input language: a typed op batch applied to a sheet that is either blank
(the whole-plan compile) or an existing file (deltas preserving untouched
content). The whole transaction — revision check, editor load, geometry
resolution, mutation, staged write, and the atomic rename onto the target — runs
inside the shared per-file edit guard, so a parallel session that committed
first turns a stale ``expected_sha256`` into a ``revision_conflict`` with nothing
written.

The op models and their in-place applier are reused verbatim from
``lib/schematic_ops.py`` (the shipped ``apply_schematic_ops`` machinery); the only
narrowing is that the wire op accepts ``wire_pins`` only — the deprecated
``connect`` alias is excluded from this surface. Post-commit, an optional
``reference`` stage exports the committed sheet on a COPY and compares it to a
reference netlist through the connectivity graph engine; a mismatch or an export
failure there is reported but never un-commits the sheet.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, cast

from mcp import types
from pydantic import Field
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import atomic_write_bytes, fsync_dir, fsync_fd
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.netlist_graph import IncludeResolver, compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.pin_legend import (
    PageCursorError,
    build_pin_legend,
    decode_page_cursor,
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
from ltspice_mcp.lib.schematic_scene import build_scene
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    StrictModel,
    ToolInput,
    format_response,
    make_include_resolver,
    registry,
    render_scene_artifact,
    resolve_runnable_netlist,
    safe_path,
    symbol_resolver_for,
)

# The blank-sheet template — identical to what ``create_schematic`` writes.
_BLANK_TEMPLATE = blank_sheet()

_DEFAULT_VIEW_LIMIT = 100


class OpWirePinsStrict(OpWirePins):
    """Draw an orthogonal wire between two pins, refusing a diagonal run, a pin
    collision, or an overlapping wire junction rather than drawing them.

    The shipped ``OpWirePins`` still accepts the deprecated ``connect`` alias;
    this consolidated surface drops it. Because the parent's applier dispatches
    on ``isinstance(op, OpWirePins)`` and reads ``op.op``, narrowing the literal
    is all that is needed — a ``connect`` payload no longer validates and never
    reaches the applier. The payload fields (and their descriptions) are the
    parent's; only the discriminator is narrowed.
    """

    op: Literal["wire_pins"] = "wire_pins"  # pyright: ignore[reportIncompatibleVariableOverride]


# The op union for this surface: the shipped models, with the wire op narrowed.
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
    | OpWirePinsStrict
    | OpAddDirective
    | OpRemoveDirective,
    Field(discriminator="op"),
]


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


class EditSchematicInput(ToolInput):
    target: str = Field(description="Path to the .asc schematic (created if absent).")
    base: Literal["existing", "blank"] = Field(
        default="existing",
        description=(
            "'existing' (default) applies the ops as deltas onto the current file, "
            "preserving untouched content. 'blank' treats the sheet as empty before "
            "applying the ops (a whole-circuit one-call build)."
        ),
    )
    expected_sha256: str | None = Field(
        default=None,
        description=(
            "REQUIRED whenever the target already exists (either base). The SHA-256 "
            "of the file you edited against, reported as 'sha256' by an inspect "
            "components/net query on the sheet and by every edit that commits; a "
            "mismatch means a peer committed first and the call returns "
            "revision_conflict with nothing written."
        ),
    )
    ops: list[ConsolidatedOp] = Field(
        description=(
            "Typed edit ops applied in order against one in-memory editor, each tagged "
            "by its 'op' field: add_component, move_component, remove_component, "
            "set_component_value, set_component_attribute, add_net_label, "
            "remove_net_label, wire_pins, remove_wire, add_directive, remove_directive. "
            "The whole batch commits atomically or not at all: the first op that fails "
            "aborts the transaction and nothing is written. " + COORDINATE_DESCRIPTION
        )
    )
    reference: str | None = Field(
        default=None,
        description=(
            "Optional netlist (.cir/.net) to verify the committed sheet against: the "
            "sheet is exported on a copy and compared for connectivity equivalence. "
            "Runs AFTER commit — a mismatch is reported but does not un-commit. The "
            "path itself is checked up front, so one outside the allowed roots is "
            "refused before the sheet is written, not after."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Resolve, validate, and compute geometry without writing. Every op is "
            "attempted so all problems surface at once; the target and its caches, "
            "snapshots, and artifacts are left untouched."
        ),
    )
    write_failed_draft: bool = Field(
        default=False,
        description=(
            "On a commit failure before the atomic rename, quarantine the would-be "
            "content to <target>.draft-<build_id>.asc for inspection."
        ),
    )
    return_views: list[Literal["touched", "pin_legend", "render"]] = Field(
        default_factory=lambda: ["touched"],
        description=(
            "Which geometry views to return. 'touched' (default) is the pin/net table "
            "for just the components this batch's ops named; 'pin_legend' is the same "
            "table for the whole sheet; 'render' is an SVG/PNG of it. A render needs a "
            "committed file, so under dry_run it reports metadata only and writes no "
            "artifact."
        ),
    )
    view_cursors: EditViewCursors | None = Field(
        default=None,
        description=(
            "Resume a paginated view by echoing back that page's next_cursor "
            "unmodified. Each cursor is bound to its own view and is checked before "
            "any work runs, so a bad one cannot surface after the sheet is committed."
        ),
    )
    view_limit: int = Field(
        default=_DEFAULT_VIEW_LIMIT,
        description=(
            "Page size for both paginated views — views.pin_legend and wiring.label_only_pins."
        ),
    )
    render_format: Literal["png", "svg"] = Field(
        default="png",
        description=(
            "Format for the 'render' view. PNG needs the optional 'raster' extra; "
            "without it the render is returned as SVG with a note naming the extra."
        ),
    )
    render_scale: float = Field(
        default=1.5,
        description="Raster scale for a PNG render (ignored for SVG).",
    )
    format: Literal["json", "text"] | None = Field(default=None, description=FORMAT_DESCRIPTION)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object"}},
        "total": {"type": "integer"},
        "returned": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "primary_truncated": {
            "type": "boolean",
            "description": (
                "Whether THIS collection has more rows. Equal to 'truncated' here; the "
                "two differ only on a page that carries a second collection."
            ),
        },
        "next_cursor": {"type": ["string", "null"]},
    },
    "required": ["items", "total", "returned", "truncated", "next_cursor"],
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["complete", "partial", "failed", "in_progress"],
        },
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
        "netlist": {"type": ["string", "null"]},
        "verification": {
            "type": "object",
            "description": (
                "Reference-compare result: equivalent is the single-glance verdict; "
                "the difference lists explain a false. Present only when a reference "
                "was supplied. 'export_error' means the sheet could not be exported."
            ),
            "properties": {
                "reference": {"type": "string"},
                "equivalent": {"type": ["boolean", "null"]},
                "structurally_equivalent": {"type": ["boolean", "null"]},
                "export_error": {"type": ["string", "null"]},
                "comparison": {"type": "object"},
            },
        },
        "wiring": {
            "type": "object",
            "description": (
                "Connectivity counts over components with resolvable geometry: of "
                "pins_total pins, pins_wired have a wire through them and "
                "pins_label_only carry a net-label but no wire. label_only_pins lists "
                "the label-only pins as an addressable, paginated page."
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
        "views": {
            "type": "object",
            "properties": {
                "touched": _PAGE_SCHEMA,
                "pin_legend": _PAGE_SCHEMA,
                "render": {"type": "object"},
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "failures": {"type": "array", "items": {"type": "object"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": {"type": "object"}},
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
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _target_mode(target))
    try:
        os.write(fd, data)
        fsync_fd(fd)
    finally:
        os.close(fd)
    return tmp


def _commit_rename(tmp: Path, target: Path) -> None:
    """Atomically move the staged file onto the target — the LAST commit step."""
    os.replace(tmp, target)
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
    return read_spice_text(net_path)


# ---------------------------------------------------------------------------
# Geometry / view builders
# ---------------------------------------------------------------------------


def _wiring_and_legend(
    editor, *, include_legend: bool
) -> tuple[dict[str, int], list[dict], list[dict]]:
    """Wiring counts, pin legend, and the label-only pins for a placed editor.

    Reuses circuit.py's ``net_partition``-backed helpers (via ``trace_nets``
    and ``build_on_wire_predicate``) rather than re-deriving connectivity, and
    hands the plain result to the pure ``lib/pin_legend`` builders. The wiring
    metric and label-only detection are contract output and always run; the
    per-component legend is built only when ``include_legend`` (its own view was
    requested), returned empty otherwise.
    """
    profile = wiring_profile(editor)
    geometry = collect_component_geometry(editor)
    nets = trace_nets(editor)
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    on_wire = build_on_wire_predicate(segments)
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    def net_name_of(coord: tuple[int, int]) -> str | None:
        names = nets.get(coord, frozenset())
        return "/".join(sorted(names)) or None

    legend = build_pin_legend(geometry, net_name_of) if include_legend else []
    label_only = find_label_only_pins(
        geometry,
        is_wired=on_wire,
        is_labeled=lambda c: c in label_coords,
        net_name_of=net_name_of,
    )
    return profile, legend, label_only


def _render_view(
    asc_path: Path,
    fmt: Literal["png", "svg"],
    scale: float,
    artifacts_dir: Path,
    *,
    resolver_path: Path | None = None,
) -> dict:
    """Render ``asc_path`` to SVG (always) or PNG (when the raster extra is
    present), writing a content-hashed artifact under ``artifacts_dir``."""
    scene = build_scene(
        asc_path,
        resolver=symbol_resolver_for(resolver_path or asc_path),
    )
    image, out_path, _ = render_scene_artifact(scene, artifacts_dir, image_format=fmt, scale=scale)
    view = dict(image.to_dict())
    if scene.diagnostics:
        view["diagnostics"] = list(scene.diagnostics)
    view["path"] = str(out_path)
    return view


def _render_committed_text(
    text: str,
    encoding: str,
    target: Path,
    fmt: Literal["png", "svg"],
    scale: float,
    artifacts_dir: Path,
) -> dict:
    """Render the transaction's bytes, never a later revision of ``target``."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltspice-edit-view-"))
    try:
        source = tmp_dir / target.name
        source.write_text(text, encoding=encoding)
        return _render_view(
            source,
            fmt,
            scale,
            artifacts_dir,
            resolver_path=target,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@dataclass(frozen=True)
class EditSchematicViews:
    """Complete requested views tied to the bytes produced by one transaction."""

    sha256: str
    wiring_profile: dict[str, int]
    pin_legend: tuple[dict[str, Any], ...]
    label_only_pins: tuple[dict[str, Any], ...]
    render: dict[str, Any] | None
    failures: tuple[dict[str, Any], ...]


async def _build_edit_views(
    args: EditSchematicInput,
    profile: dict[str, int],
    legend: list[dict],
    label_only: list[dict],
    committed_text: str,
    encoding: str,
    target: Path,
    state: SessionState,
    sheet_sha256: str,
) -> EditSchematicViews:
    """Build every requested view from transaction-owned memory.

    ``sheet_sha256`` is the digest of the same bytes the commit protocol writes;
    the caller already has it, so the sheet is encoded and hashed once per edit.
    """
    rendered: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = []
    if "render" in args.return_views:
        if args.dry_run:
            rendered = {
                "status": "dry_run",
                "note": (
                    "render artifact is written only on commit; run without dry_run to persist it."
                ),
            }
        else:
            try:
                artifacts_dir = state.working_dir / ".ltspice-mcp" / "renders"
                rendered = await asyncio.to_thread(
                    _render_committed_text,
                    committed_text,
                    encoding,
                    target,
                    args.render_format,
                    args.render_scale,
                    artifacts_dir,
                )
            except Exception as exc:  # broad by design — one view may fail independently
                failures.append({"stage": "render", "error": str(exc)})
    return EditSchematicViews(
        sha256=sheet_sha256,
        wiring_profile=profile,
        pin_legend=tuple(legend),
        label_only_pins=tuple(label_only),
        render=rendered,
        failures=tuple(failures),
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


def _present_edit_views(
    args: EditSchematicInput,
    neutral: EditSchematicViews,
    *,
    complete: bool = False,
) -> tuple[dict, dict]:
    """Apply MCP paging, or build the same page shapes without omissions."""
    cursors = args.view_cursors or EditViewCursors()
    limit = max(len(neutral.pin_legend), len(neutral.label_only_pins), 1)
    label_only_page = paginate_view(
        list(neutral.label_only_pins),
        "label_only_pins",
        cursor=None if complete else cursors.label_only_pins,
        limit=limit if complete else args.view_limit,
    )

    views: dict[str, Any] = {}
    for view in args.return_views:
        if view in ("pin_legend", "touched"):
            rows = list(neutral.pin_legend)
            if view == "touched":
                wanted = touched_refs(args.ops)
                rows = [row for row in rows if str(row.get("ref", "")).casefold() in wanted]
            views[view] = paginate_view(
                rows,
                view,
                cursor=None if complete else getattr(cursors, view),
                limit=limit if complete else args.view_limit,
            )
        elif view == "render" and neutral.render is not None:
            views["render"] = neutral.render
    return label_only_page, views


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


def _compare_reference(ref_path: Path, netlist_text: str, resolver: IncludeResolver):
    """Parse both sides and compare their connectivity graphs (blocking CPU/IO)."""
    ref_graph = parse_netlist_graph(ref_path, include_resolver=resolver)
    cand_graph = parse_netlist_graph(netlist_text, include_resolver=resolver)
    return compare_graphs(ref_graph, cand_graph)


async def _run_reference_stage(
    committed_text: str,
    encoding: str,
    ref_path: Path,
    build_id: str,
    state: SessionState,
) -> dict:
    """Export a copy of the committed sheet and compare it to ``ref_path``.

    Runs entirely on a COPY under a managed temp dir, so the exporter's own
    asc_export_lock keys on the copy's path (no reentrancy with the guard-held
    target lock). The reference path is already resolved — the handler validates
    it in pre-flight, so no path rejection can land here, after the commit. An
    export failure is captured into the returned dict; anything else escapes to
    the handler, which reports it on a committed envelope. Either way the sheet
    is already committed and stays so.
    """
    verification: dict[str, Any] = {"reference": str(ref_path)}
    export_root = state.working_dir / ".ltspice-mcp" / "edit-exports" / build_id
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
        resolver = make_include_resolver(state)
        comparison = await asyncio.to_thread(_compare_reference, ref_path, netlist_text, resolver)
        verification["equivalent"] = comparison.equivalent
        verification["structurally_equivalent"] = comparison.structurally_equivalent
        verification["comparison"] = comparison.as_dict()
        verification["_netlist"] = netlist_text
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
    surface's separate success/failure lists.
    """
    entries, abort_reason = run_op_batch(editor, ops, target, stop_on_error=not dry_run)
    results = [e for e in entries if e["ok"]]
    failures = [
        {"index": e["index"], "op": e["op"], "error": e["error"]} for e in entries if not e["ok"]
    ]
    return results, failures, abort_reason


def _mirror_commit_state(commit_state: str) -> Literal["not_started", "committed", "unknown"]:
    """Project the top-level commit_state onto the error envelope's enum.

    The top-level field uses ``committed``/``not_committed``; the shared error
    envelope uses ``not_started``/``committed``/``unknown``. A not-yet-written
    sheet mirrors to ``not_started``.
    """
    return "committed" if commit_state == "committed" else "not_started"


def _envelope(
    *,
    outcome: str,
    commit_state: str,
    target: Path,
    build_id: str,
    base: str,
    stages: list[dict],
    sha256: str | None = None,
    error: dict | None = None,
    wiring: dict | None = None,
    views: dict | None = None,
    verification: dict | None = None,
    netlist: str | None = None,
    warnings: list[str] | None = None,
    failures: list[dict] | None = None,
    observations: list[str] | None = None,
    artifacts: list[dict] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "outcome": outcome,
        "commit_state": commit_state,
        "target": str(target),
        "build_id": build_id,
        "base": base,
        "stages": stages,
        "sha256": sha256,
        "warnings": warnings or [],
        "failures": failures or [],
        "observations": observations or [],
        "artifacts": artifacts or [],
    }
    if error is not None:
        # Mirror the top-level commit_state into the error object so a failure
        # envelope is self-describing without cross-referencing the top level.
        error.setdefault("commit_state", _mirror_commit_state(commit_state))
        data["error"] = error
    if wiring is not None:
        data["wiring"] = wiring
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
    format: Literal["json", "text"] | None
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
        mcp_result=format_response(evaluation.text, evaluation.data, evaluation.format),
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
    if not args.ops:
        raise NetlistError("ops list is empty — pass at least one op.")
    _validate_view_cursors(args.view_cursors)
    # Resolve the optional reference here, with the rest of the argument checks:
    # a path outside allowed_paths is an argument fault the caller fixes by
    # resending, not a property of the sheet. Resolving it in the post-commit
    # stage instead would reject the call after the target was already written.
    reference_path = safe_path(args.reference, state) if args.reference is not None else None

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
            if expected is None:
                raise NetlistError(
                    f"{target.name} already exists; pass expected_sha256 (the SHA-256 of "
                    "the file you edited against) so a concurrent edit can't be lost. "
                    "An inspect components or net query on this sheet returns it as "
                    "'sha256'."
                )
            current = sha256_file(target)
            if current != expected:
                _stage("revision_check", False, "sha mismatch")
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            outcome="failed",
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
                        format=args.format,
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
            results, failures, abort_reason = _apply_ops(editor, args.ops, target, args.dry_run)

            # --- op failure → transactional abort (nothing written)
            if abort_reason is not None:
                state.editors.invalidate(target)
                _stage("apply_ops", False, abort_reason)
                return finish(
                    EditSchematicEvaluation(
                        data=_envelope(
                            outcome="failed",
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
                        format=args.format,
                    )
                )
            _stage("apply_ops")

            profile, legend, label_only = _wiring_and_legend(
                editor, include_legend=bool({"pin_legend", "touched"} & set(args.return_views))
            )
            warnings = [w["message"] for w in post_op_warnings(editor)]
            encoding = getattr(editor, "encoding", "utf-8") or "utf-8"
            committed_text = _render_editor_text(editor)

            # --- dry run: validate-only, nothing written, target dir untouched
            if args.dry_run:
                state.editors.invalidate(target)
                neutral_views = await _build_edit_views(
                    args,
                    profile,
                    legend,
                    label_only,
                    committed_text,
                    encoding,
                    target,
                    state,
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
                            outcome="complete",
                            commit_state="not_committed",
                            target=target,
                            build_id=build_id,
                            base=args.base,
                            stages=stages,
                            wiring=wiring,
                            views=presented_views,
                            warnings=warnings,
                            failures=failures + list(neutral_views.failures),
                            hint="Dry run — resubmit without dry_run to commit.",
                        ),
                        text=(
                            f"edit_schematic (dry run) on {target.name}: {len(results)} ops "
                            "validated; nothing saved."
                        ),
                        format=args.format,
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
                    committed_text,
                    encoding,
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
                    committed_text,
                    encoding,
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
            neutral_views = await _build_edit_views(
                args,
                profile,
                legend,
                label_only,
                committed_text,
                encoding,
                target,
                state,
                committed_sha,
            )
            wiring, presented_views = _paged_edit_views(
                args,
                profile,
                neutral_views,
                present_mcp_views=present_mcp_views,
            )
            post_commit_stage = "response"
            artifact_views = {"render": neutral_views.render} if neutral_views.render else {}
            artifacts = _artifacts_from_views(artifact_views)

            verification = None
            netlist = None
            if reference_path is not None:
                post_commit_stage = "reference"
                verification = await _run_reference_stage(
                    committed_text, encoding, reference_path, build_id, state
                )
                netlist = verification.pop("_netlist", None)
                ok = verification.get("export_error") is None and verification.get("equivalent")
                _stage("reference", bool(ok))

            hint = _commit_hint(profile, verification)
            return finish(
                EditSchematicEvaluation(
                    data=_envelope(
                        outcome="complete",
                        commit_state="committed",
                        target=target,
                        build_id=build_id,
                        base=args.base,
                        stages=stages,
                        sha256=committed_sha,
                        wiring=wiring,
                        views=presented_views,
                        verification=verification,
                        netlist=netlist,
                        warnings=warnings,
                        failures=list(neutral_views.failures),
                        artifacts=artifacts,
                        hint=hint,
                    ),
                    text=f"edit_schematic committed {target.name} (build {build_id}).",
                    format=args.format,
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
    description=(
        "Apply a typed op batch to an LTspice .asc schematic in one revision-"
        "guarded, transactional call. base='blank' builds a whole circuit from an "
        "empty sheet; base='existing' applies deltas. Pass expected_sha256 of the "
        "file you edited against (required when the target exists) — a peer that "
        "committed first yields revision_conflict with nothing written. Returns "
        "geometry facts (pin legend, wiring metric), an optional render, and an "
        "optional post-commit netlist verification against a reference."
    ),
    input_model=EditSchematicInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("consolidated",),
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
    ):
        if cursor is None:
            continue
        try:
            decode_page_cursor(cursor, kind)
        except PageCursorError as exc:
            raise NetlistError(f"invalid view_cursors.{kind}: {exc}") from exc


def _render_editor_text(editor) -> str:
    """Render the editor to .asc text via spicelib's StringIO sink."""
    buf = io.StringIO()
    editor.save_netlist(buf)
    return buf.getvalue()


def _wiring_dict(profile: dict[str, int], label_only_page: dict) -> dict:
    return {
        "wire_segments": profile["wire_segments"],
        "pins_total": profile["pins_total"],
        "pins_wired": profile["pins_wired"],
        "pins_label_only": profile["pins_label_only"],
        "label_only_pins": label_only_page,
    }


def _artifacts_from_views(views: dict) -> list[dict]:
    render = views.get("render")
    if render and render.get("path"):
        return [{"kind": "render", "path": render["path"], "format": render.get("image_format")}]
    return []


def _commit_failure_response(
    args: EditSchematicInput,
    target: Path,
    build_id: str,
    committed_text: str,
    encoding: str,
    stages: list[dict],
    error: str,
    *,
    present_mcp_views: bool,
) -> EditSchematicEvaluation:
    """Envelope for a commit failure before the rename; optionally quarantine a draft."""
    draft_path = None
    if args.write_failed_draft:
        draft_path = target.with_name(f"{target.stem}.draft-{build_id}.asc")
        with contextlib.suppress(OSError):
            draft_path.write_text(committed_text, encoding=encoding)
    artifacts = [{"kind": "draft", "path": str(draft_path)}] if draft_path else []
    return _finish_edit_evaluation(
        EditSchematicEvaluation(
            data=_envelope(
                outcome="failed",
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
                artifacts=artifacts,
                hint="The sheet was not modified; retry with the same expected_sha256.",
            ),
            text=(
                f"edit_schematic: commit failed before rename ({error}); target unchanged."
                + (f" Draft written to {draft_path.name}." if draft_path else "")
            ),
            format=args.format,
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
                outcome="partial",
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
            format=args.format,
        ),
        present_mcp_views=present_mcp_views,
    )


def _commit_hint(profile: dict[str, int], verification: dict | None) -> str:
    parts = [
        f"Committed. Of {profile['pins_total']} pins, {profile['pins_wired']} are on wires and "
        f"{profile['pins_label_only']} carry a net-label only."
    ]
    if verification is not None:
        if verification.get("export_error"):
            parts.append(f"Reference check could not export: {verification['export_error']}.")
        elif verification.get("equivalent"):
            parts.append("Reference netlist verified: equivalent.")
        else:
            parts.append("Reference netlist differs — see verification.comparison.")
    return " ".join(parts)
