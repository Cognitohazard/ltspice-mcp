"""edit_schematic — transactional, revision-guarded ``.asc`` mutation (AUTHOR).

One input language: a typed op batch applied to a sheet that is either blank
(the whole-plan compile) or an existing file (deltas preserving untouched
content). The whole transaction — revision check, editor load, geometry
resolution, mutation, staged write, and the atomic rename onto the target — runs
inside the shared per-file edit guard, so a parallel session that committed
first turns a stale ``expected_sha256`` into a ``revision_conflict`` with nothing
written.

The op models and their in-place applier are reused verbatim from
``tools/circuit.py`` (the shipped ``apply_schematic_ops`` machinery); the only
narrowing is that the wire op accepts ``wire_pins`` only — the deprecated
``connect`` alias is excluded from this surface. Post-commit, an optional
``reference`` stage exports the committed sheet on a COPY and compares it to a
reference netlist through the connectivity graph engine; a mismatch or an export
failure there is reported but never un-commits the sheet.
"""

# This module deliberately reuses the shipped apply_schematic_ops machinery from
# tools/circuit.py (op models, the in-place applier, the net-partition helpers)
# rather than duplicating it, so it imports those module-private names by design.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from mcp import types
from pydantic import Field
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import _fsync_dir, _fsync_fd, atomic_write_bytes
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.netlist_graph import IncludeResolver, compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.pin_legend import (
    PageCursorError,
    build_pin_legend,
    decode_page_cursor,
    find_label_only_pins,
    paginate_view,
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
from ltspice_mcp.tools.circuit import (
    _build_on_wire_predicate,
    _collect_component_geometry,
    _edit_guard,
    _get_asc_editor,
    _make_editor,
    _OpAddComponent,
    _OpAddDirective,
    _OpAddNetLabel,
    _OpMoveComponent,
    _OpRemoveComponent,
    _OpRemoveDirective,
    _OpRemoveNetLabel,
    _OpRemoveWire,
    _OpSetComponentAttribute,
    _OpSetComponentValue,
    _OpWirePins,
    _post_op_warnings,
    _require_asc,
    _run_op_batch,
    _trace_nets,
    _wiring_profile,
    blank_sheet,
)

# The blank-sheet template — identical to what ``create_schematic`` writes.
_BLANK_TEMPLATE = blank_sheet()

_DEFAULT_VIEW_LIMIT = 100


class _OpWirePinsStrict(_OpWirePins):
    """The wire op, ``wire_pins`` only (ID-15).

    The shipped ``_OpWirePins`` still accepts the deprecated ``connect`` alias;
    this consolidated surface drops it. Because the parent's applier dispatches
    on ``isinstance(op, _OpWirePins)`` and reads ``op.op``, narrowing the literal
    is all that is needed — a ``connect`` payload no longer validates and never
    reaches the applier.
    """

    op: Literal["wire_pins"] = "wire_pins"  # pyright: ignore[reportIncompatibleVariableOverride]


# The op union for this surface: the shipped models, with the wire op narrowed.
ConsolidatedOp = (
    _OpAddComponent
    | _OpSetComponentValue
    | _OpSetComponentAttribute
    | _OpRemoveComponent
    | _OpMoveComponent
    | _OpAddNetLabel
    | _OpRemoveNetLabel
    | _OpRemoveWire
    | _OpWirePinsStrict
    | _OpAddDirective
    | _OpRemoveDirective
)


class _ViewCursors(StrictModel):
    """Resumption cursors for the paginated views, each a page's ``next_cursor``."""

    label_only_pins: str | None = None
    pin_legend: str | None = None


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
            "of the file you edited against; a mismatch means a peer committed first "
            "and the call returns revision_conflict with nothing written."
        ),
    )
    ops: list[ConsolidatedOp] = Field(
        description=(
            "Typed edit ops applied in order against one in-memory editor. The whole "
            "batch commits atomically or not at all: the first op that fails aborts "
            "the transaction and nothing is written."
        )
    )
    reference: str | None = Field(
        default=None,
        description=(
            "Optional netlist (.cir/.net) to verify the committed sheet against: the "
            "sheet is exported on a copy and compared for connectivity equivalence. "
            "Runs AFTER commit — a mismatch is reported but does not un-commit."
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
    return_views: list[Literal["pin_legend", "render"]] = Field(
        default_factory=lambda: ["pin_legend"],
        description=(
            "Which geometry views to return. 'pin_legend' (default) is the per-"
            "component pin/net table; 'render' is an SVG/PNG of the sheet."
        ),
    )
    view_cursors: _ViewCursors | None = Field(
        default=None,
        description="Resume a paginated view by passing back its page's next_cursor.",
    )
    view_limit: int = Field(
        default=_DEFAULT_VIEW_LIMIT,
        description="Page size for the paginated views (pin_legend, label_only_pins).",
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
        _fsync_fd(fd)
    finally:
        os.close(fd)
    return tmp


def _commit_rename(tmp: Path, target: Path) -> None:
    """Atomically move the staged file onto the target — the LAST commit step."""
    os.replace(tmp, target)
    with contextlib.suppress(OSError):
        _fsync_dir(target.parent)


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

    Reuses circuit.py's ``_net_partition``-backed helpers (via ``_trace_nets``
    and ``_build_on_wire_predicate``) rather than re-deriving connectivity, and
    hands the plain result to the pure ``lib/pin_legend`` builders. The wiring
    metric and label-only detection are contract output and always run; the
    per-component legend is built only when ``include_legend`` (its own view was
    requested), returned empty otherwise.
    """
    profile = _wiring_profile(editor)
    geometry = _collect_component_geometry(editor)
    nets = _trace_nets(editor)
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    on_wire = _build_on_wire_predicate(segments)
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
    asc_path: Path, fmt: Literal["png", "svg"], scale: float, artifacts_dir: Path
) -> dict:
    """Render ``asc_path`` to SVG (always) or PNG (when the raster extra is
    present), writing a content-hashed artifact under ``artifacts_dir``."""
    scene = build_scene(asc_path, resolver=symbol_resolver_for(asc_path))
    image, out_path, _ = render_scene_artifact(scene, artifacts_dir, image_format=fmt, scale=scale)
    view = dict(image.to_dict())
    if scene.diagnostics:
        view["diagnostics"] = list(scene.diagnostics)
    view["path"] = str(out_path)
    return view


async def _paginate_views(
    args: EditSchematicInput,
    legend: list[dict],
    label_only: list[dict],
    committed_target: Path | None,
    build_id: str,
    state: SessionState,
) -> tuple[dict, dict, list[dict]]:
    """Assemble the label_only_pins page, the requested views, and per-view failures."""
    cursors = args.view_cursors or _ViewCursors()
    label_only_page = paginate_view(
        label_only, "label_only_pins", cursor=cursors.label_only_pins, limit=args.view_limit
    )

    views: dict[str, Any] = {}
    failures: list[dict] = []
    for view in args.return_views:
        if view == "pin_legend":
            views["pin_legend"] = paginate_view(
                legend, "pin_legend", cursor=cursors.pin_legend, limit=args.view_limit
            )
        elif view == "render":
            # A render needs a committed file on disk. In dry_run nothing is
            # committed, so report metadata-only rather than writing an artifact
            # (the dry_run contract leaves the target dir untouched).
            if committed_target is None:
                views["render"] = {
                    "status": "dry_run",
                    "note": (
                        "render artifact is written only on commit; run without "
                        "dry_run to persist it."
                    ),
                }
                continue
            try:
                artifacts_dir = state.working_dir / ".ltspice-mcp" / "renders"
                # build_scene + rasterize are pure file/CPU work on already-
                # committed bytes — offload so they don't stall the event loop.
                views["render"] = await asyncio.to_thread(
                    _render_view,
                    committed_target,
                    args.render_format,
                    args.render_scale,
                    artifacts_dir,
                )
            except Exception as exc:  # broad by design — a render failure is per-view, never fatal
                failures.append({"stage": "render", "error": str(exc)})
    return label_only_page, views, failures


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
    reference: str,
    build_id: str,
    state: SessionState,
) -> dict:
    """Export a copy of the committed sheet and compare it to ``reference``.

    Runs entirely on a COPY under a managed temp dir, so the exporter's own
    asc_export_lock keys on the copy's path (no reentrancy with the guard-held
    target lock). Any failure is captured into the returned dict — the sheet is
    already committed and stays so.
    """
    ref_path = safe_path(reference, state)
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
        return _get_asc_editor(target, state)
    # Construct an AscEditor from a throwaway blank template; it parses on init,
    # so the temp file can go away immediately and all mutation is in-memory.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltspice-blank-"))
    try:
        tmp_asc = tmp_dir / "blank.asc"
        tmp_asc.write_text(_BLANK_TEMPLATE, encoding="utf-8")
        # The template is .asc, so _make_editor always yields an AscEditor here.
        return cast(AscEditor, _make_editor(tmp_asc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _apply_ops(
    editor, ops, target: Path, dry_run: bool
) -> tuple[list[dict], list[dict], str | None]:
    """Apply every op in order. Returns (results, failures, abort_reason).

    Delegates the loop to the shared ``_run_op_batch`` runner (abort-on-first-
    failure unless ``dry_run``), then splits its unified entries into this
    surface's separate success/failure lists.
    """
    entries, abort_reason = _run_op_batch(editor, ops, target, stop_on_error=not dry_run)
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
    target = safe_path(args.target, state)
    _require_asc(target)
    if not args.ops:
        raise NetlistError("ops list is empty — pass at least one op.")
    _validate_view_cursors(args.view_cursors)

    build_id = generate_id("build")
    stages: list[dict] = []

    def _stage(name: str, ok: bool = True, error: str | None = None) -> None:
        """Append one commit-protocol stage entry (the ok path omits ``error``)."""
        entry: dict[str, Any] = {"stage": name, "ok": ok}
        if error is not None:
            entry["error"] = error
        stages.append(entry)

    async with _edit_guard(target):
        # --- revision guard (inside the guard so a peer's committed write is seen)
        exists = target.exists()
        expected = args.expected_sha256.lower() if args.expected_sha256 else None
        if exists:
            if expected is None:
                raise NetlistError(
                    f"{target.name} already exists; pass expected_sha256 (the SHA-256 of "
                    "the file you edited against) so a concurrent edit can't be lost."
                )
            current = sha256_file(target)
            if current != expected:
                _stage("revision_check", False, "sha mismatch")
                return format_response(
                    f"edit_schematic: revision_conflict on {target.name} — the file changed "
                    "since you read it. Re-read it and resubmit with the current sha256.",
                    _envelope(
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
                    args.format,
                )
        _stage("revision_check")

        use_template = args.base == "blank" or not exists
        editor = _build_editor(target, use_template, state)
        try:
            results, failures, abort_reason = _apply_ops(editor, args.ops, target, args.dry_run)

            # --- op failure → transactional abort (nothing written)
            if abort_reason is not None:
                state.editors.invalidate(target)
                _stage("apply_ops", False, abort_reason)
                return format_response(
                    f"edit_schematic: transaction aborted — {abort_reason}. No changes saved.",
                    _envelope(
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
                    args.format,
                )
            _stage("apply_ops")

            profile, legend, label_only = _wiring_and_legend(
                editor, include_legend="pin_legend" in args.return_views
            )
            warnings = [w["message"] for w in _post_op_warnings(editor)]
            encoding = getattr(editor, "encoding", "utf-8") or "utf-8"
            committed_text = _render_editor_text(editor)

            # --- dry run: validate-only, nothing written, target dir untouched
            if args.dry_run:
                state.editors.invalidate(target)
                label_only_page, views, view_failures = await _paginate_views(
                    args, legend, label_only, None, build_id, state
                )
                wiring = _wiring_dict(profile, label_only_page)
                return format_response(
                    f"edit_schematic (dry run) on {target.name}: {len(results)} ops validated; "
                    "nothing saved.",
                    _envelope(
                        outcome="complete",
                        commit_state="not_committed",
                        target=target,
                        build_id=build_id,
                        base=args.base,
                        stages=stages,
                        wiring=wiring,
                        views=views,
                        warnings=warnings,
                        failures=failures + view_failures,
                        hint="Dry run — resubmit without dry_run to commit.",
                    ),
                    args.format,
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
                    args, target, build_id, committed_text, encoding, stages, outcome.error or ""
                )
            _stage("stage_asc")
            if not outcome.renamed:
                state.editors.invalidate(target)
                _stage("rename", False, outcome.error)
                return _commit_failure_response(
                    args, target, build_id, committed_text, encoding, stages, outcome.error or ""
                )
            _stage("rename")
            state.editors.invalidate(target)
            # The bytes we just staged and renamed ARE the file — hash them in
            # memory instead of re-reading the target back off disk.
            committed_sha = hashlib.sha256(committed_text.encode(encoding)).hexdigest()

            # --- post-commit: everything below keeps commit_state='committed'
            label_only_page, views, view_failures = await _paginate_views(
                args, legend, label_only, target, build_id, state
            )
            wiring = _wiring_dict(profile, label_only_page)
            artifacts = _artifacts_from_views(views)

            verification = None
            netlist = None
            if args.reference is not None:
                verification = await _run_reference_stage(
                    committed_text, encoding, args.reference, build_id, state
                )
                netlist = verification.pop("_netlist", None)
                ok = verification.get("export_error") is None and verification.get("equivalent")
                _stage("reference", bool(ok))

            hint = _commit_hint(profile, verification)
            return format_response(
                f"edit_schematic committed {target.name} (build {build_id}).",
                _envelope(
                    outcome="complete",
                    commit_state="committed",
                    target=target,
                    build_id=build_id,
                    base=args.base,
                    stages=stages,
                    sha256=committed_sha,
                    wiring=wiring,
                    views=views,
                    verification=verification,
                    netlist=netlist,
                    warnings=warnings,
                    failures=view_failures,
                    artifacts=artifacts,
                    hint=hint,
                ),
                args.format,
            )
        except BaseException:
            # Any escape leaves the cached editor dirty — evict so the next read
            # re-parses from disk (the on-disk file is intact or already renamed).
            state.editors.invalidate(target)
            raise


def _validate_view_cursors(cursors: _ViewCursors | None) -> None:
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
) -> types.CallToolResult:
    """Envelope for a commit failure before the rename; optionally quarantine a draft."""
    draft_path = None
    if args.write_failed_draft:
        draft_path = target.with_name(f"{target.stem}.draft-{build_id}.asc")
        with contextlib.suppress(OSError):
            draft_path.write_text(committed_text, encoding=encoding)
    artifacts = [{"kind": "draft", "path": str(draft_path)}] if draft_path else []
    return format_response(
        f"edit_schematic: commit failed before rename ({error}); target unchanged."
        + (f" Draft written to {draft_path.name}." if draft_path else ""),
        _envelope(
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
        args.format,
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
