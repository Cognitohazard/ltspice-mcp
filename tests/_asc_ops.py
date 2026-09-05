"""Helpers for driving .asc op batches through the live ``edit_schematic`` tool.

The op appliers, the route planner, the post-op validation pass and the symbol
geometry layer are all reached the same way in production: one ``edit_schematic``
call carrying a batch of typed ops. These helpers wrap that call so a test can
say what it means (build this sheet, apply these ops, read that pin's geometry)
without repeating the revision-guard and view plumbing every time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic


def sha_of(path: Path) -> str:
    """Current sha256 of a file, for the revision guard."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _envelope(result: Any) -> dict[str, Any]:
    data = result.structuredContent
    assert data is not None, "edit_schematic must always return structuredContent"
    return dict(data)


def blank_sheet_file(state: SessionState, name: str) -> Path:
    """Write an empty ``{name}.asc`` — setup, not the behaviour under test.

    ``edit_schematic`` requires at least one op, so a test that needs a bare
    sheet to mutate afterwards writes the same blank template the tool's
    base='blank' path starts from.
    """
    from ltspice_mcp.tools.schematic_edit import _BLANK_TEMPLATE

    path = Path(state.working_dir) / f"{name}.asc"
    path.write_text(_BLANK_TEMPLATE, encoding="utf-8")
    return path


async def build_sheet(
    state: SessionState,
    name: str,
    ops: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Create ``{name}.asc`` from a blank sheet and apply ``ops`` in one call.

    With no ops the sheet is written directly (the tool refuses an empty batch).
    """
    if not ops:
        blank_sheet_file(state, name)
        return {}
    return _envelope(
        await handle_edit_schematic(
            EditSchematicInput.model_validate(
                {"target": f"{name}.asc", "base": "blank", "ops": ops, **kw}
            ),
            state,
        )
    )


async def apply_ops(
    state: SessionState,
    path: str | Path,
    ops: list[dict[str, Any]],
    **kw: Any,
) -> dict[str, Any]:
    """Apply ``ops`` to an existing sheet, satisfying the revision guard."""
    name = Path(path).name
    target = Path(state.working_dir) / name
    payload: dict[str, Any] = {"target": name, "ops": ops, **kw}
    payload.setdefault("expected_sha256", sha_of(target))
    return _envelope(
        await handle_edit_schematic(EditSchematicInput.model_validate(payload), state)
    )


def pins_of(data: dict[str, Any], ref: str, view: str = "touched") -> dict[str, tuple[int, int]]:
    """``{pin_name: (x, y)}`` for ``ref`` from a returned geometry view."""
    for row in data["views"][view]["items"]:
        if row["ref"] == ref:
            return {p["name"]: (p["x"], p["y"]) for p in row["pins"]}
    raise AssertionError(f"{ref} not in views.{view}: {data['views'][view]['items']}")


def nets_of(data: dict[str, Any], ref: str, view: str = "touched") -> dict[str, str | None]:
    """``{pin_name: net}`` for ``ref`` from a returned geometry view."""
    for row in data["views"][view]["items"]:
        if row["ref"] == ref:
            return {p["name"]: p["net"] for p in row["pins"]}
    raise AssertionError(f"{ref} not in views.{view}: {data['views'][view]['items']}")


def failure_messages(data: dict[str, Any]) -> list[str]:
    """Every per-op failure message in the batch, in op order."""
    return [f["error"] for f in data["failures"]]


def wire_segments(path: Path) -> list[tuple[int, int, int, int]]:
    """Every WIRE record in an .asc as ``(x1, y1, x2, y2)``."""
    out: list[tuple[int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "WIRE" and len(parts) >= 5:
            out.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
    return out


def has_segment(
    segments: list[tuple[int, int, int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """Whether ``segments`` contains the run between ``a`` and ``b``, either way round."""
    return (a[0], a[1], b[0], b[1]) in segments or (b[0], b[1], a[0], a[1]) in segments


# --- direct seams -----------------------------------------------------------
#
# The envelope folds per-op geometry into paginated views and flattens the
# validation pass to message strings. A test about the op runner or the
# validation pass itself therefore drives those seams directly — the same
# objects ``edit_schematic`` builds, not a stand-in for them.


def load_editor(state: SessionState, path: Path):
    """The cached AscEditor for ``path`` — the one an op batch mutates."""
    from ltspice_mcp.lib.schematic_ops import get_asc_editor

    return get_asc_editor(path, state)


def run_ops(
    state: SessionState,
    path: Path,
    ops: list[dict[str, Any]],
    *,
    stop_on_error: bool = False,
    save: bool = True,
) -> tuple[list[dict[str, Any]], str | None, Any]:
    """Apply ``ops`` through the shared op runner; return (entries, abort, editor).

    Entries are the runner's unified ``{index, op, ok, error, **op_result}``
    records, which carry the per-op geometry the response envelope pages away.
    """
    from pydantic import TypeAdapter

    from ltspice_mcp.lib.schematic_ops import run_op_batch
    from ltspice_mcp.tools.schematic_edit import ConsolidatedOp

    editor = load_editor(state, path)
    typed = TypeAdapter(list[ConsolidatedOp]).validate_python(ops)
    entries, abort = run_op_batch(editor, typed, path, stop_on_error=stop_on_error)
    if save and abort is None:
        editor.save_netlist(path)
    return entries, abort, editor


def structured_warnings(editor: Any, **kw: Any) -> list[dict[str, Any]]:
    """The post-op validation pass's structured findings for ``editor``."""
    from ltspice_mcp.lib.schematic_ops import post_op_warnings

    return post_op_warnings(editor, **kw)


def batch_view(
    state: SessionState,
    path: Path,
    ops: list[dict[str, Any]],
    *,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    """One live op batch, presented as a flat dict of its own outputs.

    Nothing here is recomputed: ``results`` are the runner's own entries,
    ``saved`` is whether it aborted, and ``validation_warnings`` is the
    validation pass's output for the editor it mutated. An aborted batch saves
    nothing, so it carries no warnings about a state that was never written.
    """
    entries, abort, editor = run_ops(state, path, ops, stop_on_error=stop_on_error)
    view: dict[str, Any] = {
        "results": entries,
        "saved": abort is None,
        "abort_reason": abort,
        "applied_count": sum(1 for e in entries if e["ok"]),
        "failed_count": sum(1 for e in entries if not e["ok"]),
        "editor": editor,
    }
    if abort is None:
        warnings = structured_warnings(editor)
        if warnings:
            view["validation_warnings"] = warnings
    return view


def apply_one(
    state: SessionState,
    path: Path,
    op: dict[str, Any],
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Apply a single op and return its facts, re-raising a per-op refusal.

    The batch runner records a ``NetlistError``/``ValueError`` against the op
    rather than propagating it. A test about one op's contract wants the refusal
    itself, so this re-raises the error the runner recorded — the same message
    the caller reads off the response's ``failures`` entry.
    """
    from ltspice_mcp.errors import NetlistError

    entries, _, _ = run_ops(state, path, [op], stop_on_error=True, save=save)
    entry = entries[0]
    if not entry["ok"]:
        raise NetlistError(str(entry["error"]))
    return entry


def add_component(
    state: SessionState,
    path: Path,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    **kw: Any,
) -> dict[str, Any]:
    """Place one component through the live op runner; returns the op's facts."""
    return apply_one(
        state,
        path,
        {"op": "add_component", "reference": reference, "symbol": symbol, "x": x, "y": y, **kw},
    )


def wire_pins(
    state: SessionState,
    path: Path,
    from_pin: str,
    to_pin: str,
    waypoints: list[dict[str, int]] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Wire two pins through the live op runner; returns the op's facts."""
    op: dict[str, Any] = {"op": "wire_pins", "from_pin": from_pin, "to_pin": to_pin, **kw}
    if waypoints is not None:
        op["waypoints"] = waypoints
    return apply_one(state, path, op)


def add_net_label(state: SessionState, path: Path, net: str, **kw: Any) -> dict[str, Any]:
    """Place one net label through the live op runner; returns the op's facts."""
    return apply_one(state, path, {"op": "add_net_label", "net": net, **kw})


async def inspect_one(state: SessionState, query: dict[str, Any]) -> dict[str, Any]:
    """Run one ``inspect`` query and return its data, asserting it succeeded."""
    from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect

    result = await handle_inspect(InspectInput.model_validate({"queries": [query]}), state)
    data = result.structuredContent
    assert data is not None
    item = data["results"][0]
    assert item["ok"], item
    return item["data"]


async def components_of(state: SessionState, path: Path, **kw: Any) -> dict[str, Any]:
    """Full component detail for a sheet, read through ``inspect``."""
    return await inspect_one(
        state, {"kind": "components", "path": str(path), "detail": "full", **kw}
    )
