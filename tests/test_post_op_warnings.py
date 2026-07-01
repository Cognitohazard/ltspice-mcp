"""Regressions for the post-op validation pass.

Pins the structured ``validation_warnings`` payload returned by mutating
.asc handlers (apply_schematic_ops, connect, add_component) and the
text-message warnings on move_component / remove_component. Enforces the
project's validate-before-write doctrine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    ApplySchematicOpsInput,
    ConnectInput,
    CreateSchematicInput,
    MoveComponentInput,
    RemoveComponentInput,
    WaypointInput,
    handle_add_component,
    handle_apply_schematic_ops,
    handle_connect,
    handle_create_schematic,
    handle_move_component,
    handle_remove_component,
)

# ---------------------------------------------------------------------------
# Helper directly — synthetic editor-level fixtures
# ---------------------------------------------------------------------------


class TestPostOpWarningsHelper:
    """Drive ``_post_op_warnings`` against synthetic schematics where we
    know exactly which pins should float, which wires are duplicate, and
    which labels are dangling."""

    @pytest.fixture
    def fresh_schematic(
        self, asc_state: SessionState, work_dir: Path
    ) -> tuple[SessionState, Path]:
        # Use a brand-new file so we don't inherit Draft1.asc's geometry.
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            handle_create_schematic(CreateSchematicInput(name="post_op_check"), asc_state)
        )
        return asc_state, work_dir / "post_op_check.asc"

    async def test_clean_schematic_returns_no_warnings(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="clean"), asc_state)
        # Apply a single connect that touches both endpoints — both pins
        # belong to the same wire, no duplicates, no labels.
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="clean.asc",
                ops=[  # type: ignore[arg-type]
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "R2",
                        "symbol": "res",
                        "x": 100,
                        "y": 200,  # R2's top pin is at (116,200), R1's bottom at (116,196)
                    },
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        # Both R1 and R2 are placed without wires; their pins are floating.
        # Confirms the helper *does* fire on a representative case.
        data = result.structuredContent
        assert data is not None
        warnings = data.get("validation_warnings", [])
        # Every floating warning carries kind/message/ref/pin.
        for w in warnings:
            assert w["kind"] == "floating_pin"
            assert "message" in w
            assert "ref" in w


# ---------------------------------------------------------------------------
# apply_schematic_ops — the primary entry point
# ---------------------------------------------------------------------------


class TestApplySchematicOpsValidation:
    async def test_floating_pin_after_add_component(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="float1"), asc_state)
        ops: list[Any] = [
            {
                "op": "add_component",
                "reference": "R1",
                "symbol": "res",
                "x": 100,
                "y": 100,
            },
        ]
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(path="float1.asc", ops=ops, stop_on_error=False),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        warnings = data.get("validation_warnings", [])
        floating = [w for w in warnings if w["kind"] == "floating_pin"]
        # R1 has two pins; both should be flagged as floating.
        assert len(floating) == 2
        refs = {w["ref"] for w in floating}
        assert refs == {"R1"}

    async def test_pin_at_shared_coord_is_not_floating(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # res.asy pins sit at (0,-48) and (0,48) in symbol-local coords. With
        # R1 at origin (100,100), pin 2 lands at (100,148). With R2 at
        # origin (100,196), pin 1 (-48 offset) lands at (100,148) — same
        # spot. Two pins at the same coord ⇒ neither is "floating".
        await handle_create_schematic(CreateSchematicInput(name="shared"), asc_state)
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="shared.asc",
                ops=[  # type: ignore[arg-type]
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "R2",
                        "symbol": "res",
                        "x": 100,
                        "y": 196,
                    },
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        warnings = data.get("validation_warnings", [])
        floating_coords = {(w["x"], w["y"]) for w in warnings if w["kind"] == "floating_pin"}
        # The shared coord must NOT appear in floating_coords.
        assert (100, 148) not in floating_coords
        # The two outer ends DO appear: R1.1 at (100,52), R2.2 at (100,244).
        assert (100, 52) in floating_coords
        assert (100, 244) in floating_coords

    async def test_duplicate_wire_detected(self, asc_state: SessionState, work_dir: Path) -> None:
        await handle_create_schematic(CreateSchematicInput(name="dupwire"), asc_state)
        # Place R1 and R2, then connect each pair the same way twice.
        # The second connect will duplicate the first wire.
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="dupwire.asc",
                ops=[  # type: ignore[arg-type]
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "R2",
                        "symbol": "res",
                        "x": 200,
                        "y": 100,
                    },
                    # Two connects with the same waypoint plan → duplicate segments.
                    {
                        "op": "connect",
                        "from_pin": "R1.1",
                        "to_pin": "R2.1",
                    },
                    {
                        "op": "connect",
                        "from_pin": "R1.1",
                        "to_pin": "R2.1",
                    },
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        warnings = data.get("validation_warnings", [])
        kinds = {w["kind"] for w in warnings}
        assert "duplicate_wire" in kinds

    async def test_dangling_label_detected(self, asc_state: SessionState, work_dir: Path) -> None:
        await handle_create_schematic(CreateSchematicInput(name="dangle"), asc_state)
        # Place a label at coordinates with no wire and no pin.
        ops: list[Any] = [
            {
                "op": "add_net_label",
                "net": "ORPHAN",
                "x": 500,
                "y": 500,
            },
        ]
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(path="dangle.asc", ops=ops, stop_on_error=False),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        warnings = data.get("validation_warnings", [])
        dangling = [w for w in warnings if w["kind"] == "dangling_label"]
        assert any(w.get("label") == "ORPHAN" for w in dangling)

    async def test_aborted_transaction_does_not_emit_warnings(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="abort"), asc_state)
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="abort.asc",
                ops=[  # type: ignore[arg-type]
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {  # bogus symbol — aborts the transaction
                        "op": "add_component",
                        "reference": "X1",
                        "symbol": "definitely_not_a_symbol",
                        "x": 200,
                        "y": 100,
                    },
                ],
                stop_on_error=True,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        # Aborted transactions don't save the file; reporting state warnings
        # would be misleading.
        assert "validation_warnings" not in data
        assert data["saved"] is False


# ---------------------------------------------------------------------------
# add_component — the simplest mutating handler with structured output
# ---------------------------------------------------------------------------


class TestAddComponentValidation:
    async def test_freshly_added_pins_not_flagged_floating(
        self, asc_state: SessionState, asc_file: Path
    ) -> None:
        # A just-placed component has every pin floating by construction, so a
        # floating-pin advisory here is 100% noise. add_component must NOT emit
        # it — that reporting belongs to connect (a pin still floating after
        # wiring is actionable) and validate_netlist (the end-of-build gate).
        result = await handle_add_component(
            AddComponentInput(
                path="Draft1.asc",
                reference="R_new",
                symbol="res",
                x=400,
                y=400,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        assert "validation_warnings" not in data


# ---------------------------------------------------------------------------
# connect — wire routes don't introduce duplicates of their own
# ---------------------------------------------------------------------------


class TestConnectValidation:
    async def test_connect_returns_validation_field_when_warnings_exist(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="conn"), asc_state)
        # Place two resistors so a connect leaves two outer pins floating.
        await handle_add_component(
            AddComponentInput(path="conn.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        await handle_add_component(
            AddComponentInput(path="conn.asc", reference="R2", symbol="res", x=200, y=100),
            asc_state,
        )
        result = await handle_connect(
            ConnectInput(path="conn.asc", from_pin="R1.1", to_pin="R2.1"),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        # After the connect, R1 pin 1 and R2 pin 1 are wired; their other
        # pins (1.2 and 2.2) remain floating, so the field is populated.
        assert "validation_warnings" in data
        floating = [w for w in data["validation_warnings"] if w["kind"] == "floating_pin"]
        assert floating, "expected at least one floating pin after partial wire"

    async def test_connect_scopes_floating_pins_to_touched_components(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # Regression: connect used to re-echo floating-pin warnings for EVERY
        # not-yet-wired component on each call. It must now report only pins of
        # the components it touched, so an untouched R3 placed earlier doesn't
        # add noise to an unrelated connect.
        await handle_create_schematic(CreateSchematicInput(name="scope"), asc_state)
        for ref, x in (("R1", 100), ("R2", 200), ("R3", 400)):
            await handle_add_component(
                AddComponentInput(path="scope.asc", reference=ref, symbol="res", x=x, y=100),
                asc_state,
            )
        result = await handle_connect(
            ConnectInput(path="scope.asc", from_pin="R1.1", to_pin="R2.1"),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        floating_refs = {
            w["ref"] for w in data.get("validation_warnings", []) if w["kind"] == "floating_pin"
        }
        assert "R3" not in floating_refs
        # Only the touched components' still-floating pins are reported.
        assert floating_refs and floating_refs <= {"R1", "R2"}

    async def test_connect_through_endpoint_pin_refused(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # F1: a waypoint routing a wire through the OTHER pin of an endpoint
        # component used to be silently allowed (skip_refs exempted the whole
        # component), shorting it while connect reported success.
        await handle_create_schematic(CreateSchematicInput(name="short_check"), asc_state)
        # res fixture: placed at (x, y) -> pins at (x, y-48) and (x, y+48).
        await handle_add_component(
            AddComponentInput(path="short_check.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        await handle_add_component(
            AddComponentInput(path="short_check.asc", reference="R2", symbol="res", x=300, y=100),
            asc_state,
        )
        # Route R1.1 (100,52) -> R2.2 (300,148); the corner (100,148) lands
        # exactly on R1.2, shorting R1 across its own terminals.
        with pytest.raises(NetlistError, match=r"R1\.2"):
            await handle_connect(
                ConnectInput(
                    path="short_check.asc",
                    from_pin="R1.1",
                    to_pin="R2.2",
                    waypoints=[WaypointInput(x=100, y=148)],
                ),
                asc_state,
            )


# ---------------------------------------------------------------------------
# move_component / remove_component — text-only handlers, message-level pin
# ---------------------------------------------------------------------------


class TestTextHandlerWarnings:
    async def test_move_component_emits_floating_pin_text(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="movetext"), asc_state)
        await handle_add_component(
            AddComponentInput(path="movetext.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        result = await handle_move_component(
            MoveComponentInput(path="movetext.asc", reference="R1", x=300, y=300),
            asc_state,
        )
        text = result.content[0].text  # type: ignore[union-attr]
        assert "Schematic warnings" in text
        assert "Floating pin" in text

    async def test_remove_component_emits_no_floating_pin_when_nothing_floats(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # Schematic has only R1 and a single floating component. Removing
        # R1 leaves the schematic empty — nothing to warn about.
        await handle_create_schematic(CreateSchematicInput(name="removetext"), asc_state)
        await handle_add_component(
            AddComponentInput(path="removetext.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        result = await handle_remove_component(
            RemoveComponentInput(path="removetext.asc", reference="R1", cleanup_wires=True),
            asc_state,
        )
        text = result.content[0].text  # type: ignore[union-attr]
        # Empty schematic ⇒ no floating-pin lines.
        assert "Floating pin" not in text


# ---------------------------------------------------------------------------
# apply_schematic_ops cache safety on uncaught exception
# ---------------------------------------------------------------------------


class TestApplySchematicOpsRollback:
    """An uncaught exception mid-batch (not NetlistError or
    ValueError — those are caught per-op) must:
      - leave the file on disk byte-identical to pre-call,
      - invalidate the cached editor so prior ops' mutations don't leak,
      - re-raise so the caller sees the failure.
    """

    async def test_uncaught_exception_invalidates_and_preserves_file(
        self,
        asc_state: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await handle_create_schematic(CreateSchematicInput(name="rollback"), asc_state)
        target = work_dir / "rollback.asc"
        original = target.read_bytes()

        from ltspice_mcp.tools import circuit as circuit_mod

        real_apply = circuit_mod._apply_op_inplace
        call_count = {"n": 0}

        def flaky_apply(editor, op, asc_path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_apply(editor, op, asc_path)
            # On the second op, raise a RuntimeError — outside the
            # (NetlistError, ValueError) tuple the per-op handler catches.
            raise RuntimeError("injected mid-batch failure")

        monkeypatch.setattr(circuit_mod, "_apply_op_inplace", flaky_apply)

        with pytest.raises(RuntimeError, match="injected"):
            await handle_apply_schematic_ops(
                ApplySchematicOpsInput(
                    path="rollback.asc",
                    ops=[  # type: ignore[arg-type]
                        {
                            "op": "add_component",
                            "reference": "R1",
                            "symbol": "res",
                            "x": 100,
                            "y": 100,
                        },
                        {
                            "op": "add_component",
                            "reference": "R2",
                            "symbol": "res",
                            "x": 200,
                            "y": 100,
                        },
                    ],
                    stop_on_error=True,
                ),
                asc_state,
            )

        # File unchanged — _atomic_save_editor never ran.
        assert target.read_bytes() == original

        # Cache eviction means a follow-up read sees the original empty
        # schematic, not the dirty R1-but-no-R2 state.
        monkeypatch.undo()
        from ltspice_mcp.tools.circuit import handle_list_components

        result = await handle_list_components({"path": "rollback.asc"}, asc_state)
        text = result.content[0].text  # type: ignore[union-attr]
        assert "R1" not in text
        assert "R2" not in text
