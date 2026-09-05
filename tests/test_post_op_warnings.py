"""Regressions for the post-op validation pass.

Pins the structured findings the pass returns — floating pins, duplicate wires,
dangling labels, labels buried in a component's box — and the wiring profile the
``edit_schematic`` envelope reports alongside them. Enforces the project's
validate-before-write doctrine.

The envelope flattens the pass to message strings, so the tests that care about a
finding's ``kind``/``ref``/coordinates call the pass itself against the editor an
op batch just mutated. That is the same object ``edit_schematic`` hands it — the
runner and the pass are driven directly, never stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.state import SessionState
from tests._asc_ops import apply_ops, build_sheet, run_ops, structured_warnings

pytestmark = pytest.mark.asyncio


def _floating(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w["kind"] == "floating_pin"]


def _kinds(warnings: list[dict]) -> set[str]:
    return {w["kind"] for w in warnings}


# ---------------------------------------------------------------------------
# floating_pin
# ---------------------------------------------------------------------------


class TestFloatingPinWarnings:
    async def test_every_floating_warning_is_fully_addressed(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # Two unwired resistors: every pin floats. Each warning must carry the
        # fields a caller needs to act without a follow-up inspection turn.
        await build_sheet(asc_state, "clean")
        _, _, editor = run_ops(
            asc_state,
            work_dir / "clean.asc",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 300, "y": 100},
            ],
        )
        warnings = _floating(structured_warnings(editor))
        assert warnings, "unwired components must produce floating-pin findings"
        for w in warnings:
            assert {"kind", "message", "ref", "pin", "x", "y"} <= set(w)

    async def test_floating_pin_after_add_component(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await build_sheet(asc_state, "float1")
        _, _, editor = run_ops(
            asc_state,
            work_dir / "float1.asc",
            [{"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100}],
        )
        floating = _floating(structured_warnings(editor))
        # R1 has two pins; both should be flagged as floating.
        assert len(floating) == 2
        assert {w["ref"] for w in floating} == {"R1"}

    async def test_pin_at_shared_coord_is_not_floating(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # res.asy pins sit at (0,-48) and (0,48) in symbol-local coords. With
        # R1 at origin (100,100), pin 2 lands at (100,148). With R2 at
        # origin (100,196), pin 1 (-48 offset) lands at (100,148) — same
        # spot. Two pins at the same coord ⇒ neither is "floating".
        await build_sheet(asc_state, "shared")
        _, _, editor = run_ops(
            asc_state,
            work_dir / "shared.asc",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 100, "y": 196},
            ],
        )
        coords = {(w["x"], w["y"]) for w in _floating(structured_warnings(editor))}
        # The shared coord must NOT appear.
        assert (100, 148) not in coords
        # The two outer ends DO appear: R1.1 at (100,52), R2.2 at (100,244).
        assert (100, 52) in coords
        assert (100, 244) in coords


# ---------------------------------------------------------------------------
# duplicate_wire
# ---------------------------------------------------------------------------


class TestDuplicateWire:
    async def test_repeating_a_wire_op_draws_nothing_and_says_so(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        """Wiring the same pair twice must leave one segment, not two.

        A duplicate connects nothing and cannot be told from a real second
        wire, so "remove the duplicate" and "remove the connection" become the
        same request — which is how one measured session disconnected a cap
        while tidying up. Not creating it is what makes that impossible."""
        await build_sheet(asc_state, "dupwire")
        entries, _, editor = run_ops(
            asc_state,
            work_dir / "dupwire.asc",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 200, "y": 100},
                # Two wire_pins ops with the same plan → would be duplicate segments.
                {"op": "wire_pins", "from_pin": "R1.1", "to_pin": "R2.1"},
                {"op": "wire_pins", "from_pin": "R1.1", "to_pin": "R2.1"},
            ],
        )
        assert "duplicate_wire" not in _kinds(structured_warnings(editor))

        first, second = entries[2], entries[3]
        assert first["wire_count"] >= 1
        assert "already_present" not in first
        assert second["wire_count"] == 0
        assert second["already_present"], "the repeat must say the segments were already there"

        sheet = (work_dir / "dupwire.asc").read_text().splitlines()
        wires = [line for line in sheet if line.startswith("WIRE ")]
        assert len(wires) == len(set(wires)), f"a duplicate segment reached the sheet: {wires}"

    async def test_duplicate_wire_still_detected_on_a_sheet_that_has_one(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        """Our own tools no longer make one; a hand-written sheet still can, so
        the detector has to keep reporting it."""
        await build_sheet(asc_state, "handdup")
        path = work_dir / "handdup.asc"
        path.write_text(path.read_text() + "WIRE 100 100 200 100\nWIRE 200 100 100 100\n")

        _, _, editor = run_ops(asc_state, path, [{"op": "add_directive", "instruction": ".op"}])
        assert "duplicate_wire" in _kinds(structured_warnings(editor))


# ---------------------------------------------------------------------------
# dangling_label
# ---------------------------------------------------------------------------


class TestDanglingLabel:
    async def test_dangling_label_detected(self, asc_state: SessionState, work_dir: Path) -> None:
        await build_sheet(asc_state, "dangle")
        # Place a label at coordinates with no wire and no pin.
        _, _, editor = run_ops(
            asc_state,
            work_dir / "dangle.asc",
            [{"op": "add_net_label", "net": "ORPHAN", "x": 500, "y": 500}],
        )
        dangling = [w for w in structured_warnings(editor) if w["kind"] == "dangling_label"]
        assert any(w.get("label") == "ORPHAN" for w in dangling)


# ---------------------------------------------------------------------------
# Aborted batches write nothing and claim nothing
# ---------------------------------------------------------------------------


class TestAbortedTransaction:
    async def test_aborted_transaction_writes_nothing_and_emits_no_warnings(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await build_sheet(asc_state, "abort")
        before = (work_dir / "abort.asc").read_bytes()
        data = await apply_ops(
            asc_state,
            "abort.asc",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {  # bogus symbol — aborts the transaction
                    "op": "add_component",
                    "reference": "X1",
                    "symbol": "definitely_not_a_symbol",
                    "x": 200,
                    "y": 100,
                },
            ],
        )
        # An aborted batch doesn't save the file; reporting schematic-state
        # warnings for a state that was never written would be misleading.
        assert data["outcome"] != "complete"
        assert data["commit_state"] != "committed"
        assert data["warnings"] == []
        assert (work_dir / "abort.asc").read_bytes() == before

    async def test_uncaught_exception_invalidates_the_cache_and_preserves_the_file(
        self,
        asc_state: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An uncaught exception mid-batch (not NetlistError or ValueError —
        those are caught per-op) must leave the file byte-identical, invalidate
        the cached editor so an earlier op's mutation doesn't leak into a later
        read, and re-raise so the caller sees the failure."""
        await build_sheet(asc_state, "rollback")
        target = work_dir / "rollback.asc"
        original = target.read_bytes()

        from ltspice_mcp.tools import circuit as circuit_mod

        real_apply = circuit_mod._apply_op_inplace
        calls = {"n": 0}

        def flaky_apply(editor, op, asc_path):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_apply(editor, op, asc_path)
            raise RuntimeError("injected mid-batch failure")

        monkeypatch.setattr(circuit_mod, "_apply_op_inplace", flaky_apply)

        with pytest.raises(RuntimeError, match="injected"):
            await apply_ops(
                asc_state,
                "rollback.asc",
                [
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
            )

        assert target.read_bytes() == original

        # Cache eviction means a follow-up read sees the original empty
        # schematic, not the dirty R1-but-no-R2 state.
        monkeypatch.undo()
        from tests._asc_ops import load_editor

        refs = {c.reference for c in load_editor(asc_state, target).components.values()}
        assert "R1" not in refs


# ---------------------------------------------------------------------------
# wire_pins refusals
# ---------------------------------------------------------------------------


class TestWirePinsRefusals:
    async def test_wire_pins_through_endpoint_pin_refused(self, asc_state: SessionState) -> None:
        # A waypoint routing a wire through the OTHER pin of an endpoint
        # component used to be silently allowed (the whole component was
        # exempted), shorting it while wire_pins reported success.
        await build_sheet(
            asc_state,
            "short_check",
            [
                # res fixture: placed at (x, y) -> pins at (x, y-48) and (x, y+48).
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 300, "y": 100},
            ],
        )
        # Route R1.1 (100,52) -> R2.2 (300,148); the corner (100,148) lands
        # exactly on R1.2, shorting R1 across its own terminals.
        data = await apply_ops(
            asc_state,
            "short_check.asc",
            [
                {
                    "op": "wire_pins",
                    "from_pin": "R1.1",
                    "to_pin": "R2.2",
                    "waypoints": [{"x": 100, "y": 148}],
                }
            ],
        )
        assert data["outcome"] != "complete"
        assert any("R1.2" in f["error"] for f in data["failures"]), data["failures"]


# ---------------------------------------------------------------------------
# Readability: label_over_component and the wiring profile
# ---------------------------------------------------------------------------


class TestReadabilityWarnings:
    """label_over_component and the wiring profile — the readability surface
    that distinguishes a routed schematic from net-label soup (every pin
    tagged, no wires drawn)."""

    async def test_label_over_component_detected(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        await build_sheet(asc_state, "labover")
        path = work_dir / "labover.asc"
        entries, _, _ = run_ops(
            asc_state,
            path,
            [{"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100}],
        )
        bb = entries[0]["bounding_box"]
        # bbox centre is strict-interior and (for res) not a pin coordinate.
        cx = bb["x"] + bb["width"] // 2
        cy = bb["y"] + bb["height"] // 2
        _, _, editor = run_ops(
            asc_state, path, [{"op": "add_net_label", "net": "BURIED", "x": cx, "y": cy}]
        )
        warnings = structured_warnings(editor)
        over = [w for w in warnings if w["kind"] == "label_over_component"]
        assert any(w.get("ref") == "R1" for w in over), warnings

    async def test_label_on_foreign_pin_inside_overlapping_bbox_not_flagged(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # R2's pin lands strictly inside R1's (overlapping) bounding box. A label
        # there is on a real pin — the flag pattern — and must NOT be reported
        # against R1, even though it is not one of R1's own pins. Guards the
        # global pin-exclusion (a per-component check would false-positive here).
        await build_sheet(asc_state, "foreignpin")
        path = work_dir / "foreignpin.asc"
        entries, _, _ = run_ops(
            asc_state,
            path,
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 100, "y": 120},
            ],
        )
        r1_bb = entries[0]["bounding_box"]
        inside = [
            p
            for p in entries[1]["pins"]
            if r1_bb["x"] < p["x"] < r1_bb["x"] + r1_bb["width"]
            and r1_bb["y"] < p["y"] < r1_bb["y"] + r1_bb["height"]
        ]
        assert inside, (r1_bb, entries[1]["pins"])  # precondition for the test
        target = inside[0]
        _, _, editor = run_ops(
            asc_state,
            path,
            [{"op": "add_net_label", "net": "0", "x": target["x"], "y": target["y"]}],
        )
        warnings = structured_warnings(editor)
        assert not [
            w
            for w in warnings
            if w["kind"] == "label_over_component"
            and (w["x"], w["y"]) == (target["x"], target["y"])
        ], warnings

    async def test_label_on_bbox_boundary_not_flagged(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # A label exactly on the bbox boundary (and not a pin) is not "inside" —
        # guards the strict `<` interior test against a `<=` regression.
        await build_sheet(asc_state, "boundary")
        path = work_dir / "boundary.asc"
        entries, _, _ = run_ops(
            asc_state,
            path,
            [{"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100}],
        )
        bb = entries[0]["bounding_box"]
        # Left edge, vertical midpoint: on the boundary, not a pin (res pins are
        # at the top/bottom mid-x).
        _, _, editor = run_ops(
            asc_state,
            path,
            [
                {
                    "op": "add_net_label",
                    "net": "EDGE",
                    "x": bb["x"],
                    "y": bb["y"] + bb["height"] // 2,
                }
            ],
        )
        warnings = structured_warnings(editor)
        assert not [w for w in warnings if w["kind"] == "label_over_component"], warnings

    async def test_label_inside_two_overlapping_boxes_reports_both(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        # A non-pin label strictly inside two overlapping component boxes must
        # surface BOTH components — no early break may hide the second.
        await build_sheet(asc_state, "twobox")
        path = work_dir / "twobox.asc"
        entries, _, _ = run_ops(
            asc_state,
            path,
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 108, "y": 100},
            ],
        )
        r1, r2 = entries[0]["bounding_box"], entries[1]["bounding_box"]
        # Centre of the overlap region: strictly inside both boxes, not a pin.
        ox1, ox2 = max(r1["x"], r2["x"]), min(r1["x"] + r1["width"], r2["x"] + r2["width"])
        oy1, oy2 = max(r1["y"], r2["y"]), min(r1["y"] + r1["height"], r2["y"] + r2["height"])
        lx, ly = (ox1 + ox2) // 2, (oy1 + oy2) // 2
        assert ox1 < lx < ox2 and oy1 < ly < oy2  # precondition for the test
        _, _, editor = run_ops(
            asc_state, path, [{"op": "add_net_label", "net": "MID", "x": lx, "y": ly}]
        )
        over_refs = {
            w["ref"] for w in structured_warnings(editor) if w["kind"] == "label_over_component"
        }
        assert over_refs == {"R1", "R2"}

    async def test_wiring_profile_flags_net_label_soup(self, asc_state: SessionState) -> None:
        # One resistor with both pins tagged by a net-label and no wires: the
        # soup signature — pins_label_only > 0, wire_segments == 0.
        data = await build_sheet(
            asc_state,
            "soup",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                # res at (100,100) has pins at (100,52) and (100,148).
                {"op": "add_net_label", "net": "IN", "x": 100, "y": 52},
                {"op": "add_net_label", "net": "OUT", "x": 100, "y": 148},
            ],
        )
        wiring = data["wiring"]
        assert wiring["wire_segments"] == 0
        assert wiring["pins_total"] == 2
        assert wiring["pins_wired"] == 0
        assert wiring["pins_label_only"] == 2

    async def test_wiring_profile_counts_wired_pins(self, asc_state: SessionState) -> None:
        # Two resistors joined by a drawn wire → those pins count as wired.
        data = await build_sheet(
            asc_state,
            "wired",
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 100},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 200, "y": 100},
                {"op": "wire_pins", "from_pin": "R1.1", "to_pin": "R2.1"},
            ],
        )
        wiring = data["wiring"]
        assert wiring["wire_segments"] >= 1
        assert wiring["pins_total"] == 4  # two resistors, two pins each
        # Only the two connected pins sit on the wire; the other two float.
        assert wiring["pins_wired"] == 2
        assert wiring["pins_label_only"] == 0
