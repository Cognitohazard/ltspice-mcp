"""Integration tests for .asc schematic editing tools using fixture symbols."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    ApplySchematicOpsInput,
    CircuitReadInput,
    ComponentInfoInput,
    ConnectInput,
    DiffCircuitInput,
    EditDirectiveInput,
    MoveComponentInput,
    NetLabelInput,
    RemoveComponentInput,
    ResetSchematicInput,
    SchematicFromNetlistInput,
    SetComponentAttributeInput,
    SymbolInfoInput,
    TraceNetInput,
    ValidateNetlistInput,
    WaypointInput,
    _build_on_wire_predicate,
    _parse_netlist_for_synth,
    _point_on_segment,
    handle_add_component,
    handle_add_net_label,
    handle_apply_schematic_ops,
    handle_component_info,
    handle_connect,
    handle_diff_circuit,
    handle_edit_directive,
    handle_list_components,
    handle_move_component,
    handle_read_circuit,
    handle_remove_component,
    handle_reset_schematic,
    handle_schematic_from_netlist,
    handle_set_component_attribute,
    handle_symbol_info,
    handle_trace_net,
    handle_validate_netlist,
)


def _copy_file(src: Path, dst: Path) -> None:
    """Sync byte-copy — keeps blocking pathlib I/O out of async test bodies."""
    dst.write_bytes(src.read_bytes())


# Relocated regression coverage from a retired test module.
RC_NETLIST = (
    "* RC low-pass filter\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.ac dec 10 1 100k\n.end\n"
)


# Relocated regression coverage from a retired test module.
def _read_bytes(p: Path) -> bytes:
    """Sync file read (keeps blocking pathlib I/O out of async test bodies)."""
    return p.read_bytes()


# Relocated regression coverage from a retired test module.
# Two FLAGs (aaa, bbb) on one physical wire -> named-net short; R1 placed away
# from any wire/label -> both pins float.
SHORTED_ASC = """Version 4
SHEET 1 880 680
WIRE 0 0 100 0
FLAG 0 0 aaa
FLAG 100 0 bbb
SYMBOL res 200 200 R0
SYMATTR InstName R1
SYMATTR Value 1k
"""

# Relocated regression coverage from a retired test module.
# R1 (pins at y=100-48 and y=100+48) fully wired to a named net and ground.
CLEAN_ASC = """Version 4
SHEET 1 880 680
WIRE 100 52 100 0
WIRE 100 148 100 200
FLAG 100 0 vin
FLAG 100 200 0
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 1k
"""

# Relocated regression coverage from a retired test module.
# A net carrying a single name plus ground ('0') is NOT a short.
GROUND_ASC = """Version 4
SHEET 1 880 680
WIRE 0 0 100 0
FLAG 0 0 vout
FLAG 100 0 0
"""


@pytest.mark.asyncio
class TestReadAscCircuit:
    async def test_reads_components(self, asc_state: SessionState, asc_file: Path):
        result = await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)
        text = result.content[0].text
        assert "C1" in text
        assert "R1" in text
        assert "V1" in text
        # Net labels
        assert "filtered" in text
        assert result.structuredContent["type"] == "asc"

    async def test_list_components_asc(self, asc_state: SessionState, asc_file: Path):
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        text = result.content[0].text
        assert "C1" in text
        assert "R1" in text


@pytest.mark.asyncio
class TestGetSymbolInfo:
    async def test_valid_symbol(self, asc_state: SessionState):
        result = await handle_symbol_info(
            SymbolInfoInput(symbol="res", x=0, y=0, rotation="R0"), asc_state
        )
        text = result.content[0].text
        assert "res" in text
        assert "Pins" in text

    async def test_unknown_symbol(self, asc_state: SessionState):
        with pytest.raises(NetlistError, match="not found"):
            await handle_symbol_info(SymbolInfoInput(symbol="bogus_xyz_zzz"), asc_state)


@pytest.mark.asyncio
class TestGetComponentInfo:
    async def test_existing(self, asc_state: SessionState, asc_file: Path):
        result = await handle_component_info(
            ComponentInfoInput(path=asc_file.name, reference="R1"), asc_state
        )
        text = result.content[0].text
        assert "R1" in text
        assert "1k" in text
        assert "Pins:" in text

    async def test_missing_ref(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_component_info(
                ComponentInfoInput(path=asc_file.name, reference="ZZZ"), asc_state
            )

    async def test_requires_asc(self, asc_state: SessionState, work_dir: Path):
        cir = work_dir / "x.cir"
        cir.write_text("R1 1 0 1k\n.END\n")
        with pytest.raises(NetlistError, match=r"requires an \.asc"):
            await handle_component_info(
                ComponentInfoInput(path=cir.name, reference="R1"), asc_state
            )


@pytest.mark.asyncio
class TestRemoveComponent:
    async def test_remove(self, asc_state: SessionState, asc_file: Path):
        result = await handle_remove_component(
            RemoveComponentInput(path=asc_file.name, reference="C1"), asc_state
        )
        assert "Removed C1" in result.content[0].text
        # Confirm not present anymore
        read = await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)
        comp_refs = [c["reference"] for c in read.structuredContent["components"]]
        assert "C1" not in comp_refs

    async def test_remove_unknown(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_remove_component(
                RemoveComponentInput(path=asc_file.name, reference="ZZZ"), asc_state
            )


@pytest.mark.asyncio
class TestMoveComponent:
    async def test_move(self, asc_state: SessionState, asc_file: Path):
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200),
            asc_state,
        )
        assert "Moved R1" in result.content[0].text

    async def test_move_with_rotation(self, asc_state: SessionState, asc_file: Path):
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200, rotation="R180"),
            asc_state,
        )
        assert "R180" in result.content[0].text

    async def test_keep_rotation(self, asc_state: SessionState, asc_file: Path):
        # Omit rotation to exercise the "keep current" branch
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=400, y=400),
            asc_state,
        )
        assert "R1" in result.content[0].text


@pytest.mark.asyncio
class TestSetComponentAttribute:
    async def test_set(self, asc_state: SessionState, asc_file: Path):
        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_file.name, reference="R1", attribute="SpiceLine", value="tol=1%"
            ),
            asc_state,
        )
        assert "SpiceLine" in result.content[0].text


@pytest.mark.asyncio
class TestDiffCircuitAttributes:
    async def test_attribute_change_detected(
        self, asc_state: SessionState, asc_file: Path, work_dir: Path
    ) -> None:
        # Regression: diff_circuit compared only the Value field, so a
        # set_component_attribute edit (SpiceLine/Value2/SpiceModel) — which
        # lands in the exported netlist — falsely showed "no differences".
        a = work_dir / "diff_a.asc"
        b = work_dir / "diff_b.asc"
        _copy_file(asc_file, a)
        _copy_file(asc_file, b)
        await handle_set_component_attribute(
            SetComponentAttributeInput(
                path="diff_b.asc", reference="R1", attribute="SpiceLine", value="tol=1"
            ),
            asc_state,
        )
        result = await handle_diff_circuit(
            DiffCircuitInput(path_a="diff_a.asc", path_b="diff_b.asc"), asc_state
        )
        data = result.structuredContent
        assert data is not None
        changed = {c["reference"]: c for c in data["components_changed"]}
        assert "R1" in changed
        assert "tol=1" in changed["R1"]["after"]
        assert changed["R1"]["before"] != changed["R1"]["after"]


@pytest.mark.asyncio
class TestAddComponent:
    async def test_add(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_component(
            AddComponentInput(
                path=asc_file.name,
                reference="R2",
                symbol="res",
                x=400,
                y=300,
                value="2k",
            ),
            asc_state,
        )
        text = result.content[0].text
        assert "Added R2" in text
        assert "2k" in text
        assert result.structuredContent["reference"] == "R2"
        assert "pins" in result.structuredContent

    async def test_add_duplicate(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="already exists"):
            await handle_add_component(
                AddComponentInput(path=asc_file.name, reference="R1", symbol="res", x=0, y=0),
                asc_state,
            )


@pytest.mark.asyncio
class TestAddNetLabel:
    async def test_add_at_xy(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="VCC", x=100, y=100),
            asc_state,
        )
        assert "VCC" in result.content[0].text

    async def test_add_ground(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="0", x=200, y=400),
            asc_state,
        )
        assert "ground" in result.content[0].text

    async def test_missing_xy_and_pin(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Either pin or both"):
            await handle_add_net_label(NetLabelInput(path=asc_file.name, net="X"), asc_state)

    async def test_remove_existing(self, asc_state: SessionState, asc_file: Path):
        # Draft1.asc has "filtered" label at (208, 128)
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="filtered", x=208, y=128, action="remove"),
            asc_state,
        )
        assert "Removed" in result.content[0].text

    async def test_remove_nonexistent(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="No"):
            await handle_add_net_label(
                NetLabelInput(path=asc_file.name, net="zz", x=999, y=999, action="remove"),
                asc_state,
            )

    async def test_duplicate_warning(self, asc_state: SessionState, asc_file: Path):
        # Add a second 'filtered' label at a different position
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="filtered", x=400, y=400),
            asc_state,
        )
        assert "Warning" in result.content[0].text


@pytest.mark.asyncio
class TestEditDirectiveCommentKind:
    async def test_add_comment_via_edit_directive(self, asc_state: SessionState, asc_file: Path):
        """Free-text annotations now go through ``edit_directive`` with
        ``kind='comment'`` instead of the old ``add_text`` tool."""
        result = await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="add",
                instruction="Test note",
                kind="comment",
                x=100,
                y=200,
            ),
            asc_state,
        )
        assert "Test note" in result.content[0].text

    async def test_comment_rejects_directive_prefix(self, asc_state: SessionState, asc_file: Path):
        """``kind='comment'`` with an instruction that starts with
        ``!`` or ``.`` is almost always a mis-typed kind — refuse and
        steer the caller to ``kind='directive'``."""
        from ltspice_mcp.errors import NetlistError

        for instruction in ("!.tran 1m", ".ac dec 100 1 1Meg"):
            with pytest.raises(NetlistError, match="looks like a SPICE directive"):
                await handle_edit_directive(
                    EditDirectiveInput(
                        path=asc_file.name,
                        action="add",
                        instruction=instruction,
                        kind="comment",
                    ),
                    asc_state,
                )

    async def test_remove_spans_directive_and_comment(
        self, asc_state: SessionState, asc_file: Path
    ):
        """``remove`` should hit comments too — previously you could
        ``add_text`` a stray ``;.foo`` line and ``edit_directive remove``
        couldn't touch it."""
        await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="add",
                instruction="zap me",
                kind="comment",
            ),
            asc_state,
        )
        await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="remove",
                instruction="regex:zap me",
            ),
            asc_state,
        )
        # Comment should be gone — re-removing yields no error since the
        # underlying spicelib calls are tolerant of misses. ASC files may
        # contain Latin-1 µ characters, so read raw bytes and replace.
        text = asc_file.read_bytes().decode("utf-8", errors="replace")  # noqa: ASYNC240
        assert "zap me" not in text


@pytest.mark.asyncio
class TestConnect:
    async def test_diagonal_rejected(self, asc_state: SessionState, asc_file: Path):
        # First add a unique net label, then try a diagonal route to it
        await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="X", x=100, y=200), asc_state
        )
        with pytest.raises(NetlistError, match="not orthogonal"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:X",
                    waypoints=[],
                ),
                asc_state,
            )

    async def test_multiple_ground_labels_error(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Multiple '0'"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_invalid_pin_format(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Invalid pin reference"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="badformat",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_unknown_component(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="ZZZ.A",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_missing_net_label(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Net label"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:nonexistent",
                    to_pin="net:0",
                ),
                asc_state,
            )

    async def test_pin_unknown(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="R1.ZZ",
                    to_pin="net:0",
                ),
                asc_state,
            )


def _wire_segments(asc_path: Path) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Parse WIRE records from an .asc on disk (sync read, keeps blocking
    pathlib I/O out of async test bodies)."""
    text = asc_path.read_bytes().decode("utf-8", errors="replace")
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for line in text.splitlines():
        if line.startswith("WIRE"):
            _, x1, y1, x2, y2 = line.split()
            segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return segments


def _has_segment(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """True if a wire segment with endpoints a and b exists in either order."""
    return (a, b) in segments or (b, a) in segments


# Absolute pin positions expected for the fixture nmos symbol (pin offsets
# D=(0,-96), G=(-48,0), S=(0,96)) placed at (400, 200), hand-computed from the
# LTspice orientation transforms (y axis points down; R90 maps (x, y) to
# (-y, x); M0 negates x before rotating). The G pin sits off the symbol's
# vertical axis, so each mirror produces a pin map distinct from its rotation
# counterpart — a sign error in any transform entry changes at least one pin.
NMOS_PIN_POSITIONS: dict[str, dict[str, tuple[int, int]]] = {
    "R0": {"D": (400, 104), "G": (352, 200), "S": (400, 296)},
    "R90": {"D": (496, 200), "G": (400, 152), "S": (304, 200)},
    "R180": {"D": (400, 296), "G": (448, 200), "S": (400, 104)},
    "R270": {"D": (304, 200), "G": (400, 248), "S": (496, 200)},
    "M0": {"D": (400, 104), "G": (448, 200), "S": (400, 296)},
    "M90": {"D": (496, 200), "G": (400, 248), "S": (304, 200)},
    "M180": {"D": (400, 296), "G": (352, 200), "S": (400, 104)},
    "M270": {"D": (304, 200), "G": (400, 152), "S": (496, 200)},
}


@pytest.mark.asyncio
class TestOrientationPlacementAndRouting:
    """add_component(rotation=...) -> cached editor -> _resolve_pin -> connect
    must agree on absolute pin coordinates for every rotation AND mirror.
    Wire endpoints on disk are checked against hand-computed positions, so a
    sign error in any orientation transform fails here — not just an
    inconsistency between add_component and connect."""

    @pytest.mark.parametrize("rotation", sorted(NMOS_PIN_POSITIONS))
    async def test_pin_map_and_wire_endpoint(
        self, asc_state: SessionState, work_dir: Path, rotation: str
    ):
        asc = work_dir / "orient.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")

        added = await handle_add_component(
            AddComponentInput(
                path="orient.asc",
                reference="M1",
                symbol="nmos",
                x=400,
                y=200,
                rotation=rotation,  # type: ignore[arg-type]  # parametrized literal
            ),
            asc_state,
        )
        assert added.structuredContent is not None
        reported = {p["name"]: (p["x"], p["y"]) for p in added.structuredContent["pins"]}
        assert reported == NMOS_PIN_POSITIONS[rotation]

        # Fixed second component, far enough from M1 that no route below can
        # collide with its pins. Fixture res pins: 1=(0,-48) -> R9.1=(700,452).
        await handle_add_component(
            AddComponentInput(path="orient.asc", reference="R9", symbol="res", x=700, y=500),
            asc_state,
        )

        # connect re-resolves M1.G from the cached editor's stored placement,
        # so the wire endpoint proves the rotation survived the round trip.
        gx, gy = NMOS_PIN_POSITIONS[rotation]["G"]
        connected = await handle_connect(
            ConnectInput(
                path="orient.asc",
                from_pin="M1.G",
                to_pin="R9.1",
                waypoints=[WaypointInput(x=gx, y=452)],
            ),
            asc_state,
        )
        sc = connected.structuredContent
        assert sc is not None
        assert sc["from"] == {"ref": "M1.G", "x": gx, "y": gy}
        assert sc["to"] == {"ref": "R9.1", "x": 700, "y": 452}

        # Re-read the file from disk: the persisted wire must start at the
        # hand-computed absolute G coordinate and land on R9.1.
        segments = _wire_segments(asc)
        assert _has_segment(segments, (gx, gy), (gx, 452)), segments
        assert _has_segment(segments, (gx, 452), (700, 452)), segments


@pytest.mark.asyncio
class TestConnectPersistsWires:
    """connect's success path must actually write WIRE records to disk —
    the rejection-path tests above only prove it validates."""

    async def test_wire_written_and_persisted(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "wire_persist.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        await handle_add_component(
            AddComponentInput(path="wire_persist.asc", reference="R1", symbol="res", x=200, y=200),
            asc_state,
        )
        await handle_add_component(
            AddComponentInput(path="wire_persist.asc", reference="R2", symbol="res", x=200, y=400),
            asc_state,
        )
        before = _wire_segments(asc)
        assert before == []  # add_component places no wires

        result = await handle_connect(
            ConnectInput(path="wire_persist.asc", from_pin="R1.2", to_pin="R2.1"),
            asc_state,
        )
        assert "Connected R1.2 to R2.1" in result.content[0].text
        sc = result.structuredContent
        assert sc is not None
        assert sc["wire_count"] == 1

        # Fixture res pins: 1=(0,-48), 2=(0,48) -> R1.2=(200,248), R2.1=(200,352).
        after = _wire_segments(asc)
        assert len(after) == len(before) + sc["wire_count"]
        assert _has_segment(after, (200, 248), (200, 352)), after


@pytest.mark.asyncio
class TestAscValueExcludesValue2:
    """Regression: read_circuit / list_components on .asc used to concatenate
    Value+Value2 into the displayed value AND duplicate Value2 under
    attributes. Read Value alone; let Value2 stay only in attributes."""

    async def test_list_components_excludes_value2(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="value2"), asc_state)
        await handle_add_component(
            AddComponentInput(
                path="value2.asc",
                reference="M1",
                symbol="nmos",
                x=200,
                y=200,
                value="NMOS_VTH04",
                attributes={"Value2": "tag1", "SpiceLine": "W=10u L=0.5u"},
            ),
            asc_state,
        )

        result = await handle_list_components({"path": "value2.asc"}, asc_state)
        comps = result.structuredContent["components"]  # type: ignore[index]
        m1 = next(c for c in comps if c["reference"] == "M1")
        # Value field is the Value SYMATTR alone, not "NMOS_VTH04 tag1".
        assert m1["value"] == "NMOS_VTH04"
        # Value2 still appears under attributes.
        assert m1["attributes"]["Value2"] == "tag1"

    async def test_single_ref_lookup_excludes_value2(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="value2_single"), asc_state)
        await handle_add_component(
            AddComponentInput(
                path="value2_single.asc",
                reference="M1",
                symbol="nmos",
                x=200,
                y=200,
                value="NMOS_VTH04",
                attributes={"Value2": "tag1"},
            ),
            asc_state,
        )

        result = await handle_list_components(
            {"path": "value2_single.asc", "reference": "M1"}, asc_state
        )
        assert result.structuredContent["value"] == "NMOS_VTH04"  # type: ignore[index]


@pytest.mark.asyncio
class TestEmptyAttributeRejected:
    """Regression: add_component with an empty SYMATTR value used to write a partial
    SYMATTR line and crash mid-write, leaving the .asc permanently
    unreadable. Reject up front."""

    async def test_empty_attribute_raises(self, asc_state: SessionState, asc_file: Path):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="empty value"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="M_bad",
                    symbol="res",
                    x=600,
                    y=600,
                    attributes={"SpiceModel": ""},
                ),
                asc_state,
            )
        assert asc_file.read_bytes() == original  # noqa: ASYNC240


@pytest.mark.asyncio
class TestEditingAscRollback:
    """Uncaught exceptions inside _editing_asc must invalidate the
    cached editor so a later read doesn't see dirty in-memory mutations,
    and the file on disk must remain intact."""

    async def test_uncaught_exception_after_mutation_invalidates_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Inject a failure after add_component has already mutated the
        # editor in-memory but before save. _post_op_warnings runs in the
        # handler body after _create_component, so raising from it
        # simulates a real spicelib internal error mid-edit.
        from ltspice_mcp.tools import circuit as circuit_mod

        original = asc_file.read_bytes()  # noqa: ASYNC240
        boom_calls = {"n": 0}

        def boom(*_a, **_kw):
            del _a, _kw
            boom_calls["n"] += 1
            raise RuntimeError("injected post-op failure")

        monkeypatch.setattr(circuit_mod, "_post_op_warnings", boom)

        with pytest.raises(RuntimeError, match="injected"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_uncommitted",
                    symbol="res",
                    x=700,
                    y=700,
                ),
                asc_state,
            )

        # The injection fired (sanity).
        assert boom_calls["n"] == 1
        # File on disk is unchanged — save runs only on the success path.
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        # Cache eviction means a fresh read doesn't see R_uncommitted.
        monkeypatch.undo()
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        assert "R_uncommitted" not in result.content[0].text


@pytest.mark.asyncio
class TestAtomicAscSave:
    """A failure while spicelib is rendering the .asc must not
    leave a partially-written file on disk."""

    async def test_save_failure_preserves_original(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from spicelib import AscEditor

        original = asc_file.read_bytes()  # noqa: ASYNC240

        # Inject a save that writes partial bytes to whatever sink it gets,
        # then raises. Two cases to defeat:
        #   1. Pre-fix path: editor.save_netlist(str(path)) opens the file
        #      directly. A partial write would land on disk. To prove the
        #      atomic-rename, route through the StringIO sink only (which
        #      atomic_write_text uses) — so a partial sink write does NOT
        #      reach the target.
        #   2. Post-fix path: editor.save_netlist(buf), then
        #      atomic_write_text(target, buf.getvalue(). On failure, the
        #      sibling temp is cleaned up and target stays intact.
        def failing_save(self_editor, sink):
            del self_editor
            # Write partial content to the sink (StringIO or file handle).
            if hasattr(sink, "write"):
                sink.write("Version 4\nSHEET 1 0 0\n!!CORRUPT!!\n")
            elif isinstance(sink, str):
                # Pre-fix code path: it would have passed a string path,
                # so spicelib opens the file directly. Simulate spicelib
                # writing partial content before crashing.
                Path(sink).write_text("Version 4\nSHEET 1 0 0\n!!CORRUPT!!\n")
            raise OSError("disk full simulation")

        monkeypatch.setattr(AscEditor, "save_netlist", failing_save)

        with pytest.raises(OSError, match="disk full"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_aborted_save",
                    symbol="res",
                    x=600,
                    y=600,
                ),
                asc_state,
            )

        # Atomic-rename guarantee: no partial write reached the target.
        assert asc_file.read_bytes() == original  # noqa: ASYNC240

    async def test_save_failure_evicts_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A save that mutates the in-memory editor then crashes
        must still invalidate the cache. Otherwise a follow-up read sees
        the unsaved component."""
        from spicelib import AscEditor

        def failing_save(*args, **_kw):
            raise OSError("disk full simulation")

        monkeypatch.setattr(AscEditor, "save_netlist", failing_save)

        with pytest.raises(OSError, match="disk full"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="R_uncommitted",
                    symbol="res",
                    x=600,
                    y=600,
                ),
                asc_state,
            )

        # Restore real save so the follow-up read works.
        monkeypatch.undo()

        # The component must NOT be visible — cache was evicted, fresh
        # read from disk shows the pre-failure state.
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        assert "R_uncommitted" not in result.content[0].text


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestSetAttributeAllowlist:
    """set_component_attribute rejects unknown attribute names."""

    async def test_rejects_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Unknown attribute"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name,
                    reference="R1",
                    attribute="NotARealAttr",
                    value="x",
                ),
                asc_state,
            )

    async def test_suggests_canonical_for_case_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Did you mean 'SpiceLine'"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name, reference="R1", attribute="spiceline", value="x"
                ),
                asc_state,
            )

    async def test_accepts_spiceline(self, asc_state: SessionState, asc_file: Path):
        # Sanity: the canonical name still works.
        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_file.name, reference="R1", attribute="SpiceLine", value="tc=10ppm"
            ),
            asc_state,
        )
        assert "SpiceLine" in result.content[0].text


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestFloatingLabelWarning:
    """add_net_label warns on labels placed away from any wire/pin."""

    async def test_warns_on_floating(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="VCC_floating", x=10, y=10),
            asc_state,
        )
        assert "floating" in result.content[0].text.lower()


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestNetConflictInConnect:
    """connect detects shorts between two named nets."""

    async def test_refuses_named_net_short(self, asc_state: SessionState):
        # Build a clean schematic with two resistors on disjoint named nets,
        # then try to connect them.
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="net_conflict_test"), asc_state)
        await handle_add_component(
            AddComponentInput(
                path="net_conflict_test.asc",
                reference="R1",
                symbol="res",
                x=100,
                y=100,
            ),
            asc_state,
        )
        await handle_add_component(
            AddComponentInput(
                path="net_conflict_test.asc",
                reference="R2",
                symbol="res",
                x=300,
                y=100,
            ),
            asc_state,
        )
        # The test fixture's stripped 'res' symbol uses numeric pin names.
        await handle_add_net_label(
            NetLabelInput(path="net_conflict_test.asc", net="LEFT", pin="R1.1"),
            asc_state,
        )
        await handle_add_net_label(
            NetLabelInput(path="net_conflict_test.asc", net="RIGHT", pin="R2.1"),
            asc_state,
        )
        with pytest.raises(NetlistError, match="Net-label conflict"):
            await handle_connect(
                ConnectInput(
                    path="net_conflict_test.asc",
                    from_pin="R1.1",
                    to_pin="R2.1",
                ),
                asc_state,
            )


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestRemoveComponentNoFalseOrphans:
    """remove_component doesn't flag wires belonging to other components."""

    async def test_other_component_pin_not_flagged(self, asc_state: SessionState, asc_file: Path):
        # Add a second resistor whose pin coincides with R1's existing wire.
        # When we remove R2, the wire connecting R1 stays — and our orphan
        # detector should NOT flag it.
        await handle_add_component(
            AddComponentInput(
                path=asc_file.name,
                reference="R2",
                symbol="res",
                x=128,
                y=112,  # same coords as R1 — pins overlap
                value="2k",
                rotation="R90",
            ),
            asc_state,
        )
        result = await handle_remove_component(
            RemoveComponentInput(path=asc_file.name, reference="R2"),
            asc_state,
        )
        # The remaining R1's wires shouldn't be flagged as orphans.
        assert "orphaned" not in result.content[0].text


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestApplySchematicOps:
    """apply_schematic_ops batches add/connect/label/directive."""

    async def test_basic_transaction(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_demo"), asc_state)

        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_demo.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                        "value": "1k",
                    },
                    {
                        "op": "add_component",
                        "reference": "C1",
                        "symbol": "cap",
                        "x": 200,
                        "y": 100,
                        "value": "1u",
                    },
                    {
                        "op": "add_directive",
                        "instruction": ".tran 1m",
                    },
                ],
            ),
            asc_state,
        )
        text = result.content[0].text
        data = result.structuredContent
        assert data["applied_count"] == 3
        assert data["failed_count"] == 0
        assert data["saved"] is True
        assert "All changes saved." in text

    async def test_stop_on_error_aborts(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_abort"), asc_state)
        # Op #1 succeeds, op #2 fails (unknown symbol). The R1 add must NOT
        # be persisted because stop_on_error defaults to True.
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_abort.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "X1",
                        "symbol": "definitely_not_a_symbol",
                        "x": 200,
                        "y": 100,
                    },
                ],
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "Transaction aborted" in result.content[0].text

        # Verify the file actually doesn't have R1 — load and check.
        from ltspice_mcp.tools.circuit import (
            CircuitReadInput,
            handle_read_circuit,
        )

        read = await handle_read_circuit(CircuitReadInput(path="batch_abort.asc"), asc_state)
        refs = {c["reference"] for c in read.structuredContent.get("components", [])}
        assert "R1" not in refs

    async def test_continue_on_error_persists_partial(
        self, asc_state: SessionState, work_dir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_partial"), asc_state)
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_partial.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 100,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "X1",
                        "symbol": "definitely_not_a_symbol",
                        "x": 200,
                        "y": 100,
                    },
                    {
                        "op": "add_component",
                        "reference": "C1",
                        "symbol": "cap",
                        "x": 300,
                        "y": 100,
                    },
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data["applied_count"] == 2
        assert data["failed_count"] == 1
        assert data["saved"] is True


# Relocated regression coverage from a retired test module.
class TestMidSegmentLabelDetected:
    """A label sitting mid-segment on a wire used to be invisible
    to ``connect``'s endpoint-only label compare. The fix is segment-
    aware: the trace dragon-swallows interest points that lie on a wire
    even if they're not at an endpoint.
    """

    def test_point_on_segment_horizontal(self) -> None:
        from ltspice_mcp.tools.circuit import _point_on_segment

        # Mid-x point on a horizontal wire.
        assert _point_on_segment((150, 100), (100, 100), (200, 100))
        # Same y but outside x-range.
        assert not _point_on_segment((300, 100), (100, 100), (200, 100))
        # Different y.
        assert not _point_on_segment((150, 101), (100, 100), (200, 100))

    def test_point_on_segment_vertical(self) -> None:
        from ltspice_mcp.tools.circuit import _point_on_segment

        assert _point_on_segment((100, 150), (100, 100), (100, 200))
        assert not _point_on_segment((100, 250), (100, 100), (100, 200))
        assert not _point_on_segment((101, 150), (100, 100), (100, 200))

    def test_named_labels_strips_ground(self) -> None:
        from ltspice_mcp.tools.circuit import _named_labels

        assert _named_labels(frozenset({"OUTP", "0"})) == {"OUTP"}
        assert _named_labels(frozenset({"0"})) == set()
        assert _named_labels(frozenset()) == set()


# Relocated regression coverage from a retired test module.
class TestParseNetlistForSynth:
    def test_basic_rc(self):
        instances, directives, skipped, _warnings = _parse_netlist_for_synth(RC_NETLIST)
        refs = {i.ref for i in instances}
        assert refs == {"V1", "R1", "C1"}
        assert not skipped
        assert any(d.lower().startswith(".ac") for d in directives)
        # Title line dropped, .end dropped.
        assert all(not d.lower().startswith(".end") for d in directives)

    def test_symbol_and_nodes_mapping(self):
        instances, *_ = _parse_netlist_for_synth(RC_NETLIST)
        by_ref = {i.ref: i for i in instances}
        assert by_ref["R1"].symbol == "res"
        assert by_ref["C1"].symbol == "cap"
        assert by_ref["V1"].symbol == "voltage"
        assert by_ref["R1"].nodes == ("in", "out")
        assert by_ref["R1"].value == "1k"

    def test_multi_token_source_value_preserved(self):
        instances, *_ = _parse_netlist_for_synth(
            "* t\nV1 in 0 SINE(0 1 1k) AC 1\nR1 in 0 1k\n.end\n"
        )
        v1 = next(i for i in instances if i.ref == "V1")
        assert v1.nodes == ("in", "0")
        assert v1.value == "SINE(0 1 1k) AC 1"

    def test_unsupported_element_skipped(self):
        instances, _, skipped, _ = _parse_netlist_for_synth(
            "* t\nM1 d g s b NMOS\nR1 a b 1k\n.end\n"
        )
        assert {i.ref for i in instances} == {"R1"}
        assert any(s["ref"] == "M1" for s in skipped)

    def test_subckt_def_warns(self):
        _, _, _, warnings = _parse_netlist_for_synth(
            "* t\nR1 a b 1k\n.subckt amp in out\nR2 in out 1k\n.ends\n.end\n"
        )
        assert any("ubcircuit" in w for w in warnings)

    def test_malformed_body_skipped_not_raised(self):
        # "R1 net(a b 1k" lexes as an instance (R prefix) but tokenize_body
        # raises on the unbalanced paren — it must be skipped, not crash.
        instances, _, skipped, _ = _parse_netlist_for_synth(
            "* t\nR1 net(a b 1k\nC1 a 0 1u\n.end\n"
        )
        assert {i.ref for i in instances} == {"C1"}
        assert any(s["ref"] == "R1" and "tokenize" in s["reason"] for s in skipped)

    def test_no_title_keeps_first_instance(self):
        # Regression: a bare netlist fragment (no '*' title) must NOT silently drop its
        # first card — that used to delete the source (V1) and leave a dead
        # circuit with no feedback.
        instances, _directives, skipped, warnings = _parse_netlist_for_synth(
            "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.ac dec 10 1 100k\n.end\n"
        )
        assert {i.ref for i in instances} == {"V1", "R1", "C1"}
        assert not skipped
        assert any("title" in w.lower() for w in warnings)

    def test_title_comment_dropped_without_warning(self):
        # RC_NETLIST has a leading '* RC low-pass filter' comment: it is the
        # conventional deck title, dropped silently, all instances kept.
        instances, _directives, _skipped, warnings = _parse_netlist_for_synth(RC_NETLIST)
        assert {i.ref for i in instances} == {"V1", "R1", "C1"}
        assert not any("title" in w.lower() for w in warnings)


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestSchematicFromNetlist:
    async def test_roundtrip_through_read_circuit(self, asc_state: SessionState, work_dir: Path):
        res = await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_rc", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["placed"] == 3
        assert set(sc["nets"]) == {"in", "out", "0"}
        assert sc["directive_count"] == 1
        assert not sc["skipped"]

        read = await handle_read_circuit(CircuitReadInput(path="synth_rc.asc"), asc_state)
        rsc = read.structuredContent
        assert rsc["type"] == "asc"
        refs = {c["reference"] for c in rsc["components"]}
        assert refs == {"R1", "C1", "V1"}
        values = {c["reference"]: c["value"] for c in rsc["components"]}
        assert values["R1"] == "1k"
        assert values["C1"] == "1u"
        assert values["V1"] == "AC 1"
        label_texts = {lbl["text"] for lbl in rsc["labels"]}
        assert {"in", "out", "0"} <= label_texts
        assert any(d.lower().startswith(".ac") for d in rsc["directives"])

    async def test_overwrite_after_read_uses_fresh_stub(
        self, asc_state: SessionState, work_dir: Path
    ):
        # Regression (Codex review): the overwrite path must populate the fresh
        # blank stub, not an editor cached from a prior read of the old content.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="ow_read", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        await handle_read_circuit(CircuitReadInput(path="ow_read.asc"), asc_state)  # caches editor
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(
                name="ow_read", content="* t\nR9 a b 2k\nC9 b 0 2u\n.end\n", overwrite=True
            ),
            asc_state,
        )
        read = await handle_read_circuit(CircuitReadInput(path="ow_read.asc"), asc_state)
        assert read.structuredContent is not None
        refs = {c["reference"] for c in read.structuredContent["components"]}
        assert refs == {"R9", "C9"}  # only the new content, not R1/C1 from the first synth

    async def test_reports_skipped_unsupported(self, asc_state: SessionState):
        content = "* t\nM1 d g s NMOS\nR1 in out 1k\nC1 out 0 1u\n.end\n"
        res = await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_skip", content=content, overwrite=True),
            asc_state,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["placed"] == 2  # R1, C1
        assert any(s["ref"] == "M1" for s in sc["skipped"])

    async def test_refuses_overwrite_by_default(self, asc_state: SessionState, work_dir: Path):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_dup", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        with pytest.raises(NetlistError, match="already exists"):
            await handle_schematic_from_netlist(
                SchematicFromNetlistInput(name="synth_dup", content=RC_NETLIST),
                asc_state,
            )

    async def test_nothing_to_place_raises(self, asc_state: SessionState):
        with pytest.raises(NetlistError, match="Nothing to place"):
            await handle_schematic_from_netlist(
                SchematicFromNetlistInput(name="synth_empty", content="* just a title\n"),
                asc_state,
            )


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestTraceNet:
    async def test_name_based_net_on_synth_output(self, asc_state: SessionState):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_rc", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        # R1.1 is on node "in" (SpiceOrder 1). V1.+ is also on "in" — they are
        # at different coordinates connected only by the shared label name.
        res = await handle_trace_net(TraceNetInput(path="trace_rc.asc", pin="R1.1"), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        refs = {p["reference"] for p in sc["pins"]}
        assert refs == {"R1", "V1"}
        assert sc["is_shorted"] is False

    async def test_trace_by_net_name(self, asc_state: SessionState):
        # net:in matches one FLAG per pin (V1.+ and R1.1) — _resolve_pin would
        # refuse the ambiguity, but trace_net seeds from a match and name-merges.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_byname", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        res = await handle_trace_net(
            TraceNetInput(path="trace_byname.asc", pin="net:in"), asc_state
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        assert {p["reference"] for p in sc["pins"]} == {"R1", "V1"}

    async def test_trace_by_missing_net_name_raises(self, asc_state: SessionState):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_miss", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        with pytest.raises(NetlistError, match="not found"):
            await handle_trace_net(
                TraceNetInput(path="trace_miss.asc", pin="net:nonexistent"), asc_state
            )

    async def test_short_detection(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "short.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nWIRE 0 0 100 0\nFLAG 0 0 a\nFLAG 100 0 b\n")
        res = await handle_trace_net(TraceNetInput(path="short.asc", x=0, y=0), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["is_shorted"] is True
        assert set(sc["labels"]) == {"a", "b"}

    async def test_empty_coordinate_raises(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "empty.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nFLAG 0 0 a\n")
        with pytest.raises(NetlistError, match="Nothing found"):
            await handle_trace_net(TraceNetInput(path="empty.asc", x=500, y=500), asc_state)


# Relocated regression coverage from a retired test module.
class TestOnWirePredicate:
    def test_matches_point_on_segment(self):
        segments = [((0, 0), (100, 0)), ((100, 0), (100, 80)), ((50, 50), (50, 50))]
        on_wire = _build_on_wire_predicate(segments)
        probes = [(0, 0), (50, 0), (100, 0), (100, 40), (100, 80), (50, 50), (10, 10), (200, 0)]
        for p in probes:
            expected = any(_point_on_segment(p, v1, v2) for v1, v2 in segments)
            assert on_wire(p) == expected, p

    def test_endpoints_and_spans(self):
        on_wire = _build_on_wire_predicate([((0, 0), (0, 100))])
        assert on_wire((0, 0))
        assert on_wire((0, 50))
        assert on_wire((0, 100))
        assert not on_wire((10, 50))


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestAddComponentFloatingFilter:
    async def test_only_new_component_floating_pins(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "build.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        # First component: both pins float.
        await handle_add_component(
            AddComponentInput(path="build.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        # Second component placed far away: its warnings must NOT re-list R1's
        # floating pins (the O(n^2) spam this fix removes).
        res = await handle_add_component(
            AddComponentInput(path="build.asc", reference="R2", symbol="res", x=400, y=100),
            asc_state,
        )
        vw = res.structuredContent.get("validation_warnings", [])
        refs = {w["ref"] for w in vw}
        assert refs <= {"R2"}
        assert "R1" not in refs


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestResetSchematic:
    async def test_revert_after_edit(self, asc_state: SessionState, asc_file: Path):
        original = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200),
            asc_state,
        )
        assert _read_bytes(asc_file) != original  # edit landed
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is True
        assert _read_bytes(asc_file) == original  # byte-exact restore

    async def test_nothing_to_revert(self, asc_state: SessionState, asc_file: Path):
        # No in-session edit captured → reverted=False, not an error.
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False
        assert res.structuredContent["bytes"] is None

    async def test_snapshot_is_pre_first_edit(self, asc_state: SessionState, asc_file: Path):
        # Two edits, then reset → restores the state before the FIRST edit.
        original = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200), asc_state
        )
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=400, y=400), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert _read_bytes(asc_file) == original

    async def test_reset_then_reedit_resnapshots(self, asc_state: SessionState, asc_file: Path):
        # After a reset the snapshot is dropped; a new edit establishes a fresh
        # restore point, and a reset with no new edit finds nothing.
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False
        after_reset = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=500, y=500), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert _read_bytes(asc_file) == after_reset

    async def test_requires_asc(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "x.cir"
        cir.write_text("* t\nR1 a b 1k\n.end\n")
        with pytest.raises(NetlistError, match=r"requires an \.asc"):
            await handle_reset_schematic(ResetSchematicInput(path="x.cir"), state_no_sim)

    async def test_synth_new_file_not_revertible_to_stub(self, asc_state: SessionState):
        # Regression: schematic_from_netlist writes a blank stub then edits it.
        # reset_schematic must NOT restore that 30-byte stub for a NEW file —
        # there's no pre-session state, so it reports reverted=False.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_reset", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        res = await handle_reset_schematic(ResetSchematicInput(path="synth_reset.asc"), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False

    async def test_synth_overwrite_reverts_to_original(
        self, asc_state: SessionState, work_dir: Path
    ):
        # overwrite=true synth over an existing file → reset restores the
        # ORIGINAL bytes (captured before the overwrite), not the blank stub.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_ow", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        original = _read_bytes(work_dir / "synth_ow.asc")
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(
                name="synth_ow", content="* t\nR9 a b 2k\nC9 b 0 2u\n.end\n", overwrite=True
            ),
            asc_state,
        )
        assert _read_bytes(work_dir / "synth_ow.asc") != original
        res = await handle_reset_schematic(ResetSchematicInput(path="synth_ow.asc"), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is True
        assert _read_bytes(work_dir / "synth_ow.asc") == original


# Relocated regression coverage from a retired test module.
class TestValidateNetlistAscTopology:
    """validate_netlist surfaces .asc shorts/floating/dangling."""

    async def test_named_net_short_and_floating_pins_flagged(
        self, asc_state: SessionState, work_dir: Path
    ):
        (work_dir / "shorted.asc").write_text(SHORTED_ASC)
        result = await handle_validate_netlist(ValidateNetlistInput(path="shorted.asc"), asc_state)
        data = result.structuredContent
        assert data is not None
        issues = data["issues"]
        # 1 short (error) + 2 floating pins (warning).
        assert data["issue_count"] >= 3, issues

        shorts = [
            i for i in issues if i["severity"] == "error" and "short" in i["message"].lower()
        ]
        assert len(shorts) == 1, issues
        assert "aaa" in shorts[0]["message"] and "bbb" in shorts[0]["message"]

        floating = [
            i
            for i in issues
            if i["severity"] == "warning" and "floating pin" in i["message"].lower()
        ]
        assert len(floating) == 2, issues

    async def test_clean_schematic_has_no_topology_issues(
        self, asc_state: SessionState, work_dir: Path
    ):
        (work_dir / "clean.asc").write_text(CLEAN_ASC)
        result = await handle_validate_netlist(ValidateNetlistInput(path="clean.asc"), asc_state)
        data = result.structuredContent
        assert data is not None
        assert data["issue_count"] == 0, data["issues"]

    async def test_ground_label_not_treated_as_short(
        self, asc_state: SessionState, work_dir: Path
    ):
        (work_dir / "gnd.asc").write_text(GROUND_ASC)
        result = await handle_validate_netlist(ValidateNetlistInput(path="gnd.asc"), asc_state)
        data = result.structuredContent
        assert data is not None
        shorts = [
            i
            for i in data["issues"]
            if i["severity"] == "error" and "short" in i["message"].lower()
        ]
        assert shorts == [], data["issues"]
