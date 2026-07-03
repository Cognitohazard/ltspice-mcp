"""Integration tests for .asc schematic editing tools using fixture symbols."""

from pathlib import Path

import pytest
from mcp.types import TextContent

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    ApplySchematicOpsInput,
    CircuitReadInput,
    ComponentInfoInput,
    ConnectInput,
    CreateSchematicInput,
    DiffCircuitInput,
    EditDirectiveInput,
    MoveComponentInput,
    NetLabelInput,
    RemoveComponentInput,
    ResetSchematicInput,
    SetComponentAttributeInput,
    SetComponentValueInput,
    SymbolInfoInput,
    TraceNetInput,
    ValidateNetlistInput,
    WaypointInput,
    _build_on_wire_predicate,
    _point_on_segment,
    handle_add_component,
    handle_add_net_label,
    handle_apply_schematic_ops,
    handle_component_info,
    handle_connect,
    handle_create_schematic,
    handle_diff_circuit,
    handle_edit_directive,
    handle_list_components,
    handle_move_component,
    handle_read_circuit,
    handle_remove_component,
    handle_reset_schematic,
    handle_set_component_attribute,
    handle_set_component_value,
    handle_symbol_info,
    handle_trace_net,
    handle_validate_netlist,
)


def _result_text(result) -> str:
    """Extract text from a tool result's first content block, asserting it is text."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _copy_file(src: Path, dst: Path) -> None:
    """Sync byte-copy — keeps blocking pathlib I/O out of async test bodies."""
    dst.write_bytes(src.read_bytes())


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
        text = _result_text(result)
        assert "C1" in text
        assert "R1" in text
        assert "V1" in text
        # Net labels
        assert "filtered" in text
        assert result.structuredContent["type"] == "asc"

    async def test_list_components_asc(self, asc_state: SessionState, asc_file: Path):
        result = await handle_list_components({"path": asc_file.name}, asc_state)
        text = _result_text(result)
        assert "C1" in text
        assert "R1" in text


@pytest.mark.asyncio
class TestGetSymbolInfo:
    async def test_valid_symbol(self, asc_state: SessionState):
        result = await handle_symbol_info(
            SymbolInfoInput(symbol="res", x=0, y=0, rotation="R0"), asc_state
        )
        text = _result_text(result)
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
        text = _result_text(result)
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
        assert "Removed C1" in _result_text(result)
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
        assert "Moved R1" in _result_text(result)

    async def test_move_with_rotation(self, asc_state: SessionState, asc_file: Path):
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200, rotation="R180"),
            asc_state,
        )
        assert "R180" in _result_text(result)

    async def test_keep_rotation(self, asc_state: SessionState, asc_file: Path):
        # Omit rotation to exercise the "keep current" branch
        result = await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=400, y=400),
            asc_state,
        )
        assert "R1" in _result_text(result)


@pytest.mark.asyncio
class TestSetComponentAttribute:
    async def test_set(self, asc_state: SessionState, asc_file: Path):
        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_file.name, reference="R1", attribute="SpiceLine", value="tol=1%"
            ),
            asc_state,
        )
        assert "SpiceLine" in _result_text(result)


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
        text = _result_text(result)
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
        assert "VCC" in _result_text(result)

    async def test_add_ground(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="0", x=200, y=400),
            asc_state,
        )
        assert "ground" in _result_text(result)

    async def test_missing_xy_and_pin(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Either pin or both"):
            await handle_add_net_label(NetLabelInput(path=asc_file.name, net="X"), asc_state)

    async def test_remove_existing(self, asc_state: SessionState, asc_file: Path):
        # Draft1.asc has "filtered" label at (208, 128)
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="filtered", x=208, y=128, action="remove"),
            asc_state,
        )
        assert "Removed" in _result_text(result)

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
        assert "Warning" in _result_text(result)


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
        assert "Test note" in _result_text(result)

    async def test_add_directive_honors_placement(self, asc_state: SessionState, asc_file: Path):
        """x/y/size on the .asc DIRECTIVE branch must place the directive at
        the given coordinates — previously only the comment branch read
        them, and spicelib's add_instruction silently picked its own spot
        and font size."""
        result = await handle_edit_directive(
            EditDirectiveInput(
                path=asc_file.name,
                action="add",
                instruction=".tran 5m",
                x=320,
                y=240,
                size=3,
            ),
            asc_state,
        )
        assert "Added directive" in _result_text(result)
        content = _read_bytes(asc_file)
        # LTspice TEXT record: "TEXT <x> <y> <align> <size> !<directive>"
        assert b"!.tran 5m" in content
        line = next(ln for ln in content.splitlines() if b"!.tran 5m" in ln)
        assert b"320 240" in line
        assert b" 3 " in line

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
        with pytest.raises(NetlistError, match="Multiple '0'") as exc_info:
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:0",
                ),
                asc_state,
            )
        # Guidance must reference the actual ambiguous net, not a canned example.
        msg = str(exc_info.value)
        assert "add_net_label op of apply_schematic_ops (net='0'" in msg
        assert "M3.S" not in msg

    async def test_multiple_label_error_names_actual_net(
        self, asc_state: SessionState, asc_file: Path
    ):
        # Two same-name labels on a non-ground net: the ambiguity guidance
        # must name that net dynamically.
        await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="SIG", x=100, y=200), asc_state
        )
        await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="SIG", x=300, y=200), asc_state
        )
        with pytest.raises(NetlistError, match="Multiple 'SIG'") as exc_info:
            await handle_connect(
                ConnectInput(
                    path=asc_file.name,
                    from_pin="net:filtered",
                    to_pin="net:SIG",
                ),
                asc_state,
            )
        assert "add_net_label op of apply_schematic_ops (net='SIG'" in str(exc_info.value)

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


def _flag_records(asc_path: Path) -> list[tuple[tuple[int, int], str]]:
    """Parse FLAG (net-label / ground) records from an .asc as ((x, y), net)."""
    text = asc_path.read_bytes().decode("utf-8", errors="replace")
    flags: list[tuple[tuple[int, int], str]] = []
    for line in text.splitlines():
        if line.startswith("FLAG"):
            _, x, y, net = line.split(maxsplit=3)
            flags.append(((int(x), int(y)), net))
    return flags


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


# Canonical device archetypes the schematic-build path MUST handle, beyond the
# passive R/C/V that dominated every battery. (symbol, ordered pin names from the
# fixture .asy.) Covers two-terminal active, three-terminal active, and the
# four-terminal controlled sources — the >2-pin classes a converter/synth that
# only understood 2-terminal devices silently dropped.
BUILD_ARCHETYPES: list[tuple[str, tuple[str, ...]]] = [
    ("diode", ("A", "K")),  # two-terminal active
    ("nmos", ("D", "G", "S")),  # three-terminal active
    ("e", ("+", "-", "P", "N")),  # controlled source (VCVS)
    ("g", ("+", "-", "NC+", "NC-")),  # controlled source (VCCS)
]


@pytest.mark.asyncio
class TestArchetypeBuildCoverage:
    """The build battery's anti-passive-bias guard. A build or synth tool that
    silently skips a device class is an absence-class bug — it has no failing
    code path, so happy-path stress tests on passive circuits never surface it
    (the netlist->asc converter skipped every active device yet passed every
    stress pass). Each non-passive archetype is placed and wired through the real
    build path here, so an unusable-for-a-class regression fails on the next run
    instead of after it ships.
    """

    @pytest.mark.parametrize(("symbol", "pin_names"), BUILD_ARCHETYPES)
    async def test_archetype_places_with_all_terminals_and_wires(
        self, asc_state: SessionState, work_dir: Path, symbol: str, pin_names: tuple[str, ...]
    ):
        asc = work_dir / f"arch_{symbol}.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")

        added = await handle_add_component(
            AddComponentInput(path=asc.name, reference="X1", symbol=symbol, x=400, y=300),
            asc_state,
        )
        assert added.structuredContent is not None
        pins = {p["name"]: (p["x"], p["y"]) for p in added.structuredContent["pins"]}
        # The symbol-geometry layer must report every terminal of the class. This
        # extends the nmos-only orientation coverage to the 2-terminal active and
        # 4-terminal controlled-source classes, so a PIN-parser regression that
        # truncated the terminal list for one of them fails here.
        assert set(pins) == set(pin_names), pins

        # Wire the LAST terminal (a non-positional name — N / NC- — on the
        # 4-terminal sources, which the positional-pin nmos orientation test
        # never reaches) to a passive load. This is the load-bearing part: the
        # terminal must resolve by NAME through connect's pin lookup and the wire
        # must persist to disk at the geometry coordinate — not merely be
        # reported in memory. The load sits collinear and outward from the pin
        # (left if the pin is on the body's left half, else right), so the single
        # straight segment leaves the body without crossing another terminal
        # (every pin of these symbols has a unique y).
        name = pin_names[-1]
        px, py = pins[name]
        out_x = px + (300 if px >= 400 else -300)
        await handle_add_component(
            AddComponentInput(path=asc.name, reference="RL", symbol="res", x=out_x, y=py + 48),
            asc_state,
        )  # res pin 1 = (0,-48) offset -> (out_x, py), collinear with X1.{name}
        connected = await handle_connect(
            ConnectInput(path=asc.name, from_pin=f"X1.{name}", to_pin="RL.1"),
            asc_state,
        )
        assert connected.structuredContent is not None
        # Re-read from disk: the named terminal resolved and the wire persisted.
        assert _has_segment(_wire_segments(asc), (px, py), (out_x, py)), (name, asc.read_text())


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
        assert "Connected R1.2 to R2.1" in _result_text(result)
        sc = result.structuredContent
        assert sc is not None
        assert sc["wire_count"] == 1

        # Fixture res pins: 1=(0,-48), 2=(0,48) -> R1.2=(200,248), R2.1=(200,352).
        after = _wire_segments(asc)
        assert len(after) == len(before) + sc["wire_count"]
        assert _has_segment(after, (200, 248), (200, 352)), after


@pytest.mark.asyncio
class TestSchematicReadability:
    """Readability eval for a schematic built the way the guide recommends:
    apply_schematic_ops + connect for the signal path, add_net_label only for
    the ground/global nets. The result must come out WIRED — not 'net-label
    soup', where every component pin floats on its own same-named FLAG and there
    are no wires. This is the regression guard for the blind spot that let a
    label-only build ship: the signal junctions have to be real WIRE records,
    and net labels stay scoped to the terminal nets.
    """

    async def test_built_schematic_is_wired_not_label_soup(
        self, asc_state: SessionState, work_dir: Path
    ):
        from ltspice_mcp.tools.circuit import CreateSchematicInput, handle_create_schematic

        await handle_create_schematic(CreateSchematicInput(name="readable"), asc_state)
        # A 3-resistor chain stacked on x=200: the two internal junctions are
        # wired by connect; only the two terminal nets (in, ground) get a label.
        # Fixture res pins: 1=(0,-48), 2=(0,48), so Rn at (200, y) has pins at
        # (200, y-48) and (200, y+48).
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="readable.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 200,
                        "y": 200,
                    },
                    {
                        "op": "add_component",
                        "reference": "R2",
                        "symbol": "res",
                        "x": 200,
                        "y": 400,
                    },
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 200,
                        "y": 600,
                    },
                    {"op": "connect", "from_pin": "R1.2", "to_pin": "R2.1"},
                    {"op": "connect", "from_pin": "R2.2", "to_pin": "R3.1"},
                    {"op": "add_net_label", "net": "in", "pin": "R1.1"},
                    {"op": "add_net_label", "net": "0", "pin": "R3.2"},
                ],
            ),
            asc_state,
        )
        assert result.structuredContent is not None
        assert result.structuredContent["failed_count"] == 0
        assert result.structuredContent["saved"] is True

        asc = work_dir / "readable.asc"
        wires = _wire_segments(asc)
        flags = _flag_records(asc)
        flag_coords = {coord for coord, _net in flags}

        # The signal path is WIRED: both internal junctions are real segments.
        assert _has_segment(wires, (200, 248), (200, 352)), wires  # R1.2 - R2.1
        assert _has_segment(wires, (200, 448), (200, 552)), wires  # R2.2 - R3.1

        # Net labels are scoped to the two terminal nets, placed at the terminal
        # pins — not one FLAG per junction.
        assert sorted(net for _coord, net in flags) == ["0", "in"]
        assert flag_coords == {(200, 152), (200, 648)}  # R1.1 (in), R3.2 (gnd)

        # The anti-soup invariant: no internal junction is realized as a label.
        for junction in ((200, 248), (200, 352), (200, 448), (200, 552)):
            assert junction not in flag_coords, f"junction {junction} labeled, not wired"


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

    @pytest.mark.parametrize("bad_value", ["", "   "])
    async def test_add_component_empty_value_rejected(
        self, asc_state: SessionState, asc_file: Path, bad_value: str
    ):
        # The `value` param writes SYMATTR Value directly; an empty/whitespace
        # value corrupts the .asc the same way an empty attribute does.
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="empty value"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name, reference="RX", symbol="res", x=600, y=600, value=bad_value
                ),
                asc_state,
            )
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)

    async def test_apply_ops_add_component_empty_value_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path=asc_file.name,
                ops=[
                    {  # type: ignore[arg-type]
                        "op": "add_component",
                        "reference": "RX",
                        "symbol": "res",
                        "x": 600,
                        "y": 600,
                        "value": "",
                    },
                ],
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "empty value" in data["results"][0]["error"]
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)

    async def test_set_component_attribute_empty_value_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        # The reported bug: set_component_attribute(Value="") wrote a 2-token
        # "SYMATTR Value " line the parser could not read back, bricking the
        # editor for that file. Reject up front; the .asc stays intact + readable.
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="empty value"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name, reference="R1", attribute="Value", value=""
                ),
                asc_state,
            )
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)

    async def test_apply_ops_set_component_attribute_empty_value_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path=asc_file.name,
                ops=[
                    {  # type: ignore[arg-type]
                        "op": "set_component_attribute",
                        "reference": "R1",
                        "attribute": "Value",
                        "value": "",
                    },
                ],
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "empty value" in data["results"][0]["error"]
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        await handle_read_circuit(CircuitReadInput(path=asc_file.name), asc_state)


@pytest.mark.asyncio
class TestSetComponentValueCreatesMissingValue:
    """Regression: set_component_value on a component added without a Value slot
    used to fail 'Component(s) not found' (the component existed). It must create
    the Value line — symmetric with add_component(value=)."""

    async def test_standalone_set_value_creates_missing_value_line(
        self, asc_state: SessionState, asc_file: Path
    ):
        from spicelib import AscEditor

        await handle_add_component(
            AddComponentInput(path=asc_file.name, reference="R9", symbol="res", x=400, y=400),
            asc_state,
        )
        result = await handle_set_component_value(
            SetComponentValueInput(path=asc_file.name, reference="R9", value="22k"),
            asc_state,
        )
        assert "R9" in _result_text(result)
        assert str(AscEditor(str(asc_file)).get_component_value("R9")) == "22k"

    async def test_apply_ops_set_value_after_valueless_add(
        self, asc_state: SessionState, asc_file: Path
    ):
        from spicelib import AscEditor

        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path=asc_file.name,
                ops=[  # type: ignore[arg-type]
                    {
                        "op": "add_component",
                        "reference": "R8",
                        "symbol": "res",
                        "x": 500,
                        "y": 400,
                    },
                    {"op": "set_component_value", "reference": "R8", "value": "33k"},
                ],
            ),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        assert data["saved"] is True
        assert data["failed_count"] == 0
        assert str(AscEditor(str(asc_file)).get_component_value("R8")) == "33k"


@pytest.mark.asyncio
class TestCreateSchematicFormat:
    """Regression: create_schematic rejected a `format` param its sibling tools
    accept (schema was additionalProperties:false with no `format`)."""

    async def test_format_text_accepted(self, asc_state: SessionState):
        result = await handle_create_schematic(
            CreateSchematicInput(name="fmt_text", format="text"), asc_state
        )
        assert "Created schematic" in _result_text(result)

    async def test_format_json_returns_structured(self, asc_state: SessionState):
        result = await handle_create_schematic(
            CreateSchematicInput(name="fmt_json", width=640, height=480, format="json"),
            asc_state,
        )
        data = result.structuredContent
        assert data is not None
        assert data["path"].endswith("fmt_json.asc")
        assert data["width"] == 640
        assert data["height"] == 480
        # The layout checklist must reach structured-aware clients (which show
        # only structuredContent), not just the text channel.
        assert "Layout checklist" in data["hint"]
        assert "add_net_label" in data["hint"]
        assert "apply_schematic_ops" in data["hint"]


@pytest.mark.asyncio
class TestEditingAscRollback:
    """Uncaught exceptions inside _editing_asc must invalidate the
    cached editor so a later read doesn't see dirty in-memory mutations,
    and the file on disk must remain intact."""

    async def test_uncaught_exception_after_mutation_invalidates_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Inject a failure after add_component has already mutated the
        # editor in-memory but before save: wrap _create_component so the real
        # in-memory mutation runs, then raise — simulating a spicelib internal
        # error mid-edit, after the editor is dirty but before the editing
        # context saves.
        from ltspice_mcp.tools import circuit as circuit_mod

        original = asc_file.read_bytes()  # noqa: ASYNC240
        boom_calls = {"n": 0}
        real_create = circuit_mod._create_component

        def boom(*a, **kw):
            real_create(*a, **kw)  # do the real in-memory mutation
            boom_calls["n"] += 1
            raise RuntimeError("injected post-op failure")

        monkeypatch.setattr(circuit_mod, "_create_component", boom)

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
        assert "R_uncommitted" not in _result_text(result)


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
        assert "R_uncommitted" not in _result_text(result)


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
        assert "SpiceLine" in _result_text(result)


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestFloatingLabelWarning:
    """add_net_label warns on labels placed away from any wire/pin."""

    async def test_warns_on_floating(self, asc_state: SessionState, asc_file: Path):
        result = await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="VCC_floating", x=10, y=10),
            asc_state,
        )
        assert "floating" in _result_text(result).lower()


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
        assert "orphaned" not in _result_text(result)


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
        text = _result_text(result)
        data = result.structuredContent
        assert data["applied_count"] == 3
        assert data["failed_count"] == 0
        assert data["saved"] is True
        assert "All changes saved." in text

    async def test_accepts_format_param(self, asc_state: SessionState):
        # Regression: apply_schematic_ops used to reject the `format` field that
        # nearly every other tool accepts, raising a validation error. It must
        # accept and honor it like the rest.
        import json

        await handle_create_schematic(CreateSchematicInput(name="fmt_demo"), asc_state)
        op1 = {"op": "add_component", "reference": "R1", "symbol": "res", "x": 0, "y": 0}
        as_json = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(path="fmt_demo.asc", ops=[op1], format="json"),  # type: ignore[arg-type]
            asc_state,
        )
        assert as_json.structuredContent["applied_count"] == 1
        # json format → the text body IS the JSON payload, not the human summary.
        assert json.loads(_result_text(as_json))["applied_count"] == 1
        op2 = {"op": "add_component", "reference": "R2", "symbol": "res", "x": 100, "y": 0}
        text_only = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(path="fmt_demo.asc", ops=[op2], format="text"),  # type: ignore[arg-type]
            asc_state,
        )
        # text format → human-readable summary body, structured payload still present.
        assert text_only.structuredContent["applied_count"] == 1
        assert "apply_schematic_ops on fmt_demo.asc" in _result_text(text_only)

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
        assert "Transaction aborted" in _result_text(result)

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

    async def test_dry_run_validates_without_saving(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CircuitReadInput,
            CreateSchematicInput,
            handle_create_schematic,
            handle_read_circuit,
        )

        await handle_create_schematic(CreateSchematicInput(name="batch_dry"), asc_state)
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="batch_dry.asc",
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
                dry_run=True,
            ),
            asc_state,
        )
        data = result.structuredContent
        # Every op is attempted in a dry run (the bad op does not stop it).
        assert data["applied_count"] == 2
        assert data["failed_count"] == 1
        assert data["saved"] is False
        assert data["dry_run"] is True
        assert "Dry run" in _result_text(result)

        # The file must be untouched — none of the would-be ops persisted.
        read = await handle_read_circuit(CircuitReadInput(path="batch_dry.asc"), asc_state)
        refs = {c["reference"] for c in read.structuredContent.get("components", [])}
        assert refs == set()


@pytest.mark.asyncio
class TestRemoveWireAndNetLabelOps:
    """remove_wire / remove_net_label apply_schematic_ops ops."""

    async def test_remove_wire_by_endpoints_and_label_by_pin_and_xy(
        self, asc_state: SessionState, work_dir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_ops"), asc_state)
        # Build R1 + C1, wire R1.2 → C1.1, label R1.1 by pin, and place a
        # second label at an explicit coordinate.
        build = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_ops.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 128,
                        "y": 128,
                    },
                    {
                        "op": "add_component",
                        "reference": "C1",
                        "symbol": "cap",
                        "x": 128,
                        "y": 320,
                    },
                    {"op": "connect", "from_pin": "R1.2", "to_pin": "C1.1"},
                    {"op": "add_net_label", "net": "in", "pin": "R1.1"},
                    {"op": "add_net_label", "net": "spare", "x": 512, "y": 512},
                ],
            ),
            asc_state,
        )
        assert build.structuredContent["saved"] is True

        # read_circuit must expose wire segments for discovery/removal.
        read = await handle_read_circuit(CircuitReadInput(path="rm_ops.asc"), asc_state)
        rsc = read.structuredContent
        assert rsc["wires"], "read_circuit should list wire segments"
        wire = rsc["wires"][0]
        # Label coordinates for the by-pin removal target.
        in_label = next(lbl for lbl in rsc["labels"] if lbl["text"] == "in")

        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_ops.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "remove_wire",
                        "x1": wire["x1"],
                        "y1": wire["y1"],
                        "x2": wire["x2"],
                        "y2": wire["y2"],
                    },
                    {"op": "remove_net_label", "pin": "R1.1"},
                    {"op": "remove_net_label", "x": 512, "y": 512},
                ],
            ),
            asc_state,
        )
        data = res.structuredContent
        assert data["saved"] is True
        assert data["failed_count"] == 0
        # Each op reports what it removed.
        by_op = {r["op"]: r for r in data["results"]}
        assert by_op["remove_wire"]["removed"] == 1
        # The by-pin removal must land on the "in" label coordinate.
        assert by_op["remove_net_label"]["ok"] is True

        read2 = await handle_read_circuit(CircuitReadInput(path="rm_ops.asc"), asc_state)
        rsc2 = read2.structuredContent
        assert rsc2["wire_count"] == 0
        assert not rsc2["wires"]
        remaining = {lbl["text"] for lbl in rsc2["labels"]}
        assert "in" not in remaining
        assert "spare" not in remaining
        # Sanity: the removed label coordinate is gone.
        assert not any(
            lbl["x"] == in_label["x"] and lbl["y"] == in_label["y"] for lbl in rsc2["labels"]
        )

    async def test_remove_wire_no_match_raises(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_nomatch"), asc_state)
        # stop_on_error default True: a no-match remove aborts the transaction.
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_nomatch.asc",
                ops=[{"op": "remove_wire", "x1": 0, "y1": 0, "x2": 16, "y2": 0}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        data = res.structuredContent
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No matching wire" in data["results"][0]["error"]

    async def test_remove_net_label_no_match_raises(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_lbl_nomatch"), asc_state)
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_lbl_nomatch.asc",
                ops=[{"op": "remove_net_label", "x": 999, "y": 999}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        data = res.structuredContent
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No net label found" in data["results"][0]["error"]

    async def test_remove_directive_round_trip(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_dir"), asc_state)
        # add_directive then remove_directive is the inverse pair the closure
        # test requires — exercise it end to end so the op actually edits.
        add = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_dir.asc",
                ops=[{"op": "add_directive", "instruction": ".tran 1m"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        assert add.structuredContent["saved"] is True
        read = await handle_read_circuit(CircuitReadInput(path="rm_dir.asc"), asc_state)
        assert any(".tran 1m" in d for d in read.structuredContent["directives"])

        rm = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_dir.asc",
                ops=[{"op": "remove_directive", "instruction": ".tran 1m"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        data = rm.structuredContent
        assert data["saved"] is True
        assert data["failed_count"] == 0
        assert data["results"][0]["removed"] == "directive"

        read2 = await handle_read_circuit(CircuitReadInput(path="rm_dir.asc"), asc_state)
        assert not any(".tran 1m" in d for d in read2.structuredContent["directives"])

    async def test_remove_directive_no_match_raises(self, asc_state: SessionState, work_dir: Path):
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_dir_nomatch"), asc_state)
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_dir_nomatch.asc",
                ops=[{"op": "remove_directive", "instruction": ".tran 999"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        data = res.structuredContent
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No directive or comment" in data["results"][0]["error"]

    async def test_remove_directive_is_exact_not_substring(
        self, asc_state: SessionState, work_dir: Path
    ):
        # The inverse must match the full directive text, not a substring:
        # removing ".tran 1" must NOT delete ".tran 10m" (spicelib's matcher
        # would, silently corrupting the simulation setup).
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_substr"), asc_state)
        await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_substr.asc",
                ops=[{"op": "add_directive", "instruction": ".tran 10m"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_substr.asc",
                ops=[{"op": "remove_directive", "instruction": ".tran 1"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        data = res.structuredContent
        # ".tran 1" is a substring of ".tran 10m" but not an exact match: refuse.
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No directive or comment" in data["results"][0]["error"]
        read = await handle_read_circuit(CircuitReadInput(path="rm_substr.asc"), asc_state)
        assert any(".tran 10m" in d for d in read.structuredContent["directives"])

    async def test_remove_directive_removes_one_of_duplicates(
        self, asc_state: SessionState, work_dir: Path
    ):
        # Inverse of a single add removes a single record: with two identical
        # directives, one remove_directive leaves exactly one.
        from ltspice_mcp.tools.circuit import (
            CreateSchematicInput,
            handle_create_schematic,
        )

        await handle_create_schematic(CreateSchematicInput(name="rm_dup"), asc_state)
        await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_dup.asc",
                ops=[  # type: ignore[arg-type]
                    {"op": "add_directive", "instruction": ".tran 1m"},
                    {"op": "add_directive", "instruction": ".tran 1m"},
                ],
            ),
            asc_state,
        )
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_dup.asc",
                ops=[{"op": "remove_directive", "instruction": ".tran 1m"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        assert res.structuredContent["saved"] is True
        read = await handle_read_circuit(CircuitReadInput(path="rm_dup.asc"), asc_state)
        assert sum(1 for d in read.structuredContent["directives"] if d == ".tran 1m") == 1


@pytest.mark.asyncio
class TestAddNetLabelOpValidation:
    """The add_net_label op is the public path now that the standalone tool is
    unregistered, so it must enforce the same rules: refuse a label that would
    short two different named nets, and surface duplicate-name / floating-
    placement warnings."""

    async def test_short_refused_via_batch(self, asc_state: SessionState):
        from ltspice_mcp.tools.circuit import CreateSchematicInput, handle_create_schematic

        await handle_create_schematic(CreateSchematicInput(name="lbl_short"), asc_state)
        # Two different named labels on the same pin coordinate would merge the
        # nets at netlist time; the second must be refused, not silently saved.
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="lbl_short.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 128,
                        "y": 128,
                    },
                    {"op": "add_net_label", "net": "a", "pin": "R1.1"},
                    {"op": "add_net_label", "net": "b", "pin": "R1.1"},
                ],
                stop_on_error=False,
            ),
            asc_state,
        )
        results = {r["index"]: r for r in res.structuredContent["results"]}
        assert results[1]["ok"] is True  # net "a" placed
        assert results[2]["ok"] is False  # net "b" would short — refused
        assert "short" in results[2]["error"].lower()

    async def test_floating_label_warning_via_batch(self, asc_state: SessionState):
        from ltspice_mcp.tools.circuit import CreateSchematicInput, handle_create_schematic

        await handle_create_schematic(CreateSchematicInput(name="lbl_float"), asc_state)
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="lbl_float.asc",
                ops=[{"op": "add_net_label", "net": "x", "x": 500, "y": 500}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        op = res.structuredContent["results"][0]
        assert op["ok"] is True
        assert any("no wire" in w.lower() for w in op.get("warnings", []))

    async def test_duplicate_label_warning_via_batch(self, asc_state: SessionState):
        from ltspice_mcp.tools.circuit import CreateSchematicInput, handle_create_schematic

        await handle_create_schematic(CreateSchematicInput(name="lbl_dup"), asc_state)
        # Same name on two distinct (unwired) pins: not a short (the netlist merges
        # same-name labels into one net) — the only cost is that a later connect
        # can't disambiguate, which the warning states without a scare.
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="lbl_dup.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 128,
                        "y": 128,
                    },
                    {"op": "add_net_label", "net": "n1", "pin": "R1.1"},
                    {"op": "add_net_label", "net": "n1", "pin": "R1.2"},
                ],
            ),
            asc_state,
        )
        op2 = res.structuredContent["results"][2]
        assert op2["ok"] is True
        warns = op2.get("warnings", [])
        # Reframed: names the duplicate but says it merges correctly and only
        # connect is ambiguous — no "short"/"will error" scare.
        assert any("already labels a net" in w and "ambiguous" in w for w in warns)


@pytest.mark.asyncio
class TestMoveRemoveOpWarnings:
    """The move/remove ops are the public path now; they must surface the same
    bbox-overlap and orphaned-wire warnings the standalone handlers did (these
    are NOT recovered by the batch's end-of-run _post_op_warnings)."""

    async def _build_pair(self, asc_state: SessionState, name: str):
        from ltspice_mcp.tools.circuit import CreateSchematicInput, handle_create_schematic

        await handle_create_schematic(CreateSchematicInput(name=name), asc_state)
        # R1 above R2, wired R1.2 -> R2.1. Fixture res pins: 1=(x,y-48), 2=(x,y+48).
        return await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path=f"{name}.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 200,
                        "y": 200,
                    },
                    {
                        "op": "add_component",
                        "reference": "R2",
                        "symbol": "res",
                        "x": 200,
                        "y": 400,
                    },
                    {"op": "connect", "from_pin": "R1.2", "to_pin": "R2.1"},
                ],
            ),
            asc_state,
        )

    async def test_move_overlap_warning_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "mv_overlap")
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="mv_overlap.asc",
                ops=[{"op": "move_component", "reference": "R2", "x": 200, "y": 200}],  # type: ignore[arg-type]  # onto R1
            ),
            asc_state,
        )
        op = res.structuredContent["results"][0]
        assert op["ok"] is True
        assert any("Overlaps" in w for w in op.get("warnings", []))

    async def test_move_orphan_warning_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "mv_orphan")
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="mv_orphan.asc",
                ops=[{"op": "move_component", "reference": "R1", "x": 600, "y": 200}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        op = res.structuredContent["results"][0]
        assert any("old pin" in w for w in op.get("warnings", []))

    async def test_remove_orphan_warning_then_cleanup_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "rm_orphan")
        # Remove without cleanup: the wire left on R1's former pin is flagged.
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_orphan.asc",
                ops=[{"op": "remove_component", "reference": "R1"}],  # type: ignore[arg-type]
            ),
            asc_state,
        )
        op = res.structuredContent["results"][0]
        assert any("orphaned" in w for w in op.get("warnings", []))

    async def test_remove_cleanup_reports_deleted_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "rm_clean")
        res = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(
                path="rm_clean.asc",
                ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                    {"op": "remove_component", "reference": "R1", "cleanup_wires": True},  # type: ignore[arg-type]
                ],
            ),
            asc_state,
        )
        op = res.structuredContent["results"][0]
        assert op["deleted_wires"] >= 1
        assert "warnings" not in op


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
async def _build_name_wired_rc(name: str, state: SessionState, work_dir: Path) -> str:
    """Build an RC schematic wired by net label (one FLAG per pin), the way a
    label-based layout connects: R1(in,out), C1(out,0), V1(in,0). Returns the
    .asc filename. Pins are connected only by shared label name, not by wires,
    so trace_net must fold same-name FLAGs together.
    """
    asc = work_dir / f"{name}.asc"
    asc.write_text("Version 4\nSHEET 1 880 680\n")
    await handle_apply_schematic_ops(
        ApplySchematicOpsInput(
            path=asc.name,
            ops=[  # type: ignore[arg-type]  # pydantic validates dicts
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 128, "y": 128},
                {"op": "add_component", "reference": "C1", "symbol": "cap", "x": 384, "y": 128},
                {
                    "op": "add_component",
                    "reference": "V1",
                    "symbol": "voltage",
                    "x": 640,
                    "y": 128,
                },
                {"op": "add_net_label", "net": "in", "pin": "R1.1"},
                {"op": "add_net_label", "net": "out", "pin": "R1.2"},
                {"op": "add_net_label", "net": "out", "pin": "C1.1"},
                {"op": "add_net_label", "net": "0", "pin": "C1.2"},
                {"op": "add_net_label", "net": "in", "pin": "V1.+"},
                {"op": "add_net_label", "net": "0", "pin": "V1.-"},
            ],
        ),
        state,
    )
    return asc.name


@pytest.mark.asyncio
class TestTraceNet:
    async def test_name_based_net_on_label_wiring(self, asc_state: SessionState, work_dir: Path):
        # R1.1 is on net "in". V1.+ is also on "in" — they are at different
        # coordinates connected only by the shared label name.
        path = await _build_name_wired_rc("trace_rc", asc_state, work_dir)
        res = await handle_trace_net(TraceNetInput(path=path, pin="R1.1"), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        refs = {p["reference"] for p in sc["pins"]}
        assert refs == {"R1", "V1"}
        assert sc["is_shorted"] is False

    async def test_trace_by_net_name(self, asc_state: SessionState, work_dir: Path):
        # net:in matches one FLAG per pin (V1.+ and R1.1) — _resolve_pin would
        # refuse the ambiguity, but trace_net seeds from a match and name-merges.
        path = await _build_name_wired_rc("trace_byname", asc_state, work_dir)
        res = await handle_trace_net(TraceNetInput(path=path, pin="net:in"), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        assert {p["reference"] for p in sc["pins"]} == {"R1", "V1"}

    async def test_trace_by_missing_net_name_raises(self, asc_state: SessionState, work_dir: Path):
        path = await _build_name_wired_rc("trace_miss", asc_state, work_dir)
        with pytest.raises(NetlistError, match="not found"):
            await handle_trace_net(TraceNetInput(path=path, pin="net:nonexistent"), asc_state)

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


def _real_symbol_dir() -> str | None:
    """Locate the real LTspice .asy library (NOT the test fixtures), or None.

    Every other test in this module runs against the tiny fabricated fixture
    .asy files registered by the ``asc_symbols`` fixture. That means a
    regression that only manifests against the *real* LTspice symbol library —
    e.g. an add_component that throws on a real symbol, or a pin-geometry drift
    between fixtures and reality — is invisible to the whole suite. This finds
    the actual library so the smoke test below runs wherever LTspice symbols
    are installed (dev boxes, WSL) and skips cleanly on bare CI.
    """
    import os

    from ltspice_mcp.lib.wsl import get_ltspice_lib_paths, is_wsl

    candidates: list[str] = []
    env = os.environ.get("LTSPICE_MCP_SYMBOL_PATHS")
    if env:
        candidates.extend(env.split(os.pathsep))
    if is_wsl():
        candidates.extend(get_ltspice_lib_paths())
    for c in candidates:
        if (Path(c) / "res.asy").is_file():
            return c
    return None


_REAL_SYM = _real_symbol_dir()


@pytest.mark.skipif(_REAL_SYM is None, reason="real LTspice symbol library not installed")
@pytest.mark.asyncio
class TestAddComponentRealSymbols:
    """add_component against the REAL LTspice symbol library, not the fixtures.

    The fixture .asy files are minimal hand-written stand-ins; this exercises
    the actual symbol parse → SchematicComponent build → .asc save → reopen
    round-trip that the rest of the suite never touches.
    """

    @pytest.fixture
    def real_state(self, state_no_sim: SessionState, work_dir: Path):
        from spicelib import AscEditor

        from ltspice_mcp.lib import symbol_geometry

        saved_paths = AscEditor.custom_lib_paths
        saved_cache = AscEditor.symbol_cache
        saved_geo = dict(symbol_geometry._symbol_cache)
        AscEditor.custom_lib_paths = [_REAL_SYM]  # type: ignore[list-item]
        AscEditor.symbol_cache = {}
        symbol_geometry._symbol_cache.clear()
        try:
            yield state_no_sim
        finally:
            AscEditor.custom_lib_paths = saved_paths
            AscEditor.symbol_cache = saved_cache
            symbol_geometry._symbol_cache.clear()
            symbol_geometry._symbol_cache.update(saved_geo)

    async def test_add_real_symbols_round_trip(self, real_state: SessionState):
        await handle_create_schematic(
            CreateSchematicInput(name="real", overwrite=True), real_state
        )
        for ref, sym, x, y, val in [
            ("R1", "res", 100, 100, "1k"),
            ("C1", "cap", 300, 100, "1n"),
            ("M1", "nmos", 500, 100, "NMOS1"),
        ]:
            result = await handle_add_component(
                AddComponentInput(path="real.asc", reference=ref, symbol=sym, x=x, y=y, value=val),
                real_state,
            )
            assert f"Added {ref}" in _result_text(result)
            # Geometry comes from parsing the real .asy; empty pins = a broken parse.
            assert result.structuredContent["pins"]

    async def test_real_resistor_pins_are_a_b(self, real_state: SessionState):
        # The fixture res uses numeric pins 1/2; the real LTspice res uses A/B.
        # Guards against the suite silently drifting onto fabricated geometry.
        result = await handle_symbol_info(
            SymbolInfoInput(symbol="res", x=0, y=0, rotation="R0"), real_state
        )
        names = {p["name"] for p in result.structuredContent["absolute_pins"]}
        assert names == {"A", "B"}

    async def test_apply_ops_add_real_symbol(self, real_state: SessionState):
        await handle_create_schematic(
            CreateSchematicInput(name="real2", overwrite=True), real_state
        )
        op = {
            "op": "add_component",
            "reference": "R1",
            "symbol": "res",
            "x": 100,
            "y": 100,
            "value": "1k",
        }
        result = await handle_apply_schematic_ops(
            ApplySchematicOpsInput(path="real2.asc", ops=[op]),  # type: ignore[arg-type]
            real_state,
        )
        assert result.structuredContent["applied_count"] == 1
        assert result.structuredContent["failed_count"] == 0
