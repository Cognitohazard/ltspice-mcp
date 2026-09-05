"""Integration tests for .asc schematic editing using fixture symbols.

The ops, the route planner, the symbol-geometry layer and the post-op validation
pass are the live implementation behind ``edit_schematic``; the tests drive them
through the shared op runner (``tests/_asc_ops.py``) and then read the sheet back
off disk, so a claim about geometry is checked against what was actually written.
"""

from pathlib import Path

import pytest
from mcp.types import TextContent

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.schematic_ops import (
    build_on_wire_predicate,
    point_on_segment,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.inspect_tools import TraceNetInput, handle_trace_net
from tests._asc_ops import (
    add_component,
    add_net_label,
    apply_one,
    batch_view,
    blank_sheet_file,
    components_of,
    inspect_one,
    wire_pins,
)


def _result_text(result) -> str:
    """Extract text from a tool result's first content block, asserting it is text."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _sheet(name: str) -> Path:
    """A sheet path relative to the session's working directory.

    The op helpers take a real path; the tests name sheets the way a caller
    does, so this resolves the name against the fixture working dir.
    """
    return _WORK_DIR / name


_WORK_DIR = Path()


@pytest.fixture(autouse=True)
def _bind_work_dir(work_dir: Path):
    """Let ``_sheet`` resolve bare sheet names for the duration of a test."""
    global _WORK_DIR
    previous = _WORK_DIR
    _WORK_DIR = work_dir
    yield
    _WORK_DIR = previous


def _sheet_facts(state: SessionState, name: str) -> dict:
    """Components, wires and labels as written to the sheet on disk.

    Reads the .asc itself rather than any in-memory model, so an assertion about
    what an op produced is an assertion about the file a simulator would open.
    """
    path = _sheet(name) if not Path(name).is_absolute() else Path(name)
    components: list[dict] = []
    wires: list[dict] = []
    labels: list[dict] = []
    directives: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "WIRE" and len(parts) >= 5:
            wires.append(
                {
                    "x1": int(parts[1]),
                    "y1": int(parts[2]),
                    "x2": int(parts[3]),
                    "y2": int(parts[4]),
                }
            )
        elif parts[0] == "FLAG" and len(parts) >= 4:
            labels.append({"x": int(parts[1]), "y": int(parts[2]), "text": parts[3]})
        elif parts[0] == "TEXT" and "!" in line:
            # "TEXT <x> <y> <align> <size> !<directive>"
            directives.append(line.split("!", 1)[1])
        elif parts[0] == "SYMATTR" and len(parts) >= 3 and parts[1] == "InstName":
            components.append({"reference": parts[2]})
    return {
        "components": components,
        "wires": wires,
        "labels": labels,
        "directives": directives,
        "wire_count": len(wires),
    }


def _copy_file(src: Path, dst: Path) -> None:
    """Sync byte-copy — keeps blocking pathlib I/O out of async test bodies."""
    dst.write_bytes(src.read_bytes())


# Relocated regression coverage from a retired test module.
def _read_bytes(p: Path) -> bytes:
    """Sync file read (keeps blocking pathlib I/O out of async test bodies)."""
    return p.read_bytes()


def _directive_anchors(content: bytes) -> list[tuple[int, int]]:
    """(x,y) anchor of every directive (``!``) TEXT record in an .asc."""
    out: list[tuple[int, int]] = []
    for line in content.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "TEXT" and "!" in line:
            out.append((int(parts[1]), int(parts[2])))
    return out


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
class TestEditDirectiveCommentKind:
    async def test_add_comment_via_edit_directive(self, asc_state: SessionState, asc_file: Path):
        """Free-text annotations now go through ``edit_directive`` with
        ``kind='comment'`` instead of the old ``add_text`` tool."""
        result = apply_one(
            asc_state,
            asc_file,
            {
                "op": "add_directive",
                "instruction": "Test note",
                "kind": "comment",
                "x": 100,
                "y": 200,
            },
        )
        assert result["instruction"] == "Test note"

    async def test_add_directive_honors_placement(self, asc_state: SessionState, asc_file: Path):
        """x/y/size on the .asc DIRECTIVE branch must place the directive at
        the given coordinates — previously only the comment branch read
        them, and spicelib's add_instruction silently picked its own spot
        and font size."""
        result = apply_one(
            asc_state,
            asc_file,
            {"op": "add_directive", "instruction": ".tran 5m", "x": 320, "y": 240, "size": 3},
        )
        assert result["op"] == "add_directive"
        content = _read_bytes(asc_file)
        # LTspice TEXT record: "TEXT <x> <y> <align> <size> !<directive>"
        assert b"!.tran 5m" in content
        line = next(ln for ln in content.splitlines() if b"!.tran 5m" in ln)
        assert b"320 240" in line
        assert b" 3 " in line

    async def test_multiline_directive_escaped_to_one_text_record(
        self, asc_state: SessionState, asc_file: Path
    ):
        """Raw newlines in directive text must be stored as LTspice's literal
        \\n escapes — a real newline splits the TEXT record and corrupts the
        .asc (surfacing later as an unrelated 'Primitive not supported')."""
        result = apply_one(
            asc_state,
            asc_file,
            {
                "op": "add_directive",
                "instruction": ".options reltol=1e-4\n.ic V(out)=0",
                "x": 600,
                "y": 600,
            },
        )
        assert result["op"] == "add_directive"
        content = _read_bytes(asc_file)
        line = next(ln for ln in content.splitlines() if b"!.options reltol=1e-4" in ln)
        assert b"!.options reltol=1e-4\\n.ic V(out)=0" in line
        # The file still parses cleanly.
        from spicelib import AscEditor

        AscEditor(str(asc_file))

    async def test_stacked_directives_auto_shift(self, asc_state: SessionState, asc_file: Path):
        # Two add_directive ops without coordinates both default to (16,16);
        # the auto-declutter must nudge the second down so no two directives
        # share an anchor (which would render them on top of each other).
        result = batch_view(
            asc_state,
            asc_file,
            [
                {"op": "add_directive", "instruction": ".tran 5m"},  # type: ignore[list-item]
                {"op": "add_directive", "instruction": ".ac dec 100 1 1meg"},  # type: ignore[list-item]
            ],
            stop_on_error=True,
        )
        assert result["saved"] is True, result
        anchors = _directive_anchors(_read_bytes(asc_file))
        assert len(anchors) >= 2
        assert len(anchors) == len(set(anchors))  # every directive anchor distinct

    async def test_standalone_directives_no_coords_do_not_stack(
        self, asc_state: SessionState, asc_file: Path
    ):
        # The standalone edit_directive path (no coordinates) routes through
        # spicelib's placement, which staggers each directive below the last;
        # repeated adds must not land on the same anchor. (.tran and .four
        # coexist — .four is not an analysis directive that .tran replaces.)
        for instr in (".tran 5m", ".four 1k V(out)"):
            apply_one(asc_state, asc_file, {"op": "add_directive", "instruction": instr})
        anchors = _directive_anchors(_read_bytes(asc_file))
        assert len(anchors) >= 2
        assert len(anchors) == len(set(anchors))  # no two directives share an anchor

    async def test_stacked_directives_detected(self, asc_state: SessionState, work_dir: Path):
        # A hand-authored .asc with two directives at the same anchor (bypasses
        # the auto-shift) must surface a stacked_directive advisory.
        from ltspice_mcp.lib.schematic_ops import (
            get_asc_editor,
            post_op_warnings,
        )

        stacked = work_dir / "stacked.asc"
        stacked.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "TEXT 16 16 Left 2 !.tran 5m\n"
            "TEXT 16 16 Left 2 !.ac dec 100 1 1meg\n"
        )
        warns = post_op_warnings(get_asc_editor(stacked, asc_state))
        stacked_w = [w for w in warns if w["kind"] == "stacked_directive"]
        assert len(stacked_w) == 1
        assert stacked_w[0]["count"] == 2
        assert (stacked_w[0]["x"], stacked_w[0]["y"]) == (16, 16)

    async def test_remove_spans_directive_and_comment(
        self, asc_state: SessionState, asc_file: Path
    ):
        """``remove`` should hit comments too — previously you could
        ``add_text`` a stray ``;.foo`` line and ``edit_directive remove``
        couldn't touch it."""
        apply_one(
            asc_state,
            asc_file,
            {"op": "add_directive", "instruction": "zap me", "kind": "comment"},
        )
        apply_one(asc_state, asc_file, {"op": "remove_directive", "instruction": "regex:zap me"})
        # Comment should be gone — re-removing yields no error since the
        # underlying spicelib calls are tolerant of misses. ASC files may
        # contain Latin-1 µ characters, so read raw bytes and replace.
        text = asc_file.read_bytes().decode("utf-8", errors="replace")  # noqa: ASYNC240
        assert "zap me" not in text


@pytest.mark.asyncio
class TestWirePins:
    async def test_diagonal_rejected(self, asc_state: SessionState, asc_file: Path):
        # First add a unique net label, then try a diagonal route to it
        add_net_label(asc_state, asc_file, "X", x=100, y=200)
        with pytest.raises(NetlistError, match="not orthogonal"):
            wire_pins(asc_state, asc_file, "net:filtered", "net:X", waypoints=[])

    async def test_multiple_ground_labels_error(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Multiple '0'") as exc_info:
            wire_pins(asc_state, asc_file, "net:filtered", "net:0")
        # Guidance must reference the actual ambiguous net, not a canned example.
        msg = str(exc_info.value)
        assert "add_net_label op of edit_schematic (net='0'" in msg
        assert "M3.S" not in msg

    async def test_multiple_label_error_names_actual_net(
        self, asc_state: SessionState, asc_file: Path
    ):
        # Two same-name labels on a non-ground net: the ambiguity guidance
        # must name that net dynamically.
        add_net_label(asc_state, asc_file, "SIG", x=100, y=200)
        add_net_label(asc_state, asc_file, "SIG", x=300, y=200)
        with pytest.raises(NetlistError, match="Multiple 'SIG'") as exc_info:
            wire_pins(asc_state, asc_file, "net:filtered", "net:SIG")
        assert "add_net_label op of edit_schematic (net='SIG'" in str(exc_info.value)

    async def test_invalid_pin_format(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Invalid pin reference"):
            wire_pins(asc_state, asc_file, "badformat", "net:0")

    async def test_unknown_component(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            wire_pins(asc_state, asc_file, "ZZZ.A", "net:0")

    async def test_missing_net_label(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Net label"):
            wire_pins(asc_state, asc_file, "net:nonexistent", "net:0")

    async def test_pin_unknown(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="not found"):
            wire_pins(asc_state, asc_file, "R1.ZZ", "net:0")

    async def test_same_instance_tie_refused(self, asc_state: SessionState, work_dir: Path):
        # A direct wire between two pins of ONE component is dropped by LTspice
        # at netlist time (verified against LTspice 26 -netlist: both pins stay
        # on separate NC nodes), so we must refuse rather than report a tie that
        # won't exist electrically. Fixture res pins: 1=(200,152), 2=(200,248).
        asc = work_dir / "self_tie.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("self_tie.asc"), "R1", "res", 200, 200)
        with pytest.raises(NetlistError, match="same-instance wire") as exc_info:
            wire_pins(asc_state, _sheet("self_tie.asc"), "R1.1", "R1.2")
        msg = str(exc_info.value)
        # Names the actual instance/pins and both remedies (waypoint / net label).
        assert "R1.1" in msg and "R1.2" in msg
        assert "waypoint" in msg and "net label" in msg
        # Nothing was written — the refusal happens before any save.
        assert _wire_segments(asc) == []

    async def test_same_instance_tie_allowed_through_waypoint(
        self, asc_state: SessionState, work_dir: Path
    ):
        # The documented remedy must actually work: bending the tie through a
        # waypoint means no single segment joins the two pins directly, so
        # LTspice keeps it. This is the escape hatch the refusal points at.
        asc = work_dir / "self_tie_wp.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("self_tie_wp.asc"), "R1", "res", 200, 200)
        result = wire_pins(
            asc_state,
            _sheet("self_tie_wp.asc"),
            "R1.1",
            "R1.2",
            waypoints=[{"x": 300, "y": 152}, {"x": 300, "y": 248}],
        )
        assert result is not None
        assert result["wire_count"] == 3
        # No segment directly joins the two pins, so none is dropped by LTspice.
        assert len(_wire_segments(asc)) == 3

    async def test_same_instance_collinear_waypoint_refused(
        self, asc_state: SessionState, work_dir: Path
    ):
        # A waypoint that stays COLLINEAR with the two pins does NOT rescue the
        # tie: LTspice merges the in-line segments back into one straight wire
        # and drops it (verified against LTspice 26). Detection collinear-merges
        # the route, so this must still be refused. Fixture res pins are on
        # x=200; (200,200) is their in-line midpoint.
        asc = work_dir / "self_tie_collinear.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("self_tie_collinear.asc"), "R1", "res", 200, 200)
        with pytest.raises(NetlistError, match="same-instance wire") as exc_info:
            wire_pins(
                asc_state,
                _sheet("self_tie_collinear.asc"),
                "R1.1",
                "R1.2",
                waypoints=[{"x": 200, "y": 200}],
            )
        assert "out of line" in str(exc_info.value).lower()
        assert _wire_segments(asc) == []

    async def test_cross_instance_wire_not_refused(self, asc_state: SessionState, work_dir: Path):
        # Guard against over-refusal: a wire between pins of TWO DIFFERENT
        # instances is a normal net LTspice keeps, and must still be allowed.
        asc = work_dir / "cross.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("cross.asc"), "R1", "res", 200, 200)
        add_component(asc_state, _sheet("cross.asc"), "R2", "res", 200, 400)
        result = wire_pins(asc_state, _sheet("cross.asc"), "R1.2", "R2.1")
        assert result is not None
        assert result["wire_count"] == 1

    async def test_route_over_own_pin_interior_refused_as_same_instance(
        self, asc_state: SessionState, work_dir: Path
    ):
        # Routing R1.1 to a DIFFERENT instance (R2.1) via a waypoint that takes the
        # first segment straight down through R1.2 puts R1's own second pin on that
        # segment's interior. LTspice splits the wire at that pin, so the R1.1->R1.2
        # sub-run is a dropped same-instance tie. The whole-route endpoints are
        # cross-instance (R1.1/R2.1), so only the interior-pin cut exposes the tie —
        # and because same-instance is checked before pin-collision, the refusal
        # names it as a same-instance wire, not a "passes through" collision.
        asc = work_dir / "route_over_own_pin.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("route_over_own_pin.asc"), "R1", "res", 200, 200)
        add_component(asc_state, _sheet("route_over_own_pin.asc"), "R2", "res", 400, 400)
        # R1.1=(200,152), R1.2=(200,248), R2.1=(400,352). Route:
        # (200,152)->(200,352)->(400,352); the first leg passes over R1.2.
        with pytest.raises(NetlistError, match="same-instance wire") as exc_info:
            wire_pins(
                asc_state,
                _sheet("route_over_own_pin.asc"),
                "R1.1",
                "R2.1",
                waypoints=[{"x": 200, "y": 352}],
            )
        msg = str(exc_info.value)
        assert "R1.1" in msg and "R1.2" in msg
        assert _wire_segments(asc) == []

    async def test_apply_ops_same_instance_wire_refused(
        self, asc_state: SessionState, work_dir: Path
    ):
        # The apply_schematic_ops wire op shares _plan_connect_route, so the
        # refusal must reach the batch surface too.
        asc = work_dir / "self_tie_ops.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("self_tie_ops.asc"), "R1", "res", 200, 200)
        result = batch_view(
            asc_state,
            _sheet("self_tie_ops.asc"),
            [
                # pydantic validates dicts
                {"op": "wire_pins", "from_pin": "R1.1", "to_pin": "R1.2"},  # type: ignore[arg-type]
            ],
            stop_on_error=True,
        )
        data = result
        assert data is not None
        assert data["saved"] is False
        assert data["results"][0]["ok"] is False
        assert "same-instance wire" in data["results"][0]["error"]


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
    """add_component(rotation=...) -> cached editor -> resolve_pin -> wire_pins
    must agree on absolute pin coordinates for every rotation AND mirror.
    Wire endpoints on disk are checked against hand-computed positions, so a
    sign error in any orientation transform fails here — not just an
    inconsistency between add_component and wire_pins."""

    @pytest.mark.parametrize("rotation", sorted(NMOS_PIN_POSITIONS))
    async def test_pin_map_and_wire_endpoint(
        self, asc_state: SessionState, work_dir: Path, rotation: str
    ):
        asc = work_dir / "orient.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")

        added = add_component(
            asc_state, _sheet("orient.asc"), "M1", "nmos", 400, 200, rotation=rotation
        )
        reported = {p["name"]: (p["x"], p["y"]) for p in added["pins"]}
        assert reported == NMOS_PIN_POSITIONS[rotation]

        # Fixed second component, far enough from M1 that no route below can
        # collide with its pins. Fixture res pins: 1=(0,-48) -> R9.1=(700,452).
        add_component(asc_state, _sheet("orient.asc"), "R9", "res", 700, 500)

        # wire_pins re-resolves M1.G from the cached editor's stored placement,
        # so the wire endpoint proves the rotation survived the round trip.
        gx, gy = NMOS_PIN_POSITIONS[rotation]["G"]
        connected = wire_pins(
            asc_state, _sheet("orient.asc"), "M1.G", "R9.1", waypoints=[{"x": gx, "y": 452}]
        )
        sc = connected
        assert sc["from_pin"] == "M1.G"
        assert sc["to_pin"] == "R9.1"

        # Re-read the file from disk: the persisted wire must start at the
        # hand-computed absolute G coordinate and land on R9.1. This is the
        # load-bearing check — it proves the rotation survived placement, the
        # cached editor, and the route planner's own pin resolution.
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

        added = add_component(asc_state, asc, "X1", symbol, 400, 300)
        pins = {p["name"]: (p["x"], p["y"]) for p in added["pins"]}
        # The symbol-geometry layer must report every terminal of the class. This
        # extends the nmos-only orientation coverage to the 2-terminal active and
        # 4-terminal controlled-source classes, so a PIN-parser regression that
        # truncated the terminal list for one of them fails here.
        assert set(pins) == set(pin_names), pins

        # Wire the LAST terminal (a non-positional name — N / NC- — on the
        # 4-terminal sources, which the positional-pin nmos orientation test
        # never reaches) to a passive load. This is the load-bearing part: the
        # terminal must resolve by NAME through wire_pins's pin lookup and the wire
        # must persist to disk at the geometry coordinate — not merely be
        # reported in memory. The load sits collinear and outward from the pin
        # (left if the pin is on the body's left half, else right), so the single
        # straight segment leaves the body without crossing another terminal
        # (every pin of these symbols has a unique y).
        name = pin_names[-1]
        px, py = pins[name]
        out_x = px + (300 if px >= 400 else -300)
        add_component(
            asc_state, asc, "RL", "res", out_x, py + 48
        )  # res pin 1 = (0,-48) offset -> (out_x, py), collinear with X1.{name}
        wire_pins(asc_state, asc, f"X1.{name}", "RL.1")
        # Re-read from disk: the named terminal resolved and the wire persisted.
        assert _has_segment(_wire_segments(asc), (px, py), (out_x, py)), (name, asc.read_text())


@pytest.mark.asyncio
class TestWirePinsPersistsWires:
    """wire_pins's success path must actually write WIRE records to disk —
    the rejection-path tests above only prove it validates."""

    async def test_wire_written_and_persisted(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "wire_persist.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        add_component(asc_state, _sheet("wire_persist.asc"), "R1", "res", 200, 200)
        add_component(asc_state, _sheet("wire_persist.asc"), "R2", "res", 200, 400)
        before = _wire_segments(asc)
        assert before == []  # add_component places no wires

        result = wire_pins(asc_state, _sheet("wire_persist.asc"), "R1.2", "R2.1")
        assert (result["from_pin"], result["to_pin"]) == ("R1.2", "R2.1")
        sc = result
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

        blank_sheet_file(asc_state, "value2")
        add_component(
            asc_state,
            _sheet("value2.asc"),
            "M1",
            "nmos",
            200,
            200,
            value="NMOS_VTH04",
            attributes={"Value2": "tag1", "SpiceLine": "W=10u L=0.5u"},
        )

        result = await components_of(asc_state, _sheet("value2.asc"))
        comps = result["components"]  # type: ignore[index]
        m1 = next(c for c in comps if c["reference"] == "M1")
        # Value field is the Value SYMATTR alone, not "NMOS_VTH04 tag1".
        assert m1["value"] == "NMOS_VTH04"
        # Value2 still appears under attributes.
        assert m1["attributes"]["Value2"] == "tag1"

    async def test_single_ref_lookup_excludes_value2(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:

        blank_sheet_file(asc_state, "value2_single")
        add_component(
            asc_state,
            _sheet("value2_single.asc"),
            "M1",
            "nmos",
            200,
            200,
            value="NMOS_VTH04",
            attributes={"Value2": "tag1"},
        )

        data = await components_of(asc_state, _sheet("value2_single.asc"))
        m1 = next(c for c in data["components"] if c["reference"] == "M1")
        # Value field is the Value SYMATTR alone, not "NMOS_VTH04 tag1".
        assert m1["value"] == "NMOS_VTH04"


@pytest.mark.asyncio
class TestEmptyAttributeHandling:
    """LTspice's format has no empty-SYMATTR-value representation (a 2-token
    line bricks the file on the next parse). CREATION paths (add_component)
    reject an empty value up front; set_component_attribute treats an empty
    value as CLEAR (removes the SYMATTR line), except InstName."""

    async def test_empty_attribute_raises(self, asc_state: SessionState, asc_file: Path):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="empty value"):
            add_component(
                asc_state, asc_file, "M_bad", "res", 600, 600, attributes={"SpiceModel": ""}
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
            add_component(asc_state, asc_file, "RX", "res", 600, 600, value=bad_value)
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        _sheet_facts(asc_state, asc_file.name)

    async def test_add_component_unknown_attribute_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        # A typo'd attribute name (Val for Value) would silently no-op at export;
        # add_component now refuses it the same way set_component_attribute does.
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="Unknown attribute"):
            add_component(asc_state, asc_file, "RX", "res", 600, 600, attributes={"Val": "10k"})
        assert asc_file.read_bytes() == original  # noqa: ASYNC240

    async def test_apply_ops_add_component_empty_value_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        result = batch_view(
            asc_state,
            asc_file,
            [
                {  # type: ignore[arg-type]
                    "op": "add_component",
                    "reference": "RX",
                    "symbol": "res",
                    "x": 600,
                    "y": 600,
                    "value": "",
                },
            ],
            stop_on_error=True,
        )
        data = result
        assert data is not None
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "empty value" in data["results"][0]["error"]
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        _sheet_facts(asc_state, asc_file.name)

    async def test_set_component_attribute_empty_value_clears(
        self, asc_state: SessionState, asc_file: Path
    ):
        # An empty value means CLEAR: the SYMATTR line is removed (LTspice's
        # format has no empty-value representation — writing a 2-token
        # "SYMATTR Value " line bricks the file on the next parse). The .asc
        # must stay readable afterwards.
        result = apply_one(
            asc_state,
            asc_file,
            {
                "op": "set_component_attribute",
                "reference": "R1",
                "attribute": "Value",
                "value": "",
            },
        )
        assert result["ok"] is True
        from spicelib import AscEditor

        assert "Value" not in AscEditor(str(asc_file)).get_component("R1").attributes
        _sheet_facts(asc_state, asc_file.name)

    async def test_instname_cannot_be_cleared(self, asc_state: SessionState, asc_file: Path):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        with pytest.raises(NetlistError, match="InstName"):
            apply_one(
                asc_state,
                asc_file,
                {
                    "op": "set_component_attribute",
                    "reference": "R1",
                    "attribute": "InstName",
                    "value": "",
                },
            )
        assert asc_file.read_bytes() == original  # noqa: ASYNC240

    async def test_apply_ops_set_component_attribute_empty_value_clears(
        self, asc_state: SessionState, asc_file: Path
    ):
        from spicelib import AscEditor

        result = batch_view(
            asc_state,
            asc_file,
            [
                {  # type: ignore[arg-type]
                    "op": "set_component_attribute",
                    "reference": "R1",
                    "attribute": "Value",
                    "value": "",
                },
            ],
            stop_on_error=True,
        )
        data = result
        assert data is not None
        assert data["saved"] is True
        assert "Value" not in AscEditor(str(asc_file)).get_component("R1").attributes
        _sheet_facts(asc_state, asc_file.name)

    async def test_apply_ops_instname_clear_rejected(
        self, asc_state: SessionState, asc_file: Path
    ):
        original = asc_file.read_bytes()  # noqa: ASYNC240
        result = batch_view(
            asc_state,
            asc_file,
            [
                {  # type: ignore[arg-type]
                    "op": "set_component_attribute",
                    "reference": "R1",
                    "attribute": "InstName",
                    "value": "",
                },
            ],
            stop_on_error=True,
        )
        data = result
        assert data is not None
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "InstName" in data["results"][0]["error"]
        assert asc_file.read_bytes() == original  # noqa: ASYNC240


@pytest.mark.asyncio
class TestSetComponentValueBehavioralSource:
    """Regression: on a .asc B-source, a value like 'V=V(in)*2' used to be
    split as a KEY=VALUE parameter and written to SpiceLine while the old
    expression stayed in Value — the netlisted B-line carried both
    expressions ("No such node") behind a success message."""

    async def test_bsource_expression_replaces_value_slot(
        self, asc_state: SessionState, work_dir: Path
    ):
        asc = work_dir / "bsrc.asc"
        asc.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL bv 100 100 R0\n"
            "SYMATTR InstName B1\n"
            "SYMATTR Value V=1\n"
        )
        result = apply_one(
            asc_state, asc, {"op": "set_component_value", "reference": "B1", "value": "V=V(in)*2"}
        )
        assert result["reference"] == "B1"
        content = asc.read_text()
        assert "SYMATTR Value V=V(in)*2" in content
        assert "SpiceLine" not in content
        assert "Value V=1\n" not in content


@pytest.mark.asyncio
class TestSetComponentValueCreatesMissingValue:
    """Regression: set_component_value on a component added without a Value slot
    used to fail 'Component(s) not found' (the component existed). It must create
    the Value line — symmetric with add_component(value=)."""

    async def test_standalone_set_value_creates_missing_value_line(
        self, asc_state: SessionState, asc_file: Path
    ):
        from spicelib import AscEditor

        add_component(asc_state, asc_file, "R9", "res", 400, 400)
        result = apply_one(
            asc_state, asc_file, {"op": "set_component_value", "reference": "R9", "value": "22k"}
        )
        assert result["reference"] == "R9"
        assert str(AscEditor(str(asc_file)).get_component_value("R9")) == "22k"

    async def test_apply_ops_set_value_after_valueless_add(
        self, asc_state: SessionState, asc_file: Path
    ):
        from spicelib import AscEditor

        result = batch_view(
            asc_state,
            asc_file,
            [  # type: ignore[arg-type]
                {
                    "op": "add_component",
                    "reference": "R8",
                    "symbol": "res",
                    "x": 500,
                    "y": 400,
                },
                {"op": "set_component_value", "reference": "R8", "value": "33k"},
            ],
            stop_on_error=True,
        )
        data = result
        assert data is not None
        assert data["saved"] is True
        assert data["failed_count"] == 0
        assert str(AscEditor(str(asc_file)).get_component_value("R8")) == "33k"


@pytest.mark.asyncio
class TestEditingAscRollback:
    """Uncaught exceptions inside _editing_asc must invalidate the
    cached editor so a later read doesn't see dirty in-memory mutations,
    and the file on disk must remain intact."""

    async def test_uncaught_exception_after_mutation_invalidates_cache(
        self, asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Inject a failure after add_component has already mutated the
        # editor in-memory but before save: wrap create_component so the real
        # in-memory mutation runs, then raise — simulating a spicelib internal
        # error mid-edit, after the editor is dirty but before the editing
        # context saves.
        from ltspice_mcp.lib import schematic_ops as circuit_mod

        original = asc_file.read_bytes()  # noqa: ASYNC240
        boom_calls = {"n": 0}
        real_create = circuit_mod.create_component

        def boom(*a, **kw):
            real_create(*a, **kw)  # do the real in-memory mutation
            boom_calls["n"] += 1
            raise RuntimeError("injected post-op failure")

        monkeypatch.setattr(circuit_mod, "create_component", boom)

        with pytest.raises(RuntimeError, match="injected"):
            add_component(asc_state, asc_file, "R_uncommitted", "res", 700, 700)

        # The injection fired (sanity).
        assert boom_calls["n"] == 1
        # File on disk is unchanged — save runs only on the success path.
        assert asc_file.read_bytes() == original  # noqa: ASYNC240
        # Cache eviction means a fresh read doesn't see R_uncommitted.
        monkeypatch.undo()
        result = await components_of(asc_state, asc_file)
        assert "R_uncommitted" not in result


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
            add_component(asc_state, asc_file, "R_aborted_save", "res", 600, 600)

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
            add_component(asc_state, asc_file, "R_uncommitted", "res", 600, 600)

        # Restore real save so the follow-up read works.
        monkeypatch.undo()

        # The component must NOT be visible — cache was evicted, fresh
        # read from disk shows the pre-failure state.
        result = await components_of(asc_state, asc_file)
        assert "R_uncommitted" not in result


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestSetAttributeAllowlist:
    """set_component_attribute rejects unknown attribute names."""

    async def test_rejects_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Unknown attribute"):
            apply_one(
                asc_state,
                asc_file,
                {
                    "op": "set_component_attribute",
                    "reference": "R1",
                    "attribute": "NotARealAttr",
                    "value": "x",
                },
            )

    async def test_suggests_canonical_for_case_typo(self, asc_state: SessionState, asc_file: Path):
        with pytest.raises(NetlistError, match="Did you mean 'SpiceLine'"):
            apply_one(
                asc_state,
                asc_file,
                {
                    "op": "set_component_attribute",
                    "reference": "R1",
                    "attribute": "spiceline",
                    "value": "x",
                },
            )

    async def test_accepts_spiceline(self, asc_state: SessionState, asc_file: Path):
        # Sanity: the canonical name still works.
        result = apply_one(
            asc_state,
            asc_file,
            {
                "op": "set_component_attribute",
                "reference": "R1",
                "attribute": "SpiceLine",
                "value": "tc=10ppm",
            },
        )
        assert result["ok"] is True


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestFloatingLabelWarning:
    """add_net_label warns on labels placed away from any wire/pin."""

    async def test_warns_on_floating(self, asc_state: SessionState, asc_file: Path):
        result = add_net_label(asc_state, asc_file, "VCC_floating", x=10, y=10)
        assert any("floating" in w.lower() for w in result["warnings"]), result


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestNetConflictInWirePins:
    """wire_pins detects shorts between two named nets."""

    async def test_refuses_named_net_short(self, asc_state: SessionState):
        # Build a clean schematic with two resistors on disjoint named nets,
        # then try to wire them together.

        blank_sheet_file(asc_state, "net_conflict_test")
        add_component(asc_state, _sheet("net_conflict_test.asc"), "R1", "res", 100, 100)
        add_component(asc_state, _sheet("net_conflict_test.asc"), "R2", "res", 300, 100)
        # The test fixture's stripped 'res' symbol uses numeric pin names.
        add_net_label(asc_state, _sheet("net_conflict_test.asc"), "LEFT", pin="R1.1")
        add_net_label(asc_state, _sheet("net_conflict_test.asc"), "RIGHT", pin="R2.1")
        with pytest.raises(NetlistError, match="Net-label conflict"):
            wire_pins(asc_state, _sheet("net_conflict_test.asc"), "R1.1", "R2.1")


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestRemoveComponentNoFalseOrphans:
    """remove_component doesn't flag wires belonging to other components."""

    async def test_other_component_pin_not_flagged(self, asc_state: SessionState, asc_file: Path):
        # Add a second resistor whose pin coincides with R1's existing wire.
        # When we remove R2, the wire connecting R1 stays — and our orphan
        # detector should NOT flag it.
        add_component(
            asc_state,
            asc_file,
            "R2",
            "res",
            128,
            112,  # same coords as R1 — pins overlap
            value="2k",
            rotation="R90",
        )
        result = apply_one(asc_state, asc_file, {"op": "remove_component", "reference": "R2"})
        # The remaining R1's wires shouldn't be flagged as orphans.
        assert "orphaned" not in result


# Relocated regression coverage from a retired test module.
@pytest.mark.asyncio
class TestApplySchematicOps:
    """apply_schematic_ops batches add/wire_pins/label/directive."""

    async def test_add_component_result_includes_placed_geometry_and_overlap_warnings(
        self, asc_state: SessionState
    ):

        blank_sheet_file(asc_state, "batch_geometry")
        result = batch_view(
            asc_state,
            _sheet("batch_geometry.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
                    "y": 100,
                },
            ],
            stop_on_error=True,
        )

        data = result
        assert data is not None
        added = data["results"][1]
        assert added["pins"]
        assert added["bounding_box"] == {"x": 84, "y": 52, "width": 32, "height": 96}
        assert added["warnings"] == ["Overlaps R1 bounding box"]

    async def test_basic_transaction(self, asc_state: SessionState, work_dir: Path):

        blank_sheet_file(asc_state, "batch_demo")

        result = batch_view(
            asc_state,
            _sheet("batch_demo.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
            stop_on_error=True,
        )
        data = result
        assert data["applied_count"] == 3
        assert data["failed_count"] == 0
        assert data["saved"] is True

    async def test_continue_on_error_persists_partial(
        self, asc_state: SessionState, work_dir: Path
    ):

        blank_sheet_file(asc_state, "batch_partial")
        result = batch_view(
            asc_state,
            _sheet("batch_partial.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
        )
        data = result
        assert data["applied_count"] == 2
        assert data["failed_count"] == 1
        assert data["saved"] is True


@pytest.mark.asyncio
class TestRemoveWireAndNetLabelOps:
    """remove_wire / remove_net_label apply_schematic_ops ops."""

    async def test_remove_wire_by_endpoints_and_label_by_pin_and_xy(
        self, asc_state: SessionState, work_dir: Path
    ):

        blank_sheet_file(asc_state, "rm_ops")
        # Build R1 + C1, wire R1.2 → C1.1, label R1.1 by pin, and place a
        # second label at an explicit coordinate.
        build = batch_view(
            asc_state,
            _sheet("rm_ops.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
                {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "C1.1"},
                {"op": "add_net_label", "net": "in", "pin": "R1.1"},
                {"op": "add_net_label", "net": "spare", "x": 512, "y": 512},
            ],
            stop_on_error=True,
        )
        assert build["saved"] is True

        # read_circuit must expose wire segments for discovery/removal.
        read = _sheet_facts(asc_state, "rm_ops.asc")
        rsc = read
        assert rsc["wires"], "read_circuit should list wire segments"
        wire = rsc["wires"][0]
        # Label coordinates for the by-pin removal target.
        in_label = next(lbl for lbl in rsc["labels"] if lbl["text"] == "in")

        res = batch_view(
            asc_state,
            _sheet("rm_ops.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
            stop_on_error=True,
        )
        data = res
        assert data["saved"] is True
        assert data["failed_count"] == 0
        # Each op reports what it removed.
        by_op = {r["op"]: r for r in data["results"]}
        assert by_op["remove_wire"]["removed"] == 1
        # The by-pin removal must land on the "in" label coordinate.
        assert by_op["remove_net_label"]["ok"] is True

        read2 = _sheet_facts(asc_state, "rm_ops.asc")
        rsc2 = read2
        assert rsc2["wire_count"] == 0
        assert not rsc2["wires"]
        remaining = {lbl["text"] for lbl in rsc2["labels"]}
        assert "in" not in remaining
        assert "spare" not in remaining
        # Sanity: the removed label coordinate is gone.
        assert not any(
            lbl["x"] == in_label["x"] and lbl["y"] == in_label["y"] for lbl in rsc2["labels"]
        )

    async def _wired_pair(self, asc_state: SessionState, name: str) -> dict:
        """R1.2 wired to C1.1 — one connection, drawn once."""

        blank_sheet_file(asc_state, name)
        batch_view(
            asc_state,
            _sheet(f"{name}.asc"),
            [  # type: ignore[arg-type]
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
                {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "C1.1"},
            ],
            stop_on_error=True,
        )
        read = _sheet_facts(asc_state, f"{name}.asc")
        return read["wires"][0]

    async def test_removing_a_duplicated_connection_is_refused_by_the_pin_it_would_float(
        self, asc_state: SessionState, work_dir: Path
    ):
        """ "Delete the duplicate" and "delete the connection" are the same
        request when a segment is drawn twice, and one measured session lost a
        cap to it. Refuse, and name the pin that would be left hanging."""
        wire = await self._wired_pair(asc_state, "dup_load_bearing")
        path = work_dir / "dup_load_bearing.asc"
        path.write_text(
            path.read_text() + f"WIRE {wire['x1']} {wire['y1']} {wire['x2']} {wire['y2']}\n"
        )

        res = batch_view(
            asc_state,
            _sheet("dup_load_bearing.asc"),
            [
                {  # type: ignore[list-item]
                    "op": "remove_wire",
                    "x1": wire["x1"],
                    "y1": wire["y1"],
                    "x2": wire["x2"],
                    "y2": wire["y2"],
                }
            ],
            stop_on_error=True,
        )
        data = res
        assert data is not None
        assert data["saved"] is False
        error = data["results"][0]["error"]
        assert "2 copies" in error
        assert "split the net" in error
        assert "R1.2" in error or "C1.1" in error

        after = _sheet_facts(asc_state, "dup_load_bearing.asc")
        assert after["wire_count"] == 2, "the refusal must leave the sheet alone"

    async def test_removing_a_duplicated_bridge_between_wired_stubs_is_refused(
        self, asc_state: SessionState, work_dir: Path
    ):
        """The cut that hurts can leave every pin still touching a wire: two
        stubs, each attached to its pin, joined only by a duplicated bridge.
        A floating-pin scan blesses that removal; the net partition must not."""
        wire = await self._wired_pair(asc_state, "dup_bridge")
        x = int(wire["x1"])
        lo, hi = sorted((int(wire["y1"]), int(wire["y2"])))
        m1, m2 = lo + 32, lo + 64
        path = work_dir / "dup_bridge.asc"
        original_line = f"WIRE {wire['x1']} {wire['y1']} {wire['x2']} {wire['y2']}"
        rebuilt = path.read_text().replace(
            original_line,
            f"WIRE {x} {lo} {x} {m1}\n"
            f"WIRE {x} {m1} {x} {m2}\n"
            f"WIRE {x} {m1} {x} {m2}\n"
            f"WIRE {x} {m2} {x} {hi}",
        )
        assert rebuilt != path.read_text(), "helper wire line not found to rewrite"
        path.write_text(rebuilt)

        res = batch_view(
            asc_state,
            _sheet("dup_bridge.asc"),
            [
                {"op": "remove_wire", "x1": x, "y1": m1, "x2": x, "y2": m2}  # type: ignore[list-item]
            ],
            stop_on_error=True,
        )
        data = res
        assert data is not None
        assert data["saved"] is False
        error = data["results"][0]["error"]
        assert "split the net" in error
        assert "R1.2" in error or "C1.1" in error

        after = _sheet_facts(asc_state, "dup_bridge.asc")
        assert after["wire_count"] == 4, "the refusal must leave the sheet alone"

    async def test_removing_a_redundant_duplicated_segment_takes_every_copy(
        self, asc_state: SessionState, work_dir: Path
    ):
        """Nothing floats when the segment carried nothing, so both copies go."""
        await self._wired_pair(asc_state, "dup_redundant")
        path = work_dir / "dup_redundant.asc"
        path.write_text(path.read_text() + "WIRE 900 900 964 900\nWIRE 964 900 900 900\n")

        res = batch_view(
            asc_state,
            _sheet("dup_redundant.asc"),
            [
                {"op": "remove_wire", "x1": 900, "y1": 900, "x2": 964, "y2": 900}  # type: ignore[list-item]
            ],
            stop_on_error=True,
        )
        data = res
        assert data is not None
        assert data["saved"] is True
        assert data["results"][0]["removed"] == 2

    async def test_the_wire_then_duplicate_then_remove_sequence_keeps_the_pin_connected(
        self, asc_state: SessionState, work_dir: Path
    ):
        """The measured sequence, end to end: route, route again, then act on
        what the response says. The repeat reports the segments as already
        present instead of warning about a duplicate, so there is nothing to
        clean up and the connection survives."""
        wire = await self._wired_pair(asc_state, "dup_sequence")

        repeat = batch_view(
            asc_state,
            _sheet("dup_sequence.asc"),
            [{"op": "wire_pins", "from_pin": "R1.2", "to_pin": "C1.1"}],
            stop_on_error=True,
        )
        data = repeat
        assert data is not None
        assert data["results"][0]["wire_count"] == 0
        assert data["results"][0]["already_present"]
        kinds = {w["kind"] for w in data.get("validation_warnings", [])}
        assert "duplicate_wire" not in kinds

        after = _sheet_facts(asc_state, "dup_sequence.asc")
        assert after["wire_count"] == 1
        assert after["wires"][0] == wire

    async def test_remove_wire_no_match_raises(self, asc_state: SessionState, work_dir: Path):

        blank_sheet_file(asc_state, "rm_nomatch")
        # stop_on_error default True: a no-match remove aborts the transaction.
        res = batch_view(
            asc_state,
            _sheet("rm_nomatch.asc"),
            [{"op": "remove_wire", "x1": 0, "y1": 0, "x2": 16, "y2": 0}],
            stop_on_error=True,
        )
        data = res
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No matching wire" in data["results"][0]["error"]

    async def test_remove_net_label_no_match_raises(self, asc_state: SessionState, work_dir: Path):

        blank_sheet_file(asc_state, "rm_lbl_nomatch")
        res = batch_view(
            asc_state,
            _sheet("rm_lbl_nomatch.asc"),
            [{"op": "remove_net_label", "x": 999, "y": 999}],
            stop_on_error=True,
        )
        data = res
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No net label found" in data["results"][0]["error"]

    async def test_remove_directive_round_trip(self, asc_state: SessionState, work_dir: Path):

        blank_sheet_file(asc_state, "rm_dir")
        # add_directive then remove_directive is the inverse pair the closure
        # test requires — exercise it end to end so the op actually edits.
        add = batch_view(
            asc_state,
            _sheet("rm_dir.asc"),
            [{"op": "add_directive", "instruction": ".tran 1m"}],
            stop_on_error=True,
        )
        assert add["saved"] is True
        read = _sheet_facts(asc_state, "rm_dir.asc")
        assert any(".tran 1m" in d for d in read["directives"])

        rm = batch_view(
            asc_state,
            _sheet("rm_dir.asc"),
            [{"op": "remove_directive", "instruction": ".tran 1m"}],
            stop_on_error=True,
        )
        data = rm
        assert data["saved"] is True
        assert data["failed_count"] == 0
        assert data["results"][0]["removed"] == "directive"

        read2 = _sheet_facts(asc_state, "rm_dir.asc")
        assert not any(".tran 1m" in d for d in read2["directives"])

    async def test_remove_directive_no_match_raises(self, asc_state: SessionState, work_dir: Path):

        blank_sheet_file(asc_state, "rm_dir_nomatch")
        res = batch_view(
            asc_state,
            _sheet("rm_dir_nomatch.asc"),
            [{"op": "remove_directive", "instruction": ".tran 999"}],
            stop_on_error=True,
        )
        data = res
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No directive or comment" in data["results"][0]["error"]

    async def test_remove_directive_is_exact_not_substring(
        self, asc_state: SessionState, work_dir: Path
    ):
        # The inverse must match the full directive text, not a substring:
        # removing ".tran 1" must NOT delete ".tran 10m" (spicelib's matcher
        # would, silently corrupting the simulation setup).

        blank_sheet_file(asc_state, "rm_substr")
        batch_view(
            asc_state,
            _sheet("rm_substr.asc"),
            [{"op": "add_directive", "instruction": ".tran 10m"}],
            stop_on_error=True,
        )
        res = batch_view(
            asc_state,
            _sheet("rm_substr.asc"),
            [{"op": "remove_directive", "instruction": ".tran 1"}],
            stop_on_error=True,
        )
        data = res
        # ".tran 1" is a substring of ".tran 10m" but not an exact match: refuse.
        assert data["saved"] is False
        assert data["failed_count"] == 1
        assert "No directive or comment" in data["results"][0]["error"]
        read = _sheet_facts(asc_state, "rm_substr.asc")
        assert any(".tran 10m" in d for d in read["directives"])

    async def test_remove_directive_removes_one_of_duplicates(
        self, asc_state: SessionState, work_dir: Path
    ):
        # Inverse of a single add removes a single record: with two identical
        # directives, one remove_directive leaves exactly one.

        blank_sheet_file(asc_state, "rm_dup")
        batch_view(
            asc_state,
            _sheet("rm_dup.asc"),
            [  # type: ignore[arg-type]
                {"op": "add_directive", "instruction": ".tran 1m"},
                {"op": "add_directive", "instruction": ".tran 1m"},
            ],
            stop_on_error=True,
        )
        res = batch_view(
            asc_state,
            _sheet("rm_dup.asc"),
            [{"op": "remove_directive", "instruction": ".tran 1m"}],
            stop_on_error=True,
        )
        assert res["saved"] is True
        read = _sheet_facts(asc_state, "rm_dup.asc")
        assert sum(1 for d in read["directives"] if d == ".tran 1m") == 1


@pytest.mark.asyncio
class TestAddNetLabelOpValidation:
    """The add_net_label op is the public path now that the standalone tool is
    unregistered, so it must enforce the same rules: refuse a label that would
    short two different named nets, and surface duplicate-name / floating-
    placement warnings."""

    async def test_short_refused_via_batch(self, asc_state: SessionState):

        blank_sheet_file(asc_state, "lbl_short")
        # Two different named labels on the same pin coordinate would merge the
        # nets at netlist time; the second must be refused, not silently saved.
        res = batch_view(
            asc_state,
            _sheet("lbl_short.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
        )
        results = {r["index"]: r for r in res["results"]}
        assert results[1]["ok"] is True  # net "a" placed
        assert results[2]["ok"] is False  # net "b" would short — refused
        assert "short" in results[2]["error"].lower()

    async def test_floating_label_warning_via_batch(self, asc_state: SessionState):

        blank_sheet_file(asc_state, "lbl_float")
        res = batch_view(
            asc_state,
            _sheet("lbl_float.asc"),
            [{"op": "add_net_label", "net": "x", "x": 500, "y": 500}],
            stop_on_error=True,
        )
        op = res["results"][0]
        assert op["ok"] is True
        assert any("no wire" in w.lower() for w in op.get("warnings", []))

    async def test_duplicate_label_warning_via_batch(self, asc_state: SessionState):

        blank_sheet_file(asc_state, "lbl_dup")
        # Same name on two distinct (unwired) pins: not a short (the netlist merges
        # same-name labels into one net) — the only cost is that a later wire_pins
        # can't disambiguate, which the warning states without a scare.
        res = batch_view(
            asc_state,
            _sheet("lbl_dup.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
            stop_on_error=True,
        )
        op2 = res["results"][2]
        assert op2["ok"] is True
        warns = op2.get("warnings", [])
        # Reframed: names the duplicate but says it merges correctly and only
        # wire_pins is ambiguous — no "short"/"will error" scare.
        assert any("already labels a net" in w and "ambiguous" in w for w in warns)


@pytest.mark.asyncio
class TestMoveRemoveOpWarnings:
    """The move/remove ops are the public path now; they must surface the same
    bbox-overlap and orphaned-wire warnings the standalone handlers did (these
    are NOT recovered by the batch's end-of-run post_op_warnings)."""

    async def _build_pair(self, asc_state: SessionState, name: str):

        blank_sheet_file(asc_state, name)
        # R1 above R2, wired R1.2 -> R2.1. Fixture res pins: 1=(x,y-48), 2=(x,y+48).
        return batch_view(
            asc_state,
            _sheet(f"{name}.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
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
                {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "R2.1"},
            ],
            stop_on_error=True,
        )

    async def test_move_overlap_warning_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "mv_overlap")
        res = batch_view(
            asc_state,
            _sheet("mv_overlap.asc"),
            [{"op": "move_component", "reference": "R2", "x": 200, "y": 200}],
            stop_on_error=True,
        )
        op = res["results"][0]
        assert op["ok"] is True
        assert any("Overlaps" in w for w in op.get("warnings", []))

    async def test_move_orphan_warning_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "mv_orphan")
        res = batch_view(
            asc_state,
            _sheet("mv_orphan.asc"),
            [{"op": "move_component", "reference": "R1", "x": 600, "y": 200}],
            stop_on_error=True,
        )
        op = res["results"][0]
        assert any("old pin" in w for w in op.get("warnings", []))

    async def test_remove_orphan_warning_then_cleanup_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "rm_orphan")
        # Remove without cleanup: the wire left on R1's former pin is flagged.
        res = batch_view(
            asc_state,
            _sheet("rm_orphan.asc"),
            [{"op": "remove_component", "reference": "R1"}],
            stop_on_error=True,
        )
        op = res["results"][0]
        assert any("orphaned" in w for w in op.get("warnings", []))

    async def test_remove_cleanup_reports_deleted_via_op(self, asc_state: SessionState):
        await self._build_pair(asc_state, "rm_clean")
        res = batch_view(
            asc_state,
            _sheet("rm_clean.asc"),
            [  # type: ignore[arg-type]  # pydantic validates dicts
                {"op": "remove_component", "reference": "R1", "cleanup_wires": True},  # type: ignore[arg-type]
            ],
            stop_on_error=True,
        )
        op = res["results"][0]
        assert op["deleted_wires"] >= 1
        assert "warnings" not in op


# Relocated regression coverage from a retired test module.
class TestMidSegmentLabelDetected:
    """A label sitting mid-segment on a wire used to be invisible
    to ``wire_pins``'s endpoint-only label compare. The fix is segment-
    aware: the trace dragon-swallows interest points that lie on a wire
    even if they're not at an endpoint.
    """

    def test_point_on_segment_horizontal(self) -> None:
        from ltspice_mcp.lib.schematic_ops import point_on_segment

        # Mid-x point on a horizontal wire.
        assert point_on_segment((150, 100), (100, 100), (200, 100))
        # Same y but outside x-range.
        assert not point_on_segment((300, 100), (100, 100), (200, 100))
        # Different y.
        assert not point_on_segment((150, 101), (100, 100), (200, 100))

    def test_point_on_segment_vertical(self) -> None:
        from ltspice_mcp.lib.schematic_ops import point_on_segment

        assert point_on_segment((100, 150), (100, 100), (100, 200))
        assert not point_on_segment((100, 250), (100, 100), (100, 200))
        assert not point_on_segment((101, 150), (100, 100), (100, 200))

    def test_named_labels_strips_ground(self) -> None:
        from ltspice_mcp.lib.schematic_ops import named_labels

        assert named_labels(frozenset({"OUTP", "0"})) == {"OUTP"}
        assert named_labels(frozenset({"0"})) == set()
        assert named_labels(frozenset()) == set()


# Relocated regression coverage from a retired test module.
async def _build_name_wired_rc(name: str, state: SessionState, work_dir: Path) -> str:
    """Build an RC schematic wired by net label (one FLAG per pin), the way a
    label-based layout connects: R1(in,out), C1(out,0), V1(in,0). Returns the
    .asc filename. Pins are connected only by shared label name, not by wires,
    so trace_net must fold same-name FLAGs together.
    """
    asc = work_dir / f"{name}.asc"
    asc.write_text("Version 4\nSHEET 1 880 680\n")
    batch_view(
        state,
        asc,
        [  # type: ignore[arg-type]  # pydantic validates dicts
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
        stop_on_error=True,
    )
    return asc.name


@pytest.mark.asyncio
class TestTraceNet:
    async def test_name_based_net_on_label_wiring(self, asc_state: SessionState, work_dir: Path):
        # R1.1 is on net "in". V1.+ is also on "in" — they are at different
        # coordinates connected only by the shared label name.
        path = await _build_name_wired_rc("trace_rc", asc_state, work_dir)
        res = await handle_trace_net(TraceNetInput(path=path, pin="R1.1"), asc_state)
        sc = res.structuredContent
        assert sc is not None
        assert sc is not None
        assert sc["labels"] == ["in"]
        refs = {p["reference"] for p in sc["pins"]}
        assert refs == {"R1", "V1"}
        assert sc["is_shorted"] is False

    async def test_trace_by_net_name(self, asc_state: SessionState, work_dir: Path):
        # net:in matches one FLAG per pin (V1.+ and R1.1) — resolve_pin would
        # refuse the ambiguity, but trace_net seeds from a match and name-merges.
        path = await _build_name_wired_rc("trace_byname", asc_state, work_dir)
        res = await handle_trace_net(TraceNetInput(path=path, pin="net:in"), asc_state)
        sc = res.structuredContent
        assert sc is not None
        assert sc is not None
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
        sc = res.structuredContent
        assert sc is not None
        assert sc is not None
        assert sc["is_shorted"] is True
        assert set(sc["labels"]) == {"a", "b"}

    async def test_empty_coordinate_raises(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "empty.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nFLAG 0 0 a\n")
        with pytest.raises(NetlistError, match="Nothing found"):
            await handle_trace_net(TraceNetInput(path="empty.asc", x=500, y=500), asc_state)

    async def test_same_instance_wire_surfaced_as_warning(
        self, asc_state: SessionState, work_dir: Path
    ):
        # An inherited .asc can already contain a same-instance tie a hand edit
        # drew (our wire_pins now refuses to create one). trace_net must not
        # silently report a connection LTspice drops — it surfaces the fact.
        # Fixture res pins: 1=(200,152), 2=(200,248); the WIRE ties them directly.
        asc = work_dir / "self_tie_trace.asc"
        asc.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 200 152 200 248\n"
            "SYMBOL res 200 200 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
        )
        res = await handle_trace_net(
            TraceNetInput(path="self_tie_trace.asc", pin="R1.1"), asc_state
        )
        sc = res.structuredContent
        assert sc is not None
        warnings = sc.get("warnings", [])
        assert len(warnings) == 1, sc
        assert "R1.1" in warnings[0] and "R1.2" in warnings[0]
        assert "LTspice drops" in warnings[0]
        # The text channel mirrors the structured warning (self-sufficiency).
        assert "LTspice drops" in _result_text(res)

    async def test_collinear_multi_segment_tie_surfaced_as_warning(
        self, asc_state: SessionState, work_dir: Path
    ):
        # A same-instance tie drawn as TWO collinear segments meeting at an
        # in-line vertex ((200,200)) is what LTspice merges + drops. trace_net
        # collinear-merges before checking, so it must still surface the fact —
        # a per-segment check alone would miss this (neither segment joins two
        # pins directly).
        asc = work_dir / "collinear_trace.asc"
        asc.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 200 152 200 200\n"
            "WIRE 200 200 200 248\n"
            "SYMBOL res 200 200 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
        )
        res = await handle_trace_net(
            TraceNetInput(path="collinear_trace.asc", pin="R1.1"), asc_state
        )
        sc = res.structuredContent
        assert sc is not None
        warnings = sc.get("warnings", [])
        assert len(warnings) == 1, sc
        assert "R1.1" in warnings[0] and "R1.2" in warnings[0]

    async def test_zero_length_wire_no_self_warning(self, asc_state: SessionState, work_dir: Path):
        # A hand-corrupted zero-length WIRE record sitting on a pin must not
        # produce a self-referential "R1.1 and R1.1 joined" warning.
        asc = work_dir / "zero_len.asc"
        asc.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 200 152 200 152\n"
            "SYMBOL res 200 200 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
        )
        res = await handle_trace_net(TraceNetInput(path="zero_len.asc", pin="R1.1"), asc_state)
        sc = res.structuredContent
        assert sc is not None
        assert sc is not None
        assert sc.get("warnings", []) == []

    async def test_same_instance_subsegment_over_interior_pin_surfaced(
        self, asc_state: SessionState, work_dir: Path
    ):
        # A wire from R1.1 runs straight DOWN through R1.2 and past it to a bare
        # end (200,300). R1.2 sits on the run's interior, where LTspice splits the
        # wire — so the R1.1->R1.2 sub-run is a same-instance tie LTspice drops.
        # The interior pin is not a wire endpoint, so a cut built from endpoints
        # alone would miss it and report no tie; the pin-on-interior cut splits the
        # run there and surfaces the dropped sub-segment. Fixture res pins:
        # 1=(200,152), 2=(200,248).
        asc = work_dir / "interior_self_tie.asc"
        asc.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 200 152 200 300\n"
            "SYMBOL res 200 200 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
        )
        res = await handle_trace_net(
            TraceNetInput(path="interior_self_tie.asc", pin="R1.1"), asc_state
        )
        sc = res.structuredContent
        assert sc is not None
        warnings = sc.get("warnings", [])
        assert len(warnings) == 1, sc
        assert "R1.1" in warnings[0] and "R1.2" in warnings[0]
        assert "LTspice drops" in warnings[0]

    async def test_foreign_interior_pin_yields_no_false_same_instance_warning(
        self, asc_state: SessionState, work_dir: Path
    ):
        # A straight run between R1's two pins (200,152)->(200,248) would read as a
        # dropped self-tie — EXCEPT a foreign pin R2.2 sits on its interior at
        # (200,200). LTspice splits the wire at that pin, so the run becomes two
        # cross-instance segments (R1.1->R2.2, R2.2->R1.2) that LTspice keeps.
        # The interior-pin cut must split there too, so no false same-instance
        # warning is raised. R2 placed at (200,152): pins 1=(200,104), 2=(200,200).
        asc = work_dir / "foreign_interior.asc"
        asc.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 200 152 200 248\n"
            "SYMBOL res 200 200 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
            "SYMBOL res 200 152 R0\n"
            "SYMATTR InstName R2\n"
            "SYMATTR Value 2k\n"
        )
        res = await handle_trace_net(
            TraceNetInput(path="foreign_interior.asc", pin="R1.1"), asc_state
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc.get("warnings", []) == []


# Relocated regression coverage from a retired test module.
class TestOnWirePredicate:
    def test_matches_point_on_segment(self):
        segments = [((0, 0), (100, 0)), ((100, 0), (100, 80)), ((50, 50), (50, 50))]
        on_wire = build_on_wire_predicate(segments)
        probes = [(0, 0), (50, 0), (100, 0), (100, 40), (100, 80), (50, 50), (10, 10), (200, 0)]
        for p in probes:
            expected = any(point_on_segment(p, v1, v2) for v1, v2 in segments)
            assert on_wire(p) == expected, p

    def test_endpoints_and_spans(self):
        on_wire = build_on_wire_predicate([((0, 0), (0, 100))])
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
        add_component(asc_state, _sheet("build.asc"), "R1", "res", 100, 100)
        # Second component placed far away: its warnings must NOT re-list R1's
        # floating pins (the O(n^2) spam this fix removes).
        res = add_component(asc_state, _sheet("build.asc"), "R2", "res", 400, 100)
        data = res
        assert data is not None
        vw = data.get("validation_warnings", [])
        refs = {w["ref"] for w in vw}
        assert refs <= {"R2"}
        assert "R1" not in refs


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
        blank_sheet_file(real_state, "real")
        for ref, sym, x, y, val in [
            ("R1", "res", 100, 100, "1k"),
            ("C1", "cap", 300, 100, "1n"),
            ("M1", "nmos", 500, 100, "NMOS1"),
        ]:
            facts = add_component(real_state, _sheet("real.asc"), ref, sym, x, y, value=val)
            assert facts["reference"] == ref
            # Geometry comes from parsing the real .asy; empty pins = a broken parse.
            assert facts["pins"]

    async def test_real_resistor_pins_are_a_b(self, real_state: SessionState):
        # The fixture res uses numeric pins 1/2; the real LTspice res uses A/B.
        # Guards against the suite silently drifting onto fabricated geometry.
        data = await inspect_one(real_state, {"kind": "symbol", "name": "res"})
        assert {p["name"] for p in data["pins_by_rotation"]["R0"]} == {"A", "B"}

    async def test_batch_adds_a_real_symbol(self, real_state: SessionState):
        blank_sheet_file(real_state, "real2")
        view = batch_view(
            real_state,
            _sheet("real2.asc"),
            [
                {
                    "op": "add_component",
                    "reference": "R1",
                    "symbol": "res",
                    "x": 100,
                    "y": 100,
                    "value": "1k",
                }
            ],
        )
        assert view["applied_count"] == 1
        assert view["failed_count"] == 0


@pytest.mark.asyncio
class TestDuplicateLabelAdvisory:
    """The documented per-pin-label style repeats one identical duplicate-label
    advisory on every add_net_label op of a net, so each op must report it."""

    async def test_each_repeat_label_op_reports_the_duplicate(
        self, asc_state: SessionState, work_dir: Path
    ):
        asc = work_dir / "labels.asc"
        asc.write_text(
            "Version 4\n"
            "SHEET 1 880 680\n"
            "WIRE 100 52 100 0\n"
            "WIRE 100 148 100 200\n"
            "WIRE 300 52 300 0\n"
            "FLAG 100 0 vin\n"
            "SYMBOL res 100 100 R0\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
        )
        view = batch_view(
            asc_state,
            asc,
            [
                {"op": "add_net_label", "net": "vin", "x": 100, "y": 52},
                {"op": "add_net_label", "net": "vin", "x": 300, "y": 0},
                {"op": "add_net_label", "net": "vin", "x": 300, "y": 52},
            ],
        )
        assert view["saved"] is True
        all_warnings = [w for r in view["results"] for w in (r.get("warnings") or [])]
        dup = [w for w in all_warnings if "already labels a net" in w]
        assert dup, all_warnings


@pytest.mark.asyncio
class TestSchematicReadability:
    """Readability eval for a schematic built the way the guide recommends:
    one edit_schematic batch, wire_pins for the signal path, add_net_label only
    for the ground/global nets. The result must come out WIRED — not 'net-label
    soup', where every component pin floats on its own same-named FLAG and there
    are no wires. This is the regression guard for the blind spot that let a
    label-only build ship: the signal junctions have to be real WIRE records,
    and net labels stay scoped to the terminal nets.
    """

    async def test_built_schematic_is_wired_not_label_soup(
        self, asc_state: SessionState, work_dir: Path
    ):
        # A 3-resistor chain stacked on x=200: the two internal junctions are
        # wired by wire_pins; only the two terminal nets (in, ground) get a
        # label. Fixture res pins: 1=(0,-48), 2=(0,48), so Rn at (200, y) has
        # pins at (200, y-48) and (200, y+48).
        view = batch_view(
            asc_state,
            blank_sheet_file(asc_state, "readable"),
            [
                {"op": "add_component", "reference": "R1", "symbol": "res", "x": 200, "y": 200},
                {"op": "add_component", "reference": "R2", "symbol": "res", "x": 200, "y": 400},
                {"op": "add_component", "reference": "R3", "symbol": "res", "x": 200, "y": 600},
                {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "R2.1"},
                {"op": "wire_pins", "from_pin": "R2.2", "to_pin": "R3.1"},
                {"op": "add_net_label", "net": "in", "pin": "R1.1"},
                {"op": "add_net_label", "net": "0", "pin": "R3.2"},
            ],
        )
        assert view["failed_count"] == 0
        assert view["saved"] is True

        asc = _sheet("readable.asc")
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
