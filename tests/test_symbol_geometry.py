"""Unit tests for symbol_geometry — rotation transforms, .asy parsing,
placed geometry computation, and pin direction inference.
"""

from pathlib import Path

import pytest

from ltspice_mcp.lib.symbol_geometry import (
    PinInfo,
    SymbolInfo,
    _apply_rotation,
    _pin_direction,
    compute_placed_geometry,
    parse_asy_file,
)

# ---------------------------------------------------------------------------
# _apply_rotation
# ---------------------------------------------------------------------------


class TestApplyRotation:
    """Verify all 8 rotation/mirror transforms against hand-computed values."""

    # Use a non-square point so we can distinguish axis swaps.
    PX, PY = 3, 7

    @pytest.mark.parametrize(
        ("rotation", "expected"),
        [
            ("R0", (3, 7)),
            ("R90", (-7, 3)),
            ("R180", (-3, -7)),
            ("R270", (7, -3)),
            ("M0", (-3, 7)),
            ("M90", (-7, -3)),
            ("M180", (3, -7)),
            ("M270", (7, 3)),
        ],
    )
    def test_transform(self, rotation: str, expected: tuple[int, int]):
        assert _apply_rotation(self.PX, self.PY, rotation) == expected

    def test_identity_origin(self):
        """Origin should be invariant under any transform."""
        for rot in ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"):
            assert _apply_rotation(0, 0, rot) == (0, 0)

    def test_r90_four_times_is_identity(self):
        """Applying R90 four times should return to the original point."""
        x, y = 5, 11
        for _ in range(4):
            x, y = _apply_rotation(x, y, "R90")
        assert (x, y) == (5, 11)

    def test_r180_is_double_r90(self):
        x, y = 4, 9
        r180 = _apply_rotation(x, y, "R180")
        r90_twice = _apply_rotation(*_apply_rotation(x, y, "R90"), "R90")
        assert r180 == r90_twice

    def test_mirror_twice_is_identity(self):
        """Applying M0 twice should return to the original point (mirror is self-inverse)."""
        x, y = 6, -2
        mx, my = _apply_rotation(x, y, "M0")
        assert _apply_rotation(mx, my, "M0") == (x, y)


# ---------------------------------------------------------------------------
# parse_asy_file
# ---------------------------------------------------------------------------


# Minimal synthetic .asy content for an NMOS transistor symbol.
NMOS_ASY = """\
Version 4
SymbolType CELL
LINE Normal 0 48 0 96
LINE Normal 0 -48 0 -96
LINE Normal -48 0 0 0
RECTANGLE Normal -64 -96 64 96
SYMATTR Prefix M
SYMATTR Description N-Channel MOSFET
PIN 0 -96 TOP 8
PINATTR PinName D
PINATTR SpiceOrder 1
PIN -48 0 LEFT 8
PINATTR PinName G
PINATTR SpiceOrder 2
PIN 0 96 BOTTOM 8
PINATTR PinName S
PINATTR SpiceOrder 3
"""


class TestParseAsyFile:
    @pytest.fixture
    def nmos_asy(self, tmp_path: Path) -> Path:
        p = tmp_path / "nmos.asy"
        p.write_text(NMOS_ASY)
        return p

    def test_basic_parse(self, nmos_asy: Path):
        info = parse_asy_file(nmos_asy)
        assert info.name == "nmos"
        assert info.description == "N-Channel MOSFET"
        assert len(info.pins) == 3

    def test_pins_sorted_by_order(self, nmos_asy: Path):
        info = parse_asy_file(nmos_asy)
        orders = [p.order for p in info.pins]
        assert orders == [1, 2, 3]

    def test_pin_names(self, nmos_asy: Path):
        info = parse_asy_file(nmos_asy)
        names = [p.name for p in info.pins]
        assert names == ["D", "G", "S"]

    def test_pin_coordinates(self, nmos_asy: Path):
        info = parse_asy_file(nmos_asy)
        pin_map = {p.name: (p.x, p.y) for p in info.pins}
        assert pin_map["D"] == (0, -96)
        assert pin_map["G"] == (-48, 0)
        assert pin_map["S"] == (0, 96)

    def test_bounding_box(self, nmos_asy: Path):
        """Bbox should span from min to max across all geometry."""
        info = parse_asy_file(nmos_asy)
        # LINE coords: (0,48), (0,96), (0,-48), (0,-96), (-48,0), (0,0)
        # RECTANGLE coords: (-64,-96), (64,96)
        # PIN coords: (0,-96), (-48,0), (0,96)
        # x range: -64..64 = 128, y range: -96..96 = 192
        assert info.bbox_width == 128
        assert info.bbox_height == 192

    def test_empty_file(self, tmp_path: Path):
        """An empty symbol file should parse without error."""
        p = tmp_path / "empty.asy"
        p.write_text("Version 4\nSymbolType CELL\n")
        info = parse_asy_file(p)
        assert info.name == "empty"
        assert info.pins == ()
        assert info.bbox_width == 0
        assert info.bbox_height == 0

    def test_pin_without_attributes(self, tmp_path: Path):
        """A PIN line without PINATTR should still be parsed."""
        content = "Version 4\nPIN 10 20 TOP 8\n"
        p = tmp_path / "bare.asy"
        p.write_text(content)
        info = parse_asy_file(p)
        assert len(info.pins) == 1
        assert info.pins[0].name == ""
        assert info.pins[0].order == 0
        assert info.pins[0].x == 10
        assert info.pins[0].y == 20

    def test_circle_and_arc_contribute_to_bbox(self, tmp_path: Path):
        content = "Version 4\nCIRCLE Normal -10 -20 30 40\nARC Normal -5 -15 25 35 0 0 10 10\n"
        p = tmp_path / "shapes.asy"
        p.write_text(content)
        info = parse_asy_file(p)
        # CIRCLE: (-10,-20), (30,40) -> x: -10..30, y: -20..40
        # ARC: (-5,-15), (25,35), (0,0), (10,10) -> x: -5..25, y: -15..35
        # Combined: x: -10..30 = 40, y: -20..40 = 60
        assert info.bbox_width == 40
        assert info.bbox_height == 60


# ---------------------------------------------------------------------------
# compute_placed_geometry
# ---------------------------------------------------------------------------


class TestComputePlacedGeometry:
    """Test absolute coordinate computation for placed components."""

    @pytest.fixture
    def simple_symbol(self) -> SymbolInfo:
        """A simple 2-pin symbol centered on the origin.

        Pin A at (0,-50), pin B at (0,50). Bbox spans (-10,-50) to (10,50)
        — width 20, height 100, with the origin at the center.
        """
        return SymbolInfo(
            name="res",
            description="Resistor",
            pins=(
                PinInfo(name="A", order=1, x=0, y=-50),
                PinInfo(name="B", order=2, x=0, y=50),
            ),
            bbox_x=-10,
            bbox_y=-50,
            bbox_width=20,
            bbox_height=100,
        )

    def test_r0_placement(self, simple_symbol: SymbolInfo):
        result = compute_placed_geometry(simple_symbol, origin_x=100, origin_y=200, rotation="R0")
        pins = {p["name"]: p for p in result["pins"]}
        assert pins["A"]["x"] == 100
        assert pins["A"]["y"] == 150  # 200 + (-50)
        assert pins["B"]["x"] == 100
        assert pins["B"]["y"] == 250  # 200 + 50

    def test_r90_placement(self, simple_symbol: SymbolInfo):
        result = compute_placed_geometry(simple_symbol, origin_x=100, origin_y=200, rotation="R90")
        pins = {p["name"]: p for p in result["pins"]}
        # R90: (x,y) -> (-y, x)
        # A: (0,-50) -> (50, 0)   -> (150, 200)
        # B: (0,50)  -> (-50, 0)  -> (50, 200)
        assert pins["A"]["x"] == 150
        assert pins["A"]["y"] == 200
        assert pins["B"]["x"] == 50
        assert pins["B"]["y"] == 200

    def test_r180_placement(self, simple_symbol: SymbolInfo):
        result = compute_placed_geometry(simple_symbol, origin_x=0, origin_y=0, rotation="R180")
        pins = {p["name"]: p for p in result["pins"]}
        # R180: (x,y) -> (-x, -y)
        # A: (0,-50) -> (0, 50)
        # B: (0,50)  -> (0, -50)
        assert pins["A"]["x"] == 0
        assert pins["A"]["y"] == 50
        assert pins["B"]["x"] == 0
        assert pins["B"]["y"] == -50

    def test_bbox_r0(self, simple_symbol: SymbolInfo):
        result = compute_placed_geometry(simple_symbol, origin_x=100, origin_y=200, rotation="R0")
        bbox = result["bounding_box"]
        # Local bbox (-10,-50)..(10,50) placed at (100,200) → (90,150)..(110,250)
        assert bbox["x"] == 90
        assert bbox["y"] == 150
        assert bbox["width"] == 20
        assert bbox["height"] == 100

    def test_bbox_contains_all_pins(self, simple_symbol: SymbolInfo):
        """Regression: every placed pin must lie within the placed bounding box."""
        for rotation in ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"):
            result = compute_placed_geometry(
                simple_symbol, origin_x=300, origin_y=400, rotation=rotation
            )
            bbox = result["bounding_box"]
            for pin in result["pins"]:
                assert bbox["x"] <= pin["x"] <= bbox["x"] + bbox["width"], (
                    f"{rotation}: pin {pin['name']} x={pin['x']} outside bbox {bbox}"
                )
                assert bbox["y"] <= pin["y"] <= bbox["y"] + bbox["height"], (
                    f"{rotation}: pin {pin['name']} y={pin['y']} outside bbox {bbox}"
                )

    def test_bbox_r90(self, simple_symbol: SymbolInfo):
        """After R90, width and height should swap."""
        result = compute_placed_geometry(simple_symbol, origin_x=0, origin_y=0, rotation="R90")
        bbox = result["bounding_box"]
        assert bbox["width"] == 100
        assert bbox["height"] == 20

    def test_pin_dir_included(self, simple_symbol: SymbolInfo):
        result = compute_placed_geometry(simple_symbol, origin_x=0, origin_y=0, rotation="R0")
        for pin in result["pins"]:
            assert "dir" in pin
            assert pin["dir"] in ("up", "down", "left", "right", "unknown")

    def test_m0_mirror(self, simple_symbol: SymbolInfo):
        """M0 mirrors x: (x,y) -> (-x, y). Vertical pins stay vertical."""
        result = compute_placed_geometry(simple_symbol, origin_x=0, origin_y=0, rotation="M0")
        pins = {p["name"]: p for p in result["pins"]}
        # M0: (0,-50) -> (0,-50); (0,50) -> (0,50)  (x=0 unaffected)
        assert pins["A"]["y"] == -50
        assert pins["B"]["y"] == 50


# ---------------------------------------------------------------------------
# _pin_direction
# ---------------------------------------------------------------------------


class TestPinDirection:
    """Test pin wire direction inference from position relative to bbox center."""

    # Bbox at origin spanning (0,0) to (100,80) — center at (50, 40)
    BBOX_X = 0
    BBOX_Y = 0
    BBOX_W = 100
    BBOX_H = 80

    def test_right_edge(self):
        # Pin far to the right of center
        d = _pin_direction(90, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        assert d == "right"

    def test_left_edge(self):
        d = _pin_direction(10, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        assert d == "left"

    def test_top_edge(self):
        # Pin far above center (y < cy)
        d = _pin_direction(50, 5, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        assert d == "up"

    def test_bottom_edge(self):
        d = _pin_direction(50, 75, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        assert d == "down"

    def test_rotation_flips_direction(self):
        """A pin on the right edge under R0 should report left under R180."""
        d_r0 = _pin_direction(90, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        d_r180 = _pin_direction(
            90, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R180"
        )
        assert d_r0 == "right"
        assert d_r180 == "left"

    def test_r90_rotates_direction(self):
        """A pin on the right edge under R0 should report down under R90."""
        d = _pin_direction(90, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R90")
        assert d == "down"

    def test_mirror_m0_flips_horizontal(self):
        """M0 mirrors x-axis: right becomes left."""
        d = _pin_direction(90, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "M0")
        assert d == "left"

    def test_centered_bbox(self):
        """A symbol whose local bbox is centered on the origin (typical LTspice symbol).

        Bbox spans (-50,-40) to (50,40); a pin at (0,-40) is on the top edge.
        """
        # Bug regression: previously, _pin_direction assumed bbox started at (0,0),
        # so a pin at (0,-40) on a centered symbol was misclassified.
        d = _pin_direction(0, -40, -50, -40, 100, 80, "R0")
        assert d == "up"

    def test_center_pin_defaults(self):
        """A pin at the exact center picks a direction (not 'unknown')."""
        d = _pin_direction(50, 40, self.BBOX_X, self.BBOX_Y, self.BBOX_W, self.BBOX_H, "R0")
        assert d in ("up", "down", "left", "right")

    def test_zero_bbox_no_crash(self):
        """Zero-size bbox should not crash (max(w,1) guard)."""
        d = _pin_direction(5, 5, 0, 0, 0, 0, "R0")
        assert d in ("up", "down", "left", "right", "unknown")
