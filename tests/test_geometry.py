"""Unit tests for the BBox dataclass and miss-caching in get_symbol_info."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ltspice_mcp.lib import symbol_geometry
from ltspice_mcp.lib.geometry import BBox

# ---------------------------------------------------------------------------
# BBox normalization, properties, and set ops
# ---------------------------------------------------------------------------


class TestBBoxNormalization:
    def test_already_normalized(self) -> None:
        b = BBox(0, 0, 10, 20)
        assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 10, 20)

    def test_swapped_x_normalizes(self) -> None:
        b = BBox(10, 0, 0, 20)
        assert (b.x1, b.x2) == (0, 10)
        assert (b.y1, b.y2) == (0, 20)

    def test_swapped_y_normalizes(self) -> None:
        b = BBox(0, 30, 10, 5)
        assert (b.y1, b.y2) == (5, 30)

    def test_negative_coords_preserved(self) -> None:
        # Symbols centered on origin have negative coords; normalization
        # must not collapse them.
        b = BBox(-50, -40, 50, 40)
        assert (b.x1, b.y1, b.x2, b.y2) == (-50, -40, 50, 40)


class TestBBoxProperties:
    def test_width_height(self) -> None:
        b = BBox(10, 20, 50, 80)
        assert b.width == 40
        assert b.height == 60

    def test_zero_size(self) -> None:
        b = BBox(5, 5, 5, 5)
        assert b.width == 0
        assert b.height == 0

    def test_center(self) -> None:
        assert BBox(0, 0, 10, 20).center == (5.0, 10.0)


class TestBBoxOverlaps:
    def test_disjoint_horizontal(self) -> None:
        assert not BBox(0, 0, 10, 10).overlaps(BBox(20, 0, 30, 10))

    def test_disjoint_vertical(self) -> None:
        assert not BBox(0, 0, 10, 10).overlaps(BBox(0, 20, 10, 30))

    def test_overlapping_interior(self) -> None:
        assert BBox(0, 0, 10, 10).overlaps(BBox(5, 5, 15, 15))

    def test_touching_edge_does_not_overlap(self) -> None:
        # Edge-touching is the standard "open" overlap convention.
        assert not BBox(0, 0, 10, 10).overlaps(BBox(10, 0, 20, 10))

    def test_one_contains_other(self) -> None:
        assert BBox(0, 0, 100, 100).overlaps(BBox(40, 40, 60, 60))


class TestBBoxOps:
    def test_union(self) -> None:
        u = BBox(0, 0, 10, 10).union(BBox(5, 5, 20, 30))
        assert u == BBox(0, 0, 20, 30)

    def test_expanded(self) -> None:
        e = BBox(0, 0, 10, 10).expanded(5)
        assert e == BBox(-5, -5, 15, 15)

    def test_contains_point_inside(self) -> None:
        assert BBox(0, 0, 10, 10).contains_point(5, 5)

    def test_contains_point_on_edge(self) -> None:
        assert BBox(0, 0, 10, 10).contains_point(0, 5)
        assert BBox(0, 0, 10, 10).contains_point(10, 10)

    def test_contains_point_outside(self) -> None:
        assert not BBox(0, 0, 10, 10).contains_point(11, 5)


class TestBBoxFactories:
    def test_from_origin_size(self) -> None:
        assert BBox.from_origin_size(5, 10, 20, 30) == BBox(5, 10, 25, 40)

    def test_from_points(self) -> None:
        b = BBox.from_points([(0, 0), (10, 5), (-5, 8)])
        assert b == BBox(-5, 0, 10, 8)

    def test_from_points_empty(self) -> None:
        assert BBox.from_points([]) is None

    def test_to_origin_size_dict(self) -> None:
        # The MCP wire format is {x, y, width, height}; this is the contract.
        d = BBox(5, 10, 25, 40).to_origin_size_dict()
        assert d == {"x": 5, "y": 10, "width": 20, "height": 30}


# ---------------------------------------------------------------------------
# Miss-caching in get_symbol_info
# ---------------------------------------------------------------------------


class TestGetSymbolInfoMissCache:
    """A failed resolution should be cached so repeated lookups don't re-walk
    the library search path. Before this fix, ``get_symbol_info`` only stored
    successful parses, leaving misses to redo ``rglob`` on every call.

    Each test swaps in a fresh cache dict (via monkeypatch) so we don't
    mutate the shared module-global one — that would race with other tests
    in the same xdist worker that rely on cached symbol lookups.
    """

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(symbol_geometry, "_symbol_cache", {})

    def test_miss_is_cached(self) -> None:
        with patch.object(symbol_geometry, "_find_asy_file", return_value=None) as find_mock:
            assert symbol_geometry.get_symbol_info("nope_xyz") is None
            assert symbol_geometry.get_symbol_info("nope_xyz") is None
            assert symbol_geometry.get_symbol_info("nope_xyz") is None
        assert find_mock.call_count == 1

    def test_miss_then_hit_separate_keys(self) -> None:
        with patch.object(symbol_geometry, "_find_asy_file", return_value=None) as find_mock:
            symbol_geometry.get_symbol_info("missing_a")
            symbol_geometry.get_symbol_info("missing_b")
        assert find_mock.call_count == 2

    def test_hit_after_miss_does_not_repath(self, tmp_path: Path) -> None:
        # First lookup misses and caches None. Then we replace the cache
        # entry with a real symbol and ensure the cached value wins.
        with patch.object(symbol_geometry, "_find_asy_file", return_value=None):
            assert symbol_geometry.get_symbol_info("res_test") is None

        asy = tmp_path / "res_test.asy"
        asy.write_text("Version 4\nLINE Normal 0 0 10 10\n")
        info = symbol_geometry.parse_asy_file(asy)
        symbol_geometry._symbol_cache["res_test"] = info

        # No filesystem walk should occur — cache hit.
        with patch.object(symbol_geometry, "_find_asy_file", side_effect=AssertionError):
            assert symbol_geometry.get_symbol_info("res_test") is info


# ---------------------------------------------------------------------------
# bbox_from_elements: typed parsing replaces the regex slurp
# ---------------------------------------------------------------------------


class TestBboxFromElements:
    def test_lines_only(self) -> None:
        from ltspice_mcp.lib.symbol_geometry import LineEl, bbox_from_elements

        elements = [LineEl(0, 0, 10, 5), LineEl(-5, -10, 0, 0)]
        assert bbox_from_elements(elements) == BBox(-5, -10, 10, 5)

    def test_with_extra_pin_points(self) -> None:
        from ltspice_mcp.lib.symbol_geometry import LineEl, bbox_from_elements

        bb = bbox_from_elements([LineEl(0, 0, 10, 10)], extra_points=[(-5, 20)])
        assert bb == BBox(-5, 0, 10, 20)

    def test_empty(self) -> None:
        from ltspice_mcp.lib.symbol_geometry import bbox_from_elements

        assert bbox_from_elements([]) is None

    def test_arc_uses_bbox_corners_only(self) -> None:
        # The typed ArcEl exposes the underlying ellipse rectangle. Even if
        # we constructed one with synthetic out-of-bbox start/end, the
        # element doesn't carry those — so they cannot leak into the bbox.
        from ltspice_mcp.lib.symbol_geometry import ArcEl, bbox_from_elements

        bb = bbox_from_elements([ArcEl(0, 0, 10, 10)])
        assert bb == BBox(0, 0, 10, 10)


# ---------------------------------------------------------------------------
# Shape parser: rejects garbage rather than slurping ints
# ---------------------------------------------------------------------------


class TestParseShape:
    @pytest.mark.parametrize(
        "line",
        [
            "LINE Normal 0 0 10 10",
            "RECTANGLE Normal -5 -10 5 10",
            "CIRCLE Normal 0 0 20 20",
            "ARC Normal 0 0 10 10 0 5 5 10",
        ],
    )
    def test_valid_shapes(self, line: str) -> None:
        from ltspice_mcp.lib.symbol_geometry import _parse_shape

        assert _parse_shape(line) is not None

    @pytest.mark.parametrize(
        "line",
        [
            "PIN 0 0 TOP 8",
            "SYMATTR Description foo",
            "Version 4",
            "",
            "LINE Normal 0",  # too few fields
            "LINE Normal a b c d",  # non-integer coords
        ],
    )
    def test_invalid_inputs(self, line: str) -> None:
        from ltspice_mcp.lib.symbol_geometry import _parse_shape

        assert _parse_shape(line) is None
