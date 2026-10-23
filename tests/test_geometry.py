"""Unit tests for the BBox dataclass, and for a symbol miss searched for once per request."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ltspice_mcp.lib import symbol_library
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_library import SymbolLibrary

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
# Misses within one request
# ---------------------------------------------------------------------------


class TestAMissIsSearchedForOncePerRequest:
    """A symbol no library holds is searched for once by a request's resolver,
    however often the request asks: without that, every reference to a missing
    symbol walks the library again. The next request searches afresh, so a
    symbol added meanwhile is found."""

    def test_one_resolver_searches_once(self, tmp_path: Path) -> None:
        library = SymbolLibrary([tmp_path])
        with patch.object(symbol_library, "find_symbol", return_value=None) as find_mock:
            resolver = library.for_sheet(None)
            assert resolver.resolve("nope_xyz") is None
            assert resolver.resolve("nope_xyz") is None
            assert resolver.load("nope_xyz") is None
        assert find_mock.call_count == 1

    def test_each_name_is_searched_for_once(self, tmp_path: Path) -> None:
        with patch.object(symbol_library, "find_symbol", return_value=None) as find_mock:
            resolver = SymbolLibrary([tmp_path]).for_sheet(None)
            resolver.resolve("missing_a")
            resolver.resolve("missing_b")
        assert find_mock.call_count == 2

    def test_the_next_request_searches_again(self, tmp_path: Path) -> None:
        library = SymbolLibrary([tmp_path])
        with patch.object(symbol_library, "find_symbol", return_value=None) as find_mock:
            library.for_sheet(None).resolve("nope_xyz")
            library.for_sheet(None).resolve("nope_xyz")
        assert find_mock.call_count == 2
