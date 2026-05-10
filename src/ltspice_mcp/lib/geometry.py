"""Axis-aligned bounding box and shared 2D-geometry helpers.

The on-the-wire bbox shape for MCP tool outputs is ``{x, y, width, height}``
(origin + extent). Internally we use ``BBox(x1, y1, x2, y2)`` because it
makes set operations (overlap, union, expand) trivial. Convert at the
serialization boundary via :meth:`BBox.to_origin_size_dict`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box, normalized so x1 <= x2 and y1 <= y2."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        # Compute all four locals before any setattr so reordering is atomic.
        nx1 = min(self.x1, self.x2)
        nx2 = max(self.x1, self.x2)
        ny1 = min(self.y1, self.y2)
        ny2 = max(self.y1, self.y2)
        if (nx1, ny1, nx2, ny2) != (self.x1, self.y1, self.x2, self.y2):
            object.__setattr__(self, "x1", nx1)
            object.__setattr__(self, "y1", ny1)
            object.__setattr__(self, "x2", nx2)
            object.__setattr__(self, "y2", ny2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def overlaps(self, other: BBox) -> bool:
        """True if the two boxes share interior area (touching edges don't count)."""
        return (
            self.x1 < other.x2 and self.x2 > other.x1 and self.y1 < other.y2 and self.y2 > other.y1
        )

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def expanded(self, margin: int) -> BBox:
        return BBox(self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)

    @classmethod
    def from_origin_size(cls, x: int, y: int, width: int, height: int) -> BBox:
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_points(cls, points: Iterable[tuple[int, int]]) -> BBox | None:
        """Smallest BBox enclosing all points; ``None`` if empty."""
        xs: list[int] = []
        ys: list[int] = []
        for px, py in points:
            xs.append(px)
            ys.append(py)
        if not xs:
            return None
        return cls(min(xs), min(ys), max(xs), max(ys))

    def to_origin_size_dict(self) -> dict[str, int]:
        """MCP tool-output shape: ``{x, y, width, height}``."""
        return {"x": self.x1, "y": self.y1, "width": self.width, "height": self.height}
