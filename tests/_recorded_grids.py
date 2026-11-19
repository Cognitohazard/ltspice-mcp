"""The grids of recorded LTspice behaviour, as ``inputs/cases.toml`` declares them.

A grid indexes the recorded cases for one question that a short list of
functions decides: its axes, the groups of axes that are crossed, and the
cells, each one item of a case (a measurement of a deck, a trace of a raw, a
line of a log) with its coordinates. It names no file of its own. The tests of
a grid read the cells' recordings; this module only reads the declaration and
says which coordinates have a recording on a build.
"""

from __future__ import annotations

import ast
import functools
import itertools
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.ltspice_recorder import CASES_FILE, INPUTS, recorded_builds

REPO = Path(__file__).resolve().parent.parent

#: Keys of a cell table or an item that are not coordinates.
_CELL_KEYS = frozenset({"cases", "at", "items", "on", "refused"})


@dataclass(frozen=True)
class Axis:
    """One axis: its values, and the value a cell that names none takes. A
    default that is not one of the values marks a cell the axis does not apply
    to; such a cell covers no coordinate of a group holding the axis."""

    name: str
    values: tuple[str, ...]
    default: str | None = None


@dataclass(frozen=True)
class Exclusion:
    """Coordinates no cell can have: those whose value on each named axis is
    one of the values listed, with the reason. With ``builds``, only on those
    builds: a rule of the code that claims one LTspice build has no
    coordinate on the other."""

    at: Mapping[str, tuple[str, ...]]
    reason: str
    builds: frozenset[str] = frozenset()

    def covers(self, coordinates: Mapping[str, str], build: str) -> bool:
        """Whether the exclusion holds ``coordinates`` on ``build``."""
        if self.builds and build not in self.builds:
            return False
        return all(
            axis in coordinates and coordinates[axis] in values for axis, values in self.at.items()
        )


@dataclass(frozen=True)
class Cell:
    """One item of one case, with its coordinates. ``on`` gives, per build,
    coordinates that differ there; ``refused`` names the builds that do not
    take the item at all, with the reason, and ``refused_itself`` those of
    them that refuse it for what the item is. A build that refuses the item
    only because another card stopped the deck fills no coordinate with it.
    ``extra`` holds the grid's own keys (``item_keys``)."""

    case: str
    item: str
    coordinates: Mapping[str, str]
    on: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    refused: Mapping[str, str] = field(default_factory=dict)
    refused_itself: frozenset[str] = frozenset()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def at(self, build: str) -> dict[str, str]:
        """The cell's coordinates on ``build``."""
        return {**self.coordinates, **self.on.get(build, {})}

    @property
    def id(self) -> str:
        return f"{self.case}:{self.item}"


@dataclass(frozen=True)
class Grid:
    key: str
    question: str
    functions: tuple[str, ...]
    axes: Mapping[str, Axis]
    groups: tuple[tuple[str, ...], ...]
    exclusions: tuple[Exclusion, ...]
    expect: Mapping[str, Mapping[str, Any]]
    item_keys: frozenset[str]
    cells: tuple[Cell, ...]
    #: The behaviours whose every case is a cell's case or is listed in
    #: ``outside`` with the reason; empty for a grid that declares none.
    behaviours: tuple[str, ...] = ()
    outside: Mapping[str, str] = field(default_factory=dict)

    def required(self, build: str | None = None) -> Iterator[tuple[int, tuple[str, ...]]]:
        """Every coordinate a cell must fill on ``build``, as (group index,
        values in the order of the group's axes): each group's cross less its
        exclusions. With no build, every coordinate some recorded build
        requires."""
        builds = [build] if build is not None else list(recorded_builds())
        for index, group in enumerate(self.groups):
            for values in itertools.product(*(self.axes[axis].values for axis in group)):
                at = dict(zip(group, values, strict=True))
                if any(not any(rule.covers(at, b) for rule in self.exclusions) for b in builds):
                    yield index, values

    def covered(self, cell: Cell, build: str) -> Iterator[tuple[int, tuple[str, ...]]]:
        """The coordinates ``cell`` fills on ``build``: one per group whose axes
        all apply to it, none where the build refused the deck for another
        card."""
        if build in cell.refused and build not in cell.refused_itself:
            return
        at = cell.at(build)
        for index, group in enumerate(self.groups):
            values = tuple(at[axis] for axis in group)
            if all(
                value in self.axes[axis].values for axis, value in zip(group, values, strict=True)
            ):
                yield index, values

    def expected(self, cell: Cell, build: str) -> Any:
        """What the grid expects of ``cell``: the entry of the first axis in
        ``expect`` whose value the cell has an entry for."""
        at = cell.at(build)
        for axis, by_value in self.expect.items():
            if at.get(axis) in by_value:
                return by_value[at[axis]]
        raise KeyError(f"{cell.id}: the grid expects nothing of {at}")

    def unrecorded(self, build: str, recorded: set[str]) -> list[tuple[int, tuple[str, ...]]]:
        """The required coordinates no cell fills with a case ``recorded`` on
        ``build`` (a refusal is a recording)."""
        filled = {
            coordinate
            for cell in self.cells
            if cell.case in recorded
            for coordinate in self.covered(cell, build)
        }
        return [coordinate for coordinate in self.required(build) if coordinate not in filled]

    def cells_of(self, case: str) -> list[Cell]:
        return [cell for cell in self.cells if cell.case == case]


def _coordinates(table: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in table.items() if key not in _CELL_KEYS}


def _merge_on(*tables: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for table in tables:
        for build, values in table.get("on", {}).items():
            merged.setdefault(build, {}).update(values)
    return merged


def _grid(key: str, data: Mapping[str, Any]) -> Grid:
    axes = {
        name: Axis(name, tuple(spec["values"]), spec.get("default"))
        for name, spec in data["axis"].items()
    }
    item_axis = data["item_axis"]
    cells: list[Cell] = []
    for block in data["cell"]:
        shared = dict(block.get("at", {}))
        for item, spec in block["items"].items():
            own: dict[str, Any] = {item_axis: spec} if isinstance(spec, str) else dict(spec)
            values = {**shared, **_coordinates(own)}
            coordinates: dict[str, str] = {}
            for axis in axes:
                value = values.get(axis, axes[axis].default)
                if value is not None:
                    coordinates[axis] = value
            extra = {k: v for k, v in values.items() if k not in axes}
            refused: dict[str, str] = {**block.get("refused", {}), **own.get("refused", {})}
            itself = frozenset(own.get("refused", {}))
            on = _merge_on(block, own)
            for case in block["cases"]:
                cells.append(Cell(case, item, coordinates, on, refused, itself, extra))
    return Grid(
        key=key,
        question=data["question"],
        functions=tuple(data["functions"]),
        axes=axes,
        groups=tuple(tuple(group) for group in data["groups"]),
        exclusions=tuple(
            Exclusion(
                {k: tuple(v) for k, v in rule["at"].items()},
                rule["reason"],
                frozenset(rule.get("builds", ())),
            )
            for rule in data.get("exclude", ())
        ),
        expect={axis: dict(values) for axis, values in data.get("expect", {}).items()},
        item_keys=frozenset(data.get("item_keys", ())),
        cells=tuple(cells),
        behaviours=tuple(data.get("behaviours", ())),
        outside=dict(data.get("outside", {})),
    )


@functools.cache
def grids() -> dict[str, Grid]:
    with (INPUTS / CASES_FILE).open("rb") as handle:
        data = tomllib.load(handle)
    return {key: _grid(key, value) for key, value in data.get("grid", {}).items()}


def grid(key: str) -> Grid:
    return grids()[key]


def _module_path(name: str) -> Path:
    return REPO / name if name.startswith("tests/") else REPO / "src/ltspice_mcp" / name


@functools.cache
def _defined_names(module: str) -> frozenset[str]:
    """The names a module defines at its top level: functions, classes and
    assignments."""
    tree = ast.parse(_module_path(module).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def named_code_exists(name: str) -> bool:
    """Whether a ``model`` or ``functions`` entry names code that is there: a
    module (``lib/log_parser.py``), or a name a module defines at its top
    level (``lib/log_parser.py:printed_form``)."""
    module, _, function = name.partition(":")
    if not _module_path(module).exists():
        return False
    return not function or function in _defined_names(module)
