"""Dotted-path field projection over result rows.

A caller naming ``fields=["value.phase_margin_deg"]`` gets back the SAME row
shape with fewer keys, never a flattened one — so code that reads
``row["value"]["phase_margin_deg"]`` reads it identically whether or not the
projection ran. Shared by ``analyze_results`` and ``run_experiments``; split out
of ``tools/_base`` because it is pure dict work with no MCP or session in it.
"""

from typing import Any

# --- dotted field projection (shared by analyze_results and run_experiments) ---
#
# A keep-plan mirrors the row's own nesting: ``None`` keeps a whole subtree, a
# nested plan keeps only the named keys inside it. Projection therefore returns
# the SAME shape with fewer keys, never a flattened one — caller code that reads
# ``row["value"]["phase_margin_deg"]`` reads it identically either way.
KeepPlan = dict[str, "KeepPlan | None"]

ABSENT = object()


def split_field_path(path: str) -> list[str]:
    r"""Segments of a dotted path, where ``\.`` is a dot INSIDE a key.

    A row key is whatever the simulator called the thing, and plenty of them
    carry dots of their own — ngspice spells a subcircuit node ``v(x1.out)``
    and a subcircuit device parameter ``@m.x1.m1[gm]``. Splitting those on
    every dot addresses a nesting that does not exist, so the escape is what
    makes the most ordinary op-point key projectable at all.
    """
    segments: list[str] = []
    current: list[str] = []
    escaped = False
    for char in path:
        if escaped:
            # Only the dot is escapable; anything else keeps its backslash, so
            # a path that never meant to escape reads back unchanged.
            current.append(char if char == "." else "\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            segments.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    segments.append("".join(current))
    return segments


def escape_field_segment(segment: str) -> str:
    r"""Spell one row key as the path segment that addresses it.

    Only the dot is escaped, mirroring the split: a backslash means nothing on
    its own there, so doubling one here would put a character in the path that
    reading it back would not remove.
    """
    return segment.replace(".", "\\.")


def keep_plan(paths: list[str]) -> KeepPlan:
    """Group dotted ``paths`` into a nested keep-plan, in first-named order."""
    return _plan_for([split_field_path(path) for path in paths])


def _plan_for(paths: list[list[str]]) -> KeepPlan:
    order: list[str] = []
    nested: dict[str, list[list[str]]] = {}
    whole: set[str] = set()
    for segments in paths:
        root, rest = segments[0], segments[1:]
        if root not in order:
            order.append(root)
        if rest:
            nested.setdefault(root, []).append(rest)
        else:
            # A bare key wins over any dotted sibling: asking for the subtree and
            # a leaf inside it means the subtree.
            whole.add(root)
    return {root: None if root in whole else _plan_for(nested[root]) for root in order}


def project_row(row: dict[str, Any], plan: KeepPlan) -> dict[str, Any]:
    """A NEW row carrying only the planned keys.

    Never mutates ``row``: the full record is still read after this call — spec
    attribution, reductions and the analyzed-identity accounting all index keys
    a projection drops — so this is a view built for emission, not an edit.
    """
    kept: dict[str, Any] = {}
    for key, sub in plan.items():
        value = row.get(key, ABSENT)
        if value is ABSENT:
            continue
        if sub is None:
            kept[key] = value
        elif isinstance(value, dict):
            nested = project_row(value, sub)
            if nested:
                kept[key] = nested
    return kept
