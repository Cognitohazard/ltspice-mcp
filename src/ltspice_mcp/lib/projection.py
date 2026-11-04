"""Dotted-path field projection over result rows.

A caller naming ``fields=["value.phase_margin_deg"]`` gets back the SAME row
shape with fewer keys, never a flattened one — so code that reads
``row["value"]["phase_margin_deg"]`` reads it identically whether or not the
projection ran. Shared by ``analyze_results`` and ``run_experiments``; split out
of ``tools/_base`` because it is pure dict work with no MCP or session in it.
"""

from collections.abc import Sequence
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


def resolve_field_paths(
    paths: Sequence[str],
    roots: Sequence[str],
    *,
    argument: str,
    bare_root: str | None = None,
) -> list[str]:
    r"""``paths`` as a projection reads them, checked against the row's own keys.

    The one reading of a dotted projection path, for every argument that takes
    one (``include.fields`` on ``analyze_results`` and on an attached
    analysis, ``run_fields`` on ``run_experiments`` and ``jobs(runs)``), each
    passing the top-level keys its rows carry as ``roots``:

    - a path with an empty segment (``'value.'``, ``'a..b'``) is refused;
    - a path starts at one of ``roots``, or is refused naming them: a
      projection that silently keeps nothing is worse than none, so an unknown
      root is an error rather than rows the caller has to explain;
    - a one-segment name that is not a root reads under ``bare_root`` when the
      row nests its values there (``'rms'`` is ``'value.rms'``).

    A dot inside a key's own name is written ``\.``. The paths come back in
    the order given, repeats kept; ``ValueError`` names ``argument``.
    """
    resolved: list[str] = []
    for path in paths:
        segments = split_field_path(path)
        if any(not segment for segment in segments):
            raise ValueError(f"{argument} path {path!r} has an empty segment")
        root = segments[0]
        if root not in roots:
            if len(segments) != 1 or bare_root is None:
                raise ValueError(
                    f"{argument} path {path!r} starts at unknown row key {root!r}; "
                    f"choose one of: {', '.join(roots)}"
                )
            # A bare name is the number's own name, which lives under bare_root.
            path = f"{bare_root}.{path}"
        resolved.append(path)
    return resolved


def at_path(row: dict[str, Any], segments: Sequence[str]) -> Any:
    """The value at ``segments``, or ``ABSENT`` when one of them is missing."""
    node: Any = row
    for segment in segments:
        if not isinstance(node, dict) or segment not in node:
            return ABSENT
        node = node[segment]
    return node


def absent_path_warnings(
    rows: Sequence[dict[str, Any]], paths: Sequence[str], *, argument: str, rows_name: str
) -> list[str]:
    """One warning per requested path that no row in ``rows`` carries.

    A root key is checked when the path is read (:func:`resolve_field_paths`),
    so only a path reaching INTO a row can miss here, and it can miss
    legitimately: a projection is one argument over rows of several shapes.
    Naming the keys that are there turns a row missing the requested key from
    a silence into the next call's argument. No rows, no warnings: nothing was
    there to miss.
    """
    if not rows:
        return []
    warnings: list[str] = []
    for path in paths:
        segments = split_field_path(path)
        if len(segments) == 1:
            continue
        if any(at_path(row, segments) is not ABSENT for row in rows):
            continue
        parent_segments = segments[:-1]
        parent = ".".join(escape_field_segment(segment) for segment in parent_segments)
        present = sorted(
            {
                key
                for row in rows
                for node in (at_path(row, parent_segments),)
                if isinstance(node, dict)
                for key in node
            }
        )
        # Naming a dotted key as it must be SPELLED, not just as it reads, is
        # the difference between a warning and a fix: the caller is one escape
        # away from the value, and no amount of staring at the key says so.
        addressable = ", ".join(escape_field_segment(key) for key in present)
        warnings.append(
            f"{argument} path {path!r} is absent from every {rows_name}; "
            + (
                f"keys present at {parent!r}: {addressable}"
                if present
                else f"no row reaches {parent!r} — a key whose own name contains a "
                r"'.' is one segment, so address it with '\.' (e.g. 'value.v(x1\.out)')"
            )
        )
    return warnings


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
