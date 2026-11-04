"""Each question about a sheet is answered in one place, checked rather than documented.

Two readers that each answer one question, such as which points of a sheet are
one net or which directives a text names, drift apart, and then one reply says
two things. These checks walk the server's source and fail where a second
answer starts.
"""

from __future__ import annotations

import ast
import functools

from tests._source_scan import modules


@functools.cache
def _called(name: str) -> list[str]:
    """Where a function or method of that name is called, as ``module:line``."""
    return [
        f"{module}:{node.lineno}"
        for module, tree in modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            or (isinstance(node.func, ast.Name) and node.func.id == name)
        )
    ]


#: Names of the grouping of a sheet's points by wiring alone, which could not
#: hold two of LTspice's rules (a wire straight between two pins of one part
#: joins nothing; of one part's pins alone on a point only the last in
#: SpiceOrder touches another part's) and was read beside ``Nets``.
_WIRING_ONLY_NETS = frozenset(
    {"partition", "label_folded_nets", "net_members", "NetPartition", "labels_per_coord"}
)


def test_which_points_are_one_net_is_answered_by_the_nets_ltspice_makes():
    """Every reader of a sheet's nets reads ``connectivity.nets_by_ltspice``
    (``Nets``, ``NamedNets``): ``inspect(kind="net")``, the pin legend, the
    sheet findings, the edit refusals and the route rules. Nothing outside
    ``lib/connectivity.py`` imports another grouping, and it defines none."""
    imported = [
        f"{module}:{node.lineno} {alias.name}"
        for module, tree in modules().items()
        if module != "lib/connectivity.py"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in _WIRING_ONLY_NETS
    ]
    assert imported == []
    defined = [
        node.name
        for node in ast.walk(modules()["lib/connectivity.py"])
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _WIRING_ONLY_NETS
    ]
    assert defined == []


def _is_parts(node: ast.expr) -> bool:
    """Whether ``node`` is a schematic editor's parts, as the server's code names
    one: ``editor.components`` or ``editor.get_components()``."""
    name = "components"
    if isinstance(node, ast.Call):
        node, name = node.func, "get_components"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "editor"
    )


def test_a_reference_is_looked_up_among_a_sheets_parts_in_one_place():
    """``sheet.references_named`` says which part a reference names, as LTspice
    compares references, for every op and for ``inspect``. A test of a name's
    membership in an editor's parts compares it exactly, and refused ``r1`` for
    a part ``R1`` that ``add_component`` said it names."""
    found = [
        f"{module}:{node.lineno}"
        for module, tree in modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        and any(_is_parts(side) for side in node.comparators)
    ]
    assert found == []


def test_whether_a_path_is_a_sheet_is_answered_by_is_asc():
    """``asc_document.is_asc`` says whether a path names a sheet; a comparison
    with ``.asc`` written anywhere else, however the suffix was folded first,
    is a second answer."""
    found = [
        f"{module}:{node.lineno}"
        for module, tree in modules().items()
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.name != "is_asc"
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(side, ast.Constant) and side.value == ".asc"
            for side in (node.left, *node.comparators)
        )
    ]
    assert found == []


def test_no_text_is_matched_by_spicelibs_instruction_removers():
    """``remove_directive`` and ``move`` select texts by ``_text_matcher``.
    spicelib's removers match directives only, one by substring and the first
    hit, the other anchored at the start and in any case."""
    assert _called("remove_Xinstruction") == []
    assert _called("remove_instruction") == []
