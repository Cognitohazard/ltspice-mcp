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


def test_which_text_a_part_attribute_is_written_with_is_answered_in_one_place():
    """``schematic_ops._attribute_texts`` holds the text of a part's attribute
    to its rule, whichever op writes it: a Value to what the part's element
    class reads whole, a Prefix to a letter. Each op that wrote a Value
    answered this itself, and only ``set_component_value`` checked it, so a
    value it refused was written by ``add_component`` and by
    ``set_component_attribute``."""
    rules = {"_validate_component_value", "_prefix_text"}
    callers = {
        f"{module}:{function.name}"
        for module, tree in modules().items()
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in rules
    }
    assert callers == {"lib/schematic_ops.py:_attribute_texts"}
    # An attribute written any other way passes no rule.
    writers = {
        f"{module}:{name}"
        for module, tree in modules().items()
        for name in _attribute_writes_in(tree)
    }
    assert writers == set(_ATTRIBUTE_WRITERS)


#: The schematic editor's methods that write a part's attributes, and the
#: methods of a part's attributes that store into them.
_ATTRIBUTE_SETTERS = frozenset(
    {"set_component_attribute", "set_component_value", "set_component_parameters"}
)
_ATTRIBUTE_STORES = frozenset({"update", "setdefault"})

#: The functions that write a part's attributes, each with the reason the text
#: it writes has passed ``_attribute_texts``.
_ATTRIBUTE_WRITERS: dict[str, str] = {
    "lib/schematic_ops.py:create_component": (
        "stores the texts _attribute_texts gave for the part about to be placed"
    ),
    "lib/schematic_ops.py:apply_op_inplace": (
        "set_component_attribute writes the text _attribute_texts gave for the op"
    ),
    "lib/schematic_ops.py:_set_or_create_value": (
        "writes the Value _apply_component_value held to _attribute_texts first"
    ),
    "lib/schematic_ops.py:_set_instance_params": (
        "sets a parameter of the Value _apply_component_value held to "
        "_attribute_texts first, in the slot that already gives it"
    ),
}


def _attribute_writes_in(tree: ast.AST, prefix: str = "") -> set[str]:
    """The functions in ``tree`` (qualified by their enclosing ones) that write
    a part's attributes: ``x.attributes[k] = v``, ``x.attributes.update(...)``
    or ``setdefault``, or a call of one of the editor's setters."""
    found: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            found |= _attribute_writes_in(node, f"{prefix}{node.name}.")
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = f"{prefix}{node.name}"
        found |= _attribute_writes_in(node, f"{name}.")
        if any(_writes_attributes(inner) for inner in _own_nodes(node)):
            found.add(name)
    return found


def _own_nodes(function: ast.AST) -> list[ast.AST]:
    """The nodes of ``function``'s body, not those of a function or class
    defined in it, which is named on its own."""
    own: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        own.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return own


def _is_attributes(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "attributes"


def _writes_attributes(node: ast.AST) -> bool:
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(
            isinstance(target, ast.Subscript) and _is_attributes(target.value)
            for target in targets
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in _ATTRIBUTE_SETTERS:
            return True
        return node.func.attr in _ATTRIBUTE_STORES and _is_attributes(node.func.value)
    return False


def test_the_attribute_write_rule_catches_what_it_names():
    tree = ast.parse(
        "def stored(part):\n"
        "    part.attributes['Value'] = '1 k'\n"
        "def added(part):\n"
        "    part.attributes['SpiceLine'] += ' tc=1'\n"
        "def updated(part, texts):\n"
        "    part.attributes.update(texts)\n"
        "def defaulted(part):\n"
        "    part.attributes.setdefault('Prefix', 'r')\n"
        "def set_directly(editor):\n"
        "    editor.set_component_attribute('R1', 'Value', '1 k')\n"
        "def set_value(editor):\n"
        "    editor.set_component_value('R1', '1 k')\n"
        "class Ops:\n"
        "    def outer(self, editor):\n"
        "        def inner():\n"
        "            editor.set_component_parameters('R1', tc=1)\n"
        "        return inner\n"
        "def read(part, other):\n"
        "    other.update(part.attributes)\n"
        "    return part.attributes.get('Value'), part.attributes.pop('InstName', None)\n"
    )
    assert _attribute_writes_in(tree) == {
        "stored",
        "added",
        "updated",
        "defaulted",
        "set_directly",
        "set_value",
        "Ops.outer.inner",
    }
