"""Each question about a sheet is answered in one place, checked rather than documented.

Two readers that each answer one question, such as which points of a sheet are
one net or which directives a text names, drift apart, and then one reply says
two things. These checks walk the server's source and fail where a second
answer starts.
"""

from __future__ import annotations

import ast
import functools

from tests._source_scan import modules, own_nodes, scopes


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


def _is_parts(node: ast.AST) -> bool:
    """Whether ``node`` is a collection of parts keyed by reference, whatever it
    is called: ``x.components`` or ``x.get_components()``, or either's
    ``.keys()``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "keys"
    ):
        node = node.func.value
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Attribute) and node.func.attr == "get_components"
    return isinstance(node, ast.Attribute) and node.attr == "components"


def _looks_up_a_part(node: ast.AST) -> bool:
    """Whether ``node`` finds a part by its reference spelled exactly: ``r in
    parts``, ``r not in parts``, ``parts[r]``, ``parts.get(r)`` or
    ``get_component(r)`` on any receiver."""
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(
            _is_parts(side) for side in node.comparators
        )
    if isinstance(node, ast.Subscript):
        return isinstance(node.ctx, ast.Load) and _is_parts(node.value)
    if isinstance(node, ast.Call):
        callee = node.func
        if isinstance(callee, ast.Name):
            return callee.id == "get_component"
        if isinstance(callee, ast.Attribute):
            return callee.attr == "get_component" or (
                callee.attr == "get" and _is_parts(callee.value)
            )
    return False


def _part_lookups_in(tree: ast.AST) -> set[str]:
    """The scopes in ``tree`` that find a part by its reference spelled exactly."""
    return {name for name, nodes in scopes(tree) if any(_looks_up_a_part(node) for node in nodes)}


#: The scopes that find a part by a reference spelled exactly, each with the
#: reason that reference is already one the parts are keyed by.
_PART_LOOKUPS: dict[str, str] = {
    "lib/schematic_ops.py:_attributes_of": (
        "reads the part apply_op_inplace resolved with part_reference"
    ),
    "lib/schematic_ops.py:_apply_component_value": (
        "sets the part apply_op_inplace resolved with part_reference"
    ),
    "lib/schematic_ops.py:_set_instance_params": (
        "sets the part _apply_component_value was given, resolved with part_reference"
    ),
    "lib/schematic_ops.py:placed_geometry": (
        "takes a reference the editor holds: one part_reference resolved, or one "
        "read from the editor's own parts"
    ),
    "lib/schematic_ops.py:element_class": (
        "takes a reference the editor holds; a part about to be placed has none yet "
        "and is read from its reference"
    ),
    "lib/schematic_ops.py:resolve_pin": "looks the part up after part_reference resolved it",
    "lib/schematic_ops.py:apply_op_inplace": (
        "each op looks its part up after part_reference resolved it"
    ),
    "lib/variations.py:_DeckTargets.component_sites": (
        "a deck's cards keyed by case-folded name, not a sheet's parts"
    ),
    "lib/variations.py:_resolve_instance_param": (
        "a deck's cards keyed by case-folded name, not a sheet's parts"
    ),
}


def test_a_reference_is_looked_up_among_a_sheets_parts_in_one_place():
    """``sheet.references_named`` says which part a reference names, as LTspice
    compares references, for every op and for ``inspect``; ``part_reference``
    applies it to an editor. A lookup by the reference as spelled (``in``,
    ``[]``, ``.get`` or ``get_component`` on any collection of parts) compares
    it exactly, and refused ``r1`` for a part ``R1`` that ``add_component`` said
    it names."""
    lookups = {
        f"{module}:{name}" for module, tree in modules().items() for name in _part_lookups_in(tree)
    }
    assert lookups == set(_PART_LOOKUPS)


def test_the_part_lookup_rule_catches_what_it_names():
    tree = ast.parse(
        "def member(editor, ref):\n"
        "    return ref in editor.components\n"
        "def absent(sheet, ref):\n"
        "    return ref not in sheet.get_components()\n"
        "def keyed(parts, ref):\n"
        "    return ref in parts.components.keys()\n"
        "def indexed(asc, ref):\n"
        "    return asc.components[ref].symbol\n"
        "def got(doc, ref):\n"
        "    return doc.components.get(ref)\n"
        "async def handler(state, ref):\n"
        "    return state.editor.get_component(ref)\n"
        "def bare(ref):\n"
        "    return get_component(ref)\n"
        "class Ops:\n"
        "    found = REF in EDITOR.components\n"
        "    def outer(self, editor, ref):\n"
        "        def inner():\n"
        "            return editor.components[ref]\n"
        "        return inner\n"
        "if 'R1' in EDITOR.components:\n"
        "    pass\n"
        "def written(editor, ref, part):\n"
        "    editor.components[ref] = part\n"
        "    for name in editor.components:\n"
        "        pass\n"
        "    return references_named(ref, editor.components), len(editor.components)\n"
    )
    assert _part_lookups_in(tree) == {
        "member",
        "absent",
        "keyed",
        "indexed",
        "got",
        "handler",
        "bare",
        "Ops",
        "Ops.outer.inner",
        "<module>",
    }


def _names_sheet_suffix(node: ast.AST) -> bool:
    """Whether ``node`` holds the text ``.asc``, in any case."""
    return any(
        isinstance(inner, ast.Constant)
        and isinstance(inner.value, str)
        and inner.value.lower() == ".asc"
        for inner in ast.walk(node)
    )


def _tests_sheet_suffix(node: ast.AST) -> bool:
    """Whether ``node`` tests a text against ``.asc``: a comparison (``==``,
    ``!=``, ``in``) with it on either side, ``endswith`` given it, or a
    ``case`` matching it."""
    if isinstance(node, ast.Compare):
        return any(_names_sheet_suffix(side) for side in (node.left, *node.comparators))
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "endswith"
            and any(_names_sheet_suffix(argument) for argument in node.args)
        )
    if isinstance(node, ast.match_case):
        return _names_sheet_suffix(node.pattern)
    return False


#: The one scope that tests a path against ``.asc``.
_SHEET_SUFFIX_OWNER = "lib/asc_document.py:is_asc"


def _sheet_suffix_tests_in(tree: ast.AST) -> set[str]:
    """The scopes in ``tree`` that test a text against ``.asc``."""
    return {
        name for name, nodes in scopes(tree) if any(_tests_sheet_suffix(node) for node in nodes)
    }


def test_whether_a_path_is_a_sheet_is_answered_by_is_asc():
    """``asc_document.is_asc`` says whether a path names a sheet; a comparison
    with ``.asc`` written anywhere else, in any function, class or module body
    and however the suffix was folded first, is a second answer."""
    tests = {
        f"{module}:{name}"
        for module, tree in modules().items()
        for name in _sheet_suffix_tests_in(tree)
    }
    assert tests == {_SHEET_SUFFIX_OWNER}


def test_the_sheet_suffix_rule_catches_what_it_names():
    tree = ast.parse(
        "def equal(path):\n"
        "    return path.suffix.lower() == '.asc'\n"
        "def unequal(path):\n"
        "    return path.suffix.casefold() != '.ASC'\n"
        "def among(path):\n"
        "    return path.suffix in ('.asc', '.net')\n"
        "def in_set(path):\n"
        "    return path.suffix.lower() not in frozenset({'.asc'})\n"
        "def ends(name):\n"
        "    return name.lower().endswith('.asc')\n"
        "def ends_any(name):\n"
        "    return name.endswith(('.net', '.asc'))\n"
        "def renamed(path):\n"
        "    return path == path.with_suffix('.asc')\n"
        "def matched(path):\n"
        "    match path.suffix:\n"
        "        case '.asc':\n"
        "            return True\n"
        "async def handler(args):\n"
        "    return args.path.endswith('.asc')\n"
        "class Kinds:\n"
        "    SHEET = SUFFIX == '.asc'\n"
        "    def outer(self, path):\n"
        "        def inner():\n"
        "            return path.suffix == '.asc'\n"
        "        return inner\n"
        "if SUFFIX == '.asc':\n"
        "    pass\n"
        "def written(path):\n"
        "    sheet = path.with_suffix('.asc')\n"
        "    return sheet, {'.asc', '.net'} - {'.asc'}, is_asc(sheet), kind == 'asc'\n"
    )
    assert _sheet_suffix_tests_in(tree) == {
        "equal",
        "unequal",
        "among",
        "in_set",
        "ends",
        "ends_any",
        "renamed",
        "matched",
        "handler",
        "Kinds",
        "Kinds.outer.inner",
        "<module>",
    }


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
        if any(_writes_attributes(inner) for inner in own_nodes(node)):
            found.add(name)
    return found


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


#: spicelib's own searches for a file a sheet names: each walks every folder
#: below the sheet's and the server's working directory, which neither LTspice
#: build looks in (``docs/spicelib_bugs.md``, Bug 37). ``_get_subcircuit`` is
#: the loader that calls them for a part's model library and a block's sheet.
_SPICELIB_SEARCHES = ("_asy_file_find", "_lib_file_find", "_get_subcircuit")


def test_a_sheets_files_are_found_by_the_sessions_library_alone():
    """Which file a sheet's symbol, or a block's own sheet, is decided by
    ``lib/symbol_library.py`` (the session's ``SymbolLibrary``), for the edit,
    the check, the inspection and the drawing alike, wherever the server
    runs. No module calls spicelib's own search for one. Ruff refuses an
    import of spicelib's search functions and of the library's own rules
    (``banned-api`` in ``pyproject.toml``); a method it cannot see is checked
    here."""
    assert [where for name in _SPICELIB_SEARCHES for where in _called(name)] == []
