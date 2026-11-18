"""Which net, trace or device a name stands for is answered in one place.

``lib/deck_nodes.py`` says which net a deck's node name is, ``lib/device_names.py``
how a reader spells a path through subcircuit instances and which device a
name belongs to, ``lib/trace_names.py`` which trace of a result a name reads,
and ``lib/spice_lex_ops.py`` what LTspice's export does to an instance's name
(the ``§`` and the added ``X``). Each is told who reads the deck or wrote the
result. Outside them no code:

- names the ground set, or compares a node name with ``0`` or ``gnd``;
- joins path segments into a trace or device name;
- sets ``xvii_names``, or passes a literal for ``xvii``;
- holds a constant that stands for a simulator when it reads names;
- adds or removes ``§`` or the added ``X``;
- builds a ``hierarchy.SemanticProfile`` but through ``SemanticProfile.of``.

The scan reads the package's syntax trees, so a docstring or comment that
names a rule is not a use of it. Each site it finds outside the owners must
be on ``ALLOWED`` with its reason; a site on the list that the scan no longer
finds fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast

from tests._source_scan import modules, scopes

#: The modules that own the answers.
OWNERS = frozenset(
    {"lib/deck_nodes.py", "lib/device_names.py", "lib/trace_names.py", "lib/spice_lex_ops.py"}
)

#: The functions of ``deck_nodes`` a caller tells which simulator reads a name.
_READERS_OF_NODES = frozenset({"node_kind", "settled_node_kind", "is_ground"})

#: Each site outside the owners, as (module, scope, rule), with its reason.
ALLOWED: dict[tuple[str, str, str], str] = {
    # The sheet's own answerer: which points of a sheet are one net, and a
    # label named GND as ground there, are LTspice's sheet rules, outside the
    # deck-and-result resolver.
    ("lib/connectivity.py", "is_ground", "ground"): "the sheet's own rule for a GND label",
    # Display code: what a drawing or a reply shows, not which net a name is.
    ("lib/schematic_scene.py", "scene_of_sheet", "ground"): "draws a sheet's ground flag",
    # Text a reply or a capability row shows, naming LTspice's export rule.
    ("lib/capabilities.py", "<module>", "section_sign"): "describes the ngspice preparation",
    ("tools/verify.py", "<module>", "section_sign"): "explains the export's added X in a reply",
    # The log readers accept a part's name as LTspice writes it, § included.
    ("lib/log_conditions.py", "<module>", "section_sign"): "matches a log line naming R§Load",
    ("lib/log_parser.py", "<module>", "section_sign"): "matches a log line naming R§Load",
    # The comparison's own key for ground: deck_nodes folded every ground
    # name to it when the deck was parsed for its reader.
    ("lib/netlist_graph.py", "_wl_refine.add_side", "ground"): "the graph's canonical ground net",
    # A reply's field path, not a circuit name.
    ("lib/projection.py", "absent_path_warnings", "path"): "joins a reply's field path",
    ("tools/inspect_tools.py", "_deck_node", "ground"): (
        "the netlist net query compares names as a sheet does (connectivity.name_key); "
        "net:NAME is the sheet side"
    ),
    ("tools/inspect_tools.py", "_net_netlist_payload", "ground"): (
        "says in the reply that the net query joined gnd to node 0"
    ),
    # The native PDK statistics run on ngspice only.
    ("lib/native_inputs.py", "NativeCaseValidator.validate", "simulator"): (
        "native statistics run only on ngspice, in the mode the policy pins"
    ),
}


def _constant(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _rules(node: ast.AST) -> set[str]:
    """The rules ``node`` itself breaks, by name."""
    found: set[str] = set()
    if isinstance(node, ast.Compare):
        compared = [node.left, *node.comparators]
        listed = [
            element
            for side in compared
            if isinstance(side, (ast.Tuple, ast.List, ast.Set))
            for element in side.elts
        ]
        if (
            any(_constant(side) in ("0", "gnd", "GND") for side in compared)
            or any((_constant(side) or "").casefold() == "gnd" for side in listed)
            or any(
                isinstance(side, ast.Name) and side.id in ("GROUND", "_GROUND")
                for side in compared
            )
        ):
            found.add("ground")
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = node.id if isinstance(node, ast.Name) else node.attr
        if name in ("GROUND_NAMES", "_GROUND_NODES"):
            found.add("ground")
        if name in ("SECTION_SIGN", "_READ_AS"):
            found.add("section_sign" if name == "SECTION_SIGN" else "simulator")
    if isinstance(node, ast.Set) and any(_constant(e) in ("gnd", "0") for e in node.elts):
        found.add("ground")
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "join"
            and _constant(func.value) in (":", ".")
            and _joins_a_name(node)
        ):
            found.add("path")
        for keyword in node.keywords:
            if keyword.arg == "xvii_names":
                found.add("xvii")
            if keyword.arg == "xvii" and isinstance(keyword.value, ast.Constant):
                found.add("xvii")
            if (
                keyword.arg == "simulator"
                and _constant(keyword.value) is not None
                and _callee(node) in _READERS_OF_NODES
            ):
                found.add("simulator")
        if _callee(node) == "Producer" and node.args and _constant(node.args[0]) is not None:
            found.add("simulator")
        named = [a.id for a in node.args if isinstance(a, ast.Name)]
        if (_callee(node) == "deck_producer" or "deck_producer" in named) and any(
            _constant(a) for a in node.args
        ):
            found.add("simulator")
        if (
            _callee(node) == "startswith"
            and node.args
            and _constant(node.args[0]) == "xx"
            and isinstance(func, ast.Attribute)
        ):
            found.add("added_x")
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and "§" in node.value:
        found.add("section_sign")
    if isinstance(node, ast.Call) and _callee(node) == "SemanticProfile":
        found.add("profile")
    if isinstance(node, ast.arguments):
        named = [*node.posonlyargs, *node.args]
        defaults = dict(
            zip(
                [a.arg for a in named][len(named) - len(node.defaults) :],
                node.defaults,
                strict=True,
            )
        )
        defaults.update(
            (a.arg, d)
            for a, d in zip(node.kwonlyargs, node.kw_defaults, strict=True)
            if d is not None
        )
        if isinstance(defaults.get("xvii"), ast.Constant):
            found.add("xvii")
    return found


def _joins_a_name(call: ast.Call) -> bool:
    """Whether a ``":".join`` or ``".".join`` builds a name inside a trace or
    a device: it is put inside ``V(...)``/``I(...)`` text, or joins what a
    path, scope or instance holds."""
    text = ast.unparse(call.args[0]) if call.args else ""
    return any(word in text.casefold() for word in ("scope", "path", "segment", "instance"))


def _sites() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for module, tree in modules().items():
        if module in OWNERS:
            continue
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
        }
        for scope, nodes in scopes(tree):
            for node in nodes:
                if id(node) in docstrings:
                    continue
                for rule in _rules(node):
                    found.add((module, scope, rule))
    return found


def test_only_the_owners_answer_what_a_name_stands_for():
    found = _sites()
    unlisted = sorted(found - ALLOWED.keys())
    assert unlisted == [], unlisted


def test_the_allow_list_holds_only_sites_that_remain():
    stale = sorted(ALLOWED.keys() - _sites())
    assert stale == [], stale


# --- One fold for names ------------------------------------------------------

#: The modules that read a node, net, trace or device name of a deck or a
#: result. In them a name is folded only by ``device_names.fold_name``.
NAME_MODULES = frozenset(
    {
        "lib/connectivity.py",
        "lib/deck_nodes.py",
        "lib/decoded_raw.py",
        "lib/device_names.py",
        "lib/hierarchy.py",
        "lib/lint_rules.py",
        "lib/log_parser.py",
        "lib/metrics.py",
        "lib/netlist_graph.py",
        "lib/raw_parser.py",
        "lib/spice_lex_views.py",
        "lib/spice_validator.py",
        "lib/trace_names.py",
    }
)

#: Words that mark what a fold is applied to as a node, net, trace or device
#: name.
_NAME_WORDS = (
    "node",
    "net",
    "port",
    "terminal",
    "trace",
    "signal",
    "sig",
    "probe",
    "global",
    "instance",
    "segment",
    "pin",
    "label",
    "device",
    "variable.name",
    "ref",
)

#: Each ``lower()``/``casefold()`` in a name module whose operand reads as a
#: name but is not one, as (module, scope, operand), with its reason.
FOLD_ALLOWED: dict[tuple[str, str, str], str] = {
    # A model or subcircuit an instance calls is not a node, net or trace:
    # how a reader matches a model's name is not this rule's question.
    (
        "lib/hierarchy.py",
        "_rewritten_for_m.callee",
        "InstanceLine.from_card(item.card).model or ''",
    ): ("the subcircuit an instance calls"),
    ("lib/lint_rules.py", "_model_missing", "instance.model"): "the model an instance names",
    ("lib/netlist_graph.py", "_diff_matched_pair", "_type_signature(cc)"): (
        "a component's letter and model, compared between two decks"
    ),
    ("lib/netlist_graph.py", "_diff_matched_pair", "_type_signature(rc)"): (
        "a component's letter and model, compared between two decks"
    ),
}


def _fold_sites() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for module, tree in modules().items():
        if module not in NAME_MODULES:
            continue
        for scope, nodes in scopes(tree):
            for node in nodes:
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("lower", "casefold")
                    and not node.args
                ):
                    continue
                operand = ast.unparse(node.func.value)
                if any(word in operand.casefold() for word in _NAME_WORDS):
                    found.add((module, scope, operand))
    return found


def test_a_name_is_folded_only_by_the_one_fold():
    """``device_names.fold_name`` folds ASCII letters and nothing more, as
    the recordings establish; ``lower()`` and ``casefold()`` fold more and
    differ from each other on names such as ``maß`` and ``mass``."""
    unlisted = sorted(_fold_sites() - FOLD_ALLOWED.keys())
    assert unlisted == [], unlisted


def test_the_fold_allow_list_holds_only_sites_that_remain():
    stale = sorted(FOLD_ALLOWED.keys() - _fold_sites())
    assert stale == [], stale


def test_no_second_reader_of_a_signal_is_left():
    """The lint's own normalizer of a signal name is gone: it reads names
    through ``device_names.deck_signal``."""
    defined = {
        node.name
        for tree in modules().values()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_normalize_signal" not in defined


def test_no_code_reads_input_noise_from_part_of_a_name():
    """Whether a trace is input-referred noise is ``trace_names.is_input_noise``,
    by the whole name: a device named ``Rinoise`` contributes ``V(Rinoise)``."""
    found = []
    for module, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if (
                    isinstance(node, ast.Compare)
                    and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and "noise" in node.left.value.casefold()
                ):
                    found.append((module, scope, ast.unparse(node)))
    assert found == [], found
