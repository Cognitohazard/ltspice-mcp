"""Which module answers each question about a deck's text, checked rather than documented.

A deck read line by line, a probe pattern written again, an include target
cut out of a directive by hand: each was a second answer to a question one
module owns, and each came to disagree with it. These checks keep the answers
with their owners. An allow-list entry is a decision written down with its
reason, and the lists may only shrink. A name only its owner may call, such as
the plain value reader, is banned in ``pyproject.toml`` instead.
"""

from __future__ import annotations

import ast
import functools
import re

from tests._source_scan import modules


@functools.cache
def _functions(tree: ast.AST, prefix: str = "") -> tuple[tuple[str, ast.AST], ...]:
    """Every function in ``tree`` with its qualified name.

    A tuple, because the cache hands every caller the same object and a
    generator is spent once its first caller has read it.
    """
    found: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = f"{prefix}{node.name}"
            found.append((name, node))
            found.extend(_functions(node, f"{name}."))
        elif isinstance(node, ast.ClassDef):
            found.extend(_functions(node, f"{prefix}{node.name}."))
    return tuple(found)


def _calls_method(node: ast.AST, method: str) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == method
        for call in ast.walk(node)
    )


def _strings(node: ast.AST) -> list[str]:
    """The string constants ``node`` is, or holds in a tuple."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Tuple):
        return [text for element in node.elts for text in _strings(element)]
    return []


# --------------------------------------------------------------------------
# A deck's cards are read through the lexer
# --------------------------------------------------------------------------

#: Functions that walk a text's lines and test each for a directive's dot or a
#: continuation's plus. Such a walk reads a deck's physical lines, so it reads
#: the title, the cards after ``.end`` and half of a continued card;
#: ``spice_lex.lex_deck`` is the reading of a deck.
_LINE_WALKS: dict[str, str] = {}


def _line_walks_in(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for name, function in _functions(tree):
        for loop in ast.walk(function):
            if not isinstance(loop, ast.For) or not _calls_method(loop.iter, "splitlines"):
                continue
            tested = [
                text
                for call in ast.walk(loop)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "startswith"
                and call.args
                for text in _strings(call.args[0])
            ]
            if any(text[:1] in {".", "+"} for text in tested):
                found.add(name)
    return found


def test_no_deck_is_read_line_by_line_outside_the_lexer():
    walks = {
        f"{module}:{name}"
        for module, tree in modules().items()
        if module != "lib/spice_lex.py"
        for name in _line_walks_in(tree)
    }
    assert walks == set(_LINE_WALKS)


def test_every_rule_sees_every_function_of_a_module():
    """The walk is cached per tree, and each rule here reads the same trees,
    so the second rule to ask must be given the functions the first was."""
    tree = modules()["lib/spice_lex.py"]
    first = [name for name, _ in _functions(tree)]
    assert "lex" in first
    assert [name for name, _ in _functions(tree)] == first


def test_the_line_walk_rule_catches_what_it_names():
    tree = ast.parse(
        "def cards(text):\n"
        "    for number, line in enumerate(text.splitlines(), 1):\n"
        "        if line.strip().startswith(('+', '*')):\n"
        "            pass\n"
        "def directives(text):\n"
        "    for line in text.splitlines():\n"
        "        if line.lstrip().lower().startswith('.tran'):\n"
        "            pass\n"
        "def comments(text):\n"
        "    for line in text.splitlines():\n"
        "        if line.startswith('*'):\n"
        "            pass\n"
    )
    assert _line_walks_in(tree) == {"cards", "directives"}


#: Functions that call ``spice_lex.lex`` itself, with what each lexes. ``lex``
#: reads every line of a text as a card, a deck's title and the cards after
#: its ``.END`` among them, so a whole deck is read through ``lex_deck`` or,
#: by an edit that writes every card back, ``lex_for_edit``.
_PLAIN_LEX: dict[str, str] = {
    "lib/library_parser.py:parse_library_file": "a model library, not a deck",
    "lib/lint_rules.py:_LintContext.files": (
        "the staged included files; the deck itself is lex_deck's. What LTspice "
        "reads past an included file's .END is not recorded"
    ),
    "lib/lint_rules.py:_models_from_staged_dependencies.walk": "an included file the walk reaches",
    "lib/montecarlo.py:render_variant_model_card": "one .model card",
    "lib/montecarlo.py:parse_model_params": "one .model card",
    "lib/spice_lex_ops.py:inject_card_before_end": "the one card to insert",
    "lib/spice_validator.py:validate_directive": "one directive",
    "lib/subckt_mismatch.py:_render_clone": "one subcircuit block, copied out of a file",
    "lib/subckt_mismatch.py:render": "a patched subcircuit block the plan rendered",
    "lib/variations.py:_validate_write_sets": "one card the hierarchy resolved",
    "lib/instance_targeting.py:TargetEditor._copy_includes": "one include card, rewritten",
    "lib/instance_targeting.py:TargetEditor.clone_model": "one .model card",
    "lib/instance_targeting.py:source_lineage": (
        "every card of a file, matched to its edited copy by line; which cards "
        "are read is the hierarchy's answer (lex_deck), and only its lines are looked up"
    ),
    "lib/instance_targeting.py:TargetEditor.__init__": (
        "every card of a file by line, for an edit to emit; the cards it edits are "
        "the hierarchy's (lex_deck)"
    ),
    "lib/native_inputs.py:_Sources.occurrence": (
        "every card of a staged file and its original, matched by line; only "
        "lines the hierarchy (lex_deck) resolved are looked up"
    ),
    "lib/pdk_native.py:_original_cards": (
        "every card of an original capture by line; only lines the hierarchy "
        "(lex_deck) resolved are looked up"
    ),
    "lib/deck_prep.py:_read_export": (
        "an LTspice export scrubbed for ngspice's parser: every card loses .backanno "
        "and the section sign, read or not, which changes nothing either simulator reads"
    ),
}


def _plain_lex_calls_in(tree: ast.AST) -> set[str]:
    """The functions in ``tree`` that call ``lex`` in their own body."""
    found: set[str] = set()
    for name, function in _functions(tree):
        pending = list(ast.iter_child_nodes(function))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            pending.extend(ast.iter_child_nodes(node))
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            called = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
            if called == "lex":
                found.add(name)
    return found


def test_a_deck_is_lexed_only_through_its_reading():
    callers = {
        f"{module}:{name}"
        for module, tree in modules().items()
        if module != "lib/spice_lex.py"
        for name in _plain_lex_calls_in(tree)
    }
    assert callers == set(_PLAIN_LEX)


def test_the_plain_lex_rule_catches_what_it_names():
    tree = ast.parse(
        "def deck(text):\n"
        "    return lex(text).cards\n"
        "def outer(text):\n"
        "    def inner():\n"
        "        return spice_lex.lex(text)\n"
        "    return inner\n"
        "def reading(text):\n"
        "    return lex_deck(text).cards\n"
    )
    assert _plain_lex_calls_in(tree) == {"deck", "outer.inner"}


#: Functions outside the lexer that read a card's text themselves: its ``body``
#: or ``raw_lines``, or ``tokenize_body`` or ``key_value_tokens`` on it. A fact
#: about a card is asked of the lexer or a view in ``spice_lex_views``, so two
#: readers cannot answer it differently. An entry gives the reason it reads
#: the text: it shows the text, it is not a deck card, it writes cards back,
#: or it is not yet moved, with the question it answers.
_CARD_TEXT_READERS: dict[str, str] = {
    "code_worker.py:execute": "not a deck card: a Python syntax tree's body",
    "errors.py:UnreadableCardError.__init__": "the refusal keeps the tokenizer error's fields",
    "lib/asc_document.py:Text.format": "not a deck card: a schematic TEXT line",
    "lib/component_value.py:_apply_b_source": (
        "moves to a view in a follow-up: an element card's fields (a B source's value)"
    ),
    "lib/component_value.py:_apply_indep_source": (
        "moves to a view in a follow-up: an element card's fields (a source's value)"
    ),
    "lib/component_value.py:apply_value_to_instance": (
        "a value the caller supplies, read against the card's own parts for an edit"
    ),
    "lib/deck_nodes.py:global_names": (
        "moves to a view in a follow-up: a directive's arguments (the names .global lists)"
    ),
    "lib/deck_prep.py:_read_export": (
        "an LTspice export scrubbed for ngspice's parser: which cards are .backanno"
    ),
    "lib/deck_staging.py:card_sections": (
        "the library-section reader, beside the include scanner it must agree with"
    ),
    "lib/deck_staging.py:scan_include_references": (
        "the include scanner: which file and section an include names"
    ),
    "lib/deck_staging.py:stage_deck.snapshot_file": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/experiment_inputs.py:_card_text": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/experiment_inputs.py:_refused_constructs": (
        "moves to a view in a follow-up: which directive a card is, for recovery's refusals"
    ),
    "lib/experiment_inputs.py:_verify_inputs": (
        "moves to a view in a follow-up: a .func card's name"
    ),
    "lib/guide.py:read": "not a deck card: a guide section",
    "lib/hierarchy.py:_models": "a model's text, kept as the record of its definition",
    "lib/hierarchy.py:_rewritten_for_m": "moves to a view in a follow-up: an element card's letter",
    "lib/hierarchy.py:_rewritten_for_m.callee": (
        "moves to a view in a follow-up: an element card's letter"
    ),
    "lib/hierarchy.py:resolve_hierarchy.expand": (
        "an instance's text, kept as the record of its definition"
    ),
    "lib/hierarchy.py:walk_active.visit": (
        "moves to a view in a follow-up: continuation handling (whether a library section "
        "card is continued)"
    ),
    "lib/hierarchy_expr.py:evaluate": "not a deck card: a Python expression tree's body",
    "lib/instance_targeting.py:TargetEditor._ancestry": (
        "an edit writing cards back: line endings of the cards it moves"
    ),
    "lib/instance_targeting.py:TargetEditor._copy_includes": (
        "an edit writing cards back: a rewritten include card replaces the old"
    ),
    "lib/instance_targeting.py:TargetEditor.clone_model": (
        "an edit writing cards back: line endings of the cards it moves"
    ),
    "lib/instance_targeting.py:apply_edit": "a value the caller supplies, not a deck line",
    "lib/instance_targeting.py:validate_payload": "a value the caller supplies, not a deck line",
    "lib/library_parser.py:parse_library_file": (
        "a model library's definitions kept as written text"
    ),
    "lib/lint_rules.py:_analysis_count_ltspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_include_relative": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_lib_section_ltspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_lib_section_ngspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_meas_function_ltspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_meas_name_ltspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_meas_ngspice_batch": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_meas_trig_degrees": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_model_missing": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_node_control_byte_ltspice": (
        "the bytes of an instance card, for the control-byte rule"
    ),
    "lib/lint_rules.py:_save_meas_coverage": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_save_pattern_unrecorded": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_step_ngspice": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_step_without_values_ltspice": (
        "moves to a view in a follow-up: a directive's arguments (.step)"
    ),
    "lib/lint_rules.py:_suffix_mega_milli": (
        "moves to a view in a follow-up: free-text scan of a body for value suffixes"
    ),
    "lib/lint_rules.py:_temp_as_param": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_unreadable_card": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/lint_rules.py:_value_suffix_findings": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/montecarlo.py:rewrite_instance_model": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/native_inputs.py:_assignments": (
        "the native PDK profile: its removal is decided, so it is left as it is"
    ),
    "lib/netlist_diff.py:_canonical_words": (
        "moves to a view in a follow-up: a card's words for comparison"
    ),
    "lib/netlist_diff.py:_directive_of": (
        "moves to a view in a follow-up: which directive a card is"
    ),
    "lib/netlist_diff.py:parse_directive": (
        "moves to a view in a follow-up: which directive a card is, and an include's target"
    ),
    "lib/netlist_graph.py:_collect_subckts": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/ngspice_driver.py:seed_option": (
        "moves to a view in a follow-up: a directive's arguments (.options seed)"
    ),
    "lib/pdk_native.py:_owned_parameters": (
        "pdk_native: its removal is decided, so it is left as it is"
    ),
    "lib/pdk_native.py:_validate_controls": (
        "pdk_native: its removal is decided, so it is left as it is"
    ),
    "lib/pdk_native.py:validate_sample": (
        "pdk_native: its removal is decided, so it is left as it is"
    ),
    "lib/plot_settings.py:render_plot_settings": "not a deck card: a .plt section",
    "lib/result_observations.py:parse_source_amplitudes": (
        "moves to a view in a follow-up: the names a .param sets (it also reads .params, "
        "which no recording covers)"
    ),
    "lib/runner_base.py:_blank_or_comment": (
        "moves to a view in a follow-up: continuation and comment handling"
    ),
    "lib/runner_base.py:_deck_has_sectioned_lib": (
        "moves to a view in a follow-up: the target and section of a .lib"
    ),
    "lib/runner_base.py:inject_logopinfo": (
        "moves to a view in a follow-up: free-text scan of a body (logopinfo)"
    ),
    "lib/runner_base.py:inject_ngspice_control_write": (
        "moves to a view in a follow-up: continuation and comment handling in a control block"
    ),
    "lib/schematic_ops.py:_apply_component_value": "a value the caller supplies, not a deck line",
    "lib/schematic_ops.py:_replace_param": "a value the caller supplies, not a deck line",
    "lib/schematic_ops.py:_set_source_value": "a value the caller supplies, not a deck line",
    "lib/schematic_ops.py:_validate_component_value": (
        "a value the caller supplies, not a deck line"
    ),
    "lib/schematic_scene.py:_place_symbol": "not a deck card: a symbol drawing",
    "lib/schematic_scene.py:scene_of_sheet": "not a deck card: a schematic TEXT",
    "lib/schematic_scene.py:sheet_view": "not a deck card: a symbol drawing",
    "lib/services.py:extract_netlist_info": (
        "moves to a view in a follow-up: an element card's fields (a key=value left over)"
    ),
    "lib/sheet.py:_placed": "not a deck card: a symbol file",
    "lib/sheet_findings.py:Part.drawn": "not a deck card: a symbol drawing",
    "lib/sheet_findings.py:_duplicate_parameters": "a schematic part's attribute, not a deck line",
    "lib/spice_validator.py:_card_directive": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "lib/spice_validator.py:_unread_connectors": (
        "moves to a view in a follow-up: free-text scan of a body for names"
    ),
    "lib/spice_validator.py:meas_unclosed_paren_refused": (
        "which cards cannot be read, as the tokenizer answers it"
    ),
    "lib/spice_validator.py:validate_directive_card": (
        "moves to a view in a follow-up: a directive's arguments (.tran and .step)"
    ),
    "lib/spice_validator.py:validate_netlist_arity": (
        "moves to a view in a follow-up: an element card's fields (a key=value left over)"
    ),
    "lib/spice_validator.py:validate_netlist_directive_refs": (
        "moves to a view in a follow-up: which directive a card is (.func, and .function, "
        "which no recording covers)"
    ),
    "lib/spice_validator.py:validate_netlist_duplicate_params": (
        "moves to a view in a follow-up: free-text scan of a body"
    ),
    "lib/subckt_mismatch.py:_append_to_code": (
        "an edit writing a card back: where its comment starts"
    ),
    "lib/subckt_mismatch.py:_deck_scale": (
        "moves to a view in a follow-up: a directive's arguments (.options scale, and "
        ".opt, which no recording covers)"
    ),
    "lib/subckt_mismatch.py:_render_clone": (
        "an edit writing cards back: a subcircuit block copied as written"
    ),
    "lib/symbol_file.py:SymbolFile.bbox": "not a deck card: a symbol drawing",
    "lib/variations.py:_closure_model_card": (
        "a model card copied as written, for a per-device variant"
    ),
    "lib/variations.py:_unread_declarations": (
        "the names a card that cannot be read may declare, which no view reads"
    ),
    "tools/inspect_tools.py:_components_netlist_payload": (
        "moves to a view in a follow-up: an element card's fields (a key=value left over)"
    ),
    "tools/verify.py:_meas_angle_findings": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "tools/verify.py:_syntax_findings": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
    "tools/verify.py:_value_suffix_findings": (
        "display: the card's text shown in a finding, message or record, not read for a fact"
    ),
}

_OWNERS = frozenset({"lib/spice_lex.py", "lib/spice_lex_ops.py", "lib/spice_lex_views.py"})
_TEXT_ATTRIBUTES = frozenset({"body", "raw_lines"})
_TEXT_READERS = frozenset({"tokenize_body", "key_value_tokens"})


def _card_text_reads_in(tree: ast.AST) -> set[str]:
    """The functions in ``tree`` that read a card's text in their own body."""
    found: set[str] = set()
    for name, function in _functions(tree):
        pending = list(ast.iter_child_nodes(function))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            pending.extend(ast.iter_child_nodes(node))
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _TEXT_ATTRIBUTES
                and isinstance(node.ctx, ast.Load)
            ):
                found.add(name)
            elif isinstance(node, ast.Call):
                callee = node.func
                called = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
                if called in _TEXT_READERS:
                    found.add(name)
    return found


def test_a_cards_text_is_read_only_by_the_lexer_and_its_views():
    readers = {
        f"{module}:{name}"
        for module, tree in modules().items()
        if module not in _OWNERS
        for name in _card_text_reads_in(tree)
    }
    assert readers == set(_CARD_TEXT_READERS)


def test_the_card_text_rule_catches_what_it_names():
    tree = ast.parse(
        "def head(card):\n"
        "    return card.body.split()[0]\n"
        "def lines(card):\n"
        "    return len(card.raw_lines)\n"
        "def parts(card):\n"
        "    return spice_lex.tokenize_body(card.body)\n"
        "def pairs(text):\n"
        "    return key_value_tokens(text)\n"
        "def write(card):\n"
        "    card.body = ''\n"
        "def view(card):\n"
        "    return ModelCard.from_card(card).params\n"
    )
    assert _card_text_reads_in(tree) == {"head", "lines", "parts", "pairs"}


# --------------------------------------------------------------------------
# Which file and section an include names is read by staging's scanner
# --------------------------------------------------------------------------


def test_an_include_target_is_read_only_by_the_staging_scanner():
    """``deck_staging.scan_include_references`` reads the file and section an
    ``.include`` or ``.lib`` names. A reader that cuts the target out of the
    directive itself takes a library's own ``.lib tt`` section declaration for
    a file to include, and reads every section of a library."""
    cutters = {
        module
        for module, tree in modules().items()
        for name, _ in _functions(tree)
        if name.rsplit(".", 1)[-1] == "_include_target"
    }
    assert cutters <= {"lib/deck_staging.py"}
    head_sets = {
        module
        for module, tree in modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Set)
        and ".include" in [text for element in node.elts for text in _strings(element)]
    }
    assert head_sets == {"lib/spice_lex.py"}


# --------------------------------------------------------------------------
# A probe in a card is read by the lexer's views
# --------------------------------------------------------------------------

#: A pattern for a ``V(...)`` or ``I(...)`` probe: a ``[VI]`` class, or a
#: ``v(``/``i(`` spelled out (not the end of a longer word), before an
#: escaped parenthesis.
_PROBE_PATTERN = re.compile(r"\[[VvIi]{2}\](?:\\s\*)?\\\(|(?:(?<![A-Za-z])|(?<=\\b))[VvIi]\\\(")

#: Modules that compile a probe pattern for text that is not a deck's card.
_PROBE_PATTERN_READERS: dict[str, str] = {
    "lib/trace_names.py": (
        "the results side: a caller's signal name or node pair V(a,b), read against a "
        "raw's traces"
    ),
    "lib/decoded_raw.py": "the results side: a trace name a raw file holds",
}

_RE_FUNCTIONS = frozenset(
    {"compile", "search", "match", "fullmatch", "finditer", "findall", "sub"}
)


def _probe_patterns_in(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _RE_FUNCTIONS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.args
        and any(_PROBE_PATTERN.search(text) for text in _strings(node.args[0]))
        for node in ast.walk(tree)
    )


def test_a_probe_in_a_card_is_read_only_by_the_lexer_views():
    """``spice_lex_views.probe_refs`` reads the probes a card names, and the
    ``.save`` view the probes it saves; a second pattern elsewhere would read
    ``V(a,b)`` one way here and another there."""
    readers = {
        module
        for module, tree in modules().items()
        if module != "lib/spice_lex_views.py" and _probe_patterns_in(tree)
    }
    assert readers == set(_PROBE_PATTERN_READERS)


def test_the_probe_pattern_rule_catches_what_it_names():
    caught = [
        r're.compile(r"\b[VI]\s*\(([^()]*)\)")',
        r're.search(r"(?i)\bv\(\w+\)", text)',
        r're.compile(r"\s*V\(\s*(\S+)\)")',
    ]
    missed = [r're.compile(r"\.save\s+(all)")', r're.compile(r"(\w+)\(")']
    assert [_probe_patterns_in(ast.parse(source)) for source in caught] == [True] * 3
    assert [_probe_patterns_in(ast.parse(source)) for source in missed] == [False] * 2
