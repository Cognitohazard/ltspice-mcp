"""Tests for ``lib/spice_lex.py`` and ``lib/spice_lex_views.py``.

Three layers of coverage:

1. Layer 3 body tokenizer — one fixture per transition-table row.
2. Layer 1 lexer — round-trip byte-identical on representative netlists.
3. Layer 2 typed views — parse + mutate + re-emit.
"""

from __future__ import annotations

import pytest

from ltspice_mcp.lib.spice_lex import (
    SpiceLexError,
    SpiceLexErrorCategory,
    Token,
    TokenKind,
    emit,
    iter_body,
    iter_by_kind,
    lex,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    MeasCard,
    ModelCard,
    ParamCard,
    SubcktCard,
    find_model,
    iter_instances,
    iter_models,
)

# ---------------------------------------------------------------------------
# Layer 3: body tokenizer
# ---------------------------------------------------------------------------


class TestTokenizeBody:
    def test_empty(self) -> None:
        assert tokenize_body("") == []

    def test_whitespace_only(self) -> None:
        assert tokenize_body("   \t  ") == []

    def test_single_bare(self) -> None:
        toks = tokenize_body("R1")
        assert toks == [Token(TokenKind.BARE, "R1", body_offset=0, body_length=2)]

    def test_multiple_bare(self) -> None:
        toks = tokenize_body("R1 n1 n2 1k")
        assert [t.kind for t in toks] == [TokenKind.BARE] * 4
        assert [t.text for t in toks] == ["R1", "n1", "n2", "1k"]

    def test_quoted_string(self) -> None:
        toks = tokenize_body('M1 d g s b "NMOS_lvt" W=10u')
        # M1, d, g, s, b → 5 BARE; then QUOTED, then KEY_VALUE.
        assert toks[5].kind == TokenKind.QUOTED
        assert toks[5].text == '"NMOS_lvt"'

    def test_standalone_equals_in_meas_when_clause(self) -> None:
        # `.MEAS WHEN mag(V(out))=0.7` — `=` is a comparison, not a
        # key-value assignment. The `=` passes through as TokenKind.EQUALS
        # because the preceding atom is PARENED, not BARE/QUOTED.
        toks = tokenize_body("WHEN mag(V(out))=0.7")
        kinds = [t.kind for t in toks]
        assert TokenKind.EQUALS in kinds
        assert TokenKind.PARENED in kinds

    def test_braced_expression(self) -> None:
        toks = tokenize_body("R1 n1 n2 {2*Rd}")
        assert toks[3].kind == TokenKind.BRACED
        assert toks[3].text == "{2*Rd}"

    def test_braced_with_nested_parens(self) -> None:
        toks = tokenize_body("R1 n1 n2 {2*max(kp_n,kp_min)}")
        assert toks[3].kind == TokenKind.BRACED
        assert toks[3].text == "{2*max(kp_n,kp_min)}"

    def test_parened_group(self) -> None:
        toks = tokenize_body(".MODEL FOO NMOS (VTO=0.7 KP=100u)")
        # .MODEL FOO NMOS as 3 BARE then PARENED
        assert toks[0].kind == TokenKind.BARE
        assert toks[3].kind == TokenKind.PARENED
        assert toks[3].text == "(VTO=0.7 KP=100u)"

    def test_key_value_simple(self) -> None:
        toks = tokenize_body("M1 d g s NMOS W=10u")
        kv = toks[-1]
        assert kv.kind == TokenKind.KEY_VALUE
        assert kv.key == "W"
        assert kv.value == "10u"
        assert kv.text == "W=10u"

    def test_key_value_with_braced_value(self) -> None:
        toks = tokenize_body("M1 d g s NMOS W={2*W0}")
        kv = toks[-1]
        assert kv.kind == TokenKind.KEY_VALUE
        assert kv.key == "W"
        assert kv.value == "{2*W0}"

    def test_key_value_with_quoted_value(self) -> None:
        toks = tokenize_body('M1 d g s b "NMOS_lvt" W=10u L="100n"')
        # find the L= KEY_VALUE
        kv = next(t for t in toks if t.kind == TokenKind.KEY_VALUE and t.key == "L")
        assert kv.value == '"100n"'

    def test_key_value_whitespace_around_eq(self) -> None:
        toks = tokenize_body("R1 n1 n2 1k TC = 0.001")
        kv = next(t for t in toks if t.kind == TokenKind.KEY_VALUE)
        assert kv.key == "TC"
        assert kv.value == "0.001"

    def test_key_value_function_call_with_whitespace_around_eq(self) -> None:
        toks = tokenize_body("V = if(V(in)>1, 5, 0)")
        assert [(t.kind, t.text) for t in toks] == [
            (TokenKind.KEY_VALUE, "V=if(V(in)>1, 5, 0)")
        ]
        assert toks[0].key == "V"
        assert toks[0].value == "if(V(in)>1, 5, 0)"

    def test_comment_trail_semicolon(self) -> None:
        toks = tokenize_body("R1 n1 n2 1k ; this is a comment")
        assert toks[-1].kind == TokenKind.COMMENT_TRAIL
        assert toks[-1].text.startswith(";")

    def test_comment_trail_dollar(self) -> None:
        toks = tokenize_body("R1 n1 n2 1k $ HSPICE-style comment")
        assert toks[-1].kind == TokenKind.COMMENT_TRAIL
        assert toks[-1].text.startswith("$")

    def test_semicolon_inside_braces_is_literal(self) -> None:
        # ; inside {} should not start a comment.
        toks = tokenize_body("R1 n1 n2 {a;b}")
        assert toks[3].kind == TokenKind.BRACED
        assert toks[3].text == "{a;b}"

    def test_paren_inside_braces_is_literal(self) -> None:
        toks = tokenize_body("R1 n1 n2 {f(x)}")
        assert toks[3].kind == TokenKind.BRACED
        assert "(" in toks[3].text and ")" in toks[3].text

    def test_unbalanced_brace_raises(self) -> None:
        with pytest.raises(SpiceLexError):
            tokenize_body("R1 n1 n2 {a*b")

    def test_unbalanced_paren_raises(self) -> None:
        with pytest.raises(SpiceLexError):
            tokenize_body(".MODEL FOO NMOS (VTO=0.7")

    def test_unterminated_quote_raises(self) -> None:
        with pytest.raises(SpiceLexError):
            tokenize_body('M1 d g s "unterminated W=10u')

    def test_stray_close_brace_raises(self) -> None:
        with pytest.raises(SpiceLexError):
            tokenize_body("R1 n1 n2 1k}")

    def test_leading_equals_passes_through(self) -> None:
        # ``=foo`` at body start has no preceding key, so the `=` passes
        # through as a standalone EQUALS token (no parse error).
        toks = tokenize_body("=foo")
        assert toks[0].kind == TokenKind.EQUALS
        assert toks[1].kind == TokenKind.BARE

    def test_quoted_with_spaces_inside(self) -> None:
        toks = tokenize_body('"a b c" "d e"')
        assert len(toks) == 2
        assert all(t.kind == TokenKind.QUOTED for t in toks)
        assert toks[0].text == '"a b c"'

    def test_comma_is_token_boundary_outside_groups(self) -> None:
        toks = tokenize_body("a,b,c")
        assert [t.text for t in toks] == ["a", "b", "c"]


class TestInterop:
    def test_cards_from_path_round_trip(self, tmp_path) -> None:
        text = "* test\nR1 a b 1k\n.END\n"
        p = tmp_path / "t.cir"
        p.write_text(text)
        from ltspice_mcp.lib.spice_lex import cards_from_path

        result = cards_from_path(p)
        assert result.warnings == []
        assert emit(result.cards) == text

    def test_write_cards(self, tmp_path) -> None:
        from ltspice_mcp.lib.spice_lex import cards_from_path, write_cards

        text = ".PARAM Vdd=5\nR1 a b 1k\n"
        cards = lex(text).cards
        out = tmp_path / "out.cir"
        write_cards(cards, out)
        # Re-read and confirm round-trip via path-based interop.
        assert cards_from_path(out).cards
        assert out.read_text() == text

    def test_cards_from_path_handles_utf16_bom(self, tmp_path) -> None:
        from ltspice_mcp.lib.spice_lex import cards_from_path

        text = ".MODEL FOO NMOS(VTO=0.7)\n"
        p = tmp_path / "utf16.lib"
        p.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        result = cards_from_path(p)
        assert any(c.kind == "model" for c in result.cards)

    def test_apply_to_editor_with_reset_netlist(self, tmp_path) -> None:
        from ltspice_mcp.lib.spice_lex import apply_to_editor

        class FakeEditor:
            def __init__(self) -> None:
                self.reloaded_from: str | None = None

            def reset_netlist(self, path) -> None:
                self.reloaded_from = str(path)

        cards = lex(".PARAM Vdd=5\n").cards
        ed = FakeEditor()
        apply_to_editor(cards, ed)
        assert ed.reloaded_from is not None

    def test_apply_to_editor_with_netlist_attribute(self) -> None:
        # When the editor has no reset_netlist, writing through the
        # ``netlist`` list-of-strings attribute pushes the new text in.
        from ltspice_mcp.lib.spice_lex import apply_to_editor

        class FakeEditor:
            def __init__(self) -> None:
                self.netlist: list[str] = ["old\n"]

        cards = lex(".PARAM Vdd=5\nR1 a b 1k\n").cards
        ed = FakeEditor()
        apply_to_editor(cards, ed)
        assert ".PARAM Vdd=5\n" in ed.netlist
        assert "R1 a b 1k\n" in ed.netlist
        assert "old\n" not in ed.netlist  # replaced, not appended

    def test_apply_to_editor_raises_when_no_reload_path(self) -> None:
        from ltspice_mcp.lib.spice_lex import apply_to_editor

        class FakeEditor:
            pass  # no reset_netlist, no netlist

        cards = lex(".PARAM Vdd=5\n").cards
        with pytest.raises(TypeError, match="cannot push card list"):
            apply_to_editor(cards, FakeEditor())


class TestOps:
    """Cross-card transformation passes (lib/spice_lex_ops.py)."""

    def test_inject_card_before_end(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end

        text = "R1 a b 1k\n.END\n"
        cards = lex(text).cards
        new = inject_card_before_end(cards, ".MODEL FOO NMOS(VTO=0.7)\n")
        assert new.kind == "model"
        # Verify it landed before the .END.
        kinds = [c.kind for c in cards]
        end_idx = kinds.index("end")
        assert kinds[end_idx - 1] == "model"

    def test_inject_card_when_no_end_appends(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end

        text = "R1 a b 1k\n"
        cards = lex(text).cards
        new = inject_card_before_end(cards, ".MODEL FOO NMOS(VTO=0.7)\n")
        assert new is cards[-1]

    def test_inject_card_no_end_no_trailing_newline(self) -> None:
        # Deck with no .END and no final newline: emit must not glue
        # the appended card onto the predecessor's last line.
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end

        text = "R1 a b 1k"  # no trailing newline
        cards = lex(text).cards
        inject_card_before_end(cards, ".MODEL FOO NMOS(VTO=0.7)\n")
        out = emit(cards)
        # The two cards must be on separate lines, not glued.
        assert "1k.MODEL" not in out
        assert "1k\n.MODEL" in out

    def test_inject_card_before_end_predecessor_missing_newline(self) -> None:
        # Same defect can hit when .END is present but the card before
        # .END lacks a trailing newline (rare but possible after
        # external editors).
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end

        # Build a card list manually: instance card without trailing \n
        # then .END. The lexer wouldn't normally produce this from
        # text, so we patch the raw_lines directly to simulate it.
        cards = lex("R1 a b 1k\n.END\n").cards
        cards[0].raw_lines = ["R1 a b 1k"]  # strip trailing newline
        inject_card_before_end(cards, ".MODEL FOO NMOS(VTO=0.7)\n")
        out = emit(cards)
        assert "1k.MODEL" not in out
        assert "1k\n.MODEL" in out

    def test_rename_subckt_atomic(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import rename_subckt

        text = ".SUBCKT INV in out\nM1 out in 0 0 NMOS\n.ENDS INV\nX1 a b INV\nX2 c d INV\n"
        cards = lex(text).cards
        n = rename_subckt(cards, "INV", "BUF")
        assert n >= 4  # opener + closer + 2 X-callers (and scope updates)
        out = emit(cards)
        assert "INV" not in out
        # Opener, closer name, and both X-callers all use BUF.
        assert out.count("BUF") >= 4

    def test_rename_subckt_updates_body_card_scopes(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import rename_subckt

        text = ".SUBCKT INV in out\nR1 in out 1k\n.ENDS INV\n"
        cards = lex(text).cards
        rename_subckt(cards, "INV", "BUF")
        # The R1 card's scope should now be ("BUF",).
        r1 = next(c for c in cards if c.name == "R1")
        assert r1.scope == ("BUF",)

    def test_rename_subckt_validates_x_callers_first(self) -> None:
        # If any X-caller fails to parse, rename_subckt must raise
        # before any mutation lands.
        from ltspice_mcp.lib.spice_lex import SpiceLexError
        from ltspice_mcp.lib.spice_lex_ops import rename_subckt

        text = ".SUBCKT INV in out\n.ENDS INV\nX1 a b INV\n"
        cards = lex(text).cards
        # Corrupt one X-caller body so InstanceLine.from_card raises.
        cards[2].body = ""  # empty body — instance parse will fail
        cards[2].body_layout = []

        opener_body_before = cards[0].body
        with pytest.raises(SpiceLexError):
            rename_subckt(cards, "INV", "BUF")
        # Opener was not mutated despite being valid.
        assert cards[0].body == opener_body_before

    def test_rename_model_top_level(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import rename_model

        text = ".MODEL NMOS1 NMOS(VTO=0.7)\nM1 d g s b NMOS1 W=10u\nM2 d g s b NMOS1 W=20u\n"
        cards = lex(text).cards
        n = rename_model(cards, "NMOS1", "NMOS_v2")
        assert n == 3  # model + 2 instances
        out = emit(cards)
        assert "NMOS1" not in out
        assert out.count("NMOS_v2") == 3

    def test_rename_model_visible_in_inner_scope(self) -> None:
        from ltspice_mcp.lib.spice_lex_ops import rename_model

        text = (
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            ".SUBCKT INV in out\n"
            "M1 out in 0 0 NMOS1 W=10u\n"
            ".ENDS INV\n"
        )
        cards = lex(text).cards
        n = rename_model(cards, "NMOS1", "NMOS_v2")
        # Outer model + inner instance.
        assert n == 2
        out = emit(cards)
        assert "NMOS1" not in out


class TestFindMatchingEnds:
    """Direct tests for the cross-card .SUBCKT / .ENDS matcher."""

    def test_simple_subckt(self) -> None:
        from ltspice_mcp.lib.spice_lex import find_matching_ends

        cards = lex(".SUBCKT INV in out\nR1 in out 1k\n.ENDS\n").cards
        opener_idx = next(i for i, c in enumerate(cards) if c.kind == "subckt")
        closer_idx = find_matching_ends(cards, opener_idx)
        assert closer_idx is not None
        assert cards[closer_idx].kind == "ends"

    def test_nested_subckt_outer_closer(self) -> None:
        from ltspice_mcp.lib.spice_lex import find_matching_ends

        text = (
            ".SUBCKT OUTER a b\n"
            ".SUBCKT INNER x y\n"
            "R1 x y 1k\n"
            ".ENDS INNER\n"
            "R2 a b 2k\n"
            ".ENDS OUTER\n"
        )
        cards = lex(text).cards
        outer_idx = next(
            i for i, c in enumerate(cards) if c.kind == "subckt" and c.name == "OUTER"
        )
        closer_idx = find_matching_ends(cards, outer_idx)
        assert closer_idx is not None
        assert cards[closer_idx].name == "OUTER"

    def test_nested_subckt_inner_closer(self) -> None:
        from ltspice_mcp.lib.spice_lex import find_matching_ends

        text = ".SUBCKT OUTER a b\n.SUBCKT INNER x y\nR1 x y 1k\n.ENDS INNER\n.ENDS OUTER\n"
        cards = lex(text).cards
        inner_idx = next(
            i for i, c in enumerate(cards) if c.kind == "subckt" and c.name == "INNER"
        )
        closer_idx = find_matching_ends(cards, inner_idx)
        assert closer_idx is not None
        assert cards[closer_idx].name == "INNER"

    def test_missing_closer_returns_none(self) -> None:
        from ltspice_mcp.lib.spice_lex import find_matching_ends

        # EOF before the closing .ENDS — lex still produces the cards
        # but find_matching_ends should report None.
        cards = lex(".SUBCKT INV in out\nR1 in out 1k\n").cards
        opener_idx = next(i for i, c in enumerate(cards) if c.kind == "subckt")
        assert find_matching_ends(cards, opener_idx) is None


class TestSpiceCardTypedAccessors:
    """``model_name`` / ``param_name`` / ``instance_ref`` / ``subckt_name`` / ``meas_label``."""

    def test_model_name(self) -> None:
        cards = lex(".MODEL FOO NMOS(VTO=0.7)\n").cards
        assert cards[0].model_name == "FOO"
        assert cards[0].instance_ref is None
        assert cards[0].param_name is None

    def test_quoted_model_name(self) -> None:
        cards = lex('.MODEL "NMOS_lvt" NMOS(VTO=0.7)\n').cards
        assert cards[0].model_name == "NMOS_lvt"
        assert ModelCard.from_card(cards[0]).name == "NMOS_lvt"
        assert find_model(cards, "NMOS_lvt") is not None

    def test_param_name(self) -> None:
        cards = lex(".PARAM Vdd=5\n").cards
        assert cards[0].param_name == "Vdd"
        assert cards[0].model_name is None

    def test_single_quoted_semicolon_is_not_comment(self) -> None:
        cards = lex(".PARAM x='a;b'\n").cards
        assert cards[0].body == ".PARAM x='a;b'"
        view = ParamCard.from_card(cards[0])
        assert view.name == "x"
        assert view.value == "'a;b'"

    def test_instance_ref(self) -> None:
        cards = lex("R1 a b 1k\n").cards
        assert cards[0].instance_ref == "R1"
        assert cards[0].subckt_name is None

    def test_subckt_name(self) -> None:
        cards = lex(".SUBCKT INV in out\n.ENDS\n").cards
        assert cards[0].subckt_name == "INV"
        assert cards[0].meas_label is None

    def test_meas_label(self) -> None:
        cards = lex(".MEAS TRAN vmax MAX V(out)\n").cards
        assert cards[0].meas_label == "vmax"
        assert cards[0].model_name is None


class TestCanonicalThenSpanEdits:
    """After ``replace_body`` the body_layout collapses to one segment;
    subsequent ``replace_span`` calls must still work on the new layout."""

    def test_set_param_after_canonical_rerender(self) -> None:
        cards = lex(".MODEL FOO NMOS(VTO=0.7 KP=100u)\n").cards
        view = ModelCard.from_card(cards[0])
        # Adding a new param triggers _canonical_rerender — body_layout
        # collapses, _param_tokens is cleared.
        view.set_param("LEVEL", "1")
        # A subsequent edit must still work; views go through canonical
        # again (no cached tokens after clear) but the result must be
        # correct.
        view.set_param("VTO", "0.8")
        out = emit(cards)
        assert "VTO=0.8" in out
        assert "KP=100u" in out
        assert "LEVEL=1" in out

    def test_replace_span_works_on_canonical_form(self) -> None:
        # After replace_body, body_layout has one segment covering the
        # whole new body. Re-deriving the view and editing should hit
        # the in-place path again on the new (single-segment) layout.
        cards = lex(".PARAM Vdd=5\n").cards
        v1 = ParamCard.from_card(cards[0])
        v1.set_value("3.3")
        # Re-derive: cached tokens were cleared after canonical fallback.
        # set_value should still work.
        v2 = ParamCard.from_card(cards[0])
        v2.set_value("12.0")
        out = emit(cards)
        assert "Vdd=12.0" in out


class TestStructuredErrors:
    def test_unbalanced_brace_carries_category_and_position(self) -> None:
        with pytest.raises(SpiceLexError) as ei:
            tokenize_body("R1 a b {2*Rd")
        assert ei.value.category == SpiceLexErrorCategory.UNBALANCED_BRACE
        assert ei.value.position == 7  # position of {
        assert "{" in ei.value.body
        assert ei.value.suggestion  # non-empty suggestion

    def test_unbalanced_paren_category(self) -> None:
        with pytest.raises(SpiceLexError) as ei:
            tokenize_body(".MODEL FOO NMOS (VTO=0.7")
        assert ei.value.category == SpiceLexErrorCategory.UNBALANCED_PAREN

    def test_unterminated_quote_category(self) -> None:
        with pytest.raises(SpiceLexError) as ei:
            tokenize_body('M1 a b "model_name')
        assert ei.value.category == SpiceLexErrorCategory.UNTERMINATED_QUOTE

    def test_stray_close_brace_category(self) -> None:
        with pytest.raises(SpiceLexError) as ei:
            tokenize_body("R1 a b 1k}")
        assert ei.value.category == SpiceLexErrorCategory.UNEXPECTED_CHAR

    def test_str_includes_position_and_suggestion(self) -> None:
        with pytest.raises(SpiceLexError) as ei:
            tokenize_body("R1 a b {open")
        s = str(ei.value)
        assert "position" in s
        assert "suggestion" in s


# ---------------------------------------------------------------------------
# Layer 1: line lexer + round-trip
# ---------------------------------------------------------------------------


SIMPLE_NETLIST = """\
* Simple test netlist
.PARAM Vdd=5
V1 vdd 0 {Vdd}
R1 vdd out 1k
C1 out 0 1n
M1 out gate 0 0 NMOS1 W=10u L=1u
.MODEL NMOS1 NMOS(VTO=0.7 KP=100u)
.TRAN 1m
.END
"""


class TestLexAndEmit:
    def test_round_trip_simple(self) -> None:
        result = lex(SIMPLE_NETLIST)
        assert result.warnings == []
        assert emit(result.cards) == SIMPLE_NETLIST

    def test_round_trip_crlf(self) -> None:
        text = SIMPLE_NETLIST.replace("\n", "\r\n")
        result = lex(text)
        assert emit(result.cards) == text

    def test_card_kinds_classified(self) -> None:
        result = lex(SIMPLE_NETLIST)
        kinds = [c.kind for c in result.cards]
        assert "comment" in kinds  # the * line
        assert "param" in kinds
        assert "instance" in kinds
        assert "model" in kinds
        assert "directive" in kinds  # .TRAN
        assert "end" in kinds

    def test_continuation_merged_into_body(self) -> None:
        text = ".MODEL FOO NMOS\n+ (VTO=0.7\n+ KP=100u)\n"
        result = lex(text)
        assert len(result.cards) == 1
        card = result.cards[0]
        assert card.kind == "model"
        assert "VTO=0.7" in card.body
        assert "KP=100u" in card.body
        # Round-trip preserves the original line layout.
        assert emit(result.cards) == text

    def test_inline_comment_stripped_from_body_but_preserved_in_raw(self) -> None:
        text = "R1 n1 n2 1k ; trailing comment\n"
        result = lex(text)
        card = result.cards[0]
        assert ";" not in card.body
        assert card.body == "R1 n1 n2 1k"
        # Round-trip keeps the comment.
        assert emit(result.cards) == text

    def test_subckt_scope_tracking(self) -> None:
        text = (
            ".SUBCKT INV in out\n"
            "M1 out in vdd vdd PMOS\n"
            "M2 out in 0 0 NMOS\n"
            ".ENDS INV\n"
            "X1 a b INV\n"
        )
        result = lex(text)
        scopes = {c.line_start: c.scope for c in result.cards}
        # .SUBCKT and .ENDS both at top scope.
        # Inner Ms inside ("INV",).
        assert scopes[1] == ()  # .SUBCKT
        assert scopes[2] == ("INV",)  # M1
        assert scopes[3] == ("INV",)  # M2
        assert scopes[4] == ()  # .ENDS
        assert scopes[5] == ()  # X1

    def test_nested_subckt_scope(self) -> None:
        text = (
            ".SUBCKT OUTER a b\n"
            ".SUBCKT INNER x y\n"
            "R1 x y 1k\n"
            ".ENDS INNER\n"
            "X1 a b INNER\n"
            ".ENDS OUTER\n"
        )
        result = lex(text)
        scopes = [c.scope for c in result.cards]
        assert scopes == [
            (),  # .SUBCKT OUTER
            ("OUTER",),  # .SUBCKT INNER (inside OUTER)
            ("OUTER", "INNER"),  # R1
            ("OUTER",),  # .ENDS INNER
            ("OUTER",),  # X1
            (),  # .ENDS OUTER
        ]

    def test_unmatched_ends_warns_not_raises(self) -> None:
        text = ".ENDS\nR1 a b 1k\n"
        result = lex(text)
        assert any(".ENDS with no matching" in w for w in result.warnings)
        # Round-trip still works.
        assert emit(result.cards) == text

    def test_eof_in_open_subckt_warns(self) -> None:
        text = ".SUBCKT FOO a b\nR1 a b 1k\n"
        result = lex(text)
        assert any("unclosed" in w for w in result.warnings)
        assert emit(result.cards) == text

    def test_cards_after_end_marked_trailing(self) -> None:
        text = "R1 a b 1k\n.END\nR2 c d 2k\n"
        result = lex(text)
        last = result.cards[-1]
        assert last.kind == "instance"
        assert last.trailing is True

    def test_meas_classified_separately(self) -> None:
        text = ".MEAS AC fc WHEN mag(V(out))=0.7\n"
        result = lex(text)
        assert result.cards[0].kind == "meas"

    def test_model_inside_subckt_has_correct_scope(self) -> None:
        text = (
            ".SUBCKT WRAP in out\n"
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            "M1 out in 0 0 NMOS1 W=10u L=1u\n"
            ".ENDS WRAP\n"
        )
        result = lex(text)
        model_cards = [c for c in result.cards if c.kind == "model"]
        assert len(model_cards) == 1
        assert model_cards[0].scope == ("WRAP",)


# ---------------------------------------------------------------------------
# Layer 2 typed views
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_parse(self) -> None:
        card = lex(".MODEL NMOS1 NMOS(VTO=0.7 KP=100u LEVEL=1)\n").cards[0]
        view = ModelCard.from_card(card)
        assert view.name == "NMOS1"
        assert view.type == "NMOS"
        assert view.params == {"VTO": "0.7", "KP": "100u", "LEVEL": "1"}
        assert view.level == 1

    def test_parse_braced_value(self) -> None:
        card = lex(".MODEL NMOS1 NMOS(VTO={vto_n} KP=100u)\n").cards[0]
        view = ModelCard.from_card(card)
        assert view.params["VTO"] == "{vto_n}"

    def test_set_param_replaces_existing(self) -> None:
        cards = lex(".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n").cards
        view = ModelCard.from_card(cards[0])
        view.set_param("VTO", 0.8)
        assert view.params["VTO"] == "0.8"
        assert cards[0].dirty
        # Re-emit and re-lex to confirm the new value sticks.
        text = emit(cards)
        assert "VTO=0.8" in text
        assert "KP=100u" in text

    def test_set_param_case_insensitive(self) -> None:
        cards = lex(".MODEL NMOS1 NMOS(VTO=0.7)\n").cards
        view = ModelCard.from_card(cards[0])
        view.set_param("vto", 0.9)
        # Original key case preserved.
        assert "VTO" in view.params
        assert view.params["VTO"] == "0.9"

    def test_set_param_appends_when_absent(self) -> None:
        cards = lex(".MODEL NMOS1 NMOS(VTO=0.7)\n").cards
        view = ModelCard.from_card(cards[0])
        view.set_param("KP", "100u")
        assert view.params["KP"] == "100u"
        text = emit(cards)
        assert "KP=100u" in text

    def test_remove_param(self) -> None:
        cards = lex(".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n").cards
        view = ModelCard.from_card(cards[0])
        view.remove_param("KP")
        assert "KP" not in view.params
        assert "KP" not in emit(cards)

    def test_clean_card_round_trips_byte_for_byte(self) -> None:
        text = ".MODEL NMOS1 NMOS  (VTO=0.7   KP=100u)\n"
        cards = lex(text).cards
        ModelCard.from_card(cards[0])  # parse without mutating
        assert emit(cards) == text  # whitespace preserved


class TestSequentialEditsCacheShift:
    """Repeated in-place edits on the same view must shift cached offsets.

    Without the shift, a second setter splices into stale offsets —
    e.g. editing IS in ``.MODEL D1 D IS=1e-14 N=1`` shifts N's body
    position by the IS length delta, and the next set_param("N") would
    then write into the wrong substring.
    """

    def test_model_two_param_edits_no_corruption(self) -> None:
        text = ".MODEL D1 D IS=1e-14 N=1\n"
        cards = lex(text).cards
        view = ModelCard.from_card(cards[0])
        view.set_param("IS", "2.3456e-14")
        view.set_param("N", "1.5")
        out = emit(cards)
        assert "IS=2.3456e-14" in out
        assert "N=1.5" in out
        # No concatenation of the two params (the corruption signature
        # was a body like "IS=2.3456N=" with N glued onto IS's value).
        assert "IS=2.3456e-14N" not in out
        re_view = ModelCard.from_card(lex(out).cards[0])
        assert re_view.params["IS"] == "2.3456e-14"
        assert re_view.params["N"] == "1.5"

    def test_instance_set_param_then_set_model(self) -> None:
        # First param edit shifts the model-token offset; the next
        # set_model must operate at the shifted position.
        text = "M1 d g s b NMOS1 W=10u L=1u\n"
        cards = lex(text).cards
        view = InstanceLine.from_card(cards[0])
        view.set_param("W", "200u")  # grows W=10u → W=200u (delta +1)
        view.set_model("NMOS_lvt")
        out = emit(cards)
        assert "NMOS_lvt" in out
        assert "W=200u" in out
        assert "L=1u" in out
        re_view = InstanceLine.from_card(lex(out).cards[0])
        assert re_view.model == "NMOS_lvt"
        assert re_view.params["W"] == "200u"
        assert re_view.params["L"] == "1u"

    def test_instance_two_param_edits(self) -> None:
        text = "M1 d g s b NMOS1 W=10u L=1u m=2\n"
        cards = lex(text).cards
        view = InstanceLine.from_card(cards[0])
        view.set_param("W", "100u")
        view.set_param("L", "0.1u")
        view.set_param("m", "4")
        out = emit(cards)
        assert "W=100u" in out
        assert "L=0.1u" in out
        assert "m=4" in out

    def test_param_card_repeated_set_value(self) -> None:
        # ParamCard caches a single token; sanity-check that repeated
        # edits on it work even though there's nothing to shift.
        cards = lex(".PARAM Vdd=5\n").cards
        view = ParamCard.from_card(cards[0])
        view.set_value("3.3")
        view.set_value("12.0")
        out = emit(cards)
        assert "Vdd=12.0" in out


class TestFormatPreservation:
    """Mutations should preserve original layout when possible."""

    def test_resistor_set_value_preserves_continuation_layout(self) -> None:
        text = "R1 n1 n2\n+ 1k\n"
        cards = lex(text).cards
        view = InstanceLine.from_card(cards[0])
        view.set_value("2k")
        out = emit(cards)
        # Multi-line layout preserved; only the `1k` slice changed.
        assert out == "R1 n1 n2\n+ 2k\n"

    def test_param_card_set_value_preserves_position(self) -> None:
        # A .PARAM card on a single line — set_value should rewrite
        # only the value substring.
        text = ".PARAM   Vdd  =  5\n"
        cards = lex(text).cards
        view = ParamCard.from_card(cards[0])
        view.set_value(3.3)
        out = emit(cards)
        # The leading whitespace and `=` spacing stays; only `5` → `3.3`.
        # set_value falls back to canonical when the original token has
        # whitespace around `=` (since text reconstruction loses it).
        # Either way the value sticks.
        assert "3.3" in out
        assert "5" not in out.split("3.3")[1]

    def test_model_set_param_in_place_preserves_continuation(self) -> None:
        text = ".MODEL NMOS1 NMOS\n+ VTO=0.7\n+ KP=100u\n"
        cards = lex(text).cards
        view = ModelCard.from_card(cards[0])
        view.set_param("VTO", 0.8)
        out = emit(cards)
        # Continuation layout preserved; only the VTO value changed.
        assert "VTO=0.8" in out
        assert out.count("\n+ ") == 2  # both continuation lines intact

    def test_dirty_after_mutation(self) -> None:
        cards = lex(".PARAM Vdd=5\n").cards
        assert not cards[0].dirty
        view = ParamCard.from_card(cards[0])
        view.set_value(3.3)
        assert cards[0].dirty

    def test_clean_after_parse_only(self) -> None:
        cards = lex(".PARAM Vdd=5\n").cards
        ParamCard.from_card(cards[0])
        assert not cards[0].dirty


class TestParamCard:
    def test_parse(self) -> None:
        card = lex(".PARAM Vdd=5\n").cards[0]
        view = ParamCard.from_card(card)
        assert view.name == "Vdd"
        assert view.value == "5"

    def test_set_value(self) -> None:
        cards = lex(".PARAM Vdd=5\n").cards
        view = ParamCard.from_card(cards[0])
        view.set_value(3.3)
        text = emit(cards)
        assert "Vdd=3.3" in text


class TestInstanceLine:
    def test_resistor(self) -> None:
        card = lex("R1 n1 n2 1k\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "R1"
        assert view.nodes == ["n1", "n2"]
        # Resistors carry value in .value, not .model.
        assert view.value == "1k"
        assert view.model is None
        assert view.params == {}

    def test_resistor_with_tc(self) -> None:
        card = lex("R1 n1 n2 1k TC=0.001\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.value == "1k"
        assert view.model is None
        assert view.params == {"TC": "0.001"}

    def test_resistor_keyed_primary_value(self) -> None:
        cards = lex("R1 n1 n2 R=1k\n").cards
        view = InstanceLine.from_card(cards[0])
        assert view.nodes == ["n1", "n2"]
        assert view.value == "1k"
        assert view.params == {"R": "1k"}
        view.set_value("2k")
        assert emit(cards) == "R1 n1 n2 R=2k\n"

    def test_b_source_function_value_with_spaced_equals(self) -> None:
        from ltspice_mcp.lib.component_value import apply_value_to_instance

        cards = lex("B1 out 0 V=0\n").cards
        apply_value_to_instance(cards[0], "V = if(V(in)>1, 5, 0)")
        assert emit(cards) == "B1 out 0 V=if(V(in)>1, 5, 0)\n"

    def test_mosfet_basic(self) -> None:
        card = lex("M1 d g s b NMOS1 W=10u L=1u\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "M1"
        assert view.nodes == ["d", "g", "s", "b"]
        assert view.model == "NMOS1"
        assert view.params == {"W": "10u", "L": "1u"}

    def test_mosfet_quoted_model(self) -> None:
        """The motivating adversarial case: `M1 d g s b "NMOS_lvt" W=10u`."""
        card = lex('M1 d g s b "NMOS_lvt" W=10u\n').cards[0]
        view = InstanceLine.from_card(card)
        assert view.nodes == ["d", "g", "s", "b"]
        assert view.model == "NMOS_lvt"  # quotes stripped from logical model

    def test_subckt_call_with_param_override(self) -> None:
        card = lex("X1 a b c MYSUB W=10u\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "X1"
        assert view.nodes == ["a", "b", "c"]
        assert view.model == "MYSUB"
        assert view.params == {"W": "10u"}

    def test_b_source_value_in_params(self) -> None:
        card = lex("B1 out 0 V={V(in)*2}\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "B1"
        assert view.nodes == ["out", "0"]
        assert view.model is None  # B sources have no model
        assert view.params == {"V": "{V(in)*2}"}

    def test_vcvs_positional_gain(self) -> None:
        # E1 out 0 in 0 10 — VCVS with positional gain.
        card = lex("E1 out 0 in 0 10\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "E1"
        assert view.nodes == ["out", "0", "in", "0"]
        assert view.value == "10"
        assert view.model is None
        assert view.params == {}

    def test_vcvs_value_form_is_params_only(self) -> None:
        # E1 out 0 VALUE={V(in)*2} — same prefix, params-only shape.
        card = lex("E1 out 0 VALUE={V(in)*2}\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.nodes == ["out", "0"]
        assert view.value is None
        assert view.model is None
        assert view.params == {"VALUE": "{V(in)*2}"}

    def test_vccs_positional_gain(self) -> None:
        card = lex("G1 out 0 in 0 1m\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.value == "1m"

    def test_cccs_positional_gain(self) -> None:
        # F1 out 0 V1 10 — F has 2 nodes + ref + gain.
        card = lex("F1 out 0 V1 10\n").cards[0]
        view = InstanceLine.from_card(card)
        assert view.ref == "F1"
        # InstanceLine doesn't distinguish nodes from source-refs; it
        # records all positional-but-the-last as nodes.
        assert view.nodes == ["out", "0", "V1"]
        assert view.value == "10"

    def test_vcvs_set_value_replaces_gain(self) -> None:
        cards = lex("E1 out 0 in 0 10\n").cards
        view = InstanceLine.from_card(cards[0])
        view.set_value("20")
        out = emit(cards)
        assert "E1 out 0 in 0 20" in out
        assert "E1 out 0 in 0 10 20" not in out  # not appended after old gain

    def test_set_model(self) -> None:
        cards = lex("M1 d g s b NMOS1 W=10u L=1u\n").cards
        view = InstanceLine.from_card(cards[0])
        view.set_model("NMOS_lvt")
        text = emit(cards)
        assert "NMOS_lvt" in text
        assert "NMOS1" not in text
        # Params survive.
        assert "W=10u" in text
        assert "L=1u" in text

    def test_set_param_replaces_existing(self) -> None:
        cards = lex("M1 d g s b NMOS1 W=10u L=1u\n").cards
        view = InstanceLine.from_card(cards[0])
        view.set_param("W", "20u")
        text = emit(cards)
        assert "W=20u" in text
        assert "W=10u" not in text


class TestSubcktCard:
    def test_basic(self) -> None:
        card = lex(".SUBCKT INV in out vdd vss\n").cards[0]
        view = SubcktCard.from_card(card)
        assert view.name == "INV"
        assert view.ports == ["in", "out", "vdd", "vss"]
        assert view.param_defaults == {}

    def test_with_params_marker(self) -> None:
        card = lex(".SUBCKT INV in out PARAMS: W=10u L=1u\n").cards[0]
        view = SubcktCard.from_card(card)
        assert view.ports == ["in", "out"]
        assert view.param_defaults == {"W": "10u", "L": "1u"}

    def test_implicit_params_no_marker(self) -> None:
        card = lex(".SUBCKT INV in out W=10u L=1u\n").cards[0]
        view = SubcktCard.from_card(card)
        assert view.ports == ["in", "out"]
        assert view.param_defaults == {"W": "10u", "L": "1u"}

    def test_set_name_local_in_place(self) -> None:
        cards = lex(".SUBCKT INV in out vdd vss\n").cards
        view = SubcktCard.from_card(cards[0])
        view.set_name_local("BUF")
        out = emit(cards)
        assert "BUF" in out
        assert "INV" not in out
        assert "in out vdd vss" in out  # ports preserved verbatim

    def test_set_param_default(self) -> None:
        cards = lex(".SUBCKT INV in out PARAMS: W=10u\n").cards
        view = SubcktCard.from_card(cards[0])
        view.set_param_default("L", "1u")
        out = emit(cards)
        assert "L=1u" in out
        assert "W=10u" in out


class TestMeasCard:
    def test_parse_with_analysis(self) -> None:
        card = lex(".MEAS AC fc WHEN mag(V(out))=0.7\n").cards[0]
        view = MeasCard.from_card(card)
        assert view.analysis == "ac"
        assert view.name == "fc"
        # mag and V are both function calls.
        names = [fc.name.lower() for fc in view.function_calls]
        assert "mag" in names
        assert "v" in names

    def test_parse_without_analysis_kind(self) -> None:
        card = lex(".MEAS gain MAX V(out)\n").cards[0]
        view = MeasCard.from_card(card)
        assert view.analysis is None
        assert view.name == "gain"

    def test_signal_refs_extracted(self) -> None:
        card = lex(".MEAS TRAN vmax MAX V(out)\n").cards[0]
        view = MeasCard.from_card(card)
        assert "out" in view.signal_refs

    def test_function_call_with_vdb_detected(self) -> None:
        # The motivating Phase 5 case: detect vdb() inside .MEAS body.
        card = lex(".MEAS AC fc WHEN vdb(out)=-3\n").cards[0]
        view = MeasCard.from_card(card)
        assert any(fc.name.lower() == "vdb" for fc in view.function_calls)

    def test_set_label_rerenders(self) -> None:
        cards = lex(".MEAS TRAN vmax MAX V(out)\n").cards
        view = MeasCard.from_card(cards[0])
        view.set_label("peak_v")
        out = emit(cards)
        assert "peak_v" in out
        assert "vmax" not in out

    def test_set_analysis_rerenders(self) -> None:
        cards = lex(".MEAS gain MAX V(out)\n").cards
        view = MeasCard.from_card(cards[0])
        view.set_analysis("tran")
        out = emit(cards)
        assert "TRAN" in out


# ---------------------------------------------------------------------------
# Convenience iterators
# ---------------------------------------------------------------------------


class TestIterators:
    def test_iter_models(self) -> None:
        text = ".MODEL NMOS1 NMOS(VTO=0.7)\n.MODEL PMOS1 PMOS(VTO=-0.7)\n"
        cards = lex(text).cards
        models = list(iter_models(cards))
        assert [m.name for m in models] == ["NMOS1", "PMOS1"]

    def test_iter_instances_by_prefix(self) -> None:
        text = "R1 a b 1k\nM1 d g s b NMOS1\nM2 d g s b NMOS1\nC1 a 0 1n\n"
        cards = lex(text).cards
        mosfets = list(iter_instances(cards, prefix="M"))
        assert [m.ref for m in mosfets] == ["M1", "M2"]

    def test_find_model_by_name_top_level(self) -> None:
        text = ".MODEL NMOS1 NMOS(VTO=0.7)\n"
        cards = lex(text).cards
        assert find_model(cards, "NMOS1") is not None
        assert find_model(cards, "nmos1") is not None  # case-insensitive
        assert find_model(cards, "PMOS1") is None

    def test_find_model_walks_outward(self) -> None:
        # Inner scope sees outer's models.
        text = ".MODEL NMOS1 NMOS(VTO=0.7)\n.SUBCKT INNER x y\nM1 x y 0 0 NMOS1\n.ENDS INNER\n"
        cards = lex(text).cards
        # From within INNER, NMOS1 must resolve.
        m = find_model(cards, "NMOS1", scope=("INNER",))
        assert m is not None
        assert m.name == "NMOS1"

    def test_find_model_inner_shadows_outer(self) -> None:
        text = (
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            ".SUBCKT INNER x y\n"
            ".MODEL NMOS1 NMOS(VTO=0.5)\n"
            "M1 x y 0 0 NMOS1\n"
            ".ENDS INNER\n"
        )
        cards = lex(text).cards
        m = find_model(cards, "NMOS1", scope=("INNER",))
        assert m is not None
        assert m.params["VTO"] == "0.5"  # inner one wins

    def test_iter_by_kind_with_scope_filter(self) -> None:
        text = (
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            ".SUBCKT INNER x y\n"
            ".MODEL NMOS2 NMOS(VTO=0.5)\n"
            ".ENDS INNER\n"
        )
        cards = lex(text).cards
        top = list(iter_by_kind(cards, "model", scope=()))
        assert [c.name for c in top] == ["NMOS1"]
        inner = list(iter_by_kind(cards, "model", scope=("INNER",)))
        assert [c.name for c in inner] == ["NMOS2"]

    def test_iter_body_recursive(self) -> None:
        text = (
            ".SUBCKT OUTER a b\n"
            ".SUBCKT INNER x y\n"
            "R1 x y 1k\n"
            ".ENDS INNER\n"
            "R2 a b 2k\n"
            ".ENDS OUTER\n"
        )
        cards = lex(text).cards
        body = list(iter_body(cards, ("OUTER",)))
        # Includes everything under OUTER: nested subckt + inner R1 + R2.
        kinds = [c.kind for c in body]
        assert "subckt" in kinds  # the nested INNER opener
        assert kinds.count("instance") == 2  # R1 + R2

    def test_iter_body_non_recursive(self) -> None:
        text = (
            ".SUBCKT OUTER a b\n"
            ".SUBCKT INNER x y\n"
            "R1 x y 1k\n"
            ".ENDS INNER\n"
            "R2 a b 2k\n"
            ".ENDS OUTER\n"
        )
        cards = lex(text).cards
        body = list(iter_body(cards, ("OUTER",), recursive=False))
        # Only direct children of OUTER, not INNER's body.
        kinds = [c.kind for c in body]
        assert "instance" in kinds  # R2
        # R1 is at scope ("OUTER","INNER") — excluded.
        refs = [c.name for c in body if c.kind == "instance"]
        assert refs == ["R2"]
