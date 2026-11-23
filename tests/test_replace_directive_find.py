"""``replace_directive`` with ``find``: one change inside a directive or comment.

``find`` and ``text`` read as the text reads, a line break as one break however
it is typed, and ``find`` must occur exactly once in the texts searched. The
op is reached two ways: through ``apply_op_inplace`` on an editor, for the
rules of one op, and through ``edit_schematic``, for what a batch, a dry run
and the commit write.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.schematic_ops import apply_op_inplace, blank_sheet
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import (
    ConsolidatedOp,
    EditSchematicInput,
    handle_edit_schematic,
)
from tests._schematic_fixtures import open_editor

_OP = TypeAdapter(ConsolidatedOp)


def _write(path: Path, *lines: str, newline: str = "\n", encoding: str = "utf-8") -> Path:
    """A blank sheet plus ``lines``, each ended with ``newline``, in ``encoding``."""
    body = (blank_sheet() + "".join(f"{line}\n" for line in lines)).replace("\n", newline)
    path.write_bytes(body.encode(encoding))
    return path


def _replace(sheet: Path, **fields: Any) -> dict[str, Any]:
    """Apply one replace_directive to ``sheet`` and save it; return its ``text`` fact."""
    editor = open_editor(sheet)
    result = apply_op_inplace(editor, _OP.validate_python({"op": "replace_directive", **fields}), sheet)
    editor.save_netlist(sheet)
    return cast("dict[str, Any]", result["text"])


def _refused(sheet: Path, **fields: Any) -> str:
    """The refusal of one replace_directive; the editor's texts are as they were."""
    editor = open_editor(sheet)
    before = [(text.text, text.coord.X, text.coord.Y) for text in editor.directives]
    with pytest.raises(NetlistError) as refused:
        apply_op_inplace(
            editor, _OP.validate_python({"op": "replace_directive", **fields}), sheet
        )
    assert [(text.text, text.coord.X, text.coord.Y) for text in editor.directives] == before
    return str(refused.value)


async def _edit(state: SessionState, name: str, ops: list[dict[str, Any]], **kw: Any) -> dict:
    """One edit_schematic call on ``name`` in the working directory."""
    target = Path(state.working_dir) / name
    result = await handle_edit_schematic(
        EditSchematicInput.model_validate(
            {
                "path": name,
                "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "ops": ops,
                **kw,
            }
        ),
        state,
    )
    assert result.structured_content is not None
    return dict(result.structured_content)


def _find(find: str, text: str, **kw: Any) -> dict[str, Any]:
    return {"op": "replace_directive", "find": find, "text": text, **kw}


_BLOCK = "TEXT 16 16 Left 2 !.param a=1\\n.param b=2\\n.param c=3"


class TestWhichTextIsChanged:
    def test_one_line_of_a_block_changes_and_the_reply_holds_that_line(self, work_dir: Path):
        sheet = _write(work_dir / "block.asc", _BLOCK, "TEXT 16 96 Left 2 !.tran 1m")
        assert _replace(sheet, find="b=2", text="b=5") == {
            "kind": "directive",
            "x": 16,
            "y": 16,
            "line": 2,
            "was": ".param b=2",
            "now": ".param b=5",
        }
        lines = sheet.read_text(encoding="utf-8").splitlines()
        assert "TEXT 16 16 Left 2 !.param a=1\\n.param b=5\\n.param c=3" in lines
        assert "TEXT 16 96 Left 2 !.tran 1m" in lines

    def test_find_matches_across_a_line_break_however_it_is_typed(self, work_dir: Path):
        sheet = _write(work_dir / "across.asc", _BLOCK)
        result = _replace(sheet, find="a=1\r\n.param b", text="a=4\n.param b")
        assert (result["line"], result["was"], result["now"]) == (
            1,
            ".param a=1\n.param b=2",
            ".param a=4\n.param b=2",
        )

    def test_find_twice_in_one_text_is_refused_with_each_place(self, work_dir: Path):
        sheet = _write(work_dir / "twice.asc", "TEXT 16 16 Left 2 !.param a=1\\n.param b=1")
        message = _refused(sheet, find="=1", text="=2")
        assert "2 times" in message
        assert "1:9" in message and "2:9" in message
        assert "make find longer, taking in neighbouring words" in message

    def test_find_in_two_texts_is_refused_with_both_and_x_y_picks_one(self, work_dir: Path):
        sheet = _write(
            work_dir / "two.asc",
            "TEXT 16 16 Left 2 !.param a=1",
            "TEXT 16 96 Left 2 ;a=1 is the default",
        )
        message = _refused(sheet, find="a=1", text="a=2")
        assert "'.param a=1' at (16, 16); 'a=1 is the default' at (16, 96)" in message
        assert "give instruction, or x and y" in message
        assert _replace(sheet, find="a=1", text="a=2", x=16, y=96)["now"] == "a=2 is the default"
        lines = sheet.read_text(encoding="utf-8").splitlines()
        assert "TEXT 16 16 Left 2 !.param a=1" in lines
        assert "TEXT 16 96 Left 2 ;a=2 is the default" in lines

    def test_instruction_names_the_one_text_searched(self, work_dir: Path):
        sheet = _write(
            work_dir / "named.asc",
            "TEXT 16 16 Left 2 !.param a=1",
            "TEXT 16 96 Left 2 ;a=1 is the default",
        )
        result = _replace(sheet, instruction="regex:^;?a=1", find="a=1", text="a=3")
        assert (result["kind"], result["y"]) == ("comment", 96)

    def test_find_nowhere_in_a_named_text_gives_that_text(self, work_dir: Path):
        sheet = _write(work_dir / "absent.asc", "TEXT 16 16 Left 2 !.tran 1m")
        message = _refused(sheet, instruction=".tran 1m", find="2m", text="3m")
        assert "'2m' is not in" in message
        assert "'.tran 1m'" in message

    def test_find_in_the_record_spelling_is_refused_with_the_reading_form(self, work_dir: Path):
        sheet = _write(work_dir / "spelled.asc", _BLOCK)
        message = _refused(sheet, find="a=1\\n.param b", text="x")
        assert "as the record spells it" in message
        assert repr("a=1\n.param b") in message


class TestTheNewText:
    def test_a_directive_the_rules_refuse_is_refused(self, work_dir: Path):
        sheet = _write(work_dir / "step.asc", "TEXT 16 16 Left 2 !.step param R list 1k 2k")
        assert _refused(sheet, find=" list 1k 2k", text="").startswith("Refusing directive")

    def test_a_comment_is_not_checked(self, work_dir: Path):
        sheet = _write(work_dir / "note.asc", "TEXT 16 16 Left 2 ;.step param R list 1k 2k")
        assert _replace(sheet, find=" list 1k 2k", text="")["now"] == ".step param R"

    def test_a_card_is_changed_and_checked_as_a_card(self, work_dir: Path):
        sheet = _write(work_dir / "card.asc", "TEXT 16 16 Left 2 !R9 a b 1k")
        assert _replace(sheet, find="1k", text="2k")["now"] == "R9 a b 2k"
        assert "TEXT 16 16 Left 2 !R9 a b 2k" in sheet.read_text(encoding="utf-8")

    def test_a_change_that_empties_the_text_names_remove_directive(self, work_dir: Path):
        sheet = _write(work_dir / "empty.asc", "TEXT 16 16 Left 2 !.tran 1m")
        assert "remove_directive" in _refused(sheet, find=".tran 1m", text="")

    @pytest.mark.parametrize("text", ["", "  "])
    def test_the_whole_text_form_refuses_an_empty_text(self, work_dir: Path, text: str):
        sheet = _write(work_dir / "whole.asc", "TEXT 16 16 Left 2 !.tran 1m")
        assert "remove_directive" in _refused(sheet, instruction=".tran 1m", text=text)

    def test_a_call_with_neither_instruction_nor_find_is_refused_at_validation(self):
        with pytest.raises(ValidationError, match="find"):
            _OP.validate_python({"op": "replace_directive", "text": ".tran 2m"})


class TestWhatTheBatchWrites:
    async def test_the_reverse_call_restores_the_record_byte_for_byte(
        self, asc_state: SessionState, work_dir: Path
    ):
        sheet = _write(work_dir / "undo.asc", _BLOCK, "TEXT 16 96 Left 2 !.tran 1m")
        before = sheet.read_bytes()
        done = await _edit(asc_state, "undo.asc", [_find("b=2", "b=5")])
        assert done["commit_state"] == "committed", done
        fact = done["results"][0]["text"]
        undone = await _edit(
            asc_state,
            "undo.asc",
            [_find(fact["now"], fact["was"], x=fact["x"], y=fact["y"])],
        )
        assert undone["commit_state"] == "committed", undone
        assert sheet.read_bytes() == before

    async def test_a_crlf_sheet_is_written_as_a_whole_text_edit_writes_it(
        self, asc_state: SessionState, work_dir: Path
    ):
        # The line endings of a saved sheet are the save path's, which this op
        # does not change: the find edit writes what the whole-text edit writes.
        lines = (_BLOCK, "TEXT 16 96 Left 2 ;note")
        by_find = _write(work_dir / "find.asc", *lines, newline="\r\n")
        by_text = _write(work_dir / "text.asc", *lines, newline="\r\n")
        before = by_find.read_bytes().decode().splitlines()
        done = await _edit(asc_state, "find.asc", [_find("c=3", "c=7")])
        assert done["commit_state"] == "committed", done
        whole = {
            "op": "replace_directive",
            "instruction": "regex:^\\.param a=1",
            "text": ".param a=1\n.param b=2\n.param c=7",
        }
        done = await _edit(asc_state, "text.asc", [whole])
        assert done["commit_state"] == "committed", done
        assert by_find.read_bytes() == by_text.read_bytes()
        after = by_find.read_bytes().decode().splitlines()
        changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
        assert changed == [(_BLOCK, _BLOCK.replace("c=3", "c=7"))]

    async def test_a_chained_batch_applies_in_order(self, asc_state: SessionState, work_dir: Path):
        sheet = _write(work_dir / "chain.asc", "TEXT 16 16 Left 2 !.param a=1")
        done = await _edit(
            asc_state, "chain.asc", [_find("a=1", "a=2"), _find("a=2", "a=3")]
        )
        assert done["commit_state"] == "committed", done
        assert [r["text"]["now"] for r in done["results"]] == [".param a=2", ".param a=3"]
        assert "TEXT 16 16 Left 2 !.param a=3" in sheet.read_text(encoding="utf-8")

    async def test_an_op_looking_for_words_an_earlier_op_took_is_refused(
        self, asc_state: SessionState, work_dir: Path
    ):
        sheet = _write(work_dir / "gone.asc", "TEXT 16 16 Left 2 !.param a=1")
        before = sheet.read_bytes()
        done = await _edit(
            asc_state, "gone.asc", [_find("a=1", "a=2"), _find("a=1", "a=9")]
        )
        assert done["commit_state"] == "not_committed"
        assert "'a=1'" in done["failures"][0]["error"]
        assert sheet.read_bytes() == before

    async def test_a_dry_run_reports_as_a_commit_and_its_later_ops_see_the_change(
        self, asc_state: SessionState, work_dir: Path
    ):
        sheet = _write(work_dir / "dry.asc", _BLOCK)
        before = sheet.read_bytes()
        ops = [_find("b=2", "b=5"), _find("b=5", "b=6")]
        dry = await _edit(asc_state, "dry.asc", ops, dry_run=True)
        assert sheet.read_bytes() == before
        assert "failures" not in dry, dry
        done = await _edit(asc_state, "dry.asc", ops)
        assert done["commit_state"] == "committed", done
        assert dry["results"] == done["results"]
        assert done["results"][1]["text"]["now"] == ".param b=6"

    async def test_a_cp1252_sheet_is_written_in_cp1252(
        self, asc_state: SessionState, work_dir: Path
    ):
        sheet = _write(
            work_dir / "micro.asc", "TEXT 16 16 Left 2 !.param c=1µ", encoding="cp1252"
        )
        done = await _edit(asc_state, "micro.asc", [_find("1µ", "2µ")])
        assert done["commit_state"] == "committed", done
        assert b"!.param c=2\xb5" in sheet.read_bytes()

    async def test_a_character_cp1252_lacks_is_refused_and_nothing_is_written(
        self, asc_state: SessionState, work_dir: Path
    ):
        sheet = _write(work_dir / "omega.asc", "TEXT 16 16 Left 2 ;c=1µ", encoding="cp1252")
        before = sheet.read_bytes()
        done = await _edit(asc_state, "omega.asc", [_find("c=1µ", "c=1µ R=2Ω")])
        assert done["commit_state"] == "not_committed", done
        # The save path names the character by its code point.
        assert "\\u03a9" in done["error"]["message"], done
        assert sheet.read_bytes() == before
