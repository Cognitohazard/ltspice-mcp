"""edit_schematic's set_plot_panes: the waveform panes LTspice opens for a sheet.

The op writes the plot settings file beside the sheet (``<sheet>.plt``) in the
same transaction as the sheet. Everything here goes through the tool; the files
LTspice wrote, used as the file already beside a sheet, are the recorded ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.plot_settings import read_plot_settings
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests._asc_ops import build_sheet, sha_of
from tests._ltspice_recorded import recorded

_TWO_PANES = [{"traces": ["V(out)"]}, {"traces": ["V(in)", "I(R1)"]}]


async def _edit(state: SessionState, target: str, ops: list[dict], **kw: Any) -> dict:
    path = Path(state.working_dir) / target
    if path.exists() and "expected_sha256" not in kw:
        kw["expected_sha256"] = sha_of(path)
    result = await handle_edit_schematic(
        EditSchematicInput.model_validate({"target": target, "ops": ops, **kw}), state
    )
    data = result.structured_content
    assert data is not None
    return dict(data)


def _panes(analysis: str, panes: list[dict]) -> dict:
    return {"op": "set_plot_panes", "analysis": analysis, "panes": panes}


@pytest.fixture
async def sheet(asc_state: SessionState) -> Path:
    await build_sheet(asc_state, "rc", [{"op": "add_net_label", "net": "out", "x": 96, "y": 96}])
    return Path(asc_state.working_dir) / "rc.asc"


async def test_the_panes_are_written_beside_the_sheet_bottom_first(asc_state, sheet: Path):
    data = await _edit(asc_state, "rc.asc", [_panes("tran", _TWO_PANES)])

    assert data["outcome"] == "complete"
    plot = sheet.with_suffix(".plt")
    (result,) = data["results"]
    assert result["index"] == 0 and result["op"] == "set_plot_panes"
    assert result["replaced_panes"] == []
    # The path the sandbox resolved, which may be spelt differently (a short
    # name on Windows, a symlinked temp root elsewhere), names this file.
    assert Path(result["plot_settings"]).samefile(plot)  # noqa: ASYNC240
    raw = plot.read_bytes()
    assert b"\r" not in raw.decode("utf-16-le").encode("utf-8")
    assert raw.decode("utf-16-le") == (
        "[Transient Analysis]\n{\n   Npanes: 2\n"
        '   {\n      traces: 2 {0,0,"V(in)"} {0,0,"I(R1)"}\n      Log: 0 0 0\n   },\n'
        '   {\n      traces: 1 {0,0,"V(out)"}\n      Log: 0 0 0\n   }\n}\n'
    )
    # The sheet is what the reply's digest says it is, plot op or not.
    assert data["sha256"] == sha_of(sheet)


async def test_an_ac_pane_gets_ltspices_own_scales_unless_it_names_them(asc_state, sheet: Path):
    await _edit(
        asc_state,
        "rc.asc",
        [_panes("ac", [{"traces": ["V(out)"]}, {"traces": ["V(in)"], "y_scale": "linear"}])],
    )
    section = read_plot_settings(sheet.with_suffix(".plt").read_bytes()).section("AC Analysis")
    assert section is not None
    assert [pane.scales for pane in section.panes] == [(1, 2, 0), (1, 0, 0)]


async def test_a_file_ltspice_26_wrote_keeps_its_other_sections(asc_state, sheet: Path):
    """LTspice 26 writes the file in UTF-8 and with a micro sign in a unit; the
    section the op does not name keeps its text, in the encoding written here."""
    existing = recorded("ltspice26", "plot/math.plt").read_bytes()
    assert "µ".encode() in existing
    plot = sheet.with_suffix(".plt")
    plot.write_bytes(existing)

    data = await _edit(asc_state, "rc.asc", [_panes("ac", [{"traces": ["V(out)"]}])])

    assert data["results"][0]["replaced_panes"] == []
    text = plot.read_bytes().decode("utf-16-le")
    assert text.startswith(existing.decode("utf-8"))
    assert text.count("[AC Analysis]") == 1


async def test_the_replaced_panes_put_back_undo_the_op(asc_state, sheet: Path):
    plot = sheet.with_suffix(".plt")
    plot.write_bytes(recorded("ltspice26", "plot/math.plt").read_bytes())
    before = read_plot_settings(plot.read_bytes()).section("Transient Analysis")

    first = await _edit(asc_state, "rc.asc", [_panes("tran", [{"traces": ["V(out)"]}])])
    replaced = first["results"][0]["replaced_panes"]
    assert replaced == [
        {"traces": ["V(in)-V(out)", "V(out)*I(R1)"], "x_scale": "linear", "y_scale": "linear"}
    ]

    second = await _edit(asc_state, "rc.asc", [_panes("tran", replaced)])
    assert second["results"][0]["replaced_panes"] == [
        {"traces": ["V(out)"], "x_scale": "linear", "y_scale": "linear"}
    ]
    after = read_plot_settings(plot.read_bytes()).section("Transient Analysis")
    assert before is not None and after is not None
    assert after.panes == before.panes


async def test_no_panes_removes_the_analysis_and_then_the_file(asc_state, sheet: Path):
    plot = sheet.with_suffix(".plt")
    await _edit(
        asc_state,
        "rc.asc",
        [_panes("tran", _TWO_PANES), _panes("ac", [{"traces": ["V(out)"]}])],
    )
    await _edit(asc_state, "rc.asc", [_panes("tran", [])])
    assert [s.name for s in read_plot_settings(plot.read_bytes()).sections] == ["AC Analysis"]

    await _edit(asc_state, "rc.asc", [_panes("ac", [])])
    assert not plot.exists()


async def test_a_dry_run_reports_and_writes_nothing(asc_state, sheet: Path):
    data = await _edit(asc_state, "rc.asc", [_panes("tran", _TWO_PANES)], dry_run=True)
    assert data["results"][0]["replaced_panes"] == []
    assert not sheet.with_suffix(".plt").exists()


async def test_one_batch_edits_the_sheet_and_its_panes(asc_state, sheet: Path):
    data = await _edit(
        asc_state,
        "rc.asc",
        [
            {"op": "add_directive", "instruction": ".tran 1m"},
            _panes("tran", [{"traces": ["V(out)"]}]),
        ],
    )
    assert data["outcome"] == "complete"
    assert ".tran 1m" in sheet.read_text(encoding="utf-8")  # noqa: ASYNC240
    assert sheet.with_suffix(".plt").is_file()


@pytest.mark.parametrize("trace", ['V("out")', "V(out)\nV(in)", " "])
async def test_a_trace_the_file_cannot_carry_aborts_the_batch(asc_state, sheet: Path, trace: str):
    sha = sha_of(sheet)
    data = await _edit(
        asc_state,
        "rc.asc",
        [
            {"op": "add_directive", "instruction": ".tran 1m"},
            _panes("tran", [{"traces": ["V(in)", trace]}]),
        ],
    )
    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    assert data["failures"][0]["index"] == 1
    assert sha_of(sheet) == sha
    assert not sheet.with_suffix(".plt").exists()


async def test_a_pane_with_no_trace_is_refused_by_the_model(asc_state, sheet: Path):
    with pytest.raises(ValueError, match="traces"):
        EditSchematicInput.model_validate(
            {"target": "rc.asc", "ops": [_panes("tran", [{"traces": []}])]}
        )


async def test_a_file_in_two_encodings_is_refused_and_left_alone(asc_state, sheet: Path):
    """LTspice XVII saving over a file in LTspice 26's encoding leaves its own
    UTF-16 section followed by the old file's bytes; the op will not guess at it."""
    mixed = recorded("ltspice17", "plot/read_utf8.plt").read_bytes()
    plot = sheet.with_suffix(".plt")
    plot.write_bytes(mixed)
    sha = sha_of(sheet)

    data = await _edit(asc_state, "rc.asc", [_panes("tran", [{"traces": ["V(out)"]}])])

    assert data["outcome"] == "failed"
    assert "outside its sections" in data["failures"][0]["error"]
    assert plot.read_bytes() == mixed
    assert sha_of(sheet) == sha


async def test_a_sheet_rename_that_fails_puts_the_old_panes_back(
    asc_state, sheet: Path, monkeypatch: pytest.MonkeyPatch
):
    plot = sheet.with_suffix(".plt")
    existing = recorded("ltspice26", "plot/math.plt").read_bytes()
    plot.write_bytes(existing)
    sha = sha_of(sheet)

    def refuse(_tmp: Path, _target: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(se, "_commit_rename", refuse)
    data = await _edit(asc_state, "rc.asc", [_panes("tran", [{"traces": ["V(out)"]}])])

    assert data["commit_state"] == "not_committed"
    assert data["error"]["stage"] == "rename"
    assert plot.read_bytes() == existing
    assert sha_of(sheet) == sha
    assert not list(sheet.parent.glob("*.staging-*"))


async def test_a_new_file_is_removed_again_when_the_sheet_rename_fails(
    asc_state, sheet: Path, monkeypatch: pytest.MonkeyPatch
):
    def refuse(_tmp: Path, _target: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(se, "_commit_rename", refuse)
    data = await _edit(asc_state, "rc.asc", [_panes("tran", [{"traces": ["V(out)"]}])])

    assert data["commit_state"] == "not_committed"
    assert not sheet.with_suffix(".plt").exists()


async def test_a_plot_settings_write_that_fails_leaves_both_files(
    asc_state, sheet: Path, monkeypatch: pytest.MonkeyPatch
):
    plot = sheet.with_suffix(".plt")
    existing = recorded("ltspice17", "plot/math.plt").read_bytes()
    plot.write_bytes(existing)
    sha = sha_of(sheet)

    def refuse(_path: Path, _data: bytes | None, _build_id: str) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(se, "_write_beside", refuse)
    data = await _edit(asc_state, "rc.asc", [_panes("tran", [{"traces": ["V(out)"]}])])

    assert data["commit_state"] == "not_committed"
    assert data["error"]["stage"] == "stage_plot_settings"
    assert "injected write failure" in data["error"]["message"]
    assert plot.read_bytes() == existing
    assert sha_of(sheet) == sha
    assert not list(sheet.parent.glob("*.staging-*"))
