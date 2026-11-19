"""A cut reply whose complete answer cannot be kept is still sent.

Writing the stored answer can fail (a full disk, a folder another program
holds). The answer was computed: the reply goes out cut as it would be, with
no cursor that leads nowhere, and ``stored_answer.not_kept`` says, with a
code, that it cannot be read on, why, and what to ask instead. One place
does this for every tool on the presenter (``presenter.present_many``).
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import presenter as presenter_module
from ltspice_mcp.lib.presenter import present
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import STORED_ANSWER_SCHEMA
from tests.test_analyze_results_through_the_presenter import _analyze, _job

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {"type": "array", "items": {"type": "string"}, "x-cut": "items"},
        "stored_answer": STORED_ANSWER_SCHEMA,
    },
}


@pytest.fixture
def full_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    real = presenter_module.atomic_write_json

    def write(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.startswith(presenter_module.STORED_PREFIX):
            raise OSError(errno.ENOSPC, "No space left on device")
        real(path, *args, **kwargs)

    monkeypatch.setattr(presenter_module, "atomic_write_json", write)


def _says_it_was_not_kept(node: dict[str, Any]) -> None:
    not_kept = node["stored_answer"]["not_kept"]
    assert not_kept["code"] == "stored_answer_not_kept"
    assert "No space left on device" in not_kept["message"]
    assert "budget" in not_kept["message"]
    assert "path" not in node["stored_answer"]
    assert all("next" not in entry for entry in node["cut"].values())


def test_a_cut_reply_is_sent_when_its_answer_cannot_be_kept(tmp_path: Path, full_disk: None):
    answer = {"rows": [f"row {index} " + "x" * 40 for index in range(100)]}
    shown = present(
        answer, _SCHEMA, target=300, tool="t", snapshot={}, store=Store(tmp_path), ttl_hours=1
    )
    reply = shown.reply
    assert 1 <= len(reply["rows"]) < 100
    assert reply["cut"]["/rows"]["returned"] == len(reply["rows"])
    _says_it_was_not_kept(reply)
    jsonschema.Draft202012Validator(_SCHEMA).validate(reply)


@pytest.mark.asyncio
async def test_analyze_results_answers_when_its_answer_cannot_be_kept(
    state_no_sim: SessionState, work_dir: Path, full_disk: None
):
    job = _job(state_no_sim, work_dir, 60)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job, "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "100u"}],
            "budget": 1000,
        },
    )
    _says_it_was_not_kept(data)
    assert "next" not in data and "cursor" not in data
    assert data["results"]["v"]["values"]


def test_an_inner_answer_not_kept_says_so_at_its_own_root(tmp_path: Path, full_disk: None):
    """A receipt and its attached analysis are two answers under one target
    (``present_many``): each says, at its root, that it was not kept."""
    from ltspice_mcp.lib.presenter import Answer, present_many

    inner = {"rows": [f"inner {index} " + "y" * 40 for index in range(60)]}
    reply = {"rows": [f"outer {index} " + "x" * 40 for index in range(60)], "inner": inner}
    outer_schema = {**_SCHEMA, "properties": {**_SCHEMA["properties"], "inner": _SCHEMA}}
    shown = present_many(
        reply,
        [Answer("", outer_schema, "outer", {}), Answer("/inner", _SCHEMA, "inner", {})],
        target=400,
        store=Store(tmp_path),
        ttl_hours=1,
    ).reply
    for node in (shown, shown["inner"]):
        if "cut" in node:
            _says_it_was_not_kept(node)
    assert "cut" in shown or "cut" in shown["inner"]
    jsonschema.Draft202012Validator(outer_schema).validate(shown)
