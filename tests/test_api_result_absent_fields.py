"""A Python API result answers a field it left out with that field's empty value.

A reply leaves out every field that says nothing (``lib/reply_fields.py``), and
the Python API's results keep that shape. Reading such a field answers the
empty value of its declared type, so a script need not guard every read; a key
the operation never declares still raises, and the result serialises exactly
as the sent dict does.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import ApiCallError
from ltspice_mcp.api._methods import ApiMethodsMixin, _CataloguedMethod
from ltspice_mcp.api._result import SentResult, sent_result
from ltspice_mcp.lib.reply_fields import IMPLIED
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from ltspice_mcp.tools._base import as_sent
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import LIVENESS_S, SyncApi, fake_simulator

pytestmark = pytest.mark.asyncio


def _rc_deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text(
        "* rc\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n", encoding="utf-8"
    )
    return deck


async def _verify(state: SessionState, work_dir: Path) -> dict[str, Any]:
    args = {"path": str(_rc_deck(work_dir)), "checks": ["syntax", "quality"]}
    return await asyncio.to_thread(lambda: SyncApi(state).verify_circuit(**args))


async def test_a_left_out_list_reads_as_empty(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    assert "failures" not in result
    assert result["failures"] == []
    assert result.get("failures") == []
    # A caller's own default is still the caller's.
    assert result.get("failures", "none") == "none"
    assert "failures" not in result, "reading a left-out field must not add it"


async def test_a_misspelt_key_still_raises(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    with pytest.raises(KeyError):
        result["failurez"]
    assert result.get("failurez") is None


async def test_the_result_serialises_as_the_sent_dict(state_no_sim: SessionState, work_dir: Path):
    result = await _verify(state_no_sim, work_dir)
    plain = json.loads(json.dumps(result))
    assert json.dumps(result) == json.dumps(plain)
    assert json.dumps(result, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert repr(result) == repr(plain)
    assert result == plain
    assert list(result) == list(plain)
    assert len(result) == len(plain)
    assert type(dict(result)) is dict
    assert type(result.copy()) is dict


def _run_args(work_dir: Path, request_id: str) -> dict[str, Any]:
    deck = work_dir / "op.cir"
    deck.write_text("* deck\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n", encoding="utf-8")
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
    }


async def test_a_run_row_answers_its_left_out_assignments(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A run with no sweep has no assignments; every route that hands a run
    row to a script answers the left-out field the same way."""
    fake_simulator(monkeypatch)
    arguments = _run_args(work_dir, "absent-assignments")
    # The run itself goes on this test's loop, which outlives it; each API
    # call below runs on a loop of its own.
    ran = await handle_run_experiments(
        RunExperimentsInput.model_validate(arguments), state_with_sim
    )
    assert ran.structured_content is not None
    assert ran.structured_content["status"] == "completed", ran.structured_content
    api = SyncApi(state_with_sim)
    # The same request again replays the finished job's receipt.
    receipt = await asyncio.to_thread(lambda: api.run_experiments(**arguments))
    job_id = receipt["job_id"]
    status = await asyncio.to_thread(lambda: api.jobs(action="status", job_id=job_id))
    waited = await asyncio.to_thread(lambda: api.wait(job_id, timeout_s=LIVENESS_S))
    raw = await asyncio.to_thread(lambda: api.jobs(action="status", job_id=job_id, raw_page=True))
    for result in (receipt, status, waited, raw):
        rows = result["runs"]["items"]
        assert rows
        for row in rows:
            assert "assignments" not in row
            assert row["assignments"] == {}
            assert row.get("assignments") == {}
            with pytest.raises(KeyError):
                row["assignmentz"]


async def test_an_error_payload_answers_its_left_out_fields(state_no_sim: SessionState):
    with pytest.raises(ApiCallError) as caught:
        await asyncio.to_thread(
            lambda: SyncApi(state_no_sim).jobs(action="status", job_id="no-such-job")
        )
    payload = caught.value.payload
    assert "observations" not in payload
    assert payload["observations"] == []
    assert payload.get("observations") == []
    with pytest.raises(KeyError):
        payload["observationz"]


# --- Guards over every catalogued operation -----------------------------------


def _catalogued() -> dict[str, _CataloguedMethod]:
    return {
        name: method
        for name, method in vars(ApiMethodsMixin).items()
        if isinstance(method, _CataloguedMethod)
    }


def _parts(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The schemas and every branch inside them."""
    out: list[dict[str, Any]] = []
    for schema in schemas:
        out.append(schema)
        for keyword in ("oneOf", "anyOf", "allOf"):
            out.extend(_parts(schema.get(keyword, [])))
    return out


def _declared(parts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fields: dict[str, list[dict[str, Any]]] = {}
    for part in parts:
        for name, sub in part.get("properties", {}).items():
            fields.setdefault(name, []).append(sub)
    return fields


def _kinds(parts: list[dict[str, Any]]) -> set[str]:
    kinds: set[str] = set()
    for part in parts:
        declared = part.get("type")
        kinds.update([declared] if isinstance(declared, str) else declared or [])
    return kinds - {"null"}


def _expected_empty(kinds: set[str]) -> Any:
    return {
        frozenset({"array"}): [],
        frozenset({"object"}): {},
        frozenset({"boolean"}): False,
    }.get(frozenset(kinds))


def _nested(path: tuple[str | None, ...]) -> Any:
    """A value holding one empty object at ``path``; None steps into a row."""
    value: Any = {}
    for step in reversed(path):
        value = [value] if step is None else {step: value}
    return value


def _at(value: Any, path: tuple[str | None, ...]) -> Any:
    for step in path:
        value = value[0] if step is None else value[step]
    return value


def _check_node(
    tool: str, path: tuple[str | None, ...], parts: list[dict[str, Any]], depth: int
) -> int:
    """Every field declared at ``path`` answers its declared empty value when
    left out; returns how many fields were checked."""
    node = _at(sent_result(_nested(path), tool), path)
    assert isinstance(node, SentResult), (tool, path)
    checked = 0
    for name, subs in _declared(parts).items():
        sub_parts = _parts(subs)
        kinds = _kinds(sub_parts)
        expected = _expected_empty(kinds)
        if name in IMPLIED:
            # Its overall answer is absent here, so it answers its own type's empty.
            assert kinds == {"boolean"}, (tool, path, name)
        assert name not in node
        assert node[name] == expected, (tool, path, name)
        assert type(node[name]) is type(expected) or isinstance(node[name], dict)
        assert node.get(name) == expected, (tool, path, name)
        checked += 1
        if depth == 0:
            continue
        if kinds == {"object"}:
            checked += _check_node(tool, (*path, name), sub_parts, depth - 1)
        rows = _parts([s["items"] for s in sub_parts if isinstance(s.get("items"), dict)])
        if kinds == {"array"} and _declared(rows):
            checked += _check_node(tool, (*path, name, None), rows, depth - 1)
    with pytest.raises(KeyError):
        node["no such field"]
    return checked


@pytest.mark.parametrize("tool", sorted(_catalogued()))
async def test_every_declared_field_answers_its_empty_value(tool: str):
    """At the top level, in nested objects and in rows, by the operation's own
    output schema, so a field a schema gains later is covered with no list here."""
    schema = get_tools()[1][tool].definition.output_schema
    assert schema is not None
    checked = _check_node(tool, (), _parts([schema]), depth=4)
    assert checked > len(schema.get("properties", {}))


async def test_a_field_declared_in_any_branch_counts_as_declared():
    """``jobs`` declares its fields per action, in branches: a field only some
    actions carry (``timed_out``, ``wait``'s) still answers."""
    schema = get_tools()[1]["jobs"].definition.output_schema
    assert schema is not None
    assert "timed_out" not in schema["properties"]
    result = sent_result({}, "jobs")
    assert result["timed_out"] is False
    with pytest.raises(KeyError):
        result["timed_outt"]


async def test_an_implied_sub_answer_reads_as_the_answer_that_implies_it():
    """A sub-answer is left out when the overall answer beside it implies it
    (``IMPLIED``): an equivalent comparison is structurally equivalent too."""
    schema = get_tools()[1]["verify_circuit"].definition.output_schema
    assert schema is not None
    paths = [
        path
        for path, parts in _object_nodes(_parts([schema]), ())
        if {"equivalent", "structurally_equivalent"} <= set(_declared(parts))
    ]
    assert paths, "verify_circuit no longer declares a comparison"
    for path in paths:
        value = _nested(path)
        _at(value, path)["equivalent"] = True
        node = _at(sent_result(value, "verify_circuit"), path)
        assert node["structurally_equivalent"] is True
        assert node["directives_equivalent"] is True
        _at(value, path)["equivalent"] = False
        node = _at(sent_result(value, "verify_circuit"), path)
        assert node["structurally_equivalent"] is False


def _object_nodes(
    parts: list[dict[str, Any]], path: tuple[str | None, ...], depth: int = 4
) -> Iterator[tuple[tuple[str | None, ...], list[dict[str, Any]]]]:
    yield path, parts
    if depth == 0:
        return
    for name, subs in _declared(parts).items():
        sub_parts = _parts(subs)
        kinds = _kinds(sub_parts)
        if kinds == {"object"}:
            yield from _object_nodes(sub_parts, (*path, name), depth - 1)
        rows = _parts([s["items"] for s in sub_parts if isinstance(s.get("items"), dict)])
        if kinds == {"array"} and _declared(rows):
            yield from _object_nodes(rows, (*path, name, None), depth - 1)


def _returning(name: str, value: Any) -> _CataloguedMethod:
    def operation(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(value, BaseException):
            raise value
        return value

    operation.__name__ = name
    return _CataloguedMethod(operation)


@pytest.mark.parametrize("tool", sorted(_catalogued()))
@pytest.mark.parametrize("raw_page", [False, True])
async def test_no_catalogued_result_leaves_run_as_a_plain_dict(tool: str, raw_page: bool):
    """Every catalogued operation is a registered tool with an output schema,
    and ``_run`` hands back its result and its error payload wrapped."""
    assert tool in get_tools()[1]
    kwargs = {"raw_page": True} if raw_page else {}
    result = _returning(tool, {"outcome": "ok"})._run(as_sent, (), kwargs)
    assert type(result) is SentResult
    error = ApiCallError("refused", payload={"outcome": "error"})
    with pytest.raises(ApiCallError) as caught:
        _returning(tool, error)._run(as_sent, (), kwargs)
    assert type(caught.value.payload) is SentResult


async def test_the_catalogued_operations_are_the_six_tools():
    assert set(_catalogued()) == {
        "run_experiments",
        "jobs",
        "analyze_results",
        "inspect",
        "edit_schematic",
        "verify_circuit",
    }
