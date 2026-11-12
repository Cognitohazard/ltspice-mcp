"""Tests for structured job lifecycle event emission.

The process's stderr logger is the only channel these events have: the MCP
logging capability that could once relay them to the client is deprecated as
of the 2026-07-28 revision and is not served.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from ltspice_mcp.api._exceptions import ApiValidationError
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.store import Store
from ltspice_mcp.server import call_tool, get_prompt, read_resource
from ltspice_mcp.state import SessionState
from tests.conftest import LIVENESS_S, SyncApi, call_tool_params, dispatch, fake_request_context


@pytest.fixture
def events_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture records from the dedicated events logger."""
    caplog.set_level(logging.INFO, logger="ltspice_mcp.events")
    return caplog


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    config = ServerConfig(
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        persist_jobs=False,
        log_level="DEBUG",
    )
    return SessionState.create(config, available={})


def _events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """Structured event payloads from the captured records."""
    return [
        record.ltspice_event  # type: ignore[attr-defined]
        for record in caplog.records
        if hasattr(record, "ltspice_event")
    ]


def _experiment(working_dir: Path, circuit: Path, *, job_id: str, status: str) -> ExperimentJob:
    case = ExperimentCase(
        case_id="case_0000",
        run_index=0,
        circuit="dut",
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="a" * 64,
        assignments={},
        status="queued",
    )
    return ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=Store(working_dir).job_record(job_id),
        cases=[case],
        sources=[
            SourceRecord(
                circuit="dut",
                path=circuit,
                sha256="b" * 64,
                staged_deck=circuit,
                manifest=[],
                simulator="FakeSim",
                dialect="ltspice",
            )
        ],
        simulator="FakeSim",
        completeness=Completeness(declared=1, expanded=1),
        status=status,  # type: ignore[arg-type]
    )


class TestEmitJobEvent:
    def test_emits_structured_payload(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_test_001", status="queued")
        emit_job_event("submitted", job, simulator="FakeSim")

        events = _events(events_caplog)
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "submitted"
        assert event["kind"] == "experiment"
        assert event["job_id"] == "exp_test_001"
        assert event["sources"] == [str(circuit)]
        assert event["simulator"] == "FakeSim"
        assert "ts" in event
        assert "duration_s" in event

    def test_duration_on_completion(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_dur", status="completed")
        emit_job_event("completed", job)
        event = _events(events_caplog)[-1]
        # duration >= 0 (started_at defaults to now)
        assert event["duration_s"] is not None
        assert event["duration_s"] >= 0

    def test_extra_kwargs_merge_into_payload(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_extra", status="running")
        emit_job_event("started", job, expanded=3, note="hello")
        event = _events(events_caplog)[-1]
        assert event["expanded"] == 3
        assert event["note"] == "hello"

    def test_payload_is_json_serialisable(
        self, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        # The payload rides a log record into whatever handler the host
        # installs; a Path left unstringified there fails at the sink, far from
        # the emit that produced it.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_json", status="running")
        emit_job_event("started", job)
        json.dumps(_events(events_caplog)[-1])


class TestJobRegistryEmitsOnAdd:
    def test_registering_an_experiment_emits_submitted(
        self, state: SessionState, events_caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _experiment(tmp_path, circuit, job_id="exp_add", status="queued")
        state.add_experiment_job(job, already_persisted=True)

        event = _events(events_caplog)[-1]
        assert event["event"] == "submitted"
        assert event["kind"] == "experiment"
        assert event["job_id"] == "exp_add"


_CAPABILITIES = {"queries": [{"kind": "capabilities"}]}


@pytest.mark.asyncio
class TestEachToolCallEmitsOne:
    """The tool, how long it took, how it ended and the size of the reply as
    sent; never the arguments."""

    async def test_an_answered_and_a_refused_call(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        ctx = fake_request_context(state_no_sim)
        answered = await call_tool(ctx, call_tool_params("inspect", _CAPABILITIES))
        refused = await call_tool(ctx, call_tool_params("run_experiments", {"missing": "field"}))
        events = _events(events_caplog)
        assert [{k: v for k, v in e.items() if k not in ("ts", "duration_s")} for e in events] == [
            {
                "event": "tool_call",
                "door": "mcp",
                "tool": "inspect",
                "outcome": "ok",
                "code": None,
                "reply_chars": len(answered.model_dump_json(by_alias=True, exclude_none=True)),
                "fields": ["queries"],
                "choices": ["queries.kind=capabilities"],
            },
            {
                # Refused before validation finished: nothing it set is known.
                "event": "tool_call",
                "door": "mcp",
                "tool": "run_experiments",
                "outcome": "error",
                "code": "invalid_arguments",
                "reply_chars": len(refused.model_dump_json(by_alias=True, exclude_none=True)),
                "fields": None,
                "choices": None,
            },
        ]

    async def test_a_cancelled_call(
        self,
        state_no_sim: SessionState,
        events_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A call the client gave up on, which the SDK cancels: the one most
        worth a record, and it has no result to measure."""
        entered = asyncio.Event()

        async def waits_until_cancelled(arguments, state):
            entered.set()
            await asyncio.Event().wait()

        inspect = state_no_sim.tool_dispatch["inspect"]
        monkeypatch.setitem(
            state_no_sim.tool_dispatch, "inspect", replace(inspect, handler=waits_until_cancelled)
        )
        call = asyncio.create_task(
            call_tool(
                fake_request_context(state_no_sim), call_tool_params("inspect", _CAPABILITIES)
            )
        )
        await asyncio.wait_for(entered.wait(), LIVENESS_S)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        (event,) = _events(events_caplog)
        assert {k: v for k, v in event.items() if k not in ("ts", "duration_s")} == {
            "event": "tool_call",
            "door": "mcp",
            "tool": "inspect",
            "outcome": "cancelled",
            "code": None,
            "reply_chars": None,
            # The stand-in handler validates nothing.
            "fields": None,
            "choices": None,
        }


def _names(event: dict) -> tuple[list[str] | None, list[str] | None]:
    return event["fields"], event["choices"]


@pytest.mark.asyncio
class TestEventNamesWhatACallSet:
    """Each tool call's record names the argument fields it set and the words
    of the server's own vocabulary its values chose, so the use of each
    optional feature can be counted from a log."""

    async def test_analyze_results_options(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        raw = state_no_sim.working_dir / "missing.raw"
        await dispatch(
            state_no_sim,
            "analyze_results",
            {
                "sources": [{"path": str(raw), "label": "a"}],
                "recipes": [
                    {
                        "key": "v",
                        "metric": "value",
                        "expr": "V(out)",
                        "reduce": ["mean"],
                        "spec": {"max": 1.0, "allow_incomplete": True},
                    },
                    {"key": "p", "metric": "plot", "signals": ["V(out)"]},
                ],
                "group_by": ["circuit"],
                "include": {"outliers": True, "fields": ["value.v"], "per_run": True},
                "budget": 4000,
            },
        )
        await dispatch(
            state_no_sim,
            "analyze_results",
            {"continue": {"result_set_id": "rs_1", "cursor": "c"}},
        )
        first, resumed = _events(events_caplog)
        assert _names(first) == (
            [
                "budget",
                "group_by",
                "include",
                "include.fields",
                "include.outliers",
                "include.per_run",
                "recipes",
                "recipes[metric=plot].key",
                "recipes[metric=plot].signals",
                "recipes[metric=value].expr",
                "recipes[metric=value].key",
                "recipes[metric=value].reduce",
                "recipes[metric=value].spec",
                "recipes[metric=value].spec.allow_incomplete",
                "recipes[metric=value].spec.max",
                "sources",
                "sources.label",
                "sources.path",
            ],
            ["recipes.metric=plot", "recipes.metric=value", "recipes[metric=value].reduce=mean"],
        )
        # The continuation is named by its wire name, ``continue``.
        assert _names(resumed) == (
            ["continue", "continue.cursor", "continue.result_set_id"],
            [],
        )

    async def test_inspect_kinds_and_their_paging(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        """Fields under a query are named under its kind, so a call that
        batches several kinds still says which one paged."""
        await dispatch(
            state_no_sim,
            "inspect",
            {
                "queries": [
                    {"kind": "simulator_docs", "name": "MEAS-REFERENCE.md", "cursor": "c"},
                    {"kind": "hierarchy", "path": "x.cir", "simulator": "ngspice", "cursor": "c"},
                    {"kind": "net", "path": "a.asc", "at": [16, 32], "cursor": "c"},
                    {"kind": "symbols", "cursor": "c"},
                    # Refused by its own validation, it is still a use: its
                    # kind and the fields it declares.
                    {"kind": "hierarchy", "path": "y.cir", "cursor": "c"},
                ]
            },
        )
        (event,) = _events(events_caplog)
        assert _names(event) == (
            [
                "queries",
                "queries[kind=hierarchy].cursor",
                "queries[kind=hierarchy].path",
                "queries[kind=hierarchy].simulator",
                "queries[kind=net].at",
                "queries[kind=net].cursor",
                "queries[kind=net].path",
                "queries[kind=simulator_docs].cursor",
                "queries[kind=simulator_docs].name",
                "queries[kind=symbols].cursor",
            ],
            [
                "queries.kind=hierarchy",
                "queries.kind=net",
                "queries.kind=simulator_docs",
                "queries.kind=symbols",
            ],
        )

    async def test_jobs_action_and_edit_ops(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        await dispatch(
            state_no_sim,
            "jobs",
            {"action": "runs", "job_id": "exp_1", "run_fields": ["status"], "cursor": "c"},
        )
        await dispatch(
            state_no_sim,
            "edit_schematic",
            {
                "path": "a.asc",
                "ops": [
                    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 0, "y": 0},
                    {"op": "remove_component", "reference": "R2"},
                ],
            },
        )
        runs, edit = _events(events_caplog)
        assert _names(runs) == (["action", "cursor", "job_id", "run_fields"], ["action=runs"])
        assert edit["choices"] == ["ops.op=add_component", "ops.op=remove_component"]
        assert "ops[op=add_component].reference" in edit["fields"]
        assert "ops[op=remove_component].reference" in edit["fields"]

    async def test_a_field_sent_at_its_default_is_not_set(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        await dispatch(
            state_no_sim,
            "inspect",
            {"queries": [{"kind": "capabilities"}], "budget": None},
        )
        (event,) = _events(events_caplog)
        assert _names(event) == (["queries"], ["queries.kind=capabilities"])

    async def test_no_argument_value_reaches_the_record(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        """Every value a caller writes here is distinctive: none may appear in
        the payload or in the message, whichever field carried it — a path, a
        label, a signal, an expression, a number, a mapping's key, a cursor,
        an id, a free-text value of a field that also takes vocabulary words."""
        secrets = [
            "Zq7path",
            "Zq7label",
            "Zq7sig",
            "Zq7key",
            "Zq7cursor",
            "Zq7job",
            "Zq7attr",
            "Zq7attrvalue",
            "Zq7ref",
            "Zq7group",
            "Zq7doc",
            "Zq7tool",
            "987654.5",
        ]
        ctx = fake_request_context(state_no_sim)
        calls = [
            (
                "analyze_results",
                {
                    "sources": [{"path": "Zq7path/a.raw", "label": "Zq7label"}],
                    "recipes": [
                        {
                            "key": "Zq7key",
                            "metric": "value",
                            "expr": "V(Zq7sig)",
                            "at": 987654.5,
                            "spec": {"max": 987654.5},
                        },
                        # A refused recipe is named from its raw keys.
                        {"key": "Zq7key2", "metric": "value", "Zq7attr": "Zq7sig"},
                    ],
                    "group_by": ["Zq7group"],
                },
            ),
            ("analyze_results", {"continue": {"result_set_id": "Zq7job", "cursor": "Zq7cursor"}}),
            ("jobs", {"action": "status", "job_id": "Zq7job"}),
            (
                "inspect",
                {
                    "queries": [
                        {"kind": "simulator_docs", "name": "Zq7doc", "cursor": "Zq7cursor"},
                        {"kind": "net", "path": "Zq7path.asc", "at": "Zq7ref.A"},
                    ]
                },
            ),
            (
                "edit_schematic",
                {
                    "path": "Zq7path.asc",
                    "ops": [
                        {
                            "op": "add_component",
                            "reference": "Zq7ref",
                            "symbol": "res",
                            "x": 987654,
                            "y": 0,
                            "attributes": {"Zq7attr": "Zq7attrvalue"},
                        }
                    ],
                },
            ),
        ]
        for tool, arguments in calls:
            await call_tool(ctx, call_tool_params(tool, arguments))
        with pytest.raises(MCPError):
            await call_tool(ctx, call_tool_params("Zq7tool", {"path": "Zq7path"}))
        events = _events(events_caplog)
        assert len(events) == len(calls) + 1
        assert all(event["fields"] for event in events[:-1])
        records = [r for r in events_caplog.records if hasattr(r, "ltspice_event")]
        text = "\n".join(
            [json.dumps(event) for event in events] + [r.getMessage() for r in records]
        )
        assert [secret for secret in secrets if secret in text] == []
        assert events[-1]["tool"] is None


@pytest.mark.asyncio
class TestResourceReadsAndPromptGetsEmitOne:
    """A read names the template it matched, a prompt get the prompt: never
    the concrete URI, whose path is the caller's, or a prompt argument."""

    async def test_a_read_names_its_template(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        ctx = fake_request_context(state_no_sim, "resources/read")
        await read_resource(ctx, types.ReadResourceRequestParams(uri="spice://config"))
        with pytest.raises(MCPError):
            await read_resource(
                ctx, types.ReadResourceRequestParams(uri="spice://results/Zq7job/signals")
            )
        with pytest.raises(MCPError):
            await read_resource(ctx, types.ReadResourceRequestParams(uri="spice://Zq7nothing"))
        events = _events(events_caplog)
        assert [{k: v for k, v in e.items() if k not in ("ts", "duration_s")} for e in events] == [
            {
                "event": "resource_read",
                "template": "spice://config",
                "outcome": "ok",
                "code": None,
            },
            {
                "event": "resource_read",
                "template": "spice://results/{job_id}/signals",
                "outcome": "error",
                "code": events[1]["code"],
            },
            {"event": "resource_read", "template": None, "outcome": "error", "code": None},
        ]
        assert events[1]["code"]
        text = json.dumps(events) + "".join(r.getMessage() for r in events_caplog.records)
        assert "Zq7" not in text

    async def test_a_prompt_get_names_its_prompt(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        ctx = fake_request_context(state_no_sim, "prompts/get")
        await get_prompt(
            ctx,
            types.GetPromptRequestParams(
                name="characterize_filter", arguments={"path": "Zq7path.cir", "node": "Zq7node"}
            ),
        )
        with pytest.raises(ValueError, match="Unknown prompt"):
            await get_prompt(ctx, types.GetPromptRequestParams(name="Zq7prompt"))
        events = _events(events_caplog)
        assert [{k: v for k, v in e.items() if k not in ("ts", "duration_s")} for e in events] == [
            {
                "event": "prompt_get",
                "prompt": "characterize_filter",
                "outcome": "ok",
                "code": None,
            },
            {"event": "prompt_get", "prompt": None, "outcome": "error", "code": None},
        ]
        text = json.dumps(events) + "".join(r.getMessage() for r in events_caplog.records)
        assert "Zq7" not in text


class TestPythonApiCallsEmitOne:
    """An operation of the Python API writes the tool call's record, through
    the door ``python_api``, with the API's own options it set."""

    def test_an_operation_and_a_refused_one(
        self, state_no_sim: SessionState, events_caplog: pytest.LogCaptureFixture
    ):
        api = SyncApi(state_no_sim)
        api.inspect(queries=[{"kind": "capabilities"}])
        api.inspect(queries=[{"kind": "capabilities"}], raw_page=True)
        with pytest.raises(ApiValidationError):
            api.inspect(queries=[])
        events = _events(events_caplog)
        assert [{k: v for k, v in e.items() if k not in ("ts", "duration_s")} for e in events] == [
            {
                "event": "tool_call",
                "door": "python_api",
                "tool": "inspect",
                "outcome": "ok",
                "code": None,
                "reply_chars": None,
                "fields": ["queries"],
                "choices": ["queries.kind=capabilities"],
                "controls": [],
            },
            {
                "event": "tool_call",
                "door": "python_api",
                "tool": "inspect",
                "outcome": "ok",
                "code": None,
                "reply_chars": None,
                "fields": ["queries"],
                "choices": ["queries.kind=capabilities"],
                "controls": ["raw_page"],
            },
            {
                "event": "tool_call",
                "door": "python_api",
                "tool": "inspect",
                "outcome": "error",
                "code": "invalid_arguments",
                "reply_chars": None,
                "fields": None,
                "choices": None,
                "controls": [],
            },
        ]
