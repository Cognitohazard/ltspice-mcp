"""Validation failures stay compact and point to the right tool surface."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib.recipes import validate_recipe
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import (
    RunExperimentsInput,
    handle_run_experiments,
)
from tests.conftest import _FakeServer, fake_simulator


@pytest.mark.asyncio
async def test_malformed_attached_recipe_error_is_compact(
    state_with_sim: SessionState,
    work_dir,
    monkeypatch: pytest.MonkeyPatch,
):
    submissions: list[str] = []
    fake_simulator(monkeypatch, submissions)
    deck = work_dir / "invalid-recipe.cir"
    deck.write_text("V1 in 0 1\n.op\n.end\n")
    large_value = "DO_NOT_ECHO_" * 200
    args = RunExperimentsInput.model_validate(
        {
            "request_id": "invalid-recipe",
            "circuits": [{"path": str(deck)}],
            "analyze": {
                "recipes": [
                    {
                        "key": "wave",
                        "metric": "waveform",
                        "max_points": 0,
                        "format": large_value,
                        "unexpected": large_value,
                    }
                ]
            },
        }
    )

    result = await handle_run_experiments(args, state_with_sim)

    assert result.isError
    assert result.structuredContent is not None
    message = result.structuredContent["error"]["message"]
    assert len(message) < 400
    assert "Field required" in message
    assert "https://" not in message
    assert "DO_NOT_ECHO_" not in message
    assert submissions == []


def test_distinct_missing_fields_are_not_deduplicated():
    with pytest.raises(ValidationError) as excinfo:
        validate_recipe({"key": "delay", "metric": "timing"})

    message = compact_validation_error(excinfo.value)

    assert message.count("Field required") == 2
    assert "timing.from" in message
    assert "timing.to" in message


def _consolidated_state(config) -> SessionState:
    config.tool_profile = "consolidated"
    return SessionState.create(config, available={})


@pytest.mark.asyncio
async def test_top_level_budget_refers_to_accepting_tools(config):
    state = _consolidated_state(config)
    arguments = {
        "path": "dut.cir",
        "budget": 500,
    }

    with (
        patch("ltspice_mcp.server.server", _FakeServer(state)),
        pytest.raises(ValueError, match="Invalid arguments") as excinfo,
    ):
        await call_tool("verify_circuit", arguments)

    message = str(excinfo.value)
    assert (
        "Field 'budget' is accepted by analyze_results, inspect, jobs, run_experiments." in message
    )


@pytest.mark.asyncio
async def test_top_level_continue_alias_refers_to_analyze_results(config):
    state = _consolidated_state(config)
    arguments = {
        "request_id": "continue-referral",
        "circuits": [{"path": "dut.cir"}],
        "continue": {"cursor": "opaque"},
    }

    with (
        patch("ltspice_mcp.server.server", _FakeServer(state)),
        pytest.raises(ValueError, match="Invalid arguments") as excinfo,
    ):
        await call_tool("run_experiments", arguments)

    message = str(excinfo.value)
    assert "Field 'continue' is accepted by analyze_results." in message


@pytest.mark.asyncio
async def test_nested_extra_field_has_no_cross_tool_referral(config):
    state = _consolidated_state(config)
    arguments = {
        "request_id": "nested-extra",
        "circuits": [{"path": "dut.cir", "budget": 500}],
    }

    with (
        patch("ltspice_mcp.server.server", _FakeServer(state)),
        pytest.raises(ValueError, match="Invalid arguments") as excinfo,
    ):
        await call_tool("run_experiments", arguments)

    message = str(excinfo.value)
    assert "circuits.0.budget" in message
    assert "is accepted by" not in message


def test_plain_value_error_is_rendered_unchanged():
    message = "unsupported recipe metric 'unknown'"

    assert compact_validation_error(ValueError(message)) == message
