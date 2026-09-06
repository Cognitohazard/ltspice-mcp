"""Validation failures stay compact and point to the right tool surface."""

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
from tests.conftest import call_tool_params, fake_request_context, fake_simulator, tool_text


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

    assert result.is_error
    assert result.structured_content is not None
    message = result.structured_content["error"]["message"]
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
    return SessionState.create(config, available={})


@pytest.mark.asyncio
async def test_top_level_budget_refers_to_accepting_tools(config):
    state = _consolidated_state(config)
    arguments = {
        "path": "dut.cir",
        "budget": 500,
    }

    result = await call_tool(
        fake_request_context(state), call_tool_params("verify_circuit", arguments)
    )

    assert result.is_error
    message = tool_text(result)
    # The field, and every tool the referral must send the caller to.
    assert "'budget'" in message
    for tool in ("analyze_results", "inspect", "jobs", "run_experiments"):
        assert tool in message, f"referral does not name {tool}: {message}"
    assert "edit_schematic" not in message


@pytest.mark.asyncio
async def test_top_level_continue_alias_refers_to_analyze_results(config):
    state = _consolidated_state(config)
    arguments = {
        "request_id": "continue-referral",
        "circuits": [{"path": "dut.cir"}],
        "continue": {"cursor": "opaque"},
    }

    result = await call_tool(
        fake_request_context(state), call_tool_params("run_experiments", arguments)
    )

    assert result.is_error
    message = tool_text(result)
    # One tool takes it, so the referral names that one and no other.
    assert "'continue'" in message
    assert "analyze_results" in message
    assert not [
        tool for tool in ("inspect", "jobs", "edit_schematic", "verify_circuit") if tool in message
    ]


@pytest.mark.asyncio
async def test_nested_extra_field_has_no_cross_tool_referral(config):
    state = _consolidated_state(config)
    arguments = {
        "request_id": "nested-extra",
        "circuits": [{"path": "dut.cir", "budget": 500}],
    }

    result = await call_tool(
        fake_request_context(state), call_tool_params("run_experiments", arguments)
    )

    assert result.is_error
    message = tool_text(result)
    assert "circuits.0.budget" in message
    assert "is accepted by" not in message


def test_plain_value_error_is_rendered_unchanged():
    message = "unsupported recipe metric 'unknown'"

    assert compact_validation_error(ValueError(message)) == message
