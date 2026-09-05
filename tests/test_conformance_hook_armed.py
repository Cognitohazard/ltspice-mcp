"""Guards that the output-schema conformance hook actually protects, and that
the end-to-end scenario script only calls tools that still exist.

The session-scoped hook in ``conftest.py`` validates every ``structuredContent``
emission against its tool's declared ``output_schema``. It can quietly stop
protecting anything in two ways: the handler-frame walk finds no match and
skips validation, or the response-helper patch never takes effect and the
unpatched helpers run. The first test pins both at once: a rejection is only
possible when the patched helper runs AND the frame walk attributes the
emission, so forcing a real schema violation through a registered handler
proves the whole chain is armed.

The second test keeps ``scenario_active_filter.py`` honest: a renamed or
removed tool would leave the script calling a name the server no longer
answers to, and the script isn't exercised by the normal suite.
"""

import importlib
import re
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import get_tools_for_profile
from ltspice_mcp.tools.analysis import SignalStatsInput, handle_signal_stats
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.experiments import JobsInput, handle_jobs
from tests.conftest import (
    NO_CONTRACT_DELEGATES,
    make_experiment_job,
    stage_recorded_fixture,
)


async def test_hook_rejects_a_schema_violating_emission(
    state_no_sim: SessionState, work_dir: Path
):
    """A registered handler emitting structuredContent that contradicts its
    output_schema is caught at emission time, not silently passed through."""
    # jobs' status branch relays a job's recorded failures verbatim, and the
    # schema types each failure's ``message`` as a string. Seeding a non-string
    # makes the real handler emit a genuinely non-conforming payload from real
    # registry state through the (patched) format_response. The cast injects
    # the bad value deliberately — the point is that it is NOT a valid str.
    job = make_experiment_job(state_no_sim, job_id="hook_specimen", status="completed")
    job.failures = [{"case_id": "case-0000", "code": "run_failed", "message": cast("str", [123])}]

    with pytest.raises(AssertionError, match="output_schema"):
        await handle_jobs(
            JobsInput.model_validate({"action": "status", "job_id": job.job_id}),
            state_no_sim,
        )


def _scenario_tool_names() -> set[str]:
    """Tool names passed to ``session.call_tool(...)`` in the scenario script."""
    source = (Path(__file__).parent / "scenario_active_filter.py").read_text()
    return set(re.findall(r"call_tool\(\s*[\"']([A-Za-z_]\w*)[\"']", source))


def test_scenario_calls_only_registered_tools():
    """Every tool the scenario script drives must exist in the live registry
    (aliases included), so a rename can't leave the script calling a dead name."""
    _, dispatch = get_tools_for_profile("consolidated")
    called = _scenario_tool_names()
    assert called, "parsed no call_tool names from the scenario — the regex has rotted"
    unknown = sorted(called - set(dispatch))
    assert not unknown, f"scenario calls tools absent from the registry: {unknown}"


# ---------------------------------------------------------------------------
# Delegated-contract attribution (the schema belongs to the handler, not to
# its registration): analyze_results delegates to internal compute adapters,
# and each adapter's structuredContent must be validated against the
# ADAPTER's declared contract, never against the delegating tool's envelope.
# ---------------------------------------------------------------------------


def _tran_recipe() -> list[dict[str, Any]]:
    return [{"key": "signal_stats", "metric": "signal_stats", "signal": "V(out)"}]


async def test_adapter_emission_validates_against_the_adapters_own_contract(
    state_no_sim: SessionState, work_dir: Path
):
    """The adapter's real payload is legal for its own declared contract and
    ILLEGAL for the delegating envelope — so a walk that misattributed the
    emission to analyze_results would raise. Both discriminations asserted."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    result = await handle_signal_stats(
        SignalStatsInput(raw_file=str(raw), signal="V(out)"), state_no_sim
    )
    payload = result.structuredContent
    assert payload is not None
    adapter_schema = getattr(handle_signal_stats, "__output_schema__", None)
    assert adapter_schema is not None, "handle_signal_stats declares no contract"
    # Legal for the adapter's contract (this direct call already ran under the
    # live hook, which attributed it to the adapter's own frame).
    jsonschema.Draft202012Validator(adapter_schema).validate(payload)
    # Illegal for the delegating envelope — the discrimination that makes the
    # attribution test below meaningful.
    envelope = getattr(handle_analyze_results, "__output_schema__", None)
    assert envelope is not None
    assert list(jsonschema.Draft202012Validator(envelope).iter_errors(payload)), (
        "adapter payload unexpectedly satisfies the analyze_results envelope — "
        "this test can no longer distinguish the two contracts"
    )
    # The same adapter emission inside analyze_results' frame passes the hook:
    # the walk stops at the adapter's contract instead of falling through.
    data = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {"sources": [{"raw_path": str(raw), "label": "dut"}], "recipes": _tran_recipe()}
        ),
        state_no_sim,
    )
    assert data.structuredContent is not None
    assert "signal_stats" in data.structuredContent["results"]


async def test_malformed_final_analyze_results_payload_still_trips_the_hook(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stopping the walk at adapter frames must not blind the hook to the
    delegating tool's OWN envelope: corrupt the final assembled payload and
    the emission is rejected at analyze_results itself."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    original_assemble = analyze_mod._assemble

    def corrupting_assemble(*args: Any, **kwargs: Any):
        data, text = original_assemble(*args, **kwargs)
        data["results"] = "not-an-object"
        return data, text

    monkeypatch.setattr(analyze_mod, "_assemble", corrupting_assemble)
    with pytest.raises(AssertionError, match="output_schema"):
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {"sources": [{"raw_path": str(raw), "label": "dut"}], "recipes": _tran_recipe()}
            ),
            state_no_sim,
        )


_CONSOLIDATED_MODULES = ("analyze", "experiments", "inspect_tools", "schematic_edit", "verify")


def _delegate_targets() -> dict[str, Any]:
    """Every handle_* name a consolidated module's source references, resolved
    to the actual handler object. Derived from source, never hand-listed, so a
    new delegation cannot dodge the closure test."""
    providers = [
        importlib.import_module(name)
        for name in (
            "ltspice_mcp.tools.analysis",
            "ltspice_mcp.lib.schematic_ops",
        )
    ]
    targets: dict[str, Any] = {}
    for mod_name in _CONSOLIDATED_MODULES:
        mod = importlib.import_module(f"ltspice_mcp.tools.{mod_name}")
        source = Path(str(mod.__file__)).read_text(encoding="utf-8")
        for name in set(re.findall(r"\bhandle_[a-z0-9_]+\b", source)):
            target = getattr(mod, name, None)
            if target is None:
                for provider in providers:
                    target = getattr(provider, name, None)
                    if target is not None:
                        break
            assert target is not None, (
                f"{mod_name} references {name}, which resolves in no tool module"
            )
            targets[name] = target
    return targets


def test_every_delegated_handler_resolves_to_a_declared_contract():
    """A future delegation that forgets its @declare_output_schema fails the
    suite instead of quietly inheriting the delegating tool's envelope."""
    targets = _delegate_targets()
    assert targets, "derived no delegate names — the source scan has rotted"
    missing = sorted(
        name
        for name, fn in targets.items()
        if getattr(fn, "__output_schema__", None) is None and name not in NO_CONTRACT_DELEGATES
    )
    assert not missing, f"delegated handlers without a declared contract: {missing}"
    # Fail-closed pins: every exemption must still be real. (The hook's
    # walk-stop set derives from this same conftest table, so walk-stop ⊆
    # exemptions holds by construction.)
    for name in NO_CONTRACT_DELEGATES:
        assert name in targets, f"exemption {name!r} is stale — no longer delegated to"
        assert getattr(targets[name], "__output_schema__", None) is None, (
            f"exemption {name!r} is slack — the handler now declares a contract"
        )
