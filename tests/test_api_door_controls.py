"""The Python API's automatic mode knows a control by the operation's model.

A wire-only control is a field the operation declares: a presentation cap, the
response dwell, or a paging cursor. A key the caller supplies as data, such as
the name of a parameter an ``assign`` variation sweeps, is never one, whatever
it is called.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import BaseModel

from ltspice_mcp.api import ApiCallError, ApiValidationError
from ltspice_mcp.api import _methods as methods_module
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze, experiments, inspect_tools, jobs, schematic_edit, verify
from tests.conftest import SyncApi

#: Every control path each operation's model declares, and its class. A field
#: added with a control's name shows up here as a difference to classify.
_EXPECTED: dict[type[BaseModel], dict[tuple[str, ...], str]] = {
    experiments.RunExperimentsInput: {
        ("budget",): "budget",
        ("execution", "wait_s"): "dwell",
        ("analyze", "include", "per_run", "cursor"): "paging",
    },
    jobs.JobsInput: {("budget",): "budget", ("cursor",): "paging"},
    jobs.JobsResumeInput: {("budget",): "budget"},
    analyze.AnalyzeResultsInput: {
        ("budget",): "budget",
        ("continuation",): "paging",
        ("continue",): "paging",
        ("continuation", "cursor"): "paging",
        ("continue", "cursor"): "paging",
        ("include", "per_run", "cursor"): "paging",
    },
    inspect_tools.InspectInput: {("budget",): "budget", ("queries", "cursor"): "paging"},
    schematic_edit.EditSchematicInput: {("view_cursors",): "paging"},
    verify.VerifyCircuitInput: {("findings_cursor",): "paging"},
}


@pytest.mark.parametrize("model", list(_EXPECTED), ids=lambda m: m.__name__)
def test_each_model_declares_exactly_its_control_paths(model: type[BaseModel]) -> None:
    assert methods_module._control_paths(model) == _EXPECTED[model]


_DATA_KEYS = ("budget", "bias_cursor", "cursor", "x_cursors", "continue", "view_cursors")


def _sweep(*names: str) -> dict[str, Any]:
    return {
        "circuits": [{"path": "rc.cir"}],
        "variations": [{"kind": "assign", "assign": {name: ["1", "2"] for name in names}}],
    }


class TestSweepParametersAreData:
    def test_parameters_named_like_controls_pass_unchanged_and_unremarked(self) -> None:
        arguments = _sweep(*_DATA_KEYS)
        before = copy.deepcopy(arguments)
        warnings = methods_module._enforce_auto_door(arguments, experiments.RunExperimentsInput)
        assert warnings == []
        assert arguments == before

    def test_the_real_controls_beside_them_are_still_applied(self) -> None:
        arguments = {**_sweep("budget"), "budget": 4000}
        warnings = methods_module._enforce_auto_door(arguments, experiments.RunExperimentsInput)
        assert [w.split(":")[0] for w in warnings] == ["budget"]
        assert "budget" not in arguments
        assert arguments["variations"][0]["assign"] == {"budget": ["1", "2"]}

    def test_a_paging_control_beside_them_is_still_refused(self) -> None:
        arguments = {**_sweep("bias_cursor"), "analyze": {"recipes": [], "include": {}}}
        arguments["analyze"]["include"]["per_run"] = {"cursor": "abc"}
        with pytest.raises(ApiValidationError, match=r"analyze\.include\.per_run\.cursor"):
            methods_module._enforce_auto_door(arguments, experiments.RunExperimentsInput)

    @pytest.mark.parametrize("name", ["budget", "bias_cursor"])
    def test_the_api_runs_such_a_sweep_past_the_door(
        self, state_no_sim: SessionState, name: str
    ) -> None:
        """Through the real method: what stops the call is the missing
        simulator, after the door, not the automatic mode's refusal."""
        (state_no_sim.working_dir / "rc.cir").write_text(
            ".param budget=1 bias_cursor=1\nR1 a 0 1k\n.op\n.end\n", encoding="utf-8"
        )
        api = SyncApi(state_no_sim)
        with pytest.raises(ApiCallError, match="No SPICE simulator") as caught:
            api.run_experiments(**_sweep(name))
        assert "ignored" not in str(caught.value)
