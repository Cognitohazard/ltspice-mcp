"""Facts the server states in more than one place are declared once.

The dwell defaults and caps the capabilities report discloses are read off the
fields and constants that enforce them; the ``error`` object of every
enveloped tool is one schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import experiments, jobs
from ltspice_mcp.tools._base import ENVELOPE_ERROR_SCHEMA, registry
from tests.conftest import capabilities_report


class TestDwellDisclosure:
    async def test_the_report_states_what_the_tools_enforce(
        self, state_no_sim: SessionState
    ) -> None:
        dwell = (await capabilities_report(state_no_sim))["limits"]["dwell"]
        assert dwell == {
            "run_experiments_wait_default_s": (
                experiments.ExperimentExecution.model_fields["wait_s"].default
            ),
            "run_experiments_wait_max_s": experiments.SUBMISSION_DWELL_CAP_S,
            "jobs_wait_default_s": jobs.JobsWaitInput.model_fields["timeout_s"].default,
            "jobs_wait_max_s": experiments.JOBS_WAIT_CAP_S,
        }

    async def test_a_changed_cap_is_what_the_report_says(
        self, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(experiments, "SUBMISSION_DWELL_CAP_S", 90.0)
        monkeypatch.setattr(experiments, "JOBS_WAIT_CAP_S", 240.0)
        dwell = (await capabilities_report(state_no_sim))["limits"]["dwell"]
        assert dwell["run_experiments_wait_max_s"] == 90.0
        assert dwell["jobs_wait_max_s"] == 240.0


def _error_objects(schema: dict[str, Any]) -> Iterator[Any]:
    """The envelope's own ``error``: at the top, or atop each action's branch."""
    for node in (schema, *schema.get("oneOf", []), *schema.get("anyOf", [])):
        error = node.get("properties", {}).get("error")
        if error is not None:
            yield error


@pytest.mark.parametrize("tool", ["run_experiments", "jobs", "edit_schematic"])
def test_each_envelope_declares_the_one_error_object(tool: str) -> None:
    import ltspice_mcp.tools  # noqa: F401  (registers every tool)

    (registered,) = (
        r
        for r in registry._registered  # pyright: ignore[reportPrivateUsage]
        if r.definition.name == tool
    )
    schema = registered.definition.output_schema
    assert schema is not None
    found = list(_error_objects(schema))
    assert found
    assert all(error == ENVELOPE_ERROR_SCHEMA for error in found), tool
