"""Integration tests for MCP resource handlers."""

from pathlib import Path

import pytest
from mcp.types import TextResourceContents

from ltspice_mcp.errors import ResultError
from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState


def _text(contents) -> str:
    """Extract text from a resource contents entry, asserting it is text."""
    assert isinstance(contents, TextResourceContents)
    return contents.text


class TestStaticResources:
    def test_returns_four_resources(self):
        resources = get_static_resources()
        assert len(resources) == 4
        names = {r.name for r in resources}
        assert names == {"netlists", "results", "models", "config"}

    def test_all_have_uri_and_mime(self):
        for r in get_static_resources():
            assert r.uri is not None
            assert r.mimeType is not None


class TestResourceTemplates:
    def test_returns_three_templates(self):
        templates = get_resource_templates()
        assert len(templates) == 3

    def test_template_uris(self):
        templates = get_resource_templates()
        uris = {t.uriTemplate for t in templates}
        assert "ltspice://netlists/{filename}" in uris
        assert "ltspice://results/{job_id}/signals" in uris
        assert "ltspice://results/{job_id}/measurements" in uris


class TestReadResource:
    def test_read_config(self, state_no_sim: SessionState):
        import json

        result = handle_read_resource("ltspice://config", state_no_sim)
        assert result.contents
        text = _text(result.contents[0])
        data = json.loads(text)
        assert "working_dir" in data
        assert "allowed_paths" in data
        assert str(state_no_sim.config.working_dir) in data["working_dir"]
        assert "detected_simulators" in data

    def test_read_netlists_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("ltspice://netlists/", state_no_sim)
        assert result.contents
        assert '"count": 0' in _text(result.contents[0])

    def test_read_netlists_with_files(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        result = handle_read_resource("ltspice://netlists/", state_no_sim)
        text = _text(result.contents[0])
        assert "rc_filter.cir" in text
        assert '"count": 1' in text

    def test_read_netlist_content(self, state_no_sim: SessionState, sample_netlist: Path):
        result = handle_read_resource(
            f"ltspice://netlists/{sample_netlist.name}", state_no_sim
        )
        text = _text(result.contents[0])
        assert "R1 in out 1k" in text

    def test_read_results_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("ltspice://results/", state_no_sim)
        assert '"count": 0' in _text(result.contents[0])

    def test_read_models_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("ltspice://models/", state_no_sim)
        assert "libraries" in _text(result.contents[0])

    def test_unknown_uri_raises(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Unknown resource URI"):
            handle_read_resource("ltspice://nonexistent", state_no_sim)

    def test_netlist_path_escape_blocked(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Unknown resource URI"):
            handle_read_resource("ltspice://netlists/../../etc/passwd", state_no_sim)

    def test_signals_no_job(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="Job not found"):
            handle_read_resource("ltspice://results/fake_job_id/signals", state_no_sim)

    def test_measurements_no_job(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="Job not found"):
            handle_read_resource("ltspice://results/fake_job_id/measurements", state_no_sim)
