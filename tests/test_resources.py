"""Integration tests for MCP resource handlers."""

from pathlib import Path

import pytest
from mcp.types import TextResourceContents

from ltspice_mcp.resources import (
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState


def _text(contents) -> str:
    """Extract text from a resource contents entry, asserting it is text."""
    assert isinstance(contents, TextResourceContents)
    return contents.text


class TestStaticResources:
    def test_returns_static_resources(self):
        resources = get_static_resources()
        assert len(resources) == 7
        names = {r.name for r in resources}
        assert names == {
            "netlists",
            "results",
            "models",
            "config",
            "recent",
            "plot_widget",
            "guide",
        }

    def test_resource_uris(self):
        resources = get_static_resources()
        uris = {str(r.uri) for r in resources}
        assert "spice://netlists/" in uris
        assert "spice://results/" in uris


class TestReadResource:
    def test_read_config(self, state_no_sim: SessionState):
        import json

        result = handle_read_resource("spice://config", state_no_sim)
        assert result.contents
        text = _text(result.contents[0])
        data = json.loads(text)
        assert "working_dir" in data
        assert "allowed_paths" in data
        assert str(state_no_sim.config.working_dir) in data["working_dir"]
        assert "detected_simulators" in data

    def test_read_netlists_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("spice://netlists/", state_no_sim)
        assert result.contents
        assert '"count": 0' in _text(result.contents[0])

    def test_read_netlists_with_files(self, state_no_sim: SessionState, sample_netlist: Path):
        result = handle_read_resource("spice://netlists/", state_no_sim)
        text = _text(result.contents[0])
        assert "rc_filter.cir" in text
        assert '"count": 1' in text

    def test_read_netlist_content(self, state_no_sim: SessionState, sample_netlist: Path):
        result = handle_read_resource(f"spice://netlists/{sample_netlist.name}", state_no_sim)
        text = _text(result.contents[0])
        assert "R1 in out 1k" in text

    def test_read_results_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("spice://results/", state_no_sim)
        assert '"count": 0' in _text(result.contents[0])

    def test_read_models_empty(self, state_no_sim: SessionState):
        result = handle_read_resource("spice://models/", state_no_sim)
        assert "libraries" in _text(result.contents[0])

    def test_read_guide(self, state_no_sim: SessionState):
        result = handle_read_resource("spice://guide", state_no_sim)
        assert result.contents
        contents = result.contents[0]
        assert isinstance(contents, TextResourceContents)
        assert contents.mimeType == "text/markdown"
        text = contents.text
        # Both strings come from the real SKILL.md body, proving the packaged
        # guide is served (not a placeholder).
        assert "Schematic layout best practices" in text
        assert "means MILLI" in text

    def test_unknown_uri_raises(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Unknown resource URI"):
            handle_read_resource("spice://nonexistent", state_no_sim)

    def test_netlist_path_escape_blocked(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Unknown resource URI"):
            handle_read_resource("spice://netlists/../../etc/passwd", state_no_sim)
