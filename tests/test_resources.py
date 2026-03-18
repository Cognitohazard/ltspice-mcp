"""Integration tests for MCP resource handlers."""

from pathlib import Path

import pytest
import pytest_asyncio

from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState


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


@pytest.mark.asyncio
class TestReadResource:

    async def test_read_config(self, state_no_sim: SessionState):
        import json

        result = await handle_read_resource("ltspice://config", state_no_sim)
        assert result.contents
        text = result.contents[0].text
        data = json.loads(text)
        assert "working_dir" in data
        assert "allowed_paths" in data
        assert str(state_no_sim.config.working_dir) in data["working_dir"]
        assert "detected_simulators" in data

    async def test_read_netlists_empty(self, state_no_sim: SessionState):
        result = await handle_read_resource("ltspice://netlists/", state_no_sim)
        assert result.contents
        assert '"count": 0' in result.contents[0].text

    async def test_read_netlists_with_files(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        result = await handle_read_resource("ltspice://netlists/", state_no_sim)
        text = result.contents[0].text
        assert "rc_filter.cir" in text
        assert '"count": 1' in text

    async def test_read_netlist_content(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        result = await handle_read_resource(
            f"ltspice://netlists/{sample_netlist.name}", state_no_sim
        )
        text = result.contents[0].text
        assert "R1 in out 1k" in text

    async def test_read_results_empty(self, state_no_sim: SessionState):
        result = await handle_read_resource("ltspice://results/", state_no_sim)
        assert '"count": 0' in result.contents[0].text

    async def test_read_models_empty(self, state_no_sim: SessionState):
        result = await handle_read_resource("ltspice://models/", state_no_sim)
        assert "libraries" in result.contents[0].text

    async def test_unknown_uri_raises(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Unknown resource URI"):
            await handle_read_resource("ltspice://nonexistent", state_no_sim)

    async def test_netlist_path_escape_blocked(self, state_no_sim: SessionState):
        with pytest.raises(ValueError):
            await handle_read_resource(
                "ltspice://netlists/../../etc/passwd", state_no_sim
            )

    async def test_signals_no_job(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Job not found"):
            await handle_read_resource(
                "ltspice://results/fake_job_id/signals", state_no_sim
            )

    async def test_measurements_no_job(self, state_no_sim: SessionState):
        with pytest.raises(ValueError, match="Job not found"):
            await handle_read_resource(
                "ltspice://results/fake_job_id/measurements", state_no_sim
            )
