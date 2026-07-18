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


class TestNetlistResourceHardening:
    def test_space_in_filename_round_trips_through_listing(self, state_no_sim: SessionState):
        # Listed URIs are percent-encoded (clients normalize via AnyUrl, which
        # would encode a raw space anyway) and dispatch percent-decodes, so
        # the exact URI the listing hands out must be readable.
        f = state_no_sim.working_dir / "rc filter.cir"
        f.write_text("* spaced\nR1 in out 1k\n.end\n")
        listing = _text(handle_read_resource("spice://netlists/", state_no_sim).contents[0])
        assert "spice://netlists/rc%20filter.cir" in listing
        result = handle_read_resource("spice://netlists/rc%20filter.cir", state_no_sim)
        assert "R1 in out 1k" in _text(result.contents[0])

    def test_utf16_netlist_decodes_cleanly(self, state_no_sim: SessionState):
        # LTspice writes UTF-16 LE artifacts; the resource must use the same
        # BOM-sniffing decode as the tool channel, not a hard-coded utf-8 read
        # that renders every character interleaved with NULs.
        f = state_no_sim.working_dir / "utf16.cir"
        f.write_bytes("* µ-title\nR1 in out 1k\n.end\n".encode("utf-16-le"))
        text = _text(handle_read_resource("spice://netlists/utf16.cir", state_no_sim).contents[0])
        assert "R1 in out 1k" in text
        assert "\x00" not in text

    def test_non_netlist_extension_rejected(self, state_no_sim: SessionState):
        f = state_no_sim.working_dir / "big.raw"
        f.write_bytes(b"\x00\x01binary")
        with pytest.raises(ValueError, match="Not a netlist file"):
            handle_read_resource("spice://netlists/big.raw", state_no_sim)

    def test_oversize_netlist_rejected(self, state_no_sim: SessionState, monkeypatch):
        import ltspice_mcp.resources as resources_mod

        f = state_no_sim.working_dir / "huge.cir"
        f.write_text("* padding\n" + "x" * 256)
        monkeypatch.setattr(resources_mod, "_NETLIST_RESOURCE_CAP_BYTES", 64)
        with pytest.raises(ValueError, match="too large"):
            handle_read_resource("spice://netlists/huge.cir", state_no_sim)
