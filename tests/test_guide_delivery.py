"""Tests for the 3-channel schematic/authoring guidance delivery.

The guidance must reach the consuming LLM without relying on a client-side
skill being installed: an always-on floor (server instructions + tool
descriptions), a just-in-time checklist (create_schematic result), and the
single-sourced ``spice://guide`` resource.
"""

import re
from importlib.resources import files
from pathlib import Path

from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.variations import MismatchRule
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.server import CONSOLIDATED_INSTRUCTIONS
from ltspice_mcp.state import SessionState

_GUIDE_ASSET = files("ltspice_mcp") / "assets" / "spice_guide.md"


class TestServerInstructionsFloor:
    def test_names_the_planes_and_keeps_the_result_trust_tail(self):
        # Always-on floor: even with no client-side skill installed, the
        # handshake teaches the three planes and ends on the result-trust
        # guidance (the tail is what Claude Code's 2048-char truncation
        # would eat first, so its presence is the budget test's partner).
        for tool in (
            "run_experiments",
            "jobs",
            "analyze_results",
            "edit_schematic",
            "verify_circuit",
            "inspect",
        ):
            assert tool in CONSOLIDATED_INSTRUCTIONS
        assert "status completed and still hold a degenerate result" in CONSOLIDATED_INSTRUCTIONS


class TestGuideIsEngineGeneral:
    """The packaged guide is the union of both engines (the per-engine skills
    stay engine-specific). These are coverage checks, not a byte-mirror — the
    guide is hand-authored, so its per-engine sections duplicate the skills'
    and can drift; the anchors below flag a section that went missing.
    """

    def test_covers_both_engines_and_fundamentals(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        assert "# SPICE Circuit Simulation Guide" in guide
        assert "## SPICE Fundamentals" in guide
        assert "## LTspice-Specific" in guide
        assert "## ngspice-Specific" in guide
        assert "LTspice vs ngspice" in guide  # the differences table

    def test_includes_each_engines_distinctive_sections(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        ltspice_anchors = ("### .asc Schematics", "### Other LTspice Quirks")
        ngspice_anchors = ("### .control / .endc Blocks", "### XSPICE", "### .save Directive")
        for anchor in ltspice_anchors + ngspice_anchors:
            assert anchor in guide, f"guide is missing section: {anchor}"


class TestMismatchExemplarMatchesTheEngineUnit:
    """The guide's worked AVT number and the engine that reads it are one class.

    ``montecarlo.py`` converts W/L to µm before dividing, so AVT is V·µm. The
    same coefficient written in V·m is 1e6 too small, and nothing errors: the
    draw is negligible, every run is the nominal deck, and the receipt says
    complete. A wrong exponent here is unfalsifiable from the result, so it is
    pinned against the engine's own field documentation instead.
    """

    # Real technology coefficients are single-digit to tens of mV·µm; a V·m
    # value lands at 1e-9 and a naive "5 mV" at 5e-3 is still inside the band.
    _PLAUSIBLE_V_UM = (1e-4, 1e-1)

    def test_exemplar_value_is_in_the_engines_unit(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        values = [float(match) for match in re.findall(r'"AVT":\s*([0-9.eE+-]+)', guide)]
        assert values, "the guide no longer ships a worked AVT exemplar"
        low, high = self._PLAUSIBLE_V_UM
        for value in values:
            assert low <= value <= high, (
                f"guide AVT exemplar {value:g} is outside the V·µm band "
                f"[{low:g}, {high:g}] — montecarlo.py divides by √(W·L) in µm²"
            )

    def test_guide_and_engine_name_the_same_unit(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        engine_description = MismatchRule.model_fields["AVT"].description or ""
        assert "V·µm" in engine_description
        assert "V·µm" in guide, "the guide states the exemplar's unit nowhere"


def _served_guide(work_dir: Path) -> str:
    """The guide as a client receives it, through the resource route."""
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir])
    state = SessionState.create(config, available={})
    contents = handle_read_resource("spice://guide", state).contents[0]
    assert isinstance(contents, types.TextResourceContents)
    return contents.text


class TestTheServedGuide:
    """One document: the SPICE facts and the passages naming this server's
    tools reach every client, and no client may be told to call a tool it
    cannot see."""

    def test_simulator_facts_are_served(self, work_dir: Path):
        guide = _served_guide(work_dir)
        for anchor in (
            "### Value Notation — CRITICAL",
            "ngspice skips `.meas` under the server's",
            "### .control / .endc Blocks",
            "### .asc Schematics",
            "LTspice vs ngspice",
        ):
            assert anchor in guide, f"the served guide is missing shared content: {anchor}"

    def test_it_maps_the_six_tools_and_replaces_the_asc_entry(self, work_dir: Path):
        guide = _served_guide(work_dir)
        assert "## Tool surface on this profile" in guide
        for tool in (
            "run_experiments",
            "jobs",
            "analyze_results",
            "inspect",
            "edit_schematic",
            "verify_circuit",
        ):
            assert tool in guide, f"the guide never names {tool}"
        assert "use the server's schematic tools (`create_schematic`" not in guide
        assert '`edit_schematic(target=..., base="blank")` starts a new sheet' in guide
