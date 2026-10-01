"""Tests for the delivery of the SPICE and authoring guidance.

The guidance must reach the consuming LLM without relying on a client-side
skill being installed: an always-on floor (the server instructions) and the
single-sourced ``spice://guide`` resource. The checks here pin facts and
routes (tool names, constructs, sections), never the sentences around them.
"""

import re
import typing
from importlib.resources import files
from pathlib import Path

from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.variations import MismatchRule
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.server import CONSOLIDATED_INSTRUCTIONS
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import ENVELOPE_CHANNELS
from ltspice_mcp.tools.schematic_edit import EditSchematicInput
from tests.conftest import ENVELOPE_TOOLS

_GUIDE_ASSET = files("ltspice_mcp") / "assets" / "spice_guide.md"


def _flat(text: str) -> str:
    """Lower-cased, with line wrapping and repeated spaces collapsed."""
    return " ".join(text.split()).lower()


def _headings(text: str) -> list[str]:
    return [line.lstrip("#").strip().lower() for line in text.splitlines() if line.startswith("#")]


def _has_heading(text: str, *words: str) -> bool:
    """Some markdown heading mentions every one of ``words``."""
    return any(all(word.lower() in heading for word in words) for heading in _headings(text))


def _has_engine_comparison_table(text: str) -> bool:
    """A markdown table whose header row has an LTspice and an ngspice column."""
    return any(
        line.startswith("|") and "ltspice" in line.lower() and "ngspice" in line.lower()
        for line in text.splitlines()
    )


class TestServerInstructionsFloor:
    def test_names_the_planes_and_keeps_the_result_trust_tail(self):
        # Always-on floor: even with no client-side skill installed, the
        # handshake names every envelope tool and keeps the result-trust
        # warning — that a completed run can still hold a degenerate result,
        # read from the envelope's channels (the tail is what Claude Code's
        # 2048-char truncation would eat first, so its presence is the budget
        # test's partner).
        for tool in ENVELOPE_TOOLS:
            assert re.search(rf"\b{tool}\b", CONSOLIDATED_INSTRUCTIONS), tool
        text = _flat(CONSOLIDATED_INSTRUCTIONS)
        assert "completed" in text and "degenerate" in text
        for channel in ENVELOPE_CHANNELS:
            assert channel in text, f"the result-trust warning never names {channel}"


class TestGuideIsEngineGeneral:
    """The packaged guide is the union of both engines (the per-engine skills
    stay engine-specific). These are coverage checks, not a byte-mirror — the
    guide is hand-authored, so its per-engine sections duplicate the skills'
    and can drift; a section that went missing fails here, a renamed heading
    does not.
    """

    def test_covers_both_engines_and_their_differences(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        assert _has_heading(guide, "fundamentals")
        assert _has_heading(guide, "ltspice")
        assert _has_heading(guide, "ngspice")
        assert _has_engine_comparison_table(guide), "the differences table is gone"

    def test_includes_each_engines_distinctive_sections(self):
        guide = _GUIDE_ASSET.read_text("utf-8")
        for construct in (".asc", ".control", "xspice", ".save"):
            assert _has_heading(guide, construct), f"no guide section covers {construct}"


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
        lines = [line.lower() for line in guide.splitlines()]
        # Value notation: M is milli, MEG is mega.
        assert any(
            re.search(r"(?<![a-z])m(?![a-z])", line) and "milli" in line for line in lines
        ), "the served guide never says M means milli"
        assert any("meg" in line and "mega" in line for line in lines)
        # The server runs ngspice in -b -r batch mode, which affects .meas.
        assert any("ngspice" in line and ".meas" in line and "-b -r" in line for line in lines)
        for construct in (".control", ".asc"):
            assert _has_heading(guide, construct), f"no served section covers {construct}"
        assert _has_engine_comparison_table(guide)

    def test_it_maps_the_six_tools_and_how_to_start_a_sheet(self, work_dir: Path):
        guide = _served_guide(work_dir)
        for tool in ENVELOPE_TOOLS:
            assert re.search(rf"\b{tool}\b", guide), f"the guide never names {tool}"
        # A new schematic is an edit_schematic call on a blank base; the value
        # is read off the live model so the guide cannot teach a stale one.
        base = EditSchematicInput.model_fields["base"].annotation
        assert "blank" in typing.get_args(base)
        assert re.search(r"edit_schematic\([^)]*base\s*=\s*['\"]blank['\"]", guide)
