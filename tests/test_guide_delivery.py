"""The guide reaches the model through every door, and the doors agree.

The guide is a core every session reads plus topic sections and task playbooks
read on demand (``lib/guide.py``). These tests pin its structure (the section
list is the files present, the index is the front matter), its delivery (the
``inspect`` guide kind, ``Api.guide()``, the ``spice://guide`` resources serve
the same text), the instructions that send a model there, and the one reminder
a session gets when it skipped them.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import types

from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import guide
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.lib.response_budget import BUDGET_MIN_TOKENS
from ltspice_mcp.lib.variations import MismatchRule
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.server import (
    CONSOLIDATED_INSTRUCTIONS,
    GUIDE_REMINDER,
    build_instructions,
    call_tool,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from tests._text import names, section
from tests.conftest import call_tool_params, fake_request_context

ROOT = Path(__file__).resolve().parents[1]
GUIDE_DIR = ROOT / "src" / "ltspice_mcp" / "assets" / "guide"

#: The core is read every session, so its size is a per-session cost. About
#: 2.5k tokens with its generated index; growth has to be a deliberate edit to
#: this number, and a section is where depth goes.
CORE_BUDGET_CHARS = 10_000


def _all_sections() -> str:
    return "\n".join(guide.read(name) for name in guide.SECTION_ORDER)


def _state(work_dir: Path) -> SessionState:
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir])
    return SessionState.create(config, available={})


class TestServerInstructions:
    def test_send_the_model_to_the_guide_first(self):
        assert 'inspect(queries=[{"kind": "guide"}])' in CONSOLIDATED_INSTRUCTIONS
        assert "Read the guide first, every session" in CONSOLIDATED_INSTRUCTIONS

    def test_keep_the_rules_that_cost_a_wrong_answer(self):
        # The tail is what a client's 2048-character truncation would eat
        # first; test_server pins every prefix shape under the budget.
        text = CONSOLIDATED_INSTRUCTIONS
        assert "edit_schematic" in text
        assert "status completed and still hold a degenerate result" in text
        assert "a refused path names the config line that widens it" in text

    def test_state_both_doors_and_their_tradeoffs(self):
        served = build_instructions({}, None)
        library = build_instructions({}, None, served=())
        for text in (served, library):
            assert "from ltspice_mcp.api import Api" in text
            assert "detach=True" in text
            assert "Tools: one sandboxed step per call" in text
        assert "outside the sandbox" in served
        # run_code off: the library runs with the caller's own authority, so
        # the authority warning that belongs to run_code is not stated.
        assert "run_code" not in library and "outside the sandbox" not in library


class TestGuideStructure:
    def test_the_section_list_is_the_files_present(self):
        present = {path.stem for path in GUIDE_DIR.glob("*.md")} - {"core"}
        assert set(guide.SECTION_ORDER) == present
        assert len(guide.SECTION_ORDER) == len(set(guide.SECTION_ORDER))

    @pytest.mark.parametrize("name", guide.SECTION_ORDER)
    def test_each_section_names_itself_and_says_when_to_read_it(self, name: str):
        fields, body = guide.split_front_matter((GUIDE_DIR / f"{name}.md").read_text("utf-8"))
        assert fields.get("name") == name
        assert fields.get("description")
        assert body.startswith("# "), "a section opens with its title"

    @pytest.mark.parametrize("name", guide.SECTION_ORDER)
    def test_each_section_is_a_known_kind(self, name: str):
        fields, _ = guide.split_front_matter((GUIDE_DIR / f"{name}.md").read_text("utf-8"))
        assert fields.get("kind", "topic") in guide.KINDS

    def test_the_index_lists_every_section_under_its_kind(self):
        core = guide.read()
        index = core[core.index("## Index") :]
        for entry in guide.sections():
            heading = index.index(guide.KINDS[entry.kind])
            line = index.index(f"- `{entry.name}`:")
            later = [
                index.index(other)
                for kind, other in guide.KINDS.items()
                if kind != entry.kind and other in index and index.index(other) > heading
            ]
            assert heading < line < min(later, default=len(index)), entry.name

    def test_the_core_fits_its_budget(self):
        core = guide.read()
        assert len(core) <= CORE_BUDGET_CHARS, (
            f"the guide core grew to {len(core)} characters (limit {CORE_BUDGET_CHARS})"
        )

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(guide.UnknownGuideSection) as caught:
            guide.read("nope")
        assert caught.value.known == guide.names()

    def test_a_path_names_no_section(self):
        """Sections are names, never paths: nothing outside the guide's own
        files can be read through it."""
        with pytest.raises(guide.UnknownGuideSection):
            guide.read("../guide/core")


#: Where a pointer to a section can appear: the code that writes descriptions
#: and hints, the guide itself, and the plugin's skill.
_POINTER_SOURCES = (
    *sorted((ROOT / "src" / "ltspice_mcp").rglob("*.py")),
    *sorted(GUIDE_DIR.glob("*.md")),
    *sorted((ROOT / "skills").rglob("*.md")),
)


#: The ways a text names a section: the prose pointer, and the two calls that
#: read one. A placeholder such as "<name>" is not a name.
_POINTER_PATTERNS = (
    r"guide section\s+'([^'<]+)'",
    r'"section":\s*"([^"<]+)"',
    r"""guide\(["']([^"'<]+)["']\)""",
)


def test_every_section_pointer_resolves():
    """A pointer is how descriptions, hints, the sections themselves and the
    plugin's skill send a reader on; one naming no section is a dead end the
    reader cannot recover from."""
    names = set(guide.names())
    dangling = []
    for path in _POINTER_SOURCES:
        text = " ".join(path.read_text(encoding="utf-8").split())
        for pattern in _POINTER_PATTERNS:
            for name in re.findall(pattern, text):
                if name not in names:
                    dangling.append(f"{path.relative_to(ROOT)}: {name!r}")
    assert not dangling, "pointers to sections the guide does not have:\n" + "\n".join(dangling)


class TestSectionContent:
    """Coverage anchors, not a byte-mirror: each flags a section that went
    missing in a reorganisation."""

    @pytest.mark.parametrize(
        ("section", "anchor"),
        [
            ("fundamentals", "## Value suffixes"),
            ("fundamentals", "On ngspice a top-level `.meas` is skipped"),
            ("ltspice", "## Other LTspice Quirks"),
            ("ngspice", "## .control / .endc Blocks"),
            ("ngspice", "## XSPICE"),
            ("ngspice", "## .save Directive"),
            ("ngspice", "LTspice vs ngspice"),
            ("schematics", "## Building and editing a sheet"),
            ("schematics", '`edit_schematic(target=..., base="blank")` starts a new sheet'),
            ("python", "detach=True"),
            # The paragraph opening this subsection was once split off into
            # another section, leaving it to start mid-sentence.
            ("tools", "Three recipes appear in the tool schema by name only"),
            ("tools", "## A sweep in one call"),
        ],
    )
    def test_anchor(self, section: str, anchor: str):
        assert anchor in " ".join(guide.read(section).split())

    def test_teaches_both_measurement_idioms(self):
        # Scalars come from a .meas in the deck, read back by the measurements
        # recipe; device small-signal parameters come from a .op run with
        # .options logopinfo, read back by operating_point. The recipes are
        # checked against the live union, so a rename fails here.
        tools, op = guide.read("tools"), guide.read("operating-points")
        assert ".meas" in tools.lower()
        assert "logopinfo" in op.lower()
        for text, recipe in ((tools, "measurements"), (op, "operating_point")):
            assert recipe in DISCRIMINANTS, f"{recipe} is no longer a recipe"
            assert names(text, recipe), f"the guide never names {recipe} where it teaches it"

    def test_teaches_the_response_budget(self):
        # The budget section names every tool that takes one, its unit and its
        # floor, each read off the code, so it can neither be gutted nor fall
        # behind the surface.
        owners = {
            name
            for name, registered in get_tools()[1].items()
            if "budget" in registered.definition.input_schema.get("properties", {})
        }
        assert owners, "no tool takes a budget any more"
        budget = section(guide.read("tools"), "budget")
        missing = sorted(tool for tool in owners if not names(budget, tool))
        assert not missing, f"the budget section never names {missing}"
        assert "token" in budget.lower()
        assert names(budget, str(BUDGET_MIN_TOKENS)), "the budget floor is not stated"

    def test_the_bench_playbook_closes_a_dc_servo(self):
        # The DC-only feedback inductor that closes the loop at DC.
        text = guide.read("bench-craft")
        assert re.search(r"^\s*LFB\s+out\s+inn\s+1T\b", text, re.IGNORECASE | re.MULTILINE)

    def test_the_bench_playbook_ships_a_template_per_analysis(self):
        # An operating-point, an open-loop AC and a closed-loop transient bench,
        # each a complete deck an agent can render.
        decks = re.findall(r"```spice\n(.*?)```", guide.read("bench-craft"), re.S)
        for analysis in (".op", ".ac", ".tran"):
            assert any(
                re.search(rf"^{re.escape(analysis)}\b", deck, re.IGNORECASE | re.MULTILINE)
                and re.search(r"^\.end\s*$", deck, re.IGNORECASE | re.MULTILINE)
                for deck in decks
            ), f"no complete {analysis} bench template"

    def test_the_bench_playbook_teaches_ngspice_batch_output(self):
        text = guide.read("bench-craft")
        # Under -b -r a measurement goes in .control as the dot-less command.
        controls = re.findall(r"^\.control\b(.*?)^\.endc\b", text, re.S | re.M | re.I)
        assert any(re.search(r"^meas\s", block, re.M | re.I) for block in controls)
        # wrdata repeats the scale column before every dumped vector.
        assert re.search(r"scale\s*,\s*v\(out\)\s*,\s*scale\s*,\s*i\(vdd\)", text, re.I)

    def test_the_bench_playbook_is_a_task(self):
        (entry,) = [entry for entry in guide.sections() if entry.name == "bench-craft"]
        assert entry.kind == "task"

    def test_names_no_absent_or_withheld_surface(self):
        """Two rules share this denylist. Absent behavior: "rerun", "case_axis"
        and "columnar" name things this surface does not have, and a guide that
        names them teaches calls that do not exist. The analysis_budget_s
        deferral knob is not taught; recovery authority is documented."""
        text = "\n".join([guide.read(), _all_sections()])
        for term in ("rerun", "columnar", "case_axis", "analysis_budget_s"):
            pattern = re.compile(
                r"\b" + r"[\s_-]?".join(re.escape(p) for p in term.split("_")) + r"\b",
                re.IGNORECASE,
            )
            hit = pattern.search(text)
            assert hit is None, f"the guide names {term!r} as {hit.group(0)!r}"

    def test_the_tools_section_maps_the_six_tools(self):
        text = guide.read("tools")
        for tool in (
            "run_experiments",
            "jobs",
            "analyze_results",
            "inspect",
            "edit_schematic",
            "verify_circuit",
        ):
            assert tool in text, f"the tools section never names {tool}"

    def test_the_core_states_the_micro_sign_rule_by_version(self):
        core = " ".join(guide.read().split())
        assert "LTspice 24 writes µ in UTF-8 and reads it back" in core
        assert "LTspice XVII misreads that µ and drops the scale" in core


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
        text = _all_sections()
        values = [float(match) for match in re.findall(r'"AVT":\s*([0-9.eE+-]+)', text)]
        assert values, "the guide no longer ships a worked AVT exemplar"
        low, high = self._PLAUSIBLE_V_UM
        for value in values:
            assert low <= value <= high, (
                f"guide AVT exemplar {value:g} is outside the V·µm band "
                f"[{low:g}, {high:g}] — montecarlo.py divides by √(W·L) in µm²"
            )

    def test_guide_and_engine_name_the_same_unit(self):
        engine_description = MismatchRule.model_fields["AVT"].description or ""
        assert "V·µm" in engine_description
        assert "V·µm" in _all_sections(), "the guide states the exemplar's unit nowhere"


def _resource_text(uri: str, state: SessionState) -> str:
    contents = handle_read_resource(uri, state).contents[0]
    assert isinstance(contents, types.TextResourceContents)
    return contents.text


async def _inspect_guide(state: SessionState, section: str | None = None) -> dict:
    query: dict = {"kind": "guide"}
    if section is not None:
        query["section"] = section
    result = await call_tool(
        fake_request_context(state), call_tool_params("inspect", {"queries": [query]})
    )
    assert not result.is_error
    assert result.structured_content is not None
    return result.structured_content


class TestTheDoorsAgree:
    @pytest.mark.parametrize("section", [None, "ltspice", "bench-craft"])
    async def test_inspect_api_and_resource_serve_one_text(
        self, work_dir: Path, section: str | None
    ):
        state = _state(work_dir)
        data = (await _inspect_guide(state, section))["results"][0]["data"]
        uri = "spice://guide" if section is None else f"spice://guide/{section}"
        assert data["text"] == Api.guide(section) == _resource_text(uri, state)
        assert data["section"] == section
        assert data["title"] == guide.title_of(section)

    async def test_an_unknown_section_fails_only_its_item(self, work_dir: Path):
        state = _state(work_dir)
        result = await call_tool(
            fake_request_context(state),
            call_tool_params(
                "inspect",
                {"queries": [{"kind": "guide", "section": "nope"}, {"kind": "guide"}]},
            ),
        )
        assert result.structured_content is not None
        bad, good = result.structured_content["results"]
        assert bad["ok"] is False and bad["error"]["code"] == "unknown_section"
        assert "bench-craft" in bad["error"]["supported"]
        assert good["ok"] is True

    def test_the_guide_reads_without_starting_the_engine(self, tmp_path: Path):
        """``python -m ltspice_mcp.api guide`` is the shell door; like the
        catalogue it must not pay for the engine, which a cold process proves."""
        probe = (
            "import runpy, sys\n"
            "sys.argv = ['ltspice_mcp.api', 'guide', 'python']\n"
            "try:\n"
            "    runpy.run_module('ltspice_mcp.api', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "heavy = sorted({m.split('.')[0] for m in sys.modules} & {'scipy', 'mcp', 'pydantic'})\n"
            "print('HEAVY', heavy)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            # A Windows pipe would otherwise encode with the ANSI code page.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=120,
            cwd=tmp_path,
        )
        assert "# Working in Python" in proc.stdout, proc.stdout + proc.stderr
        assert "HEAVY []" in proc.stdout, proc.stdout + proc.stderr

    def test_a_code_page_pipe_still_gets_the_whole_text(self, tmp_path: Path):
        """A Windows pipe encodes with the ANSI code page, which has no Γ: the
        shell door falls back to UTF-8 rather than failing mid-print."""
        proc = subprocess.run(
            [sys.executable, "-m", "ltspice_mcp.api", "guide", "rf"],
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
            timeout=120,
            cwd=tmp_path,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert "Γ" in proc.stdout.decode("utf-8")


class TestTheReadTheGuideReminder:
    async def test_the_first_reply_of_a_session_that_skipped_the_guide_carries_it(
        self, work_dir: Path
    ):
        state = _state(work_dir)
        args = {"queries": [{"kind": "capabilities", "fields": ["tool_profile"]}]}
        first = await call_tool(fake_request_context(state), call_tool_params("inspect", args))
        second = await call_tool(fake_request_context(state), call_tool_params("inspect", args))
        # Both channels: a structured-aware client reads only the hint.
        assert first.structured_content is not None
        assert GUIDE_REMINDER in first.structured_content["hint"]
        assert any(
            isinstance(block, types.TextContent) and block.text == GUIDE_REMINDER
            for block in first.content
        )
        # Once a session: the second reply is the tool's own.
        assert second.structured_content is not None
        assert GUIDE_REMINDER not in second.structured_content.get("hint", "")
        assert all(GUIDE_REMINDER not in getattr(block, "text", "") for block in second.content)

    async def test_a_session_that_read_the_guide_is_not_reminded(self, work_dir: Path):
        state = _state(work_dir)
        reply = await _inspect_guide(state)
        assert GUIDE_REMINDER not in reply.get("hint", "")
        after = await call_tool(
            fake_request_context(state),
            call_tool_params("inspect", {"queries": [{"kind": "capabilities"}]}),
        )
        assert after.structured_content is not None
        assert GUIDE_REMINDER not in after.structured_content.get("hint", "")

    async def test_reading_the_guide_resource_counts(self, work_dir: Path):
        state = _state(work_dir)
        _resource_text("spice://guide/ngspice", state)
        after = await call_tool(
            fake_request_context(state),
            call_tool_params("inspect", {"queries": [{"kind": "capabilities"}]}),
        )
        assert after.structured_content is not None
        assert GUIDE_REMINDER not in after.structured_content.get("hint", "")

    async def test_an_error_reply_carries_it_too(self, work_dir: Path):
        """A session whose first call fails still learns where the guide is:
        a failed first call is exactly when it is needed."""
        state = _state(work_dir)
        result = await call_tool(
            fake_request_context(state), call_tool_params("run_experiments", {"missing": 1})
        )
        assert result.is_error
        assert any(GUIDE_REMINDER in getattr(block, "text", "") for block in result.content)
