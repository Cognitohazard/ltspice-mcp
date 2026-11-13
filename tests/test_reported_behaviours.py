"""Behaviours users reported as working well, each held by a test.

Each test here guards one behaviour that had no test of its own, and goes
through the tool surface (``server.call_tool`` by ``dispatched``), so it holds
whatever code answers underneath. The table in ``docs/TESTING.md``, "Behaviours
users rely on", lists every such behaviour and the test that guards it,
including those guarded elsewhere.

LTspice cannot run here, so where a behaviour needs LTspice's export of a
sheet, the session's LTspice is a stand-in at the simulator boundary (the
class whose ``create_netlist`` the server calls): either one that writes the
cards of the sheet it is handed (``tests/_fake_netlister.py``) or one that
answers a recorded sheet with the export LTspice wrote for it.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._fake_netlister import netlist_text
from tests.conftest import FIXTURES_DIR, dispatch, dispatched, tool_text

pytestmark = pytest.mark.asyncio


def ltspice_exports(state: SessionState, export_of: Callable[[Path], bytes]) -> None:
    """Make the session's LTspice export a sheet as ``export_of`` says.

    The export lands in ``<name>.net`` beside the sheet, where LTspice's
    ``-netlist`` writes it; reading and comparing it is the server's own code.
    """

    class _Exporter:
        @staticmethod
        def create_netlist(path: str, timeout: float | None = None) -> str:
            sheet = Path(path)
            exported = sheet.with_suffix(".net")
            exported.write_bytes(export_of(sheet))
            return str(exported)

    state.available_simulators["ltspice"] = _Exporter


def exports_its_cards(sheet: Path) -> bytes:
    return netlist_text(sheet).encode("utf-8")


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_of(sheet: Path) -> str:
    return sheet.read_text(encoding="utf-8")


#: A diode clamp: a .model written over two lines, an analysis, and a draft
#: comment left on the sheet.
_CLAMP = (
    "Version 4\nSHEET 1 880 680\n"
    "SYMBOL voltage 96 80 R0\nSYMATTR InstName V1\nSYMATTR Value 5\n"
    "SYMBOL res 256 96 R0\nSYMATTR InstName R1\nSYMATTR Value 10k\n"
    "SYMBOL diode 400 96 R0\nSYMATTR InstName D1\nSYMATTR Value DX\n"
    "TEXT 40 400 Left 2 !.model DX D(Is=1e-14\\n+ N=1.8)\n"
    "TEXT 40 440 Left 2 !.tran 1m\n"
    "TEXT 40 480 Left 2 ;draft: check the clamp\n"
)


def clamp_reference(model: str, analysis: str) -> str:
    """The clamp's deck as an export writes it, with ``model`` and ``analysis``."""
    return (
        "* clamp\nV1 V1_1 V1_2 5\nR1 R1_1 R1_2 10k\nD1 D1_1 D1_2 DX\n"
        f"{model}\n{analysis}\n.backanno\n.end\n"
    )


@pytest.fixture
def clamp(asc_state: SessionState, work_dir: Path) -> Path:
    sheet = work_dir / "clamp.asc"
    sheet.write_text(_CLAMP, encoding="utf-8", newline="\n")
    ltspice_exports(asc_state, exports_its_cards)
    return sheet


# ---------------------------------------------------------------------------
# One call does the whole job
# ---------------------------------------------------------------------------


async def test_one_batch_removes_a_comment_rewrites_a_directive_and_checks_equivalence(
    asc_state: SessionState, clamp: Path
):
    reference = clamp_reference(".model DX D(Is=1e-14\n+ N=1.8)", ".tran 2m")

    edited = await dispatched(
        asc_state,
        "edit_schematic",
        path=clamp.name,
        expected_sha256=sha_of(clamp),
        ops=[
            {"op": "remove_directive", "instruction": "draft: check the clamp"},
            {"op": "replace_directive", "instruction": ".tran 1m", "text": ".tran 2m"},
        ],
        compare={"reference": reference},
    )

    assert edited["commit_state"] == "committed", edited
    removed, replaced = edited["results"]
    assert removed["removed_texts"] == [
        {"kind": "comment", "text": "draft: check the clamp", "x": 40, "y": 480}
    ]
    assert (replaced["text"]["was"], replaced["text"]["now"]) == (".tran 1m", ".tran 2m")
    assert edited["verification"]["equivalent"] is True, edited
    written = text_of(clamp)
    assert "draft: check the clamp" not in written
    assert "TEXT 40 440 Left 2 !.tran 2m" in written


async def test_an_edit_that_changes_model_lines_compares_equivalent_with_its_reference(
    asc_state: SessionState, clamp: Path
):
    """The .model is one text written over two lines; the edit rewrites both,
    and the export of the committed sheet matches a deck carrying the new
    model."""
    reference = clamp_reference(".model DX D(Is=2e-14\n+ N=1.9)", ".tran 1m")

    edited = await dispatched(
        asc_state,
        "edit_schematic",
        path=clamp.name,
        expected_sha256=sha_of(clamp),
        ops=[
            {
                "op": "replace_directive",
                "instruction": "regex:^\\.model DX",
                "text": ".model DX D(Is=2e-14\n+ N=1.9)",
            }
        ],
        compare={"reference": reference},
    )

    assert edited["commit_state"] == "committed", edited
    (replaced,) = edited["results"]
    assert replaced["text"]["now"] == ".model DX D(Is=2e-14\n+ N=1.9)"
    assert edited["verification"]["equivalent"] is True, edited
    assert "!.model DX D(Is=2e-14\\n+ N=1.9)" in text_of(clamp)


# ---------------------------------------------------------------------------
# A refusal or finding says exactly what to do and where
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "region",
    [
        {"left": 0, "top": 0, "right": 160, "bottom": 160},
        {"x": 0, "y": 0, "w": 160, "h": 160},
    ],
    ids=["edges", "origin_and_size"],
)
async def test_a_refused_move_region_names_its_fields_x1_y1_x2_y2(
    asc_state: SessionState, clamp: Path, region: dict[str, int]
):
    result = await dispatch(
        asc_state,
        "edit_schematic",
        {
            "path": clamp.name,
            "expected_sha256": sha_of(clamp),
            "ops": [{"op": "move", "dx": 16, "dy": 0, "region": region}],
        },
    )

    assert result.is_error
    message = tool_text(result)
    for field in ("x1", "y1", "x2", "y2"):
        assert f"region.{field}: Field required" in message, message
        assert f"region.{field} (integer, required)" in message, message
    assert text_of(clamp) == _CLAMP


async def test_a_parts_name_drawn_across_a_wire_gives_the_names_anchor_and_the_wires_ends(
    state_no_sim: SessionState, work_dir: Path
):
    """The value text across a wire is pinned in test_verify_circuit; this is
    the part's name. The name window of the suite's resistor is at (32,-32)
    from the part, drawn rightward, so R1's name starts at (232,168) and the
    wire at x=256 runs through it."""
    sheet = work_dir / "named.asc"
    sheet.write_text(
        "Version 4\nSHEET 1 880 680\n"
        "SYMBOL res 200 200 R0\nSYMATTR InstName Rload\nSYMATTR Value 1\n"
        "WIRE 256 144 256 192\n",
        encoding="utf-8",
        newline="\n",
    )

    checked = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])

    (found,) = [f for f in checked["findings"] if f["rule_id"] == "attribute_text_overlap"]
    assert found["subject"] == "Rload"
    assert found["evidence"]["detail"] == (
        "Rload's text 'Rload' at (232,168) is drawn across wire (256,144)->(256,192)"
    )
    assert found["evidence"]["coords"] == [[232, 168], [256, 144], [256, 192]]


async def test_capabilities_says_what_ngspice_is_missing_and_where_to_set_it(
    state_no_sim: SessionState,
):
    asked = await dispatched(state_no_sim, "inspect", queries=[{"kind": "capabilities"}])

    (answer,) = asked["results"]
    capabilities = answer["data"]
    ngspice = capabilities["simulators"]["ngspice"]
    assert ngspice["available"] is False
    remediation = ngspice["remediation"]
    assert remediation["config_key"] == "simulator.path"
    assert remediation["env_var"] == "LTSPICE_MCP_SIMULATOR_EXE"
    assert remediation["config_file"] == capabilities["config_path"]
    action = remediation["action"]
    assert action.startswith("Install ngspice, or set simulator.path in "), action
    assert capabilities["config_path"] in action
    assert "restart" in action


async def test_failures_warnings_and_observations_come_back_as_structured_records(
    state_no_sim: SessionState, work_dir: Path
):
    """A recorded LTspice run read by path: a signal it lacks and a
    measurement its log lacks fail as records that say what, where and what
    there is instead; the recipe's own warning points at them; a fact about
    the source is an observation with a code."""
    raw = work_dir / "rc.raw"
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.log", raw.with_suffix(".log"))

    read = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(raw)}],
        recipes=[
            {"key": "v", "metric": "value", "expr": "V(nope)", "at": "900u"},
            {"key": "m", "metric": "measurements", "names": ["nosuch"]},
        ],
    )

    signal, measurement = read["failures"]
    assert {key: signal[key] for key in ("code", "stage", "where")} == {
        "code": "recipe_failed",
        "stage": "analyze",
        "where": "rc:0",
    }
    assert signal["message"].startswith("Signal 'V(nope)' not found. Available signals: ")
    assert "V(out)" in signal["message"]
    assert measurement["where"] == "rc:0"
    assert "It holds: vfinal." in measurement["message"]
    assert read["results"]["v"]["warnings"] == [
        "No rows: this recipe failed wherever it was read; see failures at rc:0."
    ]
    (observation,) = read["observations"]
    assert observation["code"] == "path_without_deck_provenance"
    assert observation["kind"] == "provenance"
    assert "'rc'" in observation["detail"]


async def test_a_failed_edit_names_the_op_by_index_and_writes_nothing(
    asc_state: SessionState, clamp: Path
):
    edited = await dispatched(
        asc_state,
        "edit_schematic",
        path=clamp.name,
        expected_sha256=sha_of(clamp),
        ops=[
            {"op": "replace_directive", "instruction": ".tran 1m", "text": ".tran 2m"},
            {"op": "remove_directive", "instruction": "no such note"},
        ],
    )

    assert edited["commit_state"] == "not_committed", edited
    (failure,) = edited["failures"]
    assert (failure["index"], failure["op"]) == (1, "remove_directive")
    assert "'no such note'" in failure["error"]
    assert edited["error"]["code"] == "op_failed"
    assert text_of(clamp) == _CLAMP


# ---------------------------------------------------------------------------
# Checks run where the work happens
# ---------------------------------------------------------------------------

#: Each benchmark circuit LTspice exported both drawn with wires and with its
#: connections made by labels instead, and the opamp moved as well.
_REWIRED = [
    *(
        (f"benchmark/{circuit}_drawn", f"benchmark/{circuit}_labelled")
        for circuit in (
            "common_source",
            "diff_pair",
            "nmos_mirror",
            "r2r_ladder",
            "rc_lowpass",
            "rectifier",
            "rlc_line",
            "sallen_key",
            "two_stage_opamp",
            "vccs_stage",
            "vcvs_stage",
        )
    ),
    ("benchmark/two_stage_opamp_drawn", "benchmark/two_stage_opamp_moved"),
]


def ltspice_exports_recorded_sheets(state: SessionState, build: str, case_ids: list[str]) -> None:
    """The session's LTspice answers each of these recorded sheets with the
    export ``build`` wrote for it, and refuses any other sheet."""
    exports = {
        (rec.INPUTS / rec.CASES.case(case_id).source).read_bytes(): rec.recorded(
            build, f"{case_id}.net"
        )
        for case_id in case_ids
    }

    def export_of(sheet: Path) -> bytes:
        recorded = exports.get(sheet.read_bytes())
        assert recorded is not None, f"no recording of an export of {sheet}"
        return recorded.read_bytes()

    ltspice_exports(state, export_of)


@pytest.mark.parametrize(("before", "after"), _REWIRED, ids=[after for _, after in _REWIRED])
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_rewired_sheets_export_diff_shows_no_component_change(
    build: str, before: str, after: str, state_no_sim: SessionState, work_dir: Path
):
    """The same circuit drawn otherwise, compared by the export LTspice wrote
    of each sheet: the parts and their values are the same, and the circuit is
    the same circuit."""
    original = rec.stage_sheet(build, before, work_dir / "before")
    rewired = rec.stage_sheet(build, after, work_dir / "after")
    ltspice_exports_recorded_sheets(state_no_sim, build, [before, after])

    for mode in ("structural_diff", "equivalence"):
        checked = await dispatched(
            state_no_sim,
            "verify_circuit",
            path=str(rewired),
            checks=["compare"],
            compare={"reference": str(original), "mode": mode},
        )
        comparison = checked["comparison"]
        assert comparison["mode"] == mode, checked
        for key in (
            "components_added",
            "components_removed",
            "components_changed",
            "added",
            "removed",
            "value_mismatches",
        ):
            assert key not in comparison, (mode, comparison)
        assert comparison["equivalent"] is True, checked
