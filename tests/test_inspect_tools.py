"""Tests for inspect — the consolidated, honestly read-only UNDERSTAND surface.

Covers the seven query kinds' happy paths (capabilities / symbols / symbol /
net over both a .asc and a netlist / components / model search+enumerate /
reference search and its table of contents),
cursor paging with resumption and tampered/stale/cross-query cursor rejection,
mixed-batch partial failure (a denied path and an unknown kind returning
alongside good items), the model search/enumerate requirement matrix, the
symbol resolution-order precedence including a schematic-local path and a
symlink duplicate that must dedupe, the netlist net query carrying no geometry
keys at all, and capabilities key presence (pinned loosely, not by value).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import typing
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest
from spicelib import AscEditor

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import raster, wsl
from ltspice_mcp.lib.deck_staging import stage_deck
from ltspice_mcp.lib.simulator_build import SimulatorExecutable, executable_identity
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools as insp
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests.conftest import installed_simulator, needs_raster, symlink_or_skip


class FakeLT:
    """Stub LTspice class (name maps to no explicit dialect — auto-detect)."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/LTspice.exe"]


class NGspiceSimulator:
    """Stub ngspice class; the name is what dialect_for_simulator_name keys on."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/ngspice"]


def _digest(p: Path) -> str:
    """SHA-256 of a file (sync helper; keeps blocking I/O out of async tests)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _real(p: str | Path) -> Path:
    """Realpath a directory (sync helper; keeps blocking I/O out of async tests)."""
    return Path(p).resolve()


def _schema(result) -> dict:
    sc = result.structured_content
    assert sc is not None
    jsonschema.Draft202012Validator(insp._OUTPUT_SCHEMA).validate(sc)
    return sc


async def _run(state: SessionState, queries: list[dict]) -> list[dict]:
    """Call inspect, validate the envelope against its schema, return results."""
    args = InspectInput.model_validate({"queries": queries})
    result = await handle_inspect(args, state)
    data = _schema(result)
    assert data["count"] == len(queries)
    # Call-level outcome invariant (design section 2): complete iff every query
    # succeeded, partial the moment any one isolates a failure. A per-item
    # failure is never a call-level failure, so isError stays false throughout.
    any_failed = any(not item["ok"] for item in data["results"])
    assert data["outcome"] == ("partial" if any_failed else "complete")
    assert result.is_error is False
    return data["results"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cap_state(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"ltspice": FakeLT, "ngspice": NGspiceSimulator})


@pytest.fixture
def netlist(work_dir: Path) -> Path:
    p = work_dir / "amp.cir"
    p.write_text(
        "* amp\nR1 in mid 1k\nR2 mid out 2k\nC1 out 0 100n\nX1 out vdd ubuf\nV1 in 0 AC 1\n.end\n"
    )
    return p


@pytest.fixture
def libfile(work_dir: Path) -> Path:
    p = work_dir / "parts.lib"
    p.write_text(".model MyNPN NPN(BF=100)\n.subckt myamp in out vcc\nR1 in out 1k\n.ends\n")
    return p


@pytest.fixture
def isolated_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the symbol precedence to the fixture library only, so the
    enumerated names and the precedence are deterministic regardless of any
    real LTspice install on the test host."""
    monkeypatch.setattr(insp, "default_stock_paths", lambda: [])
    monkeypatch.setattr(AscEditor, "simulator_lib_paths", [], raising=False)


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


async def test_capabilities_keys_present(cap_state: SessionState):
    (res,) = await _run(cap_state, [{"kind": "capabilities"}])
    assert res["ok"] is True
    assert res["kind"] == "capabilities"
    data = res["data"]
    for key in (
        "simulators",
        "default_simulator",
        "exporter_available",
        "dialects",
        "persist_jobs",
        "allowed_paths",
        "tool_profile",
        "tool_listing",
        "limits",
        "linter_version",
        "config_path",
        "python",
        "python_api",
        "render",
    ):
        assert key in data, f"missing capabilities key {key!r}"
    # The library door, named where an agent already looks for what the
    # server can do: the import, a session on this working directory, and
    # the reference lookup that replaces guessing an op's arguments.
    assert data["python_api"]["import"] == "from ltspice_mcp.api import Api"
    assert repr(str(cap_state.working_dir)) in data["python_api"]["open"]
    assert "reference(" in data["python_api"]["reference"]
    # The exec tool is on unless the operator turned it off; the entry names
    # the key and that the change takes effect at the next start.
    assert data["python_api"]["run_code"] == {
        "enabled": True,
        "config_key": "tools.run_code",
        "restart_required": True,
    }
    assert "ltspice" in data["simulators"]
    assert data["simulators"]["ltspice"]["available"] is True
    assert data["exporter_available"] is True
    # The Python API's confirmation destination: the interpreter that
    # has the package, and whether it is a durable path or a throwaway one.
    python_facts = data["python"]
    for key in ("executable", "install_kind", "ephemeral", "package_location"):
        assert key in python_facts, f"missing python fact {key!r}"
    assert isinstance(python_facts["ephemeral"], bool)
    assert data["diagnostics"] == []
    for lim in (
        "max_experiment_cases",
        "analysis_budget_s",
        "result_set_ttl_hours",
        "max_points_returned",
        "inspect_page_size",
        "dwell",
    ):
        assert lim in data["limits"], f"missing limits key {lim!r}"


async def test_capabilities_fields_returns_only_the_named_keys(cap_state: SessionState):
    """What a caller checks after a config edit, without the rest of the report."""
    (full,) = await _run(cap_state, [{"kind": "capabilities"}])
    (picked,) = await _run(
        cap_state, [{"kind": "capabilities", "fields": ["config_path", "allowed_paths"]}]
    )
    assert picked["ok"] is True
    assert picked["data"] == {key: full["data"][key] for key in ("allowed_paths", "config_path")}


def _executables(state: SessionState) -> dict[str, SimulatorExecutable | None]:
    """What the capabilities query identifies for each available simulator."""
    return {name: executable_identity(cls) for name, cls in state.available_simulators.items()}


async def test_capabilities_without_fields_is_the_whole_report(cap_state: SessionState):
    (res,) = await _run(cap_state, [{"kind": "capabilities"}])
    assert res["data"] == insp._do_capabilities(
        cap_state, raster.raster_support(), _executables(cap_state)
    )


def test_capabilities_field_names_are_the_report_keys(cap_state: SessionState):
    """The selector's vocabulary and the report's keys are one list: a key added
    to the report without a selector name, or the reverse, fails here."""
    assert set(typing.get_args(insp.CapabilityField)) == set(
        insp._do_capabilities(cap_state, raster.raster_support(), _executables(cap_state))
    )


@pytest.mark.parametrize("fields", [["no_such_key"], []], ids=["unknown", "empty"])
async def test_a_bad_capabilities_selector_fails_only_that_item(
    cap_state: SessionState, fields: list[str]
):
    results = await _run(
        cap_state,
        [{"kind": "capabilities", "fields": fields}, {"kind": "capabilities"}],
    )
    assert results[0]["ok"] is False
    assert results[0]["error"]["code"] == "invalid_query"
    assert results[1]["ok"] is True


@needs_raster
async def test_capabilities_reports_png_rendering(cap_state: SessionState):
    """An agent deciding between an inline PNG and a file path asks here first,
    rather than rendering to find out."""
    (res,) = await _run(cap_state, [{"kind": "capabilities"}])
    assert res["data"]["render"] == {"png": True, "missing": None, "reason": None, "remedy": None}


@pytest.mark.parametrize(
    ("absence", "missing"),
    [("raster_extra_missing", "extra"), ("raster_native_missing", "native_library")],
)
async def test_capabilities_reports_what_is_missing(
    cap_state: SessionState, request: pytest.FixtureRequest, absence: str, missing: str
):
    request.getfixturevalue(absence)
    (res,) = await _run(cap_state, [{"kind": "capabilities"}])
    render = res["data"]["render"]
    assert render["png"] is False
    assert render["missing"] == missing
    # The loader's own answer, reason and per-platform remedy included.
    assert render == asdict(raster.raster_support())


async def test_capabilities_carries_startup_diagnostics(config: ServerConfig):
    """A server that started degraded — a configured simulator path that does
    not exist, a requested engine that fell back to another — says so only in
    its own stderr log, which no client reads. The capabilities query is where
    a client can see it, so the notes have to ride there."""
    state = SessionState.create(
        config,
        available={"ltspice": FakeLT},
        diagnostics=["Configured simulator path does not exist: /nope/LTspice.exe"],
    )

    (res,) = await _run(state, [{"kind": "capabilities"}])

    assert res["data"]["diagnostics"] == [
        "Configured simulator path does not exist: /nope/LTspice.exe"
    ]


async def test_capabilities_names_the_keys_that_turn_a_simulator_on(cap_state: SessionState):
    """Every known-but-undetected simulator carries remediation naming the
    SAME config keys the loader reads — the config self-diagnosis surface.
    The fixture state detects only LTspice and ngspice, so the other two
    engines are the specimens."""
    from ltspice_mcp.config import SIM_PATH_ENV

    (res,) = await _run(cap_state, [{"kind": "capabilities"}])
    data = res["data"]
    undetected = {name: info for name, info in data["simulators"].items() if not info["available"]}
    assert undetected, "fixture unexpectedly detects every known simulator"
    for name, info in undetected.items():
        remediation = info["remediation"]
        assert remediation["config_key"] == "simulator.path"
        assert remediation["env_var"] == SIM_PATH_ENV
        assert remediation["config_file"] == data["config_path"]
        assert "restart" in remediation["action"], f"{name}: fix must end in a restart"
        assert remediation["excluded_by_allowlist"] is False


# ---------------------------------------------------------------------------
# symbols / symbol
# ---------------------------------------------------------------------------


async def test_symbols_happy(asc_state: SessionState, isolated_stock: None):
    (res,) = await _run(asc_state, [{"kind": "symbols"}])
    assert res["ok"] is True
    data = res["data"]
    assert data["precedence"], "expected at least one precedence directory"
    assert {"res", "cap", "nmos"} <= set(data["symbols"])
    assert data["total"] == len(data["symbols"])  # single fixture dir, under one page


async def test_symbols_filter(asc_state: SessionState, isolated_stock: None):
    (res,) = await _run(asc_state, [{"kind": "symbols", "filter": "re"}])
    names = res["data"]["symbols"]
    assert "res" in names
    assert all("re" in n.lower() for n in names)
    assert "cap" not in names


async def test_symbol_pins_per_rotation(asc_state: SessionState):
    (res,) = await _run(asc_state, [{"kind": "symbol", "name": "res"}])
    assert res["ok"] is True
    data = res["data"]
    assert data["symbol"] == "res"
    assert data["origin"] == {"x": 0, "y": 0}
    assert set(data["pins_by_rotation"]) == set(insp.ROTATIONS)
    assert set(data["bbox_by_rotation"]) == set(insp.ROTATIONS)
    # A resistor has two pins at every rotation.
    for rot in insp.ROTATIONS:
        assert len(data["pins_by_rotation"][rot]) == 2
    # R0 and R90 place the pins at different absolute coordinates.
    assert data["pins_by_rotation"]["R0"] != data["pins_by_rotation"]["R90"]


async def test_symbol_not_found(asc_state: SessionState):
    (res,) = await _run(asc_state, [{"kind": "symbol", "name": "definitely_not_a_symbol"}])
    assert res["ok"] is False
    assert res["error"]["code"] == "symbol_not_found"


# ---------------------------------------------------------------------------
# net — .asc geometric trace vs netlist card-membership
# ---------------------------------------------------------------------------


async def test_net_asc_geometric(asc_file: Path, asc_state: SessionState):
    (res,) = await _run(asc_state, [{"kind": "net", "path": str(asc_file), "at": "net:filtered"}])
    assert res["ok"] is True
    data = res["data"]
    assert data["source"] == "schematic"
    assert "filtered" in data["labels"]
    assert "start" in data
    assert isinstance(data["pins"], list)


async def test_net_netlist_has_no_geometry_keys(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "net", "path": str(netlist), "at": "net:out"}])
    assert res["ok"] is True
    data = res["data"]
    assert data["source"] == "netlist"
    assert data["node"] == "out"
    refs = {m["reference"] for m in data["members"]}
    # R2, C1, X1 touch 'out'; R1 and V1 do not.
    assert {"R2", "C1"} <= refs
    assert "R1" not in refs and "V1" not in refs
    # The contract: a netlist net query makes NO geometry claim.
    for banned in (
        "start",
        "coordinates",
        "coordinates_truncated",
        "pins",
        "labels",
        "is_shorted",
    ):
        assert banned not in data, f"netlist net leaked geometry key {banned!r}"
    for member in data["members"]:
        assert "x" not in member and "y" not in member
        assert set(member) == {"reference", "terminal"}


@pytest.fixture
def wide_net_asc(work_dir: Path, asc_state: SessionState) -> Path:
    """A schematic whose one net carries more wire vertices than a single page.

    600 collinear segments share 601 endpoints, all on the flagged net — enough
    to make the coordinate page boundary observable without depending on the
    page size's value.
    """
    lines = ["Version 4.1", "SHEET 1 880 680"]
    lines += [f"WIRE {x} 0 {x + 16} 0" for x in range(0, 600 * 16, 16)]
    lines.append("FLAG 0 0 bignet")
    p = work_dir / "wide.asc"
    p.write_text("\n".join(lines) + "\n")
    return p


async def test_net_asc_coordinates_advance_across_pages(
    wide_net_asc: Path, asc_state: SessionState
):
    """Wire-vertex coordinates page forward instead of re-serving the first page.

    The item's cursor advances pins AND coordinates, so every vertex is
    reachable exactly once. Announcing more coordinates while handing back the
    same ones on every page is worse than plain truncation: it advertises data
    the caller has no lever to reach.
    """
    query: dict = {"kind": "net", "path": str(wide_net_asc), "at": "net:bignet"}
    (first,) = await _run(asc_state, [query])
    assert first["ok"] is True
    assert first["data"]["coordinates_truncated"] is True
    cursor = first["next_cursor"]
    assert cursor is not None, "coordinates reported truncated with no cursor to reach the rest"
    total = first["data"]["total_coordinates"]
    page1 = first["data"]["coordinates"]
    assert total > len(page1), "fixture net must exceed one coordinate page"

    (second,) = await _run(asc_state, [{**query, "cursor": cursor}])
    assert second["ok"] is True
    page2 = second["data"]["coordinates"]
    assert page2, "page 2 returned no coordinates"
    assert not [c for c in page2 if c in page1], "page 2 repeated page 1's coordinates"

    # Walk to exhaustion: every vertex appears exactly once across the pages.
    seen = list(page1) + list(page2)
    cursor = second["next_cursor"]
    pages = 2
    while cursor is not None:
        (nxt,) = await _run(asc_state, [{**query, "cursor": cursor}])
        seen.extend(nxt["data"]["coordinates"])
        cursor = nxt["next_cursor"]
        pages += 1
        assert pages < 20, "cursor failed to terminate"
    assert len(seen) == total
    assert len({(c["x"], c["y"]) for c in seen}) == total


async def test_page_counters_cover_every_collection_the_item_pages(
    wide_net_asc: Path, asc_state: SessionState
):
    """``page.returned == page.total`` must never read "done" while rows remain.

    A .asc net pages pins and wire vertices under one cursor. Counters that
    described the pins alone reported "3 of 3 returned" on a net with hundreds of
    unfetched coordinates, so a caller stopping on ``returned == total`` dropped
    them silently while ``truncated`` said otherwise.
    """
    query: dict = {"kind": "net", "path": str(wide_net_asc), "at": "net:bignet"}
    (first,) = await _run(asc_state, [query])
    page = first["page"]
    assert page["truncated"] is True
    assert page["returned"] < page["total"], "counters read complete while the item is truncated"
    assert set(page["collections"]) == {"pins", "coordinates"}
    coordinates = page["collections"]["coordinates"]
    assert coordinates["total"] == first["data"]["total_coordinates"]
    assert coordinates["returned"] == len(first["data"]["coordinates"])
    assert coordinates["truncated"] is True

    # Walking to exhaustion accounts for exactly the rows the first page promised.
    seen = page["returned"]
    cursor = first["next_cursor"]
    while cursor is not None:
        (nxt,) = await _run(asc_state, [{**query, "cursor": cursor}])
        assert nxt["page"]["total"] == page["total"]
        assert not nxt["page"]["truncated"] or nxt["page"]["returned"] < nxt["page"]["total"]
        seen += nxt["page"]["returned"]
        cursor = nxt["next_cursor"]
    assert seen == page["total"]


async def test_net_netlist_coordinates_rejected(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "net", "path": str(netlist), "at": [10, 20]}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_at"


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["amp.cir", "sub/../amp.cir", "amp.spice"],
    ids=["path", "parent-segment", "spice-suffix"],
)
async def test_components_list_netlist(
    netlist: Path, state_no_sim: SessionState, work_dir: Path, spelling: str
):
    """A path is judged by where it lands (``sub/../amp.cir`` is the deck
    itself), and ``.spice`` is what xschem and the sky130 testbenches write."""
    await asyncio.to_thread((work_dir / "sub").mkdir)
    spice = work_dir / "amp.spice"
    await asyncio.to_thread(spice.write_bytes, await asyncio.to_thread(netlist.read_bytes))
    (res,) = await _run(state_no_sim, [{"kind": "components", "path": spelling}])
    assert res["ok"] is True, res
    data = res["data"]
    assert data["detail"] == "list"
    refs = [c["reference"] for c in data["components"]]
    assert refs == sorted(refs)
    assert set(refs) == {"C1", "R1", "R2", "V1", "X1"}
    # list detail carries value only, never full-detail keys.
    for c in data["components"]:
        assert "nodes" not in c


async def test_components_full_netlist(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": str(netlist), "detail": "full"}]
    )
    by_ref = {c["reference"]: c for c in res["data"]["components"]}
    assert by_ref["R1"]["nodes"] == ["in", "mid"]
    assert by_ref["X1"]["nodes"] == ["out", "vdd"]
    assert by_ref["X1"]["model"] == "ubuf"


async def test_components_full_asc(asc_file: Path, asc_state: SessionState):
    (res,) = await _run(
        asc_state, [{"kind": "components", "path": str(asc_file), "detail": "full"}]
    )
    by_ref = {c["reference"]: c for c in res["data"]["components"]}
    assert {"C1", "R1", "V1"} <= set(by_ref)
    c1 = by_ref["C1"]
    assert c1["symbol"] == "cap"
    assert "position" in c1 and "rotation" in c1
    assert "pins" in c1 and "bounding_box" in c1


async def test_components_asc_reports_sheet_digest(asc_file: Path, asc_state: SessionState):
    # edit_schematic refuses to touch an existing sheet without its sha256, and
    # a read is the only supported way to get one — so the .asc component query
    # reports it. Netlists carry no such token and get no key.
    (res,) = await _run(asc_state, [{"kind": "components", "path": str(asc_file)}])
    assert res["data"]["sha256"] == _digest(asc_file)


async def test_components_netlist_has_no_digest(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "components", "path": str(netlist)}])
    assert "sha256" not in res["data"]


async def test_components_netlist_relays_lexer_warnings(
    work_dir: Path, state_no_sim: SessionState
):
    """The lexer reports what it had to guess about — here, a .SUBCKT that
    never closes, which means every card after it was read as subcircuit body
    and the component list is a list of the wrong scope. Reading only the
    cards drops that: the rows come back looking authoritative."""
    deck = work_dir / "unclosed.cir"
    deck.write_text("* unclosed\n.subckt AMP a b\nR1 a b 1k\nV1 a 0 1\n.end\n")

    (res,) = await _run(state_no_sim, [{"kind": "components", "path": str(deck)}])

    assert any("unclosed .SUBCKT" in note for note in res["data"]["warnings"])


async def test_components_netlist_omits_the_warnings_key_when_clean(
    netlist: Path, state_no_sim: SessionState
):
    (res,) = await _run(state_no_sim, [{"kind": "components", "path": str(netlist)}])
    assert "warnings" not in res["data"]


async def test_net_asc_reports_sheet_digest(asc_file: Path, asc_state: SessionState):
    (res,) = await _run(asc_state, [{"kind": "net", "path": str(asc_file), "at": "net:filtered"}])
    assert res["data"]["sha256"] == _digest(asc_file)


async def test_components_prefix_filter(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": str(netlist), "prefix": "R"}]
    )
    assert {c["reference"] for c in res["data"]["components"]} == {"R1", "R2"}


@pytest.mark.parametrize("prefix", ["R", "r"])
async def test_components_prefix_filter_asc_ignores_case(
    asc_file: Path, asc_state: SessionState, prefix: str
):
    (res,) = await _run(
        asc_state, [{"kind": "components", "path": str(asc_file), "prefix": prefix}]
    )
    assert res["ok"] is True, res
    assert {c["reference"] for c in res["data"]["components"]} == {"R1"}


@pytest.mark.parametrize(
    ("prefix", "names"),
    [("LX*", "prefix='LX'"), ("*", "omit it"), ("", "without spaces"), ("L X", "without spaces")],
)
async def test_components_bad_prefix(
    netlist: Path, state_no_sim: SessionState, prefix: str, names: str
):
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": str(netlist), "prefix": prefix}]
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_prefix"
    assert names in res["error"]["message"]


@pytest.fixture
async def prefixed_circuits(asc_state: SessionState, work_dir: Path) -> dict[str, Path]:
    """The same five references as a netlist and as a schematic: two share a
    multi-letter stem, and each stem also has a plain single-letter sibling."""
    from tests._asc_ops import build_sheet

    deck = work_dir / "prefixed.cir"
    deck.write_text(
        "* prefixed\nLX1 a b 1u\nLX2 b c 1u\nL1 c 0 1u\n"
        "MXO1 d g 0 0 nch\nM1 d g 0 0 nch\n.model nch nmos\n.end\n"
    )
    placements = [
        ("LX1", "ind", 128, 128),
        ("LX2", "ind", 256, 128),
        ("L1", "ind", 384, 128),
        ("MXO1", "nmos", 128, 384),
        ("M1", "nmos", 384, 384),
    ]
    await build_sheet(
        asc_state,
        "prefixed",
        [
            {"op": "add_component", "reference": ref, "symbol": sym, "x": x, "y": y}
            for ref, sym, x, y in placements
        ],
    )
    return {"netlist": deck, "asc": work_dir / "prefixed.asc"}


@pytest.mark.parametrize("source", ["netlist", "asc"])
@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("LX", {"LX1", "LX2"}),
        ("lx", {"LX1", "LX2"}),
        ("MXO", {"MXO1"}),
        ("L", {"L1", "LX1", "LX2"}),
        ("m", {"M1", "MXO1"}),
    ],
)
async def test_components_prefix_is_a_case_insensitive_reference_prefix(
    asc_state: SessionState,
    prefixed_circuits: dict[str, Path],
    source: str,
    prefix: str,
    expected: set[str],
):
    path = prefixed_circuits[source]
    (res,) = await _run(asc_state, [{"kind": "components", "path": str(path), "prefix": prefix}])
    assert res["ok"] is True, res
    assert {c["reference"] for c in res["data"]["components"]} == expected
    assert res["data"]["prefix"] == prefix


# ---------------------------------------------------------------------------
# model — search / enumerate
# ---------------------------------------------------------------------------


async def test_model_enumerate(libfile: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim, [{"kind": "model", "mode": "enumerate", "libs": [str(libfile)]}]
    )
    assert res["ok"] is True
    names = {r["name"] for r in res["data"]["results"]}
    assert "MyNPN" in names
    assert "myamp" in names


async def test_model_search_with_libs(libfile: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim,
        [{"kind": "model", "mode": "search", "query": "MyNPN", "libs": [str(libfile)]}],
    )
    assert res["ok"] is True
    assert res["data"]["results"][0]["name"] == "MyNPN"


@pytest.fixture
def simulator_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A detected simulator's own model library, outside the sandbox root."""
    lib = tmp_path_factory.mktemp("simulator_install") / "lib"
    (lib / "cmp").mkdir(parents=True)
    (lib / "cmp" / "standard.bjt").write_text(
        ".model 2N3904 NPN(BF=300)\n.model 2N3906 PNP(BF=200)\n"
    )
    (lib / "sub").mkdir()
    (lib / "sub" / "LT1001.sub").write_text(".subckt LT1001 in out\nR1 in out 1k\n.ends\n")
    # Not a library file, so a search never reads it.
    (lib / "sub" / "readme.txt").write_text(".model 2N3905 NPN(BF=1)\n")
    return lib


@pytest.fixture
def library_state(config: ServerConfig, simulator_library: Path) -> SessionState:
    """A session whose detected LTspice reports ``simulator_library`` as its own
    library."""
    installed = installed_simulator(simulator_library, base=FakeLT)
    return SessionState.create(config, available={"ltspice": installed})


async def test_model_enumerate_reads_the_simulators_own_library(
    simulator_library: Path, library_state: SessionState
):
    """Staging, the include resolver and the hierarchy reader all read the
    detected simulator's library under a default sandbox; a model lookup into
    it must not be the one read that is refused."""
    shipped = simulator_library / "cmp" / "standard.bjt"
    (res,) = await _run(
        library_state, [{"kind": "model", "mode": "enumerate", "libs": [str(shipped)]}]
    )
    assert res["ok"] is True, res
    assert {r["name"] for r in res["data"]["results"]} == {"2N3904", "2N3906"}


async def test_model_libs_outside_the_sandbox_and_the_simulator_library_denied(
    library_state: SessionState, tmp_path_factory: pytest.TempPathFactory
):
    stray = tmp_path_factory.mktemp("elsewhere") / "parts.lib"
    await asyncio.to_thread(stray.write_text, ".model STRAY NPN(BF=1)\n")
    (res,) = await _run(
        library_state, [{"kind": "model", "mode": "enumerate", "libs": [str(stray)]}]
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "path_denied"


async def test_model_search_without_libs_searches_the_simulators_own_library(
    simulator_library: Path, library_state: SessionState
):
    """With 'libs' omitted the search reads the detected simulator's library,
    and every file it names can be read back through 'libs'."""
    (res,) = await _run(library_state, [{"kind": "model", "mode": "search", "query": "2N3905"}])
    assert res["ok"] is True, res
    rows = res["data"]["results"]
    assert [r["name"] for r in rows[:2]] == ["2N3904", "2N3906"]
    assert {r["name"] for r in rows}.isdisjoint({"LT1001"})
    source = rows[0]["source_path"]
    assert Path(source) == (simulator_library / "cmp" / "standard.bjt").resolve()

    (back,) = await _run(library_state, [{"kind": "model", "mode": "enumerate", "libs": [source]}])
    assert back["ok"] is True, back
    assert "2N3904" in {r["name"] for r in back["data"]["results"]}


async def test_model_rows_have_one_shape_on_every_route(
    simulator_library: Path, library_state: SessionState
):
    """A search naming 'libs', a search of the simulator's own libraries and
    an enumerate report the same part the same way, so what a caller can do
    with a row does not depend on how it asked."""
    shipped = simulator_library / "cmp" / "standard.bjt"
    results = await _run(
        library_state,
        [
            {"kind": "model", "mode": "search", "query": "2N3904", "libs": [str(shipped)]},
            {"kind": "model", "mode": "search", "query": "2N3904"},
            {"kind": "model", "mode": "enumerate", "libs": [str(shipped)]},
        ],
    )
    named, installed, enumerated = (
        next(row for row in res["data"]["results"] if row["name"] == "2N3904") for res in results
    )
    assert named == installed
    assert {key: value for key, value in named.items() if key != "score"} == enumerated
    assert enumerated["include_directive"] == f'.include "{shipped.resolve()}"'
    assert enumerated["usage"] == "Qxxx C B E 2N3904"


@pytest.fixture
def on_wsl(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment of a WSL session: detection says WSL, and a wslpath on
    PATH spells a Linux-side file the way the real one does."""
    if sys.platform == "win32":
        pytest.skip("WSL interop runs on the Linux side; native Windows has no wslpath")
    bin_dir = tmp_path_factory.mktemp("bin")
    wslpath = bin_dir / "wslpath"
    wslpath.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('\\\\\\\\wsl.localhost\\\\Distro' + sys.argv[-1].replace('/', '\\\\'))\n"
    )
    wslpath.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(wsl, "_is_wsl_cached", True)


@pytest.mark.usefixtures("on_wsl")
async def test_model_include_directive_is_staged_on_wsl(
    simulator_library: Path, library_state: SessionState, work_dir: Path
):
    """On WSL the directive names the file as the server sees it, which is
    what staging reads. A Windows spelling made per row cost a wslpath process
    each, and for a file on the Linux side came back as a wsl.localhost path
    that staging cannot map back, so the directive a search handed out named
    nothing a run could stage."""
    (res,) = await _run(library_state, [{"kind": "model", "mode": "search", "query": "2N3904"}])
    assert res["ok"] is True, res
    directive = res["data"]["results"][0]["include_directive"]
    deck = work_dir / "tb.cir"
    await asyncio.to_thread(deck.write_text, f"* tb\nQ1 c b 0 2N3904\n{directive}\n.end\n")
    library = await asyncio.to_thread(simulator_library.resolve)

    staged = await asyncio.to_thread(
        stage_deck, deck, work_dir / "staged", [work_dir], origin=deck, simulator_roots=[library]
    )

    assert library / "cmp" / "standard.bjt" in {included.source for included in staged.includes}


async def test_model_search_without_libs_rejects_a_cursor_after_a_library_edit(
    simulator_library: Path, library_state: SessionState, monkeypatch: pytest.MonkeyPatch
):
    """The rows come out of the simulator's library files, so the cursor binds
    their revision as it binds the files named in 'libs'."""
    monkeypatch.setattr(insp, "_PAGE_SIZE", 1)
    query = {"kind": "model", "mode": "search", "query": "2N3905"}
    (first,) = await _run(library_state, [query])
    assert first["ok"] is True, first
    cursor = first["next_cursor"]
    assert cursor is not None

    shipped = simulator_library / "cmp" / "standard.bjt"
    await asyncio.to_thread(
        shipped.write_text,
        ".model 2N3903 NPN(BF=250)\n.model 2N3904 NPN(BF=300)\n.model 2N3906 PNP(BF=200)\n",
    )

    (res,) = await _run(library_state, [{**query, "cursor": cursor}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_cursor"


async def test_model_search_without_libs_or_a_simulator_finds_nothing(
    state_no_sim: SessionState,
):
    (res,) = await _run(state_no_sim, [{"kind": "model", "mode": "search", "query": "2N3904"}])
    assert res["ok"] is True, res
    assert res["data"]["results"] == []
    assert res["data"]["total"] == 0


async def test_model_search_requires_query(state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "model", "mode": "search"}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_query"
    assert "query" in res["error"]["message"]


async def test_model_enumerate_requires_libs(state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "model", "mode": "enumerate"}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_query"
    assert "libs" in res["error"]["message"]


async def test_model_enumerate_rejects_a_query_it_would_ignore(
    libfile: Path, state_no_sim: SessionState
):
    """Enumerate never filters, so accepting 'query' would echo back a filter
    that was not applied."""
    (res,) = await _run(
        state_no_sim,
        [{"kind": "model", "mode": "enumerate", "libs": [str(libfile)], "query": "MyNPN"}],
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_query"
    assert "query" in res["error"]["message"]


async def test_requirement_matrix_isolates(libfile: Path, state_no_sim: SessionState):
    """A search-missing-query and an enumerate-missing-libs each fail, while a
    valid enumerate in the same batch returns."""
    results = await _run(
        state_no_sim,
        [
            {"kind": "model", "mode": "search"},
            {"kind": "model", "mode": "enumerate"},
            {"kind": "model", "mode": "enumerate", "libs": [str(libfile)]},
        ],
    )
    assert results[0]["ok"] is False and results[0]["error"]["code"] == "invalid_query"
    assert results[1]["ok"] is False and results[1]["error"]["code"] == "invalid_query"
    assert results[2]["ok"] is True
    assert results[2]["index"] == 2


# ---------------------------------------------------------------------------
# Mixed-batch partial failure + unknown kind
# ---------------------------------------------------------------------------


async def test_unknown_kind_lists_supported(cap_state: SessionState):
    (res,) = await _run(cap_state, [{"kind": "nonsense"}])
    assert res["ok"] is False
    assert res["kind"] == "nonsense"
    assert res["error"]["code"] == "unsupported_variant"
    assert set(res["error"]["supported"]) == set(insp.SUPPORTED_KINDS)


async def test_mixed_batch_partial_failure(netlist: Path, cap_state: SessionState, work_dir: Path):
    denied = str(work_dir.parent / "outside.cir")  # outside the sandbox root
    results = await _run(
        cap_state,
        [
            {"kind": "capabilities"},
            {"kind": "net", "path": denied, "at": "net:out"},
            {"kind": "totally_made_up"},
            {"kind": "components", "path": str(netlist)},
        ],
    )
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False and results[1]["error"]["code"] == "path_denied"
    assert results[2]["ok"] is False and results[2]["error"]["code"] == "unsupported_variant"
    assert results[3]["ok"] is True
    # Every result keeps its input position.
    assert [r["index"] for r in results] == [0, 1, 2, 3]


async def test_outcome_enum_advertises_only_reachable_values(state_no_sim: SessionState):
    """The envelope must not document an outcome no code path can produce.

    Per-item isolation is inspect's contract: every query fault — including an
    unexpected one — is caught and returned as that item's error, so a batch
    where every query fails is still 'partial' with isError false. A genuine
    call-level fault raises out of the handler, and the SDK answers with isError
    and no structuredContent at all, so this envelope is never the carrier of
    'failed'. ('in_progress' is likewise absent: inspect has no async work.)
    """
    assert insp._OUTPUT_SCHEMA["properties"]["outcome"]["enum"] == ["complete", "partial"]

    # The behavioural half: nothing an individual query can do reaches a
    # call-level outcome. _run pins outcome == "partial" and isError is False.
    results = await _run(
        state_no_sim,
        [
            {"kind": "no_such_kind"},
            {"kind": "symbol", "name": "definitely_not_a_symbol"},
            {"kind": "net", "path": "/etc/passwd", "at": "net:x"},
        ],
    )
    assert [r["ok"] for r in results] == [False, False, False]


# ---------------------------------------------------------------------------
# Cursor paging: resumption, tamper, cross-query rejection
# ---------------------------------------------------------------------------


async def test_cursor_paging_and_resumption(
    netlist: Path, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(insp, "_PAGE_SIZE", 2)
    path = str(netlist)
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        query: dict = {"kind": "components", "path": path}
        if cursor is not None:
            query["cursor"] = cursor
        (res,) = await _run(state_no_sim, [query])
        assert res["ok"] is True
        seen.extend(c["reference"] for c in res["data"]["components"])
        pages += 1
        cursor = res["next_cursor"]
        if cursor is None:
            break
        assert pages < 10, "cursor failed to terminate"
    assert pages == 3  # 5 components at 2 per page
    assert set(seen) == {"C1", "R1", "R2", "V1", "X1"}
    assert len(seen) == 5  # no overlap across pages
    # A single-collection kind omits 'collections': with one collection there is
    # nothing to disambiguate, so it only restated the three counters above it.
    assert "collections" not in res["page"]


async def test_cursor_minted_before_an_edit_is_rejected(
    netlist: Path, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    """A cursor that outlived the file it paged must not resume at a stale offset.

    The token carries a row offset into a list the server re-derives on every
    page, so replaying it after an edit silently skips or repeats components.
    Binding the file's size and mtime makes that a clean rejection instead.
    """
    monkeypatch.setattr(insp, "_PAGE_SIZE", 2)
    path = str(netlist)
    (first,) = await _run(state_no_sim, [{"kind": "components", "path": path}])
    cursor = first["next_cursor"]
    assert cursor is not None

    await asyncio.to_thread(
        netlist.write_text, "* amp\nR1 in mid 1k\nR9 mid out 9k\nV1 in 0 AC 1\n.end\n"
    )

    (res,) = await _run(state_no_sim, [{"kind": "components", "path": path, "cursor": cursor}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_cursor"


@pytest.mark.parametrize(
    ("mode", "extra"),
    [("enumerate", {}), ("search", {"query": "AMP1"})],
    ids=["enumerate", "search"],
)
async def test_model_cursor_minted_before_a_library_edit_is_rejected(
    work_dir: Path,
    state_no_sim: SessionState,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    extra: dict,
):
    """A model page is rows read out of library FILES, so its cursor binds them.

    Same defect the components and net cursors close: the token carries a row
    offset into a list the server re-derives on every page, so a library edited
    between pages silently shifts the rows that offset lands on. Both modes read
    the files named in 'libs', so both must reject the stale token.
    """
    monkeypatch.setattr(insp, "_PAGE_SIZE", 1)
    lib = work_dir / "parts2.lib"
    lib.write_text(".model AMP1 NPN(BF=100)\n.model AMP2 NPN(BF=100)\n")
    query: dict = {"kind": "model", "mode": mode, "libs": [str(lib)], **extra}

    (first,) = await _run(state_no_sim, [query])
    assert first["ok"] is True
    cursor = first["next_cursor"]
    assert cursor is not None

    await asyncio.to_thread(
        lib.write_text,
        ".model AMP0 NPN(BF=90)\n.model AMP1 NPN(BF=100)\n.model AMP2 NPN(BF=100)\n",
    )

    (res,) = await _run(state_no_sim, [{**query, "cursor": cursor}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_cursor"


async def test_tampered_cursor_isolated(
    netlist: Path, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(insp, "_PAGE_SIZE", 2)
    path = str(netlist)
    (first,) = await _run(state_no_sim, [{"kind": "components", "path": path}])
    good_cursor = first["next_cursor"]
    assert good_cursor is not None
    tampered = ("A" if good_cursor[0] != "A" else "B") + good_cursor[1:]
    (res,) = await _run(state_no_sim, [{"kind": "components", "path": path, "cursor": tampered}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_cursor"


async def test_cross_query_cursor_rejected(
    netlist: Path, state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch
):
    """A cursor minted for one query must not resume a different one."""
    monkeypatch.setattr(insp, "_PAGE_SIZE", 2)
    path = str(netlist)
    # Cursor from an unfiltered components listing...
    (first,) = await _run(state_no_sim, [{"kind": "components", "path": path}])
    cursor = first["next_cursor"]
    assert cursor is not None
    # ...replayed against a prefix-filtered listing (different identity signature).
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": path, "prefix": "R", "cursor": cursor}]
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_cursor"


# ---------------------------------------------------------------------------
# Symbol precedence: schematic-local path + symlink duplicate dedup
# ---------------------------------------------------------------------------


async def test_symbol_precedence_local_and_symlink_dedup(
    work_dir: Path, asc_symbols: Path, isolated_stock: None
):
    fixture_dir = asc_symbols
    # A symlink pointing at the same fixture directory must collapse to one entry.
    symlink_dir = work_dir / "linked_syms"
    symlink_or_skip(symlink_dir, fixture_dir, target_is_directory=True)

    proj = work_dir / "proj"
    proj.mkdir()
    sheet = proj / "sheet.asc"
    sheet.write_text("Version 4\nSHEET 1 880 680\n")

    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        symbol_paths=[fixture_dir, symlink_dir],
    )
    state = SessionState.create(config, available={})

    (res,) = await _run(state, [{"kind": "symbols", "path": str(sheet)}])
    assert res["ok"] is True
    precedence = res["data"]["precedence"]

    # The schematic's own directory leads the reported precedence.
    assert precedence[0]["tier"] == "schematic-local"
    assert _real(precedence[0]["dir"]) == _real(proj)

    # fixture_dir, the symlink to it, and AscEditor.custom_lib_paths all resolve
    # to one real directory — it must appear exactly once.
    fixture_real = _real(fixture_dir)
    collapsed = [e for e in precedence if _real(e["dir"]) == fixture_real]
    assert len(collapsed) == 1


# ---------------------------------------------------------------------------
# reference — the tools' own branch vocabulary
# ---------------------------------------------------------------------------


async def test_reference_search_returns_ranked_branches_with_their_fields(
    cap_state: SessionState,
):
    """The lookup a caller reaches for when they know the measurement but not
    the recipe name. It runs through the real dispatch, so the suite's
    conformance hook checks the payload against inspect's output schema."""
    (res,) = await _run(cap_state, [{"kind": "reference", "query": "phase margin"}])
    assert res["ok"] is True and res["kind"] == "reference"
    data = res["data"]
    assert data["query"] == "phase margin"
    assert data["matches"], "phase margin matched nothing"
    top = data["matches"][0]
    assert (top["tool"], top["name"]) == ("analyze_results", "stability")
    assert "phase margin" in top["summary"].lower()
    assert "stability" in top["call"]
    fields = {field["name"]: field for field in top["fields"]}
    assert fields["signal"]["required"] is True
    assert fields["signal"]["type"] == "string"
    # A reference lookup reads nothing off disk, so it has no page to resume.
    assert "next_cursor" not in res


async def test_reference_limit_bounds_the_matches_and_reports_the_rest(
    cap_state: SessionState,
):
    (res,) = await _run(cap_state, [{"kind": "reference", "query": "gain", "limit": 2}])
    data = res["data"]
    assert data["returned"] == 2 == len(data["matches"])
    assert data["total_matches"] > 2
    assert "limit" in data["hint"]


async def test_reference_at_the_limit_cap_stops_pointing_at_limit(cap_state: SessionState):
    """A hint naming a lever already at its ceiling is noise. At the cap the
    only move left is a narrower query, so that is all the hint offers."""
    (res,) = await _run(
        cap_state,
        [{"kind": "reference", "query": "signal name", "limit": insp.REFERENCE_LIMIT_CAP}],
    )
    data = res["data"]
    assert data["total_matches"] > data["returned"], (
        "this query no longer overflows the cap, so it cannot exercise the hint"
    )
    assert "Raise 'limit'" not in data["hint"]
    assert "narrow the query" in data["hint"]


async def test_reference_lists_run_code_only_on_a_session_that_serves_it(
    cap_state: SessionState, work_dir
):
    """The index is built from the registry; the table a session hands out is
    cut to what it dispatches, so an agent is never pointed at a tool the
    operator turned off."""
    from ltspice_mcp.config import ServerConfig

    (res,) = await _run(cap_state, [{"kind": "reference", "query": "run_code"}])
    assert res["data"]["matches"][0]["tool"] == "run_code"
    (toc,) = await _run(cap_state, [{"kind": "reference"}])
    assert "run_code" in {group["tool"] for group in toc["data"]["contents"]}

    silent = SessionState.create(
        ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], run_code=False),
        available={},
    )
    (res,) = await _run(silent, [{"kind": "reference", "query": "run_code"}])
    assert "run_code" not in {match["tool"] for match in res["data"]["matches"]}
    (toc,) = await _run(silent, [{"kind": "reference"}])
    assert "run_code" not in {group["tool"] for group in toc["data"]["contents"]}


async def test_reference_without_a_query_returns_the_table_of_contents(
    cap_state: SessionState,
):
    (res,) = await _run(cap_state, [{"kind": "reference"}])
    data = res["data"]
    assert "matches" not in data
    tools = {group["tool"] for group in data["contents"]}
    assert tools == {
        "run_experiments",
        "analyze_results",
        "inspect",
        "edit_schematic",
        "verify_circuit",
        "jobs",
        # Every tool's own arguments are listed too, plot_waveform included.
        "plot_waveform",
        "run_code",
    }
    assert {group["family"] for group in data["contents"]} >= {"argument"}
    listed = sum(len(group["branches"]) for group in data["contents"])
    assert listed == data["total_branches"]
    for group in data["contents"]:
        for branch in group["branches"]:
            # A contents line is a name and one line; the field tables are what
            # a query pays for.
            assert set(branch) == {"name", "summary"}
    assert "query" in data["hint"]


async def test_reference_says_so_when_nothing_matches(cap_state: SessionState):
    (res,) = await _run(cap_state, [{"kind": "reference", "query": "zzz quuxbar"}])
    data = res["data"]
    assert data["matches"] == [] and data["total_matches"] == 0
    assert "inspect(kind='guide')" in data["hint"]


async def test_reference_limit_above_the_cap_is_rejected_for_that_item_only(
    cap_state: SessionState,
):
    good, bad = await _run(
        cap_state,
        [
            {"kind": "reference", "query": "thd"},
            {"kind": "reference", "query": "thd", "limit": insp.REFERENCE_LIMIT_CAP + 1},
        ],
    )
    assert good["ok"] is True
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_query"


async def test_reference_is_advertised_as_a_supported_kind(cap_state: SessionState):
    """An unknown kind reports the supported list, and 'reference' has to be in
    it — a lookup nothing names is a lookup nobody finds."""
    (res,) = await _run(cap_state, [{"kind": "nonsense"}])
    assert res["ok"] is False
    assert "reference" in res["error"]["supported"]
    assert "reference" in insp.SUPPORTED_KINDS
