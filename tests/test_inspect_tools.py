"""Tests for inspect — the consolidated, honestly read-only UNDERSTAND surface.

Covers the six query kinds' happy paths (capabilities / symbols / symbol /
net over both a .asc and a netlist / components / model search+enumerate),
cursor paging with resumption and tampered/stale/cross-query cursor rejection,
mixed-batch partial failure (a denied path and an unknown kind returning
alongside good items), the model search/enumerate requirement matrix, the
symbol resolution-order precedence including a schematic-local path and a
symlink duplicate that must dedupe, the netlist net query carrying no geometry
keys at all, and capabilities key presence (pinned loosely, not by value).
"""

from __future__ import annotations

import typing
from pathlib import Path

import jsonschema
import pytest
from spicelib import AscEditor

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools as insp
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect


class FakeLT:
    """Stub LTspice class (name maps to no explicit dialect — auto-detect)."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/LTspice.exe"]


class NGspiceSimulator:
    """Stub ngspice class; the name is what dialect_for_simulator_name keys on."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/ngspice"]


def _real(p: str | Path) -> Path:
    """Realpath a directory (sync helper; keeps blocking I/O out of async tests)."""
    return Path(p).resolve()


def _schema(result) -> dict:
    sc = result.structuredContent
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
    assert result.isError is False
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
        "limits",
        "linter_version",
    ):
        assert key in data, f"missing capabilities key {key!r}"
    assert "ltspice" in data["simulators"]
    assert data["simulators"]["ltspice"]["available"] is True
    assert data["exporter_available"] is True
    for lim in (
        "max_experiment_cases",
        "analysis_budget_s",
        "result_set_ttl_hours",
        "max_points_returned",
        "inspect_page_size",
        "dwell",
    ):
        assert lim in data["limits"], f"missing limits key {lim!r}"


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


async def test_net_netlist_coordinates_rejected(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "net", "path": str(netlist), "at": [10, 20]}])
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_at"


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


async def test_components_list_netlist(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(state_no_sim, [{"kind": "components", "path": str(netlist)}])
    assert res["ok"] is True
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


async def test_components_prefix_filter(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": str(netlist), "prefix": "R"}]
    )
    assert {c["reference"] for c in res["data"]["components"]} == {"R1", "R2"}


async def test_components_bad_prefix(netlist: Path, state_no_sim: SessionState):
    (res,) = await _run(
        state_no_sim, [{"kind": "components", "path": str(netlist), "prefix": "RR"}]
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_prefix"


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
    try:
        symlink_dir.symlink_to(fixture_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this filesystem")

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
