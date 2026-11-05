"""The Python API's inspect returns every row a paged query has.

``Api.inspect`` follows each query's ``next_cursor`` and joins the pages into
one result. Which data key of a page holds the rows its cursor pages is what
the join has to know, and every paged kind answers it in its ``page``. Each
query here runs through the real handlers twice: page by page, through the
API's raw-page door that returns a page as MCP sends it, and once collected,
whose rows must be every row of every page.

Before every paged item named its lists, the join kept the newest page alone
for a kind that pages one list, under a ``total`` that counted every row.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools
from tests._ltspice_window import a_window, put_windows
from tests.conftest import FIXTURES_DIR, SyncApi
from tests.test_inspect_simulator_docs import an_install


@pytest.fixture(autouse=True)
def one_row_a_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every input below then spans several pages."""
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 1)
    monkeypatch.setattr(inspect_tools, "_COORD_PAGE_SIZE", 1)


def page_by_page(api: SyncApi, query: dict[str, Any]) -> list[dict[str, Any]]:
    pages = [api.inspect(raw_page=True, queries=[query])["results"][0]]
    while "next_cursor" in pages[-1]:
        assert len(pages) < 200, "the cursor never ran out"
        resumed = {**query, "cursor": pages[-1]["next_cursor"]}
        pages.append(api.inspect(raw_page=True, queries=[resumed])["results"][0])
    return pages


def collected_whole(state: SessionState, query: dict[str, Any], *lists: str) -> None:
    """The query collected in one API call holds every row its pages hold."""
    api = SyncApi(state)
    pages = page_by_page(api, query)
    assert all(page["ok"] for page in pages), pages
    assert len(pages) > 1, "one page proves nothing about joining them"
    (collected,) = api.inspect(queries=[query])["results"]
    assert collected["ok"], collected
    for name in lists:
        rows = [row for page in pages for row in page["data"].get(name, [])]
        assert collected["data"].get(name, []) == rows, name
    assert sum(len(collected["data"].get(name, [])) for name in lists) == pages[0]["page"]["total"]


def a_deck(work_dir: Path) -> Path:
    deck = work_dir / "ladder.cir"
    deck.write_text(
        "* ladder\n" + "".join(f"R{n} a n{n} 1k\n" for n in range(5)) + ".end\n",
        encoding="utf-8",
    )
    return deck


def test_components(state_no_sim: SessionState, work_dir: Path):
    collected_whole(
        state_no_sim, {"kind": "components", "path": str(a_deck(work_dir))}, "components"
    )


def test_a_netlist_net(state_no_sim: SessionState, work_dir: Path):
    query = {"kind": "net", "path": str(a_deck(work_dir)), "at": "a"}
    collected_whole(state_no_sim, query, "members")


def test_models(state_no_sim: SessionState, work_dir: Path):
    library = work_dir / "diodes.lib"
    library.write_text("".join(f".model D{n} D(Is=1e-14)\n" for n in range(4)), encoding="utf-8")
    query = {"kind": "model", "mode": "enumerate", "libs": [str(library)]}
    collected_whole(state_no_sim, query, "results")


def test_symbols(asc_state: SessionState):
    collected_whole(asc_state, {"kind": "symbols"}, "symbols")


@pytest.mark.parametrize(
    ("query", "rows"), [({}, "docs"), ({"name": "MEAS-REFERENCE.md"}, "sections")]
)
def test_simulator_docs(
    config: ServerConfig,
    tmp_path_factory: pytest.TempPathFactory,
    query: dict[str, Any],
    rows: str,
):
    simulator = an_install(tmp_path_factory.mktemp("install"))
    state = SessionState.create(config, available={"fake": simulator})
    collected_whole(state, {"kind": "simulator_docs", **query}, rows)


def test_hierarchy(state_no_sim: SessionState, work_dir: Path):
    deck = work_dir / "blocks.cir"
    deck.write_text(
        "* blocks\n.subckt leaf p\nR1 p 0 1k\nR2 p 0 2k\n.ends leaf\nXA a leaf\nXB b leaf\n.end\n",
        encoding="utf-8",
    )
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ngspice"}
    collected_whole(state_no_sim, query, "instances")


def test_results(state_no_sim: SessionState, work_dir: Path):
    raw = work_dir / "rc.raw"
    shutil.copyfile(FIXTURES_DIR / "ltspice_tran_rc.raw", raw)
    query = {"kind": "results", "view": "signals", "path": str(raw)}
    collected_whole(state_no_sim, query, "signals")


def test_the_documents_open_in_ltspice(asc_state: SessionState, asc_file: Path, tmp_path: Path):
    designs = {str(asc_file.with_name(f"deck{n}.net")): "* deck\n" for n in range(4)}
    put_windows(asc_state, tmp_path / "world.json", [a_window(designs)])
    collected_whole(asc_state, {"kind": "open_in_ltspice"}, "designs")
