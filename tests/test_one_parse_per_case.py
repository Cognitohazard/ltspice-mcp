"""One ``analyze_results`` call parses each case's files once, whatever the
number of recipes.

The question is "has this content been parsed already", answered by the
session's result cache (``ResultCache``) and its index of source stamps, which
``parser_service.load_artifacts_sync`` consults before it starts a parser. The
count is of parser runs (``parser_service.run_parser_sync``), the one place a
case's files are read and parsed, so it is counted through the real code path.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import parser_service
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import dispatched

#: Twelve measurements of each recorded log, with the values master's reader
#: gives them: a log alone, and a log beside its results file.
_LOG_ONLY = (
    "log/meas_forms",
    {
        "m_max": 0.999955832958,
        "m_min": 0.0,
        "m_pp": 0.999955832958,
        "m_avg": 0.800332613503,
        "m_rms": 0.921750904874,
        "m_integ": 0.000899504419286,
        "m_find": 0.993323754378,
        "m_when": 7.01211869669e-05,
        "m_findwhen": 1.0,
        "m_trigtarg": 0.000219487521473,
        "m_param": 1.99991166592,
        "m_deriv": 4014.34898376,
    },
)
_WITH_RAW = (
    "log/meas_trig",
    {
        "m_cos": 3.81253878775e-06,
        "m_sin": -3.46275856263e-05,
        "m_sin_fo": -3.46275856263e-05,
        "m_sin_30": 0.5,
        "m_cos_pi": 0.998497149864,
        "m_tan_45": 1.0,
        "m_asin": 30.0,
        "m_acos": 60.0,
        "m_atan": 45.0,
        "m_atan2": 45.0,
        "m_sinh": 1.17520119364,
        "m_tanh": 0.761594155956,
    },
)

#: More cases than the cache holds entries, so a call whose recipes each read
#: every case in turn cannot lean on an entry surviving until its next recipe.
_CASES = 40


def _recipes(names: dict[str, float]) -> dict[str, Any]:
    return {
        "recipes": [{"key": name, "metric": "measurements", "names": [name]} for name in names],
        "include": {"per_run": {"limit": 100}, "fields": ["value"]},
    }


def _sources(work_dir: Path, case_id: str, cases: int, *, with_raw: bool) -> list[dict[str, Any]]:
    sources = []
    for index in range(cases):
        log = work_dir / f"case{index}.log"
        # A content of its own, which the cache keys on: the deck it names.
        recorded = rec.recorded("ltspice26", f"{case_id}.log").read_bytes()
        log.write_bytes(recorded.replace(b".cir", f"_{index}.cir".encode(), 1))
        assert log.read_bytes() != recorded
        path = log
        if with_raw:
            path = log.with_suffix(".raw")
            shutil.copyfile(rec.recorded("ltspice26", f"{case_id}.raw"), path)
        sources.append({"path": str(path), "label": f"c{index}"})
    return sources


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The sources of every parser run, in order.

    The files were written a moment ago, and a stamp is remembered only once
    the clock has moved past their times (``parser_service._settled``); the
    clock reads ten seconds on, as it would for a run that finished earlier.
    """
    monkeypatch.setattr(parser_service, "_now_ns", lambda: time.time_ns() + 10**10)
    made: list[dict[str, Any]] = []
    real = parser_service.run_parser_sync

    def counted(request: dict[str, Any], **kwargs: Any) -> Any:
        made.append(request["sources"])
        return real(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", counted)
    return made


def _values(reply: dict[str, Any], names: dict[str, float], cases: int) -> dict[str, list[Any]]:
    """Each recipe's values, of the rows the reply carries; it covers every case."""
    assert not reply.get("failures"), reply["failures"]
    assert reply["coverage"]["runs_analyzed"] == cases, reply["coverage"]
    rows = {name: reply["results"][name]["per_run"] for name in names}
    assert all(per_run["total"] == cases for per_run in rows.values()), rows
    return {
        name: [row["value"]["measured"][name] for row in per_run["items"]]
        for name, per_run in rows.items()
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "with_raw"),
    [(_LOG_ONLY, False), (_WITH_RAW, True)],
    ids=["log only", "with a results file"],
)
async def test_twelve_recipes_parse_each_case_once(
    case: tuple[str, dict[str, float]],
    with_raw: bool,
    state_no_sim: SessionState,
    work_dir: Path,
    parses: list[dict[str, Any]],
):
    case_id, names = case
    sources = _sources(work_dir, case_id, _CASES, with_raw=with_raw)
    values = _values(
        await dispatched(state_no_sim, "analyze_results", sources=sources, **_recipes(names)),
        names,
        _CASES,
    )
    assert {name: set(rows) for name, rows in values.items()} == {
        name: {value} for name, value in names.items()
    }
    # One parser run per case, which reads its log (and its results file).
    logs = sorted(str(made["log"]) for made in parses)
    assert logs == sorted(str(Path(s["path"]).with_suffix(".log")) for s in sources), len(logs)


@pytest.mark.asyncio
async def test_a_log_changed_between_calls_is_read_again(
    state_no_sim: SessionState, work_dir: Path, parses: list[dict[str, Any]]
):
    case_id, names = _LOG_ONLY
    sources = _sources(work_dir, case_id, 2, with_raw=False)
    recipes = _recipes(names)
    first = _values(
        await dispatched(state_no_sim, "analyze_results", sources=sources, **recipes), names, 2
    )
    changed = Path(sources[0]["path"])
    text = await asyncio.to_thread(changed.read_bytes)
    assert b"=0.999955832958 FROM" in text
    await asyncio.to_thread(
        changed.write_bytes, text.replace(b"=0.999955832958 FROM", b"=0.5 FROM", 1)
    )
    del parses[:]
    second = _values(
        await dispatched(state_no_sim, "analyze_results", sources=sources, **recipes), names, 2
    )
    assert str(changed) in [str(made["log"]) for made in parses]
    assert first["m_max"][0] != second["m_max"][0] and first["m_max"][1] == second["m_max"][1]
