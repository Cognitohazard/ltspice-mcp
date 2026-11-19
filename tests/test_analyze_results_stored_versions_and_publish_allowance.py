"""What a reply says of a stored analysis of another version and of
sources the bound left unchecked, and how long a publish may take.

- ``analysis_snapshot.classify`` alone decides whether a stored analysis is
  rendered, and says earlier, later or damaged as the record is.
- Rows a recipe answered are never withdrawn because the bound ended the
  check after the reads; the reply names the sources not checked again.
- The publish step of a whole call has one allowance; past it, comparisons
  are skipped and the new files replace the old. Each recipe digests the
  files it writes as it writes them, so publishing reads no new file.
- One place says how many entries come after the ones held, and one writes
  the words for entries a reply did not send.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded
from ltspice_mcp.lib import analysis_snapshot, result_store, services
from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT, STORE_VERSION
from ltspice_mcp.lib.store import envelope as store_envelope
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import stage_recorded_fixture
from tests.test_analyze_results_call_bound import _CSV
from tests.test_analyze_results_cut_replies import _EARLIER_RELEASE_ANALYSIS
from tests.test_analyze_results_through_the_presenter import _analyze

_ANSWER = {"outcome": "complete", "results": {}, "observations": [], "failures": []}

# ---------------------------------------------------------------------------
# Earlier, later or damaged, decided in one place
# ---------------------------------------------------------------------------


def _record(**fields: Any) -> dict[str, Any]:
    return store_envelope(KIND_ANALYSIS_SNAPSHOT, **fields)


@pytest.mark.parametrize(
    ("record", "verdict", "words"),
    [
        (_EARLIER_RELEASE_ANALYSIS, "earlier", "earlier release"),
        (
            _record(answer_version=analysis_snapshot.ANSWER_VERSION + 1, answer=_ANSWER),
            "later",
            "later release",
        ),
        (
            {**_record(answer=_ANSWER), "store_version": STORE_VERSION + 1},
            "later",
            "later release",
        ),
        (
            # The release before's shape, at this version: no answer to render.
            _record(answer_version=analysis_snapshot.ANSWER_VERSION, top={}, results={}),
            "damaged",
            "damaged",
        ),
    ],
)
def test_a_stored_analysis_is_called_earlier_later_or_damaged_as_it_is(
    record: dict[str, Any], verdict: str, words: str
):
    assert analysis_snapshot.classify(record) == verdict
    with pytest.raises(analyze_mod.AttachedAnalysisNotRendered, match=words):
        analyze_mod.stored_analysis(record)


def test_the_renderer_asks_classify_and_decides_nothing_itself():
    current = analysis_snapshot.envelope({"answer": _ANSWER, "snapshot": {}, "request": {}})
    assert analysis_snapshot.classify(current) == "snapshot"
    assert analyze_mod.stored_analysis(current).answer == _ANSWER
    source = Path(analyze_mod.__file__).read_text(encoding="utf-8")
    body = source[source.index("def stored_analysis(") :]
    body = body[: body.index("\ndef ")]
    assert 'stored.get("answer")' not in body.split("raise")[0], "it decides on the shape"


# ---------------------------------------------------------------------------
# The check after the reads, ended by the bound, withdraws nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answered_rows_stay_when_the_bound_ends_the_check_after_the_reads(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    real_load = services.load_artifacts
    real_verify = analyze_mod._verify_direct_sources
    checks: list[int] = []

    async def out_of_time(source: Any, state: Any, **kwargs: Any) -> Any:
        raise AnalysisDeadlineExceeded("the check after the reads ran out of time")

    async def verify(*args: Any, **kwargs: Any) -> Any:
        checks.append(1)
        if len(checks) == 1:  # the check before the reads
            return await real_verify(*args, **kwargs)
        # The check after the reads: the real one, with no time left to read.
        monkeypatch.setattr(services, "load_artifacts", out_of_time)
        try:
            return await real_verify(*args, **kwargs)
        finally:
            monkeypatch.setattr(services, "load_artifacts", real_load)

    monkeypatch.setattr(analyze_mod, "_verify_direct_sources", verify)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "500u"}],
        },
    )
    assert data["results"]["v"]["values"], "the answered rows are kept"
    assert "failures" not in data
    (note,) = [o for o in data["observations"] if o["code"] == "sources_not_rechecked"]
    assert "dut" in note["detail"]


# ---------------------------------------------------------------------------
# One allowance for the publish step of a call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_past_the_publish_allowance_comparisons_are_skipped_and_files_replaced(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [_CSV]}
    first = await _analyze(state_no_sim, request)
    path = Path(first["results"]["csv"]["values"][0]["value"]["artifact"]["path"])
    before = (await asyncio.to_thread(path.stat)).st_ino
    monkeypatch.setattr(analyze_mod, "PUBLISH_ALLOWANCE_S", 0.0)
    real = services.bounded_parse
    compared: list[Path] = []

    async def recorded(file: Path, *args: Any, **kwargs: Any) -> Any:
        if file == path:
            compared.append(file)
        return await real(file, *args, **kwargs)

    monkeypatch.setattr(services, "bounded_parse", recorded)
    again = await _analyze(state_no_sim, request)
    assert compared == [], "no comparison is made past the allowance"
    artifact = again["results"]["csv"]["values"][0]["value"]["artifact"]
    assert artifact == first["results"]["csv"]["values"][0]["value"]["artifact"]
    assert (await asyncio.to_thread(path.stat)).st_ino != before, "the new file replaced the old"


@pytest.mark.asyncio
async def test_publishing_reads_no_new_file_its_digest_was_taken_as_it_was_written(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Each recipe digests the file it writes, under its own deadline, so the
    publish step reads none of the new files: past its allowance it reads no
    file at all, and every artifact still carries its digest."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    monkeypatch.setattr(analyze_mod, "PUBLISH_ALLOWANCE_S", 0.0)
    real_parse = services.bounded_parse
    real_publish = analyze_mod._publish
    publishing: list[int] = []
    read_while_publishing: list[Path] = []

    async def recorded(file: Path, *args: Any, **kwargs: Any) -> Any:
        if publishing:
            read_while_publishing.append(file)
        return await real_parse(file, *args, **kwargs)

    async def publish(pending: Any) -> Any:
        publishing.append(1)
        try:
            return await real_publish(pending)
        finally:
            publishing.pop()

    monkeypatch.setattr(services, "bounded_parse", recorded)
    monkeypatch.setattr(analyze_mod, "_publish", publish)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{**_CSV, "key": "first"}, _CSV],
        },
    )
    assert read_while_publishing == [], "the publish step read a file"
    for key in ("first", "csv"):
        artifact = data["results"][key]["values"][0]["value"]["artifact"]
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        path = Path(artifact["path"])
        assert (await asyncio.to_thread(path.stat)).st_size == artifact["bytes"]
        digest = await asyncio.to_thread(result_store.sha256_file, path)
        assert digest == artifact["sha256"]


# ---------------------------------------------------------------------------
# One answer each
# ---------------------------------------------------------------------------


def test_one_place_counts_entries_left_out_and_one_writes_not_sent():
    tools = Path(analyze_mod.__file__).parent
    for module in tools.glob("*.py"):
        text = module.read_text(encoding="utf-8")
        assert "offset + entry.returned >=" not in text, module.name
        assert "offset + entry.returned <" not in text, module.name
        assert not re.search(r"[Nn]ot sent[:,]", text), module.name
