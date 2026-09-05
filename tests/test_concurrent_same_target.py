"""What happens when two tool calls hit the same target at the same time.

A host may fire several tool calls in one assistant turn, and the MCP SDK
dispatches each as its own task on one event loop. That puts two requests on
the same schematic, the same ``request_id``, the same job or the same sidecar
netlist simultaneously — a case the turn-at-a-time assumption never produced.
Each test here fires the real dispatch entry twice with ``asyncio.gather`` and
asserts the outcome is one of the two orderings, never a torn or duplicated
one.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FakeSim,
    fake_artifact_paths,
    fake_simulator,
    recorded_fixture_simulator,
)

pytestmark = pytest.mark.asyncio


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_values(asc_path: Path) -> set[str]:
    """Every ``SYMATTR Value`` on a sheet, as written."""
    return {
        line.split(maxsplit=2)[2]
        for line in asc_path.read_text(encoding="cp1252").splitlines()
        if line.startswith("SYMATTR Value")
    }


async def _call(state: SessionState, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke one tool through the dispatch map the server itself uses."""
    registered = state.tool_dispatch[tool]
    result = await registered.handler(payload, state)
    assert result.structured_content is not None, result.content
    return result.structured_content


def _deck(path: Path) -> Path:
    path.write_text("V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    return path


def _run_payload(deck: Path, request_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 5},
        "lint": "off",
    }
    payload.update(overrides)
    return payload


def _job_records(work_dir: Path) -> list[Path]:
    return sorted(Store(work_dir).experiments_dir.glob("exp_*.json"))


def _staged_deck_dirs(work_dir: Path) -> list[Path]:
    """Every ``runs/{job_id}/staged`` directory this working dir produced."""
    runs = Store(work_dir).runs_root()
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.glob("*/staged") if p.is_dir())


def _spy_on_staging(
    monkeypatch: pytest.MonkeyPatch, entered: asyncio.Event | None = None
) -> list[Path]:
    """Record the staging root of every deck set a submission actually stages.

    The list is the whole history, not the directories left at the end: a
    duplicate that stages and is then cleaned up still shows up here.

    ``entered`` fires as the first deck set begins staging. Staging happens
    inside the submission's request gate, so a caller that waits for it before
    submitting is provably queued on that gate rather than arriving whenever
    the scheduler ran it. The spy runs in a worker thread, hence the
    loop-thread hand-off.
    """
    from ltspice_mcp.tools import experiments as experiments_module

    real = experiments_module.stage_deck
    staged: list[Path] = []
    loop = asyncio.get_running_loop() if entered is not None else None

    def spy(source_path: Path, staging_root: Path, *args: Any, **kwargs: Any):
        staged.append(Path(staging_root))
        if loop is not None and entered is not None:
            loop.call_soon_threadsafe(entered.set)
        return real(source_path, staging_root, *args, **kwargs)

    monkeypatch.setattr(experiments_module, "stage_deck", spy)
    return staged


# ---------------------------------------------------------------------------
# 1. Two edit_schematic batches carrying the same expected_sha256
# ---------------------------------------------------------------------------


async def test_simultaneous_edits_on_one_sha_commit_exactly_one(
    asc_state: SessionState, asc_file: Path
):
    """One batch commits; the other is refused with the file's NEW digest.

    Both callers read the sheet in the same turn, so both quote the same
    ``expected_sha256``. The loser must not overwrite the winner, and its
    refusal has to carry the digest the winner produced — otherwise the retry
    needs another read to find out what to quote.
    """
    before = _sha(asc_file)

    first, second = await asyncio.gather(
        _call(
            asc_state,
            "edit_schematic",
            {
                "target": str(asc_file),
                "expected_sha256": before,
                "ops": [{"op": "set_component_value", "reference": "R1", "value": "2k"}],
            },
        ),
        _call(
            asc_state,
            "edit_schematic",
            {
                "target": str(asc_file),
                "expected_sha256": before,
                "ops": [{"op": "set_component_value", "reference": "C1", "value": "2u"}],
            },
        ),
    )

    committed = [d for d in (first, second) if d["commit_state"] == "committed"]
    refused = [d for d in (first, second) if d["commit_state"] != "committed"]
    assert len(committed) == 1, "both batches committed onto one revision"
    assert len(refused) == 1

    after = _sha(asc_file)
    assert after != before
    assert committed[0]["sha256"] == after

    error = refused[0]["error"]
    assert error["code"] == "revision_conflict"
    assert refused[0]["outcome"] == "failed"
    # The refusal carries the digest the winner just wrote, so the retry needs
    # no extra read.
    assert refused[0]["sha256"] == after
    assert "re-read" in refused[0]["hint"].lower()

    # Exactly one batch is on disk: no interleaving of the two whole-file writes.
    values = _component_values(asc_file)
    assert ("2k" in values) ^ ("2u" in values), f"both edits landed on the sheet: {values}"


# ---------------------------------------------------------------------------
# 2. Two identical submissions under one request_id
# ---------------------------------------------------------------------------


async def test_identical_request_id_submits_one_job(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Same id, same payload, fired together: one job, one record, one deck set.

    The duplicate waits on the request gate and replays what it finds there, so
    only one submission ever stages: the deck set that exists is the one the
    surviving job claims, not the survivor of two that were staged.

    The duplicate is released once the first submission is staging, so it is
    provably inside the gate's queue while the first holds it. Which of the two
    responses carries the replay note is deliberately not asserted: the note is
    a durable fact on the shared job record, so every receipt rendered after the
    duplicate arrives carries it — including the original submitter's, whose
    dwell may still be running. What the record must not do is accumulate a
    second copy of it, which is asserted below.
    """
    submissions: list[str] = []
    fake_simulator(monkeypatch, submissions)
    staging_started = asyncio.Event()
    staged = _spy_on_staging(monkeypatch, staging_started)
    deck = _deck(work_dir / "same-request.cir")
    payload = _run_payload(deck, "same-request")

    async def duplicate() -> dict[str, Any]:
        await staging_started.wait()
        return await _call(state_with_sim, "run_experiments", dict(payload))

    first, second = await asyncio.gather(
        _call(state_with_sim, "run_experiments", dict(payload)),
        duplicate(),
    )

    assert first.get("error") is None, first.get("error")
    assert second.get("error") is None, second.get("error")
    assert first["job_id"] == second["job_id"]
    assert [p.stem for p in _job_records(work_dir)] == [first["job_id"]]
    # The replay says so rather than looking like a second run — and says it
    # once, however many callers read the record after it was noted.
    replayed = [
        d
        for d in (first, second)
        if any(o["code"] == "idempotent_replay" for o in d["observations"])
    ]
    assert replayed, "neither response reported the duplicate as a replay"
    record = state_with_sim.experiment_jobs[first["job_id"]]
    notes = [o for o in record.observations if o.get("code") == "idempotent_replay"]
    assert len(notes) == 1, f"the replay was noted {len(notes)} times on one record"
    assert len(submissions) == 1, f"the loser also reached the simulator: {submissions}"
    assert len(staged) == 1, f"both submissions staged a deck set: {staged}"
    assert first["job_id"] in staged[0].parts
    assert [p.parent.name for p in _staged_deck_dirs(work_dir)] == [first["job_id"]]


def _hold_request_lock(work_dir: str, request_id: str, held: Any, hold_s: float) -> None:
    """Subprocess helper: take the store's request gate and hold it briefly."""
    from ltspice_mcp.lib.filelock import file_lock
    from ltspice_mcp.lib.store import Store as _Store

    store = _Store(Path(work_dir))
    store.ensure_root()
    with file_lock(store.request_lock(request_id)):
        held.set()
        time.sleep(hold_s)


async def test_submission_waits_on_another_process_holding_the_request_gate(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The lookup-then-insert is gated across processes, not just on this loop.

    A second server process sharing the working directory is exactly the case
    an in-process lock cannot cover, so the gate is a real file lock: a
    submission must wait for a foreign holder rather than reading the index
    around it.
    """
    fake_simulator(monkeypatch)
    deck = _deck(work_dir / "gated.cir")
    hold_s = 1.0

    ctx = mp.get_context("spawn")
    held = ctx.Event()
    holder = ctx.Process(
        target=_hold_request_lock,
        args=(str(work_dir), "gated-request", held, hold_s),
    )
    holder.start()
    try:
        assert await asyncio.to_thread(held.wait, 30), "the lock holder never took the gate"

        started = time.monotonic()
        data = await _call(state_with_sim, "run_experiments", _run_payload(deck, "gated-request"))
        waited = time.monotonic() - started
    finally:
        holder.join(timeout=30)
        assert holder.exitcode == 0

    assert data.get("error") is None, data.get("error")
    assert waited >= hold_s * 0.5, (
        f"the submission did not wait on the foreign request gate ({waited:.2f}s)"
    )


# ---------------------------------------------------------------------------
# 3. One request_id, two different payloads
# ---------------------------------------------------------------------------


async def test_same_request_id_different_payload_conflicts(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """One submits, the other is refused as an idempotency conflict.

    The contract in docs/design/mcp_surface.md is fingerprint-keyed: the same
    id with a different canonical payload is ``idempotency_conflict``. Firing
    both at once must not turn that into two jobs under one id, and the refused
    payload must be refused before it stages anything: the conflict is decided
    inside the request gate, which the loser waits on.
    """
    submissions: list[str] = []
    fake_simulator(monkeypatch, submissions)
    staged = _spy_on_staging(monkeypatch)
    first_deck = _deck(work_dir / "conflict-a.cir")
    second_deck = _deck(work_dir / "conflict-b.cir")

    first, second = await asyncio.gather(
        _call(state_with_sim, "run_experiments", _run_payload(first_deck, "reused-id")),
        _call(state_with_sim, "run_experiments", _run_payload(second_deck, "reused-id")),
    )

    accepted = [d for d in (first, second) if d.get("error") is None]
    rejected = [d for d in (first, second) if d.get("error") is not None]
    assert len(accepted) == 1, "both payloads were accepted under one request_id"
    assert rejected[0]["error"]["code"] == "idempotency_conflict"
    assert rejected[0]["error"]["commit_state"] == "not_started"

    assert [p.stem for p in _job_records(work_dir)] == [accepted[0]["job_id"]]
    assert len(submissions) == 1, f"the refused payload still ran: {submissions}"
    # A refused submission never stages: staged decks belong to a record that
    # claims them, and the refusal happens before any deck is copied.
    assert len(staged) == 1, f"the refused payload staged a deck set: {staged}"
    assert accepted[0]["job_id"] in staged[0].parts
    assert [p.parent.name for p in _staged_deck_dirs(work_dir)] == [accepted[0]["job_id"]]


# ---------------------------------------------------------------------------
# 4. Two sidecar exports of one schematic
# ---------------------------------------------------------------------------


_EXPORT_LINES = ["V1 in 0 1\n", "R1 in 0 1k\n", ".op\n", ".end\n"]


class _ExportHandoffs:
    """The three hand-offs that sequence two exports of one schematic.

    Between one export releasing the export lock and the next one reopening the
    shared ``.net`` there is a window in which the file is empty. Left to the
    scheduler that window is microseconds wide, so a test of what an export
    reports about its own output passes or fails by luck. These events make the
    ordering explicit instead.
    """

    def __init__(self) -> None:
        # The first export has begun writing: the second caller may start, and
        # will queue on the export lock the first one holds.
        self.exporting = asyncio.Event()
        # The second export has reopened the .net and truncated it, so the
        # window is open right now.
        self.truncated = asyncio.Event()
        # The first caller has produced its response; the second export may
        # stop holding the file empty and write its lines.
        self.answered = threading.Event()


def _slow_exporter(state: SessionState, handoffs: _ExportHandoffs) -> None:
    """An exporter that writes its netlist in pieces, like the real subprocess.

    LTspice truncates ``<name>.net`` the moment it opens it and fills it over
    the life of the subprocess. Two unserialised exports of one schematic would
    interleave into a torn file, and an export that measures its own output
    after releasing the export lock reads whatever the next one has written so
    far. This stand-in reproduces both, and holds the second export at zero
    bytes until the first caller has answered so that "measured after the lock"
    yields an empty file rather than a partial one that may happen to be whole
    again. It runs in a worker thread, hence the loop-thread hand-offs.
    """
    loop = asyncio.get_running_loop()
    exports = itertools.count()

    class _Exporter:
        @staticmethod
        def create_netlist(path: str, timeout: float | None = None) -> str:
            index = next(exports)
            netlist = Path(path).with_suffix(".net")
            with netlist.open("w", encoding="utf-8") as handle:
                if index == 0:
                    loop.call_soon_threadsafe(handoffs.exporting.set)
                else:
                    loop.call_soon_threadsafe(handoffs.truncated.set)
                    assert handoffs.answered.wait(30), "the first export never answered"
                for line in _EXPORT_LINES:
                    handle.write(line)
                    handle.flush()
                    time.sleep(0.02)
            return str(netlist)

    state.available_simulators["ltspice"] = _Exporter


def _hold_first_export_open_past_its_lock(
    monkeypatch: pytest.MonkeyPatch, handoffs: _ExportHandoffs
) -> None:
    """Keep the real export lock and park the first export just past its exit.

    The first export continues only once a peer has actually truncated the
    ``.net``. Anything it still has left to read about its own output is then
    read inside that window every time, so where the export is measured stops
    being a scheduling accident and becomes the thing the test decides.
    """
    from ltspice_mcp.tools import verify as verify_module

    real = verify_module.asc_export_lock
    locks = itertools.count()

    @contextlib.asynccontextmanager
    async def spy(asc_path: Path):
        index = next(locks)
        async with real(asc_path):
            yield
        if index == 0:
            await asyncio.wait_for(handoffs.truncated.wait(), 30)

    monkeypatch.setattr(verify_module, "asc_export_lock", spy)


async def test_simultaneous_sidecar_exports_leave_one_complete_netlist(
    asc_state: SessionState, asc_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both exports succeed and each reports the whole netlist it wrote.

    The second call starts once the first is inside the exporter, so it is
    provably queued on the export lock while the first holds it, and the first
    call is held just past its lock release until the second has truncated the
    file. Both orderings are therefore fixed, and the digest assertion becomes a
    statement about where an export measures itself: read the file after the
    lock is released and the digest describes the peer's truncation rather than
    this export's own output, while ``ok`` still says the export succeeded.
    """
    handoffs = _ExportHandoffs()
    _slow_exporter(asc_state, handoffs)
    _hold_first_export_open_past_its_lock(monkeypatch, handoffs)
    net_path = asc_file.with_suffix(".net")

    payload = {
        "path": str(asc_file),
        "checks": ["export"],
        "export_to": "sidecar",
    }

    async def opener() -> dict[str, Any]:
        try:
            return await _call(asc_state, "verify_circuit", dict(payload))
        finally:
            handoffs.answered.set()

    async def contender() -> dict[str, Any]:
        await handoffs.exporting.wait()
        return await _call(asc_state, "verify_circuit", dict(payload))

    first, second = await asyncio.gather(opener(), contender())

    for data in (first, second):
        assert data["export"]["ok"] is True, data
        assert data["export"]["destination"] == "sidecar"

    complete = "".join(_EXPORT_LINES)
    assert net_path.read_text(encoding="utf-8") == complete
    whole = hashlib.sha256(complete.encode()).hexdigest()
    # Each response's digest describes a complete export, not a half-written one.
    assert {first["export"]["sha256"], second["export"]["sha256"]} == {whole}


# ---------------------------------------------------------------------------
# 5. Two analyses of one completed job
# ---------------------------------------------------------------------------


async def test_simultaneous_analyses_of_one_job_agree_and_parse_once(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both reads return the same numbers, and the shared raw is parsed once.

    ``FileCache`` is single-flight per path, so two cold readers of one raw
    must not both pay the parse — and neither may see a half-built value.
    """
    recorded_fixture_simulator(monkeypatch)
    deck = _deck(work_dir / "analyzed.cir")
    receipt = await _call(state_with_sim, "run_experiments", _run_payload(deck, "analyze-twice"))
    assert receipt.get("error") is None, receipt.get("error")
    assert receipt["status"] == "completed", receipt

    parses: list[str] = []
    original = OffsetAwareRawRead.__init__

    def counting_init(self, filename, *args, **kwargs):  # type: ignore[no-untyped-def]
        parses.append(str(filename))
        return original(self, filename, *args, **kwargs)

    monkeypatch.setattr(OffsetAwareRawRead, "__init__", counting_init)
    state_with_sim.results.clear()

    request = {
        "sources": [{"job_id": receipt["job_id"], "label": "nominal"}],
        "recipes": [{"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}],
    }
    left, right = await asyncio.gather(
        _call(state_with_sim, "analyze_results", dict(request)),
        _call(state_with_sim, "analyze_results", dict(request)),
    )

    assert left["results"]["vout"] == right["results"]["vout"]
    assert left["results"]["vout"]["values"], left["results"]["vout"]
    assert len(parses) == 1, f"the shared raw was parsed {len(parses)} times: {parses}"


# ---------------------------------------------------------------------------
# 6. Two cancels of one running job
# ---------------------------------------------------------------------------


async def test_simultaneous_cancels_of_one_job_report_one_outcome(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both cancels answer through the envelope; the job cancels exactly once.

    The case is parked where it waits for a launch permit, so it is provably
    still queued when the cancels arrive instead of however far the coordinator
    got before they did. That is what makes the receipts below decidable: with a
    queued case, the call that gets there first stops it and reports that
    transition, and the second finds nothing non-terminal left. Left to the
    scheduler the case may already be running, and then both calls see work to
    stop and both claim the same transition.
    """
    callbacks: dict[str, Any] = {}
    waiting_for_a_permit = asyncio.Event()
    # Never set. The case leaves this wait by being cancelled, which is the
    # state under test: a case stopped while queued never reaches a simulator.
    permit_granted = asyncio.Event()

    def submit(self, _netlist: Path, run_filename: str, callback):
        callbacks[run_filename] = callback
        return object()

    async def park_at_the_launch_permit(self) -> None:
        # A case takes this permit before it stamps itself submitted, so a case
        # parked here is exactly a queued one.
        waiting_for_a_permit.set()
        await permit_granted.wait()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    monkeypatch.setattr(ExperimentRunner, "acquire_launch_slot", park_at_the_launch_permit)
    deck = _deck(work_dir / "cancel-twice.cir")
    receipt = await _call(
        state_with_sim,
        "run_experiments",
        _run_payload(deck, "cancel-twice", execution={"wait_s": 0}),
    )
    assert receipt["outcome"] == "in_progress", receipt
    job_id = receipt["job_id"]
    token = receipt["control_token"]
    await asyncio.wait_for(waiting_for_a_permit.wait(), 30)

    cancel = {"action": "cancel", "job_id": job_id, "control_token": token}
    first, second = await asyncio.gather(
        _call(state_with_sim, "jobs", dict(cancel)),
        _call(state_with_sim, "jobs", dict(cancel)),
    )

    for data in (first, second):
        assert data["outcome"] != "error", data
        assert data["status"] == "cancelled", data
        assert data["failures"] == []
    job = state_with_sim.experiment_jobs[job_id]
    assert job.status == "cancelled"
    assert all(case.status == "cancelled" for case in job.cases)
    # A case is cancelled once. One call carries the transitions it made; the
    # other reports none, because by the time it looked there was nothing left
    # non-terminal to stop — never a second transition out of a terminal status.
    reporting = [data for data in (first, second) if data["items"]]
    assert len(reporting) == 1, f"both cancels claimed the same transitions: {reporting}"
    assert [row["prior_status"] for row in reporting[0]["items"]] == ["queued"]
    quiet = next(data for data in (first, second) if not data["items"])
    assert "no cancellation was needed" in quiet["hint"]
    assert not callbacks, f"a case stopped while queued still reached the simulator: {callbacks}"


# ---------------------------------------------------------------------------
# 7. Two jobs sharing a one-permit runner
# ---------------------------------------------------------------------------


async def test_two_jobs_share_the_single_launch_permit(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """With ``max_parallel_sims = 1`` both jobs are accepted; one waits its turn."""
    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
        max_parallel_sims=1,
    )
    state = SessionState.create(config, available={"fake": FakeSim})

    in_flight = 0
    peak = 0

    def submit(self, _netlist: Path, run_filename: str, callback):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        raw, log = fake_artifact_paths(self.output_folder, run_filename)

        def finish() -> None:
            nonlocal in_flight
            in_flight -= 1
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            callback(RunOutcome(str(raw), str(log), raw.stat().st_size, None))

        self.loop.call_later(0.15, finish)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    first_deck = _deck(work_dir / "queue-a.cir")
    second_deck = _deck(work_dir / "queue-b.cir")

    first, second = await asyncio.gather(
        _call(state, "run_experiments", _run_payload(first_deck, "queue-a")),
        _call(state, "run_experiments", _run_payload(second_deck, "queue-b")),
    )

    assert first["job_id"] != second["job_id"]
    for data in (first, second):
        assert data.get("error") is None, data.get("error")
        assert data["status"] == "completed", data
        assert data["completeness"]["produced"] == 1
    assert peak == 1, f"{peak} simulators ran at once under a one-permit runner"
