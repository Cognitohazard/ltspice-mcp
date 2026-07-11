"""Cross-handler CONTRACT tests — a parametrized variant matrix that closes
the bug class where a fix lands for one variant (one status, one job store,
one dispatch fork, one file extension) while sibling variants stay broken.

Invariants asserted through the REAL handlers (not internals), so a fix that
lands in a leaf without wiring the entry path fails here:

1. STATUS COMPLETENESS — a terminal batch status (incl. ``interrupted``, assigned
   on restart recovery) must format without raising "unexpected status". This is
   the exact shape of the interrupted-status formatter bug (a hardcoded status
   allowlist that omitted ``interrupted``); parametrizing over every terminal
   status catches the whole class, not the one instance.

2. DUAL-STORE RESOLUTION — a handler that takes a ``job_id`` must resolve BOTH a
   single-sim job (``state.jobs``) and a batch job (``state.batch_jobs``), never
   reject one store's ids as "not found". This is the shape of the cancel_job
   bug (it resolved only the single-sim store).

3. ROUTING-FORK COVERAGE — a handler that forks on job type must reach EVERY
   fork: cancel_job's batch fork has runner-routing tests elsewhere, but the
   single-sim fork was reachable only past guards no unit test crossed.

4. DUAL-DISPATCH (.cir vs .asc) — a circuit tool that accepts both extensions
   must work through BOTH dispatch branches (spice_lex pipeline vs AscEditor),
   and a write must persist to disk, not just to a cached editor.
"""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import BatchJobError, ResultError
from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    SessionState,
    SimulationJob,
)
from ltspice_mcp.tools.advanced import GetBatchResultsInput, handle_batch_results
from ltspice_mcp.tools.analysis import MeasurementStatsInput, handle_measurement_stats
from ltspice_mcp.tools.circuit import (
    CircuitReadInput,
    ListComponentsInput,
    SetComponentValueInput,
    handle_list_components,
    handle_read_circuit,
    handle_set_component_value,
)
from ltspice_mcp.tools.simulation import (
    CancelJobInput,
    CheckJobInput,
    handle_cancel_job,
    handle_check_job,
)
from tests.conftest import (
    FIXTURES_DIR,
    LTSPICE_SWEEP_RUN_LOGS,
    LTSPICE_TRAN_RC_LOG,
    LTSPICE_TRAN_RC_VFINAL,
    make_batch_job,
    make_sim_job,
)

pytestmark = pytest.mark.asyncio

_FIXTURE_DRAFT = FIXTURES_DIR / "Draft1.asc"

# Terminal statuses a BatchJob can actually hold: TERMINAL_STATUSES minus
# 'timeout' (single-sim only, not in BatchJob.status' Literal). Derived so a new
# terminal status auto-propagates into this contract instead of escaping it.
BATCH_TERMINAL_STATUSES = sorted(TERMINAL_STATUSES - {"timeout"})

# Single-sim terminal statuses check_job can format WITHOUT result files on disk.
# 'completed' needs a real raw/log (→ covered end-to-end by the ngspice e2e tier),
# so it's excluded here. Derived so a new terminal status is caught automatically.
SINGLE_TERMINAL_STATUSES_NO_FILES = sorted(TERMINAL_STATUSES - {"completed"})


def _text(result) -> str:
    item = result.content[0]
    assert isinstance(item, types.TextContent)
    return item.text


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _make_batch(state: SessionState, *, status: str, job_id: str = "b1") -> BatchJob:
    bj = make_batch_job(job_id, status=status, total_runs=4, completed_runs=2)
    state.add_batch_job(bj)
    return bj


def _make_sim(
    state: SessionState,
    *,
    status: str,
    job_id: str = "j1",
    log_file: Path | None = None,
) -> SimulationJob:
    job = make_sim_job(job_id, status=status, log_file=log_file)
    state.jobs[job_id] = job
    return job


class TestBatchStatusCompleteness:
    """batch_results must format EVERY terminal batch status without raising
    'unexpected status'. Regression class: the interrupted-status formatter
    bug — a hardcoded status allowlist omitted a real terminal status."""

    @pytest.mark.parametrize("status", BATCH_TERMINAL_STATUSES)
    async def test_batch_results_handles_terminal_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_batch(state_no_sim, status=status)
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        text = _text(result).lower()
        assert "unexpected status" not in text
        # the status is surfaced to the caller (not swallowed)
        assert status in text


class TestSingleSimStatusCompleteness:
    """check_job must format EVERY terminal single-sim status without falling
    through to 'unexpected status'. Same class as the batch interrupted-status
    formatter bug, one store over: the single-sim formatter omitted
    'interrupted' (assigned on restart recovery, e.g. a job whose raw didn't
    survive to be promoted to completed)."""

    @pytest.mark.parametrize("status", SINGLE_TERMINAL_STATUSES_NO_FILES)
    async def test_check_job_handles_terminal_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_sim(state_no_sim, status=status)
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "unexpected status" not in _text(result).lower()


class TestJobHandlerDualStore:
    """A job_id-taking handler must resolve BOTH job stores. Regression class:
    cancel_job resolved only the single-sim store and rejected batch ids."""

    async def test_check_job_resolves_single_sim_job(self, state_no_sim: SessionState):
        _make_sim(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "not found" not in _text(result).lower()

    async def test_check_job_resolves_batch_job(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(job_id="b1"), state_no_sim)
        assert "not found" not in _text(result).lower()

    async def test_cancel_job_resolves_single_sim_job(self, state_with_sim: SessionState):
        _make_sim(state_with_sim, status="running")
        fake_runner = MagicMock(cancel=AsyncMock())
        with patch("ltspice_mcp.tools.simulation._get_or_create_runner", return_value=fake_runner):
            result = await handle_cancel_job(CancelJobInput(job_id="j1"), state_with_sim)
        assert "not found" not in _text(result).lower()

    async def test_cancel_job_resolves_batch_job(self, state_with_sim: SessionState):
        _make_batch(state_with_sim, status="running")
        fake_runner = MagicMock(cancel=AsyncMock())
        with patch.object(
            state_with_sim.runners, "get_batch_runner_for", return_value=fake_runner
        ):
            result = await handle_cancel_job(CancelJobInput(job_id="b1"), state_with_sim)
        assert "not found" not in _text(result).lower()

    async def test_measurement_stats_aggregates_batch_job(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, status="completed")
        for i, log in enumerate(LTSPICE_SWEEP_RUN_LOGS):
            bj.run_results[i] = {"log_file": str(log), "params": {"R1": float(i)}}
        result = await handle_measurement_stats(MeasurementStatsInput(job_id="b1"), state_no_sim)
        assert result.structuredContent is not None
        entry = result.structuredContent["stats"]["vfinal"]
        assert entry["total_count"] == 3
        assert entry["valid_count"] == 3

    async def test_measurement_stats_aggregates_single_sim_job(self, state_no_sim: SessionState):
        _make_sim(state_no_sim, status="completed", log_file=LTSPICE_TRAN_RC_LOG)
        result = await handle_measurement_stats(MeasurementStatsInput(job_id="j1"), state_no_sim)
        assert result.structuredContent is not None
        entry = result.structuredContent["stats"]["vfinal"]
        assert entry["total_count"] == 1
        assert entry["valid_count"] == 1
        assert entry["mean"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)

    async def test_measurement_stats_unknown_id_errors_not_found(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="Job not found: ghost"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="ghost"), state_no_sim)

    async def test_measurement_stats_running_single_sim_errors_not_completed(
        self, state_no_sim: SessionState
    ):
        _make_sim(state_no_sim, status="running")
        with pytest.raises(ResultError, match="is not completed"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="j1"), state_no_sim)


class TestCrossTypeRedirects:
    """A job id of the OTHER run type must get an honest redirect naming the
    right tool — never "not found" for a job that exists in the store."""

    async def test_batch_results_single_sim_id_redirects(self, state_no_sim: SessionState):
        _make_sim(state_no_sim, status="completed", log_file=LTSPICE_TRAN_RC_LOG)
        with pytest.raises(BatchJobError) as exc:
            await handle_batch_results(GetBatchResultsInput(job_id="j1"), state_no_sim)
        msg = str(exc.value)
        assert "single simulation job" in msg
        # Redirect names only job-id-accepting tools, never simulation_summary.
        assert "check_job" in msg
        assert "simulation_summary" not in msg


class TestCancelJobRoutingFork:
    """cancel_job forks on job type (BatchJob → batch runner, SimulationJob →
    single-sim runner). The batch fork has runner-routing tests elsewhere; the
    single-sim fork sat behind guards (unknown id, not running, no simulator)
    that every prior unit test stopped at, so it was never exercised. These
    tests cross the guards and pin the routing: the live job object itself must
    be handed to the single-sim runner's cancel."""

    @pytest.mark.parametrize("status", sorted(NON_TERMINAL_LIVE_STATUSES))
    async def test_live_single_sim_routes_to_sim_runner(
        self, status: str, state_with_sim: SessionState
    ):
        job = _make_sim(state_with_sim, status=status)
        fake_runner = MagicMock(cancel=AsyncMock())
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner", return_value=fake_runner
        ) as get_runner:
            result = await handle_cancel_job(CancelJobInput(job_id="j1"), state_with_sim)
        assert "cancelled" in _text(result).lower()
        # Resolved via the job's own netlist so the runner's output folder matches
        # the one the job launched with. simulator_class is None here: the
        # job's recorded name ("ltspice") matches no detected class, so the
        # runner falls back to the session default.
        get_runner.assert_called_once_with(state_with_sim, job.netlist, simulator_class=None)
        fake_runner.cancel.assert_awaited_once()
        # The exact job resolved from state.jobs reaches the runner —
        # not a re-looked-up copy, not a batch-runner detour.
        assert fake_runner.cancel.await_args is not None
        assert fake_runner.cancel.await_args.args[0] is job

    async def test_single_sim_fork_does_not_touch_batch_runners(
        self, state_with_sim: SessionState
    ):
        _make_sim(state_with_sim, status="running")
        fake_runner = MagicMock(cancel=AsyncMock())
        with (
            patch(
                "ltspice_mcp.tools.simulation._get_or_create_runner",
                return_value=fake_runner,
            ),
            patch.object(state_with_sim.runners, "get_batch_runner_for") as batch,
        ):
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_with_sim)
        batch.assert_not_called()
        fake_runner.cancel.assert_awaited_once()


# --- DUAL-DISPATCH (.cir vs .asc) -----------------------------------------

# Both files contain a resistor R1 with value 1k, so the same assertions run
# against both dispatch branches (spice_lex pipeline vs AscEditor).
_CIR_NETLIST = "* RC filter\nR1 in out 1k\nC1 out 0 100n\nV1 in 0 1\n.op\n.end\n"


@pytest.fixture
def circuit_file(request: pytest.FixtureRequest, work_dir: Path) -> Path:
    """Circuit file of the parametrized extension inside the allowed dir.

    ``.asc`` copies the Draft1 fixture (R1=1k) and pulls in the session-scoped
    symbol cache so AscEditor can resolve its symbols; ``.cir`` writes an
    equivalent netlist with the same R1=1k.
    """
    ext = request.param
    if ext == "asc":
        request.getfixturevalue("asc_symbols")
        dest = work_dir / "Draft1.asc"
        shutil.copy(_FIXTURE_DRAFT, dest)
        return dest
    path = work_dir / "rc_filter.cir"
    path.write_text(_CIR_NETLIST)
    return path


@pytest.mark.parametrize("circuit_file", ["cir", "asc"], indirect=True)
class TestCircuitToolDualDispatch:
    """Circuit tools that accept both .cir and .asc must work through BOTH
    extension-dispatch branches. The .cir branch (spice_lex pipeline) had unit
    coverage; the .asc branch (AscEditor) was only reachable through separate
    .asc-specific tests, so a regression in the shared dispatch seam — or an
    .asc write that mutates only the cached editor — would escape. Writes are
    verified against raw disk bytes AND re-read through a fresh SessionState
    (fresh editor cache) so the value provably comes from disk."""

    async def test_read_circuit_surfaces_known_component(
        self, circuit_file: Path, state_no_sim: SessionState
    ):
        result = await handle_read_circuit(CircuitReadInput(path=str(circuit_file)), state_no_sim)
        text = _text(result)
        assert "R1" in text
        assert "1k" in text

    async def test_list_components_finds_reference(
        self, circuit_file: Path, state_no_sim: SessionState
    ):
        result = await handle_list_components(
            ListComponentsInput(path=str(circuit_file), reference="R1"), state_no_sim
        )
        assert _text(result) == "R1 = 1k"
        assert result.structuredContent == {"reference": "R1", "value": "1k"}

    async def test_set_component_value_persists_to_disk(
        self, circuit_file: Path, state_no_sim: SessionState, config: ServerConfig
    ):
        result = await handle_set_component_value(
            SetComponentValueInput(path=str(circuit_file), reference="R1", value="4.7k"),
            state_no_sim,
        )
        assert "4.7k" in _text(result)

        # The new value reached the file itself, not just an in-memory editor.
        # (bytes, not text: Draft1.asc carries a non-UTF-8 µ byte)
        assert b"4.7k" in _read_bytes(circuit_file)

        # Re-read through the real read path with a FRESH SessionState so a
        # warm editor cache can't fake persistence.
        fresh_state = SessionState.create(config, available={})
        reread = await handle_list_components(
            ListComponentsInput(path=str(circuit_file), reference="R1"), fresh_state
        )
        assert _text(reread) == "R1 = 4.7k"
        assert reread.structuredContent == {"reference": "R1", "value": "4.7k"}


@pytest.mark.asyncio
class TestCheckJobOutputSchemaContract:
    """5. SCHEMA HONESTY — ``check_job``'s structuredContent must validate
    against its own declared output_schema for every job shape. Emitting
    ``"error": null`` (schema types it as non-nullable string) made every
    schema-validating MCP client — including the official python SDK —
    raise on every batch-job poll, running or completed.
    """

    @classmethod
    def _validator(cls):
        # Compiled once per class — jsonschema re-checks the metaschema on
        # every plain validate() call otherwise.
        import jsonschema

        from ltspice_mcp.tools import get_tools_for_profile

        cached = getattr(cls, "_cached_validator", None)
        if cached is None:
            tool_defs, _ = get_tools_for_profile("full")
            tool = next(t for t in tool_defs if t.name == "check_job")
            assert tool.outputSchema is not None
            cached = jsonschema.Draft202012Validator(tool.outputSchema)
            cls._cached_validator = cached
        return cached

    async def _validated(self, job_id: str, state: SessionState) -> dict:
        # The conformance hook in conftest also validates this emission;
        # the explicit check here is the named, hook-independent contract.
        result = await handle_check_job(CheckJobInput(job_id=job_id, format="json"), state)
        assert result.structuredContent is not None
        errors = list(self._validator().iter_errors(result.structuredContent))
        assert not errors, [e.message for e in errors]
        return result.structuredContent

    async def test_running_batch_job_validates(self, state_no_sim: SessionState):
        bj = make_batch_job("b1", status="running")
        state_no_sim.batch_jobs[bj.job_id] = bj
        data = await self._validated("b1", state_no_sim)
        # No error yet -> the key is omitted, not emitted as null.
        assert "error" not in data
        assert data["job_type"] == "sweep"

    async def test_completed_batch_job_validates(self, state_no_sim: SessionState):
        bj = make_batch_job("b2", status="completed", completed_runs=2)
        state_no_sim.batch_jobs[bj.job_id] = bj
        data = await self._validated("b2", state_no_sim)
        assert "error" not in data
        assert data["completed_runs"] == 2

    async def test_failed_batch_job_validates_with_error(self, state_no_sim: SessionState):
        bj = make_batch_job("b3", status="failed", error="sweep execution failed")
        state_no_sim.batch_jobs[bj.job_id] = bj
        data = await self._validated("b3", state_no_sim)
        assert data["error"] == "sweep execution failed"

    async def test_failed_single_job_without_error_text_validates(
        self, state_no_sim: SessionState
    ):
        # A failed job whose error was never populated must still emit a
        # string (the schema forbids null).
        job = make_sim_job("s1", status="failed", error=None)
        state_no_sim.jobs[job.job_id] = job
        data = await self._validated("s1", state_no_sim)
        assert data["error"] == "Unknown error"
