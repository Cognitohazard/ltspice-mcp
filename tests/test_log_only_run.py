"""Recorded native output completes through the coordinator and public readers."""

import asyncio
import json
import subprocess
import threading
from pathlib import Path

import pytest
from spicelib.sim.run_task import RunTask
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_runner import ExperimentRunner, ExperimentRunRequest
from ltspice_mcp.lib.experiment_types import ExperimentCase, ManifestEntry, SourceRecord
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools
from tests.conftest import FIXTURES_DIR, LIVENESS_S, staged_decks


async def test_log_only_tf_run_is_produced_reloaded_and_publicly_readable(
    state_no_sim, work_dir, monkeypatch
):
    launch = subprocess.Popen
    parser_launches = []

    def parser_only(command, *args, **kwargs):
        assert isinstance(command, list)
        assert Path(command[0]).name.casefold().startswith("python")
        assert command[1:3] == ["-m", "ltspice_mcp.lib.parser_bootstrap"]
        parser_launches.append(command)
        return launch(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", parser_only)
    fixture = (FIXTURES_DIR / "native_log_tables" / "tf_table.log").read_bytes()
    simulations = []

    def recorded_run(cls, netlist, switches, timeout, *, cwd=None, exe_log=False):
        task = threading.current_thread()
        assert isinstance(task, RunTask)
        copied = Path(netlist)
        assert b".tf V(out) V1" in copied.read_bytes()
        copied.with_suffix(".log").write_bytes(fixture)
        simulations.append((copied, task))
        return 0

    monkeypatch.setattr(NGspiceSimulator, "run", classmethod(recorded_run))
    deck = work_dir / "divider.cir"
    deck.write_text(
        "Divider transfer\nV1 in 0 1\nR1 in out 1k\nR2 out 0 2k\n.tf V(out) V1\n.end\n",
        encoding="ascii",
    )
    digest = sha256_file(deck)
    case = ExperimentCase(
        case_id="case_0000",
        run_index=0,
        circuit="divider",
        circuit_path=deck,
        staged_deck=deck,
        deck_sha256=digest,
    )
    source = SourceRecord(
        circuit="divider",
        path=deck,
        sha256=digest,
        staged_deck=deck,
        manifest=[ManifestEntry(deck, digest, staged=True, live=False, staged_path=deck)],
        simulator="NGspiceSimulator",
        dialect="ngspice",
    )
    runner = ExperimentRunner(
        asyncio.get_running_loop(), NGspiceSimulator, state_no_sim.store.runs_root(), 1
    )
    request = ExperimentRunRequest(
        state=state_no_sim,
        request_id="recorded-tf-log-only",
        fingerprint=digest,
        stage=staged_decks([case], [source]),
        simulator="NGspiceSimulator",
        max_parallel=1,
        job_id="exp_log_only",
    )
    fresh = None
    try:
        receipt = await asyncio.wait_for(asyncio.shield(runner.submit(request)), LIVENESS_S)
        job = receipt.job
        assert await runner.wait(job, timeout_s=30)
        task = state_no_sim.job_registry.live[job.job_id].task
        assert task is not None
        await task
        await state_no_sim.job_registry.drain_pending()
        assert job.status == "completed", job.failures
        assert case.status == "produced"
        assert case.raw_file is None
        assert case.log_file is not None
        assert case.log_file.read_bytes() == fixture
        assert job.completeness.submitted == job.completeness.produced == 1
        assert (
            job.completeness.failed == job.completeness.cancelled == job.completeness.skipped == 0
        )
        assert job.recovery is None
        assert len(simulations) == 1
        copied, task = simulations[0]
        assert copied.parent == job.output_folder
        assert copied != deck
        assert not copied.with_suffix(".raw").exists()
        await asyncio.to_thread(task.join, LIVENESS_S)
        assert not task.is_alive()
        assert task.retcode == 0
        observation = next(
            item for item in case.observations if item["code"] == "native_log_output"
        )
        assert observation["evidence"]["analysis_extent"] == "unknown"

        stored = json.loads(job.store_path.read_text(encoding="utf-8"))
        assert stored["cases"][0]["status"] == "produced"
        assert stored["cases"][0]["raw_file"] is None
        assert stored["cases"][0]["log_file"] == str(case.log_file)
        fresh = SessionState.create(state_no_sim.config, available={})
        assert not fresh.all_jobs
        loaded = await services.resolve_job_async(job.job_id, fresh)
        assert loaded is not job
        assert loaded.status == "completed"
        assert loaded.cases[0].status == "produced"
        assert loaded.cases[0].raw_file is None
        run = services.resolve_experiment_run(job.job_id, fresh, require_raw=False)
        assert run.raw is None
        assert run.log == case.log_file
        assert run.console == case.log_file.with_suffix(".exe.log")
        with pytest.raises(ResultError, match="did not produce a raw result"):
            services.resolve_experiment_run(job.job_id, fresh)

        reply = await inspect_tools.handle_inspect(
            inspect_tools.InspectInput.model_validate(
                {"queries": [{"kind": "results", "view": "native_tables", "job_id": job.job_id}]}
            ),
            fresh,
        )
        assert reply.structured_content is not None
        result = reply.structured_content["results"][0]
        assert result["ok"], result
        data = result["data"]
        assert data["capture_facts"]["absent"] == ["raw", "console"]
        assert data["section"]["status"] == "parsed"
        rows = data["native_tables"]
        assert [(row["entry"]["label"], row["entry"]["real"]) for row in rows] == [
            ("transfer_function", 2 / 3),
            ("output_impedance_at_v(out)", 2000 / 3),
            ("v1#input_impedance", 3000.0),
        ]
        assert all(row["analysis_extent"] == "unknown" for row in rows)
        assert parser_launches
        assert not list((fresh.store.root / "parsing").iterdir())
    finally:
        if fresh is not None:
            await fresh.shutdown()
        await state_no_sim.shutdown()
