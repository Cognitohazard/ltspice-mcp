"""Completion uses captured log facts, without launching a simulator."""

import asyncio
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from ltspice_mcp.lib import runner_base
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.log_decode import LogLimits, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from ltspice_mcp.lib.store import Store
from tests.conftest import FIXTURES_DIR, LIVENESS_S, job_done


def captured_completion_facts(root, log=None, *, text=None, console=None):
    """Exercise capture and worker decoding in a unique admitted directory."""
    directory = Store(root).parser_dir(uuid4().hex)
    directory.mkdir(parents=True)
    if text is not None:
        log = directory / "source.log"
        log.write_text(text, encoding="utf-8")
    console_path = None
    if console is not None:
        console_path = directory / "source.exe.log"
        console_path.write_text(console, encoding="utf-8")
    limits = LogLimits(2_000_000, 16_000, 20_000, 50_000, 2_000_000)
    captured = capture_inputs(
        SourceFiles(
            raw=FIXTURES_DIR / "ltspice_tran_rc.raw" if log is None and console is None else None,
            log=log,
            console=console_path,
        ),
        directory,
        input_bytes=limits.log_bytes,
        log_bytes=limits.log_bytes,
    )
    return DecodedLog(decode_logs(captured, directory, limits=limits))


@pytest.mark.parametrize("kind", ["tran", "ac", "dc", "op", "noise", "tf", "pz", "sens", "disto"])
def test_deck_snapshot_keeps_original_analyses_under_control(tmp_path, kind):
    include = tmp_path / "analysis.inc"
    include.write_text(f".{kind} witness\n.SAVE v(out)\n", encoding="ascii")
    deck = tmp_path / "test.cir"
    deck.write_text('.include "analysis.inc"\n.control\nrun\n.endc\n.end\n', encoding="ascii")

    requirements = runner_base.deck_requirements(deck)

    assert requirements.analyses == (f".{kind}",)
    assert requirements.has_save is True
    assert requirements.has_control is True
    assert requirements.raw_analyses == ()


def test_deck_snapshot_is_frozen_deduplicated_and_stops_at_end(tmp_path):
    deck = tmp_path / "test.cir"
    deck.write_text(".TRAN 1u 1m\n.tran 2u 2m\n.op\n.end\n.ac dec 10 1 10\n", encoding="ascii")
    requirements = runner_base.deck_requirements(deck)
    assert requirements.analyses == requirements.raw_analyses == (".tran", ".op")
    assert requirements.has_save is requirements.has_control is False
    with pytest.raises(AttributeError):
        requirements.has_control = True  # pyright: ignore[reportAttributeAccessIssue]
    assert runner_base.deck_requirements(None).analyses == ()


def test_classifier_uses_snapshot_after_source_deleted(tmp_path, monkeypatch):
    log = tmp_path / "run.fail"
    log.write_text("Circuit: x\nTime step too small; time = 1.7e-05\n", encoding="ascii")
    facts = captured_completion_facts(tmp_path, log)
    log.unlink()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Completion classifier reopened or probed an artifact")

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        outcome = runner_base.collect_run_outcome("", str(log), exit_code=-9, logs=facts)

    assert outcome.failure_code == "convergence_failed"
    assert outcome.failure_evidence is not None
    assert outcome.failure_evidence["exit_code"] == -9
    assert outcome.log_excerpt is not None and outcome.error is not None
    assert "Time step too small" in outcome.log_excerpt
    assert outcome.log_excerpt in outcome.error


@pytest.mark.parametrize("section", ["diagnostics", "error_context"])
def test_failed_section_cannot_certify_success(tmp_path, section):
    metadata = captured_completion_facts(tmp_path, text="Circuit: test\n").as_dict()
    metadata[section] = {
        "status": "error",
        "value": None,
        "nonfinite_count": 0,
        "error": {"type": "ValueError", "message": "synthetic section failure"},
    }
    outcome = runner_base.collect_run_outcome(
        str(tmp_path / "missing.raw"), "reported.log", exit_code=0, logs=DecodedLog(metadata)
    )
    assert outcome.error is not None
    assert "synthetic section failure" in outcome.error


def test_unavailable_loader_preserves_process_failure():
    outcome = runner_base.collect_run_outcome(
        "candidate.raw",
        "run.fail",
        exit_code=-2,
        logs=None,
        logs_error="Parser cleanup could not confirm exit",
        simulator_exception="TimeoutExpired: recorded-simulator",
    )
    assert outcome.error is not None
    assert "Parser cleanup could not confirm exit" in outcome.error
    assert outcome.failure_evidence == {
        "exit_code": -2,
        "simulator_exception": "TimeoutExpired: recorded-simulator",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_state", ["positive", "broken", "missing"])
async def test_submission_loads_only_cold_completion_once(tmp_path, monkeypatch, raw_state):
    loop = asyncio.get_running_loop()
    base = runner_base.RunnerBase(loop, type("RecordedSimulator", (), {}), tmp_path)
    deck = tmp_path / "deck.cir"
    deck.write_text(".tran 1u 1m\n.end\n", encoding="ascii")
    log = tmp_path / "actual.fail"
    log.write_text("Circuit: x\nTime step too small; time = 1.7e-05\n", encoding="ascii")
    # A positive RAW uses a normal log name; a .fail remains a failed run.
    if raw_state != "missing":
        log = log.rename(tmp_path / "actual.log")
    raw = tmp_path / "actual.raw"
    if raw_state == "positive":
        raw.write_bytes(b"recorded nonempty raw")
    elif raw_state == "broken":
        parent = tmp_path / "not-a-directory"
        parent.write_text("file", encoding="ascii")
        raw = parent / "actual.raw"
    facts = captured_completion_facts(tmp_path, log)
    calls = []

    def load(raw_path, log_path):
        calls.append((raw_path, log_path, threading.get_ident()))
        log_path.unlink()
        return facts

    class FakeHandle:
        active_tasks = ()

        def run(self, _netlist, **kwargs):
            kwargs["callback"](raw, log)

    monkeypatch.setattr(base, "_build_sim_runner", lambda **_kwargs: FakeHandle())
    received = loop.create_future()

    def submit():
        try:
            base.submit_netlist(deck, "run.cir", received.set_result, completion_logs=load)
        except Exception as exc:
            loop.call_soon_threadsafe(received.set_exception, exc)

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    thread.join(timeout=LIVENESS_S)
    assert not thread.is_alive()
    outcome = await asyncio.wait_for(received, LIVENESS_S)
    if raw_state == "missing":
        assert len(calls) == 1
        assert calls[0][:2] == (raw, log)
        assert calls[0][2] != threading.get_ident()
        assert "Time step too small" in outcome.log_excerpt
    else:
        assert calls == []
        assert outcome.raw_file == str(raw)
        assert (outcome.error is None) is (raw_state == "positive")


@pytest.mark.asyncio
async def test_submission_loader_failure_retains_identifiers_and_exit(tmp_path, monkeypatch):
    base = runner_base.RunnerBase(
        asyncio.get_running_loop(), type("RecordedSimulator", (), {}), tmp_path
    )
    deck = tmp_path / "deck.cir"
    deck.write_text(".op\n.end\n", encoding="ascii")
    raw, log = tmp_path / "candidate.raw", tmp_path / "actual.fail"

    def unavailable(*_args):
        raise ValueError("synthetic capture refusal")

    class FakeHandle:
        active_tasks = ()

        def run(self, _netlist, **kwargs):
            thread = threading.current_thread()
            monkeypatch.setattr(thread, "retcode", -2, raising=False)
            monkeypatch.setattr(thread, "exception_text", "recorded timeout", raising=False)
            kwargs["callback"](raw, log)

    monkeypatch.setattr(base, "_build_sim_runner", lambda **_kwargs: FakeHandle())
    received = asyncio.get_running_loop().create_future()
    base.submit_netlist(deck, "run.cir", received.set_result, completion_logs=unavailable)
    outcome = await asyncio.wait_for(received, LIVENESS_S)
    assert outcome.log_file == str(log)
    assert outcome.failure_evidence == {"exit_code": -2, "simulator_exception": "recorded timeout"}
    assert "synthetic capture refusal" in outcome.error


@pytest.mark.parametrize("console", [None, "Error: analysis not run\n"])
def test_absent_primary_log_cannot_certify_success(tmp_path, console):
    facts = captured_completion_facts(tmp_path, console=console)
    outcome = runner_base.collect_run_outcome(
        "missing.raw", "missing.log", exit_code=0, logs=facts
    )
    assert outcome.error is not None
    assert outcome.log_excerpt is None
    if console:
        assert outcome.failure_code == "execution_failed"


def test_missing_required_raw_preserves_save_evidence(tmp_path):
    facts = captured_completion_facts(
        tmp_path, text="Circuit: divider\nTotal elapsed time: 0.01\n"
    )
    requirements = runner_base.DeckRequirements((".tran",), True, False)
    outcome = runner_base.collect_run_outcome(
        "missing.raw", "run.log", requirements, exit_code=0, logs=facts
    )
    assert outcome.observations[0]["evidence"] == {
        "expected_artifact": "raw",
        "analyses": [".tran"],
        "has_save_list": True,
    }
    assert outcome.error is not None
    assert "List every probed node" in outcome.error
    assert "Total elapsed time" in outcome.error


def test_diagnostics_error_still_retains_stopped_excerpt(tmp_path):
    metadata = captured_completion_facts(
        tmp_path, text="Circuit: test\nrecorded stop tail\n"
    ).as_dict()
    metadata["diagnostics"] = {
        "status": "error",
        "value": None,
        "nonfinite_count": 0,
        "error": {"type": "ValueError", "message": "synthetic diagnostics failure"},
    }
    outcome = runner_base.collect_run_outcome(
        "", "run.fail", exit_code=-9, logs=DecodedLog(metadata)
    )
    assert outcome.log_excerpt is not None and outcome.error is not None
    assert outcome.failure_evidence is not None
    assert "recorded stop tail" in outcome.log_excerpt
    assert "synthetic diagnostics failure" in outcome.error
    assert outcome.failure_evidence["exit_code"] == -9


@pytest.mark.asyncio
async def test_coordinator_injects_explicit_fail_and_console_sources(
    state_no_sim,
    work_dir,
    monkeypatch,
):
    from ltspice_mcp.lib import services
    from ltspice_mcp.lib.experiment_runner import ExperimentRunner
    from tests.conftest import await_until
    from tests.test_experiment_runner import MockSimulator, _controlled_submit, _request

    runner = ExperimentRunner(asyncio.get_running_loop(), MockSimulator, work_dir, 1)
    launches = []
    callbacks, submissions = _controlled_submit(monkeypatch, runner, launches)
    receipt = await runner.submit(_request(state_no_sim, work_dir, request_id="completion-facts"))
    await await_until(lambda: bool(submissions))
    assert receipt.job.output_folder is not None
    log = receipt.job.output_folder / f"{submissions[0]}.fail"
    log.write_text("Circuit: x\nTime step too small; time = 1.7e-05\n", encoding="ascii")
    log.with_suffix(".exe.log").write_text("Error: analysis not run\n", encoding="ascii")
    missing_raw = log.with_suffix(".raw")
    seen = []
    original = services.load_logs_sync

    def load(source, state):
        seen.append((source, state))
        return original(source, state)

    monkeypatch.setattr(services, "load_logs_sync", load)
    facts = launches[0]["completion_logs"](missing_raw, log)
    source, state = seen[0]
    assert state is state_no_sim
    assert source.raw == missing_raw and source.log == log
    assert source.console == log.with_suffix(".exe.log")
    assert source.identity is None and source.trusted_job_artifact is True
    assert source.netlist == receipt.job.cases[0].staged_deck
    assert {item.role for item in facts.captured.files} == {"log", "console"}
    assert "Time step too small" in facts.value("error_context")
    assert "Error: analysis not run" in facts.value("diagnostics")["errors"]
    callbacks[submissions[0]](
        runner_base.collect_run_outcome("", str(log), exit_code=-9, logs=facts)
    )
    assert await job_done(state_no_sim, receipt.job)
    assert not list((state_no_sim.store.root / "parsing").glob("*"))
