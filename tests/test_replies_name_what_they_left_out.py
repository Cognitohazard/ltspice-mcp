"""A reply names what its answer left out, and every reply that answers one
question gives the same answer.

A field left out of a reply means "checked, and nothing to report", so each
of these was a false all-clear or two answers to one question:

- a mismatch rule that reaches a top-level device whose size the server
  cannot read (QSPICE, Xyce: W or L written as an expression) drew for the
  other devices alone;
- a per-instance ``delvto`` or ``mulu0`` (``X1:delvto``) ran on a model that
  the simulator refuses it on, or ignores it on;
- the ``summary`` recipe called the measurements of a run whose measurements
  the server does not read "missing", while the run row said why;
- a read by job did not repeat that the raw names subcircuit members otherwise
  than the deck was read;
- a recoverable run that exited after its kill grace skipped the end-of-run
  checks a run that ended any other way makes.

Each runs through the tool and through the Python API.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import ApiCallError
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import (
    LIVENESS_S,
    FakeSim,
    SyncApi,
    coordinator_returned,
    dispatched,
    fake_artifact_paths,
    needs_controlled_startup,
    replaying_simulator,
    stand_in_program,
    submit_experiment,
)
from tests.conftest import fake_host as _host
from tests.ltspice_recorder import INPUTS
from tests.test_simulator_families import _XYCE_TRAN_RAW, XYCE_LOG, _qspice_ac_qraw, _stand_in


def _refusal(data: dict[str, Any]) -> tuple[str, str]:
    """The code and message a reply refused the circuit with."""
    if "error" in data:
        return data["error"].get("detail_code", data["error"]["code"]), data["error"]["message"]
    (failure,) = {(f["code"], f["message"]) for f in data["failures"]}
    return failure


def _both_routes(
    session: Callable[[], SessionState], payload: dict[str, Any], *, refused: bool = True
) -> dict[str, dict[str, Any]]:
    """The reply to ``payload`` through ``run_experiments`` and, for a request
    that is ``refused`` before any case runs, through the Python API, each in
    a session of its own (``session``): a session's runner belongs to the loop
    that started it, and a run started on the API's throwaway loop would
    outlive it."""
    _, by_tool = asyncio.run(submit_experiment(session(), {**payload, "request_id": "by-tool"}))
    if not refused:
        return {"tool": by_tool}
    state = session()
    state.guide_read = True
    try:
        by_api = SyncApi(state).run_experiments(**{**payload, "request_id": "by-api"})
    except ApiCallError as refusal:
        by_api = refusal.payload
    return {"tool": by_tool, "api": by_api}


# --------------------------------------------------------------------------
# A mismatch rule reaches a device the server cannot size
# --------------------------------------------------------------------------

#: Two top-level devices, one sized by an expression the QSPICE and Xyce path
#: does not evaluate.
_SIZED_BY_EXPRESSION = """\
* two top-level devices, one sized by an expression
.param wn=2u
MA d g 0 0 nm W=10u L=1u
MB d g 0 0 nm W={wn} L=1u
.model nm nmos(vto=0.7 kp=100u)
Vd d 0 0.9
Vg g 0 0.9
.op
.end
"""


@pytest.fixture
def qspice(work_dir: Path) -> type:
    program = stand_in_program(work_dir / "sim" / "QSPICE64.exe", b"a QSPICE build")
    from spicelib.simulators.qspice_simulator import Qspice

    return _stand_in(Qspice, program, _qspice_ac_qraw(), "Simulation completed.\n")


@pytest.fixture
def xyce(work_dir: Path) -> type:
    from spicelib.simulators.xyce_simulator import XyceSimulator

    program = stand_in_program(work_dir / "sim" / "Xyce", b"a Xyce build")
    return _stand_in(XyceSimulator, program, b"", XYCE_LOG)


@pytest.mark.parametrize("prefix", ["M", "MB"])
@pytest.mark.parametrize("family", ["qspice", "xyce"])
def test_a_mismatch_rule_that_reaches_a_device_it_cannot_size_is_refused(
    config: ServerConfig,
    work_dir: Path,
    qspice: type,
    xyce: type,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    prefix: str,
):
    """``M`` reaches MA (literal size) and MB (W an expression): it used to
    draw for MA alone and report that spread as the rule's. ``MB`` reaches
    only MB. Both are refused before any case runs, naming MB, why, and the
    fix, through ``run_experiments`` and the Python API."""
    _host(monkeypatch, "windows")
    simulator = {"qspice": qspice, "xyce": xyce}[family]

    def session() -> SessionState:
        return SessionState.create(config, available={family: simulator})

    deck = work_dir / "sized.cir"
    deck.write_text(_SIZED_BY_EXPRESSION)
    payload = {
        "circuits": [{"path": str(deck), "id": "dut"}],
        "variations": [
            {
                "kind": "random",
                "runs": 2,
                "seed": 3,
                "rules": [{"rule": "mismatch", "prefix": prefix, "AVT": 3e-3}],
            }
        ],
        "execution": {"simulator": family, "wait_s": LIVENESS_S},
    }
    for route, reply in _both_routes(session, payload).items():
        code, message = _refusal(reply)
        assert code == "mismatch_geometry_unresolved", (route, reply)
        assert "MB" in message, route
        assert "MA" not in message.replace("MAX", ""), route
        assert "W" in message and "number" in message, route
    assert simulator.launched == []


# --------------------------------------------------------------------------
# A per-instance delvto or mulu0 on a model the simulator does not take it on
# --------------------------------------------------------------------------

_WRAPPED = """\
* a device wrapped in a subcircuit
.subckt cell d g s b
m0 d g s b cell_model w=1u l=1u
.model cell_model nmos level={level} vth0=0.5 u0=0.06
.ends cell
XN1 d g 0 0 cell
Vd d 0 0.9
Vg g 0 0.9
.op
.end
"""

#: (model level, parameter) on LTspice -> the refusal code, or None where the
#: recordings ``deck/mos_delvto_*`` and ``deck/mos_mulu0_*`` show LTspice takes it.
_LTSPICE_INSTANCE_PARAMETER = {
    # LTspice 26 stops on "Unknown parameter"; XVII says so and runs as if
    # the parameter were not there.
    (8, "delvto"): "instance_parameter_not_taken",
    (8, "mulu0"): "instance_parameter_not_taken",
    (54, "mulu0"): "instance_parameter_not_taken",
    # No recording of either build on these.
    (14, "delvto"): "instance_parameter_unestablished",
    (1, "delvto"): "instance_parameter_unestablished",
    # Both builds take it, as a shift of the signed threshold.
    (54, "delvto"): None,
}


@pytest.mark.parametrize(("level", "parameter"), sorted(_LTSPICE_INSTANCE_PARAMETER))
def test_an_ltspice_instance_parameter_runs_only_where_the_recordings_show_it_taken(
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    parameter: str,
):
    """``XN1:<parameter>`` on LTspice, through ``run_experiments`` and the
    Python API: refused before any case runs where LTspice is recorded to
    refuse or ignore it, refused as not established where no recording shows
    it, and run where the recordings show it taken."""
    launcher = type("LTspice", (FakeSim,), {})

    def session() -> SessionState:
        return SessionState.create(config, available={"ltspice": launcher})

    # A run that is not refused completes at once, on a recorded .op.
    launched = replaying_simulator(
        monkeypatch,
        rec.recorded("ltspice26", "raw/op_subckt.raw"),
        rec.recorded("ltspice26", "raw/op_subckt.log"),
    )
    deck = work_dir / "wrapped.cir"
    deck.write_text(_WRAPPED.format(level=level))
    payload = {
        "circuits": [{"path": str(deck), "id": "dut"}],
        "variations": [{"kind": "assign", "assign": {f"XN1:{parameter}": [0.9, 1.1]}}],
        "execution": {"wait_s": LIVENESS_S},
        "lint": "off",
    }
    expected = _LTSPICE_INSTANCE_PARAMETER[(level, parameter)]
    for route, reply in _both_routes(session, payload, refused=expected is not None).items():
        if expected is None:
            assert "error" not in reply and not reply.get("failures"), (route, reply)
            continue
        code, message = _refusal(reply)
        assert code == expected, (route, reply)
        assert "XN1" in message and parameter in message, route
        if expected == "instance_parameter_not_taken":
            # The fix: a mismatch rule gives the device its own model copy.
            assert "mismatch" in message and "instance" in message, route
        else:
            assert "not established" in message, route
            assert f"deck/mos_{parameter}_" in message, route
    assert bool(launched) is (expected is None)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
async def test_ngspice_refuses_an_instance_delvto_on_a_level_one_model_before_it_runs(
    work_dir: Path,
):
    """On the ngspice on PATH a level 1 device stops the run with ``unknown
    parameter (delvto)``; the server refuses that before the run, as it does
    for a mismatch draw inside a subcircuit."""
    from ltspice_mcp.lib.simulator import detect_simulators
    from tests.conftest import terminal_experiment

    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    state = SessionState.create(config, detect_simulators(config))
    deck = work_dir / "wrapped.cir"
    deck.write_text(_WRAPPED.format(level=1).replace(" vth0=0.5 u0=0.06", " vto=0.5"))
    payload = {
        "request_id": "ngspice-level-one",
        "circuits": [{"path": str(deck), "id": "dut"}],
        "variations": [{"kind": "assign", "assign": {"XN1:delvto": [0.01]}}],
        "execution": {"simulator": "ngspice", "wait_s": 90},
        "lint": "off",
    }
    reply = await terminal_experiment(state, payload)
    code, message = _refusal(reply)
    assert code == "unsupported_model", reply
    assert "XN1" in message


# --------------------------------------------------------------------------
# Why a run's measurement has no value
# --------------------------------------------------------------------------


def _recorded_run(case: str, raw_case: str, directory: Path) -> tuple[Path, Path, Path]:
    name = Path(case).name
    deck = directory / f"{name}.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    log = directory / f"{name}.log"
    shutil.copyfile(rec.recorded("ltspice26", f"{case}.log"), log)
    raw = directory / f"{name}.raw"
    shutil.copyfile(rec.recorded("ltspice26", f"{raw_case}.raw"), raw)
    return deck, raw, log


async def _run(state: SessionState, deck: Path, request_id: str) -> dict[str, Any]:
    receipt = await dispatched(
        state,
        "run_experiments",
        request_id=request_id,
        circuits=[{"path": str(deck), "id": deck.stem}],
        execution={"wait_s": LIVENESS_S},
        lint="off",
    )
    assert receipt["status"] == "completed", receipt
    return receipt


async def _every_reason(state: SessionState, receipt: dict[str, Any]) -> dict[str, Any]:
    """Why each reader of the run's measurements says it holds none: the run
    row, the ``summary`` recipe and the ``measurements`` recipe by MCP and by
    the Python API, and the Python API's ``measurements``."""
    job_id = receipt["job_id"]
    api = SyncApi(state)
    job = [{"job_id": job_id}]
    found: dict[str, Any] = {
        "run row": receipt["runs"]["items"][0].get("measurements_coverage"),
    }
    summary = {"key": "s", "metric": "summary"}
    measurements = {"key": "m", "metric": "measurements"}
    for route, read in (
        ("tool", lambda **kw: dispatched(state, "analyze_results", **kw)),
        ("api", lambda **kw: asyncio.to_thread(lambda: api.analyze_results(**kw))),
    ):
        # The reply carries the summary's coverage and reconciliation
        # observations at its top level.
        data = await read(sources=job, recipes=[summary])
        said = data.get("observations", [])
        unread = [o for o in said if o["code"] == "measurements_not_read"]
        unmet = [o for o in said if o["code"] == "unmet_request"]
        assert unmet == [], (route, unmet)
        assert len(unread) == 1, (route, data)
        found[f"{route} summary"] = unread[0]["evidence"]["reason"]
        # The whole value reports no measurement the server did not read.
        data = await read(sources=job, recipes=[summary], include={"fields": ["value"]})
        assert "failed_measurements" not in str(data), route
        (row,) = data["results"]["s"]["values"]
        assert "measurements" not in row["value"], route
        data = await read(sources=job, recipes=[measurements])
        found[f"{route} measurements"] = data
    try:
        await asyncio.to_thread(lambda: api.measurements(job_id=job_id))
        found["api measurements()"] = None
    except Exception as refused:  # the refusal names the reason
        found["api measurements()"] = str(refused)
    return found


def _observations(data: Any) -> list[dict[str, Any]]:
    """Every observation anywhere in a reply: a recipe's own sit in its value."""
    found: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("code"), str) and "detail" in data:
            found.append(data)
        for value in data.values():
            found.extend(_observations(value))
    elif isinstance(data, list):
        for value in data:
            found.extend(_observations(value))
    return found


def _same_reason(found: dict[str, Any], reason: str) -> None:
    for route, said in found.items():
        assert reason in str(said), (route, said)


@pytest.mark.parametrize("family", ["qspice", "xyce"])
async def test_a_run_whose_measurements_are_not_read_says_why_on_every_reader(
    family: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A QSPICE or Xyce run's measure output is not read: the run row, the
    ``summary`` recipe, the ``measurements`` recipe and the Python API's
    ``measurements`` all say ``simulator_unsupported``, and none calls a
    requested ``.meas`` missing or reports a value the server did not read."""
    from ltspice_mcp.lib.simulator import SIMULATORS

    state = SessionState.create(
        config, available={family: type(SIMULATORS[family].__name__, (FakeSim,), {})}
    )
    deck, _raw, log = _recorded_run("log/meas_failed", "raw/tran_uncompressed", work_dir)
    # A raw in the simulator's own format, and a log holding .meas results the
    # server must not read for this simulator.
    raw = work_dir / f"{family}.raw"
    raw.write_bytes(_qspice_ac_qraw() if family == "qspice" else _XYCE_TRAN_RAW)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state, deck, f"unread-{family}")
    _same_reason(await _every_reason(state, receipt), "simulator_unsupported")


async def test_a_run_whose_log_was_not_captured_says_why_on_every_reader(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An LTspice run that left a raw and no log: every reader says
    ``log_unread`` (``services.measurements_unread``'s branch for a log that
    was not captured), and none calls a requested ``.meas`` missing."""
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    deck, raw, _log = _recorded_run("log/meas_failed", "raw/tran_uncompressed", work_dir)

    def submit(self, _netlist: Path, run_filename: str, callback, **kwargs):
        raw_out, _ = fake_artifact_paths(self.output_folder, run_filename)
        shutil.copy(raw, raw_out)
        outcome = RunOutcome(str(raw_out), "", raw_out.stat().st_size, None)
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    receipt = await _run(state, deck, "no-log")
    _same_reason(await _every_reason(state, receipt), "log_unread")


# --------------------------------------------------------------------------
# A read by job repeats what the run said of its raw
# --------------------------------------------------------------------------


async def test_a_read_by_job_repeats_that_the_raw_names_disagree(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An LTspice job whose program names no build reads its deck as LTspice
    26 spells subcircuit members; an XVII raw from its run spells them
    without the X. The run says so, and so does every later read of the run
    by job: ``analyze_results`` and ``plot_waveform`` by MCP, and the Python
    API's ``analyze_results`` and ``load_raw``."""
    from spicelib.simulators.ltspice_simulator import LTspice

    case = "raw/save_subckt_names"
    deck = work_dir / "nested.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    launcher = type(LTspice.__name__, (LTspice,), {"spice_exe": ["C:/tools/lt.exe"]})
    state = SessionState.create(config, available={"ltspice": launcher})
    replaying_simulator(
        monkeypatch,
        rec.recorded("ltspice17", f"{case}.raw"),
        rec.recorded("ltspice17", f"{case}.log"),
    )
    receipt = await _run(state, deck, "names-disagree")
    job_id = receipt["job_id"]
    case_id = receipt["runs"]["items"][0]["case_id"]
    job = [{"job_id": job_id}]
    api = SyncApi(state)
    found = {"receipt": receipt.get("observations", [])}
    analyzed = await dispatched(
        state, "analyze_results", sources=job, recipes=[{"key": "s", "metric": "summary"}]
    )
    found["analyze_results"] = analyzed.get("observations", [])
    plotted = await dispatched(state, "plot_waveform", job_id=job_id, signals=["V(outa)"])
    found["plot_waveform"] = plotted.get("observations", [])
    by_api = await asyncio.to_thread(
        lambda: api.analyze_results(sources=job, recipes=[{"key": "s", "metric": "summary"}])
    )
    found["api analyze_results"] = by_api.get("observations", [])
    raw = await asyncio.to_thread(lambda: api.load_raw(job_id=job_id, case_id=case_id))
    found["api load_raw"] = list(raw.observations)
    for route, observations in found.items():
        codes = [item["code"] for item in observations]
        assert codes.count("raw_names_disagree") == 1, (route, codes)


# --------------------------------------------------------------------------
# Every path that ends a run makes the same checks
# --------------------------------------------------------------------------

#: A deck whose subcircuit has a net of its own, with a .save that keeps no
#: node inside it: the raw its run writes holds none.
_SUBCIRCUIT_DECK = """* a node inside nested subcircuit instances
V1 vdd 0 3
Xa vdd outa stage
.subckt stage top out
Rd top out 10k
Xin top out inner
.ends
.subckt inner p n
Rin p mid 100k
Rlo mid n 50k
.ends
.save v(outa)
.tran 1u 10u
.end
"""


def _end_checks(case: Any) -> dict[str, Any]:
    """What the end-of-run checks left on ``case``."""
    return {
        "subcircuit_nodes_not_saved": sum(
            item["code"] == "subcircuit_nodes_not_saved" for item in case.observations
        ),
        "measurements read": case.measurement_facts is not None,
        "log classified": case.log_diagnostics is not None,
    }


@needs_controlled_startup
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize("ending", ["completed", "stopped_in_grace", "late_exit"])
async def test_every_path_that_ends_a_recoverable_run_makes_the_same_checks(
    ending: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A recoverable run keeps its raw however it ends: it completes, it is
    stopped and exits within the kill grace, or it exits after the grace. Each
    path makes the same end-of-run checks, so the raw's missing subcircuit
    nodes are said whichever ended it, on ``jobs`` and the Python API's
    ``jobs``. The raw and log are the ones the ngspice on PATH writes for the
    deck; when the run exits is the test's."""
    import subprocess

    from tests.test_resume_surface import _state

    made = work_dir / "made"
    made.mkdir()
    (made / "nested.cir").write_text(_SUBCIRCUIT_DECK, encoding="utf-8", newline="\n")
    await asyncio.to_thread(
        subprocess.run,
        [
            shutil.which("ngspice") or "ngspice",
            "-b",
            "-n",
            "-r",
            "nested.raw",
            "-o",
            "nested.log",
            "nested.cir",
        ],
        cwd=made,
        check=True,
        capture_output=True,
        timeout=LIVENESS_S,
    )
    deck = work_dir / "nested.cir"
    deck.write_text(_SUBCIRCUIT_DECK, encoding="utf-8", newline="\n")
    state = _state(work_dir)
    callbacks: dict[str, Any] = {}
    launched = asyncio.Event()

    def held(self, _netlist: Path, run_filename: str, callback, **kwargs):
        raw_out, log_out = fake_artifact_paths(self.output_folder, run_filename)
        shutil.copy(made / "nested.raw", raw_out)
        shutil.copy(made / "nested.log", log_out)
        callbacks[Path(run_filename).stem] = lambda error=None: callback(
            RunOutcome(str(raw_out), str(log_out), raw_out.stat().st_size, error)
        )
        self.loop.call_soon_threadsafe(launched.set)
        return object()

    async def kill(self, token: str) -> None:
        if ending == "stopped_in_grace":
            callbacks[token]("Stopped by test simulator")

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", held)
    monkeypatch.setattr(ExperimentRunner, "_kill_case", kill)
    execution: dict[str, Any] = {"recoverable": True, "wait_s": 0}
    if ending != "completed":
        # timing: the bound under test; the run never exits on its own, and a
        # late exit waits out the default kill grace first
        execution["run_timeout_s"] = 0.05
    try:
        receipt = await dispatched(
            state,
            "run_experiments",
            request_id=f"ends-{ending}",
            circuits=[{"path": str(deck), "id": "nested"}],
            execution=execution,
            lint="off",
        )
        assert "job_id" in receipt, receipt
        job = state.all_jobs[receipt["job_id"]]
        await asyncio.wait_for(launched.wait(), LIVENESS_S)
        (token,) = callbacks
        if ending == "completed":
            callbacks[token]()
        runner = state.runners.get_experiment_runner_for(job)
        assert runner is not None
        await coordinator_returned(state, job)
        if ending == "late_exit":
            assert job.cases[0].failure_code == "kill_unconfirmed"
            callbacks[token]("killed")
        await runner.settled(job)
        (case,) = job.cases
        assert _end_checks(case) == {
            "subcircuit_nodes_not_saved": 1,
            "measurements read": True,
            "log classified": True,
        }, (ending, case.observations)
        for route, data in (
            ("jobs", await dispatched(state, "jobs", action="status", job_id=job.job_id)),
            (
                "api jobs",
                await asyncio.to_thread(
                    lambda: SyncApi(state).jobs(action="status", job_id=job.job_id)
                ),
            ),
        ):
            codes = [item["code"] for item in _observations(data)]
            assert "subcircuit_nodes_not_saved" in codes, (route, codes)
    finally:
        await state.job_registry.cancel_running(state.runners, state)
        await state.job_registry.drain_pending()


# --------------------------------------------------------------------------
# The guards: one answerer for each question
# --------------------------------------------------------------------------


def _calls(names: set[str]) -> dict[tuple[str, str], set[str]]:
    """Where in the package each of ``names`` is called or referenced, by
    (module, scope)."""
    import ast

    from tests._source_scan import modules, scopes

    found: dict[tuple[str, str], set[str]] = {}
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                name = (
                    node.attr
                    if isinstance(node, ast.Attribute)
                    else node.id
                    if isinstance(node, ast.Name)
                    else None
                )
                if name in names:
                    found.setdefault((path, scope), set()).add(name)
    return found


def test_one_function_writes_an_instance_delvto_or_mulu0():
    """Every edit that writes ``delvto`` or ``mulu0`` on a device's card is
    made by ``variations._instance_parameter_edit``, which asks
    ``require_instance_parameter`` first: no second writer can skip the
    question whether the simulator applies it to the model."""
    import ast

    from tests._source_scan import modules, scopes

    writers = set()
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "InstanceEdit"
                    and any(
                        isinstance(arg, ast.Constant) and arg.value == "parameter"
                        for arg in node.args
                    )
                ):
                    writers.add((path, scope))
    assert writers == {("lib/variations.py", "_instance_parameter_edit")}
    assert ("lib/variations.py", "_instance_parameter_edit") in _calls(
        {"require_instance_parameter"}
    )


def _ltspice_refused(build: str, parameter: str, model: str) -> bool:
    """Whether the log ``build`` wrote for ``deck/mos_<parameter>_<model>``
    says LTspice did not take the parameter."""
    data = rec.recorded(build, f"deck/mos_{parameter}_{model}.log").read_bytes()
    text = data.decode("utf-16-le") if data[1:2] == b"\0" else data.decode("latin-1")
    return "unknown parameter" in text.casefold()


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_ltspice_instance_parameter_table_is_what_the_recordings_show(build: str):
    """Each entry of ``LTSPICE_INSTANCE_PARAMETERS`` is the recordings'
    answer, on both builds: level 8 is the BSIM3 deck, level 54 the BSIM4."""
    from ltspice_mcp.lib.subckt_mismatch import LTSPICE_INSTANCE_PARAMETERS

    models = {8: "bsim3", 54: "bsim4"}
    assert {level for _, level in LTSPICE_INSTANCE_PARAMETERS} == set(models)
    for (parameter, level), taken in LTSPICE_INSTANCE_PARAMETERS.items():
        assert taken is not _ltspice_refused(build, parameter, models[int(level)]), (
            parameter,
            level,
        )


def test_every_simulator_answers_whether_it_takes_an_instance_parameter(tmp_path: Path):
    """``require_instance_parameter`` answers for every simulator the server
    runs, and refuses one it holds no evidence for: a BSIM4 device that
    LTspice and ngspice take ``delvto`` on is refused as not established on
    QSPICE and Xyce."""
    from ltspice_mcp.lib.hierarchy import (
        SemanticProfile,
        load_hierarchy,
    )
    from ltspice_mcp.lib.simulator import SIMULATORS
    from ltspice_mcp.lib.subckt_mismatch import MismatchPlanError, require_instance_parameter

    path = tmp_path / "wrapped.cir"
    path.write_text(_WRAPPED.format(level=54), encoding="utf-8")
    hierarchy = load_hierarchy(str(path), [tmp_path], SemanticProfile("ngspice", "hsa"))
    (row,) = [row for row in hierarchy.instances if row.element == "M"]
    answers = {}
    for family in sorted(SIMULATORS):
        try:
            require_instance_parameter(row, family, "delvto")
            answers[family] = None
        except MismatchPlanError as refused:
            answers[family] = refused.code
    assert answers == {
        "ltspice": None,
        "ngspice": None,
        "qspice": "instance_parameter_unestablished",
        "xyce": "instance_parameter_unestablished",
    }


#: Where a log's measurements section is read, and why each may: services
#: owns the read; the run row and the every-run check ask
#: ``measurements_unread`` first; the summary builder takes its answer as an
#: argument.
_MEASUREMENT_READERS = {
    ("lib/services.py", "measurements_unread"),
    ("lib/services.py", "read_measurements"),
    ("lib/services.py", "_unread_detail"),
    ("lib/experiment_runner.py", "ExperimentRunner._read_measurements"),
    ("tools/analyze.py", "_measurements_failed_everywhere"),
    ("lib/raw_parser.py", "build_simulation_summary"),
}


def test_every_read_of_a_logs_measurements_asks_why_they_may_not_be_read():
    """A read of the ``measurements`` section by name is made only where the
    reason it may say nothing (``measurements_unread``) is asked first or
    passed in, so no reader can call an unread measurement missing."""
    import ast

    from tests._source_scan import modules, scopes

    readers = set()
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "attr", getattr(node.func, "id", None))
                    in {"value", "section", "section_value"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "measurements"
                ):
                    readers.add((path, scope))
    assert readers == _MEASUREMENT_READERS
    asked = _calls({"measurements_unread", "source_measurements_unread"})
    for reader in readers - {
        ("lib/services.py", "measurements_unread"),
        # Called only by read_measurements, once it has the reason.
        ("lib/services.py", "_unread_detail"),
    }:
        assert reader in asked, reader


#: The codes a job, not one of its cases, carries: said on the receipt, and
#: never on a run.
_JOB_OBSERVATIONS = {
    "dangling_request_index_replaced",
    "experiment_index_write_failed",
    "external_cancellation_requested",
    "job_deadline",
    "kill_unconfirmed_capacity",
    "analysis_failed",
}

#: The modules that put an observation on a run's case.
_CASE_OBSERVATION_SOURCES = (
    "lib/experiment_runner.py",
    "lib/runner_base.py",
    "lib/deck_staging.py",
    "tools/experiments.py",
)


def test_every_observation_a_run_can_carry_is_either_read_by_job_or_says_why_not():
    """Each code a case's observations can hold is on ``READ_BY_JOB`` or on
    ``NOT_READ_BY_JOB`` with the reason it stays off, so a new observation
    cannot be left off a read by job in silence."""
    import ast

    from ltspice_mcp.lib.result_observations import NOT_READ_BY_JOB, READ_BY_JOB
    from tests._source_scan import modules

    codes: set[str] = {"subcircuit_nodes_not_saved", "saves_not_stored"}
    trees = modules()
    for path in _CASE_OBSERVATION_SOURCES:
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "code"
                        and isinstance(value, ast.Constant)
                    ):
                        codes.add(str(value.value))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_progress_fact":
                codes.update(
                    str(kw.value.value)
                    for kw in node.keywords
                    if kw.arg == "code" and isinstance(kw.value, ast.Constant)
                )
    # A read by job repeats each of these, so its kind is one the read's
    # schema holds (plot_waveform's is the narrowest).
    from ltspice_mcp.tools._base import OBSERVATIONS_SCHEMA

    allowed = set(OBSERVATIONS_SCHEMA["items"]["properties"]["kind"]["enum"])
    kinds: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            fields: dict[str, Any] = {}
            if isinstance(node, ast.Dict):
                fields = {
                    str(k.value): v.value
                    for k, v in zip(node.keys, node.values, strict=True)
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                }
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Observation":
                fields = {
                    kw.arg: kw.value.value
                    for kw in node.keywords
                    if kw.arg is not None and isinstance(kw.value, ast.Constant)
                }
            if fields.get("code") in READ_BY_JOB and "kind" in fields:
                kinds.setdefault(fields["code"], set()).add(fields["kind"])
    assert kinds.keys() == READ_BY_JOB
    assert all(found <= allowed for found in kinds.values()), kinds
    assert not READ_BY_JOB & NOT_READ_BY_JOB.keys()
    unclassified = codes - READ_BY_JOB - NOT_READ_BY_JOB.keys() - _JOB_OBSERVATIONS
    assert unclassified == set()
    assert NOT_READ_BY_JOB.keys() | READ_BY_JOB <= codes | {"run_progress"}


def test_the_end_of_run_checks_run_in_one_place():
    """Every check a run's end makes is called from ``_end_of_run`` alone,
    which each path that ends a run calls, so no ending can skip one."""
    checks = {
        "_with_logs",
        "_read_measurements",
        "_check_raw_names",
        "_check_subcircuit_nodes",
        "_apply_stopped_outcome",
    }
    callers = {where for where in _calls(checks) if where[0] == "lib/experiment_runner.py"}
    assert callers == {("lib/experiment_runner.py", "ExperimentRunner._end_of_run")}
    ending = {
        where[1] for where in _calls({"_end_of_run"}) if where[0] == "lib/experiment_runner.py"
    }
    assert ending == {
        "ExperimentRunner._run_case",
        "ExperimentRunner._stop_active_case",
        "ExperimentRunner._retire_late_exit",
    }


# --------------------------------------------------------------------------
# A recipe's observation that qualifies its number reaches a default reply
# --------------------------------------------------------------------------


async def test_a_constant_window_is_said_in_a_reply_that_leaves_the_value_out(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``V(in)`` of ``raw/tran_uncompressed`` holds 1 V after its 1 us edge:
    ``signal_stats`` over a later window gives a peak-to-peak of 0 and says the
    signal is constant there. A default reply, which leaves the row's value
    out, carries that statement at its top level once, naming the recipe, by
    MCP and by the Python API; one that shows the value carries it once too."""
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    deck, raw, log = _recorded_run("log/meas_failed", "raw/tran_uncompressed", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state, deck, "constant-window")
    job = [{"job_id": receipt["job_id"]}]
    stats = {
        "key": "flat",
        "metric": "signal_stats",
        "signal": "V(in)",
        "window": {"start": 1e-4, "end": 5e-4},
    }
    api = SyncApi(state)
    replies = {
        "tool": await dispatched(state, "analyze_results", sources=job, recipes=[stats]),
        "api": await asyncio.to_thread(lambda: api.analyze_results(sources=job, recipes=[stats])),
        "tool with the value": await dispatched(
            state,
            "analyze_results",
            sources=job,
            recipes=[stats],
            include={"fields": ["value"]},
        ),
    }
    for route, data in replies.items():
        said = [o for o in _observations(data) if o["code"] == "constant_window"]
        assert len(said) == 1, (route, data)
        assert said[0]["evidence"]["recipe"] == "flat", route
        assert said in [[o] for o in data.get("observations", [])], route


def test_every_recipe_observation_has_a_kind_the_lift_rule_places():
    """Every observation a recipe's metric can put in a row's value has a
    literal kind, and that kind is one ``analyze._value_observations``
    either lifts to the reply's top level or keeps in the value, with why."""
    import ast

    from ltspice_mcp.tools.analyze import _KEPT_KINDS, _LIFTED_KINDS
    from tests._source_scan import modules

    placed = _LIFTED_KINDS | _KEPT_KINDS.keys()
    assert not _LIFTED_KINDS & _KEPT_KINDS.keys()
    trees = modules()
    unplaced = []
    for path in (
        "lib/metrics.py",
        "lib/ac_structure.py",
        "lib/result_observations.py",
        "lib/signal_analysis.py",
        "lib/ac_analysis.py",
    ):
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.Dict):
                fields = {
                    k.value: v
                    for k, v in zip(node.keys, node.values, strict=True)
                    if isinstance(k, ast.Constant)
                }
                if "code" not in fields or "detail" not in fields:
                    continue
                kind = fields.get("kind")
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Observation":
                kind = next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
            else:
                continue
            if not (isinstance(kind, ast.Constant) and kind.value in placed):
                unplaced.append((path, node.lineno))
    assert unplaced == []


# --------------------------------------------------------------------------
# After a run, which .save entries the results do not hold
# --------------------------------------------------------------------------


def _saves_said(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in observations if item["code"] == "saves_not_stored"]


async def test_a_run_names_each_save_its_results_do_not_hold(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``raw/save_subckt_names_no_x`` on LTspice 26: the deck spells its
    ``.save`` paths as XVII does, and LTspice 26 stores nothing for them
    (recorded). The run names each, with why, on the receipt, ``jobs``, a
    read by job and the Python API; a deck with no ``.save`` card gets no
    such observation."""
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    case = "raw/save_subckt_names_no_x"
    deck = work_dir / "no_x.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    replaying_simulator(
        monkeypatch,
        rec.recorded("ltspice26", f"{case}.raw"),
        rec.recorded("ltspice26", f"{case}.log"),
    )
    receipt = await _run(state, deck, "saves-not-stored")
    job_id = receipt["job_id"]
    case_id = receipt["runs"]["items"][0]["case_id"]
    api = SyncApi(state)
    job = [{"job_id": job_id}]
    found = {
        "receipt": receipt.get("observations", []),
        "jobs": (await dispatched(state, "jobs", action="status", job_id=job_id)).get(
            "observations", []
        ),
        "analyze_results": (
            await dispatched(
                state, "analyze_results", sources=job, recipes=[{"key": "s", "metric": "summary"}]
            )
        ).get("observations", []),
        "api jobs": (
            await asyncio.to_thread(lambda: api.jobs(action="status", job_id=job_id))
        ).get("observations", []),
        "api load_raw": list(
            (
                await asyncio.to_thread(lambda: api.load_raw(job_id=job_id, case_id=case_id))
            ).observations
        ),
    }
    for route, observations in found.items():
        (said,) = _saves_said(observations)
        missing = {item["save"] for item in said["evidence"]["not_stored"]}
        assert missing == {"I(a:in:Rin)", "V(a:in:mid)", "I(a:Rd)"}, (route, said)
        assert "not_checked" not in said["evidence"], route
        assert "V(a:in:mid)" in said["detail"], route

    plain = work_dir / "plain.cir"
    shutil.copyfile(INPUTS / "raw/subckt_names_tran.cir", plain)
    replaying_simulator(
        monkeypatch,
        rec.recorded("ltspice26", "raw/subckt_names_tran.raw"),
        rec.recorded("ltspice26", "raw/subckt_names_tran.log"),
    )
    receipt = await _run(state, plain, "no-save")
    assert _saves_said(receipt.get("observations", [])) == []


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
async def test_ngspice_names_a_save_it_stored_nothing_for_and_one_it_did_not_check_live(
    work_dir: Path,
):
    """On the ngspice on PATH: a ``.save`` of a colon path stores nothing
    (measured), so the run names it; a pattern is said to be not checked,
    never guessed; a saved top-level node is held and not named."""
    from ltspice_mcp.lib.simulator import detect_simulators
    from tests.conftest import terminal_experiment

    deck = work_dir / "nested.cir"
    deck.write_text(
        _SUBCIRCUIT_DECK.replace(".save v(outa)", ".save v(outa) v(xa:xin:mid) v(xa.*)"),
        encoding="utf-8",
        newline="\n",
    )
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ngspice")
    state = SessionState.create(config, available=detect_simulators(config))
    reply = await terminal_experiment(
        state,
        {
            "request_id": "ngspice-saves",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 60},
            "lint": "off",
        },
    )
    assert reply["status"] == "completed", reply
    (said,) = _saves_said(reply.get("observations", []))
    assert [item["save"] for item in said["evidence"]["not_stored"]] == ["v(xa:xin:mid)"]
    assert [item["save"] for item in said["evidence"]["not_checked"]] == ["v(xa.*)"]


# --------------------------------------------------------------------------
# What the simulator said, and how severe: one answer
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pending a decision on the summary's published errors and warnings lists: "
        "deriving them from the run row's diagnostics changes what they hold (one "
        "entry per condition instead of one per line, and lines with no severity "
        "of the simulator's own move). See the pull request."
    ),
)
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_summary_says_what_the_run_row_says_the_simulator_said(
    build: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_bad_syntax``: each build prints an error for the ``.meas``
    it cannot evaluate, and the run produces. The summary recipe's errors and warnings, and the severity
    of each observation it relays, are derived from the diagnostics the run
    row gives: no error the row does not name, and no severity the simulator
    did not write. By MCP and by the Python API."""
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    case = "log/meas_bad_syntax"
    deck = work_dir / "timestep.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    replaying_simulator(
        monkeypatch,
        rec.recorded(build, "raw/tran_uncompressed.raw"),
        rec.recorded(build, f"{case}.log"),
    )
    receipt = await dispatched(
        state,
        "run_experiments",
        request_id=f"said-{build}",
        circuits=[{"path": str(deck), "id": "timestep"}],
        execution={"wait_s": LIVENESS_S},
        lint="off",
    )
    (row,) = receipt["runs"]["items"]
    diagnostics = row.get("diagnostics", [])
    severe = {d["first_text"] for d in diagnostics if "severity" in d}
    severities = {d["severity"] for d in diagnostics if "severity" in d}
    job = [{"job_id": receipt["job_id"]}]
    api = SyncApi(state)
    summary = {"key": "s", "metric": "summary"}
    for route, data in (
        (
            "tool",
            await dispatched(
                state,
                "analyze_results",
                sources=job,
                recipes=[summary],
                include={"fields": ["value"]},
            ),
        ),
        (
            "api",
            await asyncio.to_thread(
                lambda: api.analyze_results(
                    sources=job, recipes=[summary], include={"fields": ["value"]}
                )
            ),
        ),
    ):
        (value,) = [v["value"] for v in data["results"]["s"].get("values", [])] or [None]
        assert value is not None, (route, data)
        assert set(value.get("errors", [])) <= severe, (route, value.get("errors"), diagnostics)
        for item in _observations(data):
            if item.get("kind") == "relay" and "severity" in item:
                assert item["severity"] in severities, (route, item)
