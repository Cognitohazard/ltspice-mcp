"""Which simulator build ran: the executable a job records, the build each run
reports, and the replay check that keeps one build's results from answering
for another."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from spicelib.sim.simulator import Simulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import experiment_store, store
from ltspice_mcp.lib.simulator_build import (
    SimulatorExecutable,
    executable_identity,
    executable_path,
    reported_build,
    same_executable,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests.conftest import (
    capabilities_report,
    job_runs,
    stand_in_program,
    submit_experiment,
    submit_through_spicelib,
    terminal_experiment,
)

FIXTURES = Path(__file__).parent / "fixtures"
LTSPICE_26 = "LTspice 26.0.2 for Windows"

# Verbatim console output of a real ngspice-42 batch run (Ubuntu 24.04 package
# 42+ds-3build1), captured the way spicelib's exe_log writes it: stdout and
# stderr of ``ngspice -b -o run.log -r run.raw run.cir`` in one file. The -o
# log of the same run carries no version at all.
NGSPICE_42_CONSOLE = (
    "******\n"
    "** ngspice-42 : Circuit level simulation program\n"
    "** Compiled with KLU Direct Linear Solver\n"
    "** The U. C. Berkeley CAD Group\n"
    "** Copyright 1985-1994, Regents of the University of California.\n"
    "** Copyright 2001-2023, The ngspice team.\n"
    "** Please get your ngspice manual from https://ngspice.sourceforge.io/docs.html\n"
    "** Please file your bug-reports at http://ngspice.sourceforge.net/bugrep.html\n"
    "** Creation Date: Sun Mar 31 20:15:14 UTC 2024\n"
    "******\n"
    "\n"
    "Batch mode\n"
    "\n"
    "Simulation output goes to rawfile: run.raw\n"
    "Comments and warnings go to log-file: run.log\n"
)
NGSPICE_42 = "ngspice-42, Creation Date: Sun Mar 31 20:15:14 UTC 2024"

_DECK = "V1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebuild(path: Path, content: bytes) -> None:
    """Replace a program with another build, as an installer would.

    The modification time is moved on explicitly: on a coarse-mtime filesystem
    a rewrite inside the same tick would otherwise keep it.
    """
    before = path.stat().st_mtime
    path.write_bytes(content)
    os.utime(path, (before + 10, before + 10))


class _RecordedLTspice26(Simulator):
    """A spicelib simulator whose every run leaves a recorded LTspice 26 log and raw.

    It goes through spicelib's real SimRunner and RunTask threads and the
    server's real completion callback, so what a case records is read from
    the files a run left, as in production. ``spice_exe`` is set per test to a
    real file, the program whose identity the job records.
    """

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated
    process_name = "recorded-simulator"
    log_fixture: ClassVar[str] = "ltspice_tran_rc.log"
    raw_fixture: ClassVar[str] = "ltspice_tran_rc.raw"
    console: ClassVar[str | None] = None

    @classmethod
    def run(
        cls,
        netlist_file,
        cmd_line_switches=None,
        timeout=None,
        stdout=None,
        stderr=None,
        cwd=None,
        exe_log=False,
    ) -> int:
        netlist = Path(netlist_file)
        shutil.copy(FIXTURES / cls.log_fixture, netlist.with_suffix(".log"))
        shutil.copy(FIXTURES / cls.raw_fixture, netlist.with_suffix(".raw"))
        # spicelib writes the console capture only when asked to; the runner
        # always asks.
        if exe_log and cls.console is not None:
            netlist.with_suffix(".exe.log").write_text(cls.console)
        return 0

    @classmethod
    def valid_switch(cls, switch, switch_param) -> list:
        return []


class _RecordedNgspice42(_RecordedLTspice26):
    """An ngspice-shaped run: a log with no version, and the console banner."""

    log_fixture = "ngspice_ac_no_meas.log"
    raw_fixture = "ngspice_noise_2plot.raw"
    console = NGSPICE_42_CONSOLE


def _simulator(base: type, program: Path, *command: str) -> type:
    """A subclass of ``base`` launching ``program`` (after ``command``, as Wine is)."""
    return type(base.__name__, (base,), {"spice_exe": [*command, str(program)]})


def _state(config: ServerConfig, program: Path, *launcher: str) -> SessionState:
    """A session whose one simulator is a recorded LTspice 26 launching ``program``."""
    simulator = _simulator(_RecordedLTspice26, program, *launcher)
    return SessionState.create(config, available={"ltspice": simulator})


@pytest.fixture
def program(work_dir: Path) -> Path:
    """The executable most tests run: one build of LTspice."""
    return stand_in_program(work_dir / "sim" / "LTspice.exe", b"build one")


def _payload(deck: Path, request_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 30},
        "lint": "off",
        **extra,
    }


# ---------------------------------------------------------------------------
# What a run says about itself
# ---------------------------------------------------------------------------


class TestReportedBuild:
    def test_ltspice_names_its_version_on_the_first_line_of_its_log(self):
        # The recorded raw says only "Linear Technology Corporation LTspice";
        # the log's banner is the one that carries a version, so it wins.
        assert (
            reported_build(FIXTURES / "ltspice_tran_rc.log", FIXTURES / "ltspice_tran_rc.raw")
            == LTSPICE_26
        )

    def test_a_utf16_log_reads_the_same(self, tmp_path: Path):
        log = tmp_path / "run.log"
        text = (FIXTURES / "ltspice_tran_rc.log").read_text(encoding="utf-8")
        log.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))

        assert reported_build(log) == LTSPICE_26

    def test_ngspice_names_its_build_in_the_console_capture(self, tmp_path: Path):
        log = tmp_path / "run.log"
        shutil.copy(FIXTURES / "ngspice_ac_no_meas.log", log)
        (tmp_path / "run.exe.log").write_text(NGSPICE_42_CONSOLE)

        assert reported_build(log) == NGSPICE_42

    def test_a_failed_ngspice_run_reports_the_same_build(self, tmp_path: Path):
        # spicelib renames a failed run's log to .fail; the console capture
        # beside it is still the run's own.
        log = tmp_path / "run.fail"
        log.write_text("Circuit: * broken\n")
        (tmp_path / "run.exe.log").write_text(NGSPICE_42_CONSOLE)

        assert reported_build(log, tmp_path / "run.raw") == NGSPICE_42

    def test_the_raw_header_names_the_writer_when_the_log_does_not(self, tmp_path: Path):
        log = tmp_path / "run.log"
        shutil.copy(FIXTURES / "ngspice_ac_no_meas.log", log)

        assert (
            reported_build(log, FIXTURES / "ngspice_noise_2plot.raw")
            == "ngspice-46, Build Mar 29 2026 15:02:07"
        )

    def test_ltspice_xvii_names_itself_only_in_its_utf16_raw_header(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_bytes("Circuit: * run.net\r\n\r\n".encode("utf-16-le"))
        raw = tmp_path / "run.raw"
        header = (
            "Title: * run.net\n"
            "Date: Mon Jan 06 10:00:00 2025\n"
            "Plotname: Operating Point\n"
            "Flags: real\n"
            "No. Variables: 1\n"
            "No. Points: 1\n"
            "Offset:   0.0000000000000000e+000\n"
            "Command: Linear Technology Corporation LTspice XVII\n"
            "Variables:\n"
            "\t0\tV(out)\tvoltage\n"
            "Binary:\n"
        )
        raw.write_bytes(header.encode("utf-16-le") + b"\x00" * 8)

        assert reported_build(log, raw) == "Linear Technology Corporation LTspice XVII"

    def test_output_that_names_no_build_reports_none(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text("Circuit: * quiet\n")

        assert reported_build(log, tmp_path / "missing.raw") is None
        assert reported_build(tmp_path / "missing.log") is None
        assert reported_build(None, None) is None

    def test_only_the_head_of_an_artifact_is_read(self, tmp_path: Path):
        # Simulator output is untrusted input: a banner past the read bound is
        # not looked for, however large the file.
        log = tmp_path / "run.log"
        log.write_text("\n" * 20_000 + LTSPICE_26 + "\n")
        raw = tmp_path / "run.raw"
        raw.write_bytes(b"Title: x\n" + b"Note: filler\n" * 2_000 + b"Command: ngspice-99\n")

        assert reported_build(log, raw) is None

    def test_a_reported_build_is_one_short_line(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text("LTspice 26.0.2   for " + "Windows" * 100 + "\nCircuit: x\n")

        reported = reported_build(log)

        assert reported is not None
        assert reported.startswith("LTspice 26.0.2 for Windows")
        assert len(reported) <= 160


# ---------------------------------------------------------------------------
# Which program a simulator class launches
# ---------------------------------------------------------------------------


class TestExecutableIdentity:
    def test_the_identity_is_the_programs_own_bytes(self, tmp_path: Path):
        program = stand_in_program(tmp_path / "LTspice.exe", b"build one")

        identity = executable_identity(_simulator(_RecordedLTspice26, program))

        assert identity is not None
        assert identity.path == str(program)
        assert identity.sha256 == _sha256(program)
        assert identity.bytes == len(b"build one")
        assert identity.modified is not None

    def test_under_wine_the_program_is_the_simulator_not_the_launcher(self, tmp_path: Path):
        program = stand_in_program(tmp_path / "LTspice.exe", b"build one")

        simulator = _simulator(_RecordedLTspice26, program, "wine")

        assert executable_path(simulator) == str(program)
        identity = executable_identity(simulator)
        assert identity is not None and identity.sha256 == _sha256(program)

    def test_a_bare_command_resolves_through_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        name = "fakesim.exe" if sys.platform == "win32" else "fakesim"
        program = stand_in_program(tmp_path / "bin" / name, b"on path")
        program.chmod(0o755)
        monkeypatch.setenv("PATH", str(program.parent))
        simulator = type("Bare", (_RecordedLTspice26,), {"spice_exe": ["fakesim"]})

        identity = executable_identity(simulator)

        assert identity is not None
        assert Path(identity.path).resolve() == program.resolve()
        assert identity.sha256 == _sha256(program)

    def test_a_missing_program_keeps_only_its_path(self, tmp_path: Path):
        missing = tmp_path / "gone" / "LTspice.exe"

        identity = executable_identity(_simulator(_RecordedLTspice26, missing))

        assert identity == SimulatorExecutable(str(missing), None, None, None)

    def test_a_class_naming_no_program_has_no_identity(self):
        assert executable_identity(_RecordedLTspice26) is None
        assert executable_path(None) is None

    def test_a_rebuilt_program_is_a_different_build(self, tmp_path: Path):
        program = stand_in_program(tmp_path / "LTspice.exe", b"build one")
        simulator = _simulator(_RecordedLTspice26, program)
        before = executable_identity(simulator)

        _rebuild(program, b"build two")
        after = executable_identity(simulator)

        assert after is not None and after.sha256 == _sha256(program)
        assert not same_executable(before, after)

    def test_the_same_build_moved_is_the_same_build(self, tmp_path: Path):
        first = stand_in_program(tmp_path / "a" / "LTspice.exe", b"build one")
        moved = stand_in_program(tmp_path / "b" / "LTspice.exe", b"build one")

        assert same_executable(
            executable_identity(_simulator(_RecordedLTspice26, first)),
            executable_identity(_simulator(_RecordedLTspice26, moved)),
        )

    def test_without_digests_path_size_and_time_must_all_agree(self):
        one = SimulatorExecutable("/sim", None, 10, "2026-01-01T00:00:00-05:00")

        assert same_executable(one, SimulatorExecutable("/sim", None, 10, one.modified))
        assert not same_executable(one, SimulatorExecutable("/sim", None, 11, one.modified))
        assert not same_executable(one, SimulatorExecutable("/other", None, 10, one.modified))

    def test_an_identity_on_one_side_only_is_a_difference(self):
        one = SimulatorExecutable("/sim", "a" * 64, 10, None)

        assert not same_executable(one, None)
        assert not same_executable(None, one)
        assert same_executable(None, None)

    def test_a_recorded_identity_round_trips_and_junk_reads_as_none(self):
        one = SimulatorExecutable("/sim", "a" * 64, 10, "2026-01-01T00:00:00-05:00")

        assert SimulatorExecutable.from_record(one.to_record()) == one
        assert SimulatorExecutable.from_record(None) is None
        assert SimulatorExecutable.from_record({"path": ""}) is None
        assert SimulatorExecutable.from_record({"path": "/sim", "bytes": True}) == (
            SimulatorExecutable("/sim", None, None, None)
        )


# ---------------------------------------------------------------------------
# One run, through spicelib's real runner threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunOutcome:
    async def test_a_finished_run_carries_the_build_its_log_named(self, tmp_path: Path):
        outcome = await submit_through_spicelib(tmp_path, _RecordedLTspice26)

        assert outcome.error is None
        assert outcome.simulator_version == LTSPICE_26

    async def test_an_ngspice_run_carries_its_console_banner(self, tmp_path: Path):
        outcome = await submit_through_spicelib(tmp_path, _RecordedNgspice42)

        assert outcome.error is None
        assert outcome.simulator_version == NGSPICE_42


# ---------------------------------------------------------------------------
# The job record and what a caller reads back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecordedBuild:
    async def test_each_run_records_the_build_it_reported(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        state = _state(config, program)
        deck = work_dir / "rc.cir"
        deck.write_text(_DECK)

        receipt = await terminal_experiment(
            state,
            _payload(
                deck,
                "recorded-build",
                variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
            ),
        )

        assert receipt["status"] == "completed", receipt
        # The durable record: the program at submission, the build per case.
        await state.job_registry.drain_pending()
        record = json.loads(store.Store(work_dir).job_record(receipt["job_id"]).read_text())
        assert record["store_version"] == store.STORE_VERSION
        assert record["simulator_executable"]["path"] == str(program)
        assert record["simulator_executable"]["sha256"] == _sha256(program)
        assert [case["simulator_version"] for case in record["cases"]] == [LTSPICE_26] * 2
        # jobs(runs) reads full rows, the build among them.
        runs = await job_runs(state, receipt["job_id"])
        assert [row["simulator_version"] for row in runs] == [LTSPICE_26] * 2

    async def test_the_lean_receipt_keeps_the_build_one_opt_in_away(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        state = _state(config, program)
        deck = work_dir / "rc.cir"
        deck.write_text(_DECK)
        receipt = await terminal_experiment(state, _payload(deck, "lean-build"))
        assert receipt["status"] == "completed", receipt

        # Lean: a produced row drops it with its artifact paths.
        assert "simulator_version" not in receipt["runs"]["items"][0]
        assert "simulator_executable" not in receipt

        # Asked for: the replay under provenance names the program, and a
        # run_fields projection names the build.
        _, detailed = await submit_experiment(
            state,
            _payload(deck, "lean-build", provenance=True, run_fields=["simulator_version"]),
        )
        assert detailed["replayed"] is True
        assert detailed["simulator_executable"] == {
            "path": str(program),
            "sha256": _sha256(program),
            "bytes": len(b"build one"),
            "modified": detailed["simulator_executable"]["modified"],
        }
        assert detailed["runs"]["items"] == [{"simulator_version": LTSPICE_26}]


# ---------------------------------------------------------------------------
# Replay across a changed executable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReplayAcrossBuilds:
    """A reused request_id over a different simulator build must not return the
    earlier build's numbers.

    The fingerprint hashes the request, and nothing in the request names the
    executable, so swapping the simulator for another build and restarting left
    the fingerprint unchanged and the replay went through.
    """

    async def _first_run(
        self, config: ServerConfig, work_dir: Path, program: Path, request_id: str
    ) -> tuple[Path, dict[str, Any]]:
        state = _state(config, program)
        deck = work_dir / "rc.cir"
        deck.write_text(_DECK)
        receipt = await terminal_experiment(state, _payload(deck, request_id))
        assert receipt["status"] == "completed", receipt
        # What a restarted server reads is the record on disk.
        await state.job_registry.drain_pending()
        return deck, receipt

    async def test_a_rebuilt_executable_conflicts_after_a_restart(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        deck, first = await self._first_run(config, work_dir, program, "rebuilt")
        _rebuild(program, b"build two")

        restarted = _state(config, program)
        is_error, data = await submit_experiment(restarted, _payload(deck, "rebuilt"))

        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"
        message = data["error"]["message"]
        assert first["job_id"] in message
        assert _sha256(program)[:12] in message
        assert "control_token" not in data

    async def test_a_different_default_simulator_conflicts(
        self, config: ServerConfig, work_dir: Path
    ):
        # The request names no simulator, so it runs on whichever the server
        # defaults to; a server now defaulting to another program is a
        # different build for the same request.
        first_program = stand_in_program(work_dir / "sim" / "LTspice.exe", b"ltspice build")
        deck, _ = await self._first_run(config, work_dir, first_program, "switched")
        other = stand_in_program(work_dir / "sim" / "ngspice", b"ngspice build")

        restarted = SessionState.create(
            config, available={"ngspice": _simulator(_RecordedNgspice42, other)}
        )
        is_error, data = await submit_experiment(restarted, _payload(deck, "switched"))

        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"
        assert str(other) in data["error"]["message"]

    async def test_the_same_build_still_replays_after_a_restart(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        deck, first = await self._first_run(config, work_dir, program, "unchanged")

        restarted = _state(config, program)
        is_error, data = await submit_experiment(restarted, _payload(deck, "unchanged"))

        assert not is_error
        assert data["replayed"] is True
        assert data["job_id"] == first["job_id"]

    async def test_a_replay_needs_the_simulator_a_fresh_run_would_use(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        # With no simulator there is no build to compare the record with, so
        # the replay is answered as a fresh submission would be. The recorded
        # job is still readable.
        deck, first = await self._first_run(config, work_dir, program, "no-simulator")

        restarted = SessionState.create(config, available={})
        is_error, data = await submit_experiment(restarted, _payload(deck, "no-simulator"))
        status = await handle_jobs(
            JobsInput.model_validate({"action": "status", "request_id": "no-simulator"}),
            restarted,
        )

        assert is_error
        assert data["error"]["code"] != "idempotency_conflict"
        assert "simulator" in data["error"]["message"].lower()
        assert status.structured_content is not None
        assert status.structured_content["job_id"] == first["job_id"]

    async def test_the_request_gate_refuses_the_same_replay(
        self,
        config: ServerConfig,
        work_dir: Path,
        program: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # The pre-staging check runs before the request gate and cannot see a
        # record written after it looked; the gate re-checks under the lock.
        # Skipping the early lookup sends this replay to the gate alone.
        from ltspice_mcp.tools import experiments as experiments_mod

        deck, _ = await self._first_run(config, work_dir, program, "gated")
        _rebuild(program, b"build two")

        async def no_early_lookup(*_args: Any) -> None:
            return None

        monkeypatch.setattr(experiments_mod, "_load_matching_replay", no_early_lookup)
        restarted = _state(config, program)
        is_error, data = await submit_experiment(restarted, _payload(deck, "gated"))

        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"
        assert "sha256" in data["error"]["message"]

    async def test_a_version_2_record_reads_but_does_not_replay(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        # Version 2 of the store predates both fields. Its records still load,
        # with the build unknown, and unknown cannot be shown to match.
        deck, first = await self._first_run(config, work_dir, program, "older-record")
        path = store.Store(work_dir).job_record(first["job_id"])
        record = json.loads(path.read_text())
        record["store_version"] = 2
        del record["simulator_executable"]
        for case in record["cases"]:
            del case["simulator_version"]
        path.write_text(json.dumps(record))

        loaded = experiment_store.load_job(first["job_id"], work_dir, own_is_alive=True)
        assert loaded is not None
        assert loaded.simulator_executable is None
        assert [case.simulator_version for case in loaded.cases] == [None]

        restarted = _state(config, program)
        is_error, data = await submit_experiment(restarted, _payload(deck, "older-record"))
        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"
        assert "unrecorded executable" in data["error"]["message"]


# ---------------------------------------------------------------------------
# The capabilities report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapabilitiesBuild:
    async def test_the_version_is_what_the_last_run_on_this_executable_reported(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        state = _state(config, program)

        before = (await capabilities_report(state))["simulators"]["ltspice"]
        assert before["executable"] == str(program)
        assert before["executable_sha256"] == _sha256(program)
        assert before["version"] is None
        assert before["version_source"] is None

        deck = work_dir / "rc.cir"
        deck.write_text(_DECK)
        receipt = await terminal_experiment(state, _payload(deck, "caps-build"))
        after = (await capabilities_report(state))["simulators"]["ltspice"]

        assert after["version"] == LTSPICE_26
        assert after["version_source"] == {
            "job_id": receipt["job_id"],
            "case_id": receipt["runs"]["items"][0]["case_id"],
        }

    async def test_a_run_on_another_build_does_not_speak_for_this_one(
        self, config: ServerConfig, work_dir: Path, program: Path
    ):
        state = _state(config, program)
        deck = work_dir / "rc.cir"
        deck.write_text(_DECK)
        await terminal_experiment(state, _payload(deck, "old-build"))

        _rebuild(program, b"build two")
        caps = (await capabilities_report(state))["simulators"]["ltspice"]

        assert caps["executable_sha256"] == _sha256(program)
        assert caps["version"] is None

    async def test_under_wine_the_executable_is_the_simulator(
        self, config: ServerConfig, work_dir: Path
    ):
        program = stand_in_program(work_dir / "wine" / "LTspice.exe", b"build one")
        state = _state(config, program, "wine")

        caps = (await capabilities_report(state))["simulators"]["ltspice"]

        assert caps["executable"] == str(program)
