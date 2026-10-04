"""More than one build of a simulator family in one server, chosen per run.

``[simulator.executables]`` names further executables of a family; a run picks
one with ``execution.simulator = "family:name"``. Each named executable is a
simulator class of its own, so it never retargets the family's class (and the
runners already holding it), and its runner, launch permits, kill names,
records, library roots and exporter all follow the build the run selected.

The executables here are small files standing in for simulator programs: their
bytes are the build, and the recording simulator below writes them as the
banner its run reports, so which build ran is read back from what the run left,
as it is in production.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

import ltspice_mcp.lib.wsl as wsl_mod
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.engine import bootstrap_library_engine
from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib import proc_kill, store
from ltspice_mcp.lib.ltspice_wsl import LTspiceWSL
from ltspice_mcp.lib.proc_kill import simulator_executable_names
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    bind_named_executable,
    detect_named_simulators,
    simulator_library_roots,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import resolve_run_simulator
from ltspice_mcp.tools.experiments import RunExperimentsInput
from tests.conftest import terminal_experiment
from tests.test_parallel_sessions import _FakeProc
from tests.test_simulator_build import _capabilities, _program, _sha256, _submit

FIXTURES = Path(__file__).parent / "fixtures"
XVII_BUILD = "LTspice 17.1.8 for Windows"
LT24_BUILD = "LTspice 24.1.9 for Windows"
DEFAULT_BUILD = "LTspice 26.0.2 for Windows"
NGSPICE_BUILD = "ngspice-45"

_DECK = "* rc\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"


def _ran_on(cls: type) -> str:
    """The build the program a class launches names: the program's own bytes."""
    return Path(cls.spice_exe[-1]).read_text()


class RecordingLTspice(LTspice):
    """An LTspice whose run leaves a recorded log and raw, the log naming the
    build of the program the class launches.

    It runs through spicelib's real SimRunner threads and the server's real
    completion callback; only the process is stood in for. ``create_netlist``
    stands in for the schematic exporter, writing which program exported it.
    """

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated

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
        log = (FIXTURES / "ltspice_tran_rc.log").read_text(encoding="utf-8").splitlines()
        log[0] = _ran_on(cls)
        netlist.with_suffix(".log").write_text("\n".join(log) + "\n", encoding="utf-8")
        shutil.copy(FIXTURES / "ltspice_tran_rc.raw", netlist.with_suffix(".raw"))
        return 0

    @classmethod
    def create_netlist(
        cls,
        circuit_file,
        cmd_line_switches=None,
        timeout=None,
        stdout=None,
        stderr=None,
        cwd=None,
        exe_log=False,
    ) -> Path:
        net = Path(circuit_file).with_suffix(".net")
        exporter = Path(cls.spice_exe[-1]).name
        net.write_text(f"* exported by {exporter}\n" + _DECK.split("\n", 1)[1])
        return net


# Keeps the family's class name, as the real ngspice class's does: the name is
# what a record's dialect and the linter are chosen by.
RecordingNgspice = type(
    "NGspiceSimulator",
    (NGspiceSimulator,),
    {
        "spice_exe": [],
        "run": classmethod(
            lambda cls, netlist_file, *args, **kwargs: _ngspice_run(cls, Path(netlist_file))
        ),
    },
)


def _ngspice_run(cls: type, netlist: Path) -> int:
    shutil.copy(FIXTURES / "ngspice_ac_no_meas.log", netlist.with_suffix(".log"))
    shutil.copy(FIXTURES / "ngspice_noise_2plot.raw", netlist.with_suffix(".raw"))
    netlist.with_suffix(".exe.log").write_text(
        f"** {_ran_on(cls)} : Circuit level simulation program\n"
    )
    return 0


@pytest.fixture
def recording_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """Named executables bind to the recording classes, as they would to spicelib's.

    WSL interop is off whatever the host: the recording LTspice is an LTspice,
    which under WSL would have its runs moved to the Windows temp directory and
    its kills sent to PowerShell, neither of which these runs involve.
    """
    monkeypatch.setitem(SIMULATORS, "ltspice", RecordingLTspice)
    monkeypatch.setitem(SIMULATORS, "ngspice", RecordingNgspice)
    monkeypatch.setattr(wsl_mod, "is_wsl", lambda: False)


@pytest.fixture
def builds(work_dir: Path) -> dict[str, Path]:
    """Three LTspice builds installed side by side, at their usual file names."""
    return {
        "default": _program(work_dir / "programs" / "ADI" / "LTspice.exe", DEFAULT_BUILD.encode()),
        "xvii": _program(work_dir / "programs" / "LTC" / "XVIIx64.exe", XVII_BUILD.encode()),
        "lt24": _program(work_dir / "programs" / "LT24" / "LTspice.exe", LT24_BUILD.encode()),
    }


@pytest.fixture
def deck(work_dir: Path) -> Path:
    deck = work_dir / "rc.cir"
    deck.write_text(_DECK)
    return deck


def _state(
    config: ServerConfig, builds: dict[str, Path], executables: dict[str, Path] | None = None
) -> SessionState:
    """A session as startup builds one: the family's own executable, and the
    named ones bound from the configuration (XVII and LTspice 24 unless given)."""
    family = type("RecordingLTspice", (RecordingLTspice,), {})
    family.create_from(str(builds["default"]))
    config.simulator_executables = (
        executables
        if executables is not None
        else {"xvii": builds["xvii"], "lt24": builds["lt24"]}
    )
    diagnostics: list[str] = []
    named = detect_named_simulators(config, diagnostics)
    return SessionState.create(config, {"ltspice": family}, diagnostics, named=named)


def _payload(deck: Path, request_id: str, simulator: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 30, "simulator": simulator},
        "lint": "off",
    }


async def _record(state: SessionState, job_id: str) -> dict[str, Any]:
    await state.job_registry.drain_pending()
    return json.loads(store.Store(state.working_dir).job_record(job_id).read_text())


# ---------------------------------------------------------------------------
# Binding: one class per named executable, the family's own left alone
# ---------------------------------------------------------------------------


class TestBinding:
    def test_each_named_executable_is_a_class_launching_its_own_program(
        self, config: ServerConfig, builds: dict[str, Path]
    ):
        family = SIMULATORS["ltspice"]
        before = (list(family.spice_exe), family.process_name)
        config.simulator_executables = {"xvii": builds["xvii"], "ltspice:lt24": builds["lt24"]}
        diagnostics: list[str] = []

        named = detect_named_simulators(config, diagnostics)

        assert diagnostics == []
        assert set(named) == {"ltspice:xvii", "ltspice:lt24"}
        xvii, lt24 = named["ltspice:xvii"], named["ltspice:lt24"]
        assert Path(xvii.spice_exe[-1]) == builds["xvii"]
        assert Path(lt24.spice_exe[-1]) == builds["lt24"]
        # Binding never touches the family's class: every runner holding it,
        # in-flight cases included, still launches the program it did.
        assert (list(family.spice_exe), family.process_name) == before
        # A subclass of the family, under the family's name, so it runs,
        # parses and lints as the family does.
        assert issubclass(xvii, family) and xvii.__name__ == family.__name__

    def test_a_table_per_family_and_a_qualified_key_name_the_same_entry(
        self, work_dir: Path, builds: dict[str, Path]
    ):
        # The TOML spellings a reader would reach for, a Windows path among them.
        toml = work_dir / "ltspice-mcp.toml"
        toml.write_text(
            "[simulator.executables]\n"
            f"xvii = '{builds['xvii']}'\n"
            "[simulator.executables.ltspice]\n"
            f"lt24 = '{builds['lt24']}'\n"
        )
        config = ServerConfig.load(toml)

        named = detect_named_simulators(config, [])

        assert config.simulator_executables == {
            "xvii": builds["xvii"],
            "ltspice:lt24": builds["lt24"],
        }
        assert set(named) == {"ltspice:xvii", "ltspice:lt24"}

    def test_the_environment_form_takes_windows_paths_on_every_platform(
        self, work_dir: Path, builds: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            "LTSPICE_MCP_SIMULATOR_EXECUTABLES",
            f"xvii={builds['xvii']};ltspice:lt24={builds['lt24']};"
            r"ngspice:win=C:\Spice64\bin\ngspice_con.exe;malformed",
        )
        config = ServerConfig.load(work_dir / "absent.toml")

        assert config.simulator_executables == {
            "xvii": builds["xvii"],
            "ltspice:lt24": builds["lt24"],
            "ngspice:win": Path(r"C:\Spice64\bin\ngspice_con.exe"),
        }

    @pytest.mark.parametrize(
        ("key", "file_name", "reason"),
        [
            # The file name says nothing, so the family must be written.
            ("wrapper", "run-sim.cmd", '"<family>:wrapper"'),
            # A name that is an LTspice, bound to ngspice.
            ("ngspice:odd", "XVIIx64.exe", "looks like a ltspice executable"),
            # A family no run can be put on.
            ("qspice:q", "QSPICE64.exe", "a run can be put on"),
            # Upper case would make two spellings of one selector.
            ("ltspice:XVII!", "XVIIx64.exe", "not a valid executable name"),
        ],
    )
    def test_an_entry_that_cannot_be_bound_says_why(
        self, config: ServerConfig, work_dir: Path, key: str, file_name: str, reason: str
    ):
        exe = _program(work_dir / "programs" / file_name, b"build")
        config.simulator_executables = {key: exe}
        diagnostics: list[str] = []

        assert detect_named_simulators(config, diagnostics) == {}
        (diagnostic,) = diagnostics
        assert repr(key) in diagnostic
        assert reason in diagnostic

    def test_a_missing_program_a_folder_and_an_excluded_family_are_skipped(
        self, config: ServerConfig, work_dir: Path, builds: dict[str, Path]
    ):
        config.simulator_executables = {
            "gone": work_dir / "programs" / "missing" / "XVIIx64.exe",
            # What an empty path becomes once it is a Path: the current folder.
            "ltspice:here": Path("."),
            "ngspice:dev": _program(work_dir / "programs" / "ngspice", NGSPICE_BUILD.encode()),
        }
        config.enabled_simulators = ["ltspice"]
        diagnostics: list[str] = []

        assert detect_named_simulators(config, diagnostics) == {}
        assert any("does not exist" in note and "'gone'" in note for note in diagnostics)
        assert any("not a file" in note and "'ltspice:here'" in note for note in diagnostics)
        assert any("excluded by simulator.enabled" in note for note in diagnostics)

    async def test_the_library_api_binds_them_from_an_override(
        self, work_dir: Path, builds: dict[str, Path]
    ):
        boot = await bootstrap_library_engine(
            working_dir=work_dir,
            simulator_executables={"xvii": builds["xvii"]},
            persist_jobs=False,
            preload_recent_count=0,
        )
        try:
            assert list(boot.state.named_simulators) == ["ltspice:xvii"]
        finally:
            await boot.state.shutdown()

        with pytest.raises(TypeError, match="mapping of name to path"):
            await bootstrap_library_engine(
                working_dir=work_dir, simulator_executables=[str(builds["xvii"])]
            )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.parametrize("value", ["ltspice", "ngspice", "ltspice:xvii", "ngspice:dev-1.2"])
    def test_the_field_takes_a_family_or_a_family_and_a_name(self, value: str):
        args = RunExperimentsInput.model_validate(
            {"circuits": [{"path": "a.cir"}], "execution": {"simulator": value}}
        )
        assert args.execution.simulator == value

    @pytest.mark.parametrize("value", ["qspice", "qspice:x", "LTspice:XVII", "ltspice:", "xvii"])
    def test_the_field_refuses_anything_else(self, value: str):
        with pytest.raises(ValidationError):
            RunExperimentsInput.model_validate(
                {"circuits": [{"path": "a.cir"}], "execution": {"simulator": value}}
            )

    def test_a_family_keeps_its_own_executable_and_a_name_its_build(
        self,
        config: ServerConfig,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds, {"xvii": builds["xvii"]})

        assert resolve_run_simulator(None, state) is state.available_simulators["ltspice"]
        assert resolve_run_simulator("ltspice", state) is state.available_simulators["ltspice"]
        assert (
            resolve_run_simulator("ltspice:xvii", state) is state.named_simulators["ltspice:xvii"]
        )
        with pytest.raises(SimulationError, match=r"named: \['ltspice:xvii'\]"):
            resolve_run_simulator("ltspice:lt24", state)

    async def test_an_unknown_name_fails_the_call_naming_what_there_is(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds, {"xvii": builds["xvii"]})

        is_error, data = await _submit(state, _payload(deck, "unknown", "ltspice:lt24"))

        assert is_error
        assert data["error"]["commit_state"] == "not_started"
        assert "ltspice:xvii" in data["error"]["message"]


# ---------------------------------------------------------------------------
# Runs routed to each build, and what the records say
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRoutedRuns:
    async def test_each_run_launches_the_build_it_named_and_its_record_says_so(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds)

        receipts = {
            simulator: await terminal_experiment(
                state, _payload(deck, f"run-{simulator}", simulator)
            )
            for simulator in ("ltspice:xvii", "ltspice:lt24", "ltspice")
        }

        expected = {
            "ltspice:xvii": (builds["xvii"], XVII_BUILD),
            "ltspice:lt24": (builds["lt24"], LT24_BUILD),
            "ltspice": (builds["default"], DEFAULT_BUILD),
        }
        for simulator, receipt in receipts.items():
            assert receipt["status"] == "completed", receipt
            program, build = expected[simulator]
            record = await _record(state, receipt["job_id"])
            assert record["simulator_executable"]["path"] == str(program)
            assert record["simulator_executable"]["sha256"] == _sha256(program)
            assert [case["simulator_version"] for case in record["cases"]] == [build]
            # The family's name and dialect, whichever build ran.
            assert record["simulator"] == "RecordingLTspice"

        # One runner per build, the family's own among them, each holding the
        # server's whole cap: a named build never shares or evicts its permits.
        runners = {
            simulator_class: runner
            for (kind, simulator_class, _folder), runner in state.runners._runners.items()
            if kind == "experiment"
        }
        assert set(runners) == {
            state.available_simulators["ltspice"],
            state.named_simulators["ltspice:xvii"],
            state.named_simulators["ltspice:lt24"],
        }
        assert len({id(runner) for runner in runners.values()}) == 3
        assert all(runner.max_parallel == config.max_parallel_sims for runner in runners.values())

    async def test_the_capabilities_report_lists_each_build_and_what_it_reported(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, "caps", "ltspice:xvii"))

        caps = await _capabilities(state)

        named = caps["named_executables"]
        assert set(named) == {"ltspice:xvii", "ltspice:lt24"}
        assert named["ltspice:xvii"] == {
            "family": "ltspice",
            "version": XVII_BUILD,
            "version_source": {
                "job_id": receipt["job_id"],
                "case_id": receipt["runs"]["items"][0]["case_id"],
            },
            "dialect": None,
            "executable": str(builds["xvii"]),
            "executable_sha256": _sha256(builds["xvii"]),
        }
        # A run on one build never speaks for another, nor for the family's own.
        assert named["ltspice:lt24"]["version"] is None
        assert named["ltspice:lt24"]["executable"] == str(builds["lt24"])
        assert caps["simulators"]["ltspice"]["version"] is None
        assert caps["simulators"]["ltspice"]["executable"] == str(builds["default"])

    async def test_a_schematic_is_exported_by_the_build_that_runs_it(
        self,
        config: ServerConfig,
        work_dir: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        state = _state(config, builds, {"xvii": builds["xvii"]})
        sheet = work_dir / "rc.asc"
        sheet.write_text("Version 4\nSHEET 1 880 680\n")

        receipt = await terminal_experiment(state, _payload(sheet, "sheet", "ltspice:xvii"))

        assert receipt["status"] == "completed", receipt
        record = await _record(state, receipt["job_id"])
        (source,) = record["sources"]
        staged = await asyncio.to_thread(Path(source["staged_deck"]).read_text)
        assert staged.startswith("* exported by XVIIx64.exe")

    async def test_dialect_and_lint_follow_the_family_of_the_named_build(
        self,
        config: ServerConfig,
        work_dir: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        ngspice = _program(
            work_dir / "programs" / "ngspice-dev" / "ngspice", NGSPICE_BUILD.encode()
        )
        state = _state(config, builds, {"xvii": builds["xvii"], "ngspice:dev": ngspice})
        assert state.diagnostics == []
        # A swept deck: ngspice reads no .step, and the linter says so for it alone.
        deck = work_dir / "swept.cir"
        deck.write_text(_DECK.replace(".tran 1m", ".step param r 1k 2k 1k\n.tran 1m"))

        findings: dict[str, list[str]] = {}
        dialects: dict[str, str | None] = {}
        for simulator in ("ngspice:dev", "ltspice:xvii"):
            receipt = await terminal_experiment(
                state, {**_payload(deck, f"lint-{simulator}", simulator), "lint": "warn"}
            )
            (source,) = (await _record(state, receipt["job_id"]))["sources"]
            findings[simulator] = [finding["rule_id"] for finding in source["lint_findings"]]
            dialects[simulator] = source["dialect"]

        assert dialects == {"ngspice:dev": "ngspice", "ltspice:xvii": None}
        assert "step-ngspice" in findings["ngspice:dev"]
        assert "step-ngspice" not in findings["ltspice:xvii"]


# ---------------------------------------------------------------------------
# Replay across names and across builds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReplayAcrossNames:
    async def _first(
        self, config: ServerConfig, deck: Path, builds: dict[str, Path], request_id: str
    ) -> dict[str, Any]:
        state = _state(config, builds)
        receipt = await terminal_experiment(state, _payload(deck, request_id, "ltspice:xvii"))
        assert receipt["status"] == "completed", receipt
        await state.job_registry.drain_pending()
        return receipt

    async def test_a_request_id_replayed_on_another_name_conflicts(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        await self._first(config, deck, builds, "across-names")
        restarted = _state(config, builds)

        is_error, data = await _submit(restarted, _payload(deck, "across-names", "ltspice:lt24"))

        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"

    async def test_a_name_bound_to_another_build_conflicts(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        first = await self._first(config, deck, builds, "rebound")
        # The same name, now pointing at the other install.
        restarted = _state(config, builds, {"xvii": builds["lt24"]})

        is_error, data = await _submit(restarted, _payload(deck, "rebound", "ltspice:xvii"))

        assert is_error
        assert data["error"]["code"] == "idempotency_conflict"
        message = data["error"]["message"]
        assert first["job_id"] in message
        assert _sha256(builds["lt24"])[:12] in message

    async def test_the_same_name_on_the_same_build_replays(
        self,
        config: ServerConfig,
        deck: Path,
        builds: dict[str, Path],
        recording_families: None,
    ):
        first = await self._first(config, deck, builds, "same")
        restarted = _state(config, builds, {"xvii": builds["xvii"]})

        is_error, data = await _submit(restarted, _payload(deck, "same", "ltspice:xvii"))

        assert not is_error
        assert data["replayed"] is True
        assert data["job_id"] == first["job_id"]


# ---------------------------------------------------------------------------
# Stopping a run: the kill names the build the runner launched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScopedKill:
    async def test_each_runner_kills_its_own_builds_process(
        self,
        config: ServerConfig,
        work_dir: Path,
        builds: dict[str, Path],
        recording_families: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        state = _state(config, builds)
        loop = asyncio.get_running_loop()
        runners = {
            selector: state.runners.get_experiment_runner(
                loop, cls, work_dir / "runs", config.max_parallel_sims
            )
            for selector, cls in state.named_simulators.items()
        }
        xvii_job, lt24_job = "exp_dut_1790000000_ab12cd34", "exp_dut_1790000001_cd34ef56"
        xvii = _FakeProc(11, "XVIIx64.exe", [str(builds["xvii"]), "-Run", "-b", f"{xvii_job}.net"])
        lt24 = _FakeProc(12, "LTspice.exe", [str(builds["lt24"]), "-Run", "-b", f"{lt24_job}.net"])
        monkeypatch.setattr(proc_kill.psutil, "process_iter", lambda attrs: iter([xvii, lt24]))

        assert simulator_executable_names(runners["ltspice:xvii"].simulator_class) == {
            "xviix64.exe"
        }
        # The other build's runner, handed this job's token, does not know the
        # program by name: the gate is each runner's own executable.
        runners["ltspice:lt24"]._kill_by_token(xvii_job)
        assert not xvii.killed

        runners["ltspice:xvii"]._kill_by_token(xvii_job)
        runners["ltspice:lt24"]._kill_by_token(lt24_job)
        assert xvii.killed and lt24.killed


# ---------------------------------------------------------------------------
# WSL: the Windows program runs and dies per executable
# ---------------------------------------------------------------------------


class TestWsl:
    def test_a_named_build_launches_its_own_windows_program(
        self, work_dir: Path, builds: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ):
        family_before = list(LTspiceWSL.spice_exe)
        xvii = bind_named_executable(LTspiceWSL, "ltspice:xvii", builds["xvii"])
        launched: list[list[str]] = []
        monkeypatch.setattr("ltspice_mcp.lib.ltspice_wsl.is_wsl", lambda: True)
        monkeypatch.setattr(
            "ltspice_mcp.lib.ltspice_wsl.to_windows_path", lambda path: r"C:\runs\deck.net"
        )
        monkeypatch.setattr(
            "spicelib.sim.simulator.run_function",
            lambda command, **kwargs: launched.append(command) or 0,
        )

        assert xvii.run(work_dir / "deck.net") == 0

        (command,) = launched
        assert Path(command[0]) == builds["xvii"]
        assert command[1:] == ["-Run", "-b", r"C:\runs\deck.net"]
        assert list(LTspiceWSL.spice_exe) == family_before

    def test_the_windows_kill_also_matches_a_named_programs_own_file_name(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        queries: list[str] = []

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            queries.append(command[-1])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(wsl_mod, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl_mod.subprocess, "run", fake_run)

        wsl_mod.kill_windows_ltspice_by_token(
            "exp_dut_1790000000_ab12cd34",
            {"ltspice-24.1.exe", "xviix64.exe", "wine", "x'.exe"},
        )

        (query,) = queries
        assert "Name='ltspice-24.1.exe'" in query
        # LTspice's own names stay matched beside the class's.
        assert "Name='XVIIx64.exe'" in query and "Name='scad3.exe'" in query
        # Neither a launcher nor a name that could close the quoted string.
        assert "'wine'" not in query and "x'.exe" not in query


# ---------------------------------------------------------------------------
# Library roots: the model library of the build that runs
# ---------------------------------------------------------------------------


class TestLibraryRoots:
    @pytest.fixture
    def home(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A user profile holding both builds' libraries where each keeps its own."""
        home = work_dir / "home"
        for library in (
            home / "Documents" / "LTspiceXVII" / "lib",
            home / "AppData" / "Local" / "LTspice" / "lib",
        ):
            library.mkdir(parents=True)
        # expanduser reads USERPROFILE on Windows and HOME elsewhere.
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setattr(wsl_mod, "is_wsl", lambda: False)
        return home

    def test_each_build_stages_against_its_own_library(self, home: Path, builds: dict[str, Path]):
        xvii = bind_named_executable(LTspice, "ltspice:xvii", builds["xvii"])
        lt24 = bind_named_executable(LTspice, "ltspice:lt24", builds["lt24"])

        assert simulator_library_roots(xvii) == [
            (home / "Documents" / "LTspiceXVII" / "lib").resolve()
        ]
        assert simulator_library_roots(lt24) == [
            (home / "AppData" / "Local" / "LTspice" / "lib").resolve()
        ]

    def test_on_wsl_xvii_finds_its_library_in_the_windows_profile(
        self, work_dir: Path, builds: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ):
        profile = work_dir / "mnt" / "c" / "Users" / "someone"
        local = profile / "AppData" / "Local"
        (profile / "Documents" / "LTspiceXVII" / "lib" / "sym").mkdir(parents=True)
        (local / "LTspice" / "lib" / "sym").mkdir(parents=True)
        windows_env = {"USERPROFILE": profile, "LOCALAPPDATA": local}
        monkeypatch.setattr(wsl_mod, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl_mod, "_resolve_win_env", windows_env.get)
        monkeypatch.setenv("HOME", str(work_dir / "linux-home"))
        xvii = bind_named_executable(LTspiceWSL, "ltspice:xvii", builds["xvii"])
        lt24 = bind_named_executable(LTspiceWSL, "ltspice:lt24", builds["lt24"])

        assert simulator_library_roots(xvii) == [
            (profile / "Documents" / "LTspiceXVII" / "lib").resolve()
        ]
        assert simulator_library_roots(lt24) == [(local / "LTspice" / "lib").resolve()]
