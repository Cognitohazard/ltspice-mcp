"""Every simulator family the server supports is one a run can select.

``execution.simulator`` once named only LTspice and ngspice, so QSPICE and Xyce
ran only as the server default. The round trips here submit through
``run_experiments`` on a session whose QSPICE or Xyce class is a spicelib
simulator standing in for the program: its ``run`` writes the artifacts that
family writes, and everything around it (spicelib's SimRunner and RunTask
threads, the server's completion callback, the job record, the attached
analysis) is the production path. The stand-ins carry the spicelib class's own
name because the job records that name and reads its raw dialect back from it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, ClassVar, get_args

import pytest
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import simulator as simulator_mod
from ltspice_mcp.lib import store
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    SimulatorName,
    simulator_remediation,
)
from ltspice_mcp.lib.simulator_build import reported_build
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests.conftest import FakeSim, resolve_local_ref, terminal_experiment

# The banner Xyce 7 opens its log with (the -l file spicelib asks for), in the
# shape Xyce's startup code prints it.
XYCE_LOG = (
    "\n"
    "*****\n"
    "***** Welcome to the Xyce(TM) Parallel Electronic Simulator\n"
    "*****\n"
    "***** This is version Xyce Release 7.8.0-opensource\n"
    "***** Date: Thu Oct  1 12:00:00 2026\n"
    "*****\n"
    "***** Executing netlist rc.cir\n"
    "*****\n"
    "\n"
    "***** Reading and parsing netlist...\n"
)
XYCE_BUILD = "Xyce Release 7.8.0-opensource"
QSPICE_COMMAND = "QSPICE64"

_DECK = "* rc\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 159.155n\n.ac dec 1 10 100k\n.end\n"

# A first-order low-pass with its corner at 1 kHz, sampled where the answer is
# known: |H| = 1/sqrt(2) and -45 degrees at the corner.
_FREQUENCIES = (10.0, 1_000.0, 100_000.0)


def _low_pass(frequency: float) -> complex:
    return 1 / (1 + 1j * frequency / 1_000.0)


def _qspice_ac_qraw() -> bytes:
    """A QSPICE AC plot: ASCII header naming its writer, a double frequency axis.

    QSPICE writes the frequency as a plain double where every other simulator
    writes it complex, so the same bytes read under another dialect misalign
    every record after the first.
    """
    header = (
        "Title: * rc\n"
        "Date: Thu Oct  1 12:00:00 2026\n"
        "Plotname: AC Analysis\n"
        "Flags: complex\n"
        "No. Variables: 2\n"
        f"No. Points: {len(_FREQUENCIES)}\n"
        f"Command: {QSPICE_COMMAND}\n"
        "Variables:\n"
        "\t0\tFrequency\tfrequency\n"
        "\t1\tV(out)\tvoltage\n"
        "Binary:\n"
    ).encode("ascii")
    body = b"".join(
        struct.pack("<3d", f, _low_pass(f).real, _low_pass(f).imag) for f in _FREQUENCIES
    )
    return header + body


_TIMES = (0.0, 1e-3, 2e-3)
_VOUT = (0.0, 0.632, 0.865)


def _xyce_tran_raw() -> bytes:
    """A Xyce transient plot: every value a double, and no ``Command:`` field.

    Xyce 7.9 does not name itself in the raw, so nothing in these bytes says
    which simulator wrote them; only the job's recorded simulator does.
    """
    header = (
        "Title: * rc\n"
        "Date: Thu Oct  1 12:00:00 2026\n"
        "Plotname: Transient Analysis\n"
        "Flags: real\n"
        "No. Variables: 2\n"
        f"No. Points: {len(_TIMES)}\n"
        "Variables:\n"
        "\t0\tTIME\ttime\n"
        "\t1\tV(OUT)\tvoltage\n"
        "Binary:\n"
    ).encode("ascii")
    return header + b"".join(struct.pack("<2d", t, v) for t, v in zip(_TIMES, _VOUT, strict=True))


class _RecordedQspice(Qspice):
    """A QSPICE run: the ``.qraw`` and console log beside the deck spicelib staged."""

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated
    process_name = "QSPICE64.exe"
    launched: ClassVar[list[str]] = []

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
        cls.launched.append(netlist.name)
        # spicelib reads the raw from the deck's path with the class's own
        # raw_extension, so this is where a real run's result is looked for.
        netlist.with_suffix(cls.raw_extension).write_bytes(_qspice_ac_qraw())
        netlist.with_suffix(".log").write_text("Simulation completed.\n")
        return 0


class _RecordedXyce(XyceSimulator):
    """A Xyce run: the raw and the ``-l`` log beside the deck spicelib staged."""

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated
    process_name = "Xyce"
    launched: ClassVar[list[str]] = []

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
        cls.launched.append(netlist.name)
        netlist.with_suffix(".raw").write_bytes(_xyce_tran_raw())
        netlist.with_suffix(".log").write_text(XYCE_LOG)
        return 0


def _named_as_spicelib(base: type, program: Path) -> type:
    """A fresh subclass of ``base`` launching ``program``, under spicelib's class name."""
    spicelib_name = base.__mro__[1].__name__
    return type(spicelib_name, (base,), {"spice_exe": [str(program)], "launched": []})


def _program(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"a build of " + path.name.encode())
    return path


@pytest.fixture
def windows_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """This host is native Windows, the one place QSPICE runs."""
    monkeypatch.setattr(simulator_mod, "_platform_key", lambda: "windows")


@pytest.fixture
def qspice(work_dir: Path) -> type:
    return _named_as_spicelib(_RecordedQspice, _program(work_dir / "sim" / "QSPICE64.exe"))


@pytest.fixture
def xyce(work_dir: Path) -> type:
    return _named_as_spicelib(_RecordedXyce, _program(work_dir / "sim" / "Xyce"))


def _state(config: ServerConfig, **families: type) -> SessionState:
    """A session whose default is a stub, so every run here selects its family."""
    return SessionState.create(config, available={"ltspice": FakeSim, **families})


def _deck(work_dir: Path, name: str = "rc.cir") -> Path:
    deck = work_dir / name
    deck.write_text(_DECK)
    return deck


def _payload(deck: Path, request_id: str, simulator: str, **extra: Any) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 30, "simulator": simulator},
        **extra,
    }


async def _submit(state: SessionState, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    result = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)
    data = result.structured_content
    assert data is not None, result.content[0].text
    return bool(result.is_error), data


async def _runs(state: SessionState, job_id: str) -> list[dict[str, Any]]:
    result = await handle_jobs(
        JobsInput.model_validate({"action": "runs", "job_id": job_id}), state
    )
    assert result.structured_content is not None
    return result.structured_content["items"]


async def _capabilities(state: SessionState) -> dict[str, Any]:
    result = await handle_inspect(
        InspectInput.model_validate({"queries": [{"kind": "capabilities"}]}), state
    )
    assert result.structured_content is not None
    (item,) = result.structured_content["results"]
    assert item["ok"], item
    return item["data"]


def _record(work_dir: Path, job_id: str) -> dict[str, Any]:
    return json.loads(store.Store(work_dir).job_record(job_id).read_text())


# ---------------------------------------------------------------------------
# What the request schema names
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_named_families_are_the_supported_ones(self):
        assert get_args(SimulatorName) == tuple(SIMULATORS)

    def test_the_published_schema_enumerates_every_family(self):
        defs, _ = get_tools()
        (tool,) = [tool for tool in defs if tool.name == "run_experiments"]
        schema = tool.input_schema
        execution = resolve_local_ref(schema, schema["properties"]["execution"])
        simulator = execution["properties"]["simulator"]
        enums = [branch["enum"] for branch in simulator["anyOf"] if "enum" in branch]
        assert enums == [list(SIMULATORS)]

    @pytest.mark.parametrize("name", ["qspice", "xyce"])
    def test_qspice_and_xyce_are_accepted_by_the_request_model(self, name: str):
        args = RunExperimentsInput.model_validate(
            {"circuits": [{"path": "rc.cir"}], "execution": {"simulator": name}}
        )
        assert args.execution.simulator == name


# ---------------------------------------------------------------------------
# A run on each newly selectable family, read back through its own dialect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRoundTrip:
    async def test_a_qspice_run_parses_its_qraw_with_the_qspice_dialect(
        self, config: ServerConfig, work_dir: Path, qspice: type, windows_host: None
    ):
        state = _state(config, qspice=qspice)
        receipt = await terminal_experiment(
            state,
            _payload(
                _deck(work_dir),
                "qspice-ac",
                "qspice",
                provenance=True,
                analyze={
                    "recipes": [
                        {"key": "corner", "metric": "value", "expr": "V(out)", "at": "1k"}
                    ],
                    "include": {"per_run": {"limit": 10}},
                },
            ),
        )

        assert receipt["status"] == "completed", receipt
        assert qspice.launched, "the QSPICE stand-in was never launched"
        (source,) = receipt["source"]
        assert (source["simulator"], source["dialect"]) == ("Qspice", "qspice")
        # The value at the corner is right only if the frequency axis was read
        # as QSPICE writes it: a plain double, not a complex pair.
        stage = receipt["analysis"]
        assert stage["error"] is None, stage
        (row,) = stage["result"]["results"]["corner"]["per_run"]["items"]
        assert row["value"]["actual_x"] == pytest.approx(1_000.0)
        assert row["value"]["magnitude_linear"] == pytest.approx(2**-0.5)
        assert row["value"]["phase_deg"] == pytest.approx(-45.0)

        (run,) = await _runs(state, receipt["job_id"])
        assert run["raw"].endswith(".qraw")
        # QSPICE names its build in the raw's Command field.
        assert run["simulator_version"] == QSPICE_COMMAND
        await state.job_registry.drain_pending()
        record = _record(work_dir, receipt["job_id"])
        assert record["simulator"] == "Qspice"
        assert record["simulator_executable"]["path"] == qspice.spice_exe[-1]

    async def test_a_xyce_run_parses_with_the_dialect_its_job_recorded(
        self, config: ServerConfig, work_dir: Path, xyce: type
    ):
        state = _state(config, xyce=xyce)
        receipt = await terminal_experiment(
            state,
            _payload(
                _deck(work_dir),
                "xyce-tran",
                "xyce",
                provenance=True,
                analyze={
                    "recipes": [{"key": "vout", "metric": "value", "expr": "V(OUT)", "at": "2m"}],
                    "include": {"per_run": {"limit": 10}},
                },
            ),
        )

        assert receipt["status"] == "completed", receipt
        assert xyce.launched, "the Xyce stand-in was never launched"
        (source,) = receipt["source"]
        assert (source["simulator"], source["dialect"]) == ("XyceSimulator", "xyce")
        # Nothing in the raw names Xyce, so it parses only because the job's
        # recorded simulator supplies the dialect.
        stage = receipt["analysis"]
        assert stage["error"] is None, stage
        (row,) = stage["result"]["results"]["vout"]["per_run"]["items"]
        assert row["value"]["actual_x"] == pytest.approx(2e-3)
        assert row["value"]["value"] == pytest.approx(0.865)

        (run,) = await _runs(state, receipt["job_id"])
        assert run["simulator_version"] == XYCE_BUILD
        await state.job_registry.drain_pending()
        record = _record(work_dir, receipt["job_id"])
        assert record["simulator"] == "XyceSimulator"
        assert record["simulator_executable"]["path"] == xyce.spice_exe[-1]


# ---------------------------------------------------------------------------
# Refusals: not detected, not runnable on this host, not runnable for this file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRefusal:
    async def test_an_undetected_family_is_refused_naming_what_was_detected(
        self, config: ServerConfig, work_dir: Path
    ):
        state = _state(config)
        is_error, data = await _submit(state, _payload(_deck(work_dir), "no-xyce", "xyce"))

        assert is_error
        assert data["error"]["commit_state"] == "not_started"
        message = data["error"]["message"]
        assert "'xyce' is not available on this server" in message
        assert "detected: ['ltspice']" in message

    async def test_qspice_off_windows_is_refused_with_the_reason(
        self, config: ServerConfig, work_dir: Path, qspice: type, monkeypatch
    ):
        monkeypatch.setattr(simulator_mod, "_platform_key", lambda: "wsl")
        state = _state(config, qspice=qspice)
        is_error, data = await _submit(state, _payload(_deck(work_dir), "qspice-wsl", "qspice"))

        assert is_error
        assert data["error"]["commit_state"] == "not_started"
        message = data["error"]["message"]
        assert "'qspice' cannot run here" in message
        assert "only when this server itself runs on Windows" in message
        assert "under WSL" in message
        assert "Selectable here: ['ltspice']" in message
        assert qspice.launched == []

    async def test_a_refused_default_is_refused_too(
        self, config: ServerConfig, work_dir: Path, qspice: type, monkeypatch
    ):
        monkeypatch.setattr(simulator_mod, "_platform_key", lambda: "linux")
        config.simulator = "qspice"
        state = SessionState.create(config, available={"qspice": qspice})
        assert state.default_simulator is qspice
        payload = _payload(_deck(work_dir), "qspice-default", "qspice")
        del payload["execution"]["simulator"]

        is_error, data = await _submit(state, payload)

        assert is_error
        message = data["error"]["message"]
        assert "The server's default simulator 'qspice' cannot run here" in message
        assert "under Wine" in message
        assert qspice.launched == []

    @pytest.mark.parametrize(("family", "display"), [("xyce", "Xyce"), ("qspice", "QSPICE")])
    async def test_a_schematic_is_refused_before_ltspice_exports_it(
        self,
        config: ServerConfig,
        work_dir: Path,
        qspice: type,
        xyce: type,
        windows_host: None,
        family: str,
        display: str,
    ):
        exports: list[str] = []

        class _RecordingLTspice(FakeSim):
            @classmethod
            def create_netlist(cls, asc_file: str, **_kwargs: Any) -> Path:
                exports.append(asc_file)
                raise AssertionError("LTspice was asked to export a schematic for " + family)

        state = SessionState.create(
            config, available={"ltspice": _RecordingLTspice, "qspice": qspice, "xyce": xyce}
        )
        sheet = work_dir / "rc.asc"
        sheet.write_text("Version 4\nSHEET 1 880 680\n")

        receipt = await terminal_experiment(state, _payload(sheet, f"asc-{family}", family))

        assert receipt["completeness"]["failed"] == receipt["completeness"]["expanded"] == 1
        (failure,) = receipt["failures"]
        assert failure["code"] == "asc_export_unavailable"
        assert f"prepared for {display}" in failure["message"]
        assert "hand-written .cir/.net/.sp" in failure["message"]
        assert exports == []


# ---------------------------------------------------------------------------
# Lint keyed on the family the deck runs on
# ---------------------------------------------------------------------------


class TestLint:
    _KEYED_CAPACITOR = "* keyed\nV1 in 0 1\nR1 in out 1k\nC1 out 0 C=1u\n.tran 1m\n.end\n"

    def _arity(self, tmp_path: Path, dialect: str | None, simulator: type) -> list[str]:
        deck = tmp_path / "keyed.cir"
        deck.write_text(self._KEYED_CAPACITOR)
        findings = lint_deck(deck.read_text(), deck, dialect, simulator)
        return [f["subject"] for f in findings if f["rule_id"] == "directive-arity"]

    def test_ltspice_rejects_a_keyed_capacitor_value(self, tmp_path: Path):
        assert self._arity(tmp_path, None, SIMULATORS["ltspice"]) != []

    @pytest.mark.parametrize(
        ("dialect", "simulator"), [("qspice", Qspice), ("xyce", XyceSimulator)]
    )
    def test_the_ltspice_only_rule_is_not_applied_to_other_families(
        self, tmp_path: Path, dialect: str, simulator: type
    ):
        assert self._arity(tmp_path, dialect, simulator) == []


# ---------------------------------------------------------------------------
# What a run reports about the build that ran it
# ---------------------------------------------------------------------------


class TestReportedBuild:
    def test_xyce_names_its_release_in_its_log(self, tmp_path: Path):
        log = tmp_path / "run.log"
        log.write_text(XYCE_LOG)

        assert reported_build(log, tmp_path / "run.raw") == XYCE_BUILD

    def test_a_failed_xyce_run_reports_the_same_build(self, tmp_path: Path):
        # spicelib renames a failed run's log to .fail; the banner is still in it.
        log = tmp_path / "run.fail"
        log.write_text(XYCE_LOG + "Netlist error in file rc.cir at or near line 3\n")

        assert reported_build(log) == XYCE_BUILD


# ---------------------------------------------------------------------------
# What inspect says a run can select
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapabilities:
    async def test_each_simulator_says_whether_a_run_can_select_it(
        self, config: ServerConfig, qspice: type, xyce: type, monkeypatch
    ):
        monkeypatch.setattr(simulator_mod, "_platform_key", lambda: "linux")
        state = _state(config, qspice=qspice, xyce=xyce)

        simulators = (await _capabilities(state))["simulators"]

        assert simulators["xyce"]["selectable"] is True
        assert "refusal" not in simulators["xyce"]
        assert simulators["ltspice"]["selectable"] is True
        # Detected, and still not one a run can name here.
        assert simulators["qspice"]["available"] is True
        assert simulators["qspice"]["selectable"] is False
        assert "only when this server itself runs on Windows" in simulators["qspice"]["refusal"]
        # Not detected: nothing to select.
        assert simulators["ngspice"]["available"] is False
        assert simulators["ngspice"]["selectable"] is False

    async def test_qspice_is_selectable_on_windows(
        self, config: ServerConfig, qspice: type, windows_host: None
    ):
        simulators = (await _capabilities(_state(config, qspice=qspice)))["simulators"]

        assert simulators["qspice"]["selectable"] is True
        assert "refusal" not in simulators["qspice"]


@pytest.mark.asyncio
class TestReference:
    @pytest.mark.parametrize("query", ["xyce", "run on qspice", "choose the simulator"])
    async def test_a_family_name_finds_where_a_run_selects_it(
        self, state_with_sim: SessionState, query: str
    ):
        result = await handle_inspect(
            InspectInput.model_validate({"queries": [{"kind": "reference", "query": query}]}),
            state_with_sim,
        )
        assert result.structured_content is not None
        (item,) = result.structured_content["results"]
        top = item["data"]["matches"][0]

        assert (top["tool"], top["family"]) == ("run_experiments", "argument")
        (field,) = [field for field in top["fields"] if field["name"] == "execution.simulator"]
        assert all(repr(name) in field["type"] for name in SIMULATORS), field


class TestRemediation:
    def test_off_windows_qspice_is_not_sent_to_an_install(self, config: ServerConfig, monkeypatch):
        monkeypatch.setattr(simulator_mod, "_platform_key", lambda: "wsl")

        remediation = simulator_remediation("qspice", config)

        assert "example_value" not in remediation
        assert "only when this server itself runs on Windows" in str(remediation["action"])
        assert "restart" not in str(remediation["action"])

    def test_windows_names_where_each_family_installs(self, config: ServerConfig, windows_host):
        examples = {
            name: simulator_remediation(name, config)["example_value"] for name in SIMULATORS
        }

        assert examples["qspice"] == "C:\\Program Files\\QSPICE\\QSPICE64.exe"
        assert examples["xyce"] == "C:\\Program Files\\Xyce 7.9 NORAD\\bin\\xyce.exe"
