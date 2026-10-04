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
import re
import struct
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import simulator as simulator_mod
from ltspice_mcp.lib import store
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    detect_named_simulators,
    simulator_family,
    simulator_remediation,
)
from ltspice_mcp.lib.simulator_build import reported_build
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from ltspice_mcp.tools.experiments import RunExperimentsInput
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests.conftest import (
    FakeSim,
    capabilities_report,
    job_runs,
    ngspice_binary_raw,
    resolve_local_ref,
    stand_in_program,
    submit_experiment,
    terminal_experiment,
)

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

# A Xyce transient plot has ngspice's binary layout, every value a double, and
# Xyce 7.9 writes no ``Command:`` field: nothing in these bytes says which
# simulator wrote them, only the job's recorded simulator does.
_XYCE_TRAN_RAW = ngspice_binary_raw(
    list(zip(_TIMES, _VOUT, strict=True)), ["TIME", "V(OUT)"], declared=len(_TIMES)
)


def _stand_in(base: type, program: Path, raw: bytes, log: str) -> type:
    """A subclass of spicelib's ``base`` whose every run leaves ``raw`` and ``log``.

    It keeps spicelib's class name, because the job records that name and reads
    its raw dialect back from it, and it launches ``program``, whose identity
    the job records. The raw lands where spicelib looks for it: beside the deck
    spicelib staged, with the class's own ``raw_extension`` (``.qraw`` for
    QSPICE). ``launched`` lists the decks it ran.
    """
    launched: list[str] = []

    def run(cls: Any, netlist_file: Any, *_args: Any, **_kwargs: Any) -> int:
        netlist = Path(netlist_file)
        launched.append(netlist.name)
        netlist.with_suffix(cls.raw_extension).write_bytes(raw)
        netlist.with_suffix(".log").write_text(log)
        return 0

    return type(
        base.__name__,
        (base,),
        {
            "spice_exe": [str(program)],
            "process_name": program.name,
            "launched": launched,
            "run": classmethod(run),
        },
    )


def _host(monkeypatch: pytest.MonkeyPatch, platform_key: str) -> None:
    """Run as if the server ran on ``platform_key``: windows, wsl, linux or darwin."""
    monkeypatch.setattr(simulator_mod, "_platform_key", lambda: platform_key)


@pytest.fixture
def windows_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """This host is native Windows, the one place QSPICE runs."""
    _host(monkeypatch, "windows")


@pytest.fixture
def qspice(work_dir: Path) -> type:
    program = stand_in_program(work_dir / "sim" / "QSPICE64.exe", b"a QSPICE build")
    return _stand_in(Qspice, program, _qspice_ac_qraw(), "Simulation completed.\n")


@pytest.fixture
def xyce(work_dir: Path) -> type:
    program = stand_in_program(work_dir / "sim" / "Xyce", b"a Xyce build")
    return _stand_in(XyceSimulator, program, _XYCE_TRAN_RAW, XYCE_LOG)


def _state(config: ServerConfig, **families: type) -> SessionState:
    """A session whose default is a stub, so every run here selects its family."""
    return SessionState.create(config, available={"ltspice": FakeSim, **families})


def _deck(work_dir: Path, name: str = "rc.cir") -> Path:
    deck = work_dir / name
    deck.write_text(_DECK)
    return deck


def _payload(deck: Path, request_id: str, simulator: str | None, **extra: Any) -> dict[str, Any]:
    """A one-circuit request; ``simulator`` None leaves the choice to the server."""
    execution: dict[str, Any] = {"wait_s": 30}
    if simulator is not None:
        execution["simulator"] = simulator
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": execution,
        **extra,
    }


async def _recorded_run(
    state: SessionState, work_dir: Path, job_id: str, simulator: Any, version: str
) -> dict[str, Any]:
    """Check what the job recorded about the build that ran it; return its run row.

    The row names the build the run reported; the durable record names the
    simulator class and the program it launched.
    """
    (run,) = await job_runs(state, job_id)
    assert run["simulator_version"] == version
    await state.job_registry.drain_pending()
    record = json.loads(store.Store(work_dir).job_record(job_id).read_text())
    assert record["simulator"] == simulator.__name__
    assert record["simulator_executable"]["path"] == simulator.spice_exe[-1]
    return run


# ---------------------------------------------------------------------------
# What the request schema names
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_published_pattern_names_every_family(self):
        defs, _ = get_tools()
        (tool,) = [tool for tool in defs if tool.name == "run_experiments"]
        schema = tool.input_schema
        execution = resolve_local_ref(schema, schema["properties"]["execution"])
        published = json.dumps(execution["properties"]["simulator"])
        (pattern,) = re.findall(r'"pattern": "((?:[^"\\]|\\.)*)"', published)
        pattern = json.loads(f'"{pattern}"')
        for family in SIMULATORS:
            assert re.fullmatch(pattern, family), family
            assert re.fullmatch(pattern, f"{family}:build-2"), family

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

        # QSPICE names its build in the raw's Command field.
        run = await _recorded_run(state, work_dir, receipt["job_id"], qspice, QSPICE_COMMAND)
        assert run["raw"].endswith(".qraw")

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

        await _recorded_run(state, work_dir, receipt["job_id"], xyce, XYCE_BUILD)


# ---------------------------------------------------------------------------
# Refusals: not detected, not runnable on this host, not runnable for this file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRefusal:
    async def test_an_undetected_family_is_refused_naming_what_was_detected(
        self, config: ServerConfig, work_dir: Path
    ):
        state = _state(config)
        is_error, data = await submit_experiment(
            state, _payload(_deck(work_dir), "no-xyce", "xyce")
        )

        assert is_error
        assert data["error"]["commit_state"] == "not_started"
        message = data["error"]["message"]
        assert "'xyce' is not available on this server" in message
        assert "detected: ['ltspice']" in message

    async def test_qspice_off_windows_is_refused_with_the_reason(
        self, config: ServerConfig, work_dir: Path, qspice: type, monkeypatch
    ):
        _host(monkeypatch, "wsl")
        state = _state(config, qspice=qspice)
        is_error, data = await submit_experiment(
            state, _payload(_deck(work_dir), "qspice-wsl", "qspice")
        )

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
        _host(monkeypatch, "linux")
        config.simulator = "qspice"
        state = SessionState.create(config, available={"qspice": qspice})
        assert state.default_simulator is qspice
        is_error, data = await submit_experiment(
            state, _payload(_deck(work_dir), "qspice-default", None)
        )

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
        findings = lint_deck(self._KEYED_CAPACITOR, tmp_path / "keyed.cir", dialect, simulator)
        return [f["subject"] for f in findings if f["rule_id"] == "value-keyword-ltspice"]

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
        _host(monkeypatch, "linux")
        state = _state(config, qspice=qspice, xyce=xyce)

        simulators = (await capabilities_report(state))["simulators"]

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
        simulators = (await capabilities_report(_state(config, qspice=qspice)))["simulators"]

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
        # The entry shows the pattern the field is checked against, which
        # admits every family.
        matched = re.search(r"matching (\S+)\)$", field["type"])
        assert matched is not None, field
        assert all(re.fullmatch(matched[1], name) for name in SIMULATORS), field


@pytest.mark.asyncio
class TestNamedExecutables:
    """A further build of QSPICE or Xyce binds at startup like one of LTspice or
    ngspice, and a run naming it meets the same host check as its family."""

    def _bind(self, config: ServerConfig, work_dir: Path) -> dict[str, type]:
        config.simulator_executables = {
            "qspice:alt": stand_in_program(work_dir / "builds" / "QSPICE64.exe", b"QSPICE two"),
            "xyce:alt": stand_in_program(work_dir / "builds" / "Xyce", b"Xyce two"),
        }
        diagnostics: list[str] = []
        named = detect_named_simulators(config, diagnostics)
        assert diagnostics == []
        return named

    async def test_a_named_build_of_each_family_binds(self, config: ServerConfig, work_dir: Path):
        named = self._bind(config, work_dir)

        assert {selector: simulator_family(cls) for selector, cls in named.items()} == {
            "qspice:alt": "qspice",
            "xyce:alt": "xyce",
        }

    async def test_a_named_qspice_build_off_windows_is_refused_by_its_selector(
        self, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _host(monkeypatch, "wsl")
        state = SessionState.create(
            config, {"ltspice": FakeSim}, named=self._bind(config, work_dir)
        )

        is_error, data = await submit_experiment(
            state, _payload(_deck(work_dir), "named-qspice", "qspice:alt")
        )

        assert is_error
        message = data["error"]["message"]
        assert "'qspice:alt' cannot run here" in message
        assert "Selectable here: ['ltspice', 'xyce:alt']" in message
        named = (await capabilities_report(state))["named_executables"]
        assert named["xyce:alt"]["selectable"] is True
        assert named["qspice:alt"]["selectable"] is False
        assert "only when this server itself runs on Windows" in named["qspice:alt"]["refusal"]


class TestRemediation:
    def test_off_windows_qspice_is_not_sent_to_an_install(self, config: ServerConfig, monkeypatch):
        _host(monkeypatch, "wsl")

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
