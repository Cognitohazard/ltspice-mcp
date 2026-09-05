"""The command-line front end: one engine, two bindings.

What these pin:

- ``--json`` prints the same ``structuredContent`` the MCP tool returns, so a
  shell caller and an MCP client read identical values off one handler. The
  flag is global: it parses before or after the subcommand.
- human mode prints the handler's text summary and then the same structured
  payload pretty-printed — the data is never gated behind ``--json``, but only
  ``--json`` is parse-stable.
- ``run`` is a translation layer onto run-experiments: it builds the canonical
  payload and enters the identical dispatch and wait path, never a second
  engine. Anything beyond one deck points at run-experiments.
- exit codes classify the envelope, not a message, across the whole matrix.
- a run started from the shell blocks until it is terminal, and every early
  exit (deadline, Ctrl-C) cancels the job this process owns first — an exiting
  one-shot cannot leave a job behind because nothing would own it.
- ``--no-wait`` is refused before anything is staged.
- an invocation is a parallel session: it takes the same cross-process file
  lock and reads a peer's live job as running.
- refusals and parser-level usage errors carry the subcommand's minimal valid
  invocation, so a rejected first guess includes the way back.
- without ``--verbose`` and with no explicit LTSPICE_MCP_LOG_LEVEL, the
  ltspice_mcp logger tree is quieted to ERROR for the invocation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from mcp import types

from ltspice_mcp import cli
from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.filelock import circuit_lock_target, file_lock
from ltspice_mcp.lib.runner_base import RunnerBase
from ltspice_mcp.tools import get_tools
from tests.conftest import (
    LTSPICE_TRAN_RC_VFINAL,
    FakeSim,
    fake_simulator,
    recorded_fixture_simulator,
)

_GOOD_DECK = "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n"

# The circuit the recorded ltspice_tran_rc fixture pair was captured from: an
# RC low-pass whose log carries ``vfinal: V(out)=0.999876166042 at 0.0009``.
_RC_MEAS_DECK = (
    "* rc lowpass step\n"
    "V1 in 0 PULSE(0 1 0 1u 1u 1 2)\n"
    "R1 in out 1k\n"
    "C1 out 0 100n\n"
    ".tran 0 1m 0 5u\n"
    ".meas tran vfinal FIND V(out) AT=0.9m\n"
    ".end\n"
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sandboxed working directory the CLI will build its own session over.

    Only the environment is arranged — every test drives the real
    ``cli.run(argv)`` entry, so config load, simulator selection, the job
    registry and shutdown all run exactly as they do for a shell invocation.
    """
    import ltspice_mcp.engine as engine_mod

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LTSPICE_MCP_WORKING_DIR", str(work))
    monkeypatch.setenv("LTSPICE_MCP_ALLOWED_PATHS", str(work))
    monkeypatch.delenv("LTSPICE_MCP_CONFIG", raising=False)
    monkeypatch.delenv("LTSPICE_MCP_SYMBOL_PATHS", raising=False)
    # The CLI sets these itself; declaring them here is what gets them unset
    # again at teardown, so no later test inherits this process-wide state.
    monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "consolidated")
    monkeypatch.setenv("LTSPICE_MCP_LOG_LEVEL", "WARNING")
    # A simulator identity, not a binary: nothing here launches one. Detection
    # is patched where the shared bootstrap resolves it (server_lifespan and
    # the CLI both boot through ltspice_mcp.engine).
    monkeypatch.setattr(
        engine_mod, "detect_simulators", lambda config, diagnostics: {"ltspice": FakeSim}
    )
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def live_peer_pid():
    """A genuinely running process to stand in for a peer session's owner, so
    the liveness check reads a real pid rather than a stubbed answer."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield proc.pid
    proc.kill()
    proc.wait()


@pytest.fixture
def asc_cli_home(cli_home: Path, asc_symbols: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``cli_home`` with the fixture symbol library, so .asc paths resolve
    without reaching for a real LTspice installation."""
    monkeypatch.setenv("LTSPICE_MCP_SYMBOL_PATHS", str(asc_symbols))
    return cli_home


def _deck(work: Path, name: str = "dut.cir", body: str = _GOOD_DECK) -> Path:
    path = work / name
    path.write_text(body)
    return path


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> Any:
    captured = capsys.readouterr()
    return json.loads(captured.out)


async def _handler_payload(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one tool handler over its own fresh session — the MCP side of the
    parity comparison, reached without going through the CLI."""
    async with cli.session() as state:
        result = await state.tool_dispatch[tool].handler(dict(arguments), state)
    assert result.structuredContent is not None
    return result.structuredContent


def _without_replay_note(payload: dict[str, Any]) -> dict[str, Any]:
    """Receipt minus the observation the idempotency replay itself records —
    the one legitimate difference between a first render and a re-read of the
    same coordinator record."""
    trimmed = dict(payload)
    trimmed["observations"] = [
        item for item in payload.get("observations", []) if item.get("code") != "idempotent_replay"
    ]
    return trimmed


# ---------------------------------------------------------------------------
# Startup cost
# ---------------------------------------------------------------------------


class TestStartupCost:
    def test_help_does_not_build_a_session(self):
        """``--help`` is documentation, not a server start: it must not pay for
        config load, simulator detection or the spicelib import chain."""
        # importlib.metadata is in the list because it walks the installed
        # distributions to answer --version, and nothing but --version asks.
        probe = (
            "import sys\n"
            "from ltspice_mcp.cli import main\n"
            "try:\n"
            "    main(['--help'])\n"
            "except SystemExit:\n"
            "    pass\n"
            "loaded = [m for m in ('ltspice_mcp.state', 'ltspice_mcp.server', 'spicelib',\n"
            "                      'importlib.metadata')\n"
            "          if m in sys.modules]\n"
            "sys.stderr.write('LOADED=' + ','.join(loaded))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
        assert "LOADED=" in proc.stderr
        assert proc.stderr.rsplit("LOADED=", 1)[1] == ""
        assert "exit codes" in proc.stdout

    def test_version_still_reports_the_installed_version(self, capsys):
        """Deferred, not dropped: the flag that needs the lookup still pays it."""
        from ltspice_mcp import __version__

        with pytest.raises(SystemExit) as exc:
            cli.parse_args(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out == f"spice-mcp {__version__}\n"

    def test_help_never_advertises_cost_savings(self):
        """The CLI is a second binding over one engine; it sells coordination and
        parsed results, never a cheaper channel."""
        parser = cli.build_parser()
        text = parser.format_help().lower()
        for banned in ("token", "context window", "cheaper", "cheap", "saves you"):
            assert banned not in text, f"help text advertises {banned!r}"

    def test_json_and_run_sit_above_the_help_fold(self):
        """Every fleet agent read ``--help | head -50`` and never saw ``--json``
        or a one-shot entry point below the fold; the quick start guarantees
        both appear in the first screen."""
        head = "\n".join(cli.build_parser().format_help().splitlines()[:50])
        assert "--json" in head
        assert "spice-mcp run " in head


# ---------------------------------------------------------------------------
# --json parity with the MCP handlers
# ---------------------------------------------------------------------------


class TestJsonParity:
    async def test_verify_circuit_json_is_the_handler_payload(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        deck = _deck(cli_home)
        args = {"path": str(deck), "checks": ["syntax"]}

        code = await cli.run(["verify-circuit", json.dumps(args), "--json"])
        printed = capsys.readouterr().out

        expected = await _handler_payload("verify_circuit", args)
        assert printed == json.dumps(expected, ensure_ascii=False) + "\n"
        assert code == cli.EXIT_OK

    async def test_inspect_json_is_the_handler_payload(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        deck = _deck(cli_home)
        args = {"queries": [{"kind": "components", "path": str(deck)}]}

        await cli.run(["inspect", json.dumps(args), "--json"])
        printed = capsys.readouterr().out

        expected = await _handler_payload("inspect", args)
        assert printed == json.dumps(expected, ensure_ascii=False) + "\n"

    async def test_jobs_list_json_is_the_handler_payload(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        await cli.run(["jobs", "--action", "list", "--json"])
        printed = capsys.readouterr().out

        expected = await _handler_payload("jobs", {"action": "list"})
        assert printed == json.dumps(expected, ensure_ascii=False) + "\n"

    async def test_edit_schematic_json_is_the_handler_payload(
        self, asc_cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Two blank builds of the same ops differ only in the identifiers the
        engine mints per call, so the comparison masks those and nothing else."""
        ops = [{"op": "add_component", "reference": "R1", "symbol": "res", "x": 400, "y": 300}]

        await cli.run(
            [
                "edit-schematic",
                json.dumps({"target": str(asc_cli_home / "a.asc"), "base": "blank", "ops": ops}),
                "--json",
            ]
        )
        printed = json.loads(capsys.readouterr().out)
        expected = await _handler_payload(
            "edit_schematic",
            {"target": str(asc_cli_home / "b.asc"), "base": "blank", "ops": ops},
        )

        volatile = ("build_id", "target")
        assert {key: value for key, value in printed.items() if key not in volatile} == {
            key: value for key, value in expected.items() if key not in volatile
        }
        assert printed["target"].endswith("a.asc")

    async def test_run_experiments_json_is_the_run_receipt(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """Compared against the SAME job, re-read through the idempotency replay:
        every id, path and counter must match, because both readings render one
        coordinator record through one receipt builder. Only the observation the
        replay itself records may differ."""
        fake_simulator(monkeypatch)
        deck = _deck(cli_home)
        args = {
            "request_id": "cli-parity",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 5},
        }

        code = await cli.run(["run-experiments", json.dumps(args), "--json"])
        printed = json.loads(capsys.readouterr().out)
        assert code == cli.EXIT_OK
        assert printed["status"] == "completed"

        replayed = await _handler_payload("run_experiments", args)

        assert _without_replay_note(printed) == _without_replay_note(replayed)
        assert printed["job_id"] == replayed["job_id"]

    async def test_shorthand_flags_set_top_level_fields(self, cli_home: Path):
        """A shorthand is only another spelling of one JSON field; it must not
        be a second argument shape."""
        namespace = cli.parse_args(["verify-circuit", '{"checks": ["syntax"]}', "--path", "x.cir"])
        assert cli.build_payload(namespace) == {"checks": ["syntax"], "path": "x.cir"}

    async def test_args_can_come_from_a_file(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        deck = _deck(cli_home)
        args_file = cli_home / "args.json"
        args_file.write_text(json.dumps({"path": str(deck), "checks": ["syntax"]}))

        code = await cli.run(["verify-circuit", f"@{args_file}", "--json"])

        assert code == cli.EXIT_OK
        assert _stdout_json(capsys)["path"] == str(deck)

    async def test_human_mode_prints_text_then_the_structured_payload(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Human mode leads with the handler's text summary and then renders
        the same structuredContent pretty-printed. For the read tools the text
        channel is an ack and the data lives only in structuredContent, so a
        human mode that stopped at the text would show no result at all. Only
        ``--json`` is parse-stable; this rendering is presentation."""
        deck = _deck(cli_home)
        args = {"path": str(deck), "checks": ["syntax"]}
        await cli.run(["verify-circuit", json.dumps(args)])
        out = capsys.readouterr().out

        assert out.strip()
        assert not out.startswith("{"), "the text summary still leads"
        expected = await _handler_payload("verify_circuit", args)
        assert json.dumps(expected, ensure_ascii=False, indent=1) in out

    async def test_inspect_human_output_carries_the_geometry_numbers(
        self, asc_cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """The measured failure: a read tool's human output was an ack with the
        geometry only in structuredContent, and the one agent that engaged got
        no data and walked."""
        args = {"queries": [{"kind": "symbol", "name": "res"}]}
        await cli.run(["inspect", json.dumps(args)])
        out = capsys.readouterr().out

        assert "pins_by_rotation" in out
        expected = await _handler_payload("inspect", args)
        assert json.dumps(expected, ensure_ascii=False, indent=1) in out


# ---------------------------------------------------------------------------
# The `run` on-ramp: a translation layer onto run-experiments
# ---------------------------------------------------------------------------


class TestRunOnRamp:
    def test_run_translates_to_the_canonical_run_experiments_payload(self):
        """Pure translation: the flags become exactly the payload a caller
        would hand run-experiments, so there is no second argument shape to
        keep in step with the engine. wait_s=0 is the one key the on-ramp adds
        on its own: the CLI's bounded wait loop is the supervisor, and the
        handler's receipt dwell would hold off deadline and Ctrl-C."""
        namespace = cli.parse_args(
            ["run", "deck.cir", "--simulator", "ngspice", "--measure", "all"]
        )
        assert cli.build_payload(namespace) == {
            "circuits": [{"path": "deck.cir"}],
            "execution": {"wait_s": 0, "simulator": "ngspice"},
            "analyze": {"recipes": [{"metric": "measurements", "key": "measurements"}]},
        }

    def test_run_bare_deck_adds_only_the_no_dwell_execution(self):
        """No flags: beyond the no-dwell wait_s, keys appear only when asked
        for, so the engine's own defaults stay the single source of default
        behavior."""
        namespace = cli.parse_args(["run", "deck.cir"])
        assert cli.build_payload(namespace) == {
            "circuits": [{"path": "deck.cir"}],
            "execution": {"wait_s": 0},
        }

    def test_run_named_measure_filters_to_that_measurement(self):
        namespace = cli.parse_args(["run", "deck.cir", "--measure", "vfinal"])
        assert cli.build_payload(namespace)["analyze"] == {
            "recipes": [{"metric": "measurements", "key": "vfinal", "names": ["vfinal"]}]
        }

    async def test_run_dispatch_parity_with_run_experiments(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """`run` enters the exact dispatch/wait path run-experiments uses: the
        printed receipt must match a run-experiments reading of the same job,
        re-read through the idempotency replay. The replay side asks at the
        default dwell while the on-ramp submitted with wait_s=0, so this also
        pins that a dwell difference replays instead of conflicting (wait_s is
        excluded from the idempotency fingerprint)."""
        fake_simulator(monkeypatch)
        deck = _deck(cli_home)

        code = await cli.run(["run", str(deck), "--json"])
        printed = json.loads(capsys.readouterr().out)
        assert code == cli.EXIT_OK
        assert printed["status"] == "completed"

        replayed = await _handler_payload(
            "run_experiments",
            {"request_id": printed["request_id"], "circuits": [{"path": str(deck)}]},
        )
        assert _without_replay_note(printed) == _without_replay_note(replayed)

    async def test_run_measure_all_returns_measured_numbers_end_to_end(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """The quick-start invocation must deliver numbers, not a status: a
        receipt without a measured value is exactly the walk-away the on-ramp
        exists to fix. Runs a real deck through the full staging, dispatch,
        wait and attached-analysis path over recorded real LTspice artifacts,
        and asserts the numeric leaf."""
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(cli_home, "rc.cir", _RC_MEAS_DECK)

        code = await cli.run(["run", str(deck), "--measure", "all", "--json"])
        printed = json.loads(capsys.readouterr().out)

        assert code == cli.EXIT_OK
        analysis = printed["analysis"]
        assert analysis["status"] == "completed", analysis.get("error")
        rows = analysis["result"]["results"]["measurements"]["values"]
        assert rows[0]["value"]["stats"]["vfinal"]["mean"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)

    async def test_run_deadline_engages_without_an_authored_dwell(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """--timeout must bite promptly on the on-ramp. The handler's own
        receipt dwell defaults to 60s, and the CLI's deadline logic only
        engages after the submission call returns — so the on-ramp has to
        submit with no dwell and let the CLI's bounded wait loop supervise."""
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        began = time.monotonic()
        code = await cli.run(["run", str(deck), "--timeout", "0.3", "--json"])
        elapsed = time.monotonic() - began
        payload = _stdout_json(capsys)

        assert payload["status"] == "cancelled"
        assert code == cli.EXIT_PARTIAL
        # Generous bound: the cancel's own kill-grace is the floor. Sitting in
        # the handler's default dwell lands at 60s+, far past it.
        assert elapsed < 15.0, f"--timeout waited out the handler dwell ({elapsed:.1f}s)"

    async def test_run_interrupt_cancels_without_an_authored_dwell(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """Ctrl-C during the on-ramp's wait must cancel promptly, not after the
        handler's default receipt dwell runs out."""
        if not hasattr(signal, "SIGINT"):  # pragma: no cover - POSIX-only path
            pytest.skip("no SIGINT on this platform")
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        async def interrupt_soon() -> None:
            await asyncio.sleep(0.4)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = asyncio.create_task(interrupt_soon())
        began = time.monotonic()
        try:
            code = await cli.run(["run", str(deck), "--json"])
        finally:
            await interrupter
        elapsed = time.monotonic() - began

        assert code == cli.EXIT_INTERRUPTED
        assert _stdout_json(capsys)["status"] == "cancelled"
        assert elapsed < 15.0, f"the interrupt waited out the handler dwell ({elapsed:.1f}s)"

    async def test_run_refuses_a_payload_shaped_deck_and_points_at_run_experiments(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """One deck is the whole surface; a JSON payload in the DECK slot is a
        caller who wants run-experiments, and the refusal must say so."""
        code = await cli.run(["run", '{"circuits": [{"path": "a.cir"}]}', "--json"])
        message = _stdout_json(capsys)["error"]["message"]

        assert code == cli.EXIT_REFUSED
        assert "run-experiments" in message

    def test_run_refuses_reserved_payload_forms_even_with_leading_whitespace(self):
        """The payload guard classifies the lstripped candidate: a shell-quoted
        argument with a leading space is the same caller mistake, and letting
        it through would hand a JSON blob to deck staging as a filename."""
        for deck in (' {"circuits": []}', "  @exp.json", " -"):
            namespace = cli.parse_args(["run", deck])
            with pytest.raises(cli._Refused):
                cli.build_payload(namespace)

    async def test_run_malformed_synthesized_measure_is_refused_at_the_door(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """The synthesized analyze block passes through the same at-the-door
        validation an authored one does — a malformed block is refused before
        anything is staged, exactly as on the MCP surface."""
        submissions = fake_simulator(monkeypatch)
        deck = _deck(cli_home)

        code = await cli.run(["run", str(deck), "--measure", "", "--json"])
        payload = _stdout_json(capsys)

        assert code == cli.EXIT_REFUSED
        assert "attached analyze block" in json.dumps(payload)
        assert submissions == [], "nothing may be submitted by a refused call"


# ---------------------------------------------------------------------------
# --json is a global flag
# ---------------------------------------------------------------------------


class TestGlobalJsonFlag:
    def test_flag_parses_in_both_positions(self):
        """The subparser copies default to SUPPRESS, so a root-level flag
        survives the subcommand parse instead of being overwritten."""
        assert cli.parse_args(["--json", "jobs"]).as_json is True
        assert cli.parse_args(["jobs", "--json"]).as_json is True
        assert cli.parse_args(["jobs"]).as_json is False

    async def test_both_positions_emit_the_same_one_line_json(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        deck = _deck(cli_home)
        args = json.dumps({"path": str(deck), "checks": ["syntax"]})

        assert await cli.run(["--json", "verify-circuit", args]) == cli.EXIT_OK
        before = capsys.readouterr().out
        assert await cli.run(["verify-circuit", args, "--json"]) == cli.EXIT_OK
        after = capsys.readouterr().out

        assert before == after
        assert before.count("\n") == 1
        assert json.loads(before)["outcome"] == "complete"


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


class TestOutcomeMapping:
    """The outcome table is closed, and it covers what the tools can say."""

    def test_every_declared_outcome_has_an_exit_code(self):
        """The union of the six tools' declared ``outcome`` enums, taken from
        their own schemas: an outcome a tool can emit and this table does not
        name would exit internal-error on a perfectly good result."""
        declared: set[str] = set()

        def walk(node: Any, key: str | None = None) -> None:
            if isinstance(node, dict):
                if key == "outcome" and isinstance(node.get("enum"), list):
                    declared.update(node["enum"])
                for name, child in node.items():
                    walk(child, name)
            elif isinstance(node, list):
                for child in node:
                    walk(child, key)

        # The dispatch map, not the advertised list: the list drops
        # outputSchema, and the declared enums are what this has to read.
        # Only the envelope six declare an outcome (conftest names the seam;
        # plot_waveform predates the envelope).
        from tests.conftest import CONSOLIDATED_TOOLS

        _, dispatch = get_tools()
        schemas = {
            r.definition.name: r.definition.outputSchema
            for r in dispatch.values()
            if r.definition.name in CONSOLIDATED_TOOLS
        }
        assert set(schemas) == set(CONSOLIDATED_TOOLS)
        for schema in schemas.values():
            walk(schema)
        assert declared
        assert declared <= set(cli._EXIT_BY_OUTCOME), (
            f"outcomes with no exit code: {sorted(declared - set(cli._EXIT_BY_OUTCOME))}"
        )

    def test_an_unknown_outcome_never_reads_as_success(self):
        result = types.CallToolResult(
            content=[], structuredContent={"outcome": "quantum_superposition"}
        )
        assert cli.exit_code_for(result) == cli.EXIT_INTERNAL

    def test_a_missing_outcome_never_reads_as_success(self):
        result = types.CallToolResult(content=[], structuredContent={})
        assert cli.exit_code_for(result) == cli.EXIT_INTERNAL


class TestExitCodes:
    async def test_clean_check_exits_ok(self, cli_home: Path, capsys):
        deck = _deck(cli_home)
        code = await cli.run(["verify-circuit", "--path", str(deck), "--json"])
        assert _stdout_json(capsys)["outcome"] == "complete"
        assert code == cli.EXIT_OK

    async def test_findings_exit_partial(self, cli_home: Path, capsys):
        # A MOSFET short of its four nodes: an arity fault the syntax check
        # reports as a finding rather than a refusal, so the call is terminal
        # but did not deliver a clean circuit.
        deck = _deck(cli_home, "broken.cir", "* broken\nM1 d g 0\n.op\n.end\n")
        code = await cli.run(["verify-circuit", "--path", str(deck), "--json"])
        payload = _stdout_json(capsys)
        assert [f["rule_id"] for f in payload["findings"]] == ["element_arity"]
        assert payload["outcome"] == "partial"
        assert code == cli.EXIT_PARTIAL

    async def test_path_outside_the_sandbox_is_refused(self, cli_home: Path, capsys):
        code = await cli.run(["verify-circuit", "--path", "/etc/hostname", "--json"])
        payload = _stdout_json(capsys)
        assert [f["rule_id"] for f in payload["findings"]] == ["path_denied"]
        assert code == cli.EXIT_REFUSED

    async def test_malformed_json_is_refused_before_a_session_exists(self, cli_home: Path, capsys):
        code = await cli.run(["inspect", "{not json", "--json"])
        assert code == cli.EXIT_REFUSED
        assert _stdout_json(capsys)["error"]["code"] == "usage"

    async def test_unknown_argument_field_is_refused(self, cli_home: Path, capsys):
        code = await cli.run(["inspect", '{"queries": [], "nope": 1}', "--json"])
        message = _stdout_json(capsys)["error"]["message"]
        assert code == cli.EXIT_REFUSED
        assert "nope" in message
        # The strict-shape rejection includes the way back: a minimal valid
        # invocation of the subcommand that was refused.
        assert f"try: {cli._SUBCOMMANDS['inspect'].exemplar}" in message

    async def test_unknown_job_is_refused(self, cli_home: Path, capsys):
        code = await cli.run(["jobs", "--action", "status", "--job-id", "exp_missing", "--json"])
        payload = _stdout_json(capsys)
        assert payload["error"]["commit_state"] == "not_started"
        assert code == cli.EXIT_REFUSED

    async def test_committed_failure_exits_failed(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """A fault after submission is an execution failure, not a refusal: the
        fleet is running and the receipt carries the handles that reach it."""
        fake_simulator(monkeypatch, delay_s=None)

        async def failing_wait(self, job, timeout_s, *, wait_for="all"):
            raise OSError("dwell exploded")

        monkeypatch.setattr(ExperimentRunner, "wait", failing_wait)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps({"circuits": [{"path": str(deck)}], "execution": {"wait_s": 1}}),
                "--json",
            ]
        )
        payload = _stdout_json(capsys)

        assert payload["error"]["commit_state"] == "committed"
        assert payload["job_id"]
        assert code == cli.EXIT_FAILED

    async def test_running_peer_job_exits_unfinished(
        self, cli_home: Path, capsys, live_peer_pid: int
    ):
        """A job owned by a live peer session reads as running here, and a status
        query that finds it still running is not a success."""
        job = _running_peer_experiment(cli_home, live_peer_pid)

        code = await cli.run(["jobs", "--action", "status", "--job-id", job.job_id, "--json"])
        payload = _stdout_json(capsys)

        assert payload["status"] == "running"
        assert payload["outcome"] == "in_progress"
        assert code == cli.EXIT_UNFINISHED

    async def test_every_documented_code_is_reachable_and_distinct(self):
        """The help text is the contract; a code documented there and never
        produced, or two codes sharing a number, is a broken promise."""
        codes = {
            cli.EXIT_OK,
            cli.EXIT_INTERNAL,
            cli.EXIT_REFUSED,
            cli.EXIT_FAILED,
            cli.EXIT_PARTIAL,
            cli.EXIT_UNFINISHED,
            cli.EXIT_UNCONFIRMED,
            cli.EXIT_INTERRUPTED,
        }
        assert len(codes) == 8
        help_text = cli.build_parser().format_help()
        for code in sorted(codes):
            assert f"  {code} " in help_text or f"  {code}   " in help_text


# ---------------------------------------------------------------------------
# Refusal exemplars and parser-level usage errors
# ---------------------------------------------------------------------------


class TestRefusalExemplars:
    async def test_argparse_unknown_flag_carries_the_exemplar(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        with pytest.raises(SystemExit) as exc:
            await cli.run(["run", "deck.cir", "--nope"])
        assert exc.value.code == 2
        assert f"try: {cli._RUN_EXEMPLAR}" in capsys.readouterr().err

    async def test_argparse_extra_positional_carries_the_exemplar(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        with pytest.raises(SystemExit) as exc:
            await cli.run(["run", "a.cir", "b.cir"])
        assert exc.value.code == 2
        assert f"try: {cli._RUN_EXEMPLAR}" in capsys.readouterr().err

    async def test_argparse_error_respects_json_mode(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """A piping caller who asked for --json must not get bare prose on
        stdout even when the failure is argparse's, not a handler's."""
        with pytest.raises(SystemExit) as exc:
            await cli.run(["run", "deck.cir", "--nope", "--json"])
        assert exc.value.code == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"]["code"] == "usage"
        assert payload["error"]["stage"] == "cli"
        assert cli._RUN_EXEMPLAR in payload["error"]["message"]

    def test_every_subcommand_has_an_exemplar(self):
        """The exemplar is a required field of the subcommand table (and `run`
        rides its own definition), so a new subcommand cannot ship refusals
        with no way back; what's left to pin is the real program name."""
        assert set(cli._SUBCOMMANDS) == set(cli.COMMAND_TOOLS) - {"run"}
        for command in cli.COMMAND_TOOLS:
            assert cli._exemplar_for(command).startswith("spice-mcp ")

    async def test_handler_refusal_prints_the_exemplar_on_stderr(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        """Refusals the handler reports as a structured envelope (an unknown
        job here) must carry the way back too, not only the ones the CLI
        classifies itself — on stderr, never inside the parse-stable stdout
        payload."""
        code = await cli.run(["jobs", "--action", "status", "--job-id", "exp_missing", "--json"])
        captured = capsys.readouterr()

        assert code == cli.EXIT_REFUSED
        assert f"try: {cli._SUBCOMMANDS['jobs'].exemplar}" in captured.err
        assert "try:" not in captured.out, "the stdout payload stays the handler's"

    async def test_path_denied_refusal_prints_the_exemplar_on_stderr(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        code = await cli.run(["verify-circuit", "--path", "/etc/hostname", "--json"])
        captured = capsys.readouterr()

        assert code == cli.EXIT_REFUSED
        assert f"try: {cli._SUBCOMMANDS['verify-circuit'].exemplar}" in captured.err
        assert "try:" not in captured.out


# ---------------------------------------------------------------------------
# Quiet logging by default
# ---------------------------------------------------------------------------


class TestQuietLogging:
    """Without --verbose and with no explicit LTSPICE_MCP_LOG_LEVEL, the
    ltspice_mcp logger tree runs at ERROR for the invocation. The trigger is a
    real one: an unreadable experiment record makes the store emit the exact
    WARNING spam the fleet measured ahead of its first answer."""

    @staticmethod
    def _junk_record(work: Path) -> str:
        job_id = "exp_junk_0001"
        path = experiment_store.record_path(job_id, work)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        return job_id

    async def test_quiet_then_verbose_then_quiet_in_one_process(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LTSPICE_MCP_LOG_LEVEL")
        job_id = self._junk_record(cli_home)
        tree = logging.getLogger("ltspice_mcp")
        level_before = tree.level
        argv = ["jobs", "--action", "status", "--job-id", job_id, "--json"]

        await cli.run(argv)
        quiet = capsys.readouterr().err
        assert "[WARNING]" not in quiet
        assert "[INFO]" not in quiet

        await cli.run([*argv, "--verbose"])
        verbose = capsys.readouterr().err
        assert "Skipping unreadable experiment job" in verbose

        await cli.run(argv)
        again = capsys.readouterr().err
        assert "[WARNING]" not in again
        assert tree.level == level_before, "the logger tree must be restored"

    async def test_explicit_log_level_env_wins_over_the_default_quiet(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """An exported LTSPICE_MCP_LOG_LEVEL is the caller's decision; the CLI
        quiets only where the caller left the level to it."""
        monkeypatch.setenv("LTSPICE_MCP_LOG_LEVEL", "WARNING")
        job_id = self._junk_record(cli_home)

        await cli.run(["jobs", "--action", "status", "--job-id", job_id, "--json"])

        assert "Skipping unreadable experiment job" in capsys.readouterr().err

    async def test_run_restores_the_host_processes_root_logging(self, cli_home: Path):
        """run() is a documented reusable entry point, and the server lifespan
        it enters calls logging.basicConfig(force=True), which strips the ROOT
        logger's handlers. A host application embedding run() must get its own
        root handlers and level back."""
        root = logging.getLogger()
        sentinel = logging.NullHandler()
        saved_level = root.level
        root.addHandler(sentinel)
        root.setLevel(logging.CRITICAL)
        try:
            code = await cli.run(["jobs", "--action", "list", "--json"])
            assert code == cli.EXIT_OK
            assert sentinel in root.handlers, "the host's root handler must survive"
            assert root.level == logging.CRITICAL, "the host's root level must survive"
        finally:
            root.removeHandler(sentinel)
            root.setLevel(saved_level)


def _running_peer_experiment(work: Path, owner_pid: int) -> ExperimentJob:
    """Persist a coordinator record that a live peer process owns."""
    circuit = _deck(work, "peer.cir")
    case = ExperimentCase(
        case_id="case_0000",
        run_index=0,
        circuit=circuit.stem,
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="sha-0",
        assignments={},
        status="queued",
    )
    completeness = Completeness(declared=1, expanded=1)
    completeness.recount([case])
    job = ExperimentJob(
        job_id="exp_peer_0001",
        request_id="peer-request",
        fingerprint="f" * 64,
        canonicalizer_version=experiment_store.CANONICALIZER_VERSION,
        control_token="peer-secret",
        store_path=experiment_store.record_path("exp_peer_0001", work),
        cases=[case],
        sources=[
            SourceRecord(
                circuit=circuit.stem,
                path=circuit,
                sha256="source-sha",
                staged_deck=circuit,
                manifest=[
                    ManifestEntry(
                        path=circuit,
                        sha256="source-sha",
                        staged=True,
                        live=False,
                        staged_path=circuit,
                    )
                ],
                linter_version="test",
                simulator="FakeSim",
            )
        ],
        simulator="FakeSim",
        completeness=completeness,
        status="running",
    )
    job.owner_pid = owner_pid
    job.started_at = now()
    experiment_store.save_job(job)
    experiment_store.save_pointers(job)
    return job


# ---------------------------------------------------------------------------
# Block to terminality
# ---------------------------------------------------------------------------


class TestBlockToTerminality:
    async def test_run_blocks_until_the_job_is_terminal(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """The dwell the tool itself offers is shorter than the run; the CLI has
        to keep waiting rather than print an in-flight receipt and exit."""
        fake_simulator(monkeypatch, delay_s=0.6)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps(
                    {
                        "request_id": "cli-blocks",
                        "circuits": [{"path": str(deck)}],
                        "execution": {"wait_s": 0.1},
                    }
                ),
                "--json",
            ]
        )
        payload = _stdout_json(capsys)

        assert payload["outcome"] != "in_progress"
        assert payload["status"] == "completed"
        assert payload["completeness"]["produced"] == 1
        assert code == cli.EXIT_OK

    async def test_deadline_cancels_rather_than_orphans(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """--timeout bounds the wait, and the bound is honoured by cancelling:
        leaving would abandon a job nothing else can reach."""
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps(
                    {
                        "request_id": "cli-deadline",
                        "circuits": [{"path": str(deck)}],
                        "execution": {"wait_s": 0},
                    }
                ),
                "--timeout",
                "0.3",
                "--json",
            ]
        )
        payload = _stdout_json(capsys)

        assert payload["status"] == "cancelled"
        assert payload["outcome"] == "partial"
        assert code == cli.EXIT_PARTIAL

    async def test_interrupt_cancels_the_job_it_owns(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """Ctrl-C during the wait cancels what this process owns before exiting,
        and reports the interrupt in the exit code."""
        if not hasattr(signal, "SIGINT"):  # pragma: no cover - POSIX-only path
            pytest.skip("no SIGINT on this platform")
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        async def interrupt_soon() -> None:
            await asyncio.sleep(0.4)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = asyncio.create_task(interrupt_soon())
        try:
            code = await cli.run(
                [
                    "run-experiments",
                    json.dumps(
                        {
                            "request_id": "cli-interrupt",
                            "circuits": [{"path": str(deck)}],
                            "execution": {"wait_s": 0},
                        }
                    ),
                    "--json",
                ]
            )
        finally:
            await interrupter

        payload = _stdout_json(capsys)
        assert payload["status"] == "cancelled"
        assert code == cli.EXIT_INTERRUPTED

    async def test_no_wait_is_refused_before_anything_is_staged(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        submissions = fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps({"circuits": [{"path": str(deck)}]}),
                "--no-wait",
                "--json",
            ]
        )
        message = _stdout_json(capsys)["error"]["message"]

        assert code == cli.EXIT_REFUSED
        assert "daemon owner" in message
        assert "server_restarted" in message
        assert submissions == [], "nothing may be submitted by a refused call"
        staged = list(cli_home.glob(".ltspice-mcp/experiments/*"))  # noqa: ASYNC240
        assert not staged, "nothing may be staged"

    async def test_no_daemon_owner_ships_today(self):
        """The refusal is a statement of fact about this installation, so the
        fact has to be checked somewhere rather than assumed at the call site."""
        assert cli.daemon_owner_present() is False


# ---------------------------------------------------------------------------
# Exit-path honesty: the envelope, the code and stderr all say what happened
# ---------------------------------------------------------------------------


class TestExitPathHonesty:
    async def test_an_interrupt_does_not_wait_out_the_poll_it_landed_in(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """Ctrl-C is checked against the leg, not between legs. A dwell this
        process is already inside must not hold a stop request for its full
        length — the help text promises the cancel happens on the interrupt."""
        if not hasattr(signal, "SIGINT"):  # pragma: no cover - POSIX-only path
            pytest.skip("no SIGINT on this platform")
        # Far longer than the interrupt: without the race, this is what the run
        # would sit through before noticing.
        monkeypatch.setattr(cli, "_WAIT_LEG_S", 20.0)
        # The scoped kill reaches the Windows process table over WSL interop,
        # which is a fixed cost of the cancel that follows rather than of
        # noticing the interrupt.
        monkeypatch.setattr(RunnerBase, "_kill_by_token", lambda self, token, label="": None)
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        async def interrupt_soon() -> None:
            await asyncio.sleep(0.4)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = asyncio.create_task(interrupt_soon())
        began = time.monotonic()
        try:
            code = await cli.run(
                [
                    "run-experiments",
                    json.dumps(
                        {
                            "request_id": "cli-mid-leg",
                            "circuits": [{"path": str(deck)}],
                            "execution": {"wait_s": 0},
                        }
                    ),
                    "--json",
                ]
            )
        finally:
            await interrupter
        elapsed = time.monotonic() - began

        assert code == cli.EXIT_INTERRUPTED
        assert _stdout_json(capsys)["status"] == "cancelled"
        # The floor here is the cancel's own kill-grace (a killed case is given
        # time to confirm it died), not the leg. Waiting the leg out lands well
        # past this; noticing the interrupt lands well under it.
        assert elapsed < 15.0, f"the interrupt waited out the leg ({elapsed:.1f}s)"

    async def test_a_leg_that_ignores_its_cancellation_does_not_hold_the_interrupt(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The bound has to survive an uncooperative victim. asyncio.wait_for
        cancels and then AWAITS the cancellation, so a dwell that swallows it
        would defeat the interrupt exactly when the interrupt matters most: the
        loser is cancelled and abandoned, never awaited."""
        started = asyncio.Event()

        async def stubborn(tool: str, arguments: dict[str, Any], state: Any) -> Any:
            started.set()
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(3.0)
            return None

        monkeypatch.setattr(cli, "invoke", stubborn)
        interrupt = cli._Interrupt()

        async def fire() -> None:
            await started.wait()
            interrupt.requested.set()

        firing = asyncio.create_task(fire())
        began = time.monotonic()
        leg = await cli._wait_leg("exp_x", cast("Any", None), 3.0, interrupt)
        elapsed = time.monotonic() - began
        await firing

        assert leg is None, "an abandoned leg reports no snapshot"
        assert elapsed < 1.0, f"the interrupt awaited the cancellation ({elapsed:.1f}s)"

    async def test_a_committed_fault_reports_where_the_job_actually_got_to(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """The first receipt's error block is the report of the code that hit
        the fault and must survive. Its lifecycle fields are a different thing:
        printing 'in_progress' about a job this process then watched finish is
        the envelope contradicting the exit path that produced it."""
        original = ExperimentRunner.wait
        calls = {"n": 0}

        async def flaky_wait(
            self, job, timeout_s=None, *, wait_for: Literal["all", "runs"] = "all"
        ):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("dwell exploded")
            return await original(self, job, timeout_s, wait_for=wait_for)

        monkeypatch.setattr(ExperimentRunner, "wait", flaky_wait)
        fake_simulator(monkeypatch, delay_s=0.4)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps(
                    {
                        "request_id": "cli-committed",
                        "circuits": [{"path": str(deck)}],
                        "execution": {"wait_s": 0.1},
                    }
                ),
                "--json",
            ]
        )
        payload = _stdout_json(capsys)

        assert payload["error"]["commit_state"] == "committed"
        assert payload["error"]["message"] == "dwell exploded"
        assert payload["status"] == "completed"
        assert payload["outcome"] != cli.IN_PROGRESS
        assert payload["completeness"]["produced"] == 1
        assert code == cli.EXIT_FAILED

    async def test_an_interrupt_after_the_job_finished_reports_the_result(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """Once the job is terminal an interrupt cannot change the outcome, so
        exiting 130 with 'cancelling the run this process owns' on stderr would
        be a cancel that never happened, reported against a completed run."""
        if not hasattr(signal, "SIGINT"):  # pragma: no cover - POSIX-only path
            pytest.skip("no SIGINT on this platform")
        fake_simulator(monkeypatch, delay_s=0.2)
        deck = _deck(cli_home)
        real_invoke = cli.invoke
        replays = {"n": 0}

        async def interrupt_on_final_replay(tool: str, arguments: dict[str, Any], state: Any):
            if tool == "run_experiments":
                replays["n"] += 1
                if replays["n"] == 2:
                    # The job is terminal by now; this is the render of its
                    # finished receipt.
                    os.kill(os.getpid(), signal.SIGINT)
                    await asyncio.sleep(0.05)
            return await real_invoke(tool, arguments, state)

        monkeypatch.setattr(cli, "invoke", interrupt_on_final_replay)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps(
                    {
                        "request_id": "cli-late-interrupt",
                        "circuits": [{"path": str(deck)}],
                        "execution": {"wait_s": 0},
                    }
                ),
                "--json",
            ]
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert payload["status"] == "completed"
        assert payload["outcome"] == "complete"
        assert code == cli.EXIT_OK
        assert "interrupted" in captured.err
        assert "too late to change the outcome" in captured.err
        assert "cancelling the run" not in captured.err

    async def test_an_unconfirmed_cancel_says_so_instead_of_still_running(
        self, cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """A kill this process issued and could not confirm is not the same
        state as a job running normally, and EXIT_UNFINISHED means the latter —
        its own docstring says only a status query can end there."""

        async def unacknowledged_cancel(self, job, *, control_token=None):
            return []  # accepted, stops nothing: the job stays running

        monkeypatch.setattr(ExperimentRunner, "cancel", unacknowledged_cancel)
        monkeypatch.setattr(cli, "_CANCEL_JOIN_S", 0.3)
        fake_simulator(monkeypatch, delay_s=None)
        deck = _deck(cli_home)

        code = await cli.run(
            [
                "run-experiments",
                json.dumps(
                    {
                        "request_id": "cli-unconfirmed",
                        "circuits": [{"path": str(deck)}],
                        "execution": {"wait_s": 0},
                    }
                ),
                "--timeout",
                "0.3",
                "--json",
            ]
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        # The envelope on its own reads as a job running normally, which is
        # exactly the reading that must not reach the caller: EXIT_UNFINISHED is
        # what that outcome maps to, and its own docstring says only a status
        # query can end there.
        assert payload["outcome"] == cli.IN_PROGRESS
        assert cli._EXIT_BY_OUTCOME[cli.IN_PROGRESS] == cli.EXIT_UNFINISHED
        assert code == cli.EXIT_UNCONFIRMED
        assert payload[cli.CANCEL_CONFIRMED_KEY] is False
        assert "NOT confirmed" in captured.err
        assert "jobs --action status" in captured.err

    async def test_a_config_from_one_run_does_not_leak_into_the_next(self, cli_home: Path, capsys):
        """run() is a reusable entry point, so its environment mutation has to be
        undone: a --config left standing hands the NEXT invocation a sandbox and
        a set of limits its caller never asked for, with nothing saying why."""
        toml = cli_home / "custom.toml"
        toml.write_text("[analysis]\nanalysis_budget_s = 12.5\n")
        query = json.dumps({"queries": [{"kind": "capabilities"}]})

        assert await cli.run(["inspect", query, "--config", str(toml), "--json"]) == cli.EXIT_OK
        configured = _stdout_json(capsys)["results"][0]["data"]["limits"]
        assert configured["analysis_budget_s"] == 12.5

        assert await cli.run(["inspect", query, "--json"]) == cli.EXIT_OK
        inherited = _stdout_json(capsys)["results"][0]["data"]["limits"]
        assert inherited["analysis_budget_s"] != 12.5
        assert "LTSPICE_MCP_CONFIG" not in os.environ


# ---------------------------------------------------------------------------
# Parallel sessions
# ---------------------------------------------------------------------------


def _hold_lock_then_write(target: Path, content: bytes, hold_s: float) -> threading.Thread:
    """Peer session stand-in: hold the file's cross-process lock, write just
    before releasing. flock contention is per open file description, so a
    thread on its own fd exercises the same semantics as another process."""
    held = threading.Event()

    def peer() -> None:
        with file_lock(circuit_lock_target(target)):
            held.set()
            time.sleep(hold_s)
            target.write_bytes(content)

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    if not held.wait(5):
        raise RuntimeError("peer thread failed to take the lock")
    return thread


class TestParallelSession:
    async def test_edit_waits_for_a_peers_lock_and_sees_its_write(
        self, asc_cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """A CLI invocation is a parallel session. It must block on the peer's
        lock and then read the file the peer left, not the one it started from —
        the revision guard can only refuse a lost update if it is comparing
        against the peer's committed bytes."""
        target = asc_cli_home / "shared.asc"
        await cli.run(
            [
                "edit-schematic",
                json.dumps(
                    {
                        "target": str(target),
                        "base": "blank",
                        "ops": [
                            {
                                "op": "add_component",
                                "reference": "R1",
                                "symbol": "res",
                                "x": 400,
                                "y": 300,
                            }
                        ],
                    }
                ),
                "--json",
            ]
        )
        stale_sha = _stdout_json(capsys)["sha256"]

        peer_version = target.read_bytes() + b"TEXT -48 320 Left 2 ;peer marker\n"
        thread = _hold_lock_then_write(target, peer_version, hold_s=0.4)
        try:
            code = await cli.run(
                [
                    "edit-schematic",
                    json.dumps(
                        {
                            "target": str(target),
                            "expected_sha256": stale_sha,
                            "ops": [
                                {"op": "set_component_value", "reference": "R1", "value": "2k2"}
                            ],
                        }
                    ),
                    "--json",
                ]
            )
        finally:
            thread.join(5)
        payload = _stdout_json(capsys)

        assert payload["error"]["code"] == "revision_conflict"
        assert payload["commit_state"] == "not_committed"
        assert payload["sha256"] == sha256_file(target), "must report the peer's committed bytes"
        assert b"peer marker" in target.read_bytes(), "the peer's edit must survive"
        assert code == cli.EXIT_REFUSED

    async def test_edit_on_the_peers_revision_commits(
        self, asc_cli_home: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ):
        """The counterpart: with the peer's own sha the same edit lands, so the
        refusal above is the revision guard doing its job, not the lock failing."""
        target = asc_cli_home / "shared.asc"
        await cli.run(
            [
                "edit-schematic",
                json.dumps(
                    {
                        "target": str(target),
                        "base": "blank",
                        "ops": [
                            {
                                "op": "add_component",
                                "reference": "R1",
                                "symbol": "res",
                                "x": 400,
                                "y": 300,
                            }
                        ],
                    }
                ),
                "--json",
            ]
        )
        capsys.readouterr()

        peer_version = target.read_bytes() + b"TEXT -48 320 Left 2 ;peer marker\n"
        thread = _hold_lock_then_write(target, peer_version, hold_s=0.4)
        thread.join(5)

        code = await cli.run(
            [
                "edit-schematic",
                json.dumps(
                    {
                        "target": str(target),
                        "expected_sha256": sha256_file(target),
                        "ops": [{"op": "set_component_value", "reference": "R1", "value": "2k2"}],
                    }
                ),
                "--json",
            ]
        )
        payload = _stdout_json(capsys)

        assert payload["commit_state"] == "committed"
        assert code == cli.EXIT_OK
        assert b"peer marker" in target.read_bytes()
