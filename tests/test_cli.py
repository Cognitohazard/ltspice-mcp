"""The command-line front end: one engine, two bindings.

What these pin:

- ``--json`` prints the same ``structuredContent`` the MCP tool returns, so a
  shell caller and an MCP client read identical values off one handler.
- exit codes classify the envelope, not a message, across the whole matrix.
- a run started from the shell blocks until it is terminal, and every early
  exit (deadline, Ctrl-C) cancels the job this process owns first — an exiting
  one-shot cannot leave a job behind because nothing would own it.
- ``--no-wait`` is refused before anything is staged.
- an invocation is a parallel session: it takes the same cross-process file
  lock and reads a peer's live job as running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

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
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.tools import get_tools_for_profile
from ltspice_mcp.tools._base import circuit_lock_target
from tests.conftest import FakeSim, fake_simulator

_GOOD_DECK = "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n"


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
    import ltspice_mcp.server as server_mod

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
    # A simulator identity, not a binary: nothing here launches one.
    monkeypatch.setattr(
        server_mod, "detect_simulators", lambda config, diagnostics: {"ltspice": FakeSim}
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

        def _without_replay_note(payload: dict[str, Any]) -> dict[str, Any]:
            trimmed = dict(payload)
            trimmed["observations"] = [
                item
                for item in payload.get("observations", [])
                if item.get("code") != "idempotent_replay"
            ]
            return trimmed

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

    async def test_human_output_is_the_handler_text(
        self, cli_home: Path, capsys: pytest.CaptureFixture[str]
    ):
        deck = _deck(cli_home)
        await cli.run(["verify-circuit", "--path", str(deck)])
        captured = capsys.readouterr()
        assert captured.out.strip()
        assert not captured.out.startswith("{")


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
        _, dispatch = get_tools_for_profile("consolidated")
        schemas = {r.definition.name: r.definition.outputSchema for r in dispatch.values()}
        assert len(schemas) == 6
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
        assert code == cli.EXIT_REFUSED
        assert "nope" in _stdout_json(capsys)["error"]["message"]

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
            cli.EXIT_INTERRUPTED,
        }
        assert len(codes) == 7
        help_text = cli.build_parser().format_help()
        for code in sorted(codes):
            assert f"  {code} " in help_text or f"  {code}   " in help_text


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
