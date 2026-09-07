"""The ``run_code`` tool: a Python snippet in a warm worker that holds the engine.

Every test here drives a REAL worker process with a real ``Api`` on a
temporary working directory (no simulator), through the tool handler or the
supervisor it uses. One worker serves the module: booting the engine is the
whole cost, and what these tests pin — output caps, interrupts, busy,
reset, death, respawn — is exactly the behaviour of one long-lived worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import run_code as run_code_module
from ltspice_mcp.tools.run_code import CodeWorker, RunCodeInput, handle_run_code, worker_for

# The worker's pipes belong to one event loop: every async test here shares
# the module's loop, and the sync tests carry no mark.
ASYNC = pytest.mark.asyncio(loop_scope="module")

POSIX = os.name != "nt"


@pytest.fixture(scope="module")
def code_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("run-code")


@pytest.fixture(scope="module")
def no_detection() -> Iterator[None]:
    """The worker inherits the environment: keep its engine host-independent."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION", "1")
        yield


def _serving_state(work_dir: Path) -> SessionState:
    config = ServerConfig(
        working_dir=work_dir, allowed_paths=[work_dir], log_level="DEBUG", run_code=True
    )
    return SessionState.create(config, available={})


@pytest.fixture(scope="module")
async def state(code_dir: Path, no_detection: None) -> AsyncIterator[SessionState]:
    state = _serving_state(code_dir)
    yield state
    await state.shutdown()


async def run(state: SessionState, code: str, **fields) -> dict:
    result = await handle_run_code(RunCodeInput(code=code, **fields), state)
    assert result.structured_content is not None
    return result.structured_content


# ---------------------------------------------------------------------------
# The surface: registered always, served only when the operator turned it on
# ---------------------------------------------------------------------------


class TestSurface:
    def test_off_by_default_is_neither_advertised_nor_dispatchable(self, state_no_sim):
        assert state_no_sim.config.run_code is False
        assert "run_code" not in {d.name for d in state_no_sim.tool_defs}
        assert "run_code" not in state_no_sim.tool_dispatch
        # Its arguments do not claim ownership of any wire field either.
        assert "code" not in state_no_sim.field_owners

    def test_on_puts_it_last_on_the_surface(self, state: SessionState):
        names = [d.name for d in state.tool_defs]
        assert names[-1] == "run_code"
        assert "run_code" in state.tool_dispatch
        assert state.field_owners["code"] == ("run_code",)

    def test_the_gate_is_declared_on_the_registration(self):
        """One declaration: the served surface, the reference table, the
        capabilities entry and the instructions all read it from here."""
        from ltspice_mcp.config import ServerConfig, config_key
        from ltspice_mcp.tools._base import registry

        assert registry.gate_of("run_code") == "run_code"
        assert config_key("run_code") == "tools.run_code"
        # Every gate names a real boolean field, fail-closed at listing time.
        for registered in registry._registered:  # pyright: ignore[reportPrivateUsage]
            if registered.gate is not None:
                assert isinstance(getattr(ServerConfig(), registered.gate), bool)

    def test_empty_code_without_reset_is_refused(self):
        with pytest.raises(ValueError, match="code is empty"):
            RunCodeInput(code="   ")

    def test_the_description_states_the_authority_first(self, state: SessionState):
        definition = {d.name: d for d in state.tool_defs}["run_code"]
        assert definition.description
        first_sentence, second_sentence = definition.description.split(". ")[:2]
        assert "warm worker" in first_sentence
        assert "authority" in second_sentence
        assert definition.annotations is not None
        assert definition.annotations.destructive_hint is True
        assert definition.annotations.read_only_hint is False


# ---------------------------------------------------------------------------
# Running code
# ---------------------------------------------------------------------------


@ASYNC
class TestExecution:
    async def test_print_and_trailing_expression(self, state: SessionState):
        reply = await run(state, "print('hi')\n40 + 2")
        assert reply["status"] == "ok"
        assert reply["stdout"] == "hi\n"
        assert reply["result"] == "42"
        assert reply["error"] is None
        assert reply["worker_pid"] and reply["worker_pid"] != os.getpid()
        assert reply["exec_seq"] >= 1

    async def test_the_namespace_holds_the_engine(self, state: SessionState):
        code = (
            "ops = [n for n in ('run_experiments', 'analyze_results', 'jobs', 'inspect',"
            " 'edit_schematic', 'verify_circuit') if callable(getattr(api, n, None))]\n"
            "print(np.__name__, callable(load_raw), callable(measurements))\n"
            "(len(ops), reference('jobs')[:6])"
        )
        reply = await run(state, code)
        assert reply["status"] == "ok", reply
        assert reply["stdout"] == "numpy True True\n"
        assert reply["result"].startswith("(6, ")

    async def test_each_call_is_a_fresh_namespace(self, state: SessionState):
        await run(state, "leftover = 1")
        reply = await run(state, "leftover")
        assert reply["status"] == "error"
        assert reply["error"]["type"] == "NameError"

    async def test_stdout_keeps_head_and_tail_and_counts_the_rest(self, state: SessionState):
        reply = await run(state, "print('x' * 30000)")
        assert reply["status"] == "ok"
        assert reply["truncated"] is True
        assert reply["chars_dropped"] == 30001 - 16000
        assert "characters elided" in reply["stdout"]
        assert reply["stdout"].startswith("x" * 100)
        assert reply["stdout"].endswith("x" * 100 + "\n")
        assert "capped" in reply["hint"]

    async def test_an_error_is_reported_and_the_worker_survives(self, state: SessionState):
        before = (await run(state, "1"))["worker_pid"]
        reply = await run(state, "print('before')\n1 / 0")
        assert reply["status"] == "error"
        assert reply["error"]["type"] == "ZeroDivisionError"
        assert "ZeroDivisionError" in reply["error"]["traceback_tail"]
        assert reply["stdout"] == "before\n"
        assert reply["worker_pid"] == before
        assert (await run(state, "2 + 2"))["result"] == "4"

    async def test_system_exit_does_not_take_the_worker_down(self, state: SessionState):
        before = (await run(state, "1"))["worker_pid"]
        reply = await run(state, "raise SystemExit(0)")
        assert reply["status"] == "error"
        assert reply["error"]["type"] == "SystemExit"
        assert (await run(state, "1"))["worker_pid"] == before

    async def test_a_reply_of_non_ascii_output_fits_the_pipe(self, state: SessionState):
        """Escaped to six bytes a character, 16k of µ overran the reader's
        default 64 KiB line; the reply is UTF-8 and the reader's line limit
        is sized to the caps."""
        reply = await run(state, "print('µ' * 30000)")
        assert reply["status"] == "ok", reply
        assert reply["truncated"] is True
        assert reply["stdout"].startswith("µ" * 100)

    async def test_a_raw_fd_write_cannot_forge_a_reply(self, state: SessionState):
        forged = json.dumps({"op": "reply", "seq": 1, "status": "ok", "result": "forged"})
        reply = await run(
            state, f"import os\nos.write(1, {(forged + chr(10)).encode()!r})\nprint('clean')"
        )
        assert reply["status"] == "ok"
        assert reply["stdout"] == "clean\n"
        assert reply["result"] is None


# ---------------------------------------------------------------------------
# Time, concurrency, lifetime
# ---------------------------------------------------------------------------


@ASYNC
class TestLifetime:
    @pytest.mark.skipif(not POSIX, reason="the graceful interrupt is POSIX-only")
    async def test_timeout_interrupts_and_keeps_the_worker(self, state: SessionState):
        before = (await run(state, "1"))["worker_pid"]
        reply = await run(state, "import time\nprint('started')\ntime.sleep(30)", timeout_s=1)
        assert reply["status"] == "timeout"
        assert reply["elapsed_s"] < 5
        assert reply["stdout"] == "started\n"
        assert "timeout_s" in reply["hint"]
        after = await run(state, "1")
        assert after["worker_pid"] == before
        assert after["exec_seq"] == reply["exec_seq"] + 1
        assert after["worker_restarted"] is None

    @pytest.mark.skipif(not POSIX, reason="the graceful interrupt is POSIX-only")
    async def test_a_snippet_that_swallows_the_interrupt_is_killed(
        self, state: SessionState, monkeypatch: pytest.MonkeyPatch
    ):
        # The grace is read at call time; a short one keeps the test quick.
        monkeypatch.setattr(run_code_module, "INTERRUPT_GRACE_S", 0.5)
        before = (await run(state, "1"))["worker_pid"]
        code = (
            "import time\n"
            "try:\n    time.sleep(30)\n"
            "except KeyboardInterrupt:\n    time.sleep(30)\n"
        )
        reply = await run(state, code, timeout_s=1)
        assert reply["status"] == "timeout"
        assert reply["elapsed_s"] < 3
        after = await run(state, "'fresh'")
        assert after["status"] == "ok"
        assert after["worker_pid"] != before
        assert after["worker_restarted"] == {
            "previous_pid": before,
            "reason": "killed after a timeout",
        }
        with pytest.raises(ProcessLookupError):
            os.kill(before, 0)

    @pytest.mark.skipif(not POSIX, reason="process groups are POSIX")
    async def test_killing_the_worker_takes_its_children_with_it(self, state: SessionState):
        reply = await run(state, "import subprocess\nsubprocess.Popen(['sleep', '1000']).pid")
        assert reply["status"] == "ok", reply
        child = int(reply["result"])
        os.kill(child, 0)  # alive while the worker is
        reset = await run(state, "", reset=True)
        assert reset["status"] == "reset"
        await asyncio.sleep(0.5)
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)

    async def test_a_second_call_while_one_runs_is_busy(self, state: SessionState):
        first = asyncio.ensure_future(run(state, "import time\ntime.sleep(2)\n'first'"))
        await asyncio.sleep(0.5)
        second = await run(state, "2")
        assert second["status"] == "busy"
        assert second["running"]["phase"] == "running"
        assert second["running"]["same_code"] is False
        assert 0 <= second["running"]["elapsed_s"] < 3
        assert "One snippet" in second["hint"]
        same = await run(state, "import time\ntime.sleep(2)\n'first'")
        assert same["status"] == "busy"
        assert same["running"]["same_code"] is True
        done = await first
        assert done["status"] == "ok"
        assert done["result"] == "'first'"
        assert second["running"]["exec_seq"] == done["exec_seq"]

    async def test_reset_restarts_the_worker(self, state: SessionState):
        before = (await run(state, "1"))["worker_pid"]
        reply = await run(state, "", reset=True)
        assert reply["status"] == "reset"
        assert reply["worker_restarted"] == {"previous_pid": before, "reason": "reset requested"}
        after = await run(state, "1")
        assert after["worker_pid"] != before
        assert after["worker_restarted"] is None
        assert after["exec_seq"] == reply["exec_seq"] + 1

    async def test_worker_death_is_reported_and_the_next_call_respawns(self, state: SessionState):
        before = (await run(state, "1"))["worker_pid"]
        reply = await run(state, "import os\nos._exit(3)")
        assert reply["status"] == "error"
        assert reply["error"]["type"] == "WorkerDied"
        after = await run(state, "'back'")
        assert after["status"] == "ok"
        assert after["result"] == "'back'"
        assert after["worker_pid"] != before
        assert after["worker_restarted"] == {"previous_pid": before, "reason": "the worker exited"}
        assert "interrupted" in after["hint"]

    @pytest.mark.skipif(not POSIX, reason="the graceful interrupt is POSIX-only")
    async def test_a_cancelled_call_interrupts_and_the_next_call_is_served(
        self, state: SessionState
    ):
        before = (await run(state, "1"))["worker_pid"]
        task = asyncio.ensure_future(run(state, "import time\ntime.sleep(30)"))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The interrupted reply is drained in the background; a call landing
        # inside that window is busy in the interrupting phase, never queued.
        reply = await run(state, "1")
        for _ in range(20):
            if reply["status"] != "busy":
                break
            assert reply["running"]["phase"] == "interrupting"
            await asyncio.sleep(0.25)
            reply = await run(state, "1")
        assert reply["status"] == "ok"
        assert reply["worker_pid"] == before

    async def test_shutdown_closes_the_worker(self, code_dir: Path, no_detection: None):
        own = _serving_state(code_dir)
        reply = await run(own, "1")
        worker = worker_for(own)
        assert worker.process is not None and worker.process.returncode is None
        await own.shutdown()
        assert worker.process is None
        # The process itself is gone, not just forgotten.
        with pytest.raises(ProcessLookupError):
            os.kill(reply["worker_pid"], 0)

    async def test_a_worker_that_cannot_start_is_a_structured_error(self, tmp_path: Path):
        worker = CodeWorker(tmp_path / "missing", None)
        reply = await worker.run("1", 5.0, False)
        assert reply["status"] == "error"
        assert reply["error"]["type"] == "WorkerBootFailed"
        assert reply["worker_pid"] is None


# ---------------------------------------------------------------------------
# The worker on its own: parent death, import weight
# ---------------------------------------------------------------------------


class TestWorkerProcess:
    @pytest.mark.skipif(not POSIX, reason="the EOF watchdog interrupts with a signal on POSIX")
    def test_parent_death_exits_a_worker_mid_snippet(self, tmp_path: Path):
        env = {**os.environ, "LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "ltspice_mcp.code_worker", str(tmp_path), ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=tmp_path,
            text=True,
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None
            ready = json.loads(proc.stdout.readline())
            assert ready["op"] == "ready"
            proc.stdin.write(
                json.dumps({"op": "run", "seq": 1, "code": "import time\ntime.sleep(60)"}) + "\n"
            )
            proc.stdin.flush()
            # The parent "dies": its end of the request pipe closes.
            proc.stdin.close()
            assert proc.wait(timeout=30) == 0
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_the_worker_imports_no_server_modules(self):
        probe = (
            "import sys, ltspice_mcp.code_worker\n"
            "print(sorted(m for m in sys.modules if m == 'mcp' or m.startswith('ltspice_mcp.tools')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"
