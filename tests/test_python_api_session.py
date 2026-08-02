"""Lifecycle, lease, and event-loop bridge tests for the Python API."""

from __future__ import annotations

import asyncio
import importlib
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import ltspice_mcp.api._session as session_module
import ltspice_mcp.engine as engine_module
import ltspice_mcp.server as server_module
from ltspice_mcp.api import (
    Api,
    ApiClosedError,
    ApiError,
    ApiInterrupted,
    ApiSessionError,
    ApiValidationError,
)
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.engine import BootstrapResult
from ltspice_mcp.state import SessionState


class _StubState:
    def __init__(self) -> None:
        self.shutdown_started = threading.Event()

    async def shutdown(self) -> None:
        self.shutdown_started.set()


def _boot_result(state: object) -> BootstrapResult:
    return BootstrapResult(
        state=cast(SessionState, state),
        preloaded_circuits=0,
    )


def _patch_stub_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    state: _StubState | None = None,
) -> _StubState:
    selected = state or _StubState()

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        del kwargs
        return _boot_result(selected)

    monkeypatch.setattr(session_module, "bootstrap_engine", bootstrap)
    return selected


def _wait_for_status(api: Api, expected: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with api._lifecycle_lock:
            if api._status == expected:
                return
        time.sleep(0.005)
    pytest.fail(f"Api did not reach status {expected!r}")


def test_exception_hierarchy_and_tier_one_public_surface() -> None:
    assert issubclass(ApiSessionError, ApiError)
    assert issubclass(ApiClosedError, ApiSessionError)
    assert issubclass(ApiInterrupted, KeyboardInterrupt)
    assert issubclass(ApiValidationError, ValueError)
    assert set(session_module.Api.__module__.split(".")) >= {"ltspice_mcp", "api"}

    import ltspice_mcp.api as api_module

    assert set(api_module.__all__) == {
        "Api",
        "ApiCallError",
        "ApiClosedError",
        "ApiError",
        "ApiInternalError",
        "ApiInterrupted",
        "ApiSessionError",
        "ApiValidationError",
    }
    assert not hasattr(api_module, "RawResult")
    assert all(
        hasattr(Api, name)
        for name in (
            "run_experiments",
            "jobs",
            "wait",
            "analyze_results",
            "inspect",
            "edit_schematic",
            "verify_circuit",
        )
    )


def test_api_routes_constructor_inputs_to_library_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    state = _StubState()

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        captured.update(kwargs)
        return _boot_result(state)

    monkeypatch.setattr(session_module, "bootstrap_engine", bootstrap)
    config_path = tmp_path / "custom.toml"

    api = Api(
        working_dir=tmp_path,
        config_path=config_path,
        default_timeout=17,
    )
    api.close()

    assert captured == {
        "mode": "library",
        "working_dir": tmp_path,
        "config_path": config_path,
        "default_timeout": 17,
    }


def test_api_explicit_config_path_uses_real_shared_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    config_path = tmp_path / "selected.toml"
    config_path.write_text("[simulation]\ntimeout = 17\n", encoding="utf-8")
    monkeypatch.setattr(engine_module, "detect_simulators", lambda config, diagnostics: {})
    monkeypatch.setattr("ltspice_mcp.lib.wsl.is_wsl", lambda: False)

    with pytest.raises(TypeError, match="server startup hooks"):
        Api(working_dir=working_dir, config_path=config_path, _logger=object())

    api = Api(
        working_dir=working_dir,
        config_path=config_path,
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        assert api._state.config.config_path == config_path
        assert api._state.config.default_timeout == 17
        assert api._state.working_dir == working_dir
        assert api._state.config.allowed_paths == [working_dir]
    finally:
        api.close()


def test_concurrent_constructors_allow_exactly_one_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_bootstrap = threading.Event()
    release_bootstrap = threading.Event()
    bootstrap_calls = 0
    results: list[Api | BaseException] = []

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        nonlocal bootstrap_calls
        del kwargs
        bootstrap_calls += 1
        entered_bootstrap.set()
        await asyncio.to_thread(release_bootstrap.wait)
        return _boot_result(_StubState())

    monkeypatch.setattr(session_module, "bootstrap_engine", bootstrap)

    def construct() -> None:
        try:
            results.append(Api())
        except BaseException as exc:
            results.append(exc)

    first = threading.Thread(target=construct)
    first.start()
    assert entered_bootstrap.wait(2)
    second = threading.Thread(target=construct)
    second.start()
    second.join(2)
    assert not second.is_alive()
    release_bootstrap.set()
    first.join(2)
    assert not first.is_alive()

    winners = [result for result in results if isinstance(result, Api)]
    failures = [result for result in results if isinstance(result, BaseException)]
    try:
        assert len(winners) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ApiSessionError)
        assert bootstrap_calls == 1
    finally:
        for api in winners:
            api.close()


def test_bootstrap_failure_releases_session_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 1:
            raise RuntimeError("bootstrap failed")
        return _boot_result(_StubState())

    monkeypatch.setattr(session_module, "bootstrap_engine", bootstrap)

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        Api()

    api = Api()
    api.close()
    assert calls == 2


def test_call_preserves_durable_result_when_interrupt_precedes_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stub_bootstrap(monkeypatch)
    api = Api()
    original_result = session_module.Future.result
    calls = 0

    def interrupt_once(future, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original_result(future, timeout=timeout)

    monkeypatch.setattr(session_module.Future, "result", interrupt_once)

    async def durable_receipt() -> dict[str, str]:
        await asyncio.sleep(0.01)
        return {"job_id": "exp-preserved", "control_token": "token"}

    try:
        with pytest.raises(ApiInterrupted) as interrupted:
            api._call(durable_receipt(), preserve_interrupt=True)
        assert interrupted.value.receipt == {
            "job_id": "exp-preserved",
            "control_token": "token",
        }
        assert interrupted.value.job_id == "exp-preserved"
    finally:
        api.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_child_replaces_stale_lease_while_parent_lock_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stub_bootstrap(monkeypatch)
    parent_api = Api()
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_in_parent() -> None:
        with session_module._lease_lock:
            lock_held.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_in_parent)
    holder.start()
    assert lock_held.wait(2)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        message = b"ok"
        child_api: Api | None = None
        try:

            async def inherited_call() -> None:
                return None

            with pytest.raises(ApiSessionError):
                parent_api._call(inherited_call())
            with pytest.raises(ApiSessionError):
                parent_api.close()
            child_api = Api()
            child_api._call(inherited_call())
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}".encode()
        finally:
            if child_api is not None:
                child_api.close()
            os.write(write_fd, message)
            os.close(write_fd)
            os._exit(0 if message == b"ok" else 1)

    os.close(write_fd)
    release_lock.set()
    holder.join(2)
    assert not holder.is_alive()
    deadline = time.monotonic() + 5
    status = 0
    while time.monotonic() < deadline:
        waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if waited_pid == child_pid:
            break
        time.sleep(0.01)
    else:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        os.close(read_fd)
        parent_api.close()
        pytest.fail("fork child blocked on the inherited lease lock")

    try:
        message = os.read(read_fd, 4096)
    finally:
        os.close(read_fd)
        parent_api.close()
    assert os.waitstatus_to_exitcode(status) == 0, message.decode()
    assert message == b"ok"


@pytest.mark.asyncio
async def test_server_lifespan_and_api_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def new_state() -> SessionState:
        return SessionState.create(
            ServerConfig(
                working_dir=tmp_path,
                allowed_paths=[tmp_path],
                persist_jobs=False,
            ),
            {},
        )

    async def bootstrap(**kwargs: object) -> BootstrapResult:
        del kwargs
        return _boot_result(new_state())

    monkeypatch.setattr(session_module, "bootstrap_engine", bootstrap)
    monkeypatch.setattr(server_module, "bootstrap_engine", bootstrap)

    async with server_module.server_lifespan(server_module.server):
        with pytest.raises(ApiSessionError, match="already active"):
            Api()

    api = Api()
    try:
        with pytest.raises(ApiSessionError, match="already active"):
            async with server_module.server_lifespan(server_module.server):
                pytest.fail("server lifespan entered despite the Api lease")
    finally:
        api.close()


@pytest.mark.asyncio
async def test_server_bootstrap_failure_releases_shared_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_bootstrap(**kwargs: object) -> BootstrapResult:
        del kwargs
        raise RuntimeError("server bootstrap failed")

    monkeypatch.setattr(server_module, "bootstrap_engine", failed_bootstrap)
    with pytest.raises(RuntimeError, match="server bootstrap failed"):
        async with server_module.server_lifespan(server_module.server):
            pytest.fail("server lifespan entered after bootstrap failure")

    _patch_stub_bootstrap(monkeypatch)
    api = Api()
    api.close()


def test_double_close_calls_after_close_and_loop_thread_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_stub_bootstrap(monkeypatch)
    api = Api()

    async def close_from_loop() -> BaseException | None:
        try:
            api.close()
        except BaseException as exc:
            return exc
        return None

    assert isinstance(api._call(close_from_loop()), ApiSessionError)
    api.close()
    api.close()
    assert state.shutdown_started.is_set()

    async def after_close() -> None:
        return None

    with pytest.raises(ApiClosedError):
        api._call(after_close())
    with pytest.raises(ApiClosedError):
        api.__enter__()


def test_context_manager_closes_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _patch_stub_bootstrap(monkeypatch)

    with Api() as api:
        assert api._call(asyncio.sleep(0, result="ready")) == "ready"

    assert state.shutdown_started.is_set()
    with pytest.raises(ApiClosedError):
        api._call(asyncio.sleep(0))


def test_concurrent_calls_and_close_cancel_only_cancelable_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_stub_bootstrap(monkeypatch)
    api = Api()
    effectful_started = threading.Event()
    cancelable_started = threading.Event()
    release_effectful = threading.Event()
    call_results: dict[str, object] = {}
    close_errors: list[BaseException] = []

    async def effectful_call() -> str:
        effectful_started.set()
        await asyncio.to_thread(release_effectful.wait)
        return "effect-complete"

    async def cancelable_call() -> None:
        cancelable_started.set()
        await asyncio.Event().wait()

    def invoke_effectful() -> None:
        try:
            call_results["effectful"] = api._call(effectful_call())
        except BaseException as exc:
            call_results["effectful"] = exc

    def invoke_cancelable() -> None:
        try:
            call_results["cancelable"] = api._call(
                cancelable_call(),
                cancelable=True,
            )
        except BaseException as exc:
            call_results["cancelable"] = exc

    def close() -> None:
        try:
            api.close()
        except BaseException as exc:
            close_errors.append(exc)

    callers = [
        threading.Thread(target=invoke_effectful),
        threading.Thread(target=invoke_cancelable),
    ]
    for caller in callers:
        caller.start()
    assert effectful_started.wait(2)
    assert cancelable_started.wait(2)

    closers = [threading.Thread(target=close), threading.Thread(target=close)]
    for closer in closers:
        closer.start()
    _wait_for_status(api, "closing")
    assert not state.shutdown_started.is_set()

    async def rejected_after_close_started() -> None:
        return None

    with pytest.raises(ApiClosedError):
        api._call(rejected_after_close_started())

    release_effectful.set()
    for thread in [*callers, *closers]:
        thread.join(3)
        assert not thread.is_alive()

    assert call_results["effectful"] == "effect-complete"
    assert isinstance(call_results["cancelable"], ApiClosedError)
    assert close_errors == []
    assert state.shutdown_started.is_set()


class _RunnerOwnedState(_StubState):
    def __init__(self) -> None:
        super().__init__()
        self.runner_tasks: list[asyncio.Task[None]] = []
        self.live_task_seen_by_shutdown = False

    async def start_runner_owned_task(self) -> None:
        async def run_until_shutdown() -> None:
            await asyncio.Event().wait()

        self.runner_tasks.append(asyncio.create_task(run_until_shutdown()))
        await asyncio.sleep(0)

    async def shutdown(self) -> None:
        self.shutdown_started.set()
        self.live_task_seen_by_shutdown = any(not task.done() for task in self.runner_tasks)
        for task in self.runner_tasks:
            task.cancel()
        await asyncio.gather(*self.runner_tasks, return_exceptions=True)


def test_runner_owned_background_task_does_not_block_state_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _RunnerOwnedState()
    _patch_stub_bootstrap(monkeypatch, state)
    api = Api()

    api._call(state.start_runner_owned_task())
    api.close()

    assert state.shutdown_started.is_set()
    assert state.live_task_seen_by_shutdown


class _StubRunner:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int,
    ) -> None:
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel

    def has_active_work(self) -> bool:
        return False


def test_runner_cache_survives_repeated_bridge_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState.create(
        ServerConfig(
            working_dir=tmp_path,
            allowed_paths=[tmp_path],
            persist_jobs=False,
        ),
        {},
    )
    _patch_stub_bootstrap(monkeypatch, cast(Any, state))
    original_import = importlib.import_module
    runner_module = SimpleNamespace(SimulationRunner=_StubRunner)

    def import_module(name: str, package: str | None = None) -> Any:
        if name == "ltspice_mcp.lib.sim_runner":
            return runner_module
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    simulator_class = type("FakeSimulator", (), {})
    api = Api()

    async def get_runner() -> object:
        return state.runners.get_sim_runner(
            asyncio.get_running_loop(),
            simulator_class,
            tmp_path,
        )

    try:
        runners = [api._call(get_runner()) for _ in range(5)]
        assert all(runner is runners[0] for runner in runners)
        assert cast(_StubRunner, runners[0]).loop is api._loop
    finally:
        api.close()
