"""Process-owned engine session and synchronous event-loop bridge."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
from collections.abc import Coroutine
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, TypeVar

from ltspice_mcp.api import _detach
from ltspice_mcp.api._exceptions import (
    ApiClosedError,
    ApiInterrupted,
    ApiSessionError,
)
from ltspice_mcp.api._methods import ApiMethodsMixin
from ltspice_mcp.engine import bootstrap_library_engine
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_RESIDUAL_DRAIN_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class _LeaseRecord:
    owner: object
    pid: int


_lease_lock = threading.Lock()
_lease_record: _LeaseRecord | None = None


def _reset_lease_lock_after_fork() -> None:
    """Make the child lock usable while preserving its inherited lease record."""
    global _lease_lock
    _lease_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lease_lock_after_fork)


def acquire_session_lease(owner: object) -> int:
    """Atomically reserve this PID's single engine-session slot for ``owner``."""
    global _lease_record
    pid = os.getpid()
    with _lease_lock:
        if _lease_record is not None and _lease_record.pid == pid:
            raise ApiSessionError("An engine session is already active in this process")
        _lease_record = _LeaseRecord(owner=owner, pid=pid)
    return pid


def release_session_lease(owner: object, pid: int) -> None:
    """Release only the exact lease this owner acquired in this PID."""
    global _lease_record
    with _lease_lock:
        if (
            _lease_record is not None
            and _lease_record.owner is owner
            and _lease_record.pid == pid
            and os.getpid() == pid
        ):
            _lease_record = None


async def _cancel_residual_tasks(timeout_s: float) -> None:
    """Cancel and briefly drain loop tasks not owned by the synchronous bridge."""
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    if not tasks:
        return

    for task in tasks:
        task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for task in done:
        if not task.cancelled():
            task.exception()
    if pending:
        logger.warning("Closing the API loop with %d task(s) still pending", len(pending))


class Api(ApiMethodsMixin):
    """Synchronous owner of one engine state on a private persistent event loop."""

    def __init__(
        self,
        working_dir: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        **overrides: object,
    ) -> None:
        self._creator_pid = os.getpid()
        self._lifecycle_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._status: Literal["open", "closing", "closed"] = "open"
        self._bridge_tasks: dict[asyncio.Task[Any], bool] = {}
        self._state: SessionState
        self._boot = _detach.boot_spec(working_dir, config_path, overrides)
        self._detached_children: list[subprocess.Popen[bytes]] = []

        self._lease_pid = acquire_session_lease(self)
        try:
            self._loop = asyncio.new_event_loop()
            self._loop_ready = threading.Event()
            self._loop_stopped = threading.Event()
            self._loop_thread = threading.Thread(
                target=self._run_loop,
                name=f"ltspice-mcp-api-{self._creator_pid}",
                daemon=True,
            )
            self._loop_thread.start()
            self._loop_ready.wait()
            future = self._submit_to_loop(
                bootstrap_library_engine(
                    working_dir=working_dir,
                    config_path=config_path,
                    **overrides,
                )
            )
            self._state = future.result().state
        except BaseException:
            try:
                if hasattr(self, "_loop"):
                    self._abort_startup()
            except BaseException:
                logger.debug("API startup cleanup failed", exc_info=True)
            finally:
                release_session_lease(self, self._lease_pid)
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            asyncio.set_event_loop(None)
            self._loop.close()
            self._loop_stopped.set()

    def _abort_startup(self) -> None:
        loop_thread = getattr(self, "_loop_thread", None)
        if loop_thread is None or not loop_thread.is_alive():
            if not self._loop.is_closed():
                self._loop.close()
            return

        try:
            drain = self._submit_to_loop(_cancel_residual_tasks(_RESIDUAL_DRAIN_TIMEOUT_S))
            drain.result(timeout=_RESIDUAL_DRAIN_TIMEOUT_S * 2)
            self._shutdown_async_generators(timeout=_RESIDUAL_DRAIN_TIMEOUT_S * 2)
        except BaseException:
            logger.debug("API startup cleanup did not fully drain", exc_info=True)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            loop_thread.join()

    def _submit_to_loop(self, coroutine: Coroutine[Any, Any, _T]) -> Future[_T]:
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except BaseException:
            coroutine.close()
            raise

    def _shutdown_async_generators(self, *, timeout: float | None = None) -> None:
        coroutine = self._loop.shutdown_asyncgens()
        future = self._submit_to_loop(coroutine)
        future.result(timeout=timeout)

    def _check_process_and_thread(self) -> None:
        if os.getpid() != self._creator_pid:
            raise ApiSessionError("An Api inherited across fork cannot be used in the child")
        if threading.get_ident() == self._loop_thread.ident:
            raise ApiSessionError("Synchronous Api methods cannot run on the private loop thread")

    async def _invoke_bridge(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        cancelable: bool,
    ) -> _T:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio always supplies one here
            return await coroutine
        self._bridge_tasks[task] = cancelable
        try:
            return await coroutine
        finally:
            self._bridge_tasks.pop(task, None)

    def _call(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        cancelable: bool = False,
        cancel_on_interrupt: bool = False,
        preserve_interrupt: bool = False,
    ) -> _T:
        """Run one invocation on the private loop and block only this caller thread.

        The check here is the authoritative one — every marshalled call passes
        through it. The public methods repeat it only to fail fast, so a call
        from the loop thread or an inherited process raises ApiSessionError
        before argument validation reports anything about the arguments.
        """
        bridge: Coroutine[Any, Any, _T] | None = None
        try:
            self._check_process_and_thread()
            with self._lifecycle_lock:
                if self._status != "open":
                    raise ApiClosedError("The Api session is closing or closed")
                bridge = self._invoke_bridge(coroutine, cancelable=cancelable)
                future = self._submit_to_loop(bridge)
        except BaseException:
            if bridge is not None:
                bridge.close()
            coroutine.close()
            raise

        interrupted: KeyboardInterrupt | None = None
        while True:
            try:
                result = future.result()
                break
            except KeyboardInterrupt as exc:
                if cancel_on_interrupt:
                    future.cancel()
                    raise
                if preserve_interrupt:
                    interrupted = exc
                    continue
                raise
            except FutureCancelledError as exc:
                with self._lifecycle_lock:
                    closing = self._status != "open"
                if closing:
                    raise ApiClosedError("The Api call was cancelled during close") from exc
                raise
        if interrupted is not None:
            receipt = result if isinstance(result, dict) else None
            raise ApiInterrupted(receipt=receipt) from interrupted
        return result

    async def _shutdown_bridge(self) -> None:
        cancelable = [task for task, may_cancel in self._bridge_tasks.items() if may_cancel]
        for task in cancelable:
            task.cancel()

        bridge_tasks = list(self._bridge_tasks)
        if bridge_tasks:
            await asyncio.gather(*bridge_tasks, return_exceptions=True)

        shutdown_error: BaseException | None = None
        try:
            await self._state.shutdown()
        except BaseException as exc:
            shutdown_error = exc

        await _cancel_residual_tasks(_RESIDUAL_DRAIN_TIMEOUT_S)
        if shutdown_error is not None:
            raise shutdown_error

    def close(self) -> None:
        """Close the engine session and release its process lease."""
        self._check_process_and_thread()
        with self._lifecycle_lock:
            if self._status == "closed":
                return
            if self._status == "closing":
                wait_for_owner = True
            else:
                self._status = "closing"
                wait_for_owner = False

        if wait_for_owner:
            self._closed_event.wait()
            return

        # Detached owners are not this session's to stop: reap the ones that
        # have already exited and leave the rest running, which is what the
        # caller detached them for.
        _detach.prune(self._detached_children)

        close_error: BaseException | None = None
        try:
            future = self._submit_to_loop(self._shutdown_bridge())
            while True:
                try:
                    future.result()
                    break
                except KeyboardInterrupt as exc:
                    close_error = close_error or exc
                    if future.done():
                        break
        except BaseException as exc:
            close_error = exc
        finally:
            try:
                self._shutdown_async_generators()
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
            finally:
                try:
                    if not self._loop_stopped.is_set():
                        self._loop.call_soon_threadsafe(self._loop.stop)
                    self._loop_thread.join()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                finally:
                    with self._lifecycle_lock:
                        self._status = "closed"
                    release_session_lease(self, self._lease_pid)
                    self._closed_event.set()

        if close_error is not None:
            raise close_error

    def __enter__(self) -> Api:
        self._check_process_and_thread()
        with self._lifecycle_lock:
            if self._status != "open":
                raise ApiClosedError("The Api session is closing or closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
