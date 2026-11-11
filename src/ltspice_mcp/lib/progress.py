"""Progress a long tool call reports to the client that asked for it.

A request may ask for progress (on the wire, with ``_meta.progressToken``);
the server may then send ``notifications/progress`` for that request, each
with a ``progress`` greater than the last, and a client may restart its
timeout on each one (MCP, "Progress"). The server binds its request's
session as the reporter for each call (``reporting_to``), and the session
sends a report only to a caller that asked for one. A wait that can hold a
call open reports through :func:`watch`, which only awaits when no reporter
is bound, as in the Python API.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from typing import TypeVar

logger = logging.getLogger(__name__)

#: Sends one progress notification: ``(progress, total, message)``.
Reporter = Callable[[float, float | None, str | None], Awaitable[None]]

#: How often a watched wait reads what it reports.
POLL_S = 1.0

_T = TypeVar("_T")


class _Call:
    """One call's progress: where it goes, and the last value sent.

    A report goes out only when it is above the last one sent for the call,
    however many waits the call makes, and none goes out once the call is
    answered (``close``). A task the call started inherits this binding and
    can outlive the call, as a job's coordinator does; once the call is
    closed it holds nothing of the request.
    """

    def __init__(self, reporter: Reporter) -> None:
        self._reporter: Reporter | None = reporter
        self._last = -1.0

    def close(self) -> None:
        self._reporter = None

    async def report(self, done: int, total: int, message: str) -> None:
        """Send ``done`` of ``total`` if it is above the last report; a
        ``total`` of zero is not yet known and is left out. A report that fails,
        as when the client has gone, is logged: progress is never what a call
        answers."""
        reporter = self._reporter
        if reporter is None or done <= self._last:
            return
        self._last = done
        try:
            await reporter(done, total or None, message)
        except Exception:
            logger.debug("A progress notification was not sent", exc_info=True)


_call: ContextVar[_Call | None] = ContextVar("mcp_progress_call", default=None)


@contextlib.contextmanager
def reporting_to(reporter: Reporter) -> Iterator[None]:
    """Report the progress of the work run inside this block to ``reporter``,
    until the block ends."""
    call = _Call(reporter)
    bound = _call.set(call)
    try:
        yield
    finally:
        call.close()
        _call.reset(bound)


async def watch(work: Awaitable[_T], read: Callable[[], tuple[int, int, str]]) -> _T:
    """Await ``work``, reporting ``read()``, a ``(done, total, message)``: at
    once, so a caller learns the wait has begun and how much there is to do,
    then every ``POLL_S`` while ``done`` grows, and as the wait ends, so it
    learns where the wait left off before the result."""
    call = _call.get()
    if call is None:
        return await work
    task = asyncio.ensure_future(work)
    try:
        while True:
            await call.report(*read())
            if task.done():
                return task.result()
            await asyncio.wait({task}, timeout=POLL_S)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
