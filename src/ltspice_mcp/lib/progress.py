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

_reporter: ContextVar[Reporter | None] = ContextVar("mcp_progress_reporter", default=None)


@contextlib.contextmanager
def reporting_to(reporter: Reporter) -> Iterator[None]:
    """Report the progress of the work run inside this block to ``reporter``."""
    bound = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(bound)


async def watch(work: Awaitable[_T], read: Callable[[], tuple[int, int, str]]) -> _T:
    """Await ``work``, reporting ``read()``, a ``(done, total, message)``, each
    time ``done`` has grown since the last report.

    The first report goes out at once, so a caller learns the wait has begun
    and how much there is to do, and the last as the wait ends, so it learns
    where the wait left off before the result. A ``total`` of zero is not yet
    known and is left out. A report that fails, as when the client has gone,
    is logged and the wait goes on: progress is never what a call answers.
    """
    reporter = _reporter.get()
    if reporter is None:
        return await work
    task = asyncio.ensure_future(work)
    sent = -1

    async def report_growth() -> None:
        nonlocal sent
        done, total, message = read()
        if done <= sent:
            return
        sent = done
        try:
            await reporter(done, total or None, message)
        except Exception:
            logger.debug("A progress notification was not sent", exc_info=True)

    try:
        while True:
            await report_growth()
            finished, _ = await asyncio.wait({task}, timeout=POLL_S)
            if finished:
                await report_growth()
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
