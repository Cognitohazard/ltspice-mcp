"""Work an owner starts on the event loop and does not await itself.

A coroutine handed to ``create_task`` and forgotten is invisible: nothing can
say whether it has finished, so anything that needs its effect — a test, a
shutdown flush, a reader of the record it writes — is left to poll for a side
effect or to sleep. Every task the server starts without awaiting it goes
through a :class:`BackgroundTasks` that its owner keeps, so the owner can say
when that work has settled.

A task may carry a key (a job id) so an owner holding work for many jobs can
answer for one of them. A task's failure stays its own: ``settled`` waits for
it and does not raise it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class BackgroundTasks:
    """The tasks one owner started and has not awaited."""

    def __init__(self) -> None:
        self._tasks: dict[asyncio.Task[Any], str | None] = {}

    def spawn(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        key: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task[_T]:
        """Start ``coro`` on ``loop`` (the running loop by default) and keep it
        until it finishes."""
        task = (loop or asyncio.get_running_loop()).create_task(coro)
        self._tasks[task] = key
        task.add_done_callback(self._forget)
        return task

    def _forget(self, task: asyncio.Task[Any]) -> None:
        self._tasks.pop(task, None)

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def pending(self, key: str | None = None) -> list[asyncio.Task[Any]]:
        """The unfinished tasks, or only those started under ``key``."""
        return [task for task, owner in self._tasks.items() if key is None or owner == key]

    async def settled(self, key: str | None = None) -> None:
        """Return once no task (under ``key``, when given) is left, counting
        the ones started while this waits."""
        while pending := self.pending(key):
            await asyncio.wait(pending)
