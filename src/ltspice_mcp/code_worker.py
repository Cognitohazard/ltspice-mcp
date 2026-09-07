"""The warm Python worker behind the ``run_code`` tool.

The server spawns this module once, lazily, as

    sys.executable -m ltspice_mcp.code_worker <working-dir> <config-path-or-empty>

and keeps it for the session. It owns one ``Api`` on that working directory
(its own process, so its own engine lease and its own runner) and runs each
snippet the server sends in a fresh namespace around that same live ``Api``.
State between calls lives on disk — jobs, artifacts, files — exactly as it
does for the six tools.

Protocol: JSON lines. Requests arrive on the worker's original stdin and
replies leave on its original stdout, but through private duplicates taken at
startup. Descriptor 0 is then pointed at the null device and descriptor 1 at
stderr, so a snippet that writes to fd 1 directly, a native library that
prints, or a subprocess the snippet spawns lands in the server's log rather
than in the reply stream, and no child can inherit the reply pipe and hold it
open past this process's death. Python-level ``print`` is captured per call.

Interrupt: the server sends SIGINT to this process alone (POSIX; the worker
sits in its own session, so a terminal Ctrl-C never reaches it and its own
signal never reaches a simulator). It becomes KeyboardInterrupt in the
snippet, which the API's blocking waits honour by cancelling the run.

Parent death: the reader thread sees EOF on the request pipe when the server
dies. It interrupts the snippet, lets the main thread close the ``Api``
(cancelling the jobs this process owns) and exits within a bound whether or
not either finished.
"""

from __future__ import annotations

import ast
import io
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import IO, Any

#: Output caps, stated in the tool description so a model slices before it prints.
STDOUT_HEAD_CHARS = 12_000
STDOUT_TAIL_CHARS = 4_000
STDERR_TAIL_CHARS = 4_000
RESULT_CHARS = 4_000
TRACEBACK_CHARS = 4_000

#: After the server is gone, how long the worker gives ``Api.close()`` before
#: it exits regardless.
EXIT_GRACE_S = 10.0


class _Capture(io.TextIOBase):
    """A text sink that keeps the first ``head`` and the last ``tail`` characters.

    What was written in between is counted, not kept: the reply says how many
    characters were elided, and the elision marker sits where they were.
    """

    def __init__(self, head: int, tail: int) -> None:
        self._head_cap = head
        self._tail_cap = tail
        self._head: list[str] = []
        self._head_len = 0
        self._tail = ""
        self.total = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        n = len(s)
        self.total += n
        room = self._head_cap - self._head_len
        if room > 0:
            self._head.append(s[:room])
            self._head_len += min(room, n)
            s = s[room:]
        if s:
            self._tail = (self._tail + s)[-self._tail_cap :] if self._tail_cap else ""
        return n

    @property
    def dropped(self) -> int:
        return self.total - self._head_len - len(self._tail)

    def text(self) -> str:
        head = "".join(self._head)
        if self.dropped:
            return f"{head}\n... {self.dropped} characters elided ...\n{self._tail}"
        return head + self._tail


def _tail(text: str, cap: int) -> str:
    return text if len(text) <= cap else "..." + text[-cap:]


def _namespace(api: Any) -> dict[str, Any]:
    """The names a snippet starts with, rebuilt around the same live ``Api``."""
    import numpy as np

    return {
        "__name__": "__main__",
        "api": api,
        "np": np,
        "load_raw": api.load_raw,
        "measurements": api.measurements,
        "reference": api.reference,
    }


def empty_reply(status: str) -> dict[str, Any]:
    """The reply fields every snippet run produces, at their empty values.

    The one spelling of the shape: ``execute`` fills it in, the serve loop
    sends it as-is for a snippet the interrupt beat, and the server's tool
    builds its own reply on top of it and derives its schema's required keys
    from it.
    """
    return {
        "status": status,
        "result": None,
        "error": None,
        "stdout": "",
        "stderr": "",
        "truncated": False,
        "chars_dropped": 0,
        "elapsed_s": 0.0,
    }


def _exit_after_grace() -> None:
    """Leave within ``EXIT_GRACE_S`` whatever the main thread is doing."""
    threading.Timer(EXIT_GRACE_S, lambda: os._exit(0)).start()


def execute(code: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Run one snippet; never raise. The last statement's value, if it is an
    expression, comes back as ``result`` (its repr) — a REPL's ergonomics,
    so a snippet need not ``print`` to answer."""
    out = _Capture(STDOUT_HEAD_CHARS, STDOUT_TAIL_CHARS)
    err = _Capture(0, STDERR_TAIL_CHARS)
    reply = empty_reply("ok")
    started = time.monotonic()
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        tree = ast.parse(code, "<run_code>", "exec")
        trailing = None
        if tree.body and isinstance(last := tree.body[-1], ast.Expr):
            tree.body.pop()
            trailing = ast.Expression(last.value)
        exec(compile(tree, "<run_code>", "exec"), namespace)
        if trailing is not None:
            value = eval(compile(trailing, "<run_code>", "eval"), namespace)
            if value is not None:
                text = repr(value)
                reply["result"] = (
                    text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + "..."
                )
    except KeyboardInterrupt:
        reply["status"] = "interrupted"
    except BaseException as exc:
        reply["status"] = "error"
        reply["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": _tail(traceback.format_exc(), TRACEBACK_CHARS),
        }
    finally:
        sys.stdout, sys.stderr = saved
    reply.update(
        stdout=out.text(),
        stderr=err.text(),
        truncated=(out.dropped + err.dropped) > 0,
        chars_dropped=out.dropped + err.dropped,
        elapsed_s=round(time.monotonic() - started, 3),
    )
    return reply


def _send(replies: IO[str], reply: dict[str, Any]) -> None:
    """Write one reply line, finishing it even if an interrupt lands mid-write."""
    # UTF-8 as written: escaping every non-ASCII character to six bytes was
    # how a reply of mostly µ and ° overran the server's line reader.
    line = json.dumps(reply, ensure_ascii=False) + "\n"
    for _ in range(3):
        try:
            replies.write(line)
            replies.flush()
            return
        except KeyboardInterrupt:
            continue


def _read_requests(
    requests: IO[str],
    inbox: queue.Queue[dict[str, Any] | None],
    busy: threading.Event,
    parent_gone: threading.Event,
) -> None:
    """The reader thread: requests into the inbox; on EOF, the parent is gone.

    The interrupt on EOF is sent only while the main thread is inside a
    snippet (``busy``), and the main thread checks ``parent_gone`` after
    raising ``busy`` — so a snippet that was running is interrupted and one
    that was about to start is not started, whichever side the race falls
    on. A signal sent to an idle main thread would land outside the snippet's
    handler, which is how the first draft of this died on its own watchdog.
    """
    try:
        for line in requests:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                continue
    except (OSError, ValueError):
        pass
    parent_gone.set()
    inbox.put(None)
    if busy.is_set() and os.name != "nt":
        os.kill(os.getpid(), signal.SIGINT)
    _exit_after_grace()


def _private_channels() -> tuple[IO[str], IO[str]]:
    """Take the protocol descriptors away from everything a snippet can reach."""
    request_fd = os.dup(0)
    reply_fd = os.dup(1)
    os.set_inheritable(request_fd, False)
    os.set_inheritable(reply_fd, False)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    requests = os.fdopen(request_fd, "r", encoding="utf-8")
    replies = os.fdopen(reply_fd, "w", encoding="utf-8")
    return requests, replies


def _serve_loop(
    api: Any,
    inbox: queue.Queue[dict[str, Any] | None],
    replies: IO[str],
    busy: threading.Event,
    parent_gone: threading.Event,
) -> None:
    pending: dict[str, Any] | None = None
    while True:
        try:
            if pending is not None:
                # An interrupt landed between the snippet's end and the write
                # of its reply: the reply still goes out, as interrupted.
                pending.update(status="interrupted")
                _send(replies, pending)
                pending = None
            message = inbox.get()
            if message is None or message.get("op") == "close":
                return
            if message.get("op") != "run":
                continue
            reply: dict[str, Any] = {
                "op": "reply",
                "seq": message.get("seq"),
                **empty_reply("interrupted"),
            }
            pending = reply
            busy.set()
            try:
                if parent_gone.is_set():
                    return
                reply.update(execute(str(message.get("code", "")), _namespace(api)))
            finally:
                busy.clear()
            _send(replies, reply)
            pending = None
        except KeyboardInterrupt:
            # A signal that landed outside the snippet: it had just ended, or
            # the server's interrupt raced the reply. Nothing to stop.
            continue


def serve(working_dir: str, config_path: str | None) -> int:
    requests, replies = _private_channels()
    if os.name != "nt":
        # Installed explicitly: a server started with SIGINT ignored hands
        # that disposition down, and the interrupt would then be a no-op.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    from ltspice_mcp.lib.observability import configure_stderr_logging

    configure_stderr_logging(os.getenv("LTSPICE_MCP_LOG_LEVEL", "WARNING"))
    try:
        from ltspice_mcp.api import Api

        api = Api(working_dir=working_dir, config_path=config_path)
    except BaseException as exc:
        _send(replies, {"op": "boot_failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    # The server's stderr is this process's log too: read [logging] level
    # off the booted engine, the way the server and the detached owner do.
    configure_stderr_logging(api._state.config.log_level)  # pyright: ignore[reportPrivateUsage]
    _send(replies, {"op": "ready", "pid": os.getpid()})

    inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
    busy = threading.Event()
    parent_gone = threading.Event()
    threading.Thread(
        target=_read_requests, args=(requests, inbox, busy, parent_gone), daemon=True
    ).start()
    try:
        _serve_loop(api, inbox, replies, busy, parent_gone)
    finally:
        # Whether the server said close or vanished: no interrupt may abort
        # the close now, and the close itself is bounded by the exit timer.
        if os.name != "nt":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        _exit_after_grace()
        try:
            api.close()
        except BaseException:
            traceback.print_exc()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m ltspice_mcp.code_worker <working-dir> <config-path|''>",
            file=sys.stderr,
        )
        return 2
    code = serve(argv[0], argv[1] or None)
    # Never linger in interpreter teardown behind the engine's loop thread.
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
