"""Command-line front end over the same handlers the MCP tools dispatch to.

One engine, two bindings. Each subcommand is a thin adapter onto a registered
tool handler — the identical function ``server.call_tool`` invokes — so there is
no second implementation of staging, linting, job ownership, locking or result
parsing to keep in step. ``--json`` prints that handler's ``structuredContent``
unchanged; the human channel is the handler's own text summary.

Two rules shape everything here:

* **Block to terminality.** The coordinator, the parallelism cap and the cancel
  authority live in the process that submitted the job. A one-shot process that
  exits while its cases are running leaves nobody holding them: the job store
  sees the owner pid die and reconciles the record to failed/server_restarted
  (``experiment_store._reconcile_restart``), so the run is lost, not detached.
  Anything that launches simulations therefore waits for a terminal status, and
  every early exit — deadline, Ctrl-C — cancels what this process owns first.
  ``--no-wait`` is refused while no daemon owner exists to hand the job to.
* **A CLI invocation is a parallel session.** It takes the same cross-process
  circuit-file locks, records the same owner pid, and kills only its own
  simulator processes. Two invocations sharing a working directory coordinate
  exactly as two server sessions do — including the documented residual that
  ``max_parallel_sims`` is per process, so N concurrent invocations permit N
  times the cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - import-time weight is the point
    from collections.abc import AsyncIterator, Callable, Sequence

    from mcp import types

    from ltspice_mcp.state import SessionState

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
"""The call completed and reported every result it promised."""

EXIT_INTERNAL = 1
"""An unexpected error escaped; the message names the exception."""

EXIT_REFUSED = 2
"""The request was rejected and nothing was committed — bad arguments, a path
outside the sandbox, a revision conflict, an unknown job."""

EXIT_FAILED = 3
"""Work started and failed."""

EXIT_PARTIAL = 4
"""Terminal, but short of what was asked for: some runs, checks or recipes did
not produce a result. The payload reconciles the shortfall."""

EXIT_UNFINISHED = 5
"""The subject is still running. Only a status query can end here — a run
started by this process always blocks to a terminal state."""

EXIT_INTERRUPTED = 130
"""Ctrl-C. Anything this process owned was cancelled before exiting."""

IN_PROGRESS = "in_progress"
"""The envelope ``outcome`` of a subject that has not reached a terminal status."""

# Every ``outcome`` the six consolidated tools declare, mapped to the code that
# reports it. Closed on purpose: an outcome nobody mapped is a build that grew a
# state this front end does not understand, and reporting that as success would
# make a script treat an unknown result as a good one. Pinned against the tools'
# own schemas by tests/test_cli.py.
_EXIT_BY_OUTCOME: dict[str, int] = {
    "complete": EXIT_OK,
    "partial": EXIT_PARTIAL,
    "failed": EXIT_FAILED,
    IN_PROGRESS: EXIT_UNFINISHED,
}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

COMMAND_TOOLS: dict[str, str] = {
    "run-experiments": "run_experiments",
    "jobs": "jobs",
    "analyze-results": "analyze_results",
    "inspect": "inspect",
    "edit-schematic": "edit_schematic",
    "verify-circuit": "verify_circuit",
}
"""Subcommand name -> the tool handler it dispatches to. The underscore form of
each tool name is also accepted, so a caller who knows the MCP tool can type it."""

# How long one jobs(wait) leg blocks. The handler allows up to 300s; shorter
# legs bound how long a Ctrl-C or a --timeout deadline waits to be noticed.
_WAIT_LEG_S = 30.0

# After cancelling, how long to keep waiting for the job to actually reach a
# terminal status. Cancellation is an acknowledgement, not a join.
_CANCEL_JOIN_S = 30.0

# Recent-index writes are fire-and-forget background tasks in the dispatch path.
# A long-lived server lets them finish on its own time; a one-shot process must
# not exit mid-write, but must not hang on a contended lock either.
_RECENT_DRAIN_S = 5.0


class _Refused(Exception):
    """The request was rejected before anything ran."""


class _Failed(Exception):
    """Execution failed after work started."""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
Run SPICE experiments and read the results as structured data.

This is the ltspice-mcp engine driven from a shell instead of over MCP. One
coordinator owns each job: it expands the sweep, runs the cases in parallel up
to the configured cap, holds the cancel authority, and reconciles what came back
against what was asked for. Results come back parsed — node voltages, branch
currents, per-device small-signal parameters, measured values — so nothing has
to scrape a rawfile.

Every subcommand takes its arguments as a JSON object, the same object the
matching MCP tool takes: inline, @FILE, or - to read stdin.
"""

_EPILOG = """\
exit codes:
  0   completed; every promised result is present
  1   unexpected internal error
  2   refused — bad arguments, a denied path, a revision conflict, an unknown
      job; nothing was committed
  3   execution failed after work started
  4   terminal but incomplete — some runs, checks or recipes produced no result
  5   still running (only a status query can end here)
  130 interrupted; anything this process owned was cancelled first

waiting:
  run-experiments blocks until the job is terminal. The coordinator that owns a
  running job is this process, so exiting early would abandon it rather than
  detach it. --timeout bounds the wait and cancels the job when it expires;
  Ctrl-C does the same immediately.

parallel invocations:
  Invocations share a working directory safely — the same cross-process file
  locks, owner-pid checks and process-scoped kills the server uses. The
  concurrency cap is per process, so N invocations at once permit N times
  [simulation] max_parallel.

examples:
  spice-mcp verify-circuit --path amp.cir --json
  spice-mcp run-experiments @sweep.json --json
  spice-mcp jobs --action status --job-id exp_a1b2c3 --json
"""


class _VersionAction(argparse.Action):
    """``--version``, resolved when it is asked for.

    argparse's own version action wants the string at parser-construction time,
    which would make every ``--help`` pay for the distribution lookup.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str = "Show the version and exit.",
    ) -> None:
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        from ltspice_mcp import __version__

        print(f"spice-mcp {__version__}")
        parser.exit()


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Options every subcommand accepts."""
    parser.add_argument(
        "args_json",
        metavar="ARGS",
        nargs="?",
        help=(
            "Tool arguments as a JSON object: inline, @FILE to read a file, or - "
            "to read stdin. Omitted means no arguments."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print the result payload as JSON on stdout. This is the stable output.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to ltspice-mcp.toml (default: CWD or $LTSPICE_MCP_CONFIG).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep the configured log level. Without it startup logging is quieted to WARNING.",
    )


def _add_field(
    parser: argparse.ArgumentParser,
    flag: str,
    field: str,
    help_text: str,
    *,
    type_: Any = str,
) -> None:
    """Add a shorthand flag that sets one top-level field of the JSON argument.

    Shorthands exist only for scalar fields that a shell caller reaches for
    constantly. Everything else is expressed in the JSON object, so the CLI
    never carries a second spelling of a tool's argument shape.
    """
    parser.add_argument(
        flag, dest=f"field_{field}", metavar=field.upper(), type=type_, help=help_text
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    _add_field(
        parser, "--request-id", "request_id", "Idempotency key; reusing one replays its receipt."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help=(
            "Give up waiting after SECONDS and cancel the job. Without it the "
            "wait is unbounded, because exiting early would abandon the run."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Submit and exit without waiting. Requires a daemon owner to hand the "
            "job to; none exists today, so this is refused."
        ),
    )


def _add_jobs_options(parser: argparse.ArgumentParser) -> None:
    _add_field(parser, "--action", "action", "status, wait, cancel, list, or runs.")
    _add_field(parser, "--job-id", "job_id", "The job to address.")
    _add_field(
        parser, "--request-id", "request_id", "Address the job by its idempotency key instead."
    )
    _add_field(parser, "--timeout-s", "timeout_s", "wait: seconds to block, 0-300.", type_=float)
    _add_field(parser, "--wait-for", "wait_for", "wait: 'all' (runs and analysis) or 'runs'.")
    _add_field(parser, "--control-token", "control_token", "cancel: the receipt's token.")
    _add_field(parser, "--circuit", "circuit", "list: restrict to one circuit file.")
    _add_field(parser, "--cursor", "cursor", "list/runs: next_cursor from the previous page.")


def _add_verify_options(parser: argparse.ArgumentParser) -> None:
    _add_field(parser, "--path", "path", "Circuit to check: .asc, .cir, .net or .sp.")


class _Subcommand(NamedTuple):
    """One subparser: its one-line help, its longer description, and the
    shorthand flags it adds beyond the shared ones."""

    help: str
    description: str | None = None
    options: Callable[[argparse.ArgumentParser], None] | None = None


# Every subcommand, in the order --help lists them. The name is the hyphenated
# spelling; the underscore form of a two-word name is registered as an alias, so
# a caller who knows the MCP tool can type it.
_SUBCOMMANDS: dict[str, _Subcommand] = {
    "run-experiments": _Subcommand(
        help="Run one or more decks, optionally as a sweep, and return the measured values.",
        description=(
            "Stage the decks, lint them, expand the variation grid, run the cases in "
            "parallel, and return the receipt with any attached measurements. Blocks "
            "until the job is terminal."
        ),
        options=_add_run_options,
    ),
    "jobs": _Subcommand(
        help="Check on, wait for, or stop a run; list recent circuits and their jobs.",
        options=_add_jobs_options,
    ),
    "analyze-results": _Subcommand(
        help="Measure finished runs: metrics, comparisons and waveform extracts.",
    ),
    "inspect": _Subcommand(
        help="Read-only lookups over decks, schematics, symbols, models and libraries.",
    ),
    "edit-schematic": _Subcommand(
        help="Apply a typed op batch to an .asc schematic in one transactional call.",
    ),
    "verify-circuit": _Subcommand(
        help="Check a circuit file, optionally rendering it or comparing it to a reference.",
        options=_add_verify_options,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Deliberately free of any ltspice_mcp import: ``--help`` must not pay for the
    version lookup, simulator detection, symbol-path resolution or the spicelib
    import chain.
    """
    parser = argparse.ArgumentParser(
        prog="spice-mcp",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action=_VersionAction)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, spec in _SUBCOMMANDS.items():
        alias = name.replace("-", "_")
        command = sub.add_parser(
            name,
            aliases=[alias] if alias != name else [],
            help=spec.help,
            description=spec.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_shared_arguments(command)
        if spec.options is not None:
            spec.options(command)

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse argv, exiting on ``--help``/``--version``/usage errors.

    Usage errors exit 2 through argparse's own convention, which is the same
    code a refused request uses — both mean nothing ran.
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command is None:
        parser.error("a COMMAND is required")
    namespace.command = namespace.command.replace("_", "-")
    return namespace


# ---------------------------------------------------------------------------
# Argument payload
# ---------------------------------------------------------------------------


def build_payload(namespace: argparse.Namespace) -> dict[str, Any]:
    """Merge the JSON argument object with any shorthand flags that were set."""
    raw = namespace.args_json
    try:
        if raw is None:
            payload: Any = {}
        elif raw == "-":
            payload = json.loads(sys.stdin.read() or "{}")
        elif raw.startswith("@"):
            payload = json.loads(Path(raw[1:]).read_text(encoding="utf-8"))
        else:
            payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _Refused(f"ARGS is not valid JSON: {exc}") from None
    except OSError as exc:
        raise _Refused(f"could not read ARGS file: {exc}") from None
    if not isinstance(payload, dict):
        raise _Refused(f"ARGS must be a JSON object, got {type(payload).__name__}")

    for dest, value in vars(namespace).items():
        if dest.startswith("field_") and value is not None:
            payload[dest.removeprefix("field_")] = value
    return payload


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def prepare_environment(namespace: argparse.Namespace) -> None:
    """Apply CLI options that are expressed as configuration.

    Routing them through the environment rather than a second config path is
    what keeps the CLI's session identical to a server session: one loader, one
    set of precedence rules, one sandbox.
    """
    if namespace.config:
        os.environ["LTSPICE_MCP_CONFIG"] = namespace.config
    # The six subcommands are the consolidated tool set; that profile is what
    # registers their handlers, so it is not a user choice here.
    os.environ["LTSPICE_MCP_TOOL_PROFILE"] = "consolidated"
    if not namespace.verbose:
        # The server's startup banner is diagnostics for a long-lived process;
        # for a one-shot it is noise ahead of the answer. An explicitly exported
        # level still wins.
        os.environ.setdefault("LTSPICE_MCP_LOG_LEVEL", "WARNING")


@asynccontextmanager
async def session() -> AsyncIterator[SessionState]:
    """Enter the server's own lifespan: same config load, simulator detection,
    job registry, preload and shutdown. Shutdown is what cancels running jobs
    this process owns and flushes persistence, so the CLI inherits it rather
    than reimplementing an exit path."""
    from ltspice_mcp.server import server, server_lifespan

    async with server_lifespan(server) as context:
        yield context["state"]


def _install_interrupt_handler() -> asyncio.Event:
    """Turn the first Ctrl-C into a cancel-then-exit request.

    The handler removes itself, so a second Ctrl-C reaches Python's default and
    aborts hard — a wait that is itself stuck must stay escapable. Platforms
    without loop signal handlers (Windows) fall back to KeyboardInterrupt, which
    ``main`` maps to the same exit code.
    """
    interrupted = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(signal.SIGINT)
        interrupted.set()
        print(
            "spice-mcp: interrupted — cancelling the run this process owns.",
            file=sys.stderr,
        )

    with suppress(NotImplementedError, RuntimeError, AttributeError, ValueError):
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    return interrupted


def _remove_interrupt_handler() -> None:
    with suppress(NotImplementedError, RuntimeError, ValueError):
        asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)


async def _drain_background_writes() -> None:
    """Let in-flight recent-index writes finish before the loop closes.

    ``asyncio.wait`` rather than ``wait_for``: a straggler blocked on a
    contended cross-process lock is abandoned, not waited out.
    """
    from ltspice_mcp.server import _recent_touch_tasks

    pending = [task for task in list(_recent_touch_tasks) if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=_RECENT_DRAIN_S)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def invoke(
    tool: str, arguments: dict[str, Any], state: SessionState
) -> types.CallToolResult:
    """Call one registered tool handler, with the server's own error enrichment.

    ``state.tool_dispatch`` holds the identical ``RegisteredTool`` the MCP
    dispatcher uses, validation wrapper included; the hint and sandbox-guidance
    helpers are imported from ``server`` so the two front ends cannot drift into
    saying different things about the same failure.
    """
    from pydantic import ValidationError

    from ltspice_mcp.errors import (
        BatchJobError,
        JobNotFoundError,
        LTSpiceMCPError,
        NetlistError,
        PathSecurityError,
    )
    from ltspice_mcp.server import _get_error_hint, _notice_circuit, _path_reject_guidance

    # Errors that mean the request could not be ACCEPTED — its arguments, its
    # netlist or the job it names. Everything else that escapes a handler was
    # accepted and then could not be carried out, which is a different thing to
    # a caller deciding whether to fix the invocation or investigate the work.
    refusals = (PathSecurityError, NetlistError, JobNotFoundError, BatchJobError)

    registered = state.tool_dispatch.get(tool)
    if registered is None:
        raise _Failed(f"tool {tool!r} is not registered in this build")

    await _notice_circuit(arguments, state)
    try:
        return await registered.handler(arguments, state)
    except ValidationError as exc:
        raise _Refused(f"invalid arguments for {tool}: {exc}") from None
    except PathSecurityError as exc:
        raise _Refused(f"{exc}\n\n{_path_reject_guidance(state)}") from None
    except LTSpiceMCPError as exc:
        hint = _get_error_hint(type(exc), state.config.tool_profile) if exc.show_hint else None
        message = f"{exc}\n\n{hint}" if hint else str(exc)
        raise (_Refused(message) if isinstance(exc, refusals) else _Failed(message)) from None


def daemon_owner_present() -> bool:
    """Whether a long-lived process can take ownership of a job this call submits.

    Always false: nothing in this package publishes a daemon a one-shot could
    hand a job to. Two things would have to exist — a resident owner whose pid
    the job sidecar can record, and a handoff that records it at submission —
    and until they do, submitting without waiting produces a job whose owner is
    dead the moment this process exits.
    """
    return False


_NO_WAIT_REFUSAL = (
    "--no-wait needs a daemon owner for the job, and this installation has none.\n"
    "A job's coordinator, parallelism cap and cancel authority live in the process "
    "that submitted it. If this process exits while cases are running, the job "
    "store sees its pid die and rewrites the record to failed/server_restarted — "
    "the run is lost, not left running in the background.\n"
    "Drop --no-wait to block until the job is terminal (--timeout bounds the wait "
    "and cancels on expiry), or submit through a running ltspice-mcp server "
    "session, which outlives the call."
)


async def run_command(
    namespace: argparse.Namespace,
    payload: dict[str, Any],
    state: SessionState,
    interrupted: asyncio.Event,
) -> types.CallToolResult:
    """Dispatch one subcommand, blocking to terminality where one is launched."""
    tool = COMMAND_TOOLS[namespace.command]
    if tool != "run_experiments":
        return await invoke(tool, payload, state)
    return await _run_experiments_blocking(namespace, payload, state, interrupted)


async def _run_experiments_blocking(
    namespace: argparse.Namespace,
    payload: dict[str, Any],
    state: SessionState,
    interrupted: asyncio.Event,
) -> types.CallToolResult:
    """Submit an experiment and stay until it is terminal.

    The receipt is re-rendered by re-asking for the same request, which the
    idempotency layer replays instead of resubmitting. That keeps the printed
    payload the run-experiments envelope — the same shape the MCP tool returns —
    without a second receipt builder here.
    """
    from ltspice_mcp.lib.sweep_utils import generate_id

    # Pin the key before submitting: without one the handler mints a fresh id
    # per call, and the replay that renders the final receipt would resubmit.
    payload.setdefault("request_id", generate_id("cli"))

    result = await invoke("run_experiments", payload, state)
    data = result.structuredContent or {}
    job_id = data.get("job_id")
    if data.get("outcome") != IN_PROGRESS or not job_id:
        return result
    if getattr(namespace, "no_wait", False):
        # Reachable only once a daemon owner exists to hold the job — ``execute``
        # refuses the flag otherwise — and then the receipt is the whole answer.
        return result

    loop = asyncio.get_running_loop()
    deadline = None if namespace.timeout is None else loop.time() + namespace.timeout
    reached_terminal = await _wait_until_terminal(job_id, state, deadline, interrupted)
    if not reached_terminal:
        # Deadline, Ctrl-C, or a job that can no longer be followed. Cancel
        # before leaving: nothing else can.
        await invoke("jobs", {"action": "cancel", "job_id": job_id}, state)
        await _wait_until_terminal(job_id, state, loop.time() + _CANCEL_JOIN_S, None)

    if isinstance(data.get("error"), dict):
        # The submission already reported a fault against itself. Re-asking would
        # render from the replay path, which classifies its own faults as
        # not_started and would downgrade a committed one — losing the fact that
        # cases were running. Keep the report that knows what happened.
        return result
    return await invoke("run_experiments", payload, state)


async def _wait_until_terminal(
    job_id: str,
    state: SessionState,
    deadline: float | None,
    interrupted: asyncio.Event | None,
) -> bool:
    """Block in bounded legs until the job is terminal, the deadline passes, or
    an interrupt arrives. Returns whether terminality was reached."""
    loop = asyncio.get_running_loop()
    while interrupted is None or not interrupted.is_set():
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            return False
        leg = _WAIT_LEG_S if remaining is None else min(_WAIT_LEG_S, remaining)
        waited = await invoke(
            "jobs", {"action": "wait", "job_id": job_id, "timeout_s": leg}, state
        )
        snapshot = waited.structuredContent or {}
        if isinstance(snapshot.get("error"), dict):
            # The wait itself failed, so looping cannot make progress. Report
            # not-terminal: the caller cancels rather than spinning or leaving.
            return False
        if snapshot.get("outcome") != IN_PROGRESS:
            return True
    return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def exit_code_for(result: types.CallToolResult) -> int:
    """Classify a handler result into an exit code.

    The discriminator is the envelope, not a per-tool error string: an ``error``
    block reports whether anything was committed, and ``outcome`` reports how
    much of what was asked for came back.
    """
    data = result.structuredContent or {}
    error = data.get("error")
    if isinstance(error, dict):
        return (
            EXIT_FAILED if error.get("commit_state") in ("committed", "unknown") else EXIT_REFUSED
        )
    if result.isError:
        return EXIT_REFUSED
    return _EXIT_BY_OUTCOME.get(data.get("outcome", ""), EXIT_INTERNAL)


def emit(namespace: argparse.Namespace, result: types.CallToolResult, code: int) -> None:
    """Write the result. ``--json`` prints the handler's structuredContent
    unchanged — that is the contract; the human channel is the handler's own
    text summary, which is presentation only."""
    from ltspice_mcp.tools._base import result_text

    data = result.structuredContent
    if namespace.as_json:
        sys.stdout.write(json.dumps(data if data is not None else {}, ensure_ascii=False) + "\n")
        return
    text = result_text(result)
    if text:
        print(text)
    hint = (data or {}).get("hint")
    if code != EXIT_OK and hint and hint not in text:
        print(hint, file=sys.stderr)


def emit_error(namespace: argparse.Namespace, code: str, message: str, exit_code: int) -> int:
    """Report a failure the CLI itself classified, and return its exit code."""
    if namespace.as_json:
        payload = {"error": {"code": code, "message": message, "stage": "cli"}}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"spice-mcp: {message}", file=sys.stderr)
    return exit_code


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def execute(namespace: argparse.Namespace) -> int:
    """Run one parsed invocation and return its exit code."""
    try:
        payload = build_payload(namespace)
    except _Refused as exc:
        return emit_error(namespace, "usage", str(exc), EXIT_REFUSED)

    if getattr(namespace, "no_wait", False) and not daemon_owner_present():
        # Refused before the session exists, so nothing is staged or submitted.
        return emit_error(namespace, "no_wait_unavailable", _NO_WAIT_REFUSAL, EXIT_REFUSED)

    prepare_environment(namespace)
    try:
        async with session() as state:
            interrupted = _install_interrupt_handler()
            try:
                result = await run_command(namespace, payload, state, interrupted)
            except _Refused as exc:
                return emit_error(namespace, "refused", str(exc), EXIT_REFUSED)
            except _Failed as exc:
                return emit_error(namespace, "failed", str(exc), EXIT_FAILED)
            finally:
                _remove_interrupt_handler()
                await _drain_background_writes()
            code = exit_code_for(result)
            emit(namespace, result, code)
            return EXIT_INTERRUPTED if interrupted.is_set() else code
    except Exception as exc:  # the message is the diagnostic, not a swallow
        return emit_error(namespace, "internal", f"{type(exc).__name__}: {exc}", EXIT_INTERNAL)


async def run(argv: Sequence[str] | None = None) -> int:
    """Parse and run, without owning the event loop. The tested entry point."""
    return await execute(parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point.

    Parsing happens before the loop starts so ``--help`` and ``--version`` cost
    nothing but argparse — no config load, no simulator detection, no spicelib.
    """
    namespace = parse_args(argv)
    try:
        code = asyncio.run(execute(namespace))
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    sys.exit(code)
