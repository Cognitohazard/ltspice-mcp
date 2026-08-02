"""Command-line front end over the same handlers the MCP tools dispatch to.

One engine, two bindings. Each subcommand is a thin adapter onto a registered
tool handler — the identical function ``server.call_tool`` invokes — so there is
no second implementation of staging, linting, job ownership, locking or result
parsing to keep in step. ``run`` is the one-deck on-ramp: it translates a deck
path and a few flags into the canonical run-experiments payload and enters that
same dispatch path, a translation layer rather than a second engine. ``--json``
prints the handler's ``structuredContent`` unchanged on one line — the
parse-stable contract. Human mode prints the handler's text summary and then the
same ``structuredContent`` pretty-printed, unless ``--table`` selects the
supported analysis or jobs table view. Only ``--json`` is parse-stable.

Three rules shape everything here:

* **Block to terminality.** The coordinator, the parallelism cap and the cancel
  authority live in the process that submitted the job. A one-shot process that
  exits while its cases are running leaves nobody holding them: the job store
  sees the owner pid die and reconciles the record to failed/server_restarted
  (``experiment_store._reconcile_restart``), so the run is lost, not detached.
  Anything that launches simulations therefore waits for a terminal status, and
  every early exit — deadline, Ctrl-C — cancels what this process owns first.
  ``--no-wait`` is refused while no daemon owner exists to hand the job to.
* **Say what happened, and only that.** The printed envelope, the exit code and
  stderr describe one story. An interrupt that lands after the job is terminal
  cancels nothing and reports the result the run produced; a cancel this process
  issued and could not confirm is reported as unconfirmed rather than as a job
  politely still running.
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
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

if TYPE_CHECKING:  # pragma: no cover - import-time weight is the point
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

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

EXIT_UNCONFIRMED = 6
"""This process issued a cancel and could not confirm it: the job never reached
a terminal status inside the join window, so whether its simulator processes are
still alive is unknown. Deliberately not EXIT_UNFINISHED — that code says the
subject is running normally, and this one says nobody knows."""

EXIT_INTERRUPTED = 130
"""Ctrl-C, and it changed the outcome: a job this process owned was still
running and was cancelled. An interrupt that lands after the work is terminal
cannot cancel anything, so it reports the result it actually got instead."""

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
    "run": "run_experiments",
    "run-experiments": "run_experiments",
    "jobs": "jobs",
    "analyze-results": "analyze_results",
    "inspect": "inspect",
    "edit-schematic": "edit_schematic",
    "verify-circuit": "verify_circuit",
}
"""Subcommand name -> the tool handler it dispatches to. The underscore form of
each tool name is also accepted, so a caller who knows the MCP tool can type it.
``run`` shares run-experiments' handler because it IS run-experiments with the
payload built from flags — see :func:`build_run_payload`."""

# How long one jobs(wait) leg blocks. The handler allows up to 300s; shorter
# legs bound how long a Ctrl-C or a --timeout deadline waits to be noticed.
_WAIT_LEG_S = 30.0

# After cancelling, how long to keep waiting for the job to actually reach a
# terminal status. Cancellation is an acknowledgement, not a join.
_CANCEL_JOIN_S = 30.0

# How much longer than the dwell it asked for one wait leg may take before this
# process stops waiting on it. The handler's own timeout is the real bound; this
# is the backstop for a leg that does not honour it, because a wait that never
# returns would defeat both Ctrl-C and --timeout — the two bounds this loop
# exists to provide.
_LEG_GRACE_S = 5.0

# Wait legs this process cancelled and walked away from. Referenced only so a
# still-pending leg is not garbage-collected mid-cancellation; nothing ever
# awaits them, which is the point (see _wait_leg).
_abandoned_legs: set[asyncio.Future[Any]] = set()

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

quick start:
  spice-mcp run deck.cir --measure all --json    one deck, one run, measured values
  spice-mcp run-experiments @exp.json --json     full payloads: sweeps, Monte Carlo

This is the ltspice-mcp engine driven from a shell instead of over MCP. One
coordinator owns each job: it expands the sweep, runs the cases in parallel up
to the configured cap, holds the cancel authority, and reconciles what came back
against what was asked for. Results come back parsed — node voltages, branch
currents, per-device small-signal parameters, measured values — so nothing has
to scrape a rawfile.

Apart from run (which takes a deck path), every subcommand takes its arguments
as a JSON object, the same object the matching MCP tool takes: inline, @FILE,
or - to read stdin.
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
  6   a cancel this process issued was never confirmed; whether the job's
      simulator processes are still alive is unknown
  130 interrupted, and the interrupt cancelled a job that was still running

waiting:
  run-experiments blocks until the job is terminal. The coordinator that owns a
  running job is this process, so exiting early would abandon it rather than
  detach it. --timeout bounds the wait and cancels the job when it expires;
  Ctrl-C does the same, without waiting out the poll it landed in. An interrupt
  that arrives after the job is already terminal cancels nothing and reports the
  result the run produced.

parallel invocations:
  Invocations share a working directory safely — the same cross-process file
  locks, owner-pid checks and process-scoped kills the server uses. The
  concurrency cap is per process, so N invocations at once permit N times
  [simulation] max_parallel.

examples:
  spice-mcp run deck.cir --measure all --json
  spice-mcp verify-circuit --path amp.cir --json
  spice-mcp run-experiments @sweep.json --json
  spice-mcp jobs --action status --job-id exp_a1b2c3 --json
"""


_RUN_EXEMPLAR = "spice-mcp run deck.cir --measure all --json"
"""The on-ramp's minimal valid invocation. Every refusal and every parser-level
usage error appends its subcommand's exemplar (the other six live on their
``_SUBCOMMANDS`` rows), because a strict shape that rejects a first guess
without an example of a valid call leaves the caller no path back (the measured
walk-away). Errors with no subcommand yet get this one: the caller is at the
front door, and the on-ramp is the answer there."""


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors carry a recovery line.

    A refused request appends its subcommand's minimal valid invocation; an
    unknown flag or an extra positional must say the same thing, because the
    caller who mistyped a flag is the same caller who needs the way back. Each
    parser is told its own exemplar at construction. When the invocation asked
    for ``--json``, the error is also emitted as the one-line JSON envelope so
    a piping caller never gets bare prose on stdout; argparse no longer knows
    the full invocation at error time, so ``build_parser`` records that at
    build time.
    """

    exemplar: str = _RUN_EXEMPLAR
    json_requested: bool = False

    def error(self, message: str) -> NoReturn:
        message = f"{message}\ntry: {self.exemplar}"
        if self.json_requested:
            _emit_error_json("usage", message)
        super().error(message)


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


def _add_json_flag(parser: argparse.ArgumentParser, default: bool | str) -> None:
    """One spelling of ``--json`` for the root parser and every subparser.

    The root passes ``default=False``; subparsers pass ``argparse.SUPPRESS``,
    because a subparser default would overwrite a flag given BEFORE the
    subcommand.
    """
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=default,
        help="Machine-readable one-line receipt (structuredContent, parse-stable).",
    )


def _add_table_flag(parser: argparse.ArgumentParser, default: bool | str) -> None:
    """One spelling of ``--table`` for the root parser and every subparser."""
    parser.add_argument(
        "--table",
        dest="as_table",
        action="store_true",
        default=default,
        help="Human-readable table for analysis results and jobs list/run pages.",
    )


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """The global options, repeated on each subparser so they parse after the
    subcommand too."""
    _add_json_flag(parser, default=argparse.SUPPRESS)
    _add_table_flag(parser, default=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to ltspice-mcp.toml (default: CWD or $LTSPICE_MCP_CONFIG).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Keep the configured log level. Without it (and with no explicit "
            "LTSPICE_MCP_LOG_LEVEL) the ltspice_mcp logger tree is quieted to ERROR."
        ),
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Options every JSON-argument subcommand accepts."""
    parser.add_argument(
        "args_json",
        metavar="ARGS",
        nargs="?",
        help=(
            "Tool arguments as a JSON object: inline, @FILE to read a file, or - "
            "to read stdin. Omitted means no arguments."
        ),
    )
    _add_common_options(parser)


def _add_timeout_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help=(
            "Give up waiting after SECONDS and cancel the job. Without it the "
            "wait is unbounded, because exiting early would abandon the run."
        ),
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
    _add_timeout_option(parser)
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
    """One subparser: its one-line help, its minimal valid invocation (appended
    to every refusal and usage error), its longer description, and the
    shorthand flags it adds beyond the shared ones."""

    help: str
    exemplar: str
    description: str | None = None
    options: Callable[[argparse.ArgumentParser], None] | None = None


# Every JSON-argument subcommand, in the order --help lists them (the `run`
# on-ramp rides its own definition in build_parser). The name is the hyphenated
# spelling; the underscore form of a two-word name is registered as an alias, so
# a caller who knows the MCP tool can type it.
_SUBCOMMANDS: dict[str, _Subcommand] = {
    "run-experiments": _Subcommand(
        help="Run one or more decks, optionally as a sweep, and return the measured values.",
        exemplar="spice-mcp run-experiments @exp.json --json",
        description=(
            "Stage the decks, lint them, expand the variation grid, run the cases in "
            "parallel, and return the receipt with any attached measurements. Blocks "
            "until the job is terminal."
        ),
        options=_add_run_options,
    ),
    "jobs": _Subcommand(
        help="Check on, wait for, or stop a run; list recent circuits and their jobs.",
        exemplar="spice-mcp jobs --action list --json",
        options=_add_jobs_options,
    ),
    "analyze-results": _Subcommand(
        help="Measure finished runs: metrics, comparisons and waveform extracts.",
        exemplar="spice-mcp analyze-results @recipes.json --json",
    ),
    "inspect": _Subcommand(
        help="Read-only lookups over decks, schematics, symbols, models and libraries.",
        exemplar='spice-mcp inspect \'{"queries": [{"kind": "capabilities"}]}\' --json',
    ),
    "edit-schematic": _Subcommand(
        help="Apply a typed op batch to an .asc schematic in one transactional call.",
        exemplar="spice-mcp edit-schematic @ops.json --json",
    ),
    "verify-circuit": _Subcommand(
        help="Check a circuit file, optionally rendering it or comparing it to a reference.",
        exemplar="spice-mcp verify-circuit --path amp.cir --json",
        options=_add_verify_options,
    ),
}


def _exemplar_for(command: str) -> str:
    """The minimal valid invocation for ``command`` (any spelling parse_args
    accepts); the on-ramp's for anything else, per ``_RUN_EXEMPLAR``."""
    spec = _SUBCOMMANDS.get(command)
    return spec.exemplar if spec is not None else _RUN_EXEMPLAR


def build_parser(argv: Sequence[str] | None = None) -> _Parser:
    """Build the argument parser.

    Deliberately free of any ltspice_mcp import: ``--help`` must not pay for the
    version lookup, simulator detection, symbol-path resolution or the spicelib
    import chain.

    ``argv`` is consulted only for the ``--json`` usage-error envelope: a
    parser-level error must respect an asked-for JSON mode, and at error time
    argparse no longer knows the full invocation, so it is recorded here.
    """
    json_requested = argv is not None and "--json" in argv
    parser = _Parser(
        prog="spice-mcp",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.json_requested = json_requested
    parser.add_argument("--version", action=_VersionAction)
    # The global spelling, valid ahead of the subcommand and visible on the
    # first help screen; each subparser re-accepts it after the subcommand.
    _add_json_flag(parser, default=False)
    _add_table_flag(parser, default=False)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    run_command = sub.add_parser(
        "run",
        help="Run one deck once and return the receipt — the on-ramp for a quick check.",
        description=(
            "Translate DECK plus flags into the canonical run-experiments payload\n"
            '({"circuits": [{"path": DECK}]}) and dispatch it exactly as\n'
            "run-experiments would: same staging, same lint gate, same\n"
            "wait-to-terminality, same exit codes, same receipt. Anything beyond one\n"
            "deck — variations, several circuits, custom analysis recipes — is\n"
            "run-experiments' job."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_command.exemplar = _RUN_EXEMPLAR
    run_command.json_requested = json_requested
    run_command.add_argument(
        "deck", metavar="DECK", help="The circuit deck to run (.cir, .net or .sp)."
    )
    _add_common_options(run_command)
    run_command.add_argument(
        "--simulator",
        choices=("ngspice", "ltspice"),
        help="Engine for the run; defaults to the configured simulator.",
    )
    run_command.add_argument(
        "--measure",
        metavar="NAME|all",
        help=(
            "Return measured values with the receipt: 'all' reads back every .MEAS "
            "in the deck, a name reads back that one. Synthesizes the same attached "
            "analysis block a run-experiments 'analyze' carries."
        ),
    )
    _add_timeout_option(run_command)

    for name, spec in _SUBCOMMANDS.items():
        alias = name.replace("-", "_")
        command = sub.add_parser(
            name,
            aliases=[alias] if alias != name else [],
            help=spec.help,
            description=spec.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        command.exemplar = spec.exemplar
        command.json_requested = json_requested
        _add_shared_arguments(command)
        if spec.options is not None:
            spec.options(command)

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse argv, exiting on ``--help``/``--version``/usage errors.

    Usage errors exit 2 through argparse's own convention, which is the same
    code a refused request uses — both mean nothing ran.
    """
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(arguments)
    namespace = parser.parse_args(arguments)
    if namespace.command is None:
        parser.error("a COMMAND is required")
    namespace.command = namespace.command.replace("_", "-")
    if namespace.as_json and namespace.as_table:
        parser.exemplar = _exemplar_for(namespace.command)
        parser.error("--json and --table cannot be used together")
    return namespace


# ---------------------------------------------------------------------------
# Argument payload
# ---------------------------------------------------------------------------


def build_run_payload(namespace: argparse.Namespace) -> dict[str, Any]:
    """Translate ``run DECK`` flags into the canonical run-experiments payload.

    A translation layer, not a second engine: the dict built here is exactly
    what a caller would pass to run-experiments, and it enters the identical
    dispatch and wait path. Keys appear only when their flag was given, so the
    engine's own defaults stay the single source of default behavior — except
    ``execution.wait_s``, forced to 0: the handler's receipt dwell would hold
    off ``--timeout`` and Ctrl-C for up to its 60s default, and the CLI's own
    bounded wait loop is the supervisor here. wait_s is excluded from the
    idempotency fingerprint (it bounds only the response), so an explicit
    run-experiments replay of this request at any dwell still replays. The
    synthesized ``analyze`` block is validated by the handler like an authored
    one — a malformed block is refused before anything is staged.
    """
    deck: str = namespace.deck
    # Classify the lstripped candidate: a shell-quoted argument with a leading
    # space is the same caller mistake, and it must not reach deck staging as
    # a filename.
    candidate = deck.lstrip()
    if candidate == "-" or candidate.startswith(("@", "{")):
        raise _Refused(
            "DECK is a single deck path, not a JSON payload. Multi-circuit, "
            "variation or custom-analysis payloads belong to run-experiments: "
            f"{_exemplar_for('run-experiments')}"
        )
    payload: dict[str, Any] = {"circuits": [{"path": deck}]}
    execution: dict[str, Any] = {"wait_s": 0}
    if namespace.simulator is not None:
        execution["simulator"] = namespace.simulator
    payload["execution"] = execution
    if namespace.measure is not None:
        recipe: dict[str, Any] = {"metric": "measurements"}
        if namespace.measure == "all":
            recipe["key"] = "measurements"
        else:
            recipe["key"] = namespace.measure
            recipe["names"] = [namespace.measure]
        payload["analyze"] = {"recipes": [recipe]}
    return payload


def build_payload(namespace: argparse.Namespace) -> dict[str, Any]:
    """Merge the JSON argument object with any shorthand flags that were set."""
    if namespace.command == "run":
        return build_run_payload(namespace)
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


def prepare_environment(namespace: argparse.Namespace) -> dict[str, str]:
    """The environment variables this invocation's options amount to.

    Routing them through the environment rather than a second config path is
    what keeps the CLI's session identical to a server session: one loader, one
    set of precedence rules, one sandbox. Returned rather than applied, so the
    same mapping that says what to set says what to put back — see
    :func:`prepared_environment`.
    """
    env: dict[str, str] = {}
    if namespace.config:
        env["LTSPICE_MCP_CONFIG"] = namespace.config
    # The subcommands are the consolidated tool set; that profile is what
    # registers their handlers, so it is not a user choice here.
    env["LTSPICE_MCP_TOOL_PROFILE"] = "consolidated"
    if not namespace.verbose and "LTSPICE_MCP_LOG_LEVEL" not in os.environ:
        # The server's startup banner is diagnostics for a long-lived process;
        # for a one-shot it is noise ahead of the answer. An explicitly exported
        # level still wins. This lowers the ROOT level; the ltspice_mcp tree is
        # quieted further to ERROR in prepared_environment, keyed on this same
        # entry.
        env["LTSPICE_MCP_LOG_LEVEL"] = "WARNING"
    return env


@contextmanager
def prepared_environment(namespace: argparse.Namespace) -> Iterator[None]:
    """Apply this invocation's configuration options, then put the process back.

    ``run()`` is a documented reusable entry point, so the mutation cannot be
    left standing: a ``--config`` from one call would silently hand a later call
    a sandbox, a simulator and a set of limits its caller never asked for, and
    nothing about the second invocation would say where they came from. The
    restore set is the same mapping :func:`prepare_environment` returns, so a new
    variable cannot be applied without also being restored.
    """
    env = prepare_environment(namespace)
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    # Quiet by default: anything the caller must know belongs in the payload's
    # observations/warnings channels, not the log stream. A WARNING root still
    # passes this package's own WARNING spam (e.g. the experiment store
    # skipping legacy records), so the ltspice_mcp tree goes to ERROR — keyed
    # on the same env entry, so an explicitly exported LTSPICE_MCP_LOG_LEVEL
    # and --verbose both keep the configured behavior, and the quieting cannot
    # apply without also being restored.
    tree = logging.getLogger("ltspice_mcp")
    saved_level = tree.level if "LTSPICE_MCP_LOG_LEVEL" in env else None
    if saved_level is not None:
        tree.setLevel(logging.ERROR)
    # The server lifespan this invocation enters calls
    # logging.basicConfig(force=True), which strips the ROOT logger's handlers
    # and resets its level — acceptable for a dedicated server process, not
    # for a host application embedding run(). Snapshot and restore both.
    root = logging.getLogger()
    saved_root_level = root.level
    saved_root_handlers = list(root.handlers)
    try:
        yield
    finally:
        root.handlers[:] = saved_root_handlers
        root.setLevel(saved_root_level)
        if saved_level is not None:
            tree.setLevel(saved_level)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@asynccontextmanager
async def session() -> AsyncIterator[SessionState]:
    """Enter the server's own lifespan: same config load, simulator detection,
    job registry, preload and shutdown. Shutdown is what cancels running jobs
    this process owns and flushes persistence, so the CLI inherits it rather
    than reimplementing an exit path."""
    from ltspice_mcp.server import server, server_lifespan

    async with server_lifespan(server) as context:
        yield context["state"]


class _Interrupt:
    """A Ctrl-C request, and what this process actually did about it.

    Two facts, not one, because the exit code and the stderr line have to report
    the second: at the moment the signal lands nothing knows whether there is
    still anything to cancel, and a job that has already finished cannot be
    stopped. Only the code that owns the job sets ``cancelled``, and only then
    may this invocation claim a cancel.
    """

    def __init__(self) -> None:
        self.requested = asyncio.Event()
        self.cancelled = False


def _install_interrupt_handler() -> _Interrupt:
    """Turn the first Ctrl-C into a cancel-then-exit request.

    The handler removes itself, so a second Ctrl-C reaches Python's default and
    aborts hard — a wait that is itself stuck must stay escapable. Platforms
    without loop signal handlers (Windows) fall back to KeyboardInterrupt, which
    ``main`` maps to the same exit code.
    """
    interrupt = _Interrupt()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(signal.SIGINT)
        interrupt.requested.set()
        # Only that it was received. Whether anything gets cancelled is decided
        # by the wait loop, which says so itself.
        print("spice-mcp: interrupted.", file=sys.stderr)

    with suppress(NotImplementedError, RuntimeError, AttributeError, ValueError):
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    return interrupt


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
        compact_validation_error,
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
        raise _Refused(f"invalid arguments for {tool}: {compact_validation_error(exc)}") from None
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
    interrupt: _Interrupt,
) -> types.CallToolResult:
    """Dispatch one subcommand, blocking to terminality where one is launched."""
    tool = COMMAND_TOOLS[namespace.command]
    if tool != "run_experiments":
        return await invoke(tool, payload, state)
    return await _run_experiments_blocking(namespace, payload, state, interrupt)


CANCEL_CONFIRMED_KEY = "cancel_confirmed"
"""Payload marker: present and false when this process issued a cancel and could
not confirm it. Absent means no cancel of this kind happened — the key exists to
be checked for ``False``, not for presence. It is the one thing the CLI adds to a
handler's payload, because it is a fact about this process rather than about the
job, and no handler is in a position to report it."""

_UNCONFIRMED_CANCEL_MESSAGE = (
    "cancel was issued but NOT confirmed — the job did not reach a terminal "
    "status within {join:.0f}s of the kill, so whether its simulator processes "
    "are still running is unknown. This is not the same as 'still running': "
    "something tried to stop it. Check with 'spice-mcp jobs --action status "
    "--job-id {job_id}' before launching another run in this directory."
)

_LIFECYCLE_FIELDS: tuple[str, ...] = (
    "status",
    "outcome",
    "completeness",
    "runs",
    "failures",
    "observations",
    "artifacts",
)
"""What a terminal status read may overwrite on a receipt that already reported a
fault against itself. Everything else on that receipt — above all its ``error``
block — is the report of the code that WAS there when the fault happened, and a
later read cannot improve on it."""


async def _run_experiments_blocking(
    namespace: argparse.Namespace,
    payload: dict[str, Any],
    state: SessionState,
    interrupt: _Interrupt,
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
    reached_terminal = await _wait_until_terminal(job_id, state, deadline, interrupt)
    confirmed = True
    if not reached_terminal:
        # Deadline, Ctrl-C, or a job that can no longer be followed. Cancel
        # before leaving: nothing else can. Claimed here rather than by the
        # signal handler, because this is the point at which it is true.
        print("spice-mcp: cancelling the run this process owns.", file=sys.stderr)
        interrupt.cancelled = interrupt.requested.is_set()
        await invoke("jobs", {"action": "cancel", "job_id": job_id}, state)
        confirmed = await _wait_until_terminal(job_id, state, loop.time() + _CANCEL_JOIN_S, None)

    if isinstance(data.get("error"), dict):
        # The submission already reported a fault against itself. Re-asking would
        # render from the replay path, which classifies its own faults as
        # not_started and would downgrade a committed one — losing the fact that
        # cases were running. Keep that report, and bring only its lifecycle
        # fields up to date so it does not still say 'in_progress' about a job
        # this process watched finish.
        await _refresh_lifecycle(result, job_id, state)
    else:
        result = await invoke("run_experiments", payload, state)
    if not confirmed and result.structuredContent is not None:
        result.structuredContent[CANCEL_CONFIRMED_KEY] = False
    return result


async def _refresh_lifecycle(
    result: types.CallToolResult, job_id: str, state: SessionState
) -> None:
    """Update a standing receipt's lifecycle fields from one status read, in place.

    One cheap read, not a replay: the receipt's own ``error`` block and its
    handles stay exactly as the code that hit the fault wrote them, and only the
    fields that describe where the job GOT TO are refreshed.
    """
    data = result.structuredContent
    if data is None:
        return
    try:
        status = await invoke("jobs", {"action": "status", "job_id": job_id}, state)
    except (_Refused, _Failed):
        # A receipt that cannot be refreshed is still a receipt, and it still
        # carries the handles that reach the job. Losing it would be worse.
        return
    fresh = status.structuredContent or {}
    for name in _LIFECYCLE_FIELDS:
        if name in fresh:
            data[name] = fresh[name]


async def _wait_until_terminal(
    job_id: str,
    state: SessionState,
    deadline: float | None,
    interrupt: _Interrupt | None,
) -> bool:
    """Block in bounded legs until the job is terminal, the deadline passes, or
    an interrupt arrives. Returns whether terminality was reached."""
    loop = asyncio.get_running_loop()
    while interrupt is None or not interrupt.requested.is_set():
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            return False
        leg = _WAIT_LEG_S if remaining is None else min(_WAIT_LEG_S, remaining)
        waited = await _wait_leg(job_id, state, leg, interrupt)
        if waited is None:
            # The leg was abandoned — interrupted, or overrunning the dwell it
            # was given. Either way this loop cannot report terminality.
            return False
        snapshot = waited.structuredContent or {}
        if isinstance(snapshot.get("error"), dict):
            # The wait itself failed, so looping cannot make progress. Report
            # not-terminal: the caller cancels rather than spinning or leaving.
            return False
        if snapshot.get("outcome") != IN_PROGRESS:
            return True
    return False


async def _wait_leg(
    job_id: str,
    state: SessionState,
    leg: float,
    interrupt: _Interrupt | None,
) -> types.CallToolResult | None:
    """One dwell, raced against the interrupt. ``None`` means it did not finish.

    Racing is the whole point: checking the interrupt only between legs makes
    Ctrl-C wait out the leg it landed in, which is up to ``_WAIT_LEG_S`` of a
    process that has been told to stop.

    ``asyncio.wait`` rather than ``wait_for``, and the loser is cancelled but
    never awaited: ``wait_for`` cancels and then AWAITS the cancellation, so a
    dwell that does not cooperate defeats the bound it was given.
    """
    call: asyncio.Future[Any] = asyncio.ensure_future(
        invoke("jobs", {"action": "wait", "job_id": job_id, "timeout_s": leg}, state)
    )
    racers: set[asyncio.Future[Any]] = {call}
    if interrupt is not None:
        racers.add(asyncio.ensure_future(interrupt.requested.wait()))
    done, pending = await asyncio.wait(
        racers, timeout=leg + _LEG_GRACE_S, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
        _abandoned_legs.add(task)
        task.add_done_callback(_abandoned_legs.discard)
    if call not in done:
        return None
    return call.result()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


_TABLE_PATH_LIMIT = 56
_FACT_CHANNELS = ("error", "failures", "observations", "warnings", "hint")
_PATH_KEYS = {"circuit", "log", "path", "raw", "staged_deck"}
_ATTRIBUTION_KEYS = (
    "source",
    "case_id",
    "run_index",
    "step_index",
    "step_values",
    "assignments",
    "circuit",
    "deck_sha256",
)


def _truncate_text(value: str) -> str:
    if len(value) <= _TABLE_PATH_LIMIT:
        return value
    left = (_TABLE_PATH_LIMIT - 3) // 2
    right = _TABLE_PATH_LIMIT - 3 - left
    return f"{value[:left]}...{value[-right:]}"


def _compact_nested(value: Any, key: str | None = None) -> Any:
    """Copy nested display data, shortening path leaves without touching input."""
    if isinstance(value, dict):
        return {name: _compact_nested(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_compact_nested(item, key) for item in value]
    if isinstance(value, str) and (key in _PATH_KEYS or (key or "").endswith("_path")):
        return _truncate_text(value)
    return value


def _compact_cell(value: Any) -> str:
    """Render one cell on one line, keeping nested values deterministic."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value.replace("\r", "\\r").replace("\n", "\\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(
        _compact_nested(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _middle_truncate(value: Any) -> str:
    """Keep both the directory root and basename visible for long paths."""
    return _truncate_text(_compact_cell(value))


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    output = [line(headers), "-+-".join("-" * width for width in widths)]
    output.extend(line(row) for row in rows)
    if not rows:
        output.append("(no rows)")
    return output


def _expanded_rows(container: dict[str, Any], key: str) -> list[dict[str, Any]] | None:
    """Read object or budget-columnar rows without changing the response."""
    from ltspice_mcp.lib import response_budget

    raw_rows = container.get(key, [])
    if not isinstance(raw_rows, list):
        return None
    columns = container.get(response_budget.columnar_key(key))
    if columns is None:
        if not all(isinstance(row, dict) for row in raw_rows):
            return None
        return [dict(row) for row in raw_rows]
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        return None
    if not all(isinstance(row, list) and len(row) == len(columns) for row in raw_rows):
        return None
    return [dict(zip(columns, row, strict=True)) for row in raw_rows]


def _expanded_page(page: Any) -> dict[str, Any] | None:
    from ltspice_mcp.lib import response_budget

    if not isinstance(page, dict):
        return None
    rows = _expanded_rows(page, "items")
    if rows is None:
        return None
    expanded = dict(page)
    expanded["items"] = rows
    expanded.pop(response_budget.columnar_key("items"), None)
    return expanded


def _has_footer_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _fact_rows(
    sources: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, Any]]:
    facts: list[tuple[str, Any]] = []
    for prefix, source in sources:
        for channel in _FACT_CHANNELS:
            value = source.get(channel)
            if _has_footer_value(value):
                label = f"{prefix}.{channel}" if prefix else channel
                facts.append((label, value))
    return facts


def _finish_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    metadata: list[str],
    facts: list[tuple[str, Any]],
) -> str:
    output = _format_table(headers, rows)
    if metadata:
        output.append("")
        output.extend(metadata)
    if facts:
        output.extend(("", "Facts:"))
        output.extend(f"  {label}: {_compact_cell(value)}" for label, value in facts)
    return "\n".join(output)


def _attribution(row: dict[str, Any]) -> str:
    attributed: dict[str, Any] = {}
    for key in _ATTRIBUTION_KEYS:
        value = row.get(key)
        if value is None or value == {} or value == []:
            continue
        attributed[key] = _middle_truncate(value) if key == "circuit" else value
    return _compact_cell(attributed) if attributed else "-"


def _analysis_stat(row: dict[str, Any], default: str) -> str:
    stat = _compact_cell(row.get("stat", default))
    field = row.get("field")
    return f"{_compact_cell(field)} / {stat}" if field is not None else stat


def _analysis_table(
    data: dict[str, Any],
    *,
    fact_sources: list[tuple[str, dict[str, Any]]],
    receipt: dict[str, Any] | None = None,
) -> str | None:
    results = data.get("results")
    if not isinstance(results, dict):
        return None

    prepared: list[dict[str, Any]] = []
    for recipe_key, entry in results.items():
        if not isinstance(entry, dict):
            return None
        reduced = _expanded_rows(entry, "reduced")
        values = _expanded_rows(entry, "values")
        groups = entry.get("groups", [])
        if (
            reduced is None
            or values is None
            or not isinstance(groups, list)
            or not all(isinstance(group, dict) for group in groups)
        ):
            return None
        group_views: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for group in groups:
            group_rows = _expanded_rows(group, "reduced")
            if group_rows is None:
                return None
            group_views.append((group, group_rows))

        per_run = entry.get("per_run")
        per_run_page = _expanded_page(per_run) if per_run is not None else None
        if per_run is not None and per_run_page is None:
            return None

        spec = entry.get("spec")
        displayed_spec: dict[str, Any] | None = None
        if spec is not None:
            if not isinstance(spec, dict):
                return None
            displayed_spec = dict(spec)
            if "fail_cases" in spec:
                fail_cases = _expanded_page(spec["fail_cases"])
                if fail_cases is None:
                    return None
                displayed_spec["fail_cases"] = fail_cases

        prepared.append(
            {
                "recipe": str(recipe_key),
                "entry": entry,
                "reduced": reduced,
                "values": values,
                "groups": group_views,
                "per_run": per_run_page,
                "spec": displayed_spec,
            }
        )

    displayed_coverage: dict[str, Any] | None = None
    coverage = data.get("coverage")
    if isinstance(coverage, dict):
        displayed_coverage = dict(coverage)
        if "missing_cases" in coverage:
            missing = _expanded_page(coverage["missing_cases"])
            if missing is None:
                return None
            displayed_coverage["missing_cases"] = missing

    rows: list[tuple[str, ...]] = []
    facts = _fact_rows(fact_sources)

    def emit(recipe: str, group: str, stat: str, row: dict[str, Any]) -> None:
        rows.append(
            (
                recipe,
                group,
                stat,
                _compact_cell(row.get("value")),
                _attribution(row),
            )
        )

    for view in prepared:
        recipe = view["recipe"]
        entry = view["entry"]
        before = len(rows)
        for reduced_row in view["reduced"]:
            emit(recipe, "-", _analysis_stat(reduced_row, "reduced"), reduced_row)

        for group, group_rows in view["groups"]:
            group_cell = _compact_cell(group.get("by", {}))
            for reduced_row in group_rows:
                emit(recipe, group_cell, _analysis_stat(reduced_row, "reduced"), reduced_row)
            if "count" in group:
                rows.append((recipe, group_cell, "count", _compact_cell(group["count"]), "-"))

        for value_row in view["values"]:
            emit(recipe, "-", "value", value_row)

        if view["per_run"] is not None:
            for value_row in view["per_run"]["items"]:
                emit(recipe, "-", "per run", value_row)

        if view["spec"] is not None:
            rows.append((recipe, "-", "spec", _compact_cell(view["spec"]), "-"))

        if entry.get("steps"):
            rows.append((recipe, "-", "steps", _compact_cell(entry["steps"]), "-"))
        if len(rows) == before:
            rows.append((recipe, "-", "metric", _compact_cell(entry.get("metric")), "-"))
        if _has_footer_value(entry.get("warnings")):
            facts.append((f"results.{recipe}.warnings", entry["warnings"]))

    metadata: list[str] = []
    if displayed_coverage is not None:
        metadata.append(f"Coverage: {_compact_cell(displayed_coverage)}")
    if data.get("result_set_id"):
        metadata.append(f"Result set: {_compact_cell(data['result_set_id'])}")
    if data.get("next"):
        metadata.append(f"Next: {_compact_cell(data['next'])}")
    if receipt is not None:
        receipt_fields = {
            key: receipt.get(key)
            for key in ("job_id", "request_id", "status", "outcome")
            if key in receipt
        }
        if receipt_fields:
            metadata.insert(0, f"Receipt: {_compact_cell(receipt_fields)}")
    return _finish_table(
        ("recipe key", "group", "stat", "value", "attribution"),
        rows,
        metadata=metadata,
        facts=facts,
    )


def _jobs_table(data: dict[str, Any]) -> str | None:
    action = data.get("action")
    if action not in {"list", "runs"}:
        return None
    items = _expanded_rows(data, "items")
    if items is None:
        return None

    if action == "list":
        headers = ("path", "exists", "last_activity", "status_counts", "interrupted_job_ids")
        rows = [
            (
                _middle_truncate(row.get("path")),
                _compact_cell(row.get("exists")),
                _compact_cell(row.get("last_activity")),
                _compact_cell(row.get("status_counts")),
                _compact_cell(row.get("interrupted_job_ids")),
            )
            for row in items
        ]
    else:
        from ltspice_mcp.tools.experiments import _RUN_RECORD_SCHEMA

        headers = tuple(_RUN_RECORD_SCHEMA["properties"])

        def run_cell(row: dict[str, Any], key: str) -> str:
            value = row.get(key)
            return (
                _middle_truncate(value)
                if key in {"circuit", "raw", "log"}
                else _compact_cell(value)
            )

        rows = [tuple(run_cell(row, key) for key in headers) for row in items]

    page = {
        key: data.get(key)
        for key in ("returned", "total", "truncated", "next_cursor")
        if key in data
    }
    metadata = [f"Page: {_compact_cell(page)}"] if page else []
    if action == "runs":
        job = {
            key: data.get(key)
            for key in ("job_id", "request_id", "status", "dialect")
            if key in data
        }
        if job:
            metadata.insert(0, f"Job: {_compact_cell(job)}")
    return _finish_table(
        headers,
        rows,
        metadata=metadata,
        facts=_fact_rows([("", data)]),
    )


def _table_view(command: str, data: dict[str, Any]) -> str | None:
    """Select one explicitly supported envelope; all other shapes fall back."""
    if command == "analyze-results":
        return _analysis_table(data, fact_sources=[("", data)])
    if command in {"run", "run-experiments"}:
        analysis = data.get("analysis")
        if not isinstance(analysis, dict) or not isinstance(analysis.get("result"), dict):
            return None
        result = analysis["result"]
        return _analysis_table(
            result,
            fact_sources=[
                ("receipt", data),
                ("analysis", analysis),
                ("analysis.result", result),
            ],
            receipt=data,
        )
    if command == "jobs":
        return _jobs_table(data)
    return None


def exit_code_for(result: types.CallToolResult) -> int:
    """Classify a handler result into an exit code.

    The discriminator is the envelope, not a per-tool error string: an ``error``
    block reports whether anything was committed, and ``outcome`` reports how
    much of what was asked for came back.
    """
    data = result.structuredContent or {}
    if data.get(CANCEL_CONFIRMED_KEY) is False:
        # Ahead of everything else: whatever the envelope says about the job, it
        # was written by a read that could not see the job reach a terminal
        # state, and reporting that reading as if it settled the matter is the
        # one thing this code must not do.
        return EXIT_UNCONFIRMED
    error = data.get("error")
    if isinstance(error, dict):
        return (
            EXIT_FAILED if error.get("commit_state") in ("committed", "unknown") else EXIT_REFUSED
        )
    if result.isError:
        return EXIT_REFUSED
    return _EXIT_BY_OUTCOME.get(data.get("outcome", ""), EXIT_INTERNAL)


def emit(namespace: argparse.Namespace, result: types.CallToolResult, code: int) -> None:
    """Write the result.

    ``--json`` prints the handler's structuredContent unchanged on one line —
    that is the parse-stable contract. Human mode prints the handler's text
    summary, then the same structuredContent pretty-printed or, for the
    explicitly supported envelopes, rendered by ``--table``. The table branch
    reads but never rewrites the payload. Human mode is presentation,
    explicitly not parse-stable.
    """
    from ltspice_mcp.tools._base import result_text

    data = result.structuredContent
    if (data or {}).get(CANCEL_CONFIRMED_KEY) is False:
        # On stderr in both modes: the payload marker is for a script, this line
        # is for whoever is watching, and neither may be the only one to say it.
        print(
            "spice-mcp: "
            + _UNCONFIRMED_CANCEL_MESSAGE.format(
                join=_CANCEL_JOIN_S, job_id=(data or {}).get("job_id")
            ),
            file=sys.stderr,
        )
    if code == EXIT_REFUSED:
        # A refusal the handler reported as a structured envelope carries no
        # CLI recovery line of its own, and the stdout payload must stay
        # exactly the handler's; stderr is where this front end speaks.
        print(f"spice-mcp: try: {_exemplar_for(namespace.command)}", file=sys.stderr)
    if namespace.as_json:
        sys.stdout.write(json.dumps(data if data is not None else {}, ensure_ascii=False) + "\n")
        return
    text = result_text(result)
    if text:
        print(text)
    if data is not None:
        table = _table_view(namespace.command, data) if namespace.as_table else None
        if namespace.as_table and table is None:
            print("Table view is unavailable for this response; showing JSON.")
        print(table if table is not None else json.dumps(data, ensure_ascii=False, indent=1))
    hint = (data or {}).get("hint")
    if code != EXIT_OK and hint and hint not in text:
        print(hint, file=sys.stderr)


def _emit_error_json(code: str, message: str) -> None:
    """The parse-stable error envelope, shared by handler-level refusals and
    parser-level usage errors so the shape cannot fork between the two."""
    payload = {"error": {"code": code, "message": message, "stage": "cli"}}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def emit_error(namespace: argparse.Namespace, code: str, message: str, exit_code: int) -> int:
    """Report a failure the CLI itself classified, and return its exit code.

    A refusal appends the subcommand's minimal valid invocation: nothing was
    committed, so the next call is the whole remedy, and a rejection without an
    example of a valid call is the measured walk-away.
    """
    if exit_code == EXIT_REFUSED:
        exemplar = _exemplar_for(getattr(namespace, "command", "") or "")
        if exemplar not in message:
            message = f"{message}\ntry: {exemplar}"
    if namespace.as_json:
        _emit_error_json(code, message)
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

    try:
        with prepared_environment(namespace):
            async with session() as state:
                interrupt = _install_interrupt_handler()
                try:
                    result = await run_command(namespace, payload, state, interrupt)
                except _Refused as exc:
                    return emit_error(namespace, "refused", str(exc), EXIT_REFUSED)
                except _Failed as exc:
                    return emit_error(namespace, "failed", str(exc), EXIT_FAILED)
                finally:
                    _remove_interrupt_handler()
                    await _drain_background_writes()
                code = exit_code_for(result)
                emit(namespace, result, code)
                if code == EXIT_UNCONFIRMED:
                    # An unconfirmed kill outranks the interrupt that asked for
                    # it: 130 promises the job was cancelled, which is exactly
                    # the claim this process cannot make.
                    return code
                if not interrupt.cancelled:
                    if interrupt.requested.is_set():
                        print(
                            "spice-mcp: the interrupt arrived too late to change the "
                            "outcome; nothing was cancelled and the result above is "
                            "what the work produced.",
                            file=sys.stderr,
                        )
                    return code
                return EXIT_INTERRUPTED
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
