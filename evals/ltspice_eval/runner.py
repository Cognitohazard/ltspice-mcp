"""Running a suite: every (task, variant, rep) once, in a shuffled order.

Trials of all variants are interleaved in one random order under one
concurrency limit, so rate limits, cache warmth and the time of day fall on
every variant alike. The run's spend cap is a safety net, not a plan: when
spend reaches it, no new trial starts, running trials finish under their own
ceilings, and the run can be resumed later. It never cuts a trial short, so
it changes no outcome.
"""

from __future__ import annotations

import asyncio
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ltspice_eval import REPO_ROOT, ROW_SCHEMA
from ltspice_eval.grading import Grade, grade
from ltspice_eval.loop import (
    LoopOutcome,
    Progress,
    Settings,
    TrialError,
    connect,
    error_counts,
    run_loop,
    tool_counts,
)
from ltspice_eval.models import Model, ModelError
from ltspice_eval.rows import RunFiles
from ltspice_eval.tasks import Task
from ltspice_eval.variants import Variant, revision, write_trial_config
from ltspice_eval.workspace import (
    FileTools,
    TrialDirs,
    changed_inputs,
    make_trial_dirs,
    remove_trial_dirs,
    snapshot,
    sweep,
)

#: How much of the server's stderr an error row keeps.
STDERR_TAIL_CHARS = 4000

ModelFactory = Callable[[Task, TrialDirs], Model]


@dataclass(frozen=True)
class RunConfig:
    suite: str
    tasks: list[Task]
    variants: list[Variant]
    reps: int
    settings: Settings
    model_for: ModelFactory
    seed: int = 0
    concurrency: int = 4
    #: Stop starting trials once the run has spent this many dollars.
    max_usd: float | None = None
    keep_workspaces: bool = False
    profile: str = "api"

    def parameters(self) -> dict[str, Any]:
        """What a resumed run must repeat to be the same comparison."""
        return {
            "tasks": [task.id for task in self.tasks],
            "variants": [variant.name for variant in self.variants],
            "reps": self.reps,
            "profile": self.profile,
            "model": self.settings.model,
            "effort": self.settings.effort,
            "rendering": self.settings.rendering,
            "max_tokens": self.settings.max_tokens,
            "ceilings": self.settings.ceilings.as_row(),
            "price": asdict(self.settings.price),
            "seed": self.seed,
        }


@dataclass
class TrialResult:
    row: dict[str, Any] | None
    error: dict[str, Any] | None
    usd: float


@dataclass
class SuiteResult:
    ran: int = 0
    skipped_done: int = 0
    not_started: int = 0
    spent_usd: float = 0.0
    errors: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)


async def run_suite(config: RunConfig, files: RunFiles) -> SuiteResult:
    files.start(config.parameters())
    done = files.done()
    trials = [
        (task, variant, rep)
        for task in config.tasks
        for variant in config.variants
        for rep in range(config.reps)
    ]
    random.Random(config.seed).shuffle(trials)
    revisions = {"harness": revision(REPO_ROOT)} | {
        variant.name: revision(variant.checkout) for variant in config.variants
    }
    result = SuiteResult()
    gate = asyncio.Semaphore(config.concurrency)
    stopped = False

    async def one(order: int, task: Task, variant: Variant, rep: int) -> None:
        nonlocal stopped
        async with gate:
            if stopped:
                result.not_started += 1
                return
            outcome = await run_trial(task, variant, rep, order, config, files, revisions)
            result.ran += 1
            result.spent_usd += outcome.usd
            if outcome.row is not None:
                result.rows.append(outcome.row)
            else:
                result.errors += 1
            if config.max_usd is not None and result.spent_usd >= config.max_usd:
                stopped = True

    async with asyncio.TaskGroup() as group:
        for order, (task, variant, rep) in enumerate(trials):
            if (task.id, variant.name, rep) in done:
                result.skipped_done += 1
                continue
            group.create_task(one(order, task, variant, rep))
    return result


async def run_trial(
    task: Task,
    variant: Variant,
    rep: int,
    order: int,
    config: RunConfig,
    files: RunFiles,
    revisions: dict[str, Any],
) -> TrialResult:
    settings = config.settings
    dirs = make_trial_dirs(task)
    write_trial_config(variant, dirs.workspace)
    before = snapshot(dirs.workspace)
    progress = Progress()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    stderr_path = dirs.root / "server.stderr"
    outcome: LoopOutcome | None = None
    server_info: dict[str, Any] = {}
    failure: tuple[str, str, str] | None = None
    try:
        model = config.model_for(task, dirs)
        with stderr_path.open("w", encoding="utf-8", errors="replace") as errlog:
            async with connect(variant, dirs, errlog) as server:
                server_info = server.server_info
                outcome = await run_loop(
                    model, server, FileTools(dirs.workspace), task.prompt, settings, progress
                )
    except Exception as exc:
        failure = _classified(exc)
    leftovers = await asyncio.to_thread(sweep, dirs.home)
    wall_s = time.monotonic() - started
    stderr_tail = _tail(stderr_path)
    usd = progress.usage.usd(settings.price)
    base = {
        "schema": ROW_SCHEMA,
        "suite": config.suite,
        "task": task.id,
        "category": task.category,
        "variant": variant.name,
        "rep": rep,
        "order": order,
        "seed": config.seed,
        "profile": config.profile,
        "model": settings.model,
        "effort": settings.effort,
        "rendering": settings.rendering,
        "ceilings": settings.ceilings.as_row(),
        "started_at": started_at,
        "revision": {"server": revisions.get(variant.name), "harness": revisions.get("harness")},
        "server": server_info,
        "leftover_processes": leftovers,
    }
    trace: dict[str, Any] = {
        "task": task.id,
        "variant": variant.name,
        "rep": rep,
        "prompt": task.prompt,
        "system": progress.system,
        "tools": [tool["name"] for tool in progress.tools],
        "messages": progress.messages,
        "calls": [asdict(call) for call in progress.calls],
    }
    try:
        if failure is not None or outcome is None:
            kind, message, stack = failure or ("harness_error", "the loop returned nothing", "")
            error = base | {
                "error": kind,
                "message": message,
                "traceback": stack,
                "turns": progress.turns,
                "usd": round(usd, 6),
                "usage": progress.usage.as_row(),
                "wall_s": round(wall_s, 3),
                "server_stderr_tail": stderr_tail,
            }
            files.append_error(error)
            files.write_trace(variant.name, task.id, rep, trace | {"error": error})
            return TrialResult(row=None, error=error, usd=usd)
        verdict = await asyncio.to_thread(grade, task, dirs.workspace)
        after = await asyncio.to_thread(snapshot, dirs.workspace)
        row = (
            base
            | _outcome_row(outcome, verdict, wall_s)
            | {
                "guardrails": {"read_only_changed": changed_inputs(before, after, task.editable)},
            }
        )
        files.append_result(row)
        files.write_trace(
            variant.name,
            task.id,
            rep,
            trace | {"answer": verdict.answer, "grade": verdict.as_row()},
        )
        return TrialResult(row=row, error=None, usd=usd)
    finally:
        if config.keep_workspaces:
            print(f"kept {dirs.root}")
        else:
            left = remove_trial_dirs(dirs)
            if left:
                print(f"could not remove {dirs.root}: {left}")


def _outcome_row(outcome: LoopOutcome, verdict: Grade, wall_s: float) -> dict[str, Any]:
    """The graded part of a row. A capped trial is a fail whatever it wrote;
    ``passed_at_cap`` keeps what the grader said of it, for the report's
    reading with every capped trial counted as a pass."""
    if outcome.status == "capped":
        status, passed = "capped", False
    elif outcome.status == "refused":
        status, passed = "refused", False
    elif verdict.status == "unparseable":
        status, passed = "unparseable", False
    else:
        status, passed = "ok", verdict.passed
    calls = outcome.calls
    tokens = [call.tokens for call in calls]
    return {
        "status": status,
        "pass": passed,
        "capped": outcome.capped,
        "passed_at_cap": verdict.passed if outcome.status == "capped" else None,
        "grade": verdict.as_row(),
        "turns": outcome.turns,
        "stop_reason": outcome.stop_reason,
        "max_tokens_stops": outcome.max_tokens_stops,
        "served_models": outcome.served_models,
        "tool_calls": tool_counts(calls),
        "tool_errors": error_counts(calls),
        "lookups": {
            "reference": sum(
                call.name == "inspect" and call.kind == "reference" for call in calls
            ),
            "guide": sum(call.name == "inspect" and call.kind == "guide" for call in calls),
        },
        "result_tokens": {
            "total": sum(tokens),
            "largest": max(tokens, default=0),
            "estimated": True,
        },
        "oversized_results": sum(call.oversized for call in calls),
        "usage": outcome.usage.as_row(),
        "usd": round(outcome.usd, 6),
        "latency": {"model_s": round(outcome.model_s, 3), "wall_s": round(wall_s, 3)},
        "retries": outcome.retries,
    }


def _classified(exc: Exception) -> tuple[str, str, str]:
    """An error row's kind, message and traceback. The MCP client's task
    groups wrap what the loop raised in an ``ExceptionGroup``, so the cause is
    looked for among its leaves."""
    stack = traceback.format_exception(exc)
    for leaf in _leaves(exc):
        if isinstance(leaf, TrialError):
            return leaf.kind, str(leaf), "".join(stack)
        if isinstance(leaf, ModelError):
            return "api_error", str(leaf), "".join(stack)
    return "harness_error", f"{type(exc).__name__}: {exc}", "".join(stack)


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _leaves(inner)]
    return [exc]


def _tail(path: Any) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-STDERR_TAIL_CHARS:]
