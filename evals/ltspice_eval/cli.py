"""``python -m ltspice_eval``: run a suite, list what can run, or summarize.

Run it from ``evals/`` so the package is importable::

    cd evals
    uv run python -m ltspice_eval list
    uv run python -m ltspice_eval run --suite smoke --model oracle
    uv run python -m ltspice_eval run --suite pilot --model claude-opus-5-5 --effort high \\
        --tasks analyze-step-ac-f3db --reps 1 --max-usd 5
    uv run python -m ltspice_eval summary --suite pilot

``--model oracle`` and ``--model null`` play each task's oracle script, or a
constant answer, through the real loop and server with no API calls. Any
other model is a paid run, and needs ``--max-usd``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from statistics import mean
from typing import Any

from ltspice_eval.loop import Ceilings, Settings
from ltspice_eval.models import PRICES, AnthropicModel, Model
from ltspice_eval.rendering import RENDERINGS
from ltspice_eval.rows import RunFiles, read_rows
from ltspice_eval.runner import RunConfig, run_suite
from ltspice_eval.scripted import (
    SCRIPTED_MODEL,
    SCRIPTED_PRICE,
    ScriptContext,
    ScriptedModel,
    load_script,
    null_script,
)
from ltspice_eval.tasks import Task, load_tasks
from ltspice_eval.variants import load_variant
from ltspice_eval.workspace import TrialDirs

SCRIPTED = {"oracle", "null"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ltspice_eval", description="Evaluate the server with an agent."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a suite, or resume one")
    run.add_argument("--suite", required=True, help="the run's name, under evals/runs/")
    run.add_argument("--tasks", help="comma-separated task ids; default every task")
    run.add_argument("--variants", default="baseline", help="comma-separated variant names")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--model", required=True, help="a model id, or 'oracle' or 'null'")
    run.add_argument("--effort", help="low, medium, high, xhigh or max; default the model's")
    run.add_argument("--rendering", default="structured", choices=sorted(RENDERINGS))
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument(
        "--max-usd", type=float, help="stop starting trials once the run has spent this"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--turn-ceiling", type=int, default=Ceilings.turns)
    run.add_argument("--usd-ceiling", type=float, default=Ceilings.usd)
    run.add_argument("--wall-ceiling", type=float, default=Ceilings.wall_clock_s, help="seconds")
    run.add_argument("--keep-workspaces", action="store_true")
    run.add_argument(
        "--allow-run-code",
        action="store_true",
        help="let a variant turn run_code on; only inside a container",
    )

    commands.add_parser("list", help="list tasks and variants")

    summary = commands.add_parser("summary", help="a per-variant table of a suite's rows")
    summary.add_argument("--suite", required=True)

    args = parser.parse_args(argv)
    if args.command == "list":
        return _list()
    if args.command == "summary":
        return _summary(args.suite)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(ids=args.tasks.split(",") if args.tasks else None)
    variants = [load_variant(name) for name in args.variants.split(",")]
    if not args.allow_run_code and (running := [v.name for v in variants if v.runs_code]):
        print(f"{', '.join(running)} turn run_code on; pass --allow-run-code inside a container")
        return 2
    ceilings = Ceilings(
        turns=args.turn_ceiling, usd=args.usd_ceiling, wall_clock_s=args.wall_ceiling
    )
    if args.model in SCRIPTED:
        settings = Settings(model=SCRIPTED_MODEL, price=SCRIPTED_PRICE, ceilings=ceilings)
        model_for = _scripted(args.model)
    else:
        if args.max_usd is None:
            print("a paid run needs --max-usd, its safety net")
            return 2
        if args.model not in PRICES:
            print(f"no price for {args.model}; add it to models.PRICES")
            return 2
        settings = Settings(
            model=args.model,
            price=PRICES[args.model],
            effort=args.effort,
            rendering=args.rendering,
            ceilings=ceilings,
        )
        shared = AnthropicModel()

        def model_for(task: Task, dirs: TrialDirs) -> Model:
            return shared

    config = RunConfig(
        suite=args.suite,
        tasks=tasks,
        variants=variants,
        reps=args.reps,
        settings=settings,
        model_for=model_for,
        seed=args.seed,
        concurrency=args.concurrency,
        max_usd=args.max_usd,
        keep_workspaces=args.keep_workspaces,
    )
    files = RunFiles(args.suite)
    result = asyncio.run(run_suite(config, files))
    print(
        f"{result.ran} trials ran ({result.errors} errors), {result.skipped_done} already done, "
        f"{result.not_started} not started; ${result.spent_usd:.2f} spent. Rows: {files.results}"
    )
    return 0


def _scripted(which: str):
    def model_for(task: Task, dirs: TrialDirs) -> Model:
        script = load_script(task.oracle) if which == "oracle" else null_script
        return ScriptedModel(
            script=script,
            context=ScriptContext(task_id=task.id, workspace=dirs.workspace, home=dirs.home),
        )

    return model_for


def _list() -> int:
    from ltspice_eval import VARIANTS_DIR

    for task in load_tasks():
        needs = f" (needs {', '.join(task.requires)})" if task.requires else ""
        print(f"task     {task.id:32s} {task.category}{needs}")
    for path in sorted(VARIANTS_DIR.glob("*.toml")):
        variant = load_variant(path.stem)
        print(f"variant  {variant.name:32s} {variant.description}")
    return 0


def _summary(suite: str) -> int:
    """Pass rate and the size measures per variant. Intervals by bootstrap
    over tasks are the comparison report's job, not this table's."""
    files = RunFiles(suite)
    rows = read_rows(files.results)
    errors = read_rows(files.errors)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    print(
        f"{'variant':20s} {'trials':>6s} {'pass':>6s} {'capped':>6s} {'input tok':>10s} {'output tok':>10s} {'calls':>6s} {'tool err':>8s} {'usd':>8s}"
    )
    for name, group in sorted(by_variant.items()):
        print(
            f"{name:20s} {len(group):6d} {mean(r['pass'] for r in group):6.2f} "
            f"{sum(r['status'] == 'capped' for r in group):6d} "
            f"{mean(r['usage']['total_input'] for r in group):10.0f} "
            f"{mean(r['usage']['output'] for r in group):10.0f} "
            f"{mean(r['tool_calls']['total'] for r in group):6.1f} "
            f"{mean(r['tool_errors']['total'] for r in group):8.2f} "
            f"{sum(r['usd'] for r in group):8.2f}"
        )
    if errors:
        print(f"{len(errors)} trials ended in errors: see {files.errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
