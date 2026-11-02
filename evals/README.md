# Agent evaluation harness

Runs a model against the server on tasks with known answers, grades what the
agent leaves in its workspace, and compares server variants on the same tasks,
paired. The design, and the reasons behind it, are in
[`docs/design/agent_eval.md`](../docs/design/agent_eval.md).

Paid runs are opt-in, never part of CI. Everything below except a run with a
real model works offline.

## Commands

Run from this directory, so the `ltspice_eval` package is importable:

```bash
cd evals
uv run python -m ltspice_eval list
uv run python -m ltspice_eval run --suite smoke --model oracle --reps 1
uv run python -m ltspice_eval summary --suite smoke
```

`--model oracle` plays each task's `oracle.py` through the real loop and a real
server process, and `--model null` answers every task with a constant; neither
calls the API. The oracle passes every task and the null answer fails every
task; `tests/test_eval_harness.py` checks both.

A paid run names a model and a spend cap:

```bash
uv run python -m ltspice_eval run --suite pilot --model claude-opus-5-5 --effort high \
    --tasks analyze-step-ac-f3db --reps 1 --max-usd 5
```

`--max-usd` is a safety net, not a budget. When spend reaches it, no new trial
starts and running trials finish, so it never changes an outcome. Run the same
command again to resume: a (task, variant, rep) that has a row is skipped.

## Ceilings

A trial has no budget, only ceilings (`--turn-ceiling`, `--usd-ceiling`,
`--wall-ceiling`) far above normal use. They stop a runaway and nothing else:
the model never sees them. A trial that reaches one is `capped` and counts as
a fail. If capped trials could change a decision, re-run the whole comparison
with higher ceilings. The design's "Ceilings, not budgets" section has the rule.

## Layout

```
launch_server.py           starts a variant's server with a scrubbed environment
ltspice_eval/              the runner, the API loop, rendering, graders, rows
tasks/<id>/task.toml       prompt, category, requirements, grader
tasks/<id>/workspace/      files copied into each trial's workspace
tasks/<id>/grader/         reference.json and make_reference.py; never copied
tasks/<id>/oracle.py       a scripted solution through the tools
variants/<name>.toml       a checkout and a configuration overlay
runs/<suite>/              results.jsonl, errors.jsonl, traces/ (not committed)
```

## Adding a task

1. Put the inputs in `tasks/<id>/workspace/` and write `task.toml`. A prompt
   says what file to write and in what shape. A task whose right answer is
   that the thing cannot be done gives the form for saying so (usually
   `null`).
2. Write `grader/make_reference.py`, whose `compute(workspace)` works the
   answer out with spicelib and numpy alone (`ltspice_eval.references` has
   helpers). Never use the server's code: a variant must not be able to move
   its own grade. Then write the reference:
   `uv run python -m ltspice_eval.references <id>`.
3. Write `oracle.py`, a script that solves the task through the tools.
4. Run `uv run pytest tests/test_eval_harness.py`.
