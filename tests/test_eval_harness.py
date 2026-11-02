"""The agent evaluation harness (``evals/``), tested with no API calls.

A scripted model plays each task's oracle through the real loop and a real
server process, so these tests cover the launcher, the MCP proxy, rendering,
grading and the rows; only the model is replaced. ``docs/design/agent_eval.md``
is the design.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import psutil
import pytest
from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS

import launch_server
from ltspice_eval import TASKS_DIR, jsonjs, references
from ltspice_eval.grading import as_number, compare
from ltspice_eval.loop import Ceilings, Settings, connect
from ltspice_eval.models import ModelReply, Usage
from ltspice_eval.rendering import OVERSIZED_TOKENS, render
from ltspice_eval.rows import RunFiles, RunMismatch, read_rows
from ltspice_eval.runner import RunConfig, run_suite
from ltspice_eval.scripted import (
    SCRIPTED_MODEL,
    SCRIPTED_PRICE,
    Call,
    ScriptContext,
    ScriptedModel,
    Seen,
    Step,
    load_script,
    null_script,
)
from ltspice_eval.tasks import Task, Tolerance, load_task, load_tasks
from ltspice_eval.variants import load_variant, write_trial_config
from ltspice_eval.workspace import FileToolError, FileTools, TrialDirs, make_trial_dirs, sweep
from tests.conftest import LIVENESS_S

TASKS = load_tasks()
STEP_AC = load_task(TASKS_DIR / "analyze-step-ac-f3db")
#: Ceilings for these tests: far above what a scripted trial uses, and a wall
#: clock that caps a hang.
CEILINGS = Ceilings(turns=50, usd=1000.0, wall_clock_s=LIVENESS_S * 4)

Script = Callable[[ScriptContext], Generator[Step, list[Seen], None]]


def _config(
    tmp_path: Path,
    tasks: list[Task],
    script_for: Callable[[Task], Script],
    *,
    ceilings: Ceilings = CEILINGS,
    reps: int = 1,
    max_usd: float | None = None,
    concurrency: int = 2,
    served_model: str = SCRIPTED_MODEL,
    models: list[ScriptedModel] | None = None,
    suite: str = "test",
) -> tuple[RunConfig, RunFiles]:
    def model_for(task: Task, dirs: TrialDirs) -> ScriptedModel:
        model = ScriptedModel(
            script=script_for(task),
            context=ScriptContext(task_id=task.id, workspace=dirs.workspace, home=dirs.home),
            served_model=served_model,
        )
        if models is not None:
            models.append(model)
        return model

    config = RunConfig(
        suite=suite,
        tasks=tasks,
        variants=[load_variant("baseline")],
        reps=reps,
        settings=Settings(model=SCRIPTED_MODEL, price=SCRIPTED_PRICE, ceilings=ceilings),
        model_for=model_for,
        concurrency=concurrency,
        max_usd=max_usd,
    )
    return config, RunFiles(suite, root=tmp_path)


def _oracle(task: Task) -> Script:
    return load_script(task.oracle)


# ---------------------------------------------------------------------------
# Tasks and their references


def test_every_task_has_an_oracle_and_a_reference():
    assert TASKS, "no tasks under evals/tasks"
    for task in TASKS:
        assert task.oracle.is_file(), f"{task.id} has no oracle.py"
        assert task.grader.reference.is_file(), f"{task.id} has no reference"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
def test_a_reference_is_what_its_script_computes(task: Task):
    """References come from spicelib alone; the committed file must be what
    the task's script computes from the recorded run, so neither drifts."""
    expected = json.loads(task.grader.reference.read_text(encoding="utf-8"))
    assert references.compute(task.directory) == expected


def test_the_reference_axis_reads_a_fresh_raw():
    """``references.axis`` works on a raw no trace has been read from, where
    spicelib's own ``get_axis`` does not (``docs/spicelib_bugs.md``, Bug 35)."""
    raw = references.open_raw(STEP_AC.workspace / "step_ac.raw")
    frequency = references.axis(raw, 0)
    assert len(frequency) == 81
    assert frequency[0] == pytest.approx(10.0)


def test_a_task_that_sets_its_own_limits_is_refused(tmp_path: Path):
    directory = tmp_path / "limited"
    (directory / "grader").mkdir(parents=True)
    (directory / "task.toml").write_text(
        'id = "limited"\ncategory = "analyze"\nprompt = "x"\n'
        '[grader]\nkind = "answer"\n[limits]\nmax_turns = 30\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sets no limits"):
        load_task(directory)


# ---------------------------------------------------------------------------
# The oracle passes every task, a constant answer fails every task


async def test_the_oracle_passes_every_task(tmp_path: Path):
    config, files = _config(tmp_path, TASKS, _oracle, concurrency=4)
    result = await run_suite(config, files)
    rows = {row["task"]: row for row in read_rows(files.results)}
    assert not read_rows(files.errors), read_rows(files.errors)
    assert result.ran == len(TASKS)
    failed = {task: row["grade"] for task, row in rows.items() if not row["pass"]}
    assert not failed
    for row in rows.values():
        assert row["status"] == "ok"
        assert row["guardrails"]["read_only_changed"] == []
        assert row["leftover_processes"] == 0
        assert row["server"]["name"] == "ltspice-mcp"


async def test_a_constant_answer_fails_every_task(tmp_path: Path):
    config, files = _config(tmp_path, TASKS, lambda task: null_script, concurrency=4)
    await run_suite(config, files)
    rows = read_rows(files.results)
    assert len(rows) == len(TASKS)
    assert not [row["task"] for row in rows if row["pass"]]


async def test_a_reply_reaches_the_model_as_its_structured_content(tmp_path: Path):
    models: list[ScriptedModel] = []
    config, files = _config(tmp_path, [STEP_AC], _oracle, models=models)
    await run_suite(config, files)
    [model] = models
    tool_result = model.requests[1]["messages"][-1]["content"][0]
    [block] = tool_result["content"]
    assert json.loads(block["text"])["results"]["f"]["metric"] == "bode_filter"
    row = read_rows(files.results)[0]
    assert row["tool_calls"]["by_tool"] == {"analyze_results": 1, "write_file": 1}
    assert row["result_tokens"]["largest"] > 0


async def test_error_codes_inside_a_reply_are_counted(tmp_path: Path):
    task = load_task(TASKS_DIR / "analyze-missing-trace")
    config, files = _config(tmp_path, [task], _oracle)
    await run_suite(config, files)
    [row] = read_rows(files.results)
    assert row["pass"]
    assert row["tool_errors"]["by_code"] == {"recipe_failed": 1}


# ---------------------------------------------------------------------------
# Ceilings


def _forever(ctx: ScriptContext) -> Generator[Step, list[Seen], None]:
    """An agent in a loop: it never stops calling tools."""
    while True:
        yield Call("list_dir", {})


async def test_a_runaway_is_capped_at_the_turn_ceiling(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], lambda task: _forever, ceilings=Ceilings(turns=3))
    await run_suite(config, files)
    [row] = read_rows(files.results)
    assert row["status"] == "capped"
    assert row["capped"] == "turns"
    assert row["turns"] == 3
    assert row["pass"] is False
    assert row["passed_at_cap"] is False


async def test_a_runaway_is_capped_at_the_dollar_ceiling(tmp_path: Path):
    """Each scripted turn costs one dollar, so a 2.5 dollar ceiling stops the
    agent before its fourth turn."""
    config, files = _config(tmp_path, [STEP_AC], lambda task: _forever, ceilings=Ceilings(usd=2.5))
    await run_suite(config, files)
    [row] = read_rows(files.results)
    assert row["capped"] == "usd"
    assert row["turns"] == 3
    assert row["usd"] == pytest.approx(3.0)


class _Silent:
    """A model that never answers."""

    async def create(self, request: dict[str, Any]) -> ModelReply:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_trial_that_hangs_is_capped_at_the_wall_clock(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], _oracle)
    config = RunConfig(
        **{
            **config.__dict__,
            "model_for": lambda task, dirs: _Silent(),
            # timing: the ceiling under test is a wall-clock bound; one second
            # past the server's start is enough to show it stops the trial.
            "settings": Settings(
                model=SCRIPTED_MODEL, price=SCRIPTED_PRICE, ceilings=Ceilings(wall_clock_s=1.0)
            ),
        }
    )
    await run_suite(config, files)
    [row] = read_rows(files.results)
    assert row["capped"] == "wall_clock"
    assert row["status"] == "capped"


def _normalized(request: dict[str, Any], workspace: Path) -> str:
    """A request with what the server returned set aside: what is left is
    what the harness wrote."""
    messages = []
    for message in request["messages"]:
        content = message["content"]
        if isinstance(content, list):
            content = [
                {**block, "content": "<server reply>"}
                if block.get("type") == "tool_result"
                else block
                for block in content
            ]
        messages.append({**message, "content": content})
    text = json.dumps({**request, "messages": messages}, sort_keys=True)
    return text.replace(json.dumps(str(workspace))[1:-1], "<workspace>")


async def test_the_ceilings_do_not_reach_the_model(tmp_path: Path):
    """Two runs with different ceilings send the model the same requests, so a
    trial below its ceilings runs as it would with none."""
    seen = []
    for index, ceilings in enumerate(
        [Ceilings(turns=9137, usd=4321.5, wall_clock_s=2711.5), CEILINGS]
    ):
        models: list[ScriptedModel] = []
        config, files = _config(
            tmp_path, [STEP_AC], _oracle, ceilings=ceilings, models=models, suite=f"c{index}"
        )
        await run_suite(config, files)
        [model] = models
        seen.append([_normalized(request, model.context.workspace) for request in model.requests])
    assert seen[0] == seen[1]
    assert not any(value in text for text in seen[0] for value in ("9137", "4321.5", "2711.5"))


# ---------------------------------------------------------------------------
# The spend cap, resume, and errors


async def test_the_spend_cap_stops_new_trials_and_cuts_none(tmp_path: Path):
    config, files = _config(tmp_path, TASKS, _oracle, max_usd=2.5, concurrency=1)
    result = await run_suite(config, files)
    rows = read_rows(files.results)
    assert result.ran == 1
    assert result.not_started == len(TASKS) - 1
    [row] = rows
    assert row["status"] == "ok"
    assert row["pass"]


async def test_a_run_resumes_where_it_stopped(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], _oracle, reps=2)
    first = await run_suite(config, files)
    assert first.ran == 2
    again = await run_suite(config, files)
    assert (again.ran, again.skipped_done) == (0, 2)
    assert len(read_rows(files.results)) == 2
    changed, _ = _config(tmp_path, [STEP_AC], _oracle, reps=3)
    with pytest.raises(RunMismatch, match="reps"):
        await run_suite(changed, files)


def _kills_its_server(ctx: ScriptContext) -> Generator[Step, list[Seen], None]:
    found = [
        process
        for process in psutil.process_iter(["environ"], ad_value=None)
        if (process.info.get("environ") or {}).get("LTSPICE_MCP_HOME") == str(ctx.home)
    ]
    assert found, "the server's processes carry the trial's home"
    for process in found:
        process.kill()
    psutil.wait_procs(found, timeout=LIVENESS_S)
    yield Call("mcp__spice__inspect", {"queries": [{"kind": "guide"}]})


async def test_a_server_that_dies_is_an_error_not_a_fail(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], lambda task: _kills_its_server)
    result = await run_suite(config, files)
    assert read_rows(files.results) == []
    [error] = read_rows(files.errors)
    assert error["error"] == "server_died", error
    assert result.errors == 1
    trace = json.loads((files.traces / "baseline" / f"{STEP_AC.id}_rep0.json").read_text())
    assert trace["error"]["error"] == "server_died"


async def test_a_model_other_than_the_one_asked_for_is_an_error(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], _oracle, served_model="claude-someone-else")
    await run_suite(config, files)
    assert read_rows(files.results) == []
    [error] = read_rows(files.errors)
    assert error["error"] == "served_model_mismatch"


def _edits_an_input(ctx: ScriptContext) -> Generator[Step, list[Seen], None]:
    yield Call("write_file", {"path": "step_ac.log", "content": "changed"})
    yield Call("write_file", {"path": "answer.json", "content": "{}"})


async def test_a_changed_input_is_a_guardrail_violation(tmp_path: Path):
    config, files = _config(tmp_path, [STEP_AC], lambda task: _edits_an_input)
    await run_suite(config, files)
    [row] = read_rows(files.results)
    assert row["guardrails"]["read_only_changed"] == ["step_ac.log"]


# ---------------------------------------------------------------------------
# The launcher


def test_the_launcher_keeps_the_sdks_default_variables():
    copied = (
        launch_server.WINDOWS_INHERITED
        if sys.platform == "win32"
        else launch_server.POSIX_INHERITED
    )
    assert set(copied) == set(DEFAULT_INHERITED_ENV_VARS)


def test_the_launcher_drops_what_would_override_the_variant():
    source = {
        "PATH": "/bin",
        "HOME": "/var/empty",
        "TMP": "/tmp",
        "LANG": "C.UTF-8",
        "LTSPICE_MCP_TEXT_CHANNEL": "json",
        "LTSPICE_MCP_HOME": "/elsewhere",
        "VIRTUAL_ENV": "/venv",
        "PYTHONPATH": "/src",
        "ANTHROPIC_API_KEY": "secret",
        "SHELL": "() { :; }",
    }
    env = launch_server.server_environment(source, "/trial/home", windows=False)
    assert env == {
        "PATH": "/bin",
        "HOME": "/var/empty",
        "TMP": "/tmp",
        "LANG": "C.UTF-8",
        "LTSPICE_MCP_HOME": "/trial/home",
    }


async def test_the_server_runs_without_the_callers_overrides(tmp_path: Path, monkeypatch):
    """A caller's ``LTSPICE_MCP_*`` and ``PYTHONPATH`` do not reach the server:
    the text channel stays the summary, and a shadowing package on the
    caller's path is not the server that answers."""
    shadow = tmp_path / "shadow" / "ltspice_mcp"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "__main__.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    monkeypatch.setenv("LTSPICE_MCP_TEXT_CHANNEL", "json")
    monkeypatch.setenv("PYTHONPATH", str(shadow.parent))
    variant = load_variant("baseline")
    dirs = make_trial_dirs(STEP_AC)
    write_trial_config(variant, dirs.workspace)
    try:
        with (dirs.root / "server.stderr").open("w", encoding="utf-8") as errlog:
            async with connect(variant, dirs, errlog) as server:
                assert server.server_info["name"] == "ltspice-mcp"
                result = await server.session.call_tool(
                    "inspect", {"queries": [{"kind": "guide"}]}
                )
                assert isinstance(result, mcp_types.CallToolResult)
                text = result.content[0]
                assert isinstance(text, mcp_types.TextContent)
                assert not text.text.lstrip().startswith("{")
    finally:
        sweep(dirs.home)


# ---------------------------------------------------------------------------
# Rendering and the JSON the model reads


def _result(
    structured: Any, *, error: bool = False, text: str = "one line"
) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        structured_content=structured,
        is_error=error,
    )


def test_structured_rendering_is_the_json_claude_code_writes():
    rendered = render(_result({"a": 1.0, "b": [1e-7, 0.5]}), "structured")
    assert rendered.content == [{"type": "text", "text": '{"a":1,"b":[1e-7,0.5]}'}]


def test_content_rendering_is_the_text_blocks():
    rendered = render(_result({"a": 1}), "content")
    assert rendered.content == [{"type": "text", "text": "one line"}]


def test_an_error_reaches_the_model_as_its_text_only():
    rendered = render(
        _result({"error": "x", "code": "invalid_arguments"}, error=True, text="bad"), "structured"
    )
    assert rendered.is_error
    assert rendered.content == [{"type": "text", "text": "bad"}]


def test_an_oversized_result_is_replaced_by_a_notice():
    rendered = render(_result({"a": "x" * (OVERSIZED_TOKENS * 4 + 100)}), "structured")
    assert rendered.oversized
    assert "not shown" in rendered.content[0]["text"]


@pytest.mark.parametrize(
    ("value", "javascript"),
    [
        (1.0, "1"),
        (-0.5, "-0.5"),
        (1e-7, "1e-7"),
        (0.000001, "0.000001"),
        (123.456, "123.456"),
        (1e21, "1e+21"),
        (1.5e300, "1.5e+300"),
        (1.2345678901234568e20, "123456789012345680000"),
        (5e-324, "5e-324"),
        (2**53, "9007199254740992"),
        (2**60, "1152921504606847000"),
        (math.nan, "null"),
        (0.1 + 0.2, "0.30000000000000004"),
    ],
)
def test_numbers_are_spelled_as_javascript_spells_them(value: float, javascript: str):
    assert jsonjs.number(value) == javascript


def test_strings_and_containers_match_compact_json():
    value = {"k": ["µ", "a\nb", True, None], "n": 3}
    assert jsonjs.dumps(value) == json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# The answer grader


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("1590", 1590.0),
        ("1.59k", 1590.0),
        ("1.59 kHz", 1590.0),
        ("10 MHz", 1e7),
        ("2ms", 2e-3),
        ("1meg", 1e6),
        ("5 F", 5.0),
        ("5 fF", 5e-15),
        ("-3 dB", -3.0),
        ("4.7µ", 4.7e-6),
        ("ten", None),
        ("5 apples", None),
    ],
)
def test_a_number_may_be_written_with_its_suffix_and_unit(text: str, number: float | None):
    got = as_number(text)
    assert got == pytest.approx(number) if number is not None else got is None


def test_the_grader_compares_every_reference_key():
    reference = {"f": [100.0, 200.0], "ok": True, "name": "Low-pass", "none": None}
    tolerance = Tolerance(rel=0.02)
    assert (
        compare(
            {"f": [101, "0.199k"], "ok": "yes", "name": " low-pass ", "none": None, "x": 1},
            reference,
            tolerance,
        )
        == []
    )
    mismatches = compare({"f": [110, 200], "ok": False, "name": "band-pass"}, reference, tolerance)
    assert [m.split(":")[0] for m in mismatches] == ["$.f[0]", "$.ok", "$.name", "$.none"]
    assert compare({"f": [100.0]}, {"f": [100.0, 200.0]}, tolerance) == [
        "$.f: expected 2 items, got 1"
    ]


def test_a_per_key_tolerance_applies_to_its_key():
    reference = {"r": 1.0, "peak_v": 2.0}
    fields = {"r": Tolerance()}
    assert compare({"r": 1.01, "peak_v": 2.01}, reference, Tolerance(rel=0.01), fields) == [
        "$.r: 1.01 is not within exact equality of 1.0"
    ]


# ---------------------------------------------------------------------------
# File tools


def test_file_tools_stay_inside_the_workspace(tmp_path: Path):
    tools = FileTools(tmp_path)
    tools.call("write_file", {"path": "a/b.txt", "content": "one\ntwo\n"})
    assert "     2\ttwo" in tools.call("read_file", {"path": "a/b.txt"})
    tools.call("edit_file", {"path": "a/b.txt", "old_string": "two", "new_string": "three"})
    assert (tmp_path / "a" / "b.txt").read_text(encoding="utf-8") == "one\nthree\n"
    assert "a/" in tools.call("list_dir", {})
    with pytest.raises(FileToolError, match="outside the workspace"):
        tools.call("read_file", {"path": "../escape.txt"})
    with pytest.raises(FileToolError, match="outside the workspace"):
        tools.call("write_file", {"path": str(tmp_path.parent / "escape.txt"), "content": ""})


def test_the_scripted_model_charges_one_dollar_a_turn():
    assert Usage(input=1000).usd(SCRIPTED_PRICE) == pytest.approx(1.0)
