# Agent evaluation harness — design

Status: proposal (draft 1), not implemented.

## 1. Why, and which decisions it serves

The contract tests pin what each tool returns. They cannot tell whether a
model, given this server and nothing else, finishes real tasks, or what that
costs. Several open changes to the tool surface trade context for
discoverability and need that number before they ship:

- publishing recipes as names only, with arguments looked up through
  `inspect(kind="reference")` (the listing is about 47 KB in compact mode,
  most of it the recipe union);
- a detail level for reference search;
- the default of `[tools] text_channel` and `[tools] listing`;
- regressions from any later surface change (reply shapes, budgets, errors,
  guide text).

The harness measures, per task: whether the agent succeeded (judged from what
it left in its workspace), how many tool calls and turns it took, how many
tool errors it hit and of which code, how many tokens it spent and on what,
and how large the server's replies were. It compares server variants on the
same tasks with the same model, paired, with confidence intervals.

Non-goals: ranking models in general; testing simulator numerics (the
recorded-fixture tests do that); a CI gate (runs cost money and are opt-in).

## 2. What is under test

The server, as a client sees it: `python -m ltspice_mcp` over stdio, with a
generated `ltspice-mcp.toml`. The model and the client are the environment
the server is evaluated in, held fixed within a comparison. A variant changes
the server only: its configuration, its code revision, or both.

## 3. Architecture

```
evals/
  tasks/<task_id>/task.toml     prompt, category, requirements, grader, split
  tasks/<task_id>/workspace/    files copied into the trial's workspace
  tasks/<task_id>/grader/       reference data, never copied into a workspace
  variants/<name>.toml          server revision + config overlay
  ltspice_eval/                 runner, client profiles, graders, report rows
  runs/<suite>/<variant>/       results.jsonl, errors.jsonl, traces/ (gitignored)
```

One trial is one (task, variant, rep). Each trial:

1. creates a fresh temporary workspace and copies `workspace/` into it;
2. writes the variant's `ltspice-mcp.toml` there, with `allowed_paths` set to
   the workspace, job persistence inside it, and `LTSPICE_MCP_HOME` inside it;
3. starts the server for that variant (`uv run --project <checkout> python -m
   ltspice_mcp`, cwd = workspace), so no trial sees another's jobs or state;
4. runs the task through the chosen client profile until the agent stops, a
   turn limit, a token limit or the wall-clock ceiling;
5. stops the server, grades the workspace, writes one row (or one error row),
   and keeps the trace.

Trials run concurrently under a semaphore sized to the account's rate limit
and the machine's simulator capacity.

## 4. Client profiles

Two profiles, because clients differ in exactly the ways the surface work
cares about (whether `structuredContent` or the text block reaches the model,
how large a tool result may be, which file tools exist).

### 4.1 `api`: a controlled client (default)

A manual tool-use loop on the Anthropic Messages API (Python `anthropic`
SDK), with the server's tools proxied from an MCP `ClientSession`. Manual
rather than the SDK's tool runner, because the harness must control exactly
what each tool result looks like to the model.

- **Tools**: the server's `tools/list`, in the order served, plus harness file
  tools confined to the workspace: `read_file`, `write_file`, `edit_file`,
  `list_dir`. No shell: the agent reaches the simulator only through the
  server, which is the surface under test.
- **System prompt**: a short frozen preamble naming the workspace and the
  answer convention (§5.3), followed by the server's `initialize`
  instructions, as MCP clients place them.
- **Result rendering**, a profile option recorded on every row:
  - `structured` — the tool result is the JSON of `structuredContent`, with
    image blocks passed as images; this is what structured-aware clients
    such as Claude Code show the model;
  - `text` — only the `content` blocks, as a client that ignores
    `structuredContent` shows them.
- **Result cap**: a tool result over a configurable size (default 25,000
  tokens, as Claude Code enforces) is replaced by an error saying so,
  mimicking that client; the row counts how often it happened.
- **Caching**: `cache_control` on the last system block (caches tools and
  system together) and on the last block of the latest turn; tools are never
  reordered within a trial.
- **Model and effort**: run parameters, recorded per row from the response.
  The served `model` is asserted against the requested one on every call.

### 4.2 `claude-code`: the real client

`claude -p <prompt> --mcp-config <file> --output-format stream-json`, with a
pinned CLI version recorded per row. The allow-list is the server's tools plus
Read, Write, Edit, Glob and Grep; Bash is off by default, for the same reason
as above and because it would let the agent run ngspice itself. Usage, tool
calls and results are read from the stream. This profile is slower and less
controlled, and is used to check that `api` results transfer: the baseline is
run under both, and a variant that wins under `api` is confirmed under
`claude-code` before it ships.

## 5. Tasks

### 5.1 Categories

| category | needs | the agent is asked to | graded on |
|-|-|-|-|
| `analyze` | nothing | read a recorded `.raw`/`.log` in the workspace and report a quantity (rise time, −3 dB point per step, which step overshoots most, a failed `.meas`) | the reported value against a reference, within a tolerance |
| `measure` | ngspice | run a given deck, possibly with a sweep, and report a result (phase margin, the R1 where V(out) crosses 2.5 V) | the reported value against a reference |
| `design` | ngspice | write a deck meeting a spec (an RC low-pass at 1 kHz ±5% with E12 parts; a divider sourcing 1 mA) | the grader simulates the agent's deck itself and checks the spec and constraints |
| `debug` | ngspice | fix a deck that fails or misbehaves, given the symptom a user would report | the fixed deck runs and meets the stated behaviour |
| `schematic` | symbol fixtures | draw a netlist as an `.asc` with `edit_schematic` | the build benchmark's scorer: the sheet is the netlist (equivalence), and its sheet findings |
| `ltspice` | LTspice (Windows) | `.meas`-based work and running a sheet | as for `measure`; opt-in suite |

Every category includes cases whose right answer is a refusal or a negative
("the run did not converge", "that trace does not exist"), so a policy of
always answering cannot score well.

### 5.2 The task file

```toml
id = "analyze-step-ac-f3db"
category = "analyze"
split = "dev"                 # dev | holdout, drawn at random per category
requires = []                 # "ngspice", "ltspice"
prompt = """
step_ac.raw in your workspace is a stepped AC run of a filter. For each step,
what is the -3 dB frequency of V(out)?
"""
[grader]
kind = "answer"               # answer | deck | sheet | custom
answer_schema = "per_step_frequencies"
tolerance = { rel = 0.02 }
reference = "grader/reference.json"
[limits]
max_turns = 30
wall_clock_s = 600
```

### 5.3 How the agent answers

In every profile the agent writes `answer.json` in its workspace (the prompt
gives the expected keys); `design`, `debug` and `schematic` tasks are graded
on the files they leave. Graders read the workspace after the server has
stopped and never read the transcript for correctness. Transcript checks are
only process guardrails, reported as their own metrics: inputs marked
read-only were not changed; an `.asc` was changed only through
`edit_schematic`.

### 5.4 Where tasks come from

In order of preference: the maintainer's own past agent sessions with this
server, and notes already kept on where agents struggled with it; GitHub issues;
maintainer-written cases (three to five first, with a sentence each on what
makes them hard); then variations of those, synthesized from the guide's task
playbooks. Target 30 to 40 tasks for the first suite, a third held out.
The maintainer reviews every task and signs off before any paid run, and again
on the graders after a five-task pilot.

## 6. Grading

- Graders are independent of the server code under test. References are
  computed when a task is authored (ngspice version recorded), and a `design`
  or `debug` grader runs the agent's deck with ngspice directly, not through
  the server.
- Reference data lives in `grader/`, which is never copied into a workspace.
- Tolerances accept equivalent forms (units, `1k` vs `1000`, rounding).
- Every grader is tested offline on a known-good and a known-bad end state.

Outcomes are classified, and only model outcomes are scored:

| class | goes to | example |
|-|-|-|
| pass / fail | `results.jsonl` | wrong value, missing `answer.json`, deck misses spec |
| refusal | `results.jsonl`, own metric | the model declined |
| truncated | `results.jsonl`, `status: truncated` | hit `max_tokens` |
| harness or server error | `errors.jsonl` | server crashed, MCP transport died, grader crashed |
| timeout | `errors.jsonl` | wall-clock ceiling |

## 7. Metrics

Per row: `pass`; turns; tool calls, total and per tool; tool errors, total and
per `error.code` (including `invalid_arguments`); tool results over the cap;
size of each tool result (estimated tokens), total and largest; guide and
reference lookups; `usage` (input, output, cache read, cache write) summed
over calls and per call; served model; latency (model time and total wall
clock, separately); retries; guardrail violations.

Reported per variant: pass rate with a 95% interval; tokens, cost and tool
calls per **solved** task; tool errors per trial; paired differences against
the baseline with bootstrap intervals. Cost is derived from `usage` and the
row's model, never measured another way.

## 8. Experiment design

- **Variants** are a server revision (a git ref checked out into a worktree)
  plus a config overlay: `baseline` is master with defaults; `v1`, `v2`, … are
  named in their `change.md`.
- **Paired**: every variant runs the same tasks and reps with the same model,
  effort and profile.
- **Reps**: 3 per (task, variant) to start. With 30 tasks that is 90 trials a
  variant, a noise floor of roughly ±10 points on pass rate; efficiency
  metrics are continuous and resolve smaller effects.
- **Sensitivity check** before trusting a result: an ablation variant (tool
  descriptions and guide stripped) must score clearly worse than baseline. If
  it doesn't, the suite isn't measuring the surface.
- **Held-out**: decisions are made on `dev`; the chosen variant is confirmed
  on `holdout`, which is never used to tune text.

## 9. Output

`runs/<suite>/<variant>/results.jsonl` (one row per trial, written as each
finishes, resumable at the (task, rep) key), `errors.jsonl`, and
`traces/<task>_rep<k>.json` (the full exchange: system prompt, every tool call
and result, the final answer, and the grader's inputs and verdict). A summary
command prints the per-variant table and writes a static HTML report.

## 10. Testing the harness itself

Offline, in `tests/`, with no API calls: a scripted model replays a fixed
sequence of tool calls through the real loop and server. An oracle script
must pass every seed task, and a null script (answers at once with a constant)
must fail them. Also tested: each grader on good and bad end states, error
classification (a killed server lands in `errors.jsonl`), resume, and the
served-model assertion.

## 11. Safety and isolation

- The workspace is the only directory the harness file tools and the server's
  `allowed_paths` accept.
- `run_code` is off in every variant unless the variant is about it; such a
  variant runs only inside a container, since the snippet has the server
  process's authority.
- The `claude-code` profile runs with Bash disabled.
- Tasks and fixtures are authored by the project, so no untrusted text enters
  a prompt.

## 12. Cost

No estimate is quoted before a pilot. The first paid step is a five-task
pilot on one variant; its measured `usage` gives tokens per trial (spread, not
just mean), dollars per full run and wall clock, shown with the arithmetic,
and the full run waits for the maintainer's yes. `--max-usd` stops a run when
spend reaches it.

## 13. Platforms

The runner and graders use `pathlib`, asyncio subprocesses and UTF-8 files
written explicitly, so they run on Windows, where the `ltspice` suite runs.
`analyze` and `schematic` tasks need no simulator; `measure`, `design` and
`debug` need ngspice on PATH.

## 14. Plan

1. The maintainer supplies seed tasks and any past sessions to mine.
2. Harness skeleton, both profiles, scripted-model tests, eight seed tasks; no
   paid calls.
3. Task set (30–40) reviewed and signed off; grader pilot reviewed.
4. Paid pilot, cost estimate, consent; baseline under both profiles.
5. First decisions: `listing` compact vs full, `text_channel`; then recipe
   disclosure and reference detail as their branches land.

## 15. Open questions

1. Budget, and credentials for each profile.
2. Models to evaluate: the default model at its default effort, and which
   others.
3. Whether past session transcripts exist to mine for tasks.
4. Where it runs: Linux or WSL with ngspice for the main suite; a Windows
   machine for the `ltspice` suite.
5. Whether the task set and harness are committed (recommended; runs and
   traces stay out of git).
