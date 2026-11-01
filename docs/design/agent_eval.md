# Agent evaluation harness — design

Status: proposal (draft 3, revised after two rounds of adversarial review),
not implemented.

## 1. Why, and which decisions it serves

The contract tests pin what each tool returns. They cannot tell whether a
model, given this server and nothing else, finishes real tasks, or what that
costs. Several open changes to the tool surface trade context for
discoverability and need that number before they ship:

- publishing recipes as names only, with arguments looked up through
  `inspect(kind="reference")` (the listing is about 47 KB in compact mode,
  most of it the recipe union);
- a detail level for reference search;
- the defaults of `[tools] listing` and `[tools] text_channel`;
- regressions from any later surface change (reply shapes, budgets, errors,
  guide text).

The harness measures, per trial: whether the agent succeeded (judged from what
it left in its workspace), how many turns and tool calls it took, how many tool
errors it hit and with which code, how many tokens it read and wrote, and how
large the server's replies were. It compares server variants on the same tasks
with the same model and client, paired.

Non-goals: ranking models; testing simulator numerics (the recorded-fixture
tests do that); a CI gate (runs cost money and are opt-in).

## 2. What reaches the model: a finding that comes first

Which channel a client shows the model decides whether a reply carries any
data, and it is not the same for every client or even for every Claude Code
session. In Claude Code 2.1.316, the function that turns an MCP tool result
into the model's tool result does this when the result has
`structuredContent`:

- by default, the model gets the serialized `structuredContent` (plus any
  image blocks), and the text blocks are dropped;
- when a remotely controlled flag is on (`tengu_eventual_noodle`, default
  off), the model gets the `content` blocks instead, and `structuredContent`
  only when every text block is blank.

The server's default suits the first case. But under `text_channel =
"summary"`, a Claude Code session with that flag on reads one summary line per
reply and no data. The
eval does not need to establish this; it is a product risk to settle on its own
(set `text_channel = "json"` by default, or make the summary line carry the
data). The eval records which rendering a trial used and, in the `claude-code`
profile, reads what the model actually received from the stream rather than
assuming it.

## 3. What is under test

The server as a client sees it: `python -m ltspice_mcp` over stdio with a
generated `ltspice-mcp.toml`. Both profiles start it through one launcher,
`python -I evals/launch_server.py <variant>`, which sets its own environment and
execs the variant checkout's venv Python. The environment starts from the MCP
SDK's default list (Windows: `APPDATA HOMEDRIVE HOMEPATH LOCALAPPDATA PATH
PATHEXT PROCESSOR_ARCHITECTURE SYSTEMDRIVE SYSTEMROOT TEMP USERNAME
USERPROFILE`; elsewhere `HOME LOGNAME PATH SHELL TERM USER`), adds `TMP`,
`LANG`/`LC_ALL`, `WSL_DISTRO_NAME` and the trial's `LTSPICE_MCP_HOME`, and
drops every other `LTSPICE_MCP_*` variable (they override the TOML), plus
`VIRTUAL_ENV` and `PYTHONPATH`. A launcher is needed because Claude Code, not
the harness, decides what a stdio server inherits. The model
and the client are the environment the server is evaluated in, fixed within a
comparison. A variant changes the server only: its configuration, its code
revision, or both.

## 4. Architecture

```
evals/
  tasks/<task_id>/task.toml     prompt, category, requirements, grader
  tasks/<task_id>/workspace/    files copied into the trial's workspace
  tasks/<task_id>/grader/       reference data, never copied into a workspace
  variants/<name>.toml          server revision + config overlay
  ltspice_eval/                 runner, client profiles, graders, rows
  runs/<suite>/                 results.jsonl, errors.jsonl, traces/ (gitignored)
```

One trial is one (task, variant, rep):

1. a fresh temporary workspace receives `workspace/`, the variant's
   `ltspice-mcp.toml` (`allowed_paths` = the workspace; job store inside it)
   and a private `LTSPICE_MCP_HOME`;
2. the variant's server starts through the launcher from its own checkout,
   whose venv is synced once per run before any trial (no `uv run` at trial
   time, so concurrent trials never race on a venv or need a cache dir);
3. the client profile runs the task until the agent stops or a limit fires;
4. the server's whole process tree is stopped (a Job object on Windows),
   since detached job owners and workers can outlive it, and only then is the
   workspace graded, ignoring `.ltspice-mcp/` and `ltspice-mcp.toml`;
5. one row (or one error row) and the trace are written.

Trials of all variants are interleaved in a randomized order under one
concurrency limit, so rate limits, cache warmth and time of day fall on every
variant alike.

## 5. Client profiles

### 5.1 `api`: a controlled client (default)

A manual tool-use loop on the Anthropic Messages API (`anthropic` SDK), with
the server's tools proxied from an MCP `ClientSession`. Manual rather than the
SDK's tool runner because the harness controls exactly what each tool result
looks like.

- **Tools**: the server's `tools/list` as served, plus harness file tools
  confined to the workspace (`read_file`, `write_file`, `edit_file`,
  `list_dir`). No shell: the simulator is reached only through the server.
- **System prompt**: a frozen preamble with the answer convention, then the
  server's `initialize` instructions, as Claude Code places them (under its
  MCP server instructions heading). The workspace path goes in the first user
  message, not the system prompt, so tools and system cache across trials.
- **Rendering** of a result with `structuredContent`, recorded per row:
  `structured` (the serialized `structuredContent` plus image blocks: Claude
  Code's default) or `content` (the content blocks: Claude Code with the flag
  on, and clients that ignore `structuredContent`). Runs use `structured`
  unless a variant is about the text channel.
- **Oversized results**: Claude Code saves a tool result over 25,000 tokens
  to a file and hands the model its path. The profile records each such
  event and gives the model a short notice; it does not claim to reproduce
  the file hand-off.
- **Caching**: a breakpoint on the last system block and one on the latest
  turn; tools are never reordered within a trial.
- **Model and effort** are run parameters; the served `model` is asserted
  against the requested one on every call.

### 5.2 `claude-code`: the real client

```
claude -p <prompt> --output-format stream-json --verbose
  --strict-mcp-config --mcp-config <file naming the launcher as server "spice">
  --setting-sources "" --tools Read,Write,Edit,Glob,Grep --disallowedTools Bash
  --allowedTools "mcp__spice" --permission-mode acceptEdits
  --permission-prompts none
  --no-session-persistence --model <m> --effort <e> --max-turns <n>
  --max-budget-usd <cap>
```

run with `DISABLE_AUTOUPDATER=1`, a pinned CLI version recorded per row, and
a scratch `HOME` (an empty `.claude/`, credentials from `ANTHROPIC_API_KEY`).
`--permission-prompts none` denies anything that would prompt, and MCP tools
prompt unless allowed, hence `--allowedTools "mcp__spice"`.
`--strict-mcp-config` and empty setting sources keep the operator's MCP servers
and settings out; the scratch `HOME` keeps their `~/.claude/CLAUDE.md` out,
which no setting source governs. `--tools` also removes Task, TodoWrite and
WebFetch, which real sessions have: a fidelity trade-off, stated in the
report. Usage, tool calls and the tool results the model received are read
from the stream. This profile
confirms that an `api` result transfers: the baseline and any variant about to
ship run under it.

## 6. Tasks

### 6.1 Categories

| category | needs | the agent is asked to | graded on |
|-|-|-|-|
| `analyze` | nothing | read a recorded `.raw`/`.log` in the workspace and report a quantity (rise time, −3 dB point per step, which step overshoots most, a failed `.meas`) | the reported value against a reference, within a tolerance |
| `measure` | ngspice | run a given deck, possibly swept, and report a result (phase margin, the R1 where V(out) crosses 2.5 V) | the reported value against a reference |
| `design` | ngspice | write a deck meeting a spec (an RC low-pass at 1 kHz ±5% with E12 parts) | the grader simulates the agent's deck itself and checks spec and constraints |
| `debug` | ngspice | fix a deck that fails or misbehaves, given the symptom a user would report | the fixed deck runs and meets the stated behaviour |
| `schematic` | symbol fixtures | draw a netlist as an `.asc` with `edit_schematic` | equivalence to the netlist, and the sheet's findings (§7) |
| `ltspice` | LTspice (Windows, native) | `.meas`-based work and running a sheet | as for `measure`; opt-in suite |

Every category includes cases whose right answer is a negative ("the run did
not converge", "that trace does not exist"), so always answering cannot score
well. Task authors check what a fixture's name and its `.log` (`.step` lines,
`.meas` results) give away, and say whether a task tests log reading or trace
math.

`schematic` trials use the suite's symbol set (`tests/fixtures/symbols`, via
`[schematic] symbol_paths`) for the agent and the grader alike; LTspice's own
library changes geometry. `ltspice` trials run natively on Windows: under WSL
the server writes LTspice artifacts to a Windows temp directory outside the
workspace.

### 6.2 The task file

```toml
id = "analyze-step-ac-f3db"
category = "analyze"
requires = []                 # "ngspice", "ltspice"
prompt = """
step_ac.raw in your workspace is a stepped AC run of a filter. For each step,
what is the -3 dB frequency of V(out)? Write the answer to answer.json as
{"f3db_hz": [...]} in step order.
"""
[grader]
kind = "answer"               # answer | deck | sheet
tolerance = { rel = 0.02 }
reference = "grader/reference.json"
[limits]
max_turns = 30
wall_clock_s = 600
```

### 6.3 How the agent answers

The agent writes `answer.json` in its workspace; `design`, `debug` and
`schematic` tasks are graded on the files they leave. Graders never read the
transcript for correctness. Two transcript checks are kept as separate
guardrail metrics: inputs marked read-only were not changed, and an `.asc`
was changed only through `edit_schematic`.

### 6.4 Where tasks come from

In order: the maintainer's past agent sessions with this server and notes
kept on where agents struggled; GitHub issues; three to five maintainer-written
cases, each with a sentence on what makes it hard; then variations of those
drawn from the guide's playbooks. 30 to 40 tasks for the first suite. The
maintainer signs off the task set before any paid run, and the graders after a
five-task pilot.

## 7. Grading

**Independence.** A grader must not run the code a variant changes:

- References are computed when a task is authored, from a pinned grader
  environment, with the ngspice version recorded.
- `design` and `debug` graders append their own measurement lines to the
  agent's deck, run ngspice directly, and read the result with spicelib's raw
  reader or from ngspice's printed `.meas` output; never through the server.
- The `schematic` grader is the build benchmark's scorer, which imports the
  server's connectivity, sheet and netlist code. It runs from one pinned
  revision in its own environment, recorded on every row, whatever revision
  the variant runs. Equivalence counts parts, pins and nets; whether a value
  difference or a missing directive fails the task is set per task. On
  Windows, LTspice's own netlist export of the agent's sheet is the stronger
  check, used where available.
- Reference data lives in `grader/`, never copied into a workspace.
- Tolerances accept equivalent forms (units, `1k` vs `1000`, rounding).

**Outcomes, by cause:**

| outcome | where | status |
|-|-|-|
| pass, fail | `results.jsonl` | `ok` |
| `answer.json` missing or malformed | `results.jsonl` (fail) | `unparseable` |
| agent still working at the wall clock, turn or token limit | `results.jsonl` (fail) | `timeout`, `max_turns`, `budget` |
| refusal (classified from the transcript) | `results.jsonl` (fail), own metric | `refused` |
| server or transport died, grader crashed | `errors.jsonl` | error class |
| the run's `--max-usd` stopped the trial | `errors.jsonl` | `aborted` |

A simulator that hangs is stopped by the server's own run timeout and reaches
the agent as `simulation_failed`, so it is part of the trial, not an error.

A response stopped by `max_tokens` is one call's stop reason: the loop lets the
model continue, and only a trial that ends that way is marked.

## 8. Metrics

Per row: `pass`; status; turns; tool calls, total and per tool; tool errors,
total and per `error.code` (including `invalid_arguments`); reference and
guide lookups; size of each tool result in estimated tokens, total and
largest; oversized-result events; `usage` per call and summed (input, output,
cache read, cache write); served model; latency (model time and wall clock,
separately); retries; guardrail violations; the rendering, CLI version and
grader revision.

Reported per variant:

- pass rate, and paired differences against the baseline;
- **total input tokens** (input + cache read + cache write) and output tokens
  per trial: the primary size measure, since dollars depend on cache timing;
- tool calls and tool errors per trial;
- tokens per solved task, on the tasks every compared variant solved (at
  least one passing rep under each), averaged per task before differencing so
  3 reps do not count as 3 pairs, beside the unconditional figures, so a
  variant that skips hard tasks does not look efficient;
- intervals by bootstrap over **tasks** (resampling a task with all its
  reps), since reps of one task are correlated.

## 9. Experiment design

- **Variants**: a git ref checked out into a worktree plus a config overlay.
  `baseline` is master with defaults; `v1`, `v2`, … are described in their
  `change.md`.
- **Paired**: same tasks, reps, model, effort, profile and rendering.
- **Resolution**: with 30 tasks and 3 reps, a pass-rate difference near 50%
  has an interval of roughly ±17 points; token and tool-call measures resolve
  smaller effects, and are the expected movers for the §1 changes. The plan
  states the effect each decision needs before its run.
- **Noise floor**, measured rather than assumed: an A/A run (two variants
  with identical servers) before the first comparison.
- **Positive control**, with a known direction and of the kind the decision
  is about: a variant whose recipe schema renames one enum value that tasks
  need, so the published listing is wrong for it, must raise
  `invalid_arguments`. (Stripping the recipe argument descriptions is not a
  control: that is the recipe-disclosure change itself, whose result is the
  answer, not a test of the suite.)
- **Effect per decision**, stated before its run, with the minimum detectable
  size taken from the A/A run: for recipe disclosure, total input tokens down
  while pass rate and `invalid_arguments` stay within noise.
- **Confirmation**: no held-out split in the first version (30 tasks cannot
  spare one); a variant chosen under `api` is confirmed under `claude-code`.

## 10. Output

`runs/<suite>/results.jsonl`, one row per trial with its variant, written as
each trial ends and resumable at the (task, variant, rep) key;
`errors.jsonl`; `traces/<variant>/<task>_rep<k>.json` (system prompt, every
tool call and result as the model received it, the answer, the grader's inputs
and verdict). A summary command prints the per-variant table and writes CSV.

## 11. Testing the harness itself

Offline, in `tests/`, with no API calls:

- a scripted model replays a fixed sequence of tool calls through the real
  loop and a real server process; an oracle script must pass every seed task
  and a null script (answers at once with a constant) must fail them;
- a fake `ngspice` answers `-b -o <log> -r <raw> <netlist>` by copying
  recorded ngspice-format fixtures (the server parses logs by family, and the
  suite's recordings are mostly LTspice's, so new ones are recorded on Linux,
  where CI installs ngspice); it is named by `[simulator] path`, an `.exe` or
  `.cmd` on Windows, so `measure`-style seed tasks run in CI;
- each grader on a known-good and a known-bad end state;
- error classification (a killed server lands in `errors.jsonl`; an agent
  still calling tools at the ceiling lands in `results.jsonl` as `timeout`),
  resume, and the served-model assertion.

## 12. Safety and isolation

- The workspace is the only directory the harness file tools and the server's
  `allowed_paths` accept.
- `run_code` is off in every variant unless the variant is about it; such a
  variant runs only in a container, since the snippet has the server
  process's authority.
- The `claude-code` profile runs with Bash disabled and the operator's
  settings excluded.
- Tasks and fixtures are authored by the project.

## 13. Cost

No estimate before a pilot. The first paid step is five tasks on one variant;
its measured `usage` gives tokens per trial (spread, not just mean), dollars
per full run and wall clock, with the arithmetic shown, and the full run waits
for the maintainer's yes. `--max-usd` stops a run when spend reaches it.

## 14. Platforms

The runner and graders use `pathlib`, asyncio subprocesses and explicit
UTF-8. ngspice runs on Windows; the variant TOML sets `[simulator] path`
rather than relying on `PATH` or spicelib's search of `C:/Spice64`.

## 15. Plan

0. A command-line smoke test (one `claude-code` call on a trivial task,
   approved first): the server is allowed, its environment is the launcher's,
   the operator's settings and `CLAUDE.md` are absent, and the stream shows
   what the model received.
1. The harness, in three steps, none with paid calls:
   a. the `api` loop and MCP proxy, the `answer` grader, `analyze` tasks (no
      simulator), oracle and null scripts, result rows;
   b. the fake ngspice, the `deck` grader, `measure`, `design` and `debug`
      tasks;
   c. the `sheet` grader (its scorer exists) and `schematic` tasks.
2. The task set reviewed and signed off; a five-task grader pilot reviewed.
3. Paid pilot, cost estimate, consent; A/A run; baseline under `api`, then
   under `claude-code`.
4. First decisions: `listing` compact vs full; then recipe disclosure (with
   its positive control) and reference detail as their branches land.

Not in the first version: an HTML report, a held-out split, both renderings on
every run.

## 16. Open questions

1. Budget, and credentials for each profile.
2. Models: the default model at its default effort, and which others.
3. Past session transcripts to mine for tasks.
4. Machines: Linux or WSL with ngspice for the main suite; Windows for the
   `ltspice` suite.
5. Committing the task set and harness (recommended; runs stay out of git).
6. The `text_channel` default (§2), which is a product decision ahead of the
   eval.
