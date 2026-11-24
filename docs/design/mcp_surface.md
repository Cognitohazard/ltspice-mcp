# The MCP tool surface

This is the reference for the tools `ltspice-mcp` advertises. It covers the
shared response envelope, each tool's arguments and reply fields, and the
guarantees each tool makes.

Where this document and the source disagree, the source is correct and this
document needs fixing.

---

## 1. Shape: six operations over three planes

| Plane | Tool | One-line contract |
|-|-|-|
| EXECUTE | `run_experiments` | circuits x declared variations; durable idempotent receipt (plus inline results when the job finishes within `execution.wait_s`); preflight lint; runs read copies of the inputs taken at submission, with their hashes |
| EXECUTE | `jobs` | status / long-poll wait / cancel / resume / list / runs over receipts; resume continues the latest job of a recoverable job's resume chain, named by `job_id`; place_beside_sheet copies a case's results beside the sheet it ran |
| UNDERSTAND | `analyze_results` | runs a batch of recipes over runs and returns typed values with case and step identity, attributed reductions and spec verdicts; every recipe is evaluated in the call, and a reply too long for its target is cut and read on from the stored answer |
| UNDERSTAND | `inspect` | reads: capabilities, symbols (list and detail), net trace, components, models, result facts, the guide, and the reference entries for the tools' arguments, recipes, ops and other branches |
| AUTHOR | `edit_schematic` | applies a batch of typed ops to a sheet (blank or existing) as one transaction; committing to an existing file needs its current `expected_sha256`; returns pin and net tables, per-op results and findings |
| AUTHOR | `verify_circuit` | gate: lint/syntax, symbols, export, layout, quality, compare (equivalence or structural diff), render |

A seventh tool, `plot_waveform`, is the interactive MCP Apps waveform widget
and does not use the envelope. Its reply summarizes each plotted trace:
minimum and maximum with their positions, first and final value, and on a
transient the time-weighted mean. Past 32 traces it summarizes the first 32,
counts every plotted trace in `traces_truncated`, and names in `hint` the
`analyze_results` call that reads the rest. `attach_plot` adds a PNG of the
chart for a vision model. Traces are split into panels by declared
unit, or by the caller's `panels`. A plot with no sampled axis (an operating
point, a `.tf`, or a stepped `.op` or one-frequency `.ac`, which hold one point
a step) has no chart: it is refused with `no_axis`, which says what the plot
holds and where its values are read, and with `in_ltspice` it is opened in
LTspice without a chart, with no traces named for it and a `no_chart`
observation. With `in_ltspice`, a job's run of a sheet is put beside the sheet
by the copy `jobs(action="place_beside_sheet")` makes, with its checks and
locks, and opened from the sheet; the `ltspice` block then carries that copy's
`not_checked` and `left_in_place`. Where the copy is refused (the sheet changed
since the run or is gone, the case's results are not there, or the run was
not LTspice's), or another server process held the sheet past the lock's wait
(`sheet_locked`), nothing is put beside the sheet, the run opens on its own as
a netlist's run does, and the block's `not_placed` gives the `code` and
`message`. If LTspice's window had already closed a plot of the sheet's
results for the copy, the `note` says so.

`plot_waveform` draws several cases of one job in one chart when `runs` is
given beside `job_id`, in the grammar of an `analyze_results` source: `"all"`,
a list of run indices, or `{case_ids: [...]}`. Both tools select the cases
through one function (`services.select_cases`), so one selection reads the
same cases in both. `runs` beside `run_index`, `case_id` or `path` is refused,
and a selection of one case reads as that case named by `case_id` does. With
several cases, each trace's legend label is the signal, then the assignments
that differ among the cases drawn, then the step's values where the case is
stepped (`V(out) [R1=1000;r=22]`); where the assignments do not tell a case
apart, or more than three differ, the case id stands in for them. One function
writes that label (`points.point_label`), and a chart of one case keeps its
labels. Each trace summary names its `case_id` and `run_index`, the reply's
`cases` lists each case drawn with its `assignments` and `n_steps`, the
number of steps of the plot it drew (a `steps_not_as_recorded` observation
names a case whose run row's `steps` says otherwise: its results file was
rewritten after the run, or its record is wrong), and `cases_not_drawn` names each case
selected and not drawn, at most 20 rows with the total in
`cases_not_drawn_truncated`: `case_not_found` and `run_not_found` as
`analyze_results` names them (one row per code, with their count and the
first five), `run_not_finished` and `raw_not_produced` with the case's
`status`, `other_raw_not_written`, `result_unreadable` or `analysis_deadline`
for a results file that cannot be parsed, `plot_not_found` for one without
the `plot_index` asked for, `load_time_exhausted` for one not loaded within
the chart's bound on loading, and `step_not_found` for a case without the
`step` asked for, which applies to every case. The results files load side
by side, two at a time, through the parse service, each under its own parse
deadline, within one bound on the whole of it (twice one file's parse
deadline, 240 s). Before any loads, each first plot's header (at most its
first 256 KiB), the cases' records and the request say whether the chart
would be over its cell limit were every case drawn; where it would, it is
refused as soon as the cases loaded are over by themselves, and the files
after them are not loaded. Neither count holds a case the chart will not
draw, so neither refuses a chart that would be drawn. A signal some cases
lack is drawn for the others, and `signals_not_in_cases` names those that lack
it. A selection with nothing to draw is refused with why for each case. One
chart has one kind of axis: cases whose plots differ in analysis are refused,
named by analysis, with the advice to plot them in separate calls; cases with
different time grids are drawn together, as steps with different axes are.
The limits count series across cases and steps together; when the summary
limit cuts the traces, `cases_unsummarized` counts the cases with no trace
summarized and `hint` names the `analyze_results` call that reads the rest.
`in_ltspice` with several cases is refused: an LTspice window shows one
case's results from its sheet.

An eighth tool, `run_code`, is always registered and is served unless the
operator sets `[tools] run_code = false`. It runs a Python snippet in a warm
worker process that holds the Python API as `api` (the six operations as methods
returning complete results), for loops over runs and numpy on samples. It does
not use the envelope either. It returns `status` (ok, error, timeout, busy,
reset) and the captured output, and its tool description is its contract. The
snippet runs with the server process's file and process permissions, which
`allowed_paths` does not restrict, so turn `run_code` off on a server exposed
to anything beyond one trusted local client.

Typical sequences:

```
netlist loop:   write deck (natively) -> run_experiments -> analyze_results -> edit natively -> ...
schematic new:  inspect(symbols) -> edit_schematic{base:"blank", ops, compare} -> revise ops -> done
schematic edit: inspect existing .asc -> edit_schematic{ops, expected_sha256} -> verify_circuit
debug loop:     verify_circuit(lint) -> fix natively -> run_experiments -> analyze_results -> inspect(net)
long runs:      run_experiments (receipt) -> jobs(wait) -> analyze_results
```

---

## 2. Shared conventions

These conventions apply to all six tools.

**Envelope.** `outcome` is `complete`, `partial`, `failed` or `in_progress`;
`in_progress` is a receipt for a job that is still running. A failure includes
`error {code, message, stage, retryable?, commit_state, item_id?}`, where
`retryable` is present only when it is true (see "Fields left out" below).
`commit_state` is `not_started` when nothing was recorded or written,
`committed` when the job or the edit was recorded and a later step failed (so
do not resubmit under a new id), and `unknown` when the server cannot tell, as
when another submission holds the `request_id`. MCP's `isError` is set only
for a call-level failure; a per-item failure sets `outcome: "partial"`.

**Unknown tool names.** A call to a tool the server does not have fails before
any tool runs, with a JSON-RPC invalid-params error (`-32602`) whose message
gives the requested name and lists the available tools. An unknown resource
URI gets the same error.

**Failures outside the envelope.** Arguments that fail validation are a tool
result with `isError`, so the caller can correct the call, and its
`structuredContent` is `{error, code: "invalid_arguments", hint}`, the hint
naming the `inspect(kind="reference")` lookup for the tool's arguments. The
error names each field it refuses and the way round: an unknown field with
where the tool takes it, the tool that takes it, its new name, or else the
names its place takes, the closest first; a field sent to a branch (a `jobs`
action, an `edit_schematic` op) that does not take it with that branch's
fields; and a missing field that chooses a branch with the values it takes,
beside the item's fields that no branch takes.
`tests/test_refusals_name_the_fix.py` holds every tool to this. A
failure a tool raises instead of reporting in its envelope is `{error, code,
hint?, suggestions?}` with the failure's own code, or `internal_error` for an
unclassified one; a `path_denied` refusal is `{error, code, hint}`. A
`run_experiments` or `jobs` refusal is the envelope's `error`, and its `hint`
is the message followed by the same recovery step. A resource read that fails
this way is a JSON-RPC invalid-params error whose message is the failure
followed by that step, and whose `data` is `{code, hint?}`; an unknown URI's
error has no `data`. Every surface takes the step for a kind of failure from
one table (`errors.ERROR_HINTS`), a refused path's from the session's sandbox.

**Error codes.** `code` is a stable public name for a failure. It depends on
the kind of failure and never on the message, except that a failure inside a
named stage may report that stage's code. A result read that fails while a
deck is being staged reports `submission_failed` instead of
`result_unreadable`.

Codes by kind of failure:

| code | meaning |
|-|-|
| `path_denied` | a sandbox (`allowed_paths`) is set and the path resolves outside it, or the path cannot be resolved; a `hint` describes the change to make (for a sandbox, the config file and key that widen it), returned as an item's `error.hint`, a finding's `evidence.hint`, a failure or coverage row's `hint`, or the envelope's `hint` for a call-level refusal |
| `invalid_arguments` | the call's arguments failed validation before the tool ran |
| `netlist_invalid` | the netlist, or a component reference in it, could not be read |
| `card_unreadable` | a card the call needs cannot be split into its parts (a quote, brace or parenthesis left open, or a closer with nothing open); the message names the file, the line, the card as written, what is wrong and the fix, whichever tool met it |
| `symbol_unresolved` | the schematic opened, but a symbol it, or a hierarchical block's own sheet, places was not found |
| `simulation_failed` | the simulator could not be started, or the run failed |
| `result_unreadable` | a `.raw` or `.log` could not be read |
| `run_not_finished` | a read named a case of a job whose run has not finished (queued, running, or its end-of-run checks not yet made); the message names the case, what its simulator has written so far and the wait. The job's finished cases are read meanwhile |
| `result_file_not_found` | no file is at a `.raw` or `.log` path the caller named; the `hint` says to check the path, or to read a run of this server's by its `job_id` |
| `analysis_deadline` | a result read ran past its wall-clock bound and was abandoned |
| `plot_not_found` | a read named a `plot_index` its results file does not hold; the message says how many plots it holds. The file is readable: the request does not fit it. Given by `inspect(kind="results")` and `plot_waveform`; `analyze_results` gives such a recipe's failure the code `recipe_failed`, with this message |
| `no_axis` | the selected plot is a table (an operating point, `.tf` or `.pz` result), so it has no sweep axis to query at |
| `job_not_found` | no job with that `job_id` or `request_id` |
| `invalid_cursor` | a page cursor did not decode, or was minted for another listing, view or request, or for a file that has changed since; the `hint` says to send back the cursor the same tool returned, or the message names the argument to change |
| `idempotency_conflict` | a `request_id` was reused for a different payload |
| `request_gate_busy` | another submission holds that `request_id`; retry with the same id |
| `cancel_not_authorized` | the call reached a server process other than the one running the job (the job's owning process) and presented no matching `control_token` |
| `cancel_failed` | cancellation was authorized but could not be carried out |
| `library_error` | a component library failed to load, parse, or resolve |
| `batch_job_error` | a sweep or Monte Carlo config could not be used |
| `raster_unavailable` | a PNG was requested without a usable rasterizer: `resvg-py`, a dependency, cannot be imported (an install that lost it), and `cairosvg` is not installed or is installed without the native Cairo library it loads |
| `internal_error` | an unclassified server failure |

Codes named after the stage, used where the stage is the more useful fact:

| code | meaning |
|-|-|
| `submission_failed` | the submission failed before a job was recorded |
| `submission_committed` | the job was recorded and a later step of the call failed; follow it with `jobs(status)` |
| `receipt_failed` | the job was recorded and building its receipt failed |
| `revision_conflict` | `expected_sha256` is missing or does not match the target |
| `asc_export_unavailable` | a `.asc` circuit could not be exported to a netlist |
| `unsupported_variant` | an unknown discriminant, with the supported list |
| `ambiguous_target` | a variation target names nothing, or names more than one thing |
| `lint_blocked` | a blocking lint finding stopped a circuit's cases with `lint: "block"` |

**Page object.** No tool returns an unbounded collection. Every collection
that can be large is a page, `{items, total, returned, truncated,
next_cursor}`, and every tool that returns a page accepts its cursor as input.
`verify_circuit`, `run_experiments` and `jobs` are on the presenter instead
(below): a list is cut only when its reply is too long to send whole.
`verify_circuit`'s `findings` and `jobs(list)`'s groups are then read on from
the stored answer (§3.5, §3.2); a job's run rows are read on from the job's
own record (§3.1).

An `inspect` item's `page` names the data key its cursor pages in a
`collections {<data key>: {total, returned, truncated}}` block, which is how
a client that pages any kind (the Python API's `inspect` among them) knows
which list to join. When one page contains several collections (an `inspect`
net view pages pins and wire coordinates together), `total` and `returned`
are sums over all of them, `collections` gives each collection's counts, and
`primary_truncated` reports whether the primary collection continues. On every page, `truncated` implies `returned < total`.
Stop paging when `next_cursor` is absent, not by the counters. A page leaves
out `items` when it is empty and `truncated` when it is false, as every reply
leaves out a field that says nothing; `total` and `returned` are always
there.

**Channels.** `observations`, `warnings` and `failures` are separate channels
and are never merged. `observations` are structured facts about the data and
the solve, such as relayed simulator errors, non-finite or extreme values and
coverage gaps. `warnings` are free text describing assumptions a measurement
made, such as a clamped window or an ambiguous edge. `failures` list what
failed. No field rates whether a result is trustworthy; the server reports
facts and the caller judges them.

**Findings shape** (lint, layout, quality):
`{rule_id, severity, ok, evidence, at {file, line? | x, y?}, subject}`. `at`
and `subject` are required on every finding a caller is expected to fix.

**Self-sufficiency.** `structuredContent` contains everything the caller needs,
including any `hint`, and the text block carries it serialized. When a sandbox
(`allowed_paths`) is set, every path a caller passes is checked against it.
Work on a `.asc` takes a cross-process lock on that circuit, so calls from
different sessions on one file run one at a time, and a call that waits more
than 10 s fails with a message to retry. The lock file is in the per-user
folder (`%LOCALAPPDATA%\ltspice-mcp` on Windows, `~/.local/state/ltspice-mcp`
or `$XDG_STATE_HOME/ltspice-mcp` elsewhere, or `$LTSPICE_MCP_HOME` when set),
never beside the circuit.

**Lean by default.** The default response contains per-case headline rows
(label, the recipe's scalar values with nested lists left out, verdicts) and a
one-line completeness count. `include.fields: ["value"]` adds the full nested
`value` dicts, and `provenance` (`include.provenance` on `analyze_results`)
adds digests, staged paths and the input manifest, including the paths and
digests in `source_hashes`. No mode trims failures,
errors, observations, warnings or a completeness shortfall; the lean form
omits only the confirming detail of a success.

A fact about the target that predates the call and involves nothing the call
named is scoped rather than trimmed. The reply does not list it; it counts it
beside the channel it was left out of and gives a way to list it. The call's
own warnings and failures are never scoped. `edit_schematic`'s `preexisting`
block (§3.4) applies this rule.

A circuit absent from `lint` had no findings; findings themselves are always
emitted. Pages and run rows follow "Fields left out" below like every other
part of a reply, so a row that produced no raw has no `raw`. Use `run_fields`
for a narrower row.

**Fields left out.** Every reply leaves out the fields that say nothing, so a
reply about work that found nothing costs no tokens for saying so. A field is
left out when its value is

- an empty list or an empty object,
- a null,
- a false flag, or
- a sub-answer the overall answer beside it already gives: in a comparison,
  `structurally_equivalent` and `directives_equivalent` beside
  `equivalent: true`, and in the capabilities
  report, `selectable` beside `available: false`. Beside any other overall
  answer every sub-answer is kept, so the caller sees which part held.

An absent list therefore reads as empty and an absent flag as false. A missing
field means "checked, and nothing to report"; it never means "not checked". A
check that did not run, a step that was skipped and a value that could not be
read are each stated by a field with content (a `checks_skipped` row, a
failure, a `status`, a `reason`), so they are never left out. The rule applies
to the fields of objects and never to the items of a list, so a null sample in
a series keeps its place. An object left empty once its silent fields are gone
is left out in turn.

One function applies the rule to every reply, on its way out of the server:
`structuredContent`, the text block that serializes it, and the Python API's
complete results, which therefore have the same shape. A number that is not
finite is the exception: the reply writes it as null at its place and names it
in `warnings`. Output schemas mark every field a reply may
leave out as optional. A field is kept anyway only when its empty, null or
false value is itself the answer. The table lists those fields for each tool,
wherever they appear in its reply, and is the whole exemption:

| tool | always present, even when empty, null or false | why |
|-|-|-|
| every envelope tool | `outcome` | the call's verdict |
| `run_experiments`, `jobs` | `ok` (lint findings) | false is what the finding reports |
| | `staged` (`source[].manifest`) | false says a reference was read in place or not staged, beside its `reason` |
| | `value` (attached analysis rows) | measured data, as for `analyze_results` |
| `jobs` | `exists` (`list` groups) | false says the circuit's file is gone |
| | `resumed` | false is a resume that selected nothing to run |
| `analyze_results` | `value` (rows and `reduced`) | a recipe's measured value is kept as the recipe returns it: a null is a quantity with no number, which reductions and specs count in `no_value_count`, and a false is a measured fact, such as `exact_match: false` for a query point moved to the nearest sample |
| `inspect` | `ok` (each result) | false carries the query's error |
| | `available`, `selectable`, `exporter_available`, `png`, `persist_jobs` (capabilities) | false says what this session cannot do |
| | `allowed_paths` (capabilities) | null says no sandbox is set; asked for by name with `fields` after a config edit |
| | `in_sandbox` (`open_in_ltspice`) | false says the document is outside the sandbox |
| `edit_schematic`, `verify_circuit` | `ok` (findings, stages, `export`) | false is what the finding or stage reports |
| | `equivalent` | false is a mismatch and null is no verdict |
| | `structurally_equivalent`, `directives_equivalent` | kept beside an `equivalent` that is false or null |
| | `shown` (`open_in_ltspice`, `ltspice`) | false says LTspice does not show what was asked, with the reason |
| `plot_waveform` | `shown`, `tied` (`ltspice`) | false says the window does not show the plot, or a click on the sheet does not plot |
| `run_code` | `status` | the snippet's outcome |

The list is `ALWAYS_PRESENT` in `tools/_base.py`, and the suite fails any
reply that carries a silent field not on it.

**A reply too long to send whole.** A tool on the presenter
(`lib/presenter.py`; `verify_circuit`, `run_experiments`, `jobs` and
`analyze_results` today, `inspect` as it moves) computes its complete answer
and hands it, with its output schema, to one function that decides what is
sent.
The target is one number: the tool's `budget` argument where it has one
(at least 500 estimated tokens; a smaller one is refused), else
`[analysis] default_budget` (4000 estimated tokens). A default of 0 and no
`budget` is no target: every answer is sent whole, except the lists an MCP
reply still caps by count, each named in `cut` and continued by its cursor,
so no reply carries an unbounded collection: a job's run rows at 50, and an
`analyze_results` recipe's `per_run` page at its `limit` (50 by default) and
every other list of its answer the presenter may cut (`values`, `groups`,
`spec.fail_cases`, `spec.outliers`, `coverage.missing_cases`,
`signals_available`, `source_hashes`) at 100 (`analyze.answer_caps`). Every
other list is declared never cut, with its bound (`tests/test_reply_cuts.py`).
The Python API's automatic mode returns every row.

- An answer within the target is sent exactly as "Fields left out" makes it.
  It carries neither field below, and nothing is stored.
- An answer over the target is stored whole, as one JSON file in the store's
  `results/` folder, under the expiry of `[analysis] result_set_ttl_hours`.
  Its id is the digest of the answer and of its snapshot, what it was computed
  from, so the same answer of the same inputs is one file.
- Then the lists the schema declares `items` or `rows` (`x-cut`,
  `lib/reply_cuts.py`) are cut from the end, whole entries only, the largest
  list first, until the reply fits or each holds one entry; a list whose cut
  to one entry would not shorten the reply (a short list, whose `cut` entry
  costs more than its entries) is left whole, and a list that is over at one
  entry is held there while the next is cut. A reply is over its target only
  when the lists, each at the least the search cuts it to, are over it
  together. A larger target does not always give a larger reply: for lists
  of whole entries at targets at or above that floor it gives one at most 40
  estimated tokens smaller (`tests/test_presenter_fit_property.py`), but a
  series is redrawn at the most points that fit, so a larger target can draw
  it with fewer, and below the floor the note that says a reply is over its
  target adds its own size. A list declared
  `never` is sent whole, and nothing inside a list's entries is cut apart from
  its entry: a list inside a row goes with the row. A reply that still does
  not fit goes out over the target.
- Before any list is cut, the samples a schema declares one `series` (an
  inline waveform's axis `x` and its `series`) are redrawn at the most points
  that let the reply fit, by a function the tool supplies, every list of the
  series at the same points and never below 32 points drawn. A thinned series
  is drawn for the entries the reply keeps, is not paged and has no cursor:
  its `cut` entry, at the pointer of its axis,
  gives the points the answer holds (`total`) and those sent (`returned`), and
  the tool's words name how to get more (`max_points`, the CSV format). The
  stored answer holds the series as computed, and a page thins its own.
- Several answers can share one reply and one target
  (`presenter.present_many`): a receipt and the analysis attached to it. Each
  keeps its own tool, snapshot and records, and carries its own `cut` and
  `stored_answer` at its root, so a cut list of the attached analysis is
  stored as an answer of `analyze_results` and read on by it. One size search
  runs over the lists of all of them: a reply that fits is whole, and no
  answer is cut while room remains.

A cut reply carries two fields, declared once in `tools/_base.py`:

- `cut`: by the JSON Pointer of each list it cut, `{total, returned, offset?,
  next?}`. `offset` is the index of the first entry the reply holds, absent
  for 0; `next` is the cursor for the entries after them, absent on the last.
- `stored_answer`: `{id, path, snapshot}`. `path` is the stored JSON file,
  whose `answer` is the complete answer; `snapshot` is what it was computed
  from, such as a file's `sha256`. When the file cannot be written (a full
  disk), the reply is still sent, cut as it would be: `stored_answer` holds
  only `not_kept` (`{code: "stored_answer_not_kept", message}`, saying why
  and to ask again with a larger budget, a narrower request or, for a
  waveform, the CSV format), and no `cut` entry carries a cursor. Every page
  object still says what it left out (`truncated`, with a `hint` saying no
  cursor reads the rest), and every count stands.

A stored answer's file has its expiry as its modification time, so deciding
which have expired reads none of them. A cut reply sweeps the expired ones
at most once every `presenter.SWEEP_INTERVAL_S` (10 minutes): it spends one
stat on the sweep, or, when one is due, one stat per stored answer.

**A list the server already keeps.** A list a record of the server holds
whole, a job's run rows in its record, is a second kind of source. When it is
cut, nothing is stored for it: its cursor names the record and an offset, and
a page is read from the record when it is asked for, so it carries the record
as it stands then (for run rows, the job's `status` and `progress`) and a
caller can tell the rows may have moved since the first page. A reply whose
cut lists are all records carries no `stored_answer` and writes no file.

**Over the target.** When the parts of a reply that are never cut already
exceed the target, a reply on the presenter goes out over it and says so
once, in a `budget_not_met` observation naming what it cut. No note
says that nothing was dropped while a `cut` field is present.

A cursor names the stored answer, the list's pointer and an offset. Reading
it slices the stored file: nothing is computed again and no request is
replayed, so a page is never refused as stale. It carries `stored_answer`, so
the caller compares the snapshot with the file to tell whether the page
predates a change. A page holds the list it reads from the offset on, cut to
the target the same way, and what identifies it: `page_of`, the pointer of
that list, so a field the page leaves out is read as given by the first reply
and not as absent; `cut`, naming only that list (where the page starts, how
many it holds, the cursor for the rest); and `stored_answer`. Of the rest of
the answer it keeps only the objects on the way to the list, each with its
text, number and flag fields and the objects its schema requires in their
least form (`presenter.page_skeleton`): no other list, no observations, no
notes the first reply gave. Each other list is read on from the first
reply's own `cut` entry. The rows of a job read with `jobs(action="runs")`
are the precedent: the rows and their counts. A cursor for an answer that has expired or is missing is
refused with `invalid_cursor`, saying the stored answer is gone and how to
compute a fresh one. The tool's `hint` names three ways to read the rest: the
stored file, the call with the cursor, and the same call through the Python
API.


**Presentation fields.** Fields that control how a receipt is rendered, rather
than what runs, are excluded from the idempotency fingerprint, so repeating a
request at a different verbosity replays it instead of conflicting. On
`run_experiments` these are `provenance`, `run_fields`, `budget`,
`execution.wait_s` and `analyze.include.fields`.

**Tool listing.** `tools/list` returns each tool's definition without its
`outputSchema`, so a client learns a reply's shape from the reply. Each tool
has a `title`, the short label a client shows a person instead of the wire
name.

**Two listing modes.** `[tools] listing` (or `LTSPICE_MCP_TOOL_LISTING`) sets
how much of each definition is published. `full` publishes every field
description as written. `compact`, the default, publishes the same tools and
schemas with every per-argument description removed; structure, enums,
defaults, `required` and `$defs` are unchanged, so a client can still build a
valid call. `inspect(kind: "capabilities")` reports the session's mode as
`tool_listing`. Both listings are static; they are the same for every
connection and no call changes them. The server accepts the same calls in both
modes, and dispatch, validation and responses are identical.

**Text block.** A reply's one text block is its `structuredContent`
serialized as compact JSON, as the MCP specification recommends ("a tool that
returns structured content SHOULD also return the serialized JSON in a
TextContent block"). A failed call's text is its `{error, code, hint?,
suggestions?}` that way, so a client that shows the model only the text of
an error, as Claude Code does, still gives it the code and the recovery hint.
An image block comes ahead of it, as Anthropic's vision guidance places
images ahead of the text about them, and a reply with no `structuredContent`
keeps its text. A client that shows the model both channels sends the data
twice; the specification accepts that cost.

Three recipes, `periodic`, `return_loss` and `noise_integral`, are published
in both modes as only their `metric` value and a one-sentence description that
names their output fields and points to `inspect(kind: "reference")` for
their arguments; they accept every argument the reference lists. On the
compact listing, any branch whose published properties are all fixed values,
these three included, keeps its description, because it has no other
structure.

On the compact listing, `inspect(kind: "reference")` returns the omitted
descriptions one entry at a time. It covers the branch vocabulary (recipes,
ops, job actions and the rest) and each tool's top-level arguments, indexed as
one entry per tool under the family `argument`, so `all_steps` or
`expected_sha256` is found the same way as `stability`. Neither mode strips
tool descriptions, so `analyze_results`' description lists the recipes and
`inspect`'s names the reference lookup in both.

**Listing freshness.** `tools/list`, `resources/list`,
`resources/templates/list` and `prompts/list` are marked fresh for one hour
and `private` (`ttlMs` / `cacheScope`, SEP-2549). `resources/read` has no such
hint. These fields exist from the 2026-07-28 revision on; clients on older
revisions do not receive them.

---

## 3. Tool contracts

### 3.1 `run_experiments` — effectful fan-out

**Submission and idempotency.** `request_id` is optional. When it is omitted,
the server mints one and returns it on the receipt. A minted id works like an
explicit one. `jobs` and `analyze_results` address the run by it, and
resubmitting the same payload with it replays the job.

An explicit `request_id` is the durability key. At submission the server
persists `{request_id -> job_id, fingerprint}`, where the fingerprint is the
sha256 of the canonical (sorted-key) input payload. The same id with the same
fingerprint returns the existing receipt with `replayed: true`; that field
describes this call, and the replay adds nothing to the job's record. A record
whose owner has exited is read as `interrupted` and written back, as any read
of the job leaves it. The same id with a different fingerprint is an
`idempotency_conflict` that names the job created under that id, which `jobs`
and `analyze_results` read by its `job_id`. Ids are scoped to the server
working directory's job store, and an id is deleted with its job ("Job model"
below says when). If the client cancels a `run_experiments` request while the
call waits for results ("Receipt, then dwell" below), only the wait ends; the
job keeps running.

**Circuit paths are resolved at submission.** Each circuit's path is resolved
before a job exists. A path a sandbox refuses fails the call with `path_denied`
and the sandbox guidance, and a file that is no circuit (by its extension)
fails it with `unsupported_file`. Either way no job is made, even when other
circuits of the call are allowed. A circuit that fails after that (a file that
is gone, an export LTspice refuses, an include that cannot be staged, that is,
copied into the job's folder) fails its own cases and leaves the call's other
circuits to run. Reading a case that has no raw reports the same failure as
the case's receipt row.

**Lock the id, then stage.** A submission takes the gate for its
`request_id` (a file lock, so it works across processes sharing the working
directory), looks the id up, and stages decks only if nothing is recorded
under it; it then writes the index and the job record and releases the gate.
A second submission with the same id waits on the gate, then replays what it
finds or is refused as a conflict, without staging decks. Each `request_id`
therefore gets one set of decks. A submission that stages and then fails (a
prepared case count that differs from the count the variations expand to,
reported as `completeness_mismatch`, or a store that cannot be written)
removes the `runs/{job_id}/` tree it copied before returning the error.

A process that dies after staging and before writing the record leaves no
index entry, so the next submission with that id stages again, and the files
the dead process copied under `runs/{job_id}/` remain with no record naming
them. An artifact belongs to a job only when a job record names it. Its
location in the runs folder does not make it part of a job.

`allow_live_includes` is not a presentation field. A job that reads includes
in place (live includes, see "Input snapshot and provenance") has inputs a
stored fingerprint cannot verify, so it is not replayed.
Reusing its `request_id` is an `idempotency_conflict`, and a new `request_id`
runs the experiment again.

**Replay is scoped to the simulator build.** A request does not name the
simulator executable, so the job records the program its cases launch
(`simulator_executable`: path, size, modification time, SHA-256). A replay
compares it with the program the request would launch now, resolved as a
fresh submission resolves it, so a changed default simulator also counts. A
different build is an `idempotency_conflict` naming both programs, as a
changed deck is. Builds are compared by digest, or by path, size and time when
either side has no digest. A job record with no executable, such as one
written by an earlier release (store version 2), is refused the same way.
`jobs(status)` still reads the recorded job, and a new `request_id` runs the
request on the current build.

**Selecting a build.** `execution.simulator` takes a family (`"ltspice"`,
`"ngspice"`, `"qspice"`, `"xyce"`), which runs that family's own executable
(`[simulator] path`, or the detected install), or a family and a name
(`"ltspice:xvii"`), which runs an additional executable listed in
`[simulator.executables]`:

```toml
[simulator.executables]
xvii = "C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"   # family from the file name
"ngspice:dev" = "/opt/ngspice-dev/bin/ngspice"          # or written in the key

[simulator.executables.ltspice]                          # or a table per family
lt24 = "C:/Program Files/ADI/LTspice/LTspice.exe"
```

`LTSPICE_MCP_SIMULATOR_EXECUTABLES` takes the same entries as `name=path`,
separated by `;` on every platform (`:` already separates a family from a name
and follows a drive letter). A name consists of lower-case letters, digits,
`.`, `_` and `-`, starts with a letter or digit, and has at most 32
characters. The family is inferred from the file name, as for
`[simulator] path` (`XVIIx64.exe` and `LTspice.exe` are LTspice); a file name
that does not identify a family needs `family:name`, and one that identifies a
different simulator than its key is refused. Every family accepts named
executables, and the `[simulator] enabled` allowlist applies to them; a run on
a named QSPICE build is refused off Windows, as one on the family's own build
is (see "Simulator choice" below). The server reads and checks the entries
when it starts, as it detects simulators then. An entry is left out, with a
message in the startup diagnostics, when its name is invalid, its family
cannot be determined or is not enabled, its file does not exist, or its
selector appears twice; a run naming a left-out entry is refused with a list
of the accepted names. `inspect(kind="capabilities")` lists the accepted
entries under `named_executables`.

A named build is parsed and linted as its family. A job records the
selector its run resolved to (`simulator_selector`; the default's selector
when the request named none), beside the class name it ran (`simulator`, which
resume reads) and the executable's identity (`simulator_executable`, which
tells two builds apart byte for byte). Every reply that names a job names its
simulator once, at the level of the job, as `simulator`, and the value is the
selector: `"ltspice"`, or `"ltspice:xvii"` for a named build. Passed back as
`execution.simulator` in the same session, it runs the same executable. The
selector is a name in the configuration of the session that ran the job, and
is the same on Windows, WSL and Wine for the same configuration. A reply adds
`simulator_notes` when there is something to say about it:
`simulator_not_in_session` when the session answering has no simulator under
that name (a run naming it would be refused here); and, for a record written
before the selector was recorded, `simulator_build_not_recorded`, with the
recorded `executable`, when its executable is the path of none of the
session's simulators of its family, so only the family is given. Such a
record whose executable is the path of the family's own simulator, or of a
named one, replies with that simulator's selector. The notes are lookups in
the session's tables: no reply reads or hashes an executable to decide them,
and `provenance` carries `simulator_executable` for a caller that needs the
exact build. `inspect(kind="capabilities")` gives `default_simulator` as a
selector too, beside `simulators` (keyed by family) and `named_executables`
(keyed by selector). Each build has its own limit on concurrent runs, so each gets the
full `[simulation] max_parallel` (`limits.max_parallel_sims` in
`inspect(kind="capabilities")`), as each family does. A named LTspice build
exports a `.asc` with its own executable, and with a sandbox set, a deck run
on it may include files from that build's own model library (LTspice XVII's
`Documents\LTspiceXVII\lib`, LTspice 24's `%LOCALAPPDATA%\LTspice\lib`),
which are staged like any other include.

The selector is part of the request. A `request_id` replayed with a different
name is a different payload, and a name that points to a different executable
than it did at submission fails the executable check above; both are an
`idempotency_conflict`.

**Control token.** The receipt contains an unguessable `control_token`. The
owning process (the server process that runs the job), or any caller
presenting the token, can cancel the job, so a reconnected session can cancel
its own work. Only the original submission response and a replay with a
matching fingerprint return the token; knowing the full original payload
proves ownership. `jobs` status, wait, list and runs output never includes it,
so read access never grants control. Receipts of an ordinary job omit it once
the job is terminal. Receipts of a recoverable job, and of each job a resume
creates from it, keep it in terminal states because it authorizes resume; a
resume that creates no job returns no new token.

**Receipt, then dwell.** The receipt is durable (the job is registered and
its record written, so a cancel sent from then on is honoured, and its decks
are staged) before any case is submitted and before any waiting. The call then
waits for the job, for at most `execution.wait_s` seconds (default 60, capped
at 120); this wait is the dwell. If the job becomes fully terminal (runs and
attached analysis) within it, the terminal results are returned inline;
otherwise the receipt is returned with `outcome: "in_progress"`.

Input:

```
request_id           str, optional      idempotency key; minted when omitted
circuits             list[{path, id?}]  .cir / .net / .sp / .spice / .asc. An
                                        id the caller writes must be valid and
                                        unique or the call is refused; with
                                        none, the file stem is made valid and
                                        unique (amp.v2 -> amp_v2; two amp
                                        files -> amp, amp-2) and a
                                        circuit_id_derived observation names
                                        the id it ran as
variations           list[Variation]    Appendix A.1. assign entries combine by
                                        cartesian product; AT MOST ONE random
                                        entry per circuit; entries whose applies_to
                                        name different circuits run side by
                                        side. [] = one plain run per circuit
execution            {wait_s?, run_timeout_s?, job_deadline_s?, max_parallel?,
                      simulator?: "ltspice"|"ngspice"|"qspice"|"xyce"|
                                  "<family>:<name>", recoverable?: bool,
                      simulator_seed?: int, keep_partial_raw?: bool,
                      results_beside_sheet?: bool}
analyze              {recipes: list[Recipe], group_by?, step?, all_steps?,
                      plot_index?, dialect?, include?}
                                        attached analysis stage; `recipes` is
                                        the Appendix A.2 union `analyze_results`
                                        takes, and `step`/`all_steps` mean what
                                        they mean there, so an attached
                                        measurement and a standalone one read
                                        the same `.step` iterations. The
                                        block's one source, the experiment,
                                        is labelled `experiment`, the only
                                        label a recipe's `sources` may name.
                                        Every recipe is validated at
                                        SUBMISSION, as `analyze_results`
                                        validates it against its sources,
                                        before a deck is staged; an invalid
                                        one fails the call with
                                        `recipe_invalid`, stage `validate`,
                                        not retryable, and a set
                                        `include.per_run.cursor` is refused.
                                        The schema here lists the metric
                                        names; `analyze_results` documents
                                        each metric's fields.
lint                 "block"|"warn"|"off"  (default "block")
suppress             list[rule_id]
allow_live_includes  bool (default false)
provenance           bool (default false)
run_fields           list[dotted row path] | null
budget               int | null            approximate response-token cap
```

`budget` and `analyze.include.fields` control how the durable receipt is
rendered and do not change what runs. `budget` is the reply's target, at
least 500 estimated tokens; with none, `[analysis] default_budget` (default
4000) is the target. The receipt goes through the presenter (§1, "A reply too
long to send whole"): its run rows are cut as rows to fit, at least one always
stays, and the rest are read from the job's record.

The attached analysis is an answer of `analyze_results`: the complete answer
a later `analyze_results` call with the same request over the finished job
gives, which the job keeps with its rows unprojected, rendered in the row view
`analyze.include.fields` of each delivery. `analysis.result` is declared with
that tool's output schema, and the receipt presents it beside its own answer
under one target (§1, "Several answers"): its lists are cut as that tool cuts
them, with its caps, words and `rows_not_shown`, and stored, when cut, as an
answer of `analyze_results`, whose cursors that tool reads with `continue`.
Its inline `waveform` samples are thinned as a series (§1). Nothing the
request asked to include is dropped, and no `budget_truncated` observation
is made: a cut is said by `cut`. The job's record keeps it at a version
(`analysis_snapshot.ANSWER_VERSION`); `analysis_snapshot.classify` alone
says whether this build renders a stored analysis: `snapshot`, or why not,
`earlier` (an earlier release stored it), `later` (a later release did) or
`damaged` (a record of this version with no answer it can render). One not
rendered leaves the receipt, its status and its run rows answering as for
any job, and `analysis.error` with an `attached_analysis_not_rendered`
observation say which, and to run `analyze_results` over the job.

**Simulator choice.** `execution.simulator` names one of the four supported
families, or a named executable of one (above); the schema's pattern accepts
all four. A name is accepted when it is detected (for a named executable,
when its entry was accepted at startup) and this host can run its family.
`inspect(kind="capabilities")` reports this per simulator and per named
executable as `selectable`, with a `refusal` giving the reason when it is
false, and names the default by its selector (`default_simulator`). A job's
replies name the simulator it ran with as that same selector (`simulator`;
"Selecting a build" above). An undetected family gets a "not available" error naming what was
detected, and a detected family this host cannot run gets the reason. QSPICE
is the only such family, because it runs only when the server itself runs on
Windows, and not under WSL or Wine. Both refusals happen before anything is
staged (`commit_state: "not_started"`). The same check applies to the server
default when no simulator is named.

The choice also sets the raw dialect used to parse results (QSPICE writes
`.qraw` with a double-precision frequency axis; Xyce's raw does not name its
writer, so the job's recorded simulator identifies it) and which lint rules
apply (the `C=`/`L=` arity rule applies to LTspice decks only). A `.asc`
schematic runs as the netlist LTspice exports in its own dialect, which is
adapted for ngspice and no other simulator. On QSPICE or Xyce a schematic
circuit therefore fails with `asc_export_unavailable`; use a hand-written
`.cir`/`.net`/`.sp`. The `measurements` recipe reads `.MEAS` results only from
LTspice and ngspice logs and cannot read QSPICE or Xyce measure output; read
those quantities from the waveform (`value`, `signal_stats`) instead.

**Run timeout.** A case has no time limit unless one is set.
`execution.run_timeout_s` on the request takes precedence over
`[simulation] run_timeout` on the server (`limits.run_timeout_s` in
`inspect(kind="capabilities")`, absent when unset). A timeout deletes the
partial result, unless the job is recoverable or `execution.keep_partial_raw`
keeps it for reads (below). A running job reports each case's progress
(below), which shows when a case is stuck. `[simulation] timeout`
(`limits.export_timeout_s`) bounds only LTspice netlist export.

**Recoverable execution.** `execution.recoverable=true` makes a job that
`jobs(action="resume")` can continue after failed, cancelled or interrupted
cases. Before the job is recorded, the server validates each case's deck,
with its variations applied, and saves the exact bytes of that deck and of
every file it includes. Recovery supports ngspice on Linux and native Windows,
and on native Windows one LTspice executable, LTspice 26.0.2, identified by its
SHA-256, whose startup behaviour has been checked; other LTspice builds and
other platforms are refused with `recovery_startup_unsupported`.

For LTspice, the server copies the settings file, `simulator.ltspice_ini` or
`%APPDATA%/LTspice.ini`, into the job's folder, and each attempt runs with its
own writable copy, so the original file is not changed. The file must come
from an LTspice that has been started and had its first-start prompts
answered, so that it holds LTspice's 20-digit installation identifier, an
analytics choice and the time of the last update check. That time must be
less than 15 days old at submission, at resume and at each launch, or the job
is refused with `recovery_startup_stale`. The server does not write a
timestamp or a consent choice itself and does not answer LTspice's dialogs.

Recoverable execution refuses includes read in place
(`allow_live_includes`), `.control` blocks, external simulator modules
(`.osdi`, `.pre_osdi`, `.load`, `.hdl`, `.verilog`, XSPICE `A` devices), file
readers the server does not stage (`sfile=`, `pwlfile=`, a PWL that names a
file in its points, and every file reader on a simulator whose data cards it
does not read), a data file read in place (`recovery_live_dependency`), directives
outside the supported set, functions the server does not know, and random
functions with no seed the server records, because a resumed case could not
be relied on to run them the same way. Every card of the deck and of each file
it includes is checked, and one refusal names the constructs refused, how many
there are, and up to twelve of them. Each is listed with its code and its first
place (the file it came from as the caller wrote it, the line and the card),
then up to five more places and a count of the rest. The error's `code` is that
of the first construct listed. On ngspice, where a seed would admit a refused
random function, the refusal names the two ways to set one, and lists any seed
option of the deck's own that leaves the draws unfixed, since that option
blocks both. An include read in place is named with the
reason it was not copied. ngspice runs with `-n` (no user `.spiceinit`) and
with `SPICE_SCRIPTS`, set only for that subprocess, pointing at a system
startup file in the job's folder that contains only a comment, so the
machine's startup files do not affect the run. A `pdk_native` sample
(Appendix A.1) keeps its recorded sample index and seed. A resume or a later
launch is refused with `recovery_execution_changed` if the executable, its
command line or the platform has changed since submission. A job without
`recoverable` runs the simulator with the machine's own startup files
(LTspice's `LTspice.ini`, ngspice's system `spinit`, which loads its code
models), except a user's `.spiceinit`: every ngspice run gets `-n`, so the
compatibility mode is the one `[simulator] ngbehavior` sets. The `spinit` can
still set another. After every ngspice run the server compares the mode the
log names for the deck (its first note, or the second when a driver sources
the deck) with the flags ngspice gives the mode it passed, asked once per
mode with no startup file. A run read in another mode fails with
`ngspice_mode_mismatch`. A run whose mode could not be checked carries an
`ngspice_mode_unverified` observation whose `evidence.reason` says why:
`no_mode_in_log`, `deck_note_missing`, `log_unreadable`, `ngspice_not_asked`
or `no_log`.

`execution.simulator_seed` runs each case's saved deck through a short ngspice
script that sets the random seed (`setseed`) immediately before it loads the
deck. It requires recoverable ngspice and an integer (not a float or boolean)
in `1..2147483646`; every case and retry starts from that seed. It cannot be
combined with a `pdk_native` variation, which sets its own seeds; a request
using both is refused. Without it, no seed is set, and a recoverable deck may
not call random functions, unless the deck fixes ngspice's seed itself: an
`.option`/`.options` card at the top level of the deck file, outside a
subcircuit and a `.lib` section, that sets `seed=` to a positive integer, with
every other seed option the deck reads fixed too. ngspice reads that card when
it loads the deck, before it evaluates a parameter, and reseeds as `setseed`
does (`inp.c:eval_opt` calls `com_sseed`), so the draws repeat on every run
and retry. That is how a `random` entry's `simulator_draws` cases run under
`recoverable`. Such a deck admits the same five random functions, and, since
it runs as an ordinary batch run, any number of analyses.

A seeded deck that sets ngspice's seed itself, with `seed=` on an option card
ngspice may read, is refused at submission, and a job recorded with one is
refused at resume and launch (`recovery_seed_unsupported`): ngspice reads that
card when it loads the deck, after the driver's `setseed`, so the deck's seed
wins and the run would record a seed it never drew from, or, with
`seed=random`, draw anew on a retry (measured on ngspice-42). A card in a
subcircuit no X line calls does not count, as ngspice drops that subcircuit
unread; one in a `.lib` section counts whichever section the deck selects.

A seeded deck may use only the random functions `agauss`, `gauss`, `aunif`,
`unif` and `limit`; `.control` blocks, functions the server does not know,
external modules and transient noise (`trnoise`, `trrandom`) are refused as in
any recoverable job. ngspice's compatibility mode (`[simulator] ngbehavior`)
still decides how the deck is read. Under `hsa`, `limit(nominal, variation)`
is statistical, while the default `kiltpsa` defines a three-argument clamping
`limit`. The seed does not change the compatibility mode. A seeded deck needs
exactly one `.op`, `.ac`, `.dc` or `.tran` analysis, because the seeding
script writes only the plot ngspice produced last; `.noise`, `.step` and more
than one analysis are refused, before the job is recorded, with
`recovery_seed_analysis_unsupported`.

`jobs(action="resume", job_id=..., resume_request_id=..., control_token=...,
case_ids=None, retry_failed=false, retry_cancelled=false, wait_s=0)` continues
a recoverable job. Each resume creates a child job, so a recoverable job and
the children its resumes create form a chain, and a resume must name the
latest job of that chain once it is terminal; naming an earlier one is refused
with `recovery_stale_parent`, and one whose process still runs it with
`recovery_owner_active`. The child has every case of the parent. Its retried
cases write run files under new names (`{child job_id}_case_{run_index}`), so
the parent's files are kept, and its successful cases reuse the parent's
results after their hashes are checked. Cases that failed with
`server_restarted` (the process running the job went away: it exited, or shut
down while they ran; earlier versions recorded a shutdown as
`server_shutdown`) or `job_deadline` are retried by default; other failures
need `retry_failed` and explicit cancellations need `retry_cancelled`. If
preparing the child fails, the resume is refused and nothing is recorded under
its `resume_request_id`, so the same request can be sent again. `case_ids` is
a set; a selection naming an unknown or already-produced case is refused.
Resume reads only the saved copies, checked against their hashes, so editing
or deleting the original circuit file does not make it run successful cases
again. A replay of an ordinary submission (same `request_id`) still checks the
original circuit file.

Keep the control tokens from the receipts of the recoverable job and of each
child, terminal ones included; status, list and runs never return them. The
job's `lineage` has `root_job_id` (the first job of the chain),
`parent_job_id` and `attempt_index` (0 for the first job, one more for each
resume). Each run's `attempt` has `execution_job_id` (the job whose attempt
ran the case), `attempt_index`, `run_token` (the base name of the case's run
files) and `reused` (true when the result was taken from an earlier attempt).
Resume returns `resumed`, `replayed`, the addressed parent and `head_job_id`,
the latest job of the chain. An empty selection returns the parent with
`resumed=false`, no child token, and a hint to analyze the job with
`analyze_results`. Reusing a `resume_request_id` with the same normalized
selection and retry flags returns the same child or no-op; a changed payload
is a conflict. A resume, and a replay of one, is authorized as a cancel is: by
the server process that runs the job or by its `control_token` (otherwise
`resume_not_authorized`).
`wait_s` sets only how long the call waits and is not compared on replay.

**Job model.** An experiment job coordinates cases across one or more
circuits. It is persisted in the server working directory's store
(`{working_dir}/.ltspice-mcp/experiments/`, or the working directory's store
under `LTSPICE_MCP_STORE_DIR` when that is set) as the job record (case
records, counters, analysis state, source manifests) and two indexes into it:
`by-request/` for the idempotency lookup and `by-circuit/` for
`jobs(list, circuit=...)`, which finds every job that ran a given circuit. Run
artifacts, staged decks included, are in `runs/{job_id}/`. A schematic runs
from a snapshot of its export in the store's `exports/`. With a sandbox set,
the snapshot is allowed because the schematic is, and its relative includes
resolve from the schematic's folder, as if it were beside the schematic.

A server process keeps at most 200 finished jobs. When it registers a new job
past that number, it deletes the oldest finished ones by start time, with
their records, their `request_id` index entries and the `analyze_results`
result sets that read them. A recoverable job is never deleted.

**Attached analysis is a job stage.** A job is terminal when all its runs and
its analysis are terminal; the status sequence is
`queued -> running -> analyzing -> completed | completed_with_failures | failed | cancelled | interrupted`.
`cancelled` means a caller cancelled the job. `interrupted` means the process
running the job stopped before its runs finished: it crashed or was killed, the
server shut down, or a script closed its `Api`. A job reads the same whichever
way the process stopped: its unfinished cases fail with `server_restarted`, an
attached analysis left pending or running fails, and the job and the analysis
each carry a `server_restarted` lifecycle observation. A process that stops
once every run has finished leaves only the analysis unfinished, so the job
ends `completed_with_failures` (or `completed` when it had no analysis left
and every run produced). A job's status is read from its record; whether this
server process runs a job is never decided from the process identity the
record names, which another process may have taken since.
The dwell and `jobs(wait)` wait for full terminality by default;
`jobs(wait, wait_for: "runs")` returns once the runs are terminal: once every
case's status is terminal, which can be just before the job's status changes.
A read does not wait for the job: each case is read once its own run has
finished, its status set after the checks made when its run ends
(`services.case_unreadable`), whatever its siblings are doing. A case still
running is refused with `run_not_finished`, naming what its simulator has
written so far. Cancelling
during analysis cancels the stage and keeps the run artifacts. An analysis
failure puts the job in `completed_with_failures` with an `analysis_error`; a
new `analyze_results` call can still analyze the runs. An analysis stage the
process running the job left unfinished is not re-run. The analysis result is
stored in the job record as one rendered page, and its summary is included in
every terminal receipt.

**Input snapshot and provenance.** At submission the server stages the
primary deck (content-addressed) and every include and lib it reaches (with a
sandbox set, those inside it or in the simulator's own library), to a
recursion depth of 8. Staged copies keep their relative layout. The manifest
records `{path, sha256, staged, live, staged_path, reason, section}` for every
reference, including `.lib` section selections. Runs read only the staged
copies, so editing an original afterwards does not change the job, and every
reported number cites the hash of the deck that produced it.

By default, a reference that cannot be staged fails that circuit's submission
with an `include_unstaged` error naming the reference. An include or library
card with a quote or bracket left open, or a closer with nothing open,
always does, at its line, since which file it names is not known. With
`allow_live_includes: true`, runs read those files in place, the manifest
marks them `live: true`, and an observation reports the weaker provenance.
Every live reference is listed.

The data files a deck reads are staged with it: on LTspice a source's
`wavefile=` (or `wavefile` with no `=`) and a PWL's `file=`, on ngspice a
`filesource` model's `file=`. A relative path is read from the deck's folder,
as both simulators read it, for a card in an included file too on ngspice; on
LTspice such a card in an included file is refused (`data_file_unestablished`),
since which folder LTspice reads it from is not recorded. The file is copied
byte for byte and the staged card names the copy by its absolute path, which
an LTspice run reports as not recorded (`data_path_unrecorded`). A data file
named by an absolute path is read where it is: the manifest marks it `live:
true` and an observation (`live_data_file`) reports it. A data file that does
not exist (`data_file_missing`) or lies outside `[security] allowed_paths`
(`data_file_outside_allowed_paths`) fails that circuit's submission before it
runs, naming the card, its line and where the file was looked for.

Provenance in the response is opt-in. Digests, staged paths, the full
manifest, the linter version and the job's `simulator_executable` are returned
only with `provenance: true` on `run_experiments`, or `include.provenance:
true` on `analyze_results` (which omits the executable). The lean receipt
keeps `{circuit, path, dialect, staged_files}` and every manifest
entry that is live, has a reason, or is otherwise not an ordinary staged
reference. An analysis's `source_hashes` has one row per source,
`{manifest_id, label, plot_index, dialect, snapshot_id}` by default; the
opt-in adds the file paths, presence flags and sizes, `job_id` and the
digests. The `provenance` flag also controls the receipt's
`analysis.request`, an echo of the caller's input. Nothing a caller must act
on is omitted.

**`.asc` ingestion** needs the LTspice exporter whichever simulator runs the
deck. The steps are: lock, export under the lock, hash, stage, simulate the
staged copy, and lint the exported deck. For ngspice the staged copy is
adapted first: the `.backanno` card is dropped, and the `§` LTspice puts in
an instance name that does not start with its element letter (`R§Load`) is
removed. Without an exporter, that circuit's cases fail with
`asc_export_unavailable`.

**Deck-native settings.** Temperature (`.step temp`), `.lib` section selection
and `.nodeset`/`.ic` are written in the deck; no tool parameter sets them. The
linter's `temp-as-param` rule blocks a deck that sweeps temperature as a
parameter and gives the `.step temp` form. Model-swap corners are a variation
(Appendix A.1).

**Completeness.** Every case gets a stable `case_id`. The counters are
`{declared, expanded, submitted, produced, failed, cancelled, skipped, reused}`,
where `declared` is the number of circuits in the request and `expanded` the
number of cases after the variations are expanded; once the runs are
terminal, `produced + failed + cancelled + skipped == expanded`. Every omitted
case is listed with its `case_id` and a reason. Lifecycle status is reported
separately from completeness. `reused` counts successful cases reused from an
earlier attempt; `submitted` counts this attempt's submissions.

`progress`, `{expanded, terminal, remaining}`, is computed from the
completeness counters stored in the job record and is on every receipt,
terminal ones included; `terminal = produced + failed + cancelled + skipped`
and `remaining = expanded - terminal`.

**Stopped cases.** A case the server stops is a failure row whose code
gives the reason, which is `run_timeout`, `job_deadline` or `cancelled`
(status `cancelled`). Its message names the bound that applied, and its
`evidence` contains:

- the bound itself: `run_timeout_s` with `run_timeout_source`
  (`"request"` or `"server_default"`), or `job_deadline_s`;
- the simulator's `exit_code`, and `simulator_exception` when the launch
  raised one;
- `log_failure_code` when the log names a cause the server recognizes
  (`convergence_failed`, `singular_matrix`, ...), with that cause's evidence;
- `log_excerpt`, the end of the log or the lines around its errors.

A stop kills only that case's processes, including a wrapper or launcher
script and the simulator it starts. If the exit is not confirmed within the
kill grace (10 s by default), the row is `kill_unconfirmed` instead, with
evidence `{stop_reason, run_timeout_s | job_deadline_s, kill_grace_s}`, and
the case keeps its place in the limit on concurrent runs until the process
exits. When the exit is confirmed, within the grace or later, one
`partial_progress` observation follows for that case, read from the partial
raw before cleanup deletes it. Its evidence is `{case_id, run_index,
raw_present, raw_kept, points, last_axis_value}`, plus `{raw_bytes,
header_complete, plot, axis, stepped?}` when a raw was read. `raw_present`
reports whether a raw was there when it was read, and `raw_kept` whether it
is kept; only a recoverable job keeps a stopped case's run files. `points`
counts complete records from the file length instead of the `No. Points`
header, because ngspice leaves that header at 0 until a plot ends and on
LTspice it lags the data. `last_axis_value` is in deck coordinates. `run_timeout`, `job_deadline` and
`kill_unconfirmed` rows include a recovery `hint`.

**Keeping a stopped case's raw.** `execution.keep_partial_raw: true` keeps the
raw of a stopped case instead of deleting it, cut to the points a full read
reaches. An LTspice raw keeps the points its header's count declares, which is
as far as LTspice's own reader reads it, because LTspice XVII's records past
that count need not continue forward in time. An ngspice raw keeps every
complete record. The `partial_progress` observation then reports
`raw_kept: true`, the failure row's `evidence` adds
`partial_raw: {points, axis, last_axis_value}` describing the kept raw, and
the run row's `raw` is its path. Reads of that case (`analyze_results`,
`plot_waveform`, `inspect(kind="results")`, the Python API's `load_raw`, and
its `measurements` and `device_op_points` of the case's log) read it as they
read a produced case's raw. Each read also returns a `partial_raw`
observation, `{case_id, stop, points, axis, last_axis_value}`, and every value
it reads ends at the last kept point. A raw with no point to keep, a text
(`Values:`) raw and an LTspice Fast Access raw are deleted even with the flag.
The flag cannot be combined with `recoverable`, since a resumed job runs its
stopped cases again.

**Running cases.** While a job runs, receipts from `run_experiments`,
`jobs(status)` and `jobs(wait)` contain one `run_progress` observation per
running case, with the `partial_progress` evidence plus `running_s`, the
seconds since launch. It is read from the raw the simulator is writing when
the receipt is built, and is never stored in the job record. If `points` does
not change between two reads while `running_s` grows, the case is not
advancing; the caller decides whether to cancel it.

**A sheet's results.** A run writes its results to the job's own folder, never
beside the sheet it ran, where LTspice's window opens a sheet's results. A
receipt of a job that ran an `.asc` in LTspice contains one
`results_not_beside_sheet` observation naming the sheets of its produced cases and
`jobs(action="place_beside_sheet")`, which copies a case's results there.

`execution.results_beside_sheet: true` does that copy as part of the run.
Once every run has finished, each produced case's results are copied beside
its sheet under the sheet's name, replacing any there, as the jobs action
copies them, before the job reaches its final status, so `jobs(wait)` returns
after the copy; the job folder keeps its own, which every read of the job
takes. The receipt then has one
observation per sheet instead of `results_not_beside_sheet`:
`results_placed_beside_sheet`, whose `evidence` gives the `sheet`, the
`case_id`, the `files` copied and any `assignments` the sheet does not show,
or `results_not_placed`, whose `evidence.reason_code` is the code the jobs
action would refuse with (`sheet_changed` when the sheet was edited during
the run), `sheet_locked` when another server process held the sheet, or
`copy_failed`; the last two name the action that copies the results once that
is fixed. Neither observation costs the job its status. A cancelled job
copies nothing, and a case that produced nothing leaves its sheet alone. Nor
does a job whose process stops before the copy, even once every run has
finished: the copy waits on each sheet's lock and copies whole raws, which can
outlast the bound a stopping process gives its jobs. When the job's own
coordinator stops it after its runs, each sheet whose case produced then has a
`results_not_placed` with `reason_code` `server_restarted`, naming the action
that copies its results. A job settled without its coordinator (one whose
process crashed, or whose stop outlasted that bound) keeps no record that a
copy was asked for, so its receipt carries only the general
`results_not_beside_sheet` note, which names the same action.
A netlist's results are copied beside it the same way, under its name, where
LTspice's batch run writes them; its `results_placed_beside_sheet` adds
`evidence.not_checked`, saying that what LTspice's window does with results
beside a netlist has not been checked. The same list says when a circuit was
not compared with the file that ran because the job holds no digest for it.
The request is refused before it runs (`commit_state: "not_started"`) when the
capability table does not put the job's simulator's results beside a circuit
(`not_ltspice`); when one results file, counted by its resolved path with
case folded, would receive more than one case, by a variation, by one circuit
named twice or by two circuits of one name such as `foo.asc` and `foo.net`
(`several_cases_per_sheet`, naming the circuits and the file): a results file
holds one run's results, so the caller copies the case it wants afterwards;
and when a netlist has a sheet of its name beside it, in the job or not
(`results_belong_to_sheet`): that results file is what LTspice's window shows
as the sheet's run, so the caller runs the sheet or copies the netlist under
another name. It
cannot be combined with `recoverable`, since a resumed job's runs would
finish without it. Left false, it is not part of the request's identity.

**RunRecord** is a schema fragment shared by `run_experiments` and `jobs`:
`{case_id?, run_index?, circuit?, assignments?, status?, raw?, log?,
other_raws?, simulator_version?, diagnostics?, diagnostics_coverage?,
failed_measurements?, failed_measurement_steps?, measurements_not_established?,
measurements_not_printed?, measurements_coverage?, steps?, attempt?}`. Every key is optional because a
run-field projection can remove any of them. `status` is the case's own
status, which uses a separate vocabulary from the job's `status` on the
receipt. It is `produced` once the simulator wrote its output, `failed`,
`cancelled` or `skipped` when the case ended without it, and `queued`,
`submitted` or `running` before. A job whose cases all ran is `completed`
while each row has status `produced`; the output schema lists both
vocabularies. `other_raws` maps each kind of other raw the run wrote beside
its raw to its path (`<run>.<kind>.raw`). The kinds are `op`, a transient's
operating point, and `fra_1`, a frequency response analysis's results. Only
kinds recorded from LTspice are listed, and a run with no raw lists none.
Every read by job and case accepts `other_raw`, which reads one of them in
place of the case's raw. The reads that accept it are a job source of
`analyze_results`, `run_experiments`' attached `analyze` block,
`plot_waveform`, `inspect(kind="results")` and the Python API's `load_raw`. A
case whose run wrote no raw of that kind fails with `other_raw_not_written` (a
missing case in `analyze_results`). When `plot_waveform` with `in_ltspice`
opens one of these raws in LTspice, it opens the raw on its own rather than
from the case's sheet, so the results LTspice shows for that sheet are not
replaced. `simulator_version` is the build
the run reported in its own output. It is read from the LTspice log banner
(`LTspice 26.0.2 for Windows`), the Xyce log banner
(`Xyce Release 7.8.0-opensource`), the ngspice console banner
(`ngspice-42, Creation Date: ...`), or the raw header's `Command:` when there
is none of those (LTspice XVII, QSPICE). It is absent when the output names no
build. It is recorded per case, so an executable replaced during a job shows
as two builds. `diagnostics` is what the simulator said about the run, read
from its log and console capture when the run ended: one entry per condition
and channel,
`{code, count, first_text, channel, severity?}`, with `severity` only where
the simulator wrote one (the strongest word over the condition's
occurrences), and lines the server does not recognise counted under
`unclassified` (`docs/DESIGN.md`, "Diagnostics"). It does not change the
case's `status`: a run the simulator warned about and finished is still
`produced`. `diagnostics_coverage` is present only when that reading is not
the run's own simulator and build: `build_unrecorded`, `producer_unknown`,
`unverified`, `log_unread`, or `not_classified` for a case recorded before
logs were classified. The `summary` recipe carries the same two fields for
the same run. The measurement fields say what the run did not give of the
`.meas` its deck requests, read from its log when it ended:
`failed_measurements` names those LTspice printed as failed,
`failed_measurement_steps` gives the steps (from 0) of one that failed at some
steps of a stepped run only, `measurements_not_established` the steps whose
printed value is not established as the result (LTspice XVII prints 0 for a
failed step), and `measurements_not_printed` the requested ones the log
printed nothing for. `measurements_coverage` is `not_read` for a case recorded
before these were kept, `log_unread` when the log's measurements could not
be read, and `simulator_unsupported` for a QSPICE or Xyce run, whose measure
output the server does not read; the fields left out otherwise mean every
requested `.meas` gave a value. Every other reader of the run's measurements
gives the same reason (`services.measurements_unread`): the `summary` recipe
reports none and says why once in a `measurements_not_read` observation, and
the `measurements` recipe and the Python API's `measurements` refuse, naming
it. When a row names one, the receipt's hint names the `measurements`
recipe, and a `meas_failed_in_every_run` observation names each `.meas` that
gave no value in any run. Runs are rendered as `items: [RunRecord]`, each row an object
with the same keys at every budget.

A receipt is the job's complete answer at a moment, every run row in it
(`receipts.render_receipt_snapshot`, which knows nothing of a wait). Its run
rows are the one list the presenter cuts; `source`, `lint`, `artifacts`,
`failures` and `observations` are sent whole. When the rows are cut, the
reply's `cut` names `/runs/items` with `{total, returned, next}`, the `runs`
block's `returned`, `truncated` and `next_cursor` say the same, and the hint
names `jobs(action="runs", job_id, cursor)`, which reads the rest from the
job's record. At least one run row always stays, and a row is sent whole or
not at all.

`steps` says how a stepped case's results are stepped: `{count, parameters}`,
the number of steps its raw's first plot holds and the names of the stepped
parameters (never their values, which `analyze_results` and `plot_waveform`
read from the results). It is decided once, when the run ends, by the raw
reader: the raw's header marks a plot stepped, and the log's `.step` lines
give each step's values (`decoded_raw.steps_held`, kept on the case's record
as `ExperimentCase.steps`). Where the raw and the log disagree, or where each
step starts cannot be read, `steps` is `{not_established}` with the reason and
no number. A case whose results are not stepped carries no `steps`; a case
with a raw whose steps were not recorded when its run ended (recorded before
the server kept them, recovered after its server stopped, stopped with its
partial raw kept, one whose raw could not be read when its run ended, or one
whose end-of-run checks are still being made) carries `{not_recorded: true}`,
which `runs_not_shown` does not count. A failed read records nothing: like the
log read at the end of a run, which keeps the failure as `logs_error`, it is
not made again, so a file still locked for a moment after the simulator exits
(as on Windows) leaves its case not recorded. A stop of the job (a cancel,
its deadline, the process shutting down) ends the read at once, with nothing
recorded. The kinds a record names are `result_kinds.STEPPED_KINDS`, each
established by the recordings it lists, per LTspice build. ngspice has no
`.step`: its runs record one step. Every chart of a job's case, of one case
(`run_index`, `case_id`, or `runs` naming one) or of several, gives `n_steps`
from the plot it drew, and where this record gives another number a
`steps_not_as_recorded` observation names the case with both
(`tools/analysis._steps_not_as_recorded`, the one comparison). A chart by
`path` names no case and has no record; a chart of a `plot_index` past the
first or of an `other_raw` draws a plot this record does not describe, so
neither is compared.

**What the rows left out carry.** When run rows are left out, `runs_not_shown`
states the facts on the rows after the ones sent, from the job's whole table
whatever `run_fields` keeps: per kind (`diagnostics.<code>`,
`diagnostics_coverage.<value>`, `failed_measurements`,
`measurements_not_established`, `measurements_not_printed`,
`measurements_coverage.<value>`, `steps_not_established` for a run whose
steps were not established when it ended, and `status.<status>` for a run that ended
`failed`, `cancelled` or `skipped`), `{counts, runs?}`. `counts` gives every
kind's number of runs. `runs` names at most 3 runs over all kinds, the first
run of each of the first kinds, each `{case_id, run_index, kinds, cursor}`
with a record cursor that starts at its row; a kind every run of the job
carries, shown or not (a coverage value such as
`measurements_coverage.simulator_unsupported` on a QSPICE job), is given by
its count alone. The whole note is held to one bound,
`receipts.NOT_SHOWN_BOUND_TOKENS` (692 estimated tokens), built by
`lib/not_shown.py`, the one builder of a part naming entries left out, whatever the number
of kinds and runs, and rows are sized against the target less what the note
reserves (its size at one row, a first estimate). The whole reply at the
count found decides: the note can be larger there, and while the reply is
over, a row comes off. When no count fits beside the reserve, counts are tried
whole; a reply says it is over its target only when one row is. Rows keep their
order, so a cursor names a row by its index. It is absent when no row is left
out or none left out carries such a fact, and it is never cut and never
shortened. A receipt with no job (a routing failure) has no record: its cut
rows are stored, its `runs` block names the stored file, and its failed runs
are left to its `failures`, which name each one. Fact channels, completeness, progress, verdicts and the
ids, token and cursors a caller needs to continue are never cut. If they and
one run row exceed the target, the reply goes out over it with one
`budget_not_met` observation. The attached analysis's lists are cut beside the
run rows, each to at least one entry, and read on with `analyze_results`.

**Standing notes, said once.** Some notes describe the job or the request and
read the same on every reply about it: the `micro_sign_folded`,
`ngspice_mode_unverified`, `results_not_beside_sheet` and
`value_read_otherwise` observations, each
`lint` entry and `source` entry, and the hint's route to the `measurements`
recipe (when a row names a `.meas` that gave no value, or, on a terminal
receipt, when the deck has `.meas` cards, the request did not attach the
recipe and the simulator's measure output is read). Whether the deck has
`.meas` cards is read from the facts the job keeps for each run, never from
the deck file; a job recorded by an earlier release kept them only beside a
`.meas` that gave no value, so its receipt names the route only then. The session remembers each one it has sent, by job, code and a digest
of its text (`lib/said_once.py`; bounded, never written to disk). A reply that
repeats one sends it as its code and `said_before`, the number of replies that
carried it before, in place of its text: an observation as `{code, kind,
said_before}`, a `lint` or `source` entry as `{circuit, said_before}`, and the
route as a shorter sentence. The hint then names `jobs(action="status",
job_id)`, which sends every note in full. A note whose text changed is new and
is sent in full. Which deliveries shorten is one table: `jobs(action="status")`
and a replay by `request_id` never do; a `wait`, a `run_experiments` receipt
and a `resume` do. Each note's schema has two branches: the whole entry, with
every field it always carries, and the short one, with `said_before` and
nothing of its text. A new
session starts with nothing remembered. The Python API sends every note in
full; `raw_page=True` returns the MCP reply, shortened notes included. Failures, run rows and their diagnostics and measurement fields,
progress, and every other observation are never shortened.

**Deliveries of one answer.** The run receipt, a `wait` and a `status` are
deliveries of one job's answer. What a delivery adds is marked in the output
schema with `x-delivery`: `replayed` on a run receipt, `action` on a `jobs`
reply, `timed_out` on a wait, and on a resume `resumed`, `replayed`,
`resume_request_id`, `addressed_parent_job_id` and `head_job_id`. Set those aside, and the run receipt, the
last wait and a status of a finished job carry the same answer, read in a
session that has sent none of its standing notes. A wait holds nothing that
can stop its job: a cancelled wait leaves the job running.

The lean row flattens `value` to its scalar leaves and keeps any `artifact`
handle, both in `analyze_results` and in a `run_experiments` receipt. A
`waveform` value is kept whole, less the run identity its row already
carries, and so is a `plot` value's scalar leaves.

Receipt failure rows with the same `code` and message (numbers in the message
ignored) are merged into one row. That row is the merged failure whose
`case_id` sorts first, not the first to finish, so two identical runs report
the same excerpt. `count` is the number of failures merged, and `case_ids`
lists at most 10 of their case ids.

Output: `job_id, request_id, control_token, job_type, status,
analysis_status, simulator, simulator_notes?, dialect, outcome, source[],
completeness, progress, lint
findings (per circuit), runs (a page of RunRecords), analysis?, failures[],
observations[], artifacts[], hint, cut?`. `simulator` is the selector the job
ran with, given once for the job: neither a `source[]` entry nor a run row
repeats it ("Selecting a build" above; the source record keeps the class name
on disk).

### 3.2 `jobs` — job status and control

```
{action: "status", job_id | request_id}
{action: "wait",   job_id | request_id, timeout_s (default 60, capped at 300),
                   wait_for: "all" (default) | "runs"}
{action: "cancel", job_id | request_id, control_token?}
{action: "resume", job_id, resume_request_id, control_token?, case_ids?,
 retry_failed?, retry_cancelled?, wait_s?}
{action: "list",   circuit?: path, limit? (capped at 50), cursor?}
{action: "runs",   job_id | request_id, cursor?, run_fields?}   cursor absent = first page
{action: "place_beside_sheet", job_id | request_id, case_id?}
```

An action on one job takes `job_id`, `request_id` or both; when both are given
and name different jobs, the call fails with `selector_conflict`.

Each action accepts only its own fields and rejects others; every action also
accepts `budget`. The input schema is one flat object, because Claude's API
refuses a tool whose input schema has `oneOf`, `allOf` or `anyOf` at its top
level and a client then drops the tool. The tool's description lists each
action's fields and marks the ones an action requires, generated from the
action models, so the compact listing, which drops argument descriptions,
still says which action takes which field. A field two actions describe
differently names each action before its sentence. The tool refuses a field
the action does not take, naming both. The output shape is discriminated on
the echoed `action`:

- `status` / `wait` return the §3.1 receipt **without** `control_token`, plus,
  for `wait`, `timed_out`. The receipt includes `progress`, terminal states
  included, and follows the same rules for its target, run rows and standing
  notes; `status` sends every standing note in full.
- `cancel` returns a page with one row per case that had not finished,
  `{case_id, run_index, prior_status, status}`.
- `resume` returns the receipt of the addressed parent or of the new child,
  with `resumed`, `replayed`, `resume_request_id`, `addressed_parent_job_id`
  and `head_job_id`. A receipt that created a child contains the child's token,
  even once the child is terminal. §3.1 covers selection and replay.
- `list` returns a page of circuit groups
  `{path, exists, last_activity, status_counts, interrupted_job_ids,
  recent_jobs, recent_jobs_total}`; with no filter, the recently touched
  circuits. `interrupted_job_ids` lists the circuit's jobs with status
  `interrupted`. `recent_jobs` gives at most 5 of the circuit's jobs, newest
  activity first, each `{job_id, status, request_id, simulator,
  simulator_notes?, finished_at}`, and `recent_jobs_total` counts them all.
- `runs` returns the job's `RunRecord` rows, read from its record when the
  call is made, with the job's `status`, `progress` and `simulator` (with any
  `simulator_notes`) at that moment. They
  are cut as rows to the target like a receipt's, at least one kept; `cut`
  names `/items` with its `offset`, and `next` is the cursor for the rows
  after. A receipt's `cut` cursor for `/runs/items` is the same kind of
  cursor. A cursor names the job and an offset, so it is refused for another
  job, and a page of a job still running may show rows that changed since the
  first page. `run_fields` projects each row, as on `run_experiments`; pass it
  again with each cursor.
- `list` is sent whole when it fits, else stored and cut as groups, and
  `cursor` takes a `cut` entry's `next` to read the stored list on. `limit`
  caps the groups one reply sends, at most 50.
- `place_beside_sheet` copies one case's results beside the circuit it ran,
  sheet or netlist, under its name, where LTspice keeps them
  (`sheet_results.files_beside`): the raw, the log, and each other raw the run
  wrote (`<name>.op.raw`). It returns `{case_id, sheet, files, assignments,
  not_checked, left_in_place}`, `sheet` naming the circuit and `not_checked`
  what the copy did not check, with
  one `{from, to, replaced}` per file copied; `assignments` is what the
  case's variations set, which the sheet does not show. `not_checked` always
  says that the files the circuit includes or uses were not compared with the
  ones the run read, since only the circuit file is. `left_in_place` names
  each of the circuit's result files beside it that this run did not write
  (an `.op.raw` an earlier run left, say): it is not removed, and
  `not_checked` says that what LTspice's window does with it is not recorded.
  `case_id` may be left
  out of a job that ran one case (otherwise `case_id_required`). The copy is
  made under the circuit's lock and then the results file's, held until the
  copy has ended even when the call is cancelled, and the job
  keeps its own. Every placement takes this one path
  (`sheet_results.place_checked`): this action, `execution.results_beside_sheet`
  and `plot_waveform` with `in_ltspice`. A case that
  produced no results (`case_not_produced`) or whose results are no longer
  there (`results_missing`), a case of a job whose runs are not all terminal
  yet (`job_not_terminal`: a job run with `execution.results_beside_sheet`
  copies its cases then, so a placement waits for it, although a read of a
  finished case does not), another simulator's run (`not_ltspice`), a netlist whose results
  file is a sheet's (`results_belong_to_sheet`), and a circuit that is missing
  (`sheet_missing`) or changed since the job read it (`sheet_changed`) are
  refused, and nothing is written. The sheet's `.plt` is left as it is.

`wait` blocks for at most `timeout_s`, up to 300 s per call. A timeout is not
a failure; the response has `timed_out` set and the job keeps running.

**Capped values.** `execution.wait_s` above 120, `timeout_s` above 300, a
`list` `limit` above 50 and a `per_run.limit` above 100 are served at the cap
instead of being refused. The response reports the cap as a warning on
`run_experiments` and `jobs`, and in the `hint` on `analyze_results`, which
has no top-level `warnings`. The other tools hold their caps the same way: an
`inspect` results `limit` above 100 or reference `limit` above 20 (in the
item's `data.warnings`), a `plot_waveform` `max_points` above 2000000 (in
`warnings`), a `run_code` `timeout_s` above 600 (in `warnings` and the
`hint`), a `waveform` recipe's `max_points` above the server's
`[analysis] max_points` and a `thd` recipe's `harmonics` above 50 (each in the
recipe's `warnings`), and a `verify_circuit` `render.scale` above 4 (in
`warnings` and the `hint`). The schema declares no `maximum` for these fields.

Only the owning process or a caller with a valid control token can cancel;
any other caller gets `cancel_not_authorized`. Once a cancel is acknowledged,
no further case is submitted, queued cases are cancelled and active ones are
killed. A wait on a job this server process does not run re-reads that job's
record every 0.5 s and returns the same output; a record whose owner has
exited is read as `interrupted`.

### 3.3 `analyze_results` — batched read

**Identity model.** Across the surface, `case_id` / `run_index` identify one
case of a job (one circuit with one set of variation values), `plot_index`
selects one plot in a raw file, `step_index` / `step_values` select one
iteration of the deck's own `.step`, and a sampled axis (time, frequency or
the swept source) lies within a step. A table plot, one whose first trace is
not a sweep axis (such as an `.op`, `.tf` or `.pz` result), has no sampled
axis. `snapshot_id` identifies the contents of a source's RAW, log and
console files as they were read. A path names one results file, so
`plot_waveform`, `inspect(kind="results")` and the Python API's `load_raw`
refuse a `run_index` other than 0, a `case_id` or an `other_raw` beside it,
and the API's `measurements` and `device_op_points` a `run_index` or a
`case_id`, in the same words, rather than read the file as though they had
not been sent. `analyze_results` refuses an `other_raw` beside a path in
those words too, and reports a run other than 0 of a path source as a
missing case.

Input:

```
sources    list[{job_id? | path?, runs?: "all"|[int]|{case_ids}, label?,
                 plot_index?: int|null, dialect?: str|null,
                 other_raw?: "fra_1"|"op"}]
                          path names a file the server did not run: a .log
                          (in any case) is read alone, any other file as a
                          RAW with the .log of the same name
                          other_raw (job sources only) reads each case's raw
                          of that kind, as its row's other_raws lists it,
                          instead of its raw; a case that wrote none is a
                          missing case with code other_raw_not_written
recipes    list[Recipe]   Appendix A.2; unique key; optional per-recipe
                          sources: [label]
group_by   list[assignment param | "circuit" | step-axis name]
step       {axis, value} | null   for a deck carrying `.step`: read the one
                                  iteration at that axis value. Default is the
                                  first. A number, or a string in SPICE
                                  notation ('2k'), picks the first step whose
                                  value is that number to the six significant
                                  digits a log prints a step value with
                                  (step_values.same_step_value); any other
                                  string must equal a step's value as text.
                                  An axis or a value the run does not have
                                  is refused with the ones it has
all_steps  bool (default false)   evaluate at every `.step` iteration; mutually
                                  exclusive with `step`
include    {per_run?: {limit? (capped at 100), cursor?} | bool, outliers?, signals_available?,
            provenance?, fields?: [dotted row path]}
budget     int | null
continue   {result_set_id, cursor}   reads on in a cut reply: a page of the
                                     stored answer; request fields sent with
                                     it must be the whole request that gave it
```

`continue` is the wire spelling; the Python attribute is `continuation`.

`plot_index` is null or a strict nonnegative integer; when it is omitted, the
first plot is selected only if a RAW recipe reads one. A source whose `path`
is a `.log` refuses plot selection. `dialect` is null or one of `ltspice`, `ngspice`,
`qspice`, `xyce`; an explicit dialect must agree with the raw file's header
and with the simulator the job recorded. `plot_waveform` (with `runs`
beside a `job_id`) and the attached `run_experiments.analyze` block accept the
same selectors and the same
`{axis, value}` step selector; `plot_waveform`'s `step` also takes a 0-based
index. Analysis type, axis and units come from the selected plot's header. A
recipe never gives a table plot an axis, so read its quantities through the
API (`RawResult.table()`) or `inspect(kind="results", view="table")`. CSV
export writes them without an axis column. Complex values outside an AC
analysis keep their real and imaginary parts rather than being read as AC
gain and phase.

A result set stores the request, each source's file hashes and the selected
plot, so a continuation reads the same plot. Each row of a RAW recipe reports
the `dialect` of the plot it read. Each `source_hashes` row reports the
`dialect` the caller gave, or else the one the job recorded, and has none
when neither names one, because the row is made without decoding the raw. Both include
`plot_index`. A `source_hashes` row includes `snapshot_id`; a default result
row leaves it, like `deck_sha256`, to `include.fields`, and its `source`
label names the `source_hashes` row that has it. `step_values` come from the `.step` lines of the run's own
log as read when the result set was created, string values included; no other
log is read for them. A run LTspice stores a point a step (a stepped `.op`,
`.tf`, or one-frequency `.ac` or `.noise`) keeps its step values in the raw
instead, under the same names: a stepped temperature is `temp` either way, as
the deck's `.step temp` names it, and `step` and `group_by` also take
`temperature` for it.

The `measurements` recipe reads the `.meas` results in each source's log once,
including a `.log` `path` source and a finished job case with no RAW. Its rows
have no `plot_index`. With neither `step` nor `all_steps`, a row holds each
measurement's statistics over every step and no step fields. With `all_steps` the recipe returns one row per step, and with
`step` the row of the step the selector picks, each with its `step_index` and
the `step_values` of the log's `.step` line for that step: row n of a stepped
`.meas` table is the step of the n-th `.step` line on both LTspice builds. A
log whose `.step` lines do not number its tables' rows is refused for `step`
and `all_steps`; LTspice writes no `.step` line for a stepped `.op`. Tables the simulator prints in its log are read through
`inspect(kind="results", view="native_tables")`, not through a recipe. In a
mixed request, a RAW recipe fails on its own when the RAW is missing or
malformed, and valid log measurements are still returned. A source's
`source_hashes` row records whether its RAW, log and console output exist and
their hashes, and needs no readable RAW. The server does not take the
producing simulator from a log's contents or from the session's default
simulator, so only a job source has one. `include.provenance` reports
`explicit_dialect` and `producing_dialect` separately; `measurements` rows
report only the producing dialect, which a `path` source does not have. The
check of every source after the recipes read it reports `source_drift` for a
source whose log or console file changed, or appeared after being absent, as
it does for a changed RAW.

When a run has one value per `.meas` (each case of a sweep with no `.step`
does), its `measurements` row contains `measured: {name: value}`, the number a
reduction or spec reads for that name, and `at: {name: point}` where the log
printed an `AT` point beside a result that is its value. A name the log holds
no number for has a null `measured` entry, not an absent one, and the row
names it in one of five places, by why:

- `failed_measurements: [name]`: LTspice printed it as failed at the step the
  row reads, as `Measurement "name" FAIL'ed` or as `failed` in that step's row
  of a stepped table (LTspice 24 and later; XVII prints 0 there, see
  `not_established`).
- `unreadable: {name: {line, reason}}`: LTspice printed something other than
  a number where the result belongs. The entry gives the log line and what
  it printed; only that entry fails, and every other measurement of the log
  reads. A WHEN threshold written as a parameter, an expression or (on
  LTspice 26) a braced expression is not one: it is the level a WHEN with no
  FIND searched for, and its result is the crossing.
- `answer_unknown: {name: reason}`: which field of what the log printed holds
  the result is not known, for a form or a simulator no recording shows, or
  a deck card the printed form contradicts.
- `not_established: {name: {steps, reason}}`: what the log printed at those
  steps (from 0) is not established as the result. LTspice XVII prints `0` in
  a stepped table both for a step whose measurement failed and for a crossing
  at 0 s (recorded: `log/meas_when_step_fails`, `log/meas_when_step_forms`),
  so a `0` in a stepped WHEN's table a build that may be XVII wrote is not
  taken. No recording shows how XVII prints a failed step of any other
  table, so a `0` there is not taken either, with a reason that says so.
  LTspice 24 and later print `failed` for a failure, so their `0` is a
  result. The summary's
  entries and the Python API's `measurements()` carry the same
  `not_established` with those steps null in `values`, and an `inspect`
  measurements row carries its reason.
- `unreadable` also names a number that is not finite, such as a stepped
  FRA's gain margin of `infdB`, quoted in its reason: the log decoder
  (`log_parser.parse_measurements`) holds it as null and gives that one
  reason, which every route carries (`steps` names the steps it concerns
  where that is not every step).
- `printed_once: {name: {step_count, step?, reason}}`: the log printed the
  value once for a run of several steps. LTspice 26 prints one line for each
  measurement of a stepped `.op`, and no `.step` line; the line holds the
  first step's value (recorded: `log/meas_step_op`, compared with the
  results file of `raw/step_op`), so `step` is 0 for LTspice 26 and
  absent, with the reason saying it is not established, for any other build
  and for a log no evidence names. The other steps' values are not in the log, and the reason
  names the recipe that reads them from the results. The summary's entries,
  an `inspect` measurements row and the Python API's `measurements()` carry
  the same statement.

A stepped run's `measurements` row gives each name's statistics in `stats`
(and so does any row read with `include.fields=["value"]`). They are over the
steps with a number (`valid_count`); the other steps are counted by the same
reasons, as the log decoder decides them (`log_types.no_number_reasons`, from
the one list `log_types.NO_NUMBER_REASONS`): `failure_count` counts the steps
LTspice printed failed, and `no_number: {reason: count}` counts every other
step under the name of the row field that gives its reason (`unreadable`,
`answer_unknown`, `not_established`), a count of 0 left out. `total_count`
is `valid_count` plus `failure_count` plus every count in `no_number`. A
`not_established` count beside `failure_count` makes it a lower bound: each
such step may be a failure or a real 0. A value printed once for several
steps is a number, so it is counted in `valid_count`. The Python API's
`compute_measurement_stats` takes the reasons as `reasons`; a null in a
caller's own list that comes with no reason is counted under
`no_number.reason_not_given`, never as a failure.

`failed_measurements` means "LTspice printed it as failed" in every reply
that carries it. The summary, `inspect(kind="results", view="measurements")`
and the Python API's `measurements` name each measurement LTspice printed as
failed at any step, and an entry whose stepped table printed `failed` lists
those steps, from 0, in `failed_steps`. The summary does so at whichever
`step` it reads: its `measurements` are the whole run's, every step's value,
and the step selects what it reads from the RAW (range, `point_count`, AC
bandwidth) and its `temp_c`. A `measurements` row read at one step names only
what failed at that step, so at a step where a measurement was taken the row
does not name it and the summary beside it does. `inspect` marks each value
LTspice printed as failed with `failed: true` on its row, so a failed value
reads apart from one that is null for another reason. Every measurement the
log printed has an entry; a log whose measurements cannot all be given one
fails the measurements read, naming the ones it lacks.

Which field of a printed measurement holds its result is decided once, by
`log_parser.measurement_answer`, from the printed form, the producer and
build that wrote the log, and the run's deck card where it is known; every
reply reads it from there. LTspice 26 and XVII (recorded) print a WHEN with
no FIND, unstepped, as `name: expr=level AT point`, and its result is the
point; a FIND, a DERIV ... AT and a FIND ... WHEN print the result after `=`
and their point after a lowercase `at`; a stepped table prints each step's
result in its first column, a WHEN's crossing included. The measurements
recipe aggregates the field so named (`aggregated_field`). Each entry of the
summary's `measurements`, the Python API's `measurements()` and each row of
`inspect(kind="results", view="measurements")` names it in `answer` (`value`
or `at`), or says why it is not known in `answer_unknown`, and then the recipe
gives that name no number and no `aggregated_field`. A bare path is read
with the build its log's first line names (LTspice 24 and later open a log
with a banner); the summary also reads the build a raw's `Command:` names. A
log that names no build, as XVII's does not, is read with every recorded
LTspice build's rules, which agree on every recorded form, and may be XVII's,
so a `0` in its stepped tables is `not_established`. A DERIV ... WHEN prints its derivative after `=` and its point
after a lowercase `at`, as a DERIV ... AT does (recorded:
`log/meas_deriv_when`), so its result is its value with the deck known or
not. The same `unreadable {line,
reason}` is on the entry in the summary's `measurements`, in
`inspect(kind="results", view="measurements")` rows and in the Python API's
`measurements`, where it says the entry's `values` were printed as other than
a number. An entry whose `answer` is `at` (a WHEN whose threshold is written
as a parameter or an expression) carries it beside a crossing that reads, so
the recipe's row, which names why a result is null, does not list it
(`log_parser.result_unreadable`). The default row omits the per-name `stats` block;
`include.fields=["value"]` returns it, and reductions and specs read it either
way. A stepped run's row has `stats`, with `failed_measurements` and
`unreadable` over all its steps.

`step` and `all_steps` apply to the whole call, so every recipe reads the same
iteration. `run_experiments`' attached `analyze` block takes
the same two arguments (§3.1). Both are part of the request a stored answer
answers, so a page reads the steps its first reply read.

- `include.fields` paths root at one of `source`, `case_id`, `run_index`,
  `step_index`, `step_values`, `assignments`, `circuit`, `deck_sha256`,
  `plot_index`, `dialect`, `snapshot_id`, `value`; an unknown root is rejected
  rather than returning empty rows, and a bare name reads under `value`
  (`rms` is `value.rms`). A dot inside a key's own name is escaped
  as `\.`, so a subcircuit node or a device parameter is spelled
  `value.voltages.v(x1\.out)` or `value.device_op_points.@m\.x1\.m1[gm]`.
  The attached `analyze` block of `run_experiments` reads its
  `include.fields` the same way, and `run_fields` on `run_experiments` and
  `jobs(runs)` follows the same rules over a run row's keys (`case_id`,
  `run_index`, `circuit`, `assignments`, `native_statistics`, `status`,
  `raw`, `log`, `other_raws`, `simulator_version`, `diagnostics`,
  `diagnostics_coverage`, `attempt`), with no bare
  names. A path that no row on the page reaches is reported in `warnings`
  with the keys that are there.
- `include.fields` changes only the row view. A fact a row's value states
  (`failed_measurements`, `not_finite`, `unreadable`, `answer_unknown`,
  `not_established`, `printed_once`, `diagnostics`, `diagnostics_coverage`)
  is kept by the lean row and by any `include.fields`.
- Every recipe of a call is evaluated in it, and its complete answer is
  presented (§1). A cursor is a cut entry's `next`, a page's `next_cursor`
  (the same cursor) or a `rows_not_shown` row's `cursor`. `continue` with it
  reads a page of the stored answer: nothing is evaluated and no source is
  read, so a page is never refused or emptied because a source moved, and it
  carries `stored_answer.snapshot` (the result set, the request and a digest
  of the sources it read). `include.per_run.cursor` reads the same page, sent
  with the request that gave it, `fields` included; another request is
  refused. Request fields sent with `continue` must be the whole request that
  gave the cursor. A cursor from an earlier release, or one whose stored
  answer expired (`[analysis] result_set_ttl_hours`, default 24), is refused
  with `invalid_cursor` and the way to start again: the same request without
  the cursor.
- A first reply's `next` and `cursor` name the first cut list's cursor; a
  page's name the cursor that reads on in the list the page reads, and are
  absent on its last page, so following them ends. Every other cut list reads
  on from its own `cut` entry. Each page object (a recipe's `per_run`, `spec.fail_cases`,
  `coverage.missing_cases`) says what the cut holds of it in `returned`,
  `truncated` and `next_cursor` (`spec.fail_cases` in a `hint`). These keep
  their shapes until the next breaking release.
- When rows of a recipe are left out and some carry a fact, `rows_not_shown`
  names them, by recipe key: `{counts, rows?}`, built by `lib/not_shown.py`
  as `runs_not_shown` is. `counts` gives every kind's number of rows
  (`failed_measurements`, `unreadable`, `diagnostics.<code>`, ...); `rows`
  names the first row of each of at most 3 kinds over all recipes, each by
  `source`, `case_id`, `run_index` and `step_index` with a cursor that starts
  at it; a kind every row of its recipe carries is given by its count alone.
  The whole part is held to `analyze.ROWS_NOT_SHOWN_BOUND_TOKENS` (965
  estimated tokens): named rows come off from the last, then the recipes
  after the first are summed under `*` (starred again if a recipe is keyed
  so). It is never cut.
- A repeated run index, `case_ids` entry, `group_by` dimension or
  `include.fields` path is read once, and the `hint` reports the duplicate. A
  source without a `label` is labelled with its job_id or its raw file's stem,
  with a `-2` suffix where two would collide; a caller-supplied label must be
  unique.
- Validation, results and errors are per recipe, so one bad recipe fails only
  that item.
- Failure rows are grouped by reason. Rows with the same `code`, `stage` and
  `message` (compared with numbers normalized, as a repeated log diagnostic
  is) merge into one row. It keeps the first row by `where`, verbatim, and
  adds `wheres` (up to 10 distinct places) and `count` (the number of failure
  records merged). A recipe that fails on every run of a sweep, such as on a
  signal no run contains, is one row rather than one per run, so the
  failures channel, which is never cut, does not grow with the run
  count. An MCP reply then caps rows at 100,
  with a `failures_truncated` observation that gives the true total and says
  that `api.analyze_results` returns every record. The Python API's complete
  `analyze_results` result lists every record, one per place; an analysis
  attached to an experiment keeps every record, and each door gives them as
  it does for `analyze_results`, a failure whose message ends in an `api`
  snippet saying where the snippet runs for the session that shows it.
- An observation lifted from a recipe's rows names at most 3 of them in
  `evidence.rows`, each with its `index` in the recipe's rows (and a cursor
  when those rows are cut), with `row_count` the number it was made of; one
  every row of its recipe carries says `every_row` and names none. The Python
  API names every row.
- `reduced` is never cut: the request bounds it, one row per field and
  statistic, and a summary is never missing while its rows are shown. The
  counts of `spec` are fields, never cut. Every other list of rows is cut by
  the presenter, `groups`, `spec.outliers`, `signals_available` and
  `source_hashes` included.
- Reductions are attributed:
  `reduced[] = {stat, value, case_id, run_index, step_index?, step_values?,
  assignments, no_value_count?}`. `no_value_count` counts the recipe's rows
  with no number for that field (a failed `.meas`, a null, a missing key),
  which the statistic excludes; it is omitted when every row has a number.
- Spec verdicts:
  `{field, min?, max?, pass_count, fail_count, no_value_count, fail_cases: Page,
  verdict: "pass"|"fail"|"indeterminate", allow_incomplete?}`. `pass_count +
  fail_count + no_value_count` equals the recipe's row count. The verdict is
  `indeterminate` when coverage is incomplete or any row has no number for the
  field, unless `allow_incomplete` is set, which judges the rows that have a
  number and still reports the count. A verdict stored with a job by an
  earlier release has no `no_value_count`; it skipped rows with no number.
- One time limit bounds the whole call, `[analysis] analysis_budget_s`
  (default 60 s), on top of the deadline on each parse: reading the sources,
  the checks of every source before and after the reads, every recipe and
  the signal listing share it, and each step gets what is left of it. A
  recipe that runs out of it fails with `analysis_deadline`, and each recipe
  the limit leaves no time for fails alone with the same code (stage
  `deadline`), naming the fix: ask for it in a call of its own. The recipes
  evaluated before it are answered. The check after the reads always runs
  (at least 50 ms). A source it reads and finds changed loses its rows, with
  `source_drift`; a source the bound leaves it no time to read again keeps
  the rows its recipes answered, and a `sources_not_rechecked` observation
  names it. A signal listing the bound ends fails with `analysis_deadline` at
  stage `signals`, naming the sources it did not read; a source whose
  signals could not be read is named with the reason, under that reading's
  code; `signals_available` lists only the runs read. A recipe takes the
  digest of each file it writes as it writes it, within its own time. The
  publish step after the bound has one allowance for the whole call,
  `analyze.PUBLISH_ALLOWANCE_S` (30 s), to compare the new files with those
  already published; past it, the comparisons not made are skipped and the
  new files replace the old, so no answered recipe loses its file. One call
  takes at most the bound plus that allowance, and the renames. A repeat of a
  request writes to the same artifact paths; an artifact already published
  there with the same bytes is kept as it is, never replaced or reopened for
  writing, so a file held open by another program does not fail the repeat;
  when the comparison cannot finish, the file is replaced. A publish's temp
  that could not be removed (on Windows, a digest that ran out of time can
  still hold it) is removed by `result_store.cleanup` once it is
  `STALE_PENDING_S` (6 hours) old: at startup and before every result set. The result set the call
  records (its request and source digests) names the folder its CSV and plot
  artifacts are written to; its id is a digest of what it records, so the
  same request over the same files gives the same id and the same bytes. A
  result set that reads a job is deleted, with its artifacts, when the job
  is ("Job model", §3.1); one that reads only `path` sources expires after
  `[analysis] result_set_ttl_hours` (default 24).
- A result set captures each source's RAW, log and console output, including
  their absence, with a content hash for each. Every later read checks each
  file's device, inode, size, modification time and change time (on Windows,
  the NTFS change time). A source whose files all match is answered from cache;
  any other source is read again and compared with the capture, and one that
  changed reports `source_drift`. No tool can set the change time, so a
  same-size rewrite that restores the modification time is still reported as
  `source_drift`. A file whose times cannot be read in full is read again on
  every call, and one that changed within a filesystem timestamp tick of being
  captured (100 ms, or two seconds on a filesystem with whole-second times) is
  read again on the next call.
- A same-size rewrite through a handle the writer still holds open, on a
  filesystem that defers its timestamps until that handle closes (Windows
  sets a file time only then), is not detected. Neither is a change by a
  writer whose filesystem clock runs more than that tick behind this
  machine's. A deadline never lets a changed source pass; the source is read
  again or reported.
- Each case of a job is read once its own run has finished, whatever the job's
  status: when it produced results, or kept a partial raw when it was
  stopped. A case still queued or running is a `coverage.missing_cases` row
  with code `run_not_finished`, whose `detail` names what its simulator has
  written so far and whose `evidence` carries `run_progress`' fields; a case
  that ended without results is a row with code `raw_not_produced`. One such
  row costs only its own case. `coverage.runs_running` counts the runs asked
  for that have not finished, so a value over the runs analyzed is not taken
  for the result of a job that has not ended; it is left out when every run
  asked for has finished.
- A source that names nothing is a `coverage.missing_cases` row coded as
  every tool reports the same failure, with that failure's `hint`: an unknown
  `job_id` is `job_not_found`, and a `path` with no file is
  `result_file_not_found`.
- Before any recipe runs, the server reads every source once to record its
  file hashes and the simulator errors its log reports. A timeout during this
  first read raises `analysis_deadline` and creates no result set or cursor;
  retry the original request with fewer sources or a larger
  `[analysis] analysis_budget_s`. The timeout is not stored, so a retry reads
  the sources again.
- A `path` source has no job provenance, so its rows have no `deck_sha256`,
  and an observation says why.
- Full-resolution data is returned as artifact handles
  `{path, content_type, sha256, bytes}`. Inline waveform data is always
  decimated to at most `max_points`, with `points_returned` and
  `points_total` reported, and a reply too long for its target thins it
  further as a series (§1). An inline value is `{format, x, series[{signal, y}],
  points_returned, points_total}`: every trace of one plot is read at the
  same points, so the axis is sent once, as `x`. An AC series has
  `magnitude_db` and `phase_deg` for `y`, and another complex one `real` and
  `imag`. A 32-bit trace's samples are written in their shortest 32-bit
  form, which reads back as the same 32-bit value.

Output: `outcome, coverage {runs_requested, runs_analyzed, runs_running?, missing_cases: Page},
results {key -> {metric, units, reduced[], groups?, spec?,
per_run?: Page, values?, warnings}}, observations[],
failures[] {code, stage, where, message, wheres?, count?},
signals_available? [{manifest_id, signals[]}],
source_hashes, result_set_id, rows_not_shown?, cut?, stored_answer?, cursor?,
next?, hint`.

`results` has an entry for every recipe the call asked for, in the order they
were asked for, whether or not it produced rows. A recipe that produced none
because it failed, whether it was invalid, its source could not be read or it
failed on every run, has empty rows and a warning `No rows: ...` naming
the `failures` places that explain it; one whose source named an unknown job
has empty rows and the gap in `coverage.missing_cases`. The Python API and an
attached analysis give the same entries.

### 3.4 `edit_schematic` — transactional sheet mutation

`edit_schematic` applies a batch of typed ops to a sheet. Creating a file and
editing one are the same operation; `base` selects which.

```
path                .asc path (created if absent)
base                "existing" (default) | "blank"
                    blank = treat the sheet as empty before applying ops;
                    existing = deltas preserving untouched content
expected_sha256     REQUIRED to commit to an existing target, under either
                    base; a mismatch is revision_conflict and nothing is
                    written. Both refusals — missing and mismatched — report
                    the target's current sha256, so a retry needs no extra
                    read. A dry run (an op-less read is one) writes nothing,
                    so it needs no token: it returns the current sha256 and
                    reports a mismatched token in observations
ops                 list[Op] — Appendix A.3
compare             {reference, anchors?, rtol, mode?} — post-commit netlist
                    compare ("equivalence" | "structural_diff"), inside
                    the transaction; the exported netlist is returned
                    only when the verdict is not a match. A dry run
                    compares the sheet the batch would write
dry_run             resolve, validate and return geometry; no write
return_views        subset ["touched", "pin_legend", "preexisting"],
                    default ["touched"]
view_cursors        {label_only_pins?, pin_legend?, touched?, preexisting?} —
                    each a next_cursor from a previous page of that view, and
                    echoing one returns that view even when return_views
                    omits it; preexisting also takes preexisting.cursor
view_limit          page size for the paginated views (default 100)
```

`compare` takes the same object as `verify_circuit`'s `compare`, with the same
field names. On a dry run, an op-less read included, it compares the sheet the
batch would write, from the same staged copy, and nothing is saved. When an op
failed the batch would write nothing, so `verification.compare_error` says the
sheet was not compared.

**The exported netlist is returned only when the comparison did not match.**
The compare stage exports the committed sheet to a netlist. It exports a copy,
staged with the project-local `.asy`/`.lib`/`.sub`/`.inc`/`.mod` files under the
sheet's folder, as `verify_circuit`'s `managed` export stages them, so a symbol
saved beside the sheet is found; an `.asc` reference is staged the same way
from its own folder. On `equivalent: true` the reply omits it and no option
includes it; `verify_circuit` with the `export` check writes it and returns
its path. On a mismatch, a compare error or no verdict, it is returned as
`netlist`. An export failure produces no netlist to return. The Python API
follows the same rule; its complete reply differs from the MCP reply only in
having unpaginated views.

**No rendering.** `verify_circuit` renders sheets; `edit_schematic` does not.
A failed batch writes nothing, not even a draft file, and the response names
the failed stage and the reason.

`touched` is the pin/net table for the parts whose wiring the batch could
change, which are the parts its ops place, move, remove, wire or label. A part
whose value or attributes alone changed keeps its pins where they were and
joined as they were, so a batch of value changes lists no pins, in `touched`
or in `wiring.label_only_pins`; its label-only pins are counted under
`preexisting`. A pin's `net` is the names the labels of its net carry, absent
when none does: the text of every label anywhere on the net, each spelling of
a name among them, joined with `/` (`OUT/out`). A label-only pin's row in
`wiring.label_only_pins` and in `preexisting` names its net the same way, not
by the labels on its own point, and `inspect(kind: "net")` lists the same
`labels`. Nets are the ones LTspice netlists: labels of one name, in any
case, are one net (the `node-name-case` recording; an export keeps each
label's own spelling), a wire straight between two pins of one part joins
nothing, and of one part's pins alone on a point only the last in SpiceOrder
touches another part's. LTspice names such a net only when it exports the sheet (`N001`,
`NC_01` for a pin on nothing), so the name comes from an export and is never
guessed. A batch with a `compare` exports the committed sheet, and each pin on
a net that no label names then also has `net_export`, its node in that export.
The whole-sheet `pin_legend` is returned only on request, and is the only
whole-sheet pin-to-net table on the surface.
`inspect(kind: "components", detail: "full")` gives pin coordinates without
net names, and `inspect(kind: "net")` traces one net at a time.

**Findings are scoped to the edit.** The sheet findings are the ones
`verify_circuit` reports for the same sheet, with the same wording. They cover
a part whose symbol is not found, a wire LTspice leaves out of the netlist,
floating pins, loose wire ends, dangling labels, duplicate wires, overlapping
parts, a wire through a part, a label or text inside a part, stacked
directives, and a net joined only by labels. The edit computes them on the
text it is about to write. The findings and the label-only-pin list describe
the whole sheet, and on an existing sheet most of them predate the call. The
transaction therefore computes both before and after the ops, each on the
sheet as the server writes it out (so the server's own formatting of the file
raises no new finding), and reports a finding only if the batch made or
changed it, that is, if no finding with the same rule, parts and subject
existed before. A finding's subject is what it is about apart from where it
is, for a rule whose places move with the parts it names or whose facts hold
an attribute's words: which attribute's text, and what it is drawn across;
which pin floats; the parameter and its copies. So a part that is moved or
given a new value keeps the findings it had, and a finding the batch left as
it was is not the batch's, whichever part it names. For any other rule the
subject is the finding's points and facts, and any change makes it new. All
other findings are listed in the `preexisting` block, with the label-only pins
the batch neither changed nor named. A label-only pin is the batch's when its
row is new (the pin was not label-only before, moved, or its net's names
changed), when an op that may rewire it names its part, or when an op names
its point:

```
preexisting {count, findings, label_only_pins, cursor?}
```

`rules_run` maps every sheet rule that ran to its number of findings on the
resulting sheet, zero included, so a rule that found nothing can be told from
one that did not run, and a rule whose findings are all preexisting still
appears. Its counts sum to the sheet findings in `warnings` plus
`preexisting.findings`. `verify_circuit` returns the same field for the sheet
rules of the checks it ran, counted before any cap.

Preexisting findings are counted rather than dropped, and the counts
reconcile. `wiring.pins_total`, `pins_wired` and `pins_label_only` cover the
whole sheet, and
`label_only_pins.total + preexisting.label_only_pins == pins_label_only`.
To list them, pass `cursor` back as `view_cursors.preexisting`, or name
`preexisting` in `return_views` to get the first page in the same call. The
cheapest such call is an op-less read (`ops: []`). Because nothing in it is
new, its pages cover the whole sheet. Either way the reply has
`views.preexisting`, which lists the findings first and then the label-only
pins, with `kind: "label_only_pin"`. Each finding is a row with its `kind`,
its sentence, every part (`ref`, and `refs` when there are several) and point
(`x`/`y` and `points`, or a wire's `from`/`to`) it names, and the rule's
facts. A `base: "blank"` build starts from an empty sheet, so nothing is
preexisting. An op judges nothing itself. What it did is in `results`, and
`warnings` holds only the sheet findings.

The op union is discriminated on `op`, so an unknown op fails with one error
listing every op kind, and a bad field on a known op is reported against that
kind only.

Fixed rules:

- Commit protocol: the new sheet is written to a temporary file beside the
  target, then the `.plt` is written when the batch sets panes, and the
  `.asc` is renamed over the target last, so the transaction is as atomic as
  one file rename. If that rename fails, the `.plt` gets its old content back.
  The reply's `sha256` is the digest of the bytes written.
- Vocabulary: grid 16; rotations `R0..M270`; pins as `REF.PIN`, where `PIN` is
  the pin's name or, when no pin has that name and it is all digits, its
  1-based SpiceOrder, so `X1.2` works on a block whose pins are lettered; the
  legal symbols are the ones `inspect` reports.
- Route refusals: an endpoint that does not resolve, a diagonal, a pin passed
  over, a run along an existing wire, another net's wiring touched, two named
  nets joined, and a straight wire between two pins of one part, which LTspice
  leaves out of the netlist. A route through a part's body is not refused; the
  finished sheet's `wire_through_symbol` finding reports it.
- Every reply has the `wiring` counts with the `label_only_pins` page, and
  every finding has `at` and `subject`.
- Retry: resubmitting with the original `expected_sha256` returns
  `revision_conflict` if and only if the earlier call committed.

**There is no undo.** Recovering from a committed edit requires the caller's
own copy of the sheet.

**A sheet open in LTspice.** LTspice reads a sheet once, when it opens it.
From then on, a window showing the target keeps its own copy, so a write to
the file does not change the window, and the window's next save writes the old
sheet over the new file. A transaction that replaces the sheet therefore
checks for such a window, while it holds the sheet's lock, through the MCP
bridge LTspice ships from 26.1. A transaction that leaves the sheet unchanged
(a read with no ops, or a batch of only `set_plot_panes`) replaces nothing in
a window's copy and does not check. For one that replaces the sheet:

- **Before the commit**, the window's copy is compared with the file. A
  window that differs either has unsaved changes or was opened before the file
  last changed, and committing would leave two sheets, each missing the
  other's changes. The call then fails with `open_window_differs` at stage
  `window_check`, naming a few of the differing entries, and writes nothing;
  its `hint` names `inspect(kind="open_in_ltspice")` with this path and the
  window's process as `pid` as the call that reads every entry. A dry run
  reports the same difference in a note, with the same call.
  A call with no ops and `open_window` set to the user's choice settles it;
  like any write to an existing sheet it quotes the file's sha256 as
  `expected_sha256`. Every reply that sends the caller to this call writes it
  out with each argument it needs (`_base.open_window_call`), naming where
  that sha256 is: the refused call's own, a commit's reply, or
  `inspect(kind="open_in_ltspice")` for a reply that carries none.
  With `"save"`, LTspice saves the sheet from that window, as Ctrl+S does,
  and the reply gives the file's new `sha256`. Where more than one window
  holds a different copy, `window_pid` names the process whose copy the user
  chose, the other windows are handed the saved sheet, and a call without it,
  or with a process that holds no different copy, fails with
  `window_not_chosen`, listing each window's difference. A box
  LTspice shows instead of saving fails the call with `window_not_saved`,
  quoting the box, and the server closes the box with Cancel, which leaves the
  window's copy as it was.
  With `"discard"`, the server closes the sheet in each window that differs
  without saving and opens it again from the file, which is unchanged; a
  window whose copy changes meanwhile is left open and fails the call with
  `window_not_discarded`. Either way the caller reads the sheet again before
  editing it. A dry run reports the
  difference in `observations` and is not refused. The comparison is by
  content rather than text, because LTspice returns its own serialization of
  a sheet. That serialization differs from the file it read in the first
  line, the order of wires and attributes, attributes with no value, the
  micro sign and off-grid text, and none of those differences counts as a
  change.
- **After the commit**, the committed sheet replaces the window's copy;
  LTspice records this as one step in that window's undo history.
  `open_in_ltspice` has one entry per window, `{pid, version, shown,
  reopened?, reason?}`. A window that does not take the committed sheet is
  closed and opened again from the file (`reopened: true`), if it still holds
  the sheet as it was before the commit; its undo history goes with it. A
  window that could not be updated either way (`shown: false`) does not fail
  the call, because the file is committed, but the entry and the `hint` warn
  that the window still shows the old sheet and that saving from it would
  overwrite the edit.

The file remains authoritative. `expected_sha256` is the file's digest, the
window is never read in place of the file, and a later save from the window is
an external write like any other (LTspice rewrites the sheet in its own form,
so the digest changes and the next edit must read the file again). An edit
never starts LTspice and never opens the sheet in a window that did not have
it. The window check applies only when the server runs natively on Windows
with LTspice 26.1 or later; elsewhere, or with
`[schematic] sync_open_window = false`, there is no check and
`inspect(kind: "capabilities")` gives the reason under `open_window_sync`. If
the bridge does not respond, the edit is still committed and `observations`
reports that the window could not be checked.

**Per-op facts.** `results` has one entry per op that found something on the
sheet, keyed by `index` and `op` as a `failures` entry is. A `wire_pins` entry
lists the segments it found `already_present` and the `junctions` it made,
and a removal's entry reports how much it removed; `remove_directive`'s
entry names each text it removed, with its kind and place (`removed_texts`),
and `replace_directive`'s gives the text it changed, with its kind and place
and what it `was` and is `now`. After a `find` edit, `line` is the first line
it touched (from 1), `was` and `now` hold only the touched lines, and `undo`
is the op that puts the words back, or `undo_refused` says why a write of the
old words would be refused. A text is
given as it reads, a line break as a line break.
Everything else an op reports is in the views. `applied` counts the ops the
batch applied, every op of a committed batch, so an op with nothing to report
is counted though it has no `results` entry. `results` is not a page because,
like `failures`, its length is bounded by the batch the caller sent. The
Python API returns the same list.

**Wire ops.** A routed segment identical to one already on the sheet is not
drawn again; the op's `results` entry lists it under `already_present`, and
the rest of the route is drawn. Removing an exact segment that exists more
than once removes every copy, unless that would split a net in two, in which
case the op is refused, naming pins on each side, and nothing is written. A
segment that exists once is always removed, because that is an explicit
disconnection.

**T-junctions.** LTspice's netlister connects a wire end, a pin or a label
anywhere along a wire, including its interior, to that wire, and does not
connect two wires that only cross; a label at a crossing connects both wires.
Every tool that reads connectivity applies this rule, so a T needs no wire
split. `wire_pins` accepts `{x, y}` for either endpoint. The point must touch a
wire, a pin or a label; on a wire's interior the new segment ends there and the wire is
left whole, so `remove_wire` of that segment restores the sheet. The op's
`results` entry lists the T under `junctions` as
`{x, y, via: "tee", wire}`, where `wire` is the segment joined. The new
segment must leave the wire at a right angle, and a point where wires of two
nets cross is refused, because a wire ending there would join both.

Wherever else the route touches existing wiring (a waypoint on a wire, a
segment through another wire's end or a lone label, or a segment over a pin
already wired to an endpoint), it connects to that wiring, so the op checks
the touch against the nets the two endpoints already join. A touch on
one of those nets is a redundant junction. It is drawn and reported as a
`junctions` entry in the op's `results` entry, whose `via` gives
what it touched (`waypoint`, `wire_end`, `label` or `pin`) and whose `wire`,
`label` or `pin` identifies it. A touch on any other net is refused, and the
error names the wire and the `{x, y}` endpoint that would make the same T
deliberately, because the op does not join a net no argument asked for. A
route crossing a wire where neither ends is drawn; LTspice does not connect
the crossing, so the nets stay separate, and the finished sheet reports it
(`unjoined_crossing`). Route length is not reported.

Ops do not repeat sheet findings. A route through a part, a part placed or
moved over another, a part's text drawn across a wire, a part or other text, a
label on nothing, a crossing left unjoined, a pin or label that joins wires
where they cross, a pin placed or moved inside a wire, a wire left diagonal,
and a net given a second name are each reported once, by the sheet finding
(`wire_through_symbol`, `symbol_overlap`, `attribute_text_overlap`,
`dangling_label`, `unjoined_crossing`, `joined_at_crossing`, `pin_inside_wire`,
`diagonal_wire`, `net_with_two_names`), whichever op made it. Findings are computed on the sheet the batch leaves, so a state the
batch only passes through is not reported. A label placed before the wire
that reaches it is not dangling, and a net relabelled in remove-and-add pairs
has one name at the end. A repeated route that is already drawn through a
part is counted under `preexisting`, because the batch drew nothing.

**Labels.** `add_net_label` refuses a label that would join two named nets,
either by giving a net a name another net already has or by placing a new name
where two named nets cross, because the netlist would short them. A new name
on an already named net joins nothing; it is placed, and the sheet reports a
net with two names (`net_with_two_names`). When `wire_pins` refuses to join
two named nets, its error gives the fix that op accepts, which is to relabel
one side with the other's name. A label, ground included, at a point where
two wires cross joins them, so a ground flag on a crossing of two named nets
is refused, as a named label or a pin there is. Where at most one of the two
wires has a name, the label is placed and the sheet reports the join
(`joined_at_crossing`), as it does for a pin. Grounding a named net is not
such a join, because it is what the flag requests, and a second named net may
be grounded after the first. A net whose only label is a ground flag carries
no name either, so a name, a pin or a route that joins it to a named net only
grounds that net. A net is known in these checks by the names its own labels
carry, as the `move` check has always known it. Names already joined, by a wire between two
labels or by a shared name, are one net, so joining them again is no short.
`wire_pins` passes over a pin, a wire's end or a label already on the net of
either of its ends as a junction, and refuses one on any other net.
`inspect(kind: "net")` follows the nets as LTspice netlists them: a wire
straight between two pins of one part joins nothing, and a trace from one of a
part's pins stacked on one point answers that pin's own net. At a point on a
wire's interior it traces that wire's net and names the wire under
`snapped_to_wire`. A point on two nets is refused, saying why: two nets' wires
cross there, it is inside a wire LTspice drops, or it holds a part's stacked
pins.

**Waveform panes.** `set_plot_panes{analysis, panes}` writes the plot
settings file that LTspice's waveform window reads when the sheet runs, which
is the `.plt` beside the sheet, with the sheet's name. `panes` lists panes top to
bottom, each `{traces, x_scale?, y_scale?}`; `analysis` is `tran`, `ac`, `dc`
(a DC sweep) or `noise`. The op replaces that analysis's section and leaves
the file's other sections as they were, as LTspice does when it saves one;
`[]` removes the section, and removing the last section removes the file. Its
`results` entry gives the file (`plot_settings`) and the panes the section
held (`replaced_panes`) in the op's input form, so passing them back as
`panes` restores the section. The sheet's bytes do not change, and the reply's
`sha256` is still the sheet's.

In the file format that LTspice 26 and XVII read, panes are listed bottom
first; a trace's id and axis are written as 0, since each build assigns them
itself; a trace is read only up to its first space, so a trace containing
whitespace is refused; each pane has a `Log` line with the build's default for
the analysis; no axis range is written, since a run of the sheet sets every
axis range. A pane created while a build's waveform grid setting is on has a
`GridStyle` line, and a pane without one shows no grid whatever the setting.
The op has no grid argument; new panes keep the line when every pane they
replace had the same one. The file is written as UTF-16 LE with LF line ends
and no byte order mark, which both builds read and neither damages when saving
over it. A file that mixes two encodings, as LTspice XVII leaves when it saves
over a UTF-8 file, is refused.

The `.plt` is written in the sheet's transaction, under its own cross-process
lock, so a batch that sets panes commits both files or neither. A batch of
only `set_plot_panes` leaves the sheet file untouched, and a `.plt` whose
content the batch leaves unchanged is not written. The `.plt` is read only
when an op sets panes. A `.plt` that cannot be read or written fails the batch
at `apply_ops` or `stage_plot_settings`, and nothing is written.

`edit_schematic` edits schematics only; netlists are edited as text, and
`verify_circuit` lints them but never changes them. A whole circuit can be
built in one `edit_schematic{base: "blank"}` call.

Output: `outcome, commit_state ("committed" | "not_committed"), path,
sha256, build_id, stages[] (the stages that did not
complete; absent on success), netlist? (only when a
compare did not confirm equivalence), verification?,
wiring {pins_total, pins_wired, pins_label_only, label_only_pins: Page},
rules_run {rule: count}, preexisting {count, findings, label_only_pins, cursor?},
views {touched?: Page, pin_legend?: Page, preexisting?: Page}, applied, results[],
open_in_ltspice? (one entry per LTspice window that had the sheet open),
warnings, failures, observations, hint`.

### 3.5 `verify_circuit` — gate

```
path          .asc | .cir | .net | .sp | .spice
checks        subset [syntax, symbols, export, layout, quality, compare]
compare       {reference, anchors?, rtol,
               mode: "equivalence"|"structural_diff"}
render        {format: "png"|"svg", scale?, max_pixels?,
               mode: "with_checks"|"only",
               delivery: "artifact"|"inline"|"both"}
              `true` selects the default policy; `false` or omitted renders
              nothing
in_ltspice    also open the file in the LTspice window the user has running,
              in front; default false
export_to     "managed" (default) | "sidecar"
findings_cursor
              a cut entry's next: reads the findings after those from the
              stored answer, without checking again; give the same path
```

`compare` (`{reference, anchors, rtol, mode}`) means the same here as in
`edit_schematic`.

**In LTspice, on request.** `in_ltspice` opens the checked file, sheet or
netlist, in the running LTspice window and brings it to the front, through the
bridge LTspice ships. The reply's `ltspice` block reports the result:
`{shown, path, pid, version, already_open?, differs_from_file?, difference?,
reason?}`. A sheet the window already had open is brought to the front as the
window has it, because LTspice does not read a file again; its copy is
compared with the file, and `differs_from_file` is set when the user is
looking at something other than what was checked; `difference` then names a
few entries, and the `hint` names `inspect(kind="open_in_ltspice")` with this
path and the block's `pid` as the call that reads every entry. With no window open,
LTspice is started in view (`started: true`); with `[schematic]
start_ltspice` off, the block reports that no window is open and the check
results are still returned. `in_ltspice` never fails the call.

`managed`, the default `export_to`, leaves the caller's files alone. It
copies the schematic and the project-local `.asy`/`.lib`/`.sub`/`.inc`/`.mod`
files under its folder into a scratch directory in the store, exports there,
and writes nothing beside the caller's file (the lock it takes while copying
is in the per-user folder). A compare of that export resolves relative
includes from the schematic's folder, not from the scratch copy.
`run_experiments` exports a schematic the same way. `sidecar` overwrites the
schematic's `<name>.net` under lock and returns `{path, sha256,
diff_vs_prior?}`. LTspice writes the same file beside a schematic it runs from
its own window; asking for it makes the call destructive, as the annotation
table shows. Both modes report in `diff_vs_prior` what changed since the
schematic's previous export: `sidecar` against the `.net` it replaces,
`managed` against the previous managed export of the same schematic, which
the store keeps. It is the `structural_diff` delta below, each part and
directive in its scope, plus `connections_changed`: each pin of a part both
exports have that moved onto or off a net both exports name (ground
included), as `{reference, pin, before, after}`, net names in lower case. The
pins are read from the wiring comparison `equivalence` runs, with each net
both exports name the same anchored by its name and a net the exporter
numbered (`N001`, `NC_01`), which another export may number otherwise,
matched by the parts on it. A move between two numbered nets is not named,
since which pin the comparison would report there depends on the order it
reads the parts; `not_compared` says the wiring differs there, as it does
when an export cannot be read as a circuit. `diff_vs_prior` holds no
difference list when nothing changed and is absent when there was no
previous export.

Rendering uses the project's own SVG-to-PNG renderer; the `render` policy
sets format, scale, pixel cap, and whether the image is returned inline or as
a file. Text is drawn as LTspice draws it, based on screenshots of both
builds. It uses Tahoma Bold at LTspice's size for each text size and lays out
each justification on the part's placement, so a `Left` window on a part
placed `R180` or `M0` ends at its point, and one on a part turned a quarter
turn reads bottom to top. A multi-line comment's lines are one line apart, and
a `Bottom` one ends at its point. The file is written to the per-user folder's
`renders/`, never to the working directory or beside the sheet, and is deleted
a week later; a render regenerates it. `inline` and `both` return a PNG as an
image block, and an SVG as its markup in `render.svg`. A PNG is made with
`resvg-py`, a dependency, or, where that cannot be imported, with `cairosvg`
and the native Cairo library it loads; without either the render falls back
to SVG, and `render.note` names what is missing and its remedy. The fallback is also a `render` failure when
the caller named `format: "png"`. A default that falls back is not a failure,
because the caller asked for nothing the server could not do.
`inspect(kind:"capabilities")` reports the same facts as
`render: {png, missing, reason, remedy}` before anything is drawn.

`equivalence` compares the two netlists as graphs of parts and nets: the
component set, values, normalized parameters, which pins share a node
(matched by the wiring, not by node names), the number of terminals, and
`anchors`, the named nets (ports, rails, outputs) that must also match by
name; ground always must. A subcircuit instance both circuits have is
compared by the parameters it runs with, as a part is, in `param_mismatches`:
each it gives (an `X` card's `params:`, with or without the keyword), else its
definition's `PARAMS:` default, as LTspice reads an instance. So a default
reaches the comparison only where an instance leaves the parameter out. It
also compares the directives, each read as `structural_diff` reads it (below),
scope by scope: the top level's, and those of each definition an instance both
circuits have calls, wherever it is defined, inline or in an included file,
and whatever each side names it. A definition's own directives are its body's,
each listed with `(in .subckt NAME)` after it, and the instance that paired
two definitions of different names. An `.include` or `.lib` line
is a directive, but one whose file (or section) defines subcircuits and nothing
else is compared by those definitions instead. Both modes compare a `.param`
moved from a body to the top level as a difference. They differ where each
reads the deck: a definition moved into such an include is a difference in
`structural_diff` and not here, and a subcircuit the circuit does not call is
compared there only.
`equivalent` is true only when the wiring, the parts and values and the
directives all match. Beside a false one, `structurally_equivalent`
says whether the wiring matched and `directives_equivalent` whether the
directives did, and `directives_added` and `directives_removed` list the
directives only this circuit has and those only the reference has.
In either mode `values_read_otherwise` summarizes the values of each side's
files that an LTspice build the session may run reads as another number than
the comparison does, one summary per build and kind of reading (see
`inspect(kind="hierarchy")`). A reference given as text has no stored bytes
and is summarized by nothing; a sheet is summarized by its own bytes.
`structural_diff` reads no included file, and its `not_compared` says so.
`structural_diff` reports the added, removed and changed components and
directives of the two files as written, each in its scope: the top level, or
the `.subckt` definition it sits in, paired with the definition of the same
name in the same scope on the other side, a nested definition a scope of its
own. An entry inside a definition says which: a string ends in
`(in .subckt NAME)`, and an object carries `subckt`, nested names joined with
`/`. So two elements of one reference in two blocks of an exported sheet are
two elements. It does not compare nodes, and its `not_compared` says so in the
data; it also names a definition either deck gives more than once in one
scope, which it does not compare on either side (which of them stands for
which is not known; `equivalent` is then null), and says that included files'
cards are not read when a deck includes one. The modes differ on a deck they
cannot read.
`equivalence` returns a compare failure with no `comparison` block, while
`structural_diff` diffs the deck as empty, so the delta is still returned,
`equivalent` is null, and a warning names the deck that failed. In both cases
the call's outcome is `partial` and the response gives the reason.

When a card gives a parameter more than once, `equivalence` reports it in
`duplicate_params` once, however many instances use the card: its reference
and the subcircuit it is in (null at the top level), the side, the key and
every copy in card order. The comparison does not model which copy a simulator reads, so the
parameter compares equal only when the result is the same whichever copy is
read, which holds when both sides have the same copies in the same order or
every copy has one value. Otherwise a `param_mismatches` row names a
reference copy and a candidate copy that differ.

Both modes compare netlists. An `.asc` under test is compared as its LTspice
export, so asking to compare one asks for `export` too, with
`checks: ["compare"]` as with the default list: the export runs and reports as
a check that ran, with its `export` block and findings, and is written where
`export_to` says. An `.asc` reference is exported the same way, on a staged
copy, never beside the caller's file; without LTspice the compare fails rather
than diffing a schematic's attributes against a netlist's cards. Both modes
read the cards the simulator reads: line 1 of a deck is its title and is not
read, nor is a card after `.end`. `equivalence` follows includes as staging
reads them, so `.lib <file> <section>` brings in that section's definitions
only, and of two definitions of one subcircuit name the body compared is the
one the session's simulator keeps in force (ngspice the first, LTspice XVII the
last of two in one deck), the others listed in `duplicate_subckts`; where the
simulator keeps none (LTspice 26 refuses the deck) or what it does is not
recorded (any other build or place, QSPICE, Xyce), no body is read, the
instances are compared as black boxes, and the entry's `reason` says why. A
name defined both inside a subcircuit (local to it) and elsewhere is read
neither way, since the comparison keys definitions by name, and the entry
says so. `structural_diff` lexes
both decks, so a `+` continuation belongs to the card it continues, and
compares directives as parsed rather than as text, so spacing, case, comma
separators and assignment order in `.model`/`.param` are ignored. It ignores
the lines the netlister adds to every export, which are `.backanno`, the
install-path `.lib …/cmp/standard.*`, and a parameterless default model such
as `.model NMOS NMOS` when the reference declares no model of that name. A
component's signature is its model or value plus its instance parameters,
without nodes (`equivalence` is the mode that compares wiring); a parameter
given twice contributes both copies, so a second copy counts as a change.

Output: findings in the shared shape, `cut`, `stored_answer` and
`findings_omitted` when the reply was too long to send whole (below), a
comparison block per mode, a render block `{path,
sha256, source_sha256, width, height, downscaled, returned_inline, svg?,
note, …}`, a scene summary, `outcome` and `hint`.

`comparison` is present exactly when `compare` is in `checks_run`. When it is
not, a `checks_skipped` row says why (no reference supplied, not requested,
not applicable, or no exported netlist to compare, which names whether the
export ran) or a `failures` row at stage `compare` gives the error, so a
reply without a comparison always says which.

**Outcome.** `complete` means every check the caller asked for ran, with no
failure, no finding at error or warning severity, and no comparison short of
`equivalent: true`. A check asked for that did not run is a shortfall: one
listed in `checks`, the comparison asked for by supplying `compare` with the
default list, and the export that comparing an `.asc` reads. Its
`checks_skipped` row or `failures` row says why, and `hint` names it. The
outcome is then `partial` when some check ran or the drawing was returned, and
`failed` when no check ran and no drawing was returned, whether the checks
were skipped or failed. `isError` stays unset in both cases, because the call
was valid and the reply says why each check did not run. Skipping a check the
caller did not ask for is not a shortfall: one left out of `checks`, one the
default list leaves out because it does not apply to the file kind, and every
check under `render.mode: "only"`. Neither is a check on the default list that
this session cannot run, such as `export` without LTspice when nothing is
compared: its `checks_skipped` row says so, and naming it in `checks` makes
its absence a shortfall.

In the render block, `sha256` is the image's digest and `source_sha256` the
digest of the sheet file this call drew. Rendering reads the file on disk, and
another session sharing the working directory may commit between
`edit_schematic` returning and this call running; compare `source_sha256` with
the `sha256` `edit_schematic` returned to confirm the image shows that
revision. The export block's `sha256` is the netlist's.

The layout check reads a part's name, value, Value2, SpiceModel, SpiceLine
and SpiceLine2 where LTspice draws them, through their windows, taking a value
from the part's symbol when the part gives none, with each extent estimated
from the font's widths rather than measured off its glyphs. Any of them but
the name written as `""` has no text, and LTspice draws none, not even the
symbol's. A symbol's own text, its pin names and an attribute in any other
window are not read, which the layout check's `observations` say, and so does
a render's.
`attribute_text_overlap` reports text that a wire or a line another
part draws passes through, or that lies over a ground flag's triangle, a
rectangle or circle another part draws or other text, by more than the
estimate's slack. A ground flag is taken as the triangle both builds draw at
the default text size (`draw/ground_flags`): a base 18 units either side of
its point and an apex 18 units away, turned as LTspice turns it. The render
draws the same triangle.
`sheet_text_overlap` reports a comment, directive or net label drawn across
another, by the same estimate and slack; texts at one anchor are
`stacked_directive`'s. The layout check states in `observations` what its scan
compares. No op moves a part's text apart from the part, so the remedy is to
move the part or the item its text crosses. `wire_through_symbol` and `text_in_symbol_body` likewise test
what a part draws, not its box, so a wire or a text anchor in the empty corner
beside an amplifier's triangle is not in the part.

Findings are sent whole when the reply fits the target, however many a rule
has; the check returns every finding, in the order the checks found them,
and that order is their identity. A reply over the target is presented as
"A reply too long to send whole" (§2) describes, with the snapshot
`{path, sha256, reference_sha256?}`: the checked file's digest, and the
reference's when one was compared (of its text when the reference is netlist
text). `findings` is declared `items`, cut from the end; `observations`,
`warnings`, `failures`, `checks_run` and `checks_skipped` are `never` lists
and go out whole. The `hint` counts every finding the check found, not only
those sent, and names the stored file, the `findings_cursor` call and its
Python form.

`findings_cursor` takes a `cut` entry's `next`. The call reads the next
findings from the stored answer and runs no check; it needs the same `path`,
and a cursor for another circuit's answer is refused with `invalid_cursor`.
The page's `hint` says which findings it holds and that they were read from
the stored answer. `findings_omitted`, keyed by `rule_id`, keeps the shape it
had while findings were capped per rule, `{omitted, total, next_cursor}`, and
is filled from the cut: `omitted` counts that rule's findings after the ones
sent, `total` all of them, and `next_cursor` is the `cut` entry's `next`,
which continues the whole list rather than that rule alone. It and
`findings_cursor` go at the next breaking release.

`api.verify_circuit` returns every finding and takes no `findings_cursor`
unless called with `raw_page=True`, which returns the reply a client reads.

**No `baseline` filter.** `verify_circuit` reports on the whole file. To find
what changed relative to a copy the caller kept, diff the two finding lists,
which include `rule_id`, `at` and `subject`. Diff complete lists: the stored
answer's when the reply was cut, or the findings `api.verify_circuit`
returns, which are never cut.

### 3.6 `inspect` — pure read

`inspect` takes `queries: list[Query]` and returns a result per item, with
cursors where listed:

```
{kind: "capabilities", fields?}
    simulators: for each, its executable and sha256; `version`, the build
    the latest run on that executable reported, with `version_source`
    naming the job and case (absent until a run; the executable is never
    launched to ask); `selectable`, whether run_experiments'
    execution.simulator may name it, with `refusal` when this host cannot
    run the family (detected or not), and for an undetected one, whose
    `available: false` already says it cannot be selected, the
    `remediation` that enables it. named_executables: the same facts and
    the family for each [simulator.executables] entry accepted at startup,
    keyed by its execution.simulator selector ("ltspice:xvii"). Also:
    exporter presence (`exporter_available`); `open_window_sync: {available,
    reason, starts_ltspice}` (whether edit_schematic keeps a sheet open in an
    LTspice window in step with its file, and whether in_ltspice starts
    LTspice when no window is open); dialects; `simulator_capabilities:
    {families: {<family>: {<capability>: {kind, status, reason?}}}, evidence}`
    (what each detected family can do here: every capability the server
    offers, `baseline` or `refinement`, as `supported`, a declared `gap`,
    `unverified`, or `unavailable_here` when this host rules it out, with
    the reason; the evidence behind each entry, and the undetected families,
    are at the `spice://capabilities` resource and
    `spice://capabilities/{family}`); job persistence
    (`persist_jobs`); the sandbox (allowed_paths, null when none is set);
    `tool_profile`; the tool listing this session was served
    (`tool_listing`); limits; linter_version; the startup diagnostics
    (`diagnostics`), which show whether the server started degraded; and
    `render: {png, missing, reason, remedy}` (whether verify_circuit can
    make the PNG it inlines and, if not, whether `resvg-py` cannot be
    imported or only cairosvg is installed, without native Cairo). `fields` (e.g.
    ["allowed_paths", "config_path"], to check after a config edit) returns
    only those top-level keys; without it the whole report is returned
{kind: "symbols", path?, filter?, cursor?}
    the symbol names the libraries resolve, and resolution order; `path` adds
    the sheet's own folder, where only a symbol directly in it is found by its
    bare name, to the reported precedence
{kind: "symbol", name, path?, rotation?}
    pins per rotation, bbox, origin; a `rotation` (one of R0, R90, R180, R270,
    M0, M90, M180, M270, as edit_schematic names them) gives `pins_by_rotation`
    and `bbox_by_rotation` that rotation alone, and omitted gives all eight. Any
    other value fails the item as `invalid_query`, naming the eight
{kind: "net", path, at: "REF.PIN" | "net:NAME" | "NAME" | [x, y], cursor?}
    .asc gives a geometric trace; an [x, y] on a wire's interior traces that
    wire, reported as snapped_to_wire; a netlist gives the membership of the
    deck's own cards (not those in a .subckt body) and no geometry, with node
    names compared without case and `gnd` read as node 0, and a warning
    saying ngspice's kiltpsa mode may not read `gnd` so. On a .asc, PIN is a
    pin name or, failing that,
    the pin's 1-based SpiceOrder; on a netlist it is a 1-based terminal number,
    and a name after one of the deck's references (`M1.D`) is refused as
    `invalid_at` unless a node is called that.
    A bare NAME is read as net:NAME on both: a net label on a .asc, a node
    name on a netlist. A name no label of the sheet carries, or no card of
    the deck's own is on, is refused on both as `not_found` rather
    than answered as an empty net, as is a pin of a part neither holds; ground is a node of every deck, and the
    netlist's refusal lists its nodes and says what the query does not
    read: a subcircuit body that uses the name, a `.global` net only such
    a body uses, a file the deck includes, a card it could not read
{kind: "components", path, prefix?, detail: "list"|"full", cursor?}
    `prefix` keeps references that start with it, case-insensitively, on
    both a .asc and a netlist: `"M"` for every MOSFET, `"LX"` for LX1, LX2
    but not L1. It is plain text; a wildcard is refused with the plain
    spelling to use. A netlist's components are the cards the simulator
    reads, as the net query's members are: neither the deck's first line,
    its title, nor a card after `.end`. They are the top level's cards; a
    card of a `.subckt` body follows them with `subcircuit`, the names of
    the bodies it sits in, outermost first, and a reference is one card
    only within its scope. The cards of a subcircuit defined more than
    once in one scope are left out, since no reference there names one
    card, and a warning names it. On a netlist, `detail: "full"` adds each card's
    `nodes`, `model` and `params`, keyed as written (a key given twice in
    one spelling reads as its last copy), and `repeated_params` for a
    parameter the card gives more than once, compared without case: every
    copy as `key=value`, in card order. Which copy a simulator reads is not
    established. On a .asc, a part whose symbol has no pins to place says
    why in `symbol_problem` at either detail: `"not found"`, or a name that
    two files or folders differing only in letter case could mean, both named
{kind: "hierarchy", path, simulator: "ltspice"|"ngspice"|"ltspice:<name>", ngbehavior?, instance?, prefix?, cursor?}
    expanded netlist instances; `instance` is an exact reference-segment list
    selecting a subtree, and `prefix` is the `components` rule applied to
    each instance's own reference (the last segment), so a one-letter
    prefix still selects an element type. LTspice names are spelled for the
    session's LTspice, or the named build, as its raw names them
    (`profile.xvii_names`; `profile.names_read_as` and `names_note` say
    which recorded build the spelling follows when the build has none)
{kind: "model", mode: "search"|"enumerate", query?, libs?, cursor?}
    search requires query and fuzzy-matches it; enumerate requires libs and,
    given a query, lists the names containing it (case-insensitive). A
    search without libs reads the detected simulators' own model libraries;
    with a sandbox set, libs may name a file inside it or inside one of
    those libraries, so every source_path a search returns can be read back
    through libs. Both modes return the same row: name, type,
    source_path, include_directive (source_path as the server sees it, which
    is what staging reads), ports, params, and for a .MODEL its device_type
    and usage; a search adds score
{kind: "results", path? | job_id?, run_index?, case_id?, other_raw?: "fra_1"|"op",
 plot_index?: int|null, dialect?: str|null,
 view?: "plots"|"signals"|"table"|"measurements"|"native_tables", prefix?, limit?, cursor?}
    one RAW path the sandbox allows or one finished experiment case;
    other_raw reads the case's raw of that kind in place of its raw, RAW
    views only. plots returns the complete plot inventory in file order,
    with a stepped LTspice plot's kind (stepped_kind: stepped_sweep,
    stepped_op, stepped_tf, stepped_ac_one_frequency,
    stepped_noise_one_frequency, stepped_run_op) and the variables that
    hold its step values (step_variables); signals returns the selected
    trace descriptors, marking the axis; table returns the quantities of a
    table plot (no sampled axis), the first one included rather than taken
    as an axis, with complex values as {real, imag}. prefix filters
    signals and table by name. Each view is paged, and its cursor works
    only for the same file contents (snapshot_id), view and selection. For
    measurements and native_tables, path is a log path, the RAW is
    optional, and plot_index must be omitted. measurements pages the
    recorded values, range and AT points by value_index (not by step), and
    marks a value LTspice printed as failed `failed: true` (a FAIL'ed
    `.meas`, or `failed` in that row of its stepped table); the page's
    `failed_measurements` names each measurement with such a value.
    native_tables pages the value blocks ngspice prints in the log, as
    `name = value` entries and `Index frequency` rows, each with its
    block's first and last line (`line_start`, `line_end`), what ended the
    block (`closed_by`) and its printed labels; a unit the log does not
    print is left out. Neither adds plot or step identifiers or inferred
    formulas, and prefix filters literal measurement or quantity labels.
    The page includes `capture_facts` (whether each file was present, and
    its digest), `scan` (whether the whole log was read) and `section`
    (`parsed`, `absent` or `error` for this part of the log); a section
    that fails to decode is a query error, never an empty successful
    result.
{kind: "reference", query?, limit? (default 5, cap 20)}
    the tools' vocabulary: each tool's top-level arguments, plus the
    branches — recipes, ops, variation kinds, query kinds, checks and job
    actions. A plain-words `query` returns the closest entries with their full
    field tables; no `query` returns the table of contents, one line per entry
{kind: "guide", section?, heading?}
    the packaged guide: no `section` returns the core a
    session reads first, ending in an index of the topic sections and task
    playbooks; a `section` from that index returns that section. An unknown name
    fails the item as `unknown_section`, listing the names that exist. A
    `heading` with the `section` returns that heading's part of it, down to the
    next heading of the same or a higher level, matched by its text whatever its
    case, and the reply's `heading` names it as the section spells it. A heading
    the section does not have fails the item as `unknown_heading`, listing the
    section's headings under `supported`; a `heading` with no `section` fails
    as `invalid_query`. A pointer in a reply may name a heading:
    "guide section 'schematics', heading '...'"
{kind: "simulator_docs", name?, cursor?}
    the reference documents the simulator's vendor installs with it; for
    LTspice 26.1 and later, about fifteen Markdown files about the program
    (keyboard shortcuts, menus, the schematic file format, .MEAS, the waveform
    viewer, troubleshooting). No `name` lists them as {name, title,
    description} with the `source` directory; a `name` from that list returns
    the document as `sections`, each {heading, text}, cut at its second-level
    headings and paged by section. An unknown name fails as `unknown_document`,
    listing the names; an install with none fails as
    `simulator_docs_unavailable`
{kind: "open_in_ltspice", path?, pid?, cursor?}
    the documents open in the LTspice windows on this machine: `windows`
    (how many are running), `total`, and `designs`, each {path, kind:
    "schematic"|"netlist"|"other", active (the one in front in its window),
    pid, version, in_sandbox}, paged by `cursor`. The cursor is bound to the
    documents open, by window and path, so a page asked after one was opened
    or closed fails as `invalid_cursor`. A sheet the server may read adds its `sha256`
    and `differs_from_file`. When the window's copy differs from the file,
    `difference` names a few entries in a sentence, and `only_in_window` and
    `only_in_file` list the differing entries (a symbol with its
    attributes, or one other line), up to 25 a side, with
    `difference_omitted` counting the rest and a `hint` naming the call
    that reads them. That call names the sheet: with a `path` (a `.asc`) the
    query answers for the window that has it open, `pid` choosing one
    where more than one has, as {path, pid, version, sha256,
    differs_from_file}, and when they differ `difference`, `only_in_window`
    and `only_in_file` with every entry, both lists paged under one cursor
    (`page.collections` says which continues). The cursor is bound to the
    difference itself, and the window and the file are read again for every
    page, so a page asked after either copy changed fails as
    `invalid_cursor`. A path no window has open fails as `sheet_not_open`,
    and more than one window without a `pid` as `window_not_chosen`, naming
    the processes; `pid` without a `path`, or a path that is not a `.asc`,
    fails as `invalid_query`. Fails as
    `open_windows_unavailable` where there is no bridge to ask (not
    Windows, LTspice before 26.1, the setting off) and as
    `open_windows_unreachable` when the bridge does not answer
```

`path` is required except on `capabilities`, `symbols`, `symbol`,
`reference`, `guide`, `simulator_docs` and `open_in_ltspice`; `results`
accepts either `path` or `job_id`.

**Vendor documents.** `simulator_docs` reads the documents from the
simulator's install when asked, so they describe the installed build and are
available only where the simulator installed them; they are not part of the
guide. A name is looked up only among the files the directory lists.

**Open documents.** `open_in_ltspice` lists files; every tool still reads the
file, not the window's copy. When `differs_from_file` is set, a run or a check
on that path reads something other than what the user sees, and
`edit_schematic` refuses to commit while that window differs
(`open_window_differs`, §3.4). The window's copy is compared with the file but
never returned as a sheet. When a sandbox is set, a document outside it is
listed by name, but neither the file nor the window's copy is read. A query
that cannot be answered fails rather than returning an empty list.

**The guide.** `inspect(kind: "guide")`, `Api.guide()` and the
`spice://guide` resources serve the same text. A read through `inspect` or a
resource is recorded on the server's session. A read through `Api.guide()`,
inside `run_code` too, is not, and a restarted server starts with none. If no
read has reached the server since it started, its first tool reply contains one
reminder to read the guide, in the text and in the structured `hint`, worded
for that: an agent that has read it carries on.

---

## 4. Cross-cutting contracts

**Annotations.**

| tool | readOnly | destructive | idempotent | openWorld |
|-|-|-|-|-|
| `run_experiments` | false | false | false (without a repeated `request_id` the same arguments start new work; with one, the original job is replayed) | true |
| `jobs` | false | true (cancel) | true | false |
| `analyze_results` | false (artifact writes) | false | true | false |
| `edit_schematic` | false | true | false | false |
| `verify_circuit` | false (render, export) | true (export_to: sidecar) | true | false |
| `inspect` | true | false | true | false |

**Ownership.** Any caller can read every persisted job; only the owning
process or a holder of the control token can cancel one, and no read returns
the token.

**Caps.** Findings, failures and observations arrays are capped, and
truncation is always declared.

**Untrusted parses.** Every raw or log parse, including the one made when a
run finishes, runs in a separate process under a deadline and a memory limit
(2 GiB), in addition to the whole-call budgets. A source whose files are
unchanged is answered from cache (§3.3).

**Schemas.** Every field a caller can set is in the tool's `inputSchema`,
which may use `oneOf`, literal discriminants, `additionalProperties: false`,
branch-local defaults and `$defs`. MCP resources are documentation and do not
define the schema. A property named `title` is published like any other field.

**Evolution.** The defaults, units and meaning of a variant (one value of a
discriminant, such as a recipe's `metric`) never change. New semantics get a
new discriminant; an unknown one returns `unsupported_variant` with the
supported list.

**Shorthand values.** The input schema declares each of these, so a client
can discover them:

| tool | field | added spelling |
|-|-|-|
| `verify_circuit` | `render` | `true` = default policy; `false`/omitted = no render |
| `analyze_results` | `include` | a bare list of flag names becomes `{name: true}` |
| `analyze_results` | `include.per_run` | `true` = the default page |
| `run_experiments` | `analyze.include` | the same two spellings |

These are shorthand values, not alternative argument names; no argument on
the surface has an alias.

An unrecognized flag name fails with the list of valid flags, noting that
`fields` takes row paths rather than a boolean. A `render` value that is
neither a policy nor a boolean is refused, and the error lists the accepted
forms and the `mode` values.

**Progress.** A call that asks for progress (a `_meta.progressToken`) and
waits on a job, `jobs` wait or resume or `run_experiments` holding the call
open, gets a `notifications/progress` when the wait begins, then at most once
a second while more of the job's cases have finished, and as the wait ends if
more have: `progress` is the number of cases finished, `total` the cases the
job expanded to (left out until it is known), and `message` says both with
the job id. A client may restart its timeout on each one; none is sent while
no case finishes, as during one long case or the job's attached analysis.
A call's reports only rise, however many waits it makes, and none is sent
once the call is answered, even by a task it started, such as the job's
coordinator. No other call reports
progress, and a call that did not ask is sent none.

**Server logging.** The server logs only to stderr, at `WARNING` by default;
`[logging] level = "INFO"` or `LTSPICE_MCP_LOG_LEVEL` sets another level. It
neither advertises nor serves the MCP logging capability
(`logging/setLevel`, the `logging` capability and `notifications/message`),
and `logging/setLevel` returns method-not-found. Everything a caller needs is
in the tool result.

At `INFO`, the `ltspice_mcp.events` logger writes one record for each tool call,
with a structured payload on the record's `ltspice_event` attribute:
`event: "tool_call"`, `door`, `tool`, `duration_s`, `outcome`, the error's
`code`, `reply_chars`, the length of the result as sent, `fields` and
`choices`. `door` is `mcp`. `outcome` is `ok` or `error` for a call that
returned a result, `cancelled` for one the client cancelled, and `raised` for
one answered with a protocol error, such as an unknown tool name; those two
have no `code` or `reply_chars`. A tool name the server does not serve is
recorded as null.

`fields` lists the argument fields the call set, as sorted dotted paths, each
once: nested fields count, list positions and mapping keys are dropped, and a
field left out or sent at its default value is not set. The fields of an item
tagged as one member of a union are named under its tag, so a call that
batches several kinds says which set what: `queries[kind=hierarchy].cursor`,
`recipes[metric=value].spec.allow_incomplete`, `ops[op=add_component].x`. An
item the handler refuses on its own (one `inspect` query, one recipe) is still
named, by its tag and those of its keys its member declares. `choices` lists,
as `path=word`, each value that picked a word of the server's own vocabulary,
a `Literal` or enum of the field's type: `action=wait`,
`queries.kind=simulator_docs`, `recipes.metric=plot`,
`recipes[metric=value].reduce=mean`, `ops.op=add_component`. A call refused
before its arguments validated has both null.

The record carries no argument value: no path, file name, label, signal,
expression, number, cursor, job or result-set id, mapping key or free text,
and no part of the reply. Results the reply reports, such as an
`indeterminate` spec verdict or a cut `fail_cases` list, are not recorded;
what the call asked for to reach them is (`spec.allow_incomplete`,
`include.per_run`). A field typed as free text records its name only, even
when its value happens to be a word the server also uses (`group_by`).

A resource read writes `event: "resource_read"` with the URI `template` it
matched (`spice://results/{job_id}/signals`, `spice://config`), never the URI
itself, and a prompt get writes `event: "prompt_get"` with the `prompt` name,
never its arguments; each has `duration_s`, `outcome` (`ok`, `error` for a
read or get answered with a protocol error, `cancelled`, `raised`) and the
error's `code`. A URI or prompt name the server does not serve is null.

A Python API operation (`run_experiments`, `jobs`, `analyze_results`,
`inspect`, `edit_schematic`, `verify_circuit`, also what a `run_code` snippet
calls) writes the same `tool_call` record with `door: "python_api"`, the
names in the arguments as the caller wrote them (before the automatic mode
drops a presentation control or submits with `execution.wait_s` of 0),
`reply_chars` null, and `controls`, the API's own options it set (`raw_page`,
`wait`, `detach`). An operation the API runs for its caller, such as the
polls of `wait` or of `run_experiments` waiting for its job, writes none.
A call the automatic mode refuses (one that sends a paging control) still
names what the caller set, since the names are read before that check. An
API error that carries a tool's error payload is `error` with that payload's
`code`, a refused argument is `error` with `invalid_arguments`, and an
interrupt is `cancelled`. The API's other methods (`wait`, `load_raw`,
`measurements`, `device_op_points`, `reference`, `guide`) write none. A
`run_code` snippet's records reach the server's stderr through its worker,
which logs at the server's level. A detached run (`detach=True`) is submitted
again by the process spawned to own it, an ordinary API caller, so its log
holds a second `run_experiments` record for the same job.
The same logger writes a job's lifecycle events.

---

## 5. Out of scope

The six tools do not edit deck text, offer temperature or lib-section sweep
parameters, autoroute or tidy a drawing, rate results as trustworthy, keep
library session state, provide per-wire interactive tools, write partially to
a named target, or return unbounded collections. They do not validate
testbench physics. The linter checks SPICE and does not check the measurement
setup.

A circuit to run or check is passed as a file path; only a comparison's
`reference` also takes netlist text. A client without file tools writes the
netlist with `run_code`. The surface assumes an agent on the same machine as
the server.

---

## 6. The linter

Each lint rule has a `rule_id` and a disposition, `blocking` or `warning`,
which its findings report as `severity` (`error` or `warning`).
Only rules for deterministic failures recorded from real simulator runs block.
Suppression is per call, and `linter_version` is reported in provenance.

| rule | disposition | condition |
|-|-|-|
| `save-meas-coverage` | blocking | a `.meas` reads a signal the run's simulator will not store, as the build that runs it reads names (`device_names.deck_signal`). Under a `.save` list (no `.save` saves all (`all`, `*`), names a signal by a pattern, or cannot be read): a signal no `.save` stores. A voltage between two nodes, `V(a,b)`, is its two nodes in both (both builds measure it from a saved `V(a)` and `V(b)`, and `.save V(a,b)` saves both); a voltage against ground, `V(a,0)`, stands as written, since neither build measures it with only `V(a)` saved. A name inside subcircuit instances is stored only in the build's own spelling: LTspice 26 with each instance's X (`xa:xin:mid`), XVII without (`a:in:mid`), ngspice with dots (`xa.xin.mid`); with or without a `.save` list, a `.meas` of a name LTspice spells otherwise, or whose first instance names no instance in either spelling, reads nothing on either build. Names are compared in ASCII case only. Where the build's spelling is not established, the signal is reported by `save-meas-spelling-unestablished` instead |
| `save-pattern-unrecorded` | warning | any `.save` of the deck names signals by a pattern (`V(*)`, `I(R*)`), no `.save` saves all or cannot be read, and a `.meas` reads a signal no `.save` names: which signals a pattern saves is not recorded, so the signal is reported as not checked rather than refused, whichever kind of signal the pattern names |
| `save-name-not-stored` | warning | a `.save` name the run's simulator stores nothing for: inside subcircuit instances, LTspice's other build's spelling, a first instance that names no instance in either spelling, or on ngspice a path spelled with colons, which ngspice-42 drops without a word. The finding gives the simulator's own spelling where it is known |
| `save-meas-spelling-unestablished` | warning | a `.save` or `.meas` name inside subcircuit instances whose reading by the run's simulator is not established, so no rule refuses on it: a dotted path in an LTspice deck, a deck for an LTspice build whose spelling is not recorded or not known, or a first segment that names an instance in both spellings or one exported with a `§` |
| `meas-subckt-node-unsaved-ltspice` | warning | LTspice, a deck with no `.save` list: a `.meas` of a node inside a subcircuit, spelled as the build stores it, which LTspice keeps only with its "Save subcircuit node voltages" setting, off by default on both builds; the measurement fails without it |
| `meas-ngspice-batch` | warning | ngspice: the deck runs and only the top-level `.meas` is skipped, which the run reports when it is read |
| `meas-trig-degrees` | blocking | LTspice: a `.meas` that calls `sin`, `cos`, `tan`, `asin`, `acos`, `atan` or `atan2`, whose angle LTspice 26 and XVII read in degrees on their default settings where a B source reads radians; `verify_circuit`'s `syntax` check reports the same as `meas_trig_degrees` |
| `lib-section-ngspice` | blocking | ngspice in a mode it reads with its `lt`, `ps` or `s3` flag set, `kiltpsa` (the default) among them: a `.lib <file> <section>` in the deck or any file it reaches |
| `lib-without-section-ngspice` | blocking | ngspice in a mode that does not read `.lib` as an include (native, `hsa`, any mode with `s3`): a `.lib <file>` that names no section, which ngspice stops on ("unimplemented dot command '.lib'"), or an `.include` of a file that declares sections, whose own section line it stops on. The finding names the fix: a section the file declares, or `.include` for a file with none |
| `lib-sections-dropped-ngspice` | blocking | ngspice in a mode that reads `.lib` as an include (`lt` or `ps` set, `kiltpsa` among them): a `.lib <file>` with no section, or an `.include`, of a file that declares sections. ngspice reads the file's own `.lib <section>` line as an include of a missing file and reads none of the file's cards; ngspice-42 then runs on without them (a wrong number), ngspice-45 stops. In the deck or any file it reaches |
| `analysis-count-ltspice` | blocking | refused by LTspice 26 and XVII: two of `.tran`/`.ac`/`.dc`/`.noise` (`.op` may appear with one) |
| `meas-function-ltspice` | blocking | refused by LTspice 26 and XVII: `vdb()`, `phase()` or `group_delay()` in a `.meas` |
| `meas-name-ltspice` | blocking | refused by LTspice 26 and XVII: a `.meas` named `e`, `k`, `pi` or `q`, constants of the expression evaluator |
| `meas-unclosed-paren-ltspice` | blocking | a `.meas` with a parenthesis left open. LTspice 26 refused the recorded one before the run and ran nothing, and XVII ran that deck without the measurement; any such `.meas` is refused, since LTspice 26's error names the parenthesis, not the form. `verify_circuit`'s `syntax` check reports the same as a `directive_syntax` error, and `edit_schematic` refuses to write it |
| `lib-section-ltspice` | blocking | refused by LTspice 26 and XVII: a `.lib` line of more than one word that names no file. LTspice reads the rest of the line as one path, quoted or not, so `.lib file section` looks for a file named `file section` and stops; a path holding a space that does exist runs |
| `step-without-values-ltspice` | blocking | refused by LTspice 26 and XVII: a `.step` with nothing after it, or `.step param <name>` with no values. `edit_schematic` refuses to write it and `verify_circuit`'s `syntax` check reports it |
| `byte-85-ltspice` | warning | LTspice: a byte 0x85 in an 8-bit deck or include with a card after it on the line, which LTspice 24 and later read as a line break and XVII does not; not reported when the session's LTspice is known to be XVII |
| `node-control-byte-ltspice` | warning | LTspice: a node name in an 8-bit file containing a byte from 0x80 to 0x9F, a control character to LTspice, which LTspice 26 refuses and XVII saves under a name the server spells differently |
| `model-missing` | blocking at staging | a missing model; the model token is read after a BJT/JFET/MOSFET area factor or `off`, and before a subckt call's `params:` |
| `element-arity` | blocking | fewer nodes than terminals |
| `bsource-value-prefix` | blocking | no `V=`/`I=`, or on LTspice `R=`/`P=` |
| `value-keyword-ltspice` | blocking | LTspice: `C=`/`L=` as the primary value |
| `value-expression-remnant` | warning | tokens left after the first `key=value`, which a value edit would drop |
| `include-relative` | warning | an include or library reference with a relative path, which depends on where the deck is placed |
| `suffix-mega-milli` | warning | a number followed by an upper-case `M` that does not start `Meg` or `Mil`, which SPICE reads as milli |
| `step-ngspice` | warning | ngspice: a `.step`, which ngspice ignores in batch mode, so the deck runs once at the base value |
| `node-non-ascii-ngspice` | warning | ngspice: a node name holding a character beyond ASCII, each byte of which ngspice-42 replaces with `_`, so two names that differ only there are one node, shorted with no word in its output |
| `temp-as-param` | blocking | a deck that sweeps temperature as a parameter (§3.1) |
| `duplicate-param` | warning | an instance, `.model` or `.subckt` card giving one parameter more than once, in any case; which copy the simulator reads is not known, and `verify_circuit` reports the same as `duplicate_parameter` |
| `subckt-defined-twice` | blocking | a subcircuit name defined more than once at the top level of the deck, which the reader refuses: LTspice 26 (recorded). `verify_circuit`'s `syntax` check reports the same as `subckt_defined_twice`. ngspice (the first) and LTspice XVII (the last of two) run a definition, which every check reads, and get no finding |
| `subckt-defined-twice-unrecorded` | warning | a subcircuit name defined more than once in one scope, where what the reader does is not recorded: an LTspice build other than 26 and XVII, one whose build is not known yet, QSPICE, Xyce, or for LTspice two definitions inside one subcircuit or more than two of one name. No check reads any of the bodies; `verify_circuit` reports the same as `subckt_defined_twice_unrecorded`. A definition repeated across the deck and an include is not linted |
| `unreadable-card` | warning | a card no rule can split into its parts: an unclosed quote, brace or parenthesis, or a closer with nothing open. The other rules leave it out. On a part, `.subckt` header or `.global` its words count as connected and defined nodes and names, so no rule reports something on a guess about it; a variation or random rule on a part, `.model` or `.param` name a card like it may declare is refused (`ambiguous_target`). It covers the deck and its staged includes, and an include card in a live include, for which `model-missing` then says nothing. What a simulator does with such a card is recorded only for the `.meas` of `meas-unclosed-paren-ltspice`; `verify_circuit` reports the same as `unreadable_card` |
| `value-suffix-mojibake` | blocking | a suffix showing the file was decoded in an encoding it was not written in, such as the `Âµ` a UTF-8 `µ` becomes under cp1252; the simulator reads the bare number, off by a factor of 1e6 |
| `value-suffix-nonascii` | warning | any other symbol after a number, such as `10Ω` or `25°C`, read as the bare number, which is usually the intended value. Staging spells a `µ`/`μ` as `u` before the deck is linted; for a deck run elsewhere `verify_circuit` reports it as one observation per file, or as a warning per value when a reader the server knows of (LTspice XVII) would decode the file differently |

`element-arity`, `bsource-value-prefix`, `value-keyword-ltspice` and
`value-expression-remnant` come from one check of each element line, and each
is a separate rule so that suppressing one never silences another. Every rule
reads the cards the simulator reads: a card continued on `+` lines is one
card, and neither line 1 of the deck, the title line both simulators skip, nor
a card after `.end` is read. `verify_circuit`'s `syntax` and `quality` checks
read the deck the same way.

---

## 7. Appendix A — payload grammars

The variation and recipe unions are strict discriminated unions, inlined into
each tool's `inputSchema`.

### A.1 Variation

A case's id is `<circuit id>-case-<number>` (`variations.format_case_id`): the
number counts that circuit's cases from 0, padded to four digits and wider
past 9999, and names the case's staged deck (`case-<number>` and the deck's
own extension, `.cir`, `.net` or `.sp`). The circuit
id is the circuit's `id`, else its file stem made valid
(`sanitize_circuit_id`), with `-2`, `-3` for a repeated stem
(`derive_circuit_ids`, said in a `circuit_id_derived` observation). A run
row's `run_index` counts the whole job's cases from 0, circuits in the order
given. The order (`expand_variations`): circuits in the order given; entries
in the order given, the last changing fastest; inside one grid entry the last
target changing fastest; a zip entry pairing its targets; the samples of a
`random` or `pdk_native` entry changing fastest of all, wherever that entry
stands. A resume copies the job's cases and expands nothing again, so each
keeps its id, run index and order. These are a contract a caller may write
ids against before a job runs; `tests/test_case_ids_and_order.py` names the
test that pins each.

```
{kind: "assign", id?, combine: "grid"|"zip" (default grid),
 applies_to?: [circuit id],
 assign?: dict[target -> list[number | SI-suffix string]],
 instances?: [{instance: [reference segment], attribute: "value"|"model"|"parameter",
               parameter?: name, values: [number | string]}]}

  Target resolution is deterministic, with no implicit precedence.
  Syntactically explicit forms are selected before bare-target resolution:
    "REF@model", or the glob "M*@model"  -> instance model swap
       (the corner idiom: {"M*@model": ["NTT", "NSS", "NFF"]})
       A name the swap writes (here, or an `instances` target with
       attribute "model") is held to what recordings show LTspice read:
       letters, digits and `_ . : -`, starting with a letter, digit or `_`,
       and not a number with at most a scale suffix (`2222`, `1k`, `1e3`;
       `2N5486` and `1k5` have more after the number and are accepted);
       any other is refused at input (`spice_lex.written_model_name_refusal`).
       A Monte Carlo `rule: "model"` names a card the deck declares, so it
       takes any name the lexer reads (`spice_lex.is_model_name`).
    "INSTANCE:{delvto|mulu0}"            -> explicit per-instance mismatch
       value; "X1:delvto" for a single-FET body, or the qualified
       "X1.M0:delvto" for a multi-FET body. Supported only through
       exactly one X -> M wrapper level as this convenience spelling, and
       only where the simulator is known to apply the parameter to the
       device's model (`subckt_mismatch.require_instance_parameter`): on
       ngspice a BSIM3/4 model (`unsupported_model` otherwise); on LTspice
       `delvto` on a level 54 model, the one its recordings show taken
       (`instance_parameter_not_taken` where they show it refused or
       ignored, `instance_parameter_unestablished` where none covers the
       level). Use an ordered `instances` selector
       for arbitrary supported depth; never construct a selector by joining
       segments with punctuation. Refused on QSPICE and Xyce
       (`mismatch_inside_subcircuit_unmodelled`).
  Otherwise, a declared .param name -> parameter substitution;
  else a component reference        -> value substitution;
  else                              -> ambiguous_target.
  A part LTspice exports as `R§Load` is also named by `RLoad` (its name on
  an ngspice copy, and in verify_circuit) and by `Load` (its name on the
  sheet and in LTspice's results); a name two parts answer to is refused as
  ambiguous_target. The same holds for a random `component` rule's target.
  Every target, and every rule's match, is resolved among the cards the
  simulator reads: a part, `.param` or `.model` named only by the deck's
  title line or after its `.end` is not one, so a target naming it is refused
  as one that names nothing.

  At least one assign or instances target is required. `parameter` is required
  only for attribute "parameter". An `instances` target selects one instance
  of the expanded netlist; the case's staged deck gets a renamed copy of each
  subcircuit on that instance's path, so other instances of those subcircuits
  and the original files are unchanged. Conflicting fields, and writes to both
  an instance and an instance inside it, are refused.
  combine: "zip" requires equal list lengths across both target forms.

{kind: "pdk_native", id, runs: int >= 1, seed: int >= 0,
 profile, mode: "nominal"|"mismatch"|"process"|"combined",
 sample_start: int >= 0 (default 0), applies_to?: [circuit id]}

  Each circuit takes at most one pdk_native or random entry. pdk_native ids
  are unique; seed is a required integer (not a float or boolean) no greater
  than 2**63-1. Run i of the entry draws sample index sample_start + i,
  whatever order the cases run in. See the native PDK contract below for the
  supported profile and the deck it needs.

{kind: "random", id?, runs: int >= 1, seed?: int, sample_start?: int >= 0,
 applies_to?: [circuit id], rules?: [RandomRule], simulator_draws?: bool}
                               at most ONE random entry per circuit
                               rules may be empty only with simulator_draws

RandomRule:
  {rule: "component", target: ref | glob, tolerance, scale:
   "relative"|"absolute", distribution: "normal"|"gaussian"|"uniform"}
     tolerance is a plus-or-minus bound; normal is N(0, tol/3) truncated at
     plus-or-minus tol (the 3-sigma convention); relative is a fraction of
     nominal, absolute a bound in source units. There is no lognormal
     distribution.
  {rule: "param",  target: param, same fields}
  {rule: "model",  target: model, param: model-param, same fields}
  {rule: "mismatch", instance?: [reference segment], ...}
                            exact instance selection, or prefix, never both.
                            A MOS path selects one device; a wrapper path
                            selects every MOS device inside it. The caller
                            supplies the Pelgrom coefficients and area
                            floor; they are not read from a PDK's models.
```

The `INSTANCE:{delvto|mulu0}` assignment form is never a random-rule target.
Random mismatch on a top-level MOS (an `M` card in the deck) gives each
device its own copy of its `.model`, with the drawn threshold shift and
current-factor scale applied to the parameters the rule names; an `M` card
of the same name inside a `.subckt` body is another device and is left as
it is. A
`rule: "mismatch"` can also select a MOS inside subcircuits by its exact
instance list, or every MOS inside a subcircuit instance by that instance's
list. On QSPICE and Xyce, whose reading of a subcircuit body the server does
not model, such a rule (an `instance` list, or a `prefix` that names an `X`
instance) is refused with `mismatch_inside_subcircuit_unmodelled` before any
case runs; run the deck on LTspice or ngspice. On ngspice such a device gets its draw as `delvto` and `mulu0` instance
parameters, which needs its model to resolve to a BSIM3 or BSIM4 model.
LTspice takes neither `mulu0` nor `delvto` on BSIM3, so there the device gets
its own copy of its `.model` inside a private copy of its subcircuit, as a
top-level device does, which needs a model with one definition; a binned
model is refused with `model_missing`. The draw uses the device's W and L
after `assign` variations are applied. The area a draw stands for is `m·W·L`,
because the simulator builds m parallel copies from one card and they share
the draw. LTspice ignores `m` on an X line, so there m is the device's own.
ngspice, as measured on ngspice-42, drops the subcircuits no instance reaches
and then reads the deck once in order, includes and `.lib` sections in
place. An X line that carries `m` makes it rewrite the subcircuit called,
unless that one declares `m` (on its `.subckt` line or in a body `.param`):
the subcircuit gets an `m` parameter, default 1, and `m={m}` is appended to
each C, D, F, G, I, J, L, M, Q, R, X and Z card of its body; an A, B, E, H,
K, S, T, V or W card is left as written, and a card of another letter there
has an unknown count. An X line so given
`m` rewrites its own callee in turn only if it sits later in the deck than
the line that rewrote its body. The appended `m` replaces a card's own
outside the HSPICE modes (the `kiltpsa` spicelib runs by default) and
multiplies it in them (`ngbehavior` `hsa`); an X line's `m` on a subcircuit
that declares `m` is an ordinary parameter. So m is the device's `m` after
that rewrite, read in the scope its X lines set. A deck where the rewrite
gives `m` to an X line calling a subcircuit that takes no parameters is
refused with `multiplier_rewrite_stops`, as ngspice stops on it. An inner `m={mult}` resolves in the scope
its X line sets. An `m` that does not resolve to a positive number refuses
the request with `mismatch_geometry_unresolved`. On ngspice, a case whose
edits, the private subcircuit copies a nested target needs among them, would
change what a card no edit writes builds is refused with
`multiplier_rewrite_changed`, naming a fix where one keeps the circuit. A deck with no semantic profile (QSPICE, Xyce)
counts the product of the X line's `m` and the device's own; neither
simulator's rule is measured. Draws across circuits are independent;
correlated Monte Carlo across decks is not supported.

A random case's draws follow from the seed, its circuit id and its sample
index, never from the case's position among the request's cases. Every case
of one sample draws the same deviation whatever its `assign` values, so a
sweep compares the same die at each corner, and adding a corner leaves the
other draws as they were. The value a deviation gives still follows what the
case assigns: a relative tolerance scales with the assigned nominal, and a
mismatch draw as 1/sqrt(m·W·L) of the assigned geometry. Sample indices start
at `sample_start` (default 0; an integer, not a float, boolean or string), and
with the same seed and circuit id `runs: 1` and `sample_start: k` replay
sample k alone. An entry without `seed` gets one the server chooses after it
has looked the request up for a replay, so a retry under the same
`request_id` returns the job it made. Every case the entry expands to records
its sample index as `_random_run`, its seed as `_random_seed`, when the
entry has one, its `id` as `_random_id` and, with `simulator_draws`, its
ngspice seed as `_random_simulator_seed`, in its assignments, a case skipped
before it ran included. A circuit whose deck fails before its variations
expand (an include that cannot be staged) records no assignments on its
cases; their failure says why.

`simulator_draws: true` (default false) also makes ngspice draw the random
functions written in the deck (`agauss`, `gauss`, `aunif`, `unif`), a PDK's
statistical models among them, anew for each sample. Each case's deck gets
`.options seed=<n>` before its `.END`, which ngspice reads before it evaluates
the deck's parameters (since ngspice-30, `inp.c`). The seed is derived from the
entry's `seed`, the circuit id and the sample index, in `1..2147483646`, so
every case of one sample runs with the same seed and `runs: 1` with
`sample_start: k` replays sample k's draws too. Same seed, same deck and same
ngspice build give the same draws; a corner that changes which random
functions the deck evaluates, or their order, changes what that seed draws.
The entry's `rules` then may be empty, or add the server's own draws to
ngspice's. A run on another simulator, or with `execution.simulator_seed`,
which seeds every case alike, is refused with `simulator_draws_unsupported`
before any job is recorded. A deck whose own option cards set `seed=` (outside
a subcircuit no X line calls) fails its cases with `deck_sets_simulator_seed`,
naming each card: ngspice reads
every seed option and the first in the deck wins (measured on ngspice-42), so
the deck's seed would fix every sample's draws or be overridden, depending on
where it sits.

The receipt compares the samples of a `random` or `pdk_native` entry by what
they computed, one `assign` point at a time. Each produced sample keeps a
digest of its raw: every plot's variables and points, a complex frequency or
other sampled axis by its real part where its imaginary part is one no
simulator computes as a coordinate (zero, or the subnormal number ngspice 42
writes) and by both parts otherwise, and nothing else of its headers. Where every sample
compared at a point computed one result, an `identical_samples` observation
(`kind: "sampling"`) lists the point in its `evidence.groups` as `circuit`,
`assigned` (that point's `assign` values, left out when the entry has none)
and `samples` (how many were compared): their draws changed nothing the
simulator saved there. A point whose samples give two results or more is not
listed, since some draw changed what was saved. A circuit whose every
compared point computed one result is one group with its `points`. The
detail speaks for each circuit: "All samples of …" when every sample of every
point ran and was compared, "Every sample of … that was compared" when some
were not, and "Samples of … computed the same results at the assign points
listed for them" when other points differ; on a job with runs still to finish
it says "so far". At most ten points are listed, in the order the cases ran,
with `points_truncated` giving how many there are. A produced sample with no
digest (recorded before samples were compared, or a raw the reader cannot
walk to its end) at a point with another produced sample is counted in a
`samples_not_compared` observation. Neither observation appears when every
sample was compared and no point computed one result.

### A.2 Recipe

There are 22 recipes (discriminant values). The shared fields `key` (required
and unique), `sources?`, `reduce`, `field` and `spec` are accepted only where
the recipe's reducer category (scalar, multi-field, keyed or variable-length)
allows them; non-reducible recipes reject `reduce` and `spec` at validation.
`.step` selection is not a recipe field; it is set once per call (§3.3).

`field` names the number a `reduce` or `spec` reads, with the same meaning in
every category. A multi-field or keyed recipe requires it for `spec`, which is
one verdict on one number; when given, it also restricts `reduce` to that
field. Without it, `reduce` covers every field or key, one row per field and
statistic. Two names for one number (`edges`' `rise_time` and `fall_time`, a
disturbance's `deviation` and `undershoot`) are reduced once. An edge is
reduced under the direction its row measured, and the others under the first
name. A scalar recipe takes no `field`, since it has one number.

Without `field`, the fields `reduce` covers depend on the category. A keyed
recipe's keys are what the run asked for (each `.meas` in the deck, or each
quantity its bias point reports), so every key in any row is reduced, and
a `.meas` that failed in every run reduces to null with its `no_value_count`
(§3.3). A multi-field recipe's fields are a fixed table, and a circuit may
have no value for some of them (a lowpass has no low cutoff, a step response
no disturbance figures); such a field is omitted when no row has a number for
it, and reported when the caller names it.

| discriminant | run type | own required fields | notes |
|-|-|-|-|
| `summary` | any | — | full summary payload: sim type, ranges, signals, measurements, Fourier, AC bandwidth, diagnostics. Fourier is each `.four` table as the build printed it, phase included; LTspice 26 and XVII print phase in different conventions. What the log says (measurements, `failed_measurements`, Fourier, diagnostics) is the whole run's at any `step`; the step selects the range, `point_count`, `temp_c` and AC bandwidth. `errors`, `warnings` and `meas_errors` are built from the run's conditions, the source of `diagnostics`: each distinct line the simulator wrote an error or a warning word on is listed by that word (`log_conditions.said_lines`, `said_as`), a line with no word is in neither list, and a relayed line carries that word as its `severity`, or none. Past 50 distinct lines of one condition, `lines_left_out` gives how many more there were, by code. For QSPICE and Xyce, with no recorded wording, a line is read by the generic patterns the server used before it had recordings, under `simulator_error` and `simulator_warning`. A run whose results file holds no point (the simulator stopped before its first solution: `ltspice17/log/err_timestep`, `ltspice26/log/err_source_loop`) is summarised from its log alone, with `results_hold_no_point: true` and no plot fields; with no log beside the file, `log_not_found: true` |
| `measurements` | any | — | `names?`, `histogram_bins?` (0 = none); per run, `stats` for each `.meas` name, keyed lower-cased as the log reader holds it. A name in `names` or `field` is matched in any case, and one the log does not hold fails the run with the names it holds. A run with one value per name also has `measured` {name: value} and `at`, and a null there is named under `failed_measurements` (LTspice printed it failed) or `unreadable`, and a value printed once for a stepped run carries `printed_once`; its default row shows only those (§3.3) |
| `value` | any | `expr` — one trace or node pair, not arithmetic | `at`, required when the axis has more than one sample; reads the nearest sample, no interpolation; a bias-point run is read by name; step-aware |
| `signal_stats` | tran, dc, ac, noise | `signal` | `window?`, on tran and dc only; `quantiles?`, tran only — levels in [0, 1], weighted by time, not by sample; each adds `q<percent>` (0.99 is `q99`, 0.999 is `q99_9`) and two or more add `quantile_peak_to_peak`, all reducible. `min`/`max`/`peak_to_peak` stay the sample extremes. An AC or FRA run gives the magnitude in dB and the phase; a noise run has no `mean`. `metrics.signal_stats_defined` decides which runs it reads, over a window or whole |
| `edges` | tran | `signal` | `levels?`, `edge?`, `window?` |
| `timing` | tran | `from{signal, edge, level}`, `to{...}` | `nth?`, `window?` |
| `periodic` | tran | `signal` | `window?`; period, frequency, duty cycle |
| `transient_response` | tran | `signal`, `mode: "step"\|"disturbance"` | `input` is required for `disturbance` and rejected for `step`; `window?` |
| `thd` | tran | `signal` | `fundamental_hz?`, `harmonics?` (default 7, held to 50), `window?` |
| `tone` | tran | `signal`, exactly one of `frequency_hz` / `frequency_param` | `frequency_param` names a parameter whose value at the step read (its `.step` value) or else the case's assignment is that run's frequency; a run with neither is refused. `window?`; cut to the whole periods of `frequency_hz` that fit from the window's first sample and resampled by linear interpolation onto evenly spaced times, as `thd` does by default, so samples weigh by time. `amplitude` (peak, signal unit), `phase_deg` in (-180, 180] for `dc + amplitude*sin(2*pi*f*t + phase_deg)` with `t` the run's own time from zero, `sin_component`, `cos_component`, `dc`, plus `t_start`, `t_end`, `n_cycles`. Reducible leaves: `amplitude`, `phase_deg`, `dc`, `sin_component`, `cos_component`; a spread of `phase_deg` that straddles 180 degrees reduces the wrapped value, the components do not wrap. A ratio `A/B` (or `1/A`) is the phasor of `A` over that of `B` over the same whole periods: `amplitude` is the gain, `phase_deg` the shift, the components the quotient's real and imaginary parts, and there is no `dc`; a denominator with no component at the frequency is refused as singular, and one with under 1e-3 of its peak there is warned of |
| `time_steps` | tran | — | reads the time axis, so no `signal`; `threshold_s?`, `window?`. Per run and per step: `points`, `min_step` and `min_step_at` (where it starts), `max_step`, `median_step`, `zero_steps` (pairs of stored points at one time, never a step), `t_start`, `t_end`; with `threshold_s`, `steps_below`, `first_below_at`, `last_below_at`. `stored` is `services.stored_time_points`' answer: `accepted`, `reduced` (LTspice XVII under waveform compression), `print_step` (ngspice under `.options interp`, read from the deck and the option cards ngspice reads, `ngspice_driver.read_option_cards`) or `not_checked` (LTspice 26, an unknown build or simulator, an unread card, or a run saved from a `.tran` start time), and every answer but `accepted` adds a `time_steps_coverage` observation (kind `coverage`) to the reply. ngspice stores no step it rejected (measured with `.options acct`); for LTspice that is not established. Reducible leaves: `min_step`, `median_step`, `max_step`, `points`, `zero_steps`, and `steps_below` with `threshold_s` |
| `bode_filter` | ac | `signal` | filter characteristics: fc, bw, Q, type; a cutoff is read at the half-power level (-3.0103 dB) below the passband |
| `bode_point` | ac | `signal`, `at_hz` | gain and phase at a frequency |
| `bode_crossing` | ac | `signal`, exactly one of `level_db` / `level_deg`; `ref_hz` with `level_db` | `level_deg` scans the unwrapped phase, so a crossing past 180 degrees is found once rather than at every wrap; `phase_deg` remains accepted as an alias. `ref_hz` measures `level_db` from the gain at that frequency, read as `bode_point` reads it, so the half-power point below a reference is one call; the value's `level` is then the absolute level crossed, and its `reference` gives `frequency_hz`, `magnitude_db` and the relative `level_db`. A `ref_hz` outside the sweep is refused |
| `bode_slope` | ac | `signal`, `from_hz`, `to_hz` | dB per decade |
| `stability` | ac | `signal` | phase margin = 180 degrees + phase at unity gain, positive being stable. Reducible leaves: `phase_margin_deg`, `gain_margin_db`, `unity_gain_hz`, `dc_gain_db`. On FRA results the crossover and phase margin are compared with the ones LTspice printed for the same analyzer and step: the reply's `observations` carry `fra_margin_differs` (kind `reconciliation`, both values and the forms of the signal whose margin agrees) when they differ beyond the printed digits, or `fra_margin_not_checked` (kind `coverage`, with why) when the log printed none; nothing when they agree |
| `ac_structure` | ac | `signal` | structural poles and zeros |
| `resonance` | ac | `signal` | resonant frequency, Q, peak; the bandwidth Q is taken from is read at the half-power level below the peak |
| `return_loss` | ac | `signal` | `z0?` (default 50); return loss in dB, VSWR, reflection coefficient |
| `noise_integral` | noise | — | `signal?`, `from_hz?`, `to_hz?` |
| `operating_point` | op | — | `device?` scopes the reply to one device, named by its path (`x1.m1`) on either LTspice build; `device_names` maps each device inside a subcircuit to its name in the log's table and in the raw |
| `waveform` | any | `signals` | `max_points?` (default 2000, held to `[analysis] max_points`), `format: "inline"\|"csv"`, `window?`. Inline is bounded decimation only, with `points_returned` / `points_total` declared and the axis sent once as `x`; a budget may thin it further; `csv` returns an artifact handle |
| `plot` | any | `signals` | `title?`, `log_x?`, `span?` — returns an artifact handle |

A `signal`, `signals` entry or `expr` names one trace as the raw stores it, or
a node-pair voltage `V(a,b)`. No simulator writes a pair as a trace, so every
reader computes `V(a) - V(b)` from the same raw, step and axis, complex on an
AC run. `V(a,0)` and `V(a,gnd)` are `V(a)`, a missing node is named, and a
`.noise` run refuses a pair because spectral densities do not subtract. The AC
recipes and `tone` also accept a ratio `A/B` of two such signals, a reciprocal
`1/A`, and one leading `-` that negates the whole expression
(`metrics.signal_expression` is the one reader of these; no other operator is
read). Other
trace math is done in Python on `RawResult.trace`. A not-found error for an
input that looks like an expression includes the Python snippet and names
where to run it, which is `run_code` when the session serves it and the
library otherwise.

### A.3 Edit ops

The fifteen op kinds, discriminated on `op`, are `add_component`,
`move_component`, `replace_component`, `move`, `remove_component`, `set_component_value`,
`set_component_attribute`, `add_net_label`, `remove_net_label`, `wire_pins`,
`remove_wire`, `add_directive`, `remove_directive`, `replace_directive`,
`set_plot_panes`. The batch
is atomic, so the first op that fails aborts the transaction and nothing is
written.

An op names a part by its reference, as `REF` or in a pin as `REF.PIN`, and
so does `inspect(kind="net")` on a sheet. The part spelled so is named, else
the one whose reference differs only in case, since LTspice compares
references without case; `add_component` refuses a reference another case of
which is taken. A reference that names no part is refused with the list of
the parts there are.

`move` shifts any mix of records by one offset, `dx` and `dy`. It selects
parts by `references`, net labels and grounds by point (`labels`), wires by
their two ends (`wires`), and directives and comments by exact text or
`regex:` (`texts`). A `region` also selects what lies inside it, edges
included, testing a part by its whole box, a wire by its two ends, a label by
its point and a text by its anchor. With `drag`, an end of any other wire that
lies on what moves (a moved pin, label or wire end, a point along a wire that
moves whole, or with `region` any end inside it) moves too, so the wire stays
attached, and so does a label or ground on any of those points. A wire both
of whose ends move moves whole, taking what lies along it. A wire dragged off
its axis is reported as diagonal. A part's attribute text moves with it.
Something named that is not there is refused, and so is a move that would
join two nets that each have a name (names compared as LTspice compares them,
in any case), before anything changes. With `drag`, a move that would cut
anything that stays off from what moves is refused as well, with what would
part, what stays and how to take it along: a part whose pin touches a moved
pin with no wire, lies inside a wire that moves, or is held by a region only
in part; a label or a wire's end along a wire that stretches rather than
moving whole. Naming every part of the net that stays in `references` moves
the whole net. The result's `moved` counts what
moved and lists the parts. Applying the opposite offset to what it moved puts
everything back. `move_component` places one part at a position and can turn
it.

`replace_component` gives a part a new `symbol` where it stands. The part
keeps its reference, place, turn and attributes, and its value unless `value`
gives one; the old symbol's attribute text placements are dropped, so the new
symbol places its own. Each old pin is paired with the new pin of the same
name, in any case, or as `pins` pairs them (old pin to new pin, each by name
or SpiceOrder). A paired pin that moved is wired back to where its old
wiring ends, by a route the `wire_pins` planner accepts: straight, round one
corner, or across halfway with two; the result's `wires` lists what was
drawn. The swap must then leave each paired pin joined to what its old pin
was joined to, each unpaired new pin joined to nothing, and every other net
as it was; otherwise it is refused with what the pin would join, and the
sheet is left as it was. The result's `was` is the old symbol, `left_at`
gives the place of each old pin that had wiring and no partner, whose wiring
is left there joined to nothing of the part, and `unpaired_pins` names the
new pins left unconnected. Applied again with `was` as the symbol it puts the
symbol back; the wires it drew stay, for `remove_wire`.

`remove_wire` removes an exact segment; a segment drawn more than once goes
whole, and is refused when that would cut its net in two, since asking to
remove a duplicate is not asking to cut a connection. `remove_component` takes
`cleanup_wires` to also delete what its removal leaves dangling, and it is the
only op that cleans up. The freed points are the part's pin points that no
other part's pin is on. From them, until nothing changes: a wire with an end
on a freed point goes unless taking it splits a net between pins and labels
(`nets.net_change`), and each end of a taken wire that holds no pin and no
label becomes a freed point. A label or ground flag on a freed point or along
a taken wire goes when, on the sheet left, it sits on no pin and no wire.
Nothing not reached from a freed point is touched, and the result does not
depend on the order of the sheet's records. The op's result lists
`removed_wires` (one per record taken), `removed_labels`, and `kept_wires`
(each wire left with an end on a freed point or a pin; `joins` names one member
of each piece of the net it holds together: its first pin in sheet order, else
its first label, with the pieces in that order too, as `nets.net_change` lists
them). Each is sent whole: it is bounded by the wires and labels reached from
one part's own pins. Within one call, a `remove_wire` or `remove_net_label`
whose target an earlier removal op took succeeds with `already_removed_by`
(that op's index); a removal that finds nothing otherwise is refused, naming
the wires at the given ends (ten, then how many more) or the pins and wires
at the label's point, and pointing to `inspect(kind="net")` to list them.
`add_directive` takes `kind`, `directive` (simulated) or `comment` (ignored by
LTspice). `remove_directive` and `move`'s `texts` name directives and comments
by one rule. A literal names every text equal to it, space around either
ignored, as the text reads or as the file spells it, with a line break as the
`\n` escape and a backslash doubled. A `regex:` prefix names every text the
pattern is found in (`re.search`) as the file spells it, case-sensitive:
`(?i)` ignores case and `^` anchors the pattern at the start of the text. `remove_directive` removes every text it names and
lists each in `removed_texts`. `replace_directive` gives one text new words
(`text`) and keeps its record: its place, size, justification and kind. Its
`instruction` names the text by the same rule and must name exactly one, or
`x` and `y` pick the named text anchored there. An instruction that names
several texts is refused with the place of each, up to ten and the count of
the rest, so a caller picks one rather than writing one text over many; texts
on one anchor cannot be told apart, so one of them is moved first. A
directive's new text is checked as `add_directive`'s is, and a refused op
changes nothing. Applied again with its result's `x` and `y`, `now` as
`instruction` and `was` as `text`, it puts the text back, unless `now` begins
with `regex:`, which names a pattern. A new text that is empty or only white
space is refused, naming `remove_directive`.

With `find`, `replace_directive` changes only those words: `text` replaces
them and the rest of the text stays. `find` and `text` are read as the text
reads, CR and CRLF each one line break, and matched literally and
case-sensitive. `find` is looked for in the one text `instruction` names
(`x` and `y` picking among several), else in the texts anchored at `x` and
`y`, else in every directive and comment on the sheet, and must be found
exactly once. Not found, the refusal names the text `instruction` named and,
when `find` is in the file's spelling, gives the reading form; found in
several texts, it lists them (ten, then a count) and asks for `instruction`,
or `x` and `y`; found several times in one text, it gives the count and up to
ten line:column places and asks for a longer `find`. A directive's new whole
text is checked as above; a comment is not. `instruction` is optional with
`find`, and a call with neither is refused. Its result's `undo` is a
`replace_directive` op that puts the words back and succeeds: `now` as `find`
and `was` as `text`, at the text's `x` and `y`, when `now` is not empty and is
found once among the texts there; otherwise the whole new text as
`instruction` (a pattern of its record when it begins with `regex:`) and the
whole old text as `text`. A whole-text edit strips its `text`, so when the old
text begins or ends with a space or a line break, that op also gives the whole
new text as `find`, and the old text comes back exactly. The check before a
write wins over the reverse call: when the old text is one a write would
refuse (an unrunnable directive the edit made runnable), the result gives no
`undo` but `undo_refused`, the reason that check gives; the old words are
still in `was`.

A text is read as both LTspice builds read a TEXT record, which they draw and
export alike (recorded as `text-backslash`): `\n` is a line break, even inside
a Windows path such as `C:\new`, `\\` is one backslash, and any other
backslash is itself. A reply gives a text as it reads. A text is written so
that it reads as given: each backslash doubled and each line break as `\n`,
as LTspice writes a TEXT record. So a line break is given as a line break;
the two characters `\n` are written as a backslash and an `n`.

`set_component_value` splits a trailing `KEY=VALUE` off the value it is given
and sets it wherever the part already gives that key (`Value`, `Value2`,
`SpiceLine` or `SpiceLine2`, matched case-insensitively), in every copy found.
A key the part does not give is added to `SpiceLine`. A part that gives a key
more than once gets the sheet's `duplicate_parameter` finding, which the reply
lists because the op names the part. A behavioural source's value is one
expression and replaces `Value` entirely. A voltage or current source's value
replaces only its value: its own parameters stay (a voltage source's `Rser=`
and `Cpar=`, either source's AC magnitude, written `AC=2` or `AC 2`, and a
current source's `load`), which both LTspice builds read after the forms of
value recorded as `source-parameters`. A parameter the new value gives is set
in every copy the part has, and a `KEY=VALUE` that is part of the value, such
as a PWL's `TIME_SCALE_FACTOR=`, stays in it. The card an `assign` variation
or a value edit of an instance inside a subcircuit writes keeps them by the
same rule (`component_value.edited_source`). A source that plays a .wav file
has no value to set, and the op is refused.

Every op that writes a part's `Value` holds it to one rule: `add_component`,
by `value` or among its `attributes`, `set_component_value` and
`set_component_attribute`. LTspice writes the `Value` after the part's nodes,
so a space in it may only separate what the part's element class reads as one
value: a source's spec, the area factor of a BJT, JFET or diode, the `off`
of a transistor or diode, a model name followed by `KEY=VALUE` parameters,
and anything inside braces, quotes or parentheses. Anything else, such as a resistor's `1 k`, puts a
token where LTspice reads another node and is refused before the sheet
changes. The class is the one the part's own `Prefix` gives, in the same op
or before it, else its symbol's. Setting or clearing a `Prefix` with
`set_component_attribute` changes the class, so the part's own `Value` is
held to the new class, and an op that would leave a `Value` the new class
splits is refused the same way.

### A.4 Inspect queries

§3.6 lists every query kind.

---

## 8. Known gaps

- The one native statistical profile is the pinned Sky130 NMOS profile below;
  other device families and multiplicity or finger conventions are not
  supported.
- No correlated Monte Carlo draws across two decks.
- No loop-gain probe templates.
- An attached analysis stage left unfinished when the process running its job
  stopped is not re-run; run `analyze_results` over the job instead.
- Request-index records are deleted only with their job ("Job model",
  §3.1), so the index can grow with every submission.
- Ops cannot define a new hierarchical block (its symbol and child sheet);
  they can place an existing subcircuit symbol.
- Two wires that cross can be joined only by ending one on the other. There
  is no op for a junction at a crossing; `wire_pins` draws a route across a
  wire where neither ends without joining them, as LTspice does, and warns.
- `remove_wire`'s point form removes the segments that end at the point; a
  wire whose interior passes through it is kept. At a T that removes the stem
  and keeps the wire it joined, but at a pin on a wire's interior it leaves
  that wire in place although the two are connected.
- No op edits a record in place. Renaming a part, renaming a net's labels or
  changing a directive's text is a remove and an add, which restores only what
  the add op takes and changes the order of the sheet's records.

## Hierarchy discovery contract

`inspect(queries=[{"kind":"hierarchy", "path":"amplifier.cir",
"simulator":"ltspice", "instance":["XA","Xleaf"], "prefix":"M"}])`
lists every instance of the expanded netlist, with each subcircuit call
replaced by the elements inside it, reading the deck and the files and `.lib`
sections it includes; the `components` and `net` queries describe the cards
of one file as written instead. The query is read-only and accepts `.cir`,
`.net`, `.sp` and `.spice`; export a `.asc` first. It does not stage a deck or
change a library. Per-instance variation is a separate operation (Appendix
A.1).

Identity is a list of reference segments, matched case-insensitively with the
original spelling preserved. A punctuation-joined display name is not a
selector. Each `data.instances` row contains:

| Field | Meaning |
|-|-|
| `instance`, `reference`, `element` | The path of reference segments from the top level, the reference on the card, the element letter |
| `source` | The card's file and line, its `.lib` section, and the subcircuit that contains it (`definition`) |
| `raw` | The element card as written |
| `nodes`, `ports` | Each node's `scope` (the instance path it belongs to), `name` and `voltage_trace`; for a subcircuit call, each port of the definition mapped, in order, to the node it connects to |
| `model` | The model name on the card, the file and line of the definition used (`source`), or a `reason` when the definition cannot be determined; `family` lists the bins of a binned model |
| `value` | The element's value, apart from its `key=value` parameters |
| `parameters`, `environment` | Each `key=value` on the card, and every parameter of the scope the instance opens (a subcircuit call) or sits in, each as its expression and numeric fact |
| `geometry` | A MOSFET's W and L as written on the card, multiplied by the scale and given in metres |
| `address` | The `device` name the `operating_point` recipe takes and the `.save` line a run needs to record it (`save`), or the `reason` there is none |

Numeric facts have `expression`, `value`, `unit`, `status` and `reason`. An
unresolved value is left out, never a guessed zero: its `status` is
`unresolved` and its `reason` says why. Discovery evaluates only
expressions built from finite SPICE numbers and suffixes, parameter names,
outer braces or single quotes, parentheses, unary signs, arithmetic
`+ - * /`, and at most one power operator (`**`, or `^` on ngspice). On
LTspice, an expression that contains `^` is left unresolved. Functions,
missing names, parameters that depend on each other in a cycle and other
expressions stay unresolved. MOS W and L do not account for `nf` or `m`. An
ngspice `.option scale` whose value cannot be resolved, for example because it
names a parameter that does not exist, leaves W and L unresolved; it is never
taken as 1. An explicit LTspice `scale` is refused in both `.option` and
`.options`; use explicit SI dimensions. The `mil` suffix is 25.4 micrometres.

A `key=value` on a subcircuit (`X`) line is evaluated in the calling scope, so
`W=W` takes the caller's `W`; on LTspice this also holds for a value that
names another parameter on the same line. On ngspice, a value on an `X` line
that refers to another parameter on the same line is reported unresolved,
because discovery does not evaluate parameters as completely as a simulator
does. Inside the subcircuit, ngspice takes a parameter from the `X` line
first, then from a `.param` in the subcircuit body, then from the default on
the `.subckt` line; LTspice takes the `X` line first, then the `.subckt`
default, then the body's `.param`. A parameter that refers to another uses
that parameter's value after this order is applied. A parameter declared
twice in one scope (allowed in ngspice `hsa` mode) and a key given twice on
one line fail the query.

`data.profile` records the declared simulator, the effective `ngbehavior` and
`evaluation_context: "initial_deck"`, the deck as written before any run
changes it. No installed executable is needed. For ngspice, omit `ngbehavior`
to use the mode the server runs ngspice with; setting it changes only how this
query reads the deck, so configure simulation to use the same mode. The
reader's rules were established against ngspice for `hsa`, `kiltpsa` and the
empty string (native SPICE mode).

`data.values_read_otherwise` says which values of the files read an LTspice
build this session may run reads as another number than the reply gives
(`lib/stored_values.py`). The server reads a micro sign or a Greek mu at a
value's suffix as micro, as a run through it does (staging spells it `u`).
LTspice XVII decodes a deck stored as UTF-8, with or without a byte order
mark, as 8-bit text, so it reads such a sign as two letters that are no scale
and takes the bare number (`deck/micro_utf8`, `deck/micro_utf8_bom`,
`deck/greek_mu_utf8`). The builds a session may run are its LTspice
executables, the detected one and each named one
(`services.session_ltspice_builds`): one identified as before 24 is LTspice
XVII, a later one LTspice 26, and one whose build nothing shows stands for
every recorded build, each summary about it carrying the `note` that the
session's build was not identified. A session with no LTspice is told
nothing.

The list holds one summary per build and kind of reading (code
`value_read_otherwise`), at most four whatever the number of files and
values. A summary states its values completely by a `rule`, with no list:
every value of the files read whose scale suffix is a micro sign, as a
two-terminal part's own value (`reading: established`, where the recordings
show the number the build takes), or in any other position or an encoding no
recording shows for the build (`reading: not_established`). It gives the
totals (`count` values in `files` files), at most three files in `by_file`,
each with its codec, its count and at most three `examples` (value, line, the
server's number, the text the build reads, the number it takes), and
`files_not_listed`. A file's `listed_by` names the call that lists each of its
values with the build's reading, `verify_circuit` with the syntax check, and
is present only for a netlist that call checks: an included library or a
sheet has none. The syntax check's micro-sign finding gives each value its
reading by every build the session may run; its presence and severity are
unchanged. The list is deck-level and is carried on every page, as `profile`
and `inputs` are. The same list rides on `verify_circuit`'s and
`edit_schematic`'s `comparison`, and on a run's receipt as one
`value_read_otherwise` observation per summary, over every staged file.

A sheet read as the server's export of it is stated by the sheet's own bytes,
not the export's: the export is the server's file, and LTspice 26 writes a
part's `u` as a micro sign there whatever the sheet says. A value counts where
the server read the export's sign as micro and the sheet's own text for it,
read by the sheet parser (the part's value attribute, or the directive; a
comment is never a value), holds that value with the sign; it is stated at
that text's line. Both builds read a UTF-8 sheet byte by byte
(`export/micro_utf8`, `export/greek_mu_utf8`) and a cp1252 or UTF-16 sheet's
sign as micro (`export/micro_cp1252`, `export/micro_utf16le`). A sheet that
cannot be read is listed under `values_read_otherwise_not_checked`, since an
absent summary reads as checked.

When an ngspice executable is available, `data.ngspice_check` reports how the
values discovery resolved compare with the ones ngspice reads
(`lib/hierarchy_check.py`). ngspice reads the deck through a driver that
`source`s it in the query's mode, as a run would, and prints each value: a
resistor's `value` (ngspice's `resistance`), a MOSFET's `geometry` W and L
(`w`, `l`, both scaled), and the copies discovery counts for either card,
which instance mismatch uses (`multiplier`, not a reply field), against
ngspice's `m`. The copies are compared on every card, so the check also
tests which subcircuits discovery says ngspice rewrites for an `X` line's
`m`, and how, which differs by mode. The driver runs no analysis, so
ngspice prints 0 for an `m` nothing set, which is read as one copy where the
card writes no `m`. ngspice reads a 0 ohm resistor as 1 milliohm, so a 0 ohm
value is not compared. `status` is `agrees`, `differs` (with `differences`, each the
`instance`, the `field` it is about, and the `reader` and `ngspice` values) or
`not_run`. A `not_run` result has a `reason` and a `detail` that explains it:
`no_ngspice`, `nothing_to_compare`, `path_unsupported` (a quote, `$`, `` ` ``,
a backslash, a brace, a tab or a line break in the path, or a non-ASCII path
on Windows), `ngspice_failed`, `inputs_changed` (a file changed before ngspice
read it), `deck_not_read` (ngspice printed no value), `mode_unchecked`, or
`mode_differs` (a startup file ngspice reads set another mode). `compared`
counts the values compared when the check ran. `not_compared` counts the rest
by reason: `no_ngspice_name` (an element the reader gives no ngspice device
name), `unresolved`, `zero_resistance`, `no_ngspice_m` (one copy of a
device ngspice has but gives no `m` parameter, such as B3SOIDD, which builds
one) and `no_ngspice_value` (ngspice printed none, or 0 for a W its device
sets only in an analysis, as MOS levels 4, 5 and 6 do). Nodes, models and
sources are not compared. A result,
a failure to run ngspice included, is kept per deck content, mode and
executable, so later pages do not ask again. The check covers the whole deck,
not only the selected instances.

The server lower-cases a mode wherever it takes one, in this argument and in
`[simulator] ngbehavior` alike, so `HSA` names `hsa`. In any other mode than
those above, the query
answers only when ngspice was asked for every value and gave the same ones, and
is otherwise refused with what the check found. The reader then refuses `gnd`,
as in `kiltpsa`, and a `.param` declared twice. A run in such a mode needs no
hierarchy reading and is not affected. An instance edit is refused, and a
recoverable run is refused before the job is recorded
(`recovery_ngbehavior_unsupported`), because both read the deck by these
rules. A `.lib file section` line needs a mode ngspice reads without its
`lt`, `ps` or `s3` flag (`hsa` or native mode): `lt` and `ps` read it as two
plain includes, and `s3` does not read `.lib` at all. The server reads a mode's flags as ngspice
does (`lib/ngspice_compat.py`): letters match case-sensitively, `spe` clears `ps`, `lt`, `ki` and `eg`, then `ps` clears
`hs`, and `mc` turns off every flag but `s3`. So `spelt` and `ltmc` read a
section, `hsps` replaces a card's `m` instead of multiplying it, and `s3mc`
skips a section. The deck's first line is its title and is skipped; an
included file has no title line, so its first line is read as a card. In
ngspice native and `hsa` modes, `gnd` is ground like `0` everywhere, even
where a subcircuit has a port named `gnd`; on LTspice such a port is that
port. Discovery refuses `gnd` in `kiltpsa` mode because its ground-alias
behavior differs across supported ngspice versions; use node `0` for portable
connectivity. An LTspice query rejects `ngbehavior`, which is ngspice-only.
With a `.step` in the deck, every numeric fact is unresolved, because its
value depends on the step. A `.control` block, `.alter`, `.altermod`, `.del`,
`.delete`, `.if`, `.elseif`, `.else` or `.endif` fails the query.

Nodes are reported for subcircuit calls (`X`, including `PARAMS:`), a MOSFET
written as four nodes, a model and `key=value` parameters, a two-node R, C or
L with one value, and a two-node V or I source. Other element forms are
listed with `raw` and a reason their connectivity or model is unsupported; a
device named on a line to control the element is never reported as a net. A
subcircuit defined at top level may be called from inside another, but a
`.subckt` defined inside another `.subckt` fails the query. A `.model` inside
a `.subckt` is allowed and is used there before a top-level model of the same
name. Recursive subcircuit calls; a missing include, `.lib` section or called
subcircuit; an include cycle; a malformed `.subckt`/`.ends` or `.lib`/`.endl`
pair; and a model or reference defined twice among the cards in use fail
the query. Of a subcircuit defined twice at the top level, the definition the
reader keeps is read (ngspice the first, LTspice XVII the last of two in one
deck); the query fails, saying why, where the reader refuses the deck
(LTspice 26) or what it does is not recorded (any other LTspice build, more
than two definitions, or a definition in an included file). A binned model (`nch.1`, `nch.2`, ...) is reported with
every bin under `model.family` and a reason that the bin is not selected,
rather than as one model. Cards after the root `.end` are ignored.

Discovery is limited to 16 MiB of file content read, 100,000 cards, include
depth 8, 20,000 instances in the expanded netlist and a subcircuit nesting
depth of 32. Expressions are limited to 2,048 characters, 128 syntax-tree
nodes, a chain of 32 parameters that depend on each other, and a power
exponent of at most 64 in magnitude. Exceeding a structural limit fails the
whole query; exceeding an expression limit leaves the affected numeric fact
unresolved.

`data.inputs` lists the files read, the SHA-256 of the exact bytes parsed
from each, where each include reference resolved, and, for each name a
one-word `.lib` line gives, whether a file of that name existed, which
decides whether the line includes a file or opens a section. The files are
read one after another, not as one snapshot of the whole set. A cursor is
valid only for those file contents, the same profile and the same filters;
any rewrite of an included file, even one with the same size and timestamp,
makes it invalid. The Python API collects all pages with the same evaluator:

```python
with Api(working_dir=project) as api:
    discovered = api.inspect(queries=[{
        "kind": "hierarchy", "path": "amplifier.cir",
        "simulator": "ngspice", "ngbehavior": "hsa",
        "instance": ["XA", "Xleaf", "M0"],
    }])
    device = discovered["results"][0]["data"]["instances"][0]
```

For a supported nested MOS, ngspice gives `device: "m.xa.xleaf.m0"`
and `.save @m.xa.xleaf.m0[gm]`; LTspice gives
`device: "xa:xleaf:m0"` and no directive, since the server asks LTspice 26
for the device table on every `.op` run (once it knows the build, see
`logopinfo_withheld`) and XVII prints it unasked and stops on
`.options logopinfo`. Add any directive given to the deck before
`run_experiments`, then pass `address.device` to the `analyze_results`
`operating_point` recipe. On LTspice, that selector also matches any longer
path that ends with it, so an instance whose whole path is the end of another
instance's path (`m1` when `x1:m1` also exists) gets no address. On ngspice,
gm is read from the trace the `.save` line records in the raw; on LTspice,
from the device table in the log. For a nested resistor the directive is
`.save @r.xa.xleaf.r1[i]` on ngspice, `.save I(xa:xleaf:r1)` on LTspice 26
and `.save I(a:leaf:r1)` on LTspice XVII, because each build's `.save`
accepts a name inside a subcircuit only in the spelling its raw uses
(recorded: `raw/save_subckt_names*`). A node's `voltage_trace` uses the same
spelling. An address is given only for a resistor or a MOSFET whose every
path segment is a name of two or more characters made of letters, digits and
`_`, not starting with a digit; any other instance is still listed, with a
reason the address is unavailable. A node's `voltage_trace` is the
simulator's spelling of that node's trace and does not mean that a run saved
that signal.


## Native PDK experiment contract

The supported profile is `sky130-e6f9c887-ngspice-v1`, which is the generated
Sky130A model set at open_pdks revision
`e6f9c8876da77220403014b116761b0b2d79aab4`, covering the
`sky130_fd_pr__nfet_01v8` transistor. The server checks the bytes of the PDK
files the deck reads against SHA-256 hashes packaged with the profile.
Install the matching PDK separately and, with a sandbox set, include its
location in the sandbox.

The deck has one `.lib` line naming `libs.tech/ngspice/sky130.lib.spice`,
with the section for the mode:

| Mode | Authored section and control |
|---|---|
| `nominal` | `tt` |
| `mismatch` | `tt_mm` |
| `process` | `mc` |
| `combined` | `mc`, followed by root `.param mc_mm_switch=1` |

The server verifies that line and checks that every transistor in the deck is
a `sky130_fd_pr__nfet_01v8` whose inner device uses the profile's model bins,
with W and L that resolve to positive values, scale `1e-6`, and `m`, `mult`
and `nf` all 1. A case whose deck does not meet these conditions fails with
`pdk_native_validation`, with the reason in `evidence.reason` (for example
`coverage` or `geometry`). `assign` variations may change the deck's own
values, such as W and L. A change to a card in a PDK file, or to a control the
server sets (`seed`, `randomseed`, `mc_mm_switch`, `mc_pr_switch`,
`ngbehavior`, `ng_nomodcheck`, `scale`, or a top-level `.param` in the PDK
files), refuses the request with `pdk_native_request`. Mismatch coefficients
the caller chooses belong in a `random` variation instead of this profile.

The deck must have one `.op`, `.tran` or `.ac`. Every file it reads must be
staged, so live includes are refused, and so are `.control`, `.step`,
`.alter`, `.dc`, `.noise`, `.pz`, `.sens`, `.tf`, `.disto`, `.four`, `.reset`,
`.setseed` and a deck that sets one of the controls above itself (other than
the one `mc_mm_switch` the `combined` mode needs). Each sample starts a new
ngspice process with `-n` (no startup files), sets `hsa` and the sample's
seed before reading models, then runs the analysis. Each sample counts
against the limit on concurrent runs and is cancelled and timed out like any
other case.

Each sample's ngspice seed is derived from the entry's `seed`, the circuit id,
the entry's `id`, the hashes of the input files, the assignments, profile,
mode, `.lib` section and sample index. To reproduce one sample independently,
keep the circuit id and the entry's `id`, the input files' bytes,
assignments, profile, mode and `seed`; set `runs: 1` and `sample_start` to
that sample's index. Parallelism and job ids do not change the sample. Exact
replay holds only on the recorded simulator, build and platform, not across
simulator releases. A request in which two different samples would get the
same derived seed is refused before launch.

Each case has `native_statistics`. Its `requested` block (profile, mode, ids,
`seed` and sample index) is kept for failed and skipped cases too.
`validated` (the derived seed, the input and model digests, and the
transistors covered) and `prepared` (the hashes of the files the run reads)
appear only once those steps have succeeded, and `unavailable` names each
missing fact with the reason. Default receipts give a compact form, with
counts and digests in place of the coverage list, the dependency list and
file paths; `jobs(action="runs", ..., run_fields=["native_statistics"])`
returns them in full, and `run_fields` can select parts of it there too. The
job record keeps the full form, so it survives a server restart, and the
source identity of each result read from the case includes the compact form.
The compact `dependency_digest` is a hash over the pairs of original PDK file
and staged file SHA-256, counting each staged copy; the full form lists every
staged file, including several copies of one original. A sample whose
preparation fails fails its case with `pdk_native_preparation` rather than
the submission, and the PDK's internal random draws are not reported
(`unavailable.internal_draws`).
