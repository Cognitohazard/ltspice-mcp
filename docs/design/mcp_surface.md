# The MCP tool surface

This is the contract for the tools `ltspice-mcp` advertises: the shared
response envelope, each tool's argument shape, and why each rule is there.
Read it before changing any of them.

The code is the authority. Where this document and the source disagree, the
source wins and this document is the thing to fix.

---

## 1. Shape: six operations over three planes

| Plane | Tool | One-line contract |
|-|-|-|
| EXECUTE | `run_experiments` | circuits x declared variations; durable idempotent receipt (plus inline results within a bounded dwell); preflight lint; snapshot provenance |
| EXECUTE | `jobs` | status / long-poll wait / cancel / list / runs over receipts; resolves `job_id` or `request_id` |
| UNDERSTAND | `analyze_results` | recipe batch over runs, returning typed values with case/step identity, attributed reductions and spec verdicts; bounded and continuable |
| UNDERSTAND | `inspect` | reads: capabilities, symbols (list and detail), net trace, components, models |
| AUTHOR | `edit_schematic` | typed op batch onto a sheet (blank or existing); revision-guarded, transactional; geometry facts back |
| AUTHOR | `verify_circuit` | gate: lint/syntax, symbols, export, layout, quality, compare (equivalence or structural diff), render |

A seventh tool, `plot_waveform`, is registered alongside them. It is the
interactive MCP Apps waveform widget; it predates this envelope and is
deliberately outside it.

Canonical loops:

```
netlist loop:   write deck (natively) -> run_experiments -> analyze_results -> edit natively -> ...
schematic new:  inspect(symbols) -> edit_schematic{base:"blank", ops, reference, render} -> revise ops -> done
schematic edit: inspect existing .asc -> edit_schematic{ops, expected_sha256} -> verify_circuit
debug loop:     verify_circuit(lint) -> fix natively -> run_experiments -> analyze_results -> inspect(net)
long runs:      run_experiments (receipt) -> jobs(wait) -> analyze_results
```

Happy-path call counts: characterization is 2-3 calls per design iteration; a
new schematic is 2 calls (inspect symbols, then build with a reference netlist
and a render); debugging is lint, fix natively, run, analyze.

---

## 2. Shared conventions

Normative for all six tools.

**Envelope.** `outcome` is one of `complete`, `partial`, `failed`,
`in_progress` — `in_progress` meaning a successfully returned, still-running
receipt. Failures carry
`error {code, message, stage, retryable, commit_state, item_id?}` where
`commit_state` is `not_started`, `committed` or `unknown`. MCP's `isError` is
reserved for call-level failures; a per-item failure sets `outcome: "partial"`
instead.

**Error codes.** `code` is a stable, public name for a failure. Which one a
failure gets is decided by the exception's *type*, never by its message: every
class in `errors.py` declares a `code`, and that class code is the default a
handler may override with the stage that failed — a result read that fails
while a deck is being staged reports `submission_failed`, not
`result_unreadable`. Adding a code is routine; renaming or removing one is a
client-visible change and needs a CHANGELOG entry. The complete vocabulary,
envelope codes and `observations` codes together, is frozen by
`tests/test_error_codes.py::FROZEN_ERROR_CODES`.

From the error hierarchy:

| code | meaning |
|-|-|
| `path_denied` | the path resolves outside `allowed_paths` |
| `netlist_invalid` | the netlist, or a component reference in it, could not be read |
| `symbol_unresolved` | the schematic opened, but a symbol, sub-sheet or library it refers to was not found |
| `simulation_failed` | the simulator could not be started, or the run failed |
| `result_unreadable` | a `.raw` or `.log` could not be read |
| `analysis_deadline` | a result read ran past its wall-clock bound and was abandoned |
| `no_axis` | the result is an operating point: no sweep axis to query at |
| `job_not_found` | no job with that `job_id` or `request_id` |
| `idempotency_conflict` | a `request_id` was reused for a different payload |
| `cancel_not_authorized` | neither the owning process nor a matching `control_token` |
| `cancel_failed` | cancellation was authorized but could not be carried out |
| `library_error` | a component library failed to load, parse, or resolve |
| `batch_job_error` | a sweep or Monte Carlo config could not be used |
| `raster_unavailable` | a PNG was asked for without the optional `raster` extra |
| `internal_error` | an unclassified server failure |

Named by the stage instead, where the stage is the more useful fact:
`submission_failed`, `receipt_failed`, `revision_conflict`,
`asc_export_unavailable`, `unsupported_variant` (with the supported list),
`ambiguous_target`, `parse_deadline`, `lint_blocked`.

**Page object.** Nothing returns an unbounded collection. Every potentially
large one is a page: `{items, total, returned, truncated, next_cursor}`, and
every tool that returns a page accepts the matching cursor input — pagination
is callable, not merely declared.

When one page carries more than one collection (an `inspect` net view pages
pins and wire coordinates together), `total` and `returned` sum across all of
them and a `collections {<data key>: {total, returned, truncated}}` block
breaks it down; `primary_truncated` says whether *this* collection continues.
That is the conforming reading of "every potentially large collection": before
it, the counters described only the first collection while `truncated` covered
the pair, so a caller stopping at `returned == total` silently lost the rest of
the second one. The invariant now held on every page is
**`truncated` implies `returned < total`**, and the schema tells callers to
stop on `next_cursor`, not on the counters.

**Channels.** `observations`, `warnings` and `failures` are three distinct
channels and are never merged. See the result-trust rules in `CLAUDE.md` and
`lib/result_observations.py` for which fact belongs on which.

**Findings shape** (lint, layout, quality):
`{rule_id, severity, ok, evidence, at {file, line? | x, y?}, subject}`. `at`
and `subject` are required on anything a caller is expected to fix.

**Self-sufficiency.** `structuredContent` carries everything the caller needs,
including a `hint`; the text channel is presentation only. Every user-supplied
path goes through `safe_path`, and `.asc` work runs under `circuit_file_lock`.

**Lean by default.** The default response is the answer channel: per-case
headline rows (label, promoted headline scalars, verdicts) plus a one-line
completeness count. Detail sits behind named opt-ins — full nested `value`
dicts, full run records, the `source_hashes` identity echo, and `provenance`.
Never trimmed in any mode: failures, errors, observations, warnings, and any
completeness shortfall. Lean cuts success ceremony, never the fact channels.

The reason is response cost: every response byte is re-read on every later
turn of the conversation, and measurement put the tool path at several times
the cost of a hand-rolled shell table for identical numbers even with the
opt-in levers applied. Two consequences worth knowing:

- A completed row drops its raw/log artifact paths (reachable through
  `jobs(runs)` with `run_fields`), but a **non-completed** row keeps them:
  failure entries carry only `{case_id, code, message}`, so the failed row's
  log path is its diagnostic and belongs on the fact channel.
- An empty per-circuit lint entry is ceremony and is dropped. A circuit absent
  from `lint` is clean; findings always emit.

Pagination is exempt from the lean rule: `next_cursor` is a
nullable-key-always-present, because the omit convention caused a client
`KeyError` past the first run page and a null key costs about 20 bytes. Lean
applies to rows, the identity echo and nested values, not to cursor-shape
safety.

**Presentation fields.** Fields that choose how a receipt is *rendered* rather
than what *runs* are excluded from the idempotency fingerprint, so re-asking
for the same experiment at a different verbosity replays it instead of
conflicting. On `run_experiments` these are `provenance`, `run_fields`,
`budget`, `execution.wait_s` and `analyze.include.fields`
(`RunExperimentsInput.PRESENTATION_FIELDS`).

---

## 3. Tool contracts

### 3.1 `run_experiments` — effectful fan-out

**Submission and idempotency.** `request_id` is optional; omitted, the server
mints one and echoes it on the receipt, so a one-off spot check pays no
idempotency ceremony. The auto id is a real id: `jobs` and `analyze_results`
address the run by it as usual, and it writes the same request-index record as
an explicit id, through one uniform code path.

Passed explicitly, `request_id` is the durability key. At submission the
server persists `{request_id -> job_id, fingerprint}`, where the fingerprint is
the sha256 of the canonical (sorted-key) input payload. Same id with the same
fingerprint returns the existing receipt (an idempotent recovery, surfaced as
an observation); same id with a different fingerprint is an
`idempotency_conflict`. Scope is the server working directory's job store;
retention matches job retention. Transport cancellation ends only the dwell,
never the durable job.

The fingerprint's canonicalizer version bumps only when the canonical
representation of a *previously valid* request changes. A presentation field
excluded from canonicalization from its first valid day changes no old
canonical bytes and never triggers a bump.

`allow_live_includes` is deliberately *not* a presentation field: a job whose
inputs cannot be proven cannot be replayed on the strength of a stored
fingerprint, so reusing its `request_id` runs the experiment again.

**Control token.** The receipt carries an unguessable `control_token`. Cancel
authority is the owning process *or* a presented control token, so a
reconnected session can still cancel its own work. The token is returned only
by the original submission response and by an idempotent replay whose
fingerprint matches — knowing the full original payload is the proof of
ownership. `jobs` status/wait/list/runs output never includes it: read
visibility must not grant cancel authority. It is carried only while the job is
non-terminal, since a terminal receipt's token authorizes nothing.

**Receipt-then-dwell.** The receipt — job registered, persisted, cancel barrier
raised, snapshots staged — is durable *before* any case is submitted and before
any waiting. `execution.wait_s` (default 60, cap 120) bounds the dwell. If the
job reaches full terminality (runs *and* attached analysis) inside it, terminal
results return inline; otherwise the receipt returns with
`outcome: "in_progress"`. Why 60: the wait must end well before the client
gives up on the request, or the response carrying the job id is lost.

Input:

```
request_id           str, optional      idempotency key; minted when omitted
circuits             list[{path, id?}]  .cir / .net / .sp / .asc
variations           list[Variation]    Appendix A.1. assign entries combine by
                                        cartesian product; AT MOST ONE random
                                        entry per call (the product of two
                                        random families is ill-defined).
                                        [] = one plain run per circuit
execution            {wait_s?, run_timeout_s?, job_deadline_s?, max_parallel?,
                      simulator?: "ltspice"|"ngspice"}
analyze              {recipes, group_by?, include?} attached analysis stage
lint                 "block"|"warn"|"off"  (default "block")
suppress             list[rule_id]
allow_live_includes  bool (default false)
provenance           bool (default false)
run_fields           list[dotted row path] | null
budget               int | null            approximate response-token cap
```

`budget` and `analyze.include.fields` select a rendering of the durable
receipt, not what runs. A call that sets no budget renders under a
config-settable server default (`[analysis] default_budget`, default 4000)
engaged at the trim rung only — empty presentation blocks and the identity
echo, never facts, never caller opt-ins. An explicit caller budget overrides
the default entirely and may descend the full ladder.

**Job model.** An experiment job is a coordinator spanning multiple circuits.
Its persistence home is the server working directory's store
(`{working_dir}/.ltspice-mcp/jobs/`), holding the coordinator record — request
index, case records, counters, analysis state, source manifests. Each source
circuit's sidecar directory gets a lightweight pointer so
`jobs(list, circuit=...)` can find it. Owner-pid liveness and foreign-session
read rules are unchanged from the simulation-job model.

**Attached analysis is a job stage.** Job terminality means all runs terminal
*and* analysis terminal; the status sequence is
`running -> analyzing -> completed | completed_with_failures | failed | cancelled`.
The dwell and `jobs(wait)` default to full terminality;
`jobs(wait, wait_for: "runs")` returns at runs-terminal. Cancelling during
analysis cancels the stage and the run artifacts survive. An analysis failure
puts the job in `completed_with_failures` with an `analysis_error`; the runs
stay analyzable through a fresh `analyze_results` call. After a server restart
a pending or running analysis stage is marked failed with a restart
observation — there is no automatic re-execution. The analysis result persists
as a bounded artifact and its summary is inlined in terminal payloads.

The point of attaching analysis is that one call can carry simulation plus
measurement; without it, analysis always costs a second call.

**Input snapshot and provenance.** At submission the server stages the primary
deck (content-addressed) plus the includes and libs that resolve inside the
allowed roots, to a recursion depth of 3; staged copies preserve relative
topology. The manifest records `{path, sha256, staged, live, staged_path,
reason, section}` for every reference, including `.lib` section selections.
Runs read only those copies, so editing the original afterwards cannot change
what the job meant, and every reported number cites the hash of the deck that
produced it.

By default any reference that cannot be staged **fails that circuit's
submission** with an `include_unstaged` error naming the reference — "staged
copies only" holds strictly. A caller may pass `allow_live_includes: true`, in
which case runs read those files live, the manifest marks them `live: true`,
and an observation says the provenance is explicitly weaker. Live references
are enumerated, never silently trusted. Reject the fail-by-default rule and the
hashes can lie; reject staging altogether and vendor-library decks cannot run.

Provenance is opt-in on the response. Digests, staged paths, the full manifest
and the linter version are emitted only under `provenance: true` on
`run_experiments`, or `include.provenance: true` on `analyze_results`. The
lean receipt keeps `{circuit, path, simulator, dialect, staged_files}` plus
every manifest entry that *discloses* something — live, carrying a reason, or
otherwise not an ordinary staged reference — so it fails closed. An analysis's
`source_hashes` keeps a minimal `{manifest_id, label}` join table by default,
because rows need their attribution legend, with `log_present` and `job_id`
joining the digests behind the opt-in. The `provenance` flag also gates the
receipt's `analysis.request` echo, which is the caller's own input replayed
back. Measurement put provenance at 29% of receipt bytes on a real run;
everything remains reachable and nothing actionable is silently dropped.

**`.asc` ingestion** requires the LTspice exporter regardless of which engine
runs the deck: lock, export under lock, hash, stage, simulate the staged copy
(ngspice gets a sanitized staged copy), lint the exported deck. With no
exporter, that circuit's cases fail with `asc_export_unavailable`.

**Deck-native stances.** Temperature (`.step temp`), `.lib` section selection
and `.nodeset`/`.ic` are things the caller writes in the deck; a tool parameter
would duplicate one line the caller can already write, so there is none. The
linter explains the `.step temp` idiom instead of running a deck that tries to
sweep temperature as a parameter. Model-swap corners are different — running
one deck N times with device models substituted per run has no one-line deck
equivalent — so they stay a tool-performed variation (Appendix A.1).

**Completeness.** Every case gets a stable `case_id`. The counters are
`{declared, expanded, submitted, produced, failed, cancelled, skipped}` with
the run-terminal invariant
`produced + failed + cancelled + skipped == expanded`; every omission names its
`case_id` and a reason. Lifecycle status is reported separately from
completeness.

`progress` is derived from that durable completeness snapshot and is present on
receipts through terminal states:
`{expanded, terminal, remaining, declared, submitted, produced, failed,
cancelled, skipped}`, where `terminal = produced + failed + cancelled + skipped`
and `remaining = expanded - terminal`.

**RunRecord** is a standalone schema fragment shared by `run_experiments` and
`jobs`:
`{case_id?, run_index?, circuit?, assignments?, status?, raw?, log?}`. The keys
are optional because a requested run-field projection may remove any of them.
Its object rendering is `items: [RunRecord]`; its budgeted columnar rendering
is `items_columns: [key]` plus `items: [[value]]`, each value position named by
`items_columns`. Both renderings carry the same page metadata and cursor
semantics.

Under a `budget`, receipts keep the common envelope and negotiate the shared
trim -> answer -> columnar -> shrink ladder. The runs page and
attached-analysis row pages may use their columnar variants; fact channels,
completeness, progress, verdicts, coverage and recovery handles are protected.
If the irreducible floor exceeds the budget, the floor is returned with
`budget_not_met` rather than facts being dropped.

An `artifact` handle survives the lean row. The lean row flattens `value` to
its scalar leaves, and a handle is a dict; dropped with them, the `plot` recipe
answered with a series count and no way to reach the file it had just written.
A handle is the one value a caller cannot recompute from the response, so it is
kept — on the standalone `analyze_results` path and inside a `run_experiments`
receipt alike.

A collapsed failure row picks its representative by case id rather than by
whichever case finished first, so two identical runs report the same excerpt.
`count` is still the true number and `case_ids` still reports its own cap.

Output: `job_id, request_id, control_token, status, outcome, source[],
completeness, progress, lint findings (per circuit), runs (a page of
RunRecords or their columnar rendering), analysis?, failures[],
observations[], artifacts[], hint`.

### 3.2 `jobs` — control plane

```
{action: "status", job_id | request_id}
{action: "wait",   job_id | request_id, timeout_s (default 60, cap 300),
                   wait_for: "all" (default) | "runs"}
{action: "cancel", job_id | request_id, control_token?}
{action: "list",   circuit?: path, limit?, cursor?}
{action: "runs",   job_id | request_id, cursor?}   cursor absent = first page
```

Each action accepts only its own fields and rejects the rest. Output shapes are
discriminated on the echoed `action`:

- `status` / `wait` return the receipt snapshot of §3.1 **without**
  `control_token`, plus `analysis_status` and (for `wait`) `timed_out`. The
  snapshot includes `progress` through terminal states and admits the same
  budgeted receipt variants.
- `cancel` returns a page of per-run kill receipts.
- `list` returns a page of circuit groups
  `{path, exists, last_activity, status_counts, interrupted_job_ids,
  recent_jobs, recent_jobs_total}`. With no filter this is the
  recently-touched-circuits view. `recent_jobs` (newest first, capped, with
  `recent_jobs_total` alongside) is the route from a completed job back to its
  id; counts alone left "find the run I did earlier" answerable only by listing
  the store directory.
- `runs` returns a page of run records, using the standalone `RunRecord`
  fragment and admitting either its object or columnar rendering under a
  budget.

`timeout_s` blocks up to 300 s per call, because one blocked call replaces a
dozen status polls. Timing out is not a failure: the response comes back with
`timed_out` set and the job keeps running.

Cancel authority is the owning process or a valid control token; otherwise
`cancel_not_authorized`. The acknowledgement guarantees that no further case
enters submission: queued cases become cancelled, active ones get a
token-scoped kill. Waiting on a job owned by another process degrades to
bounded internal sidecar-refresh polling — the owner's event is not signalled
across processes — under the same output contract.

### 3.3 `analyze_results` — batched read

Identity model, normative across the surface: `case_id` / `run_index` is the
outer fan-out, `step_index` / `step_values` the inner `.step`, and the sample
axis is within a step.

Input:

```
sources    list[{job_id? | raw_path?, runs?: "all"|[int]|{case_ids}, label}]
recipes    list[Recipe]   Appendix A.2; unique key; optional per-recipe
                          sources: [label]
group_by   list[assignment param | "circuit" | step-axis name]
include    {per_run?: {limit?, cursor?} | bool, outliers?, signals_available?,
            provenance?, fields?: [dotted row path]}
budget     int | null
continue   {result_set_id, cursor}   resumes a budget-truncated call; mutually
                                     exclusive with sources/recipes
```

`continue` is the wire spelling; the Python attribute is `continuation`.

- `include.fields` is a presentation-only row view. The selected view is
  encoded inside every opaque cursor that resumes it, and every continuation
  applies that view over the neutral stored request. `fields` is excluded from
  per-run cursor request identity, so replaying page 1 with a different
  projection can mint a continuation for that new view.
- On a full-request `include.per_run.cursor` call, an omitted `fields`
  inherits the cursor's view. An explicit `fields` that differs from the
  embedded view is rejected with guidance to replay page 1 — never silently
  ignored. A cursor with no embedded view falls back to the stored request's
  view. The `continue` input surface itself does not change.
- A continuation replays the execution request stored in the result set and
  takes its presentation view from the cursor. Other request fields are
  rejected rather than accepted and dropped: raising `include.per_run.limit` on
  resume is the obvious thing to try, and silently ignoring it hands back a
  page the caller did not ask for.
- Validation, results and errors are per recipe: one bad recipe fails that item
  only.
- Reductions are attributed:
  `reduced[] = {stat, value, case_id, run_index, step_index?, step_values?,
  assignments}`.
- Spec verdicts:
  `{field, min?, max?, pass_count, fail_count, fail_cases: Page,
  verdict: "pass"|"fail"|"indeterminate", allow_incomplete?}` — indeterminate on
  incomplete coverage unless the caller opts out.
- A whole-call compute budget sits atop the per-parse deadlines. Exceeding it
  returns partial results plus a `result_set_id`, persisted in the working-dir
  store and expiring with jobs; an expired one has to be re-requested. Ask for
  more than one call's worth — every waveform of a 300-run Monte Carlo, say —
  and you get what was computed plus a continuation handle rather than an hour
  of compute.
- A `raw_path` source has no job provenance, so its rows carry
  `deck_sha256: null` plus an observation. Provenance is never fabricated.
- Bulk fidelity travels as artifact handles
  `{path, content_type, sha256, bytes}`. Inline waveform data is only ever
  bounded decimation under `max_points`, with `points_returned` and
  `points_total` declared.

Output: `outcome, coverage {runs_requested, runs_analyzed, missing_cases: Page},
results {key -> {metric, units, reduced[], groups?, steps?, spec?,
per_run?: Page, warnings}}, observations[], failures[], signals_available?,
source_hashes, result_set_id?, hint`.

The name `analyze_results` is broader than the naming rule prefers. It was kept
because in every measured run the model chose the right tool; the failures were
in how arguments were filled, never in which tool was picked. Renaming would
cost re-learning with no measured problem to fix.

### 3.4 `edit_schematic` — transactional sheet mutation

One input language: a typed op batch applied to a sheet. Creating a new file
and editing an empty one are the same operation, so there is no create/edit
mode split — only a parameter.

```
target              .asc path (created if absent)
base                "existing" (default) | "blank"
                    blank = treat the sheet as empty before applying ops;
                    existing = deltas preserving untouched content
expected_sha256     REQUIRED whenever target exists, under either base;
                    a mismatch is revision_conflict and nothing is written.
                    Both refusals — missing and mismatched — report the
                    target's current sha256, so a retry needs no extra read
ops                 list[Op] — Appendix A.4
reference           post-commit netlist compare, inside the transaction
dry_run             resolve, validate and return geometry; no write
write_failed_draft  quarantine a failed batch to
                    <target>.draft-<build_id>.asc (build_id server-assigned
                    and echoed)
return_views        subset ["touched", "pin_legend", "render"],
                    default ["touched"]
view_cursors        {label_only_pins?, pin_legend?, touched?} — each a
                    next_cursor from a previous page of that view
budget              int | null
```

`touched` is the pin/net table scoped to the references the op batch addressed.
The whole-sheet `pin_legend` was about 2,700 characters of unrequested default
per edit, so it stays available only by explicit request. An `occupancy` grid
view was specified and then measured: both arms of the comparison produced zero
placement defects — the transaction guards already prevent the defect class —
and the grid arm was strictly less efficient, so the variant was removed from
the enum rather than shipped.

The op union is discriminated on `op`. Undiscriminated, one mistyped op
produced an error per branch — thirty-odd of them, truncated — so the caller
learned neither the op vocabulary nor what their payload lacked, and the
observed recovery was importing a private module. Now an unknown op is one
error naming every kind, and a bad field on a known op reports against that
kind alone.

Unchanged fundamentals: the commit protocol (content-hashed support assets,
with the `.asc` rename last, so a multi-asset transaction rides a single-file
atomic primitive); the vocabulary (grid 16; rotations `R0..M270`;
`REF.PINNAME` pins; the legal symbol set is what `inspect` reports); the
validation errors (unresolvable endpoint, diagonal wire, wire over a symbol
body, pin collision, junction overlap); the wiring metric with
`label_only_pins`; findings carrying `at` and `subject`; and the retry story —
re-submitting with the original `expected_sha256` yields `revision_conflict`
if and only if the earlier call committed.

**There is no undo tool.** The surface has no `reset_schematic`. Instead every
mutating call presents the file's current hash, so a stale file is caught
*before* writing; a failed edit never touches the target; and the caller can
rebuild from its own ops. What is lost is a one-call "undo everything this
session". Specifying a real restore feature — snapshot ids, lifetimes,
cross-session rules — was judged worse half-done than absent.

**Wire ops.** A routed segment identical to one already on the sheet is not
drawn a second time; the response reports it under `already_present` and
`wire_count` counts only what was drawn. Removing an exact segment that exists
more than once removes every copy, but only when that leaves no pin newly
floating — otherwise the op refuses, naming the pin, with nothing written. A
segment that exists once is still removed unconditionally: that is an explicit
disconnection, not a tidy-up. A duplicate connects nothing and is
indistinguishable from a real second wire, so "delete the duplicate" and
"delete the connection" were otherwise the same request.

**Domain rule: the AUTHOR plane edits schematics and not netlists.** Both
`.asc` and `.cir` are text files, so the split is not about file format. The
criterion is where the tool adds something beyond text editing. For schematics
that is nearly everything — pin coordinates from the `.asy` plus rotation,
collision and junction checks, routing, rendering, the wiring metric. For
netlists it is nothing: an agent edits SPICE text natively as well as any tool
could. So `edit_schematic` aids schematic editing and offers no netlist
editing, and `verify_circuit` lints netlists but never mutates them. Future
schematic dialects (xschem `.sch`) enter behind this same ops surface, because
they share the property that made the tool worth having: geometry the caller
cannot cheaply resolve from text.

Hierarchy, when it arrives, needs one grammar extension *inside* the ops
language —
`{op: "define_block", name, ports: [{name, dir?}], ops: [...child-sheet ops]}`,
generating the block's `.asy` and child sheet, with instancing being the
existing `add_component` against that symbol. One language, hierarchical by
nesting, and no second document to keep in sync. A declarative
whole-circuit input document was prototyped and withdrawn for exactly that
reason: after create and edit merged, it was a second input language on the
same tool. A large ops batch expresses the same one-call whole-circuit build,
and it was measured doing so — every archetype plus a 51-component synthetic
builds in one `edit_schematic{base: "blank"}` call with zero rejections. What
that costs is block *definition* (ops can instance an existing subcircuit
symbol but cannot define a new block) and a whole-document validation pass.

Output: `outcome, target, sha256, build_id, stages[], netlist?, verification?,
wiring {pins_total, pins_wired, pins_label_only, label_only_pins: Page},
views?, warnings, failures, observations, artifacts, hint`.

### 3.5 `verify_circuit` — gate

```
path          .asc | .cir | .net | .sp
checks        subset [syntax, symbols, export, layout, quality, compare]
reference     path?
compare_mode  "equivalence" | "structural_diff"
anchors       list[str]?
rtol          float
render        {mode: "with_checks"|"only", delivery: "artifact"|"inline"|"both",
               format: "png"|"svg", scale?, max_pixels?}
              `true` selects the default policy; `false` or omitted renders
              nothing
export_to     "managed" (default) | "sidecar"
budget        int | null
```

`managed` export is non-destructive: it exports into a staged scratch directory
and leaves the caller's files untouched. `sidecar` overwrites the deck's `.net`
under lock and returns `{path, sha256, diff_vs_prior?}`; that makes the call
destructive, which the annotation table reflects.

Rendering uses the project's own SVG-to-PNG renderer; the `render` policy
controls format, scale, pixel cap, and whether the image comes back inline or
as a file.

Output: findings in the shared shape, a comparison block per mode, a render
block `{path, sha256, width, height, downscaled}`, a scene summary, `outcome`
and `hint`.

The historical per-rule finding cap applies only to layout and quality issue
rules and to `dropped_wire`; `dropped_wire` carries no truncation observation.

### 3.6 `inspect` — pure read

`queries: list[Query]`, with per-item results and cursors where listed:

```
{kind: "capabilities"}
    simulators and versions, exporter presence, dialects, persistence,
    allowed roots, profile, limits, linter_version, and the startup
    diagnostics that say whether this server started degraded
{kind: "symbols", path?, filter?, cursor?}
    legal symbol names and resolution order; `path` adds schematic-local
    directories to the reported precedence
{kind: "symbol", name, path?}
    pins per rotation, bbox, origin
{kind: "net", path, at: "REF.PIN" | "net:NAME" | [x, y], cursor?}
    .asc gives a geometric trace; a netlist gives card membership and makes
    no geometry claims
{kind: "components", path, prefix?, detail: "list"|"full", cursor?}
{kind: "model", mode: "search"|"enumerate", query?, libs?, cursor?}
    search requires query; enumerate requires libs
```

`path` is required except on `capabilities`, `symbols` and `symbol`.

**Why `inspect` and `edit_schematic` are separate.** The boundary is
read-versus-write, not amount of aid. `inspect` aids before or without mutation
— vocabulary discovery (which symbols exist, their pins per rotation) and
reading a foreign sheet (components, net truth) — and it is the surface's only
honestly read-only tool, which the annotation gating depends on.
`edit_schematic` aids *at* mutation time — coordinate resolution, validation,
default routing, warnings, the wiring metric — inside a transaction that must
carry destructive annotations. Merging them would put read-only lookups behind
a destructive-annotated tool. One apparent redundancy is resolved by the
canonical-route rule: `inspect(symbol)` is the vocabulary lookup, and
`edit_schematic` with `dry_run` is a whole-batch pre-commit validation, not a
symbol browser.

---

## 4. Cross-cutting contracts

**Annotations.**

| tool | readOnly | destructive | idempotent | openWorld |
|-|-|-|-|-|
| `run_experiments` | false | false | true (via request_id) | true |
| `jobs` | false | true (cancel) | true | false |
| `analyze_results` | false (artifact writes) | false | true | false |
| `edit_schematic` | false | true | false | false |
| `verify_circuit` | false (render, sidecar) | true (export_to: sidecar) | true (managed mode) | false |
| `inspect` | true | false | true | false |

**Ownership.** Visibility covers all persisted jobs; cancel authority is the
owning process or a control token, and the token is never disclosed through a
read.

**Caps.** Findings, failures and observations arrays are capped with declared
truncation.

**Untrusted parses.** Every raw or log parse — completion summaries included —
runs under a deadline and cooldown, on top of the whole-call budgets.

**Schema residency.** Every authorable field lives in the tool's
`inputSchema`: `oneOf`, literal discriminants, `additionalProperties: false`,
branch-local defaults, `$defs` allowed. MCP resources are documentation, not
schema. A property named `title` is not stripped from a published schema — the
title-annotation stripper used to filter that key at every dict level,
including inside `properties`, so a recipe field that the server accepted and
the handler read appeared nowhere in the schema dump.

**Evolution.** A variant's defaults, units and meaning never change. New
semantics get a new discriminant; an unknown one returns `unsupported_variant`
with the supported list.

**Argument spellings the schema advertises.** Each is a `BeforeValidator` with
`json_schema_input_type`, so a client can *discover* the alternative rather
than merely have it tolerated:

| tool | field | added spelling |
|-|-|-|
| `verify_circuit` | `render` | `true` = default policy; `false`/omitted = no render |
| `analyze_results` | `include` | a bare list of flag names becomes `{name: true}` |
| `analyze_results` | `include.per_run` | `true` = the default page |
| `run_experiments` | `analyze.include` | the same two spellings |

An unrecognized flag name still fails, enumerating the valid set and naming
`fields` as the one that takes row paths rather than a boolean. A `render`
value that is neither a policy nor a boolean is refused with the accepted
spellings and the `mode` values inline.

**Server logging.** The server's default stderr level is `WARNING`. A long INFO
startup banner on a door where the server's stderr is the caller's own stderr
gets answered with a blanket `2>/dev/null`, which then hides real tracebacks.
`[logging] level = "INFO"` or `LTSPICE_MCP_LOG_LEVEL` restores it; MCP protocol
log notifications are a separate channel and are unaffected.

---

## 5. What the six tools deliberately do not do

No deck text editing. No temperature or lib-section sweep knobs. No autorouting
or aesthetics. No trust verdicts. No library session state. No per-wire
interactive tooling. No partial writes to a named target. No unbounded returns.
No testbench-physics validation — the linter checks SPICE, not measurement
setup.

Circuits are passed as file paths; netlist text cannot be pasted into a tool
call. The agents this surface targets write files directly, and a pasted copy
would be a second version of the truth. The consequence is that the surface
assumes an agent on the same machine as the server.

---

## 6. The linter

Rules are a registry with dispositions:
`{rule_id, disposition: blocking | warning | observation, phase, provenance}`.
Only deterministic harvested failures block. Suppression is per call, and
`linter_version` travels in provenance.

Seed rules: `save-meas-coverage` (blocking), `meas-ngspice-batch` (blocking,
ngspice), `lib-section-ngspice` (blocking, ngspice in `kiltpsa` mode),
`model-missing` (blocking at staging), `directive-arity` (blocking),
`include-relative` (warning), `suffix-mega-milli` (warning), `temp-as-param`
(warning), and `op-degenerate` (a post-run observation with neutral evidence —
device list, currents, threshold, step — whose hint mentions `.nodeset`).

---

## 7. Where the old tools went

The pre-0.6.0 surface had 49 tools. The mapping:

- **`run_experiments`**: `run_simulation`, `run_sweep`, `run_montecarlo`,
  `configure_sweep`, `configure_montecarlo`.
- **`jobs`**: `check_job`, `cancel_job`, `recent` (the list-with-no-filter
  recent-circuits view).
- **`analyze_results` recipes**: `bode_metrics`, `stability_metrics`,
  `signal_stats`, `edge_metrics`, `timing_between`, `periodic_metrics`, `thd`,
  `noise_integral`, `operating_point`, `measurement_stats`, `query_value`,
  `get_waveform`, `export_waveform`, `batch_results`, `simulation_summary`
  (as `metric: "summary"`, including Fourier, AC bandwidth and suggestions),
  `ac_structure`, `resonance`, `return_loss`, `transient_response`.
- **`edit_schematic`**: `create_schematic`, `apply_schematic_ops` (op models
  carried over verbatim), `wire_pins`.
- **`verify_circuit`**: `export_netlist` (sidecar delivery plus diff),
  `validate_netlist`, `diff_circuit` (as `structural_diff`),
  `render_schematic`.
- **`inspect`**: `trace_net`, `symbol_info`, `component_info`
  (`detail: "full"`), `find_model` (`search` and `enumerate`),
  `list_components`, `server_status` (as `capabilities`).
- **Dropped, with the loss stated**: `create_netlist`, `read_circuit`,
  `set_component_value`, `parameter`, `edit_directive`,
  `load_library` / `unload_library` / `list_libraries`, and `reset_schematic`
  (the in-session byte-restore hatch, superseded by revision guards and
  quarantine drafts).

`plot_waveform` was not folded in; it stayed a registered tool.

---

## 8. Appendix A — payload grammars

The variation and recipe unions are strict discriminated unions written for
this surface. Their field *semantics* bind to the existing engines
(`lib/montecarlo.py` sampling math, the analysis adapters, the reducer
categories in `lib/recipes.py`); their *shapes* are contracted here and inlined
into `inputSchema`. They exist because free-form `options` dictionaries were
what produced the only measured argument failures.

### A.1 Variation

```
{kind: "assign", id?, combine: "grid"|"zip" (default grid),
 applies_to?: [circuit id],
 assign: dict[target -> list[number | SI-suffix string]]}

  Target resolution is deterministic, with no silent precedence.
  Syntactically explicit forms are selected before bare-target resolution:
    "REF@model", or the glob "M*@model"  -> instance model swap
       (the corner idiom: {"M*@model": ["NTT", "NSS", "NFF"]})
    "INSTANCE:{delvto|mulu0}"            -> explicit per-instance mismatch
       value; "X1:delvto" for a single-FET body, or the qualified
       "X1.M0:delvto" for a multi-FET body. Supported only on
       ngspice-compatible BSIM3/4 devices through exactly one X -> M wrapper
       level, and only as a target key in this mapping.
  Otherwise, a declared .param name -> parameter substitution;
  else a component reference        -> value substitution;
  else                              -> ambiguous_target.

  combine: "zip" requires equal list lengths.

{kind: "random", id?, runs: int >= 1, seed?: int, applies_to?: [circuit id],
 rules: [RandomRule]}          at most ONE random entry per call

RandomRule:
  {rule: "component", target: ref | glob, tolerance, scale:
   "relative"|"absolute", distribution: "normal"|"gaussian"|"uniform"}
     tolerance is a plus-or-minus bound; normal is N(0, tol/3) truncated at
     plus-or-minus tol (the 3-sigma convention); relative is a fraction of
     nominal, absolute a bound in source units. There is no lognormal — the
     engine does not have one.
  {rule: "param",  target: param, same fields}
  {rule: "model",  target: model, param: model-param, same fields}
  {rule: "mismatch", ...}   the Pelgrom-form mismatch rule: prefix, avt, ak,
                            params, area floor. Not a pair glob.
```

The `INSTANCE:{delvto|mulu0}` assignment form is never a random-rule target.
Random mismatch on flat devices follows the ordinary flat-device path; a
`rule: "mismatch"` prefix that descends into an X-wrapped PDK device inherits
the ngspice-compatible BSIM3/4 and one-level `X -> M` constraints above. Draws
across circuits are independent — correlated Monte Carlo across decks is out of
scope.

### A.2 Recipe

21 discriminant values. Shared fields — `key` (required and unique),
`sources?`, `step` XOR `all_steps`, `reduce`, `reduce_field`, `spec` — are
accepted only where the reducer category allows: the per-variant accepts-matrix
binds to the scalar / multi-field / keyed / variable-length categories in
`lib/recipes.py`, and non-reducible variants reject `reduce` and `spec` at
validation.

| discriminant | run type | own required fields | notes |
|-|-|-|-|
| `summary` | any | — | full summary payload: sim type, ranges, signals, measurements, Fourier, AC bandwidth, diagnostics, suggestions |
| `measurements` | any | — | `names?`, `histogram_bins?` (0 = none); returns the `.meas` table plus `failed_measurements` |
| `value` | any | `expr` | `at?`; step-aware |
| `signal_stats` | tran | `signal` | `window?` |
| `edges` | tran | `signal` | `levels?`, `edge?`, `window?` |
| `timing` | tran | `from{signal, edge, level}`, `to{...}` | `nth?`, `window?` |
| `periodic` | tran | `signal` | `window?`; period, frequency, duty cycle |
| `transient_response` | tran | `signal`, `mode: "step"\|"disturbance"` | `input` is required for `disturbance` and rejected for `step`; `window?` |
| `thd` | tran | `signal` | `fundamental_hz?`, `harmonics?` (default 7), `window?` |
| `bode_filter` | ac | `signal` | filter characteristics: fc, bw, Q, type |
| `bode_point` | ac | `signal`, `at_hz` | gain and phase at a frequency |
| `bode_crossing` | ac | `signal`, exactly one of `level_db` / `level_deg` | `level_deg` scans the UNWRAPPED phase, so a crossing past 180 degrees is found once rather than at every wrap; `phase_deg` remains accepted as an alias |
| `bode_slope` | ac | `signal`, `from_hz`, `to_hz` | dB per decade |
| `stability` | ac | `signal` | phase margin = 180 degrees + phase at unity gain, positive being stable. Reducible leaves: `phase_margin_deg`, `gain_margin_db`, `unity_gain_hz`, `dc_gain_db` |
| `ac_structure` | ac | `signal` | structural poles and zeros |
| `resonance` | ac | `signal` | resonant frequency, Q, peak |
| `return_loss` | ac | `signal` | `z0?` (default 50); return loss in dB, VSWR, reflection coefficient |
| `noise_integral` | noise | — | `signal?`, `from_hz?`, `to_hz?` |
| `operating_point` | op | — | `device?` — scoping to one device is the difference between a few hundred bytes and tens of KB on a real opamp |
| `waveform` | any | `signals` | `max_points?` (default 2000), `format: "inline"\|"csv"`, `window?`. Inline is bounded decimation only, with `points_returned` / `points_total` declared; `csv` returns an artifact handle |
| `plot` | any | `signals` | `title?`, `log_x?`, `span?` — returns an artifact handle |

### A.3 Edit ops

Eleven op kinds, discriminated on `op`: `add_component`, `move_component`,
`remove_component`, `set_component_value`, `set_component_attribute`,
`add_net_label`, `remove_net_label`, `wire_pins`, `remove_wire`,
`add_directive`, `remove_directive`. The batch commits atomically or not at
all: the first op that fails aborts the transaction and nothing is written.

Carried over from the pre-consolidation op models: exact-segment `remove_wire`,
`cleanup_wires`, directive placement kinds, and literal-default removal
matching with `regex:` as an opt-in prefix. The deprecated `connect` spelling of
`wire_pins` is *excluded* from this union.

### A.4 Inspect queries

§3.6 is exhaustive.

---

## 9. Known gaps

Recorded so they are not mistaken for oversights:

- Foundry-PDK subcircuit mismatch Monte Carlo. Wanted for bandgap statistics;
  a large separate project.
- Correlated Monte Carlo draws across two decks.
- Canned loop-gain probe templates.
- Re-running an attached analysis stage after a server restart.
- Request-index records are not pruned today, so spot-check volume grows the
  index. That is the same growth class as job sidecars.
