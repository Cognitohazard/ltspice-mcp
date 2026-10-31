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
| UNDERSTAND | `analyze_results` | runs a batch of recipes over runs and returns typed values with case and step identity, attributed reductions and spec verdicts; replies are bounded and can be continued |
| UNDERSTAND | `inspect` | reads: capabilities, symbols (list and detail), net trace, components, models, result facts, the guide, and the reference entries for the tools' arguments, recipes, ops and other branches |
| AUTHOR | `edit_schematic` | applies a batch of typed ops to a sheet (blank or existing) as one transaction; committing to an existing file needs its current `expected_sha256`; returns pin and net tables, per-op results and findings |
| AUTHOR | `verify_circuit` | gate: lint/syntax, symbols, export, layout, quality, compare (equivalence or structural diff), render |

A seventh tool, `plot_waveform`, is the interactive MCP Apps waveform widget
and does not use the envelope. Its reply summarizes each plotted trace:
minimum and maximum with their positions, first and final value, and on a
transient the time-weighted mean. `attach_plot` adds a PNG of the chart for a
vision model. Traces are split into panels by declared
unit, or by the caller's `panels`. A plot with no sampled axis (an operating
point, a `.tf`, or a stepped `.op` or one-frequency `.ac`, which hold one point
a step) has no chart: it is refused with `no_axis`, which says what the plot
holds and where its values are read, and with `in_ltspice` it is opened in
LTspice without a chart, with no traces named for it and a `no_chart`
observation.

An eighth tool, `run_code`, is always registered and is served unless the
operator sets `[tools] run_code = false`. It runs a Python snippet in a warm
worker process that holds the engine as `api` (the six operations as methods
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
`error {code, message, stage, retryable, commit_state, item_id?}`.
`commit_state` is `not_started` when nothing was recorded or written,
`committed` when the job or the edit was recorded and a later step failed (so
do not resubmit under a new id), and `unknown` when the server cannot tell, as
when another submission holds the `request_id`. MCP's `isError` is set only
for a call-level failure; a per-item failure sets `outcome: "partial"`.

**Unknown tool names.** A call to a tool the server does not have fails before
any tool runs, with a JSON-RPC invalid-params error (`-32602`) whose message
gives the requested name and lists the available tools. An unknown resource
URI gets the same error.

**Error codes.** `code` is a stable public name for a failure. It depends on
the kind of failure and never on the message, except that a failure inside a
named stage may report that stage's code. A result read that fails while a
deck is being staged reports `submission_failed` instead of
`result_unreadable`.

Codes by kind of failure:

| code | meaning |
|-|-|
| `path_denied` | a sandbox (`allowed_paths`) is set and the path resolves outside it, or the path cannot be resolved; a `hint` describes the change to make (for a sandbox, the config file and key that widen it), returned as an item's `error.hint`, a finding's `evidence.hint`, a failure or coverage row's `hint`, or the envelope's `hint` for a call-level refusal |
| `netlist_invalid` | the netlist, or a component reference in it, could not be read |
| `symbol_unresolved` | the schematic opened, but a symbol or library it refers to was not found |
| `simulation_failed` | the simulator could not be started, or the run failed |
| `result_unreadable` | a `.raw` or `.log` could not be read |
| `analysis_deadline` | a result read ran past its wall-clock bound and was abandoned |
| `no_axis` | the selected plot is a table (an operating point, `.tf` or `.pz` result), so it has no sweep axis to query at |
| `job_not_found` | no job with that `job_id` or `request_id` |
| `idempotency_conflict` | a `request_id` was reused for a different payload |
| `request_gate_busy` | another submission holds that `request_id`; retry with the same id |
| `cancel_not_authorized` | the call reached a server process other than the one running the job (the job's owning process) and presented no matching `control_token` |
| `cancel_failed` | cancellation was authorized but could not be carried out |
| `library_error` | a component library failed to load, parse, or resolve |
| `batch_job_error` | a sweep or Monte Carlo config could not be used |
| `raster_unavailable` | a PNG was requested without a usable rasterizer, because the optional `raster` extra is not installed, or only `cairosvg` is installed without the native Cairo library it loads |
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

When one page contains several collections (an `inspect` net view pages pins
and wire coordinates together), `total` and `returned` are sums over all of
them, a `collections {<data key>: {total, returned, truncated}}` block gives
each collection's counts, and `primary_truncated` reports whether the primary
collection continues. On every page, `truncated` implies `returned < total`.
Stop paging when `next_cursor` is null, not by the counters.

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
including any `hint`; the text channel is presentation only. When a sandbox
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

An empty per-circuit lint entry is omitted, so a circuit absent from `lint`
had no findings; findings themselves are always emitted.

Pagination is exempt from the lean rule, so `next_cursor` is always present
and is null when there is no next page.

Run rows are also exempt, so every run row has the same keys regardless of its
status, the tool that returns it or the budget. Use `run_fields` for a
narrower row.

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
describes this call, and the replay does not change the job's record. The same
id with a different fingerprint is an `idempotency_conflict` that names the job
created under that id, which `jobs` and `analyze_results` read by its
`job_id`. Ids are scoped to the server working directory's job store, and an
id is deleted with its job ("Job model" below says when). If the client
cancels a `run_experiments` request while the call waits for results
("Receipt, then dwell" below), only the wait ends; the job keeps running.

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

A named build is recorded, parsed and linted as its family, so a record's
`simulator` is the family and `simulator_executable` distinguishes two
builds. Each build has its own limit on concurrent runs, so each gets the
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
                      simulator_seed?: int, keep_partial_raw?: bool}
analyze              {recipes: list[Recipe], group_by?, step?, all_steps?,
                      plot_index?, dialect?, include?}
                                        attached analysis stage; `recipes` is
                                        the Appendix A.2 union `analyze_results`
                                        takes, and `step`/`all_steps` mean what
                                        they mean there, so an attached
                                        measurement and a standalone one read
                                        the same `.step` iterations. Every
                                        recipe is validated at SUBMISSION,
                                        before a deck is staged.
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
rendered and do not change what runs. A reply over its budget is rendered
again with each rung of a fixed ladder in turn until it fits. The trim rung
removes empty presentation blocks and the `source_hashes` rows (the identity
echo), the answer rung drops the caller's opt-ins that enlarge the reply
(such as `include.per_run`, `include.outliers` and `include.provenance`), and
the shrink rung lowers the number of rows on every page. A call with no
budget is rendered under the server default (`[analysis] default_budget`,
default 4000), which applies only the trim rung. Trim never removes facts or
caller opt-ins. An explicit budget replaces the default and may use all three
rungs.

**Simulator choice.** `execution.simulator` names one of the four supported
families, or a named executable of one (above); the schema's pattern accepts
all four. A name is accepted when it is detected (for a named executable,
when its entry was accepted at startup) and this host can run its family.
`inspect(kind="capabilities")` reports this per simulator and per named
executable as `selectable`, with a `refusal` giving the reason when it is
false. An undetected family gets a "not available" error naming what was
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
`inspect(kind="capabilities")`, `null` when unset). A timeout deletes the
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
readers (`file=`, `sfile=`, `pwlfile=`, a PWL that names a file), directives
outside the supported set, functions the server does not know, and random
functions with no seed the server records, because a resumed case could not
be relied on to run them the same way. ngspice runs with `-n` (no user
`.spiceinit`) and with `SPICE_SCRIPTS`, set only for that subprocess, pointing
at a system startup file in the job's folder that contains only a comment, so
the machine's startup files do not affect the run. A `pdk_native` sample
(Appendix A.1) keeps its recorded sample index and seed. A resume or a later
launch is refused with `recovery_execution_changed` if the executable, its
command line or the platform has changed since submission. A job without
`recoverable` runs the simulator with the machine's own startup files.

`execution.simulator_seed` runs each case's saved deck through a short ngspice
script that sets the random seed (`setseed`) immediately before it loads the
deck. It requires recoverable ngspice and an integer (not a float or boolean)
in `1..2147483646`; every case and retry starts from that seed. It cannot be
combined with a `pdk_native` variation, which sets its own seeds; a request
using both is refused. Without it, no seed is set, and a recoverable deck may
not call random functions.

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
`server_restarted` or `job_deadline` are retried by default; other failures
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
is a conflict. A replay is authorized as the original call is, by the job's
owning process or its `control_token` (otherwise `resume_not_authorized`).
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
`queued -> running -> analyzing -> completed | completed_with_failures | failed | cancelled`,
and a job whose server stopped before its runs finished is `interrupted`.
The dwell and `jobs(wait)` wait for full terminality by default;
`jobs(wait, wait_for: "runs")` returns once the runs are terminal. Cancelling
during analysis cancels the stage and keeps the run artifacts. An analysis
failure puts the job in `completed_with_failures` with an `analysis_error`; a
new `analyze_results` call can still analyze the runs. After a server restart,
a pending or running analysis stage is marked failed with a restart
observation and is not re-run. The analysis result is stored in the job
record as one rendered page, and its summary is included in every terminal
receipt.

**Input snapshot and provenance.** At submission the server stages the
primary deck (content-addressed) and every include and lib it reaches (with a
sandbox set, those inside it or in the simulator's own library), to a
recursion depth of 8. Staged copies keep their relative layout. The manifest
records `{path, sha256, staged, live, staged_path, reason, section}` for every
reference, including `.lib` section selections. Runs read only the staged
copies, so editing an original afterwards does not change the job, and every
reported number cites the hash of the deck that produced it.

By default, a reference that cannot be staged fails that circuit's submission
with an `include_unstaged` error naming the reference. With
`allow_live_includes: true`, runs read those files in place, the manifest
marks them `live: true`, and an observation reports the weaker provenance.
Every live reference is listed.

Provenance in the response is opt-in. Digests, staged paths, the full
manifest, the linter version and the job's `simulator_executable` are returned
only with `provenance: true` on `run_experiments`, or `include.provenance:
true` on `analyze_results` (which omits the executable). The lean receipt
keeps `{circuit, path, simulator, dialect, staged_files}` and every manifest
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

**RunRecord** is a schema fragment shared by `run_experiments` and `jobs`:
`{case_id?, run_index?, circuit?, assignments?, status?, raw?, log?,
other_raws?, simulator_version?, attempt?}`. Every key is optional because a
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
is none of those (LTspice XVII, QSPICE). It is null when the output names no
build. It is recorded per case, so an executable replaced during a job shows
as two builds. Runs are rendered as `items: [RunRecord]`, each row an object
with the same keys at every budget.

Under a `budget`, receipts keep the common envelope and step down the
trim -> answer -> shrink ladder described under the input above. The trim rung
empties the attached analysis's `source_hashes` (one identity row per run);
`source` stays, because it lists the includes that were read in place, not
staged, or staged with a reason. The runs page and the attached analysis's
row surfaces may be shrunk to fewer rows; fact channels, completeness,
progress, verdicts, coverage and the ids, token and cursors a caller needs to
continue are never cut. If the floor (the smallest
response the ladder can produce) exceeds the budget, the floor is returned
with `budget_not_met`; facts are not dropped.

One page limit applies to every row surface of a receipt, which means the run
page and each attached recipe's values, groups, failing cases and missing
cases. The shrink rung lowers it to the largest limit that fits, which may be
zero. A receipt's rows preview data that other calls page (`jobs(runs)`,
`analyze_results` over the `job_id`), so the floor has no per-case rows. In
the floor, `completeness` and `runs.total` count the runs, `runs.next_cursor`
starts `jobs(runs)` at the first run, each recipe's warning counts its omitted
rows, and reductions and spec verdicts remain. The floor's size does not
depend on the number of cases. `jobs(runs)` and `jobs(list)` keep at least one
row at their floor.

The lean row flattens `value` to its scalar leaves and keeps any `artifact`
handle, both in `analyze_results` and in a `run_experiments` receipt.

Receipt failure rows with the same `code` and message (numbers in the message
ignored) are merged into one row. That row is the merged failure whose
`case_id` sorts first, not the first to finish, so two identical runs report
the same excerpt. `count` is the number of failures merged, and `case_ids`
lists at most 10 of their case ids.

Output: `job_id, request_id, control_token, status, outcome, source[],
completeness, progress, lint findings (per circuit), runs (a page of
RunRecords), analysis?, failures[], observations[], artifacts[], hint`.

### 3.2 `jobs` — job status and control

```
{action: "status", job_id | request_id}
{action: "wait",   job_id | request_id, timeout_s (default 60, capped at 300),
                   wait_for: "all" (default) | "runs"}
{action: "cancel", job_id | request_id, control_token?}
{action: "resume", job_id, resume_request_id, control_token?, case_ids?,
 retry_failed?, retry_cancelled?, wait_s?}
{action: "list",   circuit?: path, limit? (capped at 50), cursor?}
{action: "runs",   job_id | request_id, cursor?}   cursor absent = first page
{action: "place_beside_sheet", job_id | request_id, case_id?}
```

An action on one job takes `job_id`, `request_id` or both; when both are given
and name different jobs, the call fails with `selector_conflict`.

Each action accepts only its own fields and rejects others; every action also
accepts `budget`. The input schema declares one branch per action, keyed by
`action`, and the output shape is discriminated on the echoed `action`:

- `status` / `wait` return the §3.1 receipt **without** `control_token`, plus
  `analysis_status` and, for `wait`, `timed_out`. The receipt includes
  `progress`, terminal states included, and follows the same budget rules.
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
  activity first, each `{job_id, status, request_id, finished_at}`, and
  `recent_jobs_total` counts them all.
- `runs` returns a page of `RunRecord` rows.
- `place_beside_sheet` copies one case's results beside the `.asc` it ran,
  under the sheet's name, where LTspice keeps a sheet's own results and opens
  them from: the raw, the log, and each other raw the run wrote
  (`<sheet>.op.raw`). It returns `{case_id, sheet, files, assignments}`, with
  one `{from, to, replaced}` per file copied; `assignments` is what the
  case's variations set, which the sheet does not show. `case_id` may be left
  out of a job that ran one case (otherwise `case_id_required`). The copy is
  made under the sheet's lock, and the job keeps its own. A case that
  produced no results (`case_not_produced`) or whose results are no longer
  there (`results_missing`), a netlist's run (`not_a_sheet`),
  another simulator's run (`not_ltspice`), and a sheet that is missing
  (`sheet_missing`) or changed since the job read it (`sheet_changed`) are
  refused, and nothing is written. The sheet's `.plt` is left as it is.

`wait` blocks for at most `timeout_s`, up to 300 s per call. A timeout is not
a failure; the response has `timed_out` set and the job keeps running.

**Capped values.** `execution.wait_s` above 120, `timeout_s` above 300, a
`list` `limit` above 50 and a `per_run.limit` above 100 are served at the cap
instead of being refused. The response reports the cap as a warning on
`run_experiments` and `jobs`, and in the `hint` on `analyze_results`, which
has no top-level `warnings`. The schema declares no `maximum` for these
fields.

Only the owning process or a caller with a valid control token can cancel;
any other caller gets `cancel_not_authorized`. Once a cancel is acknowledged,
no further case is submitted, queued cases are cancelled and active ones are
killed. A wait on a job owned by another process re-reads that job's record
every 0.5 s and returns the same output.

### 3.3 `analyze_results` — batched read

**Identity model.** Across the surface, `case_id` / `run_index` identify one
case of a job (one circuit with one set of variation values), `plot_index`
selects one plot in a raw file, `step_index` / `step_values` select one
iteration of the deck's own `.step`, and a sampled axis (time, frequency or
the swept source) lies within a step. A table plot, one whose first trace is
not a sweep axis (such as an `.op`, `.tf` or `.pz` result), has no sampled
axis. `snapshot_id` identifies the contents of a source's RAW, log and
console files as they were read.

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
                                  first
all_steps  bool (default false)   evaluate at every `.step` iteration; mutually
                                  exclusive with `step`
include    {per_run?: {limit? (capped at 100), cursor?} | bool, outliers?, signals_available?,
            provenance?, fields?: [dotted row path]}
budget     int | null
continue   {result_set_id, cursor}   resumes a budget-truncated call; request
                                     fields sent with it must be the stored
                                     request's own
```

`continue` is the wire spelling; the Python attribute is `continuation`.

`plot_index` is null or a strict nonnegative integer; when it is omitted, the
first plot is selected only if a RAW recipe reads one. A source whose `path`
is a `.log` refuses plot selection. `dialect` is null or one of `ltspice`, `ngspice`,
`qspice`, `xyce`; an explicit dialect must agree with the raw file's header
and with the simulator the job recorded. `plot_waveform` and the attached
`run_experiments.analyze` block accept the same selectors and the same
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
`dialect` the caller gave, or else the one the job recorded, which may be null
because the row is made without decoding the raw. Both include `plot_index`
and `snapshot_id`. `step_values` come from the `.step` lines of the run's own
log as read when the result set was created, string values included; no other
log is read for them.

The `measurements` recipe reads the `.meas` results in each source's log once,
including a `.log` `path` source and a finished job case with no RAW. Its rows
have null `plot_index`. With neither `step` nor `all_steps`, a row holds each
measurement's statistics over every step and null step fields (the default row
omits them). With `all_steps` the recipe returns one row per step, and with
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
report only the producing dialect, which is null for a `path` source. A continuation reports `source_drift` for a source whose
log or console file changed, or appeared after being absent, as it does for a
changed RAW.

When a run has one value per `.meas` (each case of a sweep with no `.step`
does), its `measurements` row contains `measured: {name: value}`, the number a
reduction or spec reads for that name; `at: {name: point}` where the log
printed a crossing or `AT` point with it; and `failed_measurements: [name]`,
each name the log has no number for (LTspice printed it as FAIL'ed, or
printed a non-finite value). A failed name's `measured` entry is null, not
absent, and a null there is always listed in `failed_measurements`. The
default row omits the per-name `stats` block; `include.fields=["value"]`
returns it, and reductions and specs read it either way. A stepped run's row
has only `stats`.

`step` and `all_steps` apply to the whole call, so every recipe reads the same
iteration. `run_experiments`' attached `analyze` block takes
the same two arguments (§3.1). Both are stored in the result set, so a
continuation replays them.

- `include.fields` paths root at one of `source`, `case_id`, `run_index`,
  `step_index`, `step_values`, `assignments`, `circuit`, `deck_sha256`,
  `plot_index`, `dialect`, `snapshot_id`, `value`; an unknown root is rejected
  rather than returning empty rows. A dot inside a key's own name is escaped
  as `\.`, so a subcircuit node or a device parameter is spelled
  `value.voltages.v(x1\.out)` or `value.device_op_points.@m\.x1\.m1[gm]`.
- `include.fields` changes only the row view. Every cursor records the
  `fields` view it was made for, and a continuation applies that view to the
  stored request. A `per_run` cursor is checked against the request without
  its `fields`, so sending page 1 again with different `fields` returns
  cursors for the new view.
- On a full request containing `include.per_run.cursor`, an omitted `fields`
  takes the cursor's view. An explicit `fields` that differs from the cursor's
  view is rejected with a hint to replay page 1. A cursor with no embedded
  view uses the stored request's view. The `continue` input itself does not
  change.
- A continuation replays the request stored in the result set and takes its
  view from the cursor. Request fields sent with it are compared with the
  stored request. An echo of the original call is accepted, and a differing
  field, such as a raised `include.per_run.limit`, is rejected by name.
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
  failures channel, which no budget rung trims, does not grow with the run
  count. The page then caps rows at 100,
  with a `failures_truncated` observation. The Python API's complete
  `analyze_results` result lists every record, one per place; an analysis
  attached to an experiment is stored as the page renders it.
- Under a `budget`, the shrink rung lowers one shared limit on every row
  surface (each recipe's `values`, `groups`, `per_run` page and
  `spec.fail_cases`, and `coverage.missing_cases`) to the largest value whose
  rows fit, down to one row; the `per_run` and missing-cases pages continue
  through their cursors. `reduced` is never shrunk.
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
- Each call has a total time limit, `[analysis] analysis_budget_s` (default
  60 s), on top of the deadline on each parse. Exceeding it returns partial
  results and a `result_set_id`, stored in the working directory's store. A
  result set that reads a job is deleted when the job is ("Job model", §3.1);
  one that reads only `path` sources expires after
  `[analysis] result_set_ttl_hours` (default 24). After it is gone, send the
  request again. A request for more than one call can compute, such as every
  waveform of a 300-run Monte Carlo, returns what was computed and that
  continuation handle.
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
- A job's artifacts are read only once the job is terminal and the case
  produced them.
- Before any recipe runs, the server reads every source once to record its
  file hashes and the simulator errors its log reports. A timeout during this
  first read raises `analysis_deadline` and creates no result set or cursor;
  retry the original request with fewer sources or a larger
  `[analysis] analysis_budget_s`. The timeout is not stored, so a retry reads
  the sources again. After that read, every call completes at least one recipe
  evaluation, and work left over when the budget runs out continues through
  the returned `result_set_id`.
- A `path` source has no job provenance, so its rows have
  `deck_sha256: null` and an observation.
- Full-resolution data is returned as artifact handles
  `{path, content_type, sha256, bytes}`. Inline waveform data is always
  decimated to at most `max_points`, with `points_returned` and
  `points_total` reported.

Output: `outcome, coverage {runs_requested, runs_analyzed, missing_cases: Page},
results {key -> {metric, units, reduced[], groups?, steps?, spec?,
per_run?: Page, warnings}}, observations[],
failures[] {code, stage, where, message, wheres?, count?}, signals_available?,
source_hashes, result_set_id?, hint`.

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
                    only when the verdict is not a match
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
field names.

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
`preexisting`. A pin's `net` is the name a label gives its net, null when none
does. LTspice names such a net only when it exports the sheet (`N001`,
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
of the parts the batch did not rewire or name:

```
preexisting {count, findings, label_only_pins, cursor}
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
  `window_check`, naming a few of the differing entries, and writes nothing.
  The user resolves it by saving the window to keep its version, or by
  closing it without saving to keep the file's. A dry run reports the
  difference in `observations` and is not refused. The comparison is by
  content rather than text, because LTspice returns its own serialization of
  a sheet. That serialization differs from the file it read in the first
  line, the order of wires and attributes, attributes with no value, the
  micro sign and off-grid text, and none of those differences counts as a
  change.
- **After the commit**, the committed sheet replaces the window's copy;
  LTspice records this as one step in that window's undo history.
  `open_in_ltspice` has one entry per window, `{pid, version, shown,
  reason?}`. A window that could not be updated (`shown: false`) does not fail
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
entry names each text it removed, with its kind and place (`removed_texts`).
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
`{x, y, via: "endpoint", wire}`, where `wire` is the segment joined. The new
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
where they cross, a wire left diagonal, and a net given a second name are each
reported once, by the sheet finding (`wire_through_symbol`, `symbol_overlap`,
`attribute_text_overlap`, `dangling_label`, `unjoined_crossing`,
`joined_at_crossing`, `diagonal_wire`, `net_with_two_names`), whichever op
made it. Findings are computed on the sheet the batch leaves, so a state the
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
be grounded after the first. `inspect(kind: "net")` at a point on a wire's
interior traces that wire's net and names the wire under `snapped_to_wire`; a
point where two nets' wires cross is refused as ambiguous.

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
complete; empty on success), netlist? (only when a
compare did not confirm equivalence), verification?,
wiring {pins_total, pins_wired, pins_label_only, label_only_pins: Page},
rules_run {rule: count}, preexisting {count, findings, label_only_pins, cursor},
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
export_to     "sidecar" (default) | "managed"
```

`compare` (`{reference, anchors, rtol, mode}`) means the same here as in
`edit_schematic`.

**In LTspice, on request.** `in_ltspice` opens the checked file, sheet or
netlist, in the running LTspice window and brings it to the front, through the
bridge LTspice ships. The reply's `ltspice` block reports the result:
`{shown, path, pid, version, already_open, differs_from_file?, difference?,
reason?}`. A sheet the window already had open is brought to the front as the
window has it, because LTspice does not read a file again; its copy is
compared with the file, and `differs_from_file` is set when the user is
looking at something other than what was checked. With no window open,
LTspice is started in view (`started: true`); with `[schematic]
start_ltspice` off, the block reports that no window is open and the check
results are still returned. `in_ltspice` never fails the call.

`sidecar`, the default `export_to`, overwrites the schematic's `<name>.net`
under lock and returns `{path, sha256, diff_vs_prior?}`. LTspice writes the
same file beside a schematic it runs from its own window, so the default
export writes nothing such a run would not; it does make the call
destructive, as the annotation table shows. `run_experiments` exports a
schematic the managed way and writes nothing beside it. `managed` leaves the
caller's files alone. It copies the schematic and the project-local
`.asy`/`.lib`/`.sub`/`.inc`/`.mod` files under its folder into a scratch
directory in the store, exports there, and writes nothing beside the caller's
file (the lock it takes while copying is in the per-user folder). A compare of
that export resolves relative includes from the schematic's folder, not from
the scratch copy.

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
image block, and an SVG as its markup in `render.svg`. A PNG needs the
`raster` extra (`resvg-py`), or `cairosvg` with the native Cairo library it
loads; without either the render falls back to SVG, and `render.note` names
what is missing and its remedy. The fallback is also a `render` failure when
the caller named `format: "png"`. A default that falls back is not a failure,
because the caller asked for nothing the server could not do.
`inspect(kind:"capabilities")` reports the same facts as
`render: {png, missing, reason, remedy}` before anything is drawn.

`equivalence` compares the two netlists as graphs of parts and nets: the
component set, values, normalized parameters, which pins share a node
(matched by the wiring, not by node names), the number of terminals, and
`anchors`, the named nets (ports, rails, outputs) that must also match by
name; ground always must. `structural_diff` reports the added, removed and
changed components and directives. The modes differ on a deck they cannot
read.
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
export, and an `.asc` reference is exported the same way, on a staged copy,
never beside the caller's file; without LTspice the compare fails rather than
diffing a schematic's attributes against a netlist's cards. `structural_diff`
lexes both decks, so a `+` continuation belongs to the card it continues, and
compares directives as parsed rather than as text, so spacing, case, comma
separators and assignment order in `.model`/`.param` are ignored. It ignores
the lines the netlister adds to every export, which are `.backanno`, the
install-path `.lib …/cmp/standard.*`, and a parameterless default model such
as `.model NMOS NMOS` when the reference declares no model of that name. A
component's signature is its model or value plus its instance parameters,
without nodes (`equivalence` is the mode that compares wiring); a parameter
given twice contributes both copies, so a second copy counts as a change.

Output: findings in the shared shape, a comparison block per mode, a render
block `{path, sha256, source_sha256, width, height, downscaled,
returned_inline, svg?, note, …}`, a scene summary, `outcome` and
`hint`.

In the render block, `sha256` is the image's digest and `source_sha256` the
digest of the sheet file this call drew. Rendering reads the file on disk, and
another session sharing the working directory may commit between
`edit_schematic` returning and this call running; compare `source_sha256` with
the `sha256` `edit_schematic` returned to confirm the image shows that
revision. The export block's `sha256` is the netlist's.

The layout check reads a part's attribute text where LTspice draws it, with
its extent estimated from the font's widths rather than measured off its
glyphs. `attribute_text_overlap` reports text that a wire, a line another
part draws or a ground flag passes through, or that lies over a rectangle or
circle another part draws or over other text, by more than the estimate's
slack. A ground flag is taken as the triangle the render draws, turned as
LTspice turns it; its size is the render's and is not recorded. Comments,
directives and labels are checked only where attribute text crosses them, and
the layout check reports this limit in `observations`. No op moves a part's
text apart from the part, so the remedy is to move the part or the item its
text crosses. `wire_through_symbol` and `text_in_symbol_body` likewise test
what a part draws, not its box, so a wire or a text anchor in the empty corner
beside an amplifier's triangle is not in the part.

A cap of 25 findings per rule applies to the layout and quality rules, to the
value-suffix rules (`value_suffix_micro_sign`, `value_suffix_mojibake`,
`value_suffix_nonascii`) and to `dropped_wire`. A capped rule other than
`dropped_wire` adds an observation `<rule>: showing 25 of <n> findings`;
`dropped_wire` adds none, and `rules_run` gives its full count.

**No `baseline` filter.** `verify_circuit` reports on the whole file. To find
what changed relative to a copy the caller kept, diff the two finding lists,
which include `rule_id`, `at` and `subject`. Diff the findings
`api.verify_circuit` returns (through `run_code` or the Python API), which are
complete and never capped; capped lists do not diff reliably.

### 3.6 `inspect` — pure read

`inspect` takes `queries: list[Query]` and returns a result per item, with
cursors where listed:

```
{kind: "capabilities", fields?}
    simulators: for each, its executable and sha256; `version`, the build
    the latest run on that executable reported, with `version_source`
    naming the job and case (null until a run; the executable is never
    launched to ask); `selectable`, whether run_experiments'
    execution.simulator may name it, with `refusal` when this host cannot
    run the family (detected or not), and for an undetected one the
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
    make the PNG it inlines and, if not, whether the `raster` extra is
    missing or only cairosvg is, without native Cairo). `fields` (e.g.
    ["allowed_paths", "config_path"], to check after a config edit) returns
    only those top-level keys; without it the whole report is returned
{kind: "symbols", path?, filter?, cursor?}
    legal symbol names and resolution order; `path` adds schematic-local
    directories to the reported precedence
{kind: "symbol", name, path?}
    pins per rotation, bbox, origin
{kind: "net", path, at: "REF.PIN" | "net:NAME" | "NAME" | [x, y], cursor?}
    .asc gives a geometric trace; an [x, y] on a wire's interior traces that
    wire, reported as snapped_to_wire; a netlist gives the membership of the
    deck's own cards (not those in a .subckt body) and no geometry, with node
    names compared without case and `gnd` read as node 0, and a warning
    saying ngspice's kiltpsa mode may not read `gnd` so. On a .asc, PIN is a
    pin name or, failing that,
    the pin's 1-based SpiceOrder; on a netlist it is a 1-based terminal number.
    A bare NAME is read as net:NAME on both: a net label on a .asc, a node
    name on a netlist
{kind: "components", path, prefix?, detail: "list"|"full", cursor?}
    `prefix` keeps references that start with it, case-insensitively, on
    both a .asc and a netlist: `"M"` for every MOSFET, `"LX"` for LX1, LX2
    but not L1. It is plain text; a wildcard is refused with the plain
    spelling to use. On a netlist, `detail: "full"` adds each card's
    `nodes`, `model` and `params`, keyed as written (a key given twice in
    one spelling reads as its last copy), and `repeated_params` for a
    parameter the card gives more than once, compared without case: every
    copy as `key=value`, in card order. Which copy a simulator reads is not
    established
{kind: "hierarchy", path, simulator: "ltspice"|"ngspice"|"ltspice:<name>", ngbehavior?, instance?, prefix?, cursor?}
    expanded netlist instances; `instance` is an exact reference-segment list
    selecting a subtree, and `prefix` is the `components` rule applied to
    each instance's own reference (the last segment), so a one-letter
    prefix still selects an element type. LTspice names are spelled for the
    session's LTspice, or the named build, as its raw names them
    (`profile.xvii_names`)
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
    recorded values, range and AT points by value_index (not by step).
    native_tables pages the value blocks ngspice prints in the log, as
    `name = value` entries and `Index frequency` rows, each with its
    block's first and last line (`line_start`, `line_end`), what ended the
    block (`closed_by`) and its printed labels; a unit the log does not
    print is null. Neither adds plot or step identifiers or inferred
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
{kind: "guide", section?}
    the packaged guide: no `section` returns the core a
    session reads first, ending in an index of the topic sections and task
    playbooks; a `section` from that index returns that section. An unknown name
    fails the item as `unknown_section`, listing the names that exist
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
{kind: "open_in_ltspice"}
    the documents open in the LTspice windows on this machine: `windows`
    (how many are running), `total`, and `designs`, each {path, kind:
    "schematic"|"netlist"|"other", active (the one in front in its window),
    pid, version, in_sandbox}. A sheet the server may read adds its `sha256`
    and `differs_from_file`. When the window's copy differs from the file,
    `difference` names a few entries in a sentence, and `only_in_window` and
    `only_in_file` list the differing entries (a symbol with its
    attributes, or one other line), up to 25 a side, with
    `difference_omitted` counting the rest. Fails as
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
resource is recorded on the session; if a session has not read the guide, its
first tool reply contains one reminder to read it, in the text and in the
structured `hint`.

---

## 4. Cross-cutting contracts

**Annotations.**

| tool | readOnly | destructive | idempotent | openWorld |
|-|-|-|-|-|
| `run_experiments` | false | false | false (without a repeated `request_id` the same arguments start new work; with one, the original job is replayed) | true |
| `jobs` | false | true (cancel) | true | false |
| `analyze_results` | false (artifact writes) | false | true | false |
| `edit_schematic` | false | true | false | false |
| `verify_circuit` | false (render, export) | true (export_to: sidecar, the default) | true | false |
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

**Server logging.** The server logs only to stderr, at `WARNING` by default;
`[logging] level = "INFO"` or `LTSPICE_MCP_LOG_LEVEL` sets another level. It
neither advertises nor serves the MCP logging capability
(`logging/setLevel`, the `logging` capability and `notifications/message`),
and `logging/setLevel` returns method-not-found. Everything a caller needs is
in the tool result.

---

## 5. Out of scope

The six tools do not edit deck text, offer temperature or lib-section sweep
parameters, autoroute or tidy a drawing, rate results as trustworthy, keep
library session state, provide per-wire interactive tools, write partially to
a named target, or return unbounded collections. They do not validate
testbench physics. The linter checks SPICE and does not check the measurement
setup.

Circuits are passed as file paths; netlist text cannot be pasted into a tool
call. The surface assumes an agent on the same machine as the server.

---

## 6. The linter

Each lint rule has a `rule_id` and a disposition, `blocking` or `warning`,
which its findings report as `severity` (`error` or `warning`).
Only rules for deterministic failures recorded from real simulator runs block.
Suppression is per call, and `linter_version` is reported in provenance.

| rule | disposition | condition |
|-|-|-|
| `save-meas-coverage` | blocking | the deck has a `.save` that is not `all` or `*`, and a `.meas` reads a signal no `.save` names |
| `meas-ngspice-batch` | warning | ngspice: the deck runs and only the top-level `.meas` is skipped, which the run reports when it is read |
| `meas-trig-degrees` | blocking | LTspice: a `.meas` that calls `sin`, `cos`, `tan`, `asin`, `acos`, `atan` or `atan2`, whose angle LTspice 26 and XVII read in degrees on their default settings where a B source reads radians; `verify_circuit`'s `syntax` check reports the same as `meas_trig_degrees` |
| `lib-section-ngspice` | blocking | ngspice in `kiltpsa` mode |
| `analysis-count-ltspice` | blocking | refused by LTspice 26 and XVII: two of `.tran`/`.ac`/`.dc`/`.noise` (`.op` may appear with one) |
| `meas-function-ltspice` | blocking | refused by LTspice 26 and XVII: `vdb()`, `phase()` or `group_delay()` in a `.meas` |
| `meas-name-ltspice` | blocking | refused by LTspice 26 and XVII: a `.meas` named `e`, `k`, `pi` or `q`, constants of the expression engine |
| `lib-section-ltspice` | blocking | refused by LTspice 26 and XVII: `.lib file section`, which LTspice reads as one file name |
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
| `temp-as-param` | blocking | a deck that sweeps temperature as a parameter (§3.1) |
| `duplicate-param` | warning | an instance, `.model` or `.subckt` card giving one parameter more than once, in any case; which copy the simulator reads is not known, and `verify_circuit` reports the same as `duplicate_parameter` |
| `value-suffix-mojibake` | blocking | a suffix showing the file was decoded in an encoding it was not written in, such as the `Âµ` a UTF-8 `µ` becomes under cp1252; the simulator reads the bare number, off by a factor of 1e6 |
| `value-suffix-nonascii` | warning | any other symbol after a number, such as `10Ω` or `25°C`, read as the bare number, which is usually the intended value. Staging spells a `µ`/`μ` as `u` before the deck is linted; for a deck run elsewhere `verify_circuit` reports it as one observation per file, or as a warning per value when a reader the server knows of (LTspice XVII) would decode the file differently |

`element-arity`, `bsource-value-prefix`, `value-keyword-ltspice` and
`value-expression-remnant` come from one check of each element line, and each
is a separate rule so that suppressing one never silences another. Rules
that read elements or values skip line 1 of the deck, the title line both
simulators skip.

---

## 7. Appendix A — payload grammars

The variation and recipe unions are strict discriminated unions, inlined into
each tool's `inputSchema`.

### A.1 Variation

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
    "INSTANCE:{delvto|mulu0}"            -> explicit per-instance mismatch
       value; "X1:delvto" for a single-FET body, or the qualified
       "X1.M0:delvto" for a multi-FET body. Supported only on
       ngspice-compatible BSIM3/4 devices through exactly one X -> M wrapper
       level as this convenience spelling. Use an ordered `instances` selector
       for arbitrary supported depth; never construct a selector by joining
       segments with punctuation.
  Otherwise, a declared .param name -> parameter substitution;
  else a component reference        -> value substitution;
  else                              -> ambiguous_target.

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

{kind: "random", id?, runs: int >= 1, seed?: int, applies_to?: [circuit id],
 rules: [RandomRule]}          at most ONE random entry per circuit

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
current-factor scale applied to the parameters the rule names. A
`rule: "mismatch"` can also select a MOS inside subcircuits by its exact
instance list, or every MOS inside a subcircuit instance by that instance's
list. Such a device gets its draw as `delvto` and `mulu0` instance
parameters, which needs its model to resolve to an ngspice-compatible BSIM3
or BSIM4 model, and the draw uses the device's W and L after `assign`
variations are applied. Draws across circuits are independent; correlated
Monte Carlo across decks is not supported.

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
| `summary` | any | — | full summary payload: sim type, ranges, signals, measurements, Fourier, AC bandwidth, diagnostics. Fourier is each `.four` table as the build printed it, phase included; LTspice 26 and XVII print phase in different conventions |
| `measurements` | any | — | `names?`, `histogram_bins?` (0 = none); per run, `stats` for each `.meas` name. A run with one value per name also has `measured` {name: value}, `at` and `failed_measurements`, and its default row shows only those (§3.3) |
| `value` | any | `expr` — one trace or node pair, not arithmetic | `at`, required when the axis has more than one sample; reads the nearest sample, no interpolation; a bias-point run is read by name; step-aware |
| `signal_stats` | tran | `signal` | `window?`, `quantiles?` — levels in [0, 1], weighted by time, not by sample; each adds `q<percent>` (0.99 is `q99`, 0.999 is `q99_9`) and two or more add `quantile_peak_to_peak`, all reducible. `min`/`max`/`peak_to_peak` stay the sample extremes |
| `edges` | tran | `signal` | `levels?`, `edge?`, `window?` |
| `timing` | tran | `from{signal, edge, level}`, `to{...}` | `nth?`, `window?` |
| `periodic` | tran | `signal` | `window?`; period, frequency, duty cycle |
| `transient_response` | tran | `signal`, `mode: "step"\|"disturbance"` | `input` is required for `disturbance` and rejected for `step`; `window?` |
| `thd` | tran | `signal` | `fundamental_hz?`, `harmonics?` (default 7), `window?` |
| `tone` | tran | `signal`, `frequency_hz` | `window?`; cut to the whole periods of `frequency_hz` that fit from the window's first sample and resampled by linear interpolation onto evenly spaced times, as `thd` does by default, so samples weigh by time. `amplitude` (peak, signal unit), `phase_deg` in (-180, 180] for `dc + amplitude*sin(2*pi*f*t + phase_deg)` with `t` the run's own time from zero, `sin_component`, `cos_component`, `dc`, plus `t_start`, `t_end`, `n_cycles`. Reducible leaves: `amplitude`, `phase_deg`, `dc`, `sin_component`, `cos_component`; a spread of `phase_deg` that straddles 180 degrees reduces the wrapped value, the components do not wrap |
| `bode_filter` | ac | `signal` | filter characteristics: fc, bw, Q, type |
| `bode_point` | ac | `signal`, `at_hz` | gain and phase at a frequency |
| `bode_crossing` | ac | `signal`, exactly one of `level_db` / `level_deg` | `level_deg` scans the unwrapped phase, so a crossing past 180 degrees is found once rather than at every wrap; `phase_deg` remains accepted as an alias |
| `bode_slope` | ac | `signal`, `from_hz`, `to_hz` | dB per decade |
| `stability` | ac | `signal` | phase margin = 180 degrees + phase at unity gain, positive being stable. Reducible leaves: `phase_margin_deg`, `gain_margin_db`, `unity_gain_hz`, `dc_gain_db` |
| `ac_structure` | ac | `signal` | structural poles and zeros |
| `resonance` | ac | `signal` | resonant frequency, Q, peak |
| `return_loss` | ac | `signal` | `z0?` (default 50); return loss in dB, VSWR, reflection coefficient |
| `noise_integral` | noise | — | `signal?`, `from_hz?`, `to_hz?` |
| `operating_point` | op | — | `device?` scopes the reply to one device, named by its path (`x1.m1`) on either LTspice build; `device_names` maps each device inside a subcircuit to its name in the log's table and in the raw |
| `waveform` | any | `signals` | `max_points?` (default 2000), `format: "inline"\|"csv"`, `window?`. Inline is bounded decimation only, with `points_returned` / `points_total` declared; `csv` returns an artifact handle |
| `plot` | any | `signals` | `title?`, `log_x?`, `span?` — returns an artifact handle |

A `signal`, `signals` entry or `expr` names one trace as the raw stores it, or
a node-pair voltage `V(a,b)`. No simulator writes a pair as a trace, so every
reader computes `V(a) - V(b)` from the same raw, step and axis, complex on an
AC run. `V(a,0)` and `V(a,gnd)` are `V(a)`, a missing node is named, and a
`.noise` run refuses a pair because spectral densities do not subtract. The AC
recipes also accept a ratio `A/B` of two such signals and a leading `-`. Other
trace math is done in Python on `RawResult.trace`. A not-found error for an
input that looks like an expression includes the Python snippet and names
where to run it, which is `run_code` when the session serves it and the
library otherwise.

### A.3 Edit ops

The thirteen op kinds, discriminated on `op`, are `add_component`,
`move_component`, `move`, `remove_component`, `set_component_value`,
`set_component_attribute`, `add_net_label`, `remove_net_label`, `wire_pins`,
`remove_wire`, `add_directive`, `remove_directive`, `set_plot_panes`. The batch
is atomic, so the first op that fails aborts the transaction and nothing is
written.

`move` shifts any mix of records by one offset, `dx` and `dy`. It selects
parts by `references`, net labels and grounds by point (`labels`), wires by
their two ends (`wires`), and directives and comments by exact text or
`regex:` (`texts`). A `region` also selects what lies inside it, edges
included, testing a part by its box, a wire by its two ends, a label by its
point and a text by its anchor. With `drag`, an end of any other wire that
lies on what moves (a moved pin, label or wire end, a point along a wire that
moves whole, or with `region` any end inside it) moves too, so the wire stays
attached, and so does a label or ground on any of those points; a wire
dragged off its axis is reported as diagonal. A part's attribute text moves
with it. Something named that is not there is refused, and so is a move that
would join two nets that each have a name (names compared as LTspice compares
them, in any case), before anything changes. The result's `moved` counts what
moved and lists the parts. Applying the opposite offset to what it moved puts
everything back. `move_component` places one part at a position and can turn
it.

`remove_wire` removes an exact segment. `remove_component` takes
`cleanup_wires` to also delete the wires its removal leaves dangling.
`add_directive` takes `kind`, `directive` (simulated) or `comment` (ignored by
LTspice). Removals match text literally by default; a `regex:` prefix matches
a pattern instead.

`set_component_value` splits a trailing `KEY=VALUE` off the value it is given
and sets it wherever the part already gives that key (`Value`, `Value2`,
`SpiceLine` or `SpiceLine2`, matched case-insensitively), in every copy found.
A key the part does not give is added to `SpiceLine`. A part that gives a key
more than once gets the sheet's `duplicate_parameter` finding, which the reply
lists because the op names the part. A behavioural source's value is one
expression and replaces `Value` entirely.

### A.4 Inspect queries

§3.6 lists every query kind.

---

## 8. Known gaps

- The one native statistical profile is the pinned Sky130 NMOS profile below;
  other device families and multiplicity or finger conventions are not
  supported.
- No correlated Monte Carlo draws across two decks.
- No loop-gain probe templates.
- An attached analysis stage interrupted by a server restart is not re-run;
  run `analyze_results` over the job instead.
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
unresolved value is null, never a guessed zero. Discovery evaluates only
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
supported modes are `hsa`, `kiltpsa` and the empty string (native SPICE
mode); others are refused. A `.lib file section` line needs a mode without
`lt` or `ps` in its name (`hsa` or native mode), because those modes read it
as two plain includes. The deck's first line is its title and is skipped; an
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
pair; and a subcircuit, model or reference defined twice among the cards in
use fail the query. A binned model (`nch.1`, `nch.2`, ...) is reported with
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
for the device table on every `.op` run and XVII prints it unasked and stops
on `.options logopinfo`. Add any directive given to the deck before
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
