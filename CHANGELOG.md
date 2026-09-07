# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [0.6.0] - 2026-09-07

One engine behind two interfaces. The MCP surface is eight tools in place of
the 49-tool and 41-tool profiles: six operations arranged over execute,
understand and author, the waveform widget, and a Python snippet runner. The
same six operations are importable in-process as `ltspice_mcp.api.Api`, with
complete results where the wire pages. Every job is one durable experiment
across declared variations; every result is a typed recipe; every schematic
edit is a transactional batch. This release breaks the 0.5 tool names; the
first section says where each one went.

### Breaking: one tool surface replaces the `full` and `agentic` profiles

The `full` (49-tool) and `agentic` (41-tool) MCP tool profiles are gone. The
server now advertises eight tools:

|Tool|What it does|
|-|-|
|`run_experiments`|Stage decks, expand declared variations (sweeps, corners, Monte Carlo), and run them as one durable job|
|`jobs`|Job lifecycle by `job_id` or `request_id`: `status`, `wait` (long-poll), `cancel`, `runs` (per-run pages), and `list` (recent circuits and their job counts)|
|`analyze_results`|Typed measurement recipes over finished runs and `.raw` files, with reductions, group splits, and spec verdicts|
|`inspect`|Read-only lookups: `capabilities`, `symbols`, `symbol`, `net`, `components`, `model`, `reference`|
|`edit_schematic`|A transactional, revision-guarded batch of typed ops on one `.asc` sheet|
|`verify_circuit`|Lint, symbol and pin resolution, netlist export, layout and quality checks, comparison against a reference, optional rendering|
|`plot_waveform`|An interactive HTML chart of a run, for a person to look at|
|`run_code`|A Python snippet run in a worker process that holds the engine as `api`, for loops over runs and numpy on samples; on by default, `[tools] run_code = false` removes it|

The same six operations are importable in-process as
`from ltspice_mcp.api import Api` — one engine, two interfaces.

**How to keep the old tools.** Pin `ltspice-mcp==0.5.*`.

**What happens if you don't change your config.** `[tools] profile` is not a
recognized key any more. A config that still sets it — to `"full"`,
`"agentic"`, or anything else — loads with the key ignored, like any other
key the server does not read, and the server starts normally with the eight
tools.

**Where each old tool went.**

|Old tools|Now|
|-|-|
|`run_simulation`, `configure_sweep`, `run_sweep`, `configure_montecarlo`, `run_montecarlo`|`run_experiments`|
|`check_job`, `cancel_job`, `batch_results`, `recent`|`jobs`|
|`simulation_summary`, `measurement_stats`, `query_value`, `signal_stats`, `get_waveform`, `export_waveform`, `edge_metrics`, `timing_between`, `periodic_metrics`, `transient_response`, `thd`, `bode_metrics`, `stability_metrics`, `resonance`, `return_loss`, `ac_structure`, `noise_integral`, `operating_point`|`analyze_results` recipes, one per `metric`|
|`server_status`, `symbol_info`, `component_info`, `trace_net`, `list_components`, `find_model`|`inspect` kinds|
|`create_schematic`, `apply_schematic_ops`, `wire_pins`|`edit_schematic` (a blank sheet is `base: "blank"`; wiring is the `wire_pins` op)|
|`export_netlist`, `validate_netlist`, `diff_circuit`|`verify_circuit` checks|

CSV waveform export is the `waveform` recipe with `format: "csv"`, which writes
every sample to a file and returns its path.

### Added

- The Python API. `from ltspice_mcp.api import Api` boots the same engine
  in-process on a working directory and exposes the six operations as
  synchronous methods with complete results where the wire pages or caps,
  plus `load_raw` and `measurements` for numpy access to a run and the AC and
  transient metric functions under their existing names. It reads and writes
  the same job records as a server in the same directory, so the two can run
  side by side; a job belongs to the process that submitted it and `close()`
  cancels what it owns, exactly like server shutdown. `__all__` is the
  stability boundary and is pinned. The full contract is
  `docs/design/python_api.md`.
- `run_experiments` runs one experiment across declared variations — strict
  assignments plus one random or Monte Carlo dimension — as a single durable
  job. `request_id` is optional: omit it and a fresh id is generated and echoed
  on the receipt; pass your own to make submission idempotent, so a retry with
  the same id, the same arguments, and unchanged source decks replays the
  existing receipt instead of running anything again.
- A `run_experiments` call dwells up to `execution.wait_s` (default 60 s,
  maximum 120 s) and returns the results inline when the job finishes in that
  time. Otherwise it returns a receipt with a `job_id` and the job keeps
  running; follow it with `jobs(action="wait")`. `wait_s: 0` returns
  immediately.
- `run_experiments` takes a per-run simulator (`execution.simulator`:
  `"ltspice"` or `"ngspice"`), so one deck can be cross-checked on a second
  engine without changing config. Runners are cached per (kind, simulator,
  output folder), so a second engine's runner does not evict the first's
  in-flight concurrency and cancel state, and a cancel resolves the runner by
  the job's own recorded simulator. Result reads follow the job too: the raw
  dialect comes from the simulator the job actually ran on, not the session
  default, so an ngspice run under an LTspice default parses correctly and the
  other way round.
- `Api.run_experiments(wait=False, detach=True)` hands the job to a small
  owner process that outlives the caller: the caller validates the request,
  the owner submits, supervises the job to a terminal status, and exits. The
  caller's `close()` leaves a detached job alone; `jobs(action="cancel")`
  from any process stops it through the existing foreign-owner path; a
  running server sees it as another session's live job. The receipt names
  the owner's pid and log file. Requires `[state] persist_jobs`.

- A `run_code` tool, on by default (`[tools] run_code`,
  `LTSPICE_MCP_RUN_CODE`): it runs a Python snippet in a warm worker process
  that holds the engine as `api` (the same six ops as methods, complete
  results, plus `np`, `load_raw`, `measurements`, `reference`), for loops over
  runs and numpy on samples. Each call is a fresh namespace around the same
  live engine; `timeout_s` (default 60, max 600) interrupts the snippet and
  cancels a run it waits on; a second call during one is answered `busy`;
  `reset` replaces the worker. Output is capped and says what it dropped. The
  snippet runs with the server process's own file and process authority, not
  the sandbox: permission it in the client the way you would a shell, and set
  `run_code = false` when the server is reachable by more than one trusted
  client, for example through a proxy. The capabilities report names the key
  (`python_api.run_code`), the reference table lists the tool only on a
  session that serves it, and the registration itself declares the gate, so
  the served surface, the reference table, the capabilities entry and the
  instructions all derive from that one declaration.
- `[tools] listing` (env `LTSPICE_MCP_TOOL_LISTING`) selects how much of each
  tool definition the tool list carries. `compact`, the default, advertises
  the same tools and the same schemas with every per-argument description
  removed: structure, enums, defaults, `required` and `$defs` are intact, so a
  client can still build a valid call, and the server validates and answers
  exactly as before. It takes roughly 45% off what a session loads before it
  can call anything; descriptions are read on demand through
  `inspect(kind: "reference")` and carried on every validation error. Measured
  on the same day against `full`: the same answers at the same design
  quality, one listing-caused mismatch in 22 requests. `full` puts every
  description back on the wire. Both listings are static.
- `inspect(kind="reference")`: a searchable lookup over the tools' own
  vocabulary — each tool's top-level arguments, plus the branches (analysis
  recipes, schematic ops, variation kinds, query kinds, checks and job
  actions). A plain-words `query` ("phase margin", "connect two pins") returns
  the closest entries with their fields, types, defaults, bounds and units, and
  an argument name ("all_steps", "expected_sha256") reaches its tool's own
  table; with no `query` it returns the table of contents. `limit` defaults to
  5 and caps at 20. The index is built from the same input models the wire
  validates against, so nothing callable can be missing from it. It is the
  route to any argument's meaning on the `compact` tool listing, where
  per-argument descriptions are not published.
- `inspect(kind: "capabilities")` reports the Python API under `python_api`:
  the import line, the session call on this working directory, and where an
  op's arguments are read (`api.reference`, `help`, `inspect.signature`).
- `inspect(kind="capabilities")` reports `diagnostics`: the startup notes (a
  bad configured simulator path, a requested engine that fell back, WSL
  auto-detection) that say whether the server started degraded. They were
  written only to the server's own log.

- `inspect(kind: "capabilities")` reports the effective ngspice compatibility
  mode (`ngbehavior`), so a command-line ngspice run that fails differently can
  be compared with the server's setting.
- `inspect(kind: "capabilities")` also returns `config_path` and lists every
  known-but-undetected simulator with a remediation naming the exact config key
  (`simulator.path`), the environment variable, the config file, a
  platform-appropriate example executable path, and the restart requirement. The
  text is built from the same constants the config loader reads, so it cannot
  name a key that does not exist. When a non-empty `simulator.enabled` allowlist
  is the reason an engine is off, the remediation says that instead of pointing
  at an install. The WSL no-simulator error now names `simulator.path` alongside
  the environment variable.
- `inspect(kind: "capabilities")` gains a `python` block — executable, install
  kind, whether the interpreter path is ephemeral, package location — so a
  caller can check that environment before switching to the in-process API. The
  server instructions carry a one-line pointer
  (`from ltspice_mcp.api import Api`), and a finished `run_experiments` receipt
  that expanded ten or more cases points at the in-process API in its `hint`,
  because per-call overhead adds up over a loop of many cases.
- `verify_circuit`'s render block carries `source_sha256`, the digest of the
  sheet it drew. Rendering reads the file on disk, so a peer that committed
  between `edit_schematic` returning and this call running was invisible:
  the block's `sha256` is the image's and the export block's is the netlist's.
  Compare `source_sha256` with the `sha256` `edit_schematic` returned to know
  the picture is of the revision you wrote.
- `verify_circuit`'s `quality` check runs on a `.cir`/`.net`/`.sp` netlist,
  not only on a schematic. Three connectivity rules that had been written but
  never called report as findings: a node wired to a single element terminal
  (`dangling_node`, an observation — a bias fragment or a test stub leaves
  nodes open on purpose), a `V(...)`/`I(...)` reference in a directive naming
  something no element declares (`undefined_reference`, a warning — the
  unlabelled net that exports as `N00x` while the `.meas` still asks for
  `V(vref)` and measures nothing), and a net with two or more terminals that
  reaches ground through no DC-conductive element (`floating_net`, a warning —
  its operating point is undefined). The default check set for a netlist is
  `syntax` and `quality`; pass `checks` to narrow it.
- A run that fails because the simulator could not open an `.include` or
  `.lib` reports the failure code `missing_include` instead of the generic
  `execution_failed`, with the file it could not find as evidence. When that
  deck selects a `.lib` section and the run was ngspice in an LTspice or
  PSPICE compatibility `ngbehavior`, the code is `ngspice_lib_section` and the
  receipt's hint names the fix: `[simulator] ngbehavior = "hsa"` (or
  `LTSPICE_MCP_NGBEHAVIOR=hsa`) and restart, or `set ngbehavior=hsa` in a
  `.spiceinit` in the run directory. Those modes read `.lib <file> <section>`
  — the standard PDK corner idiom — as two plain includes and drop the
  section, so the corner select came back as a missing file with nothing
  pointing at the cause.

- Every tool carries a display title, the short label a client shows a person
  in place of the wire name (Run Simulations, Analyze Results, Edit Schematic,
  and so on).
- The tool, resource, template and prompt listings tell a client how long they
  stay fresh (`ttlMs` / `cacheScope`: one hour, private). They are built once
  at startup and cannot change while the process runs, so a client no longer
  has to re-list them every turn. Clients on earlier revisions are unaffected.
- `server/discover` answers with instructions naming the simulators actually
  detected, the same as the initialize handshake.

- A validation error that names a branch (a recipe, an op, a check, a query
  kind, a job action) now ends with that branch's field table, so a caller
  corrects the call from the error instead of looking the branch up first.
  Served on every path: the tool call, the Python API, and the attached
  analysis of `run_experiments`.
- The rule "`field` is required once `reduce` or `spec` is given" is stated in
  the recipe schemas themselves (`dependentRequired`), so a client on the
  compact listing, which carries no descriptions, still sees it.
- A random rule's `tolerance`, `scale` and `distribution` say what they mean
  on the field itself: the tolerance is a fraction of the nominal (or in the
  value's units), and for a normal draw it is the 3-sigma bound. The
  convention was stated only in a design document; an agent asked for a
  Monte Carlo at a stated sigma read the package source to find it.
- The Python API drops the two MCP presentation controls, `budget` and
  `execution.wait_s`, and says so in the result's `warnings`, instead of
  refusing the call. Neither is part of a request's identity, so an MCP call
  replayed through the API with them attached is the same request. Paging
  controls are still refused, with `raw_page=True` named as the remedy.
- `.asc` schematics now run on ngspice. The LTspice netlist export is converted
  for ngspice by removing the LTspice-only `.backanno` command (ngspice aborts
  on it) and translating `µ` suffixes and `§` name prefixes. The converted
  netlist is written to a separate `{name}.ngspice.net` sidecar instead of
  replacing the shared `.net`, so concurrent LTspice and ngspice runs of one
  schematic cannot pick up a netlist generated for the other simulator.
- A clean simulator exit that wrote results only to the log (for example an
  ngspice `.control` script) now completes with the log-parsed measurements
  instead of reporting "Simulation failed (no output generated)". A run with no
  raw file whose log carries errors still fails. When the deck asked for an
  analysis that must produce a `.raw` and none appeared, the run fails with a
  `missing_required_raw` observation, a log excerpt, and the known `.save`-list
  workaround.
- Timed-out runs now include diagnostics. The timeout response derives the
  run's log path (the completion callback does not record it for a job that is
  already terminal) and shows the end of the log. Cleanup of a killed run keeps
  the `.log`, `.exe.log`, and `.fail` files and removes the possibly
  multi-gigabyte `.raw` and netlist copies. A killed run exits with a nonzero
  status, so spicelib renames its log to `.fail`; the job record is updated with
  the renamed path once the process exit is observed, so the excerpt stays
  readable.
- Windowed transient time axes are rebased to deck time. LTspice stores
  `.tran 0 <tstop> <tstart>` output with the axis starting at 0 and the true
  start only in the raw header's `Offset:` field, which nothing applied — a
  196–202 µs window read as 0–6 µs everywhere. All raw loads now add the offset
  back, so window arguments and reported times are in deck coordinates.
- The `transient_response` recipe takes `mode: "step" | "disturbance"`.
  `"step"` is the classic pulse response. `"disturbance"` is for a regulated
  output under a load or line step (LDO, PMIC), where the output returns to its
  own level and so the step mode correctly reports null metrics; it measures
  droop and overshoot against the pre-disturbance baseline (explicit, or taken
  automatically from the leading window) and the recovery time back into a
  settle band. It reports null recovery, with the reason in `warnings`, when the
  output never re-enters the band or the band is undefined (zero baseline, no
  absolute band).
- The `return_loss` recipe reports reflection metrics — Γ magnitude and phase,
  return loss in dB, VSWR — from an AC impedance trace measured under the
  documented 1 A probe, against a reference `z0` (default 50 Ω). It evaluates a
  given frequency or scans for the worst match across the sweep, flags a
  reversed probe (negative-real Zin), and reports null return loss and VSWR at
  the perfect-match and total-reflection limits. It also reports the |Zin|
  extrema across the sweep (`zin_min_mag_ohm`, `zin_max_mag_ohm`, and their
  frequencies) and, when the worst-match scan lands on a nearly purely reactive
  point (|Γ| ≈ 1 for any real z0), says so and points at a meaningful z0 choice
  for power and filter ports, where the 50 Ω RF default is rarely the right
  reference.
- The `timing` recipe aggregates over all sequential edge pairs — `pair_count`,
  `delay_min`, `delay_max`, `delay_mean`, and the times of the extremes — for
  dead-time and minimum-off audits across a pulse train. Previously only the
  first crossing pair was measured.
- The `summary` recipe reports the run temperature (`temp_c`) and nominal
  temperature (`tnom_c`) parsed from the simulator log, so tempco, noise, and
  leakage tasks don't have to assume 27 °C. It also declares `ac_signal_used`,
  and it declares, attaches, and renders `suggestions` (model-resolution
  candidates for unresolved references) on completed runs as well as on
  failures — the success path previously computed them and dropped them.
- The `signal_stats` recipe reports `t_at_min` and `t_at_max`, the time of the
  minimum and maximum sample, so a transient droop or peak can be located in
  time from one call. (Named `t_at_*` rather than `t_min`/`t_max`, which would
  read as window bounds next to `t_start_used` and `t_end_used`.) It also emits
  a `constant_window` observation when a signal is constant across the whole
  analyzed window, min equal to max — for example a latched or degenerate DC
  solution that reads as a flat line. The observation states the fact only; it
  does not judge the result.
- A run summary flags a voltage trace whose peak exceeds every independent
  voltage source in the deck, supply rails included, by a large ratio. This
  identifies a moderate divergence, for example an undamped LC response that
  grows to hundreds of volts from a millivolt-scale drive but stays below the
  absolute extreme-value threshold. The observation reports the source name and
  amplitude and does not rate the result. It applies only to `.tran` and `.op`,
  where comparing a node voltage with a source amplitude is meaningful.
- `meas_batch_abort` observation: when one `.meas` directive fails to parse,
  LTspice abandons the whole `.meas` batch, and even directives earlier in the
  deck come back missing. The summary now links the misses to the failing
  directive instead of listing N independent-looking gaps.
- The AC recipes accept a leading `-` on the signal expression (`-V(out)`, or
  `-V(vout)/V(vsense)`): the complex wave is negated, a 180° phase flip, so a
  loop gain probed through an inverting sense or a reversed impedance probe
  reads in its natural convention without a behavioral inverter node in the
  deck.
- Per-step AC entries carry `step_values`, the `.step` name=value point from the
  log, next to the bare step index. LTspice runs a `.step ... list` sorted
  ascending rather than in declared order, so index-only labeling could
  attribute curves to the wrong list positions.
- `resonance` peaks carry `magnitude_linear` alongside `magnitude_db` — |Z| in
  ohms under the 1 A impedance probe, or |H| for a transfer function — which
  distinguishes a dBΩ peak from a dB dip. The recipe already promised the field;
  it now returns it.
- Parallel-session coordination: independent server processes (for example
  several coding agents sharing one directory) no longer interfere with each
  other.
  - Circuit-file mutations and `.asc` exports take a cross-process file lock
    (sidecar `.ltspice-mcp/locks/`), so concurrent edits of the same file from
    two sessions serialize on the latest content instead of silently losing one
    session's edit. Pin and route geometry resolves inside the lock, so it
    reflects a peer's just-completed move; an export locks the sidecar `.net`
    it overwrites as well as the `.asc`. If the lock is still held after a 10 s
    wait, the call fails with a "locked by another ltspice-mcp process" error.
  - Job sidecars record the owning server's pid. A running job whose owner is
    still alive now loads in other sessions as `running` (previously it was
    mislabeled `interrupted`), refreshes from its sidecar when its status or
    results are read or listed, and is excluded from other sessions' shutdown
    cleanup.
  - `psutil` is now a direct dependency; it was already installed as a spicelib
    transitive.
- The `edit_schematic` `set_component_attribute` op with an empty value now
  clears the attribute, removing its SYMATTR line — the only representation of
  "no value" the `.asc` format can read back. InstName remains protected.
- Placed schematic directives no longer overlap. When two directives would use
  the same anchor, which is the common case when several are added without
  explicit coordinates, the second is moved. A `stacked_directive` advisory also
  reports exact-anchor coincidences that arrive by other paths, such as a
  hand-authored `.asc` or a move op.
- The bundled SPICE guide documents the LTspice-first noise-figure method:
  `NF_dB = 20*log10(V(onoise)/V(Rs))`, using the per-source noise contribution
  traces LTspice writes into `.noise` raw files, with no temperature constant or
  4kT term needed (checked against a reference divider at 3.0103 dB). The
  engine-neutral `4kT·Rs` form remains the fallback and the only method on
  ngspice.

### Changed

- The sandbox follows the config file while the server runs: `[security]
  allowed_paths` is re-read whenever `ltspice-mcp.toml` changes, so a refused
  path names the exact line to add and says it takes effect on the next call.
  Before, the only self-serve route the refusal offered was copying the file,
  because widening the sandbox needed a restart the agent cannot perform.
  The server instructions state the sandbox rule in one sentence.
- `edit_schematic`'s `compare` is `verify_circuit`'s: `mode`
  (`equivalence` or `structural_diff`), `anchors` and `rtol`, run by the same
  comparison engine, and `verification.comparison` carries the same payload
  (a `structural_diff` delta with the verdict derived from it). A compare that
  could not run reports `verification.compare_error` instead of raising.
- `include.fields` accepts a bare name as the number it names: a path without a
  dot that is not a row key reads under `value`, so `phase_margin_worst_deg`
  means `value.phase_margin_worst_deg`. A dotted path with an unknown root is
  still refused.

- The default sandbox is the working directory plus the Claude Code scratch
  directory (`<tempdir>/claude-<uid>`). Claude Code tells an agent to write
  throwaway files there, outside the working directory, so a deck authored
  there used to be refused and copied in first. The generated config documents
  the default in a comment and leaves `allowed_paths` unset; setting it
  replaces the default. Applies to the server and the Python API alike.

- `analyze_results` takes `step` and `all_steps` as call-level arguments instead
  of per-recipe ones, and `run_experiments`' attached `analyze` block takes the
  same two, so an attached measurement and a standalone one read the same
  `.step` iterations. A run's step axis belongs to the run, not to each
  measurement taken on it, so the choice is made once and every recipe in the
  call reads it. The selection travels in the stored result set, so a
  continuation replays it. Because the attached block now hashes two more keys,
  the request canonicalizer moves to version 4: a `request_id` stored under
  version 3 raises an idempotency conflict instead of replaying.
- A recipe names the number a reduction or a spec reads once, as `field`. It
  replaces `reduce_field` and `spec.field`, which said the same thing and had to
  agree; `spec` keeps `min`, `max` and `allow_incomplete`. A multi-field recipe
  requires `field` as soon as `reduce` or `spec` is given; a keyed recipe
  requires it for `spec`, and where it is given it narrows that recipe's
  reduction to the named key too; a scalar recipe takes none.
- The published JSON Schema no longer carries `"default": null` annotations or
  `discriminator.mapping` tables: `required` and each branch's own `const`
  already say both. Nothing a call may send changed, and together with the
  argument removals above the tool listing is smaller in both modes.

- Calling a tool name the server does not have now answers a JSON-RPC
  invalid-params error (-32602) naming the unknown tool and listing the eight
  that exist, instead of an error-flagged tool result. A lookup failure has no
  tool to attribute a result to, and this matches how an unknown resource URI
  is already answered.

- Moved to the MCP Python SDK 2, which serves protocol revision 2026-07-28
  alongside the older initialize handshake. Clients on either revision are
  served the same tools.
- A tool that rejects its arguments now says `Invalid arguments for <tool>: ...`
  where it said `Input validation error: ...`. The SDK stopped validating a call
  against the published schema, so the tool's own model reports it; both are
  generated from that model, so nothing is checked less strictly.
- Reading a resource URI the server does not serve now answers the JSON-RPC
  invalid-params code (-32602). The 2026-07-28 revision dropped the separate
  resource-not-found code earlier revisions used.

- `analyze_results`' description names every recipe with the plain words a
  caller searches for, so a host matching a request against tool descriptions
  can route "phase margin", "distortion" or "bias point" to this tool. The
  handshake instructions name the reference lookup.
- A job whose owner process has exited but has not yet been collected by its
  parent now reads as interrupted rather than running. Liveness used to ask
  only whether the pid existed, and a finished child keeps its pid until the
  process that started it collects it.

- Every tool's response is built from one envelope (`outcome`, `failures`,
  `observations`, `warnings`, `hint`) with one outcome rule, and a finding's
  location carries the same fields on every tool. The advertised schemas
  did not change; the internal modules behind the tools were split so that
  each tool is its own file.

- One on-disk store. Everything the server writes under a working directory's
  `.ltspice-mcp/` is laid out by one `Store`: `experiments/` (job records, a
  request index, a per-circuit index, cancellation markers), `runs/{job_id}/`
  (every artifact one job produced, with its staged decks), `results/`,
  `detached/`, `verify/` (with its `renders/` beneath it), `edit-exports/`,
  and `locks/`, stamped with a single `store_version`. The per-circuit pointer files that let a circuit's sidecar
  find working-directory jobs are gone; the store's own index does that.

- **The `.asc` edit engine moved into the core.** The `.asc` edit engine is now `ltspice_mcp.lib.schematic_ops`, and every
  name another module imports from it is public. It lived in the tool layer,
  which meant core modules importing back up into the tools package (a latent
  import cycle that depended on import order). Nothing about `edit_schematic`,
  `inspect`, or `verify_circuit` changes; the one visible difference is in the
  JSON Schema `edit_schematic` advertises, where the internal `$defs` keys for
  the op shapes lost their leading underscore (`_OpAddComponent` is now
  `OpAddComponent`). References resolve exactly as before.
- `jobs` publishes each action's own argument shape. `status`, `wait`,
  `cancel`, `list`, and `runs` are separate branches of one schema keyed on
  `action`, instead of nine optional fields policed after the fact by the
  server. Every call that was accepted before is still accepted, unchanged;
  a call that is not now names the legal actions, or the exact field that
  action does not take, before it is dispatched.
- `run_experiments`' `analyze.recipes` is validated against the same typed
  recipe grammar `analyze_results` takes, so a malformed recipe is refused
  before any deck is staged instead of after the simulation has run. The
  schema advertises the metric names and points at `analyze_results` for the
  fields each one takes, rather than carrying a second copy of the grammar
  that every client would download in every session.
- One `jobs` call reads its job once. The MCP response and the in-process
  API result are two renderings of the same evaluation, so they can no
  longer describe two different moments of the same job.

- Every error class declares a stable `code`. No code a client sees has
  changed; the full vocabulary (146 codes) is pinned by
  `tests/test_error_codes.py`, and renaming or removing one is a breaking
  change that will be listed here. See `docs/design/mcp_surface.md`, "Error
  codes".

- The `mcp` dependency no longer pulls the `cli` extra: six packages fewer
  (typer, rich, markdown-it-py, mdurl, shellingham, annotated-doc) for a
  server that never imported them.
- New documents: `THIRD_PARTY_NOTICES.md`, the tool-surface and Python API
  contracts under `docs/design/`, and the spicelib bug ledger, published as
  `docs/spicelib_bugs.md`.
- The advertised tool definitions are exactly what the source says. Every
  description declared on a tool or one of its arguments is served verbatim on
  the `full` listing, and the `compact` listing removes descriptions rather
  than rewording them, so reading the models tells you what a client is shown, and
  the one text also feeds `api.reference('...')` and the `spice://guide`
  resource. What keeps the listing small is that the descriptions themselves
  are short: each states the unit, the convention, the default, and how the
  field interacts with its siblings, and the fuller explanation lives in
  `docs/design/mcp_surface.md` or the guide with a pointer on the field. Output
  schemas are still not advertised — response shapes are learned from
  responses, and dropping them is what pays for the argument prose.
  `tests/test_consolidated_contracts.py` holds an upper bound per tool, so the
  listing cannot grow without someone raising a number.
- Three `analyze_results` recipes — `noise_integral`, `periodic`, and
  `return_loss` — advertise only their `metric` and a one-line documentation
  pointer, about 700 characters less schema per session. They remain fully
  callable with every field they had; the full argument trees are in
  `api.reference('analyze_results')` and the `spice://guide` resource.
- `import ltspice_mcp.api` and `Api()` boot lazily: the package `__init__` is a
  PEP 562 lazy table, `SessionState` builds its tool surface on first access,
  the API method layer resolves tool modules through deferred imports, and scipy
  is imported at its call sites. A cold `Api()` drops from about 1.5 s to about
  0.4 s and no longer imports scipy or the MCP SDK at all. Method docstrings
  install on the first catalogue read or operation call; until then `help()` on
  a method shows only its signature.
- The README now describes the MCP server and the Python library as two ways to
  use one engine, opens with a runnable Python-API sweep example next to the MCP
  quick start, and cites published studies for the main design choices. PyPI
  metadata and the Desktop-extension description now mention the Python library
  and one-call sweeps and Monte Carlo. The README also no longer claims all
  circuit editing is simulator-free: netlist (`.cir`/`.net`) editing needs no
  simulator, but `.asc` schematic editing needs LTspice's `.asy` symbol
  libraries.
- The `value` recipe returns an `exact_match` flag, `false` when the requested
  time, frequency, or sweep value snapped to a different sample. On a coarse
  sweep this matters: a `.dc temp` run that silently snaps 27 °C to 25 °C biases
  a tempco reading. The `at` field now documents that it addresses the run's
  primary sweep axis — time, frequency, or the `.dc` sweep variable — and
  `step_axis` documents that it is for stepped (`.step`) sweeps rather than a
  bare `.dc` or `.ac` primary axis; the no-step error now points at `at` instead
  of a bare "axis not found".
- The `set_component_value` op warns when a GUI opamp complexity label such as
  `Level.2` is written to a subcircuit's Value, which LTspice emits as a stray
  positional token and which produces a cryptic "sub-circuit name is not
  defined" error at netlist time. The warning now fires on any Value written to
  a symbol whose model is chosen via `SpiceModel` — `UniversalOpamp2`, for
  example — not only on the `Level.N` label, because any value there corrupts
  identically. A library part that carries its subckt name in Value, with no
  `SpiceModel`, is left alone.
- The `add_component` op rejects an unknown attribute name — a typo such as
  `Val` for `Value` — instead of silently dropping it at export time, matching
  `set_component_attribute`.
- The `thd` recipe labels its per-harmonic magnitudes with the signal's native
  `unit` (V or A); the ratios, percentages, and dB values stay dimensionless.
- An axis-backed recipe run against a stepped `.op` whose raw collapsed to step
  0 points its no-axis error at the `.dc` conversion, matching the hint the
  `summary` recipe already emits.
- The SPICE guide gained several sections and corrections: the two-directive
  `.meas` argmax idiom (finding the frequency or time *of* a maximum), the dBΩ
  reading of an impedance trace under the 1 A probe, how to export quantized and
  staircase waveforms, and a cross-link to the `return_loss` recipe.
  `ac_structure`'s `net_order` is documented as a real computed order — 0 or
  negative is possible — never a sentinel. It adds notes on the three- versus
  four-terminal transistor symbols (`nmos4`, `pnp4`, and the rest), the diode
  symbol's built-in default-`D` model collision, and the `AC <mag>` small-signal
  source syntax; the `.meas MAX` signed-trace case, where `MAX I(...)` on an
  always-negative current returns the least-negative sample rather than the peak
  magnitude, so wrap it in `abs()`; and how to steer bistable circuits —
  bandgaps, mirrors, latches — to the intended DC root when `.nodeset` alone will
  not. Its named-nets rule now matches the runtime guidance: repeating a
  same-name net label validly ties distant pins, because the netlist merges
  same-name labels into one net, and with duplicate labels a wire should target a
  component pin rather than `net:NAME`.
- The duplicate-label error names the actual net in its guidance instead of a
  canned `net='0'` / `M3.S` example.
- The schematic layout guidance in `spice://guide`, the LTspice skill, and the
  new-sheet checklist now recommends delegating placement and wiring to a
  separate agent when the harness supports one, because the build is mechanical,
  detailed work that tends to end up as net labels everywhere when done inline.
  It also asks for a verification step before that agent returns: the
  `verify_circuit` export must match the source netlist, and
  `inspect(kind: "net")` must report no multi-label shorts.
- Auto-generated and example config files no longer write a live
  `max_parallel = 4` key; it is now a comment documenting the real default,
  which is the number of CPU cores capped at 8. The written key silently pinned
  every auto-generated setup back to 4 concurrent simulations on multi-core
  hosts.
- The `format` parameter description states that `'text'` is the default and
  that structured content is identical for both formats.
- The `operating_point` recipe always includes `warnings`, as `[]` on a clean
  run, instead of omitting the key.
- The `circuit-mcp` and `ngspice-mcp` alias packages publish only after the test
  workflow passes — previously they were ungated — pin `ltspice-mcp` to exactly
  the co-released version at build time, and wait for that version to be
  installable from PyPI first, so a published alias can no longer resolve an
  older canonical package under a newer alias version.
- The README, changelog, security policy, packaging text, developer docs,
  skills, and packaged guide were rewritten in plain language: shorter
  sentences, no slogans or metaphors, internal shorthand replaced with what it
  refers to. Facts, code, and tested instructions are unchanged.

### Fixed

- The no-simulator message names the setting that points at a simulator
  executable (`LTSPICE_MCP_SIMULATOR_EXE`, `simulator.path`) on every
  platform. Only the WSL wording did, so on a native Linux, macOS or Windows
  host an agent read "install ngspice" with no route for the simulator it
  already had.
- A host with the `raster` extra installed but no native cairo library now
  degrades a PNG render to SVG with a note, the way a host without the extra
  does. cairocffi reports the missing library as an OSError from the import,
  which the optional-dependency guard let escape, so a render call failed
  outright and the raster tests failed at collection on any machine without
  libcairo.
- A value spelled with the micro sign (`20µ`, `4.7µF`) parses as micro. That
  is how LTspice's netlist exporter writes `u`, so an exported current source
  had no nominal a Monte Carlo `component` rule could perturb: every case
  failed before submission with `random_nominal_unavailable`. The Greek mu is
  accepted too.
- A compare reference written outside the sandbox is refused with the
  alternative named: the deck's text itself, passed as `compare.reference`.
  A client's scratch directory is usually outside `allowed_paths`, and both
  tools now resolve the reference the same way, so `edit_schematic` accepts
  literal netlist text as its description always said (it used to treat the
  text as a path).
- A recipe's `field` accepts the result row's own key. `stability` reports
  `phase_margin_worst_deg`, and passing that name back for a reduction or a
  spec was refused in favour of `phase_margin_deg`; either spelling now names
  the number (the one reducible name is carried downstream). A key two fields
  read, such as `transition_time` under `edges`, is still refused as ambiguous.
- `edit_schematic` accepts an empty `ops` list when `return_views` is set: the
  batch runs as a dry run and returns the requested pin table. The whole-sheet
  table is only reachable through this tool, and reading it used to need a
  throwaway op.

- Two `run_experiments` calls sharing a `request_id` and fired at the same
  time now stage one set of decks between them, not two. A submission claims
  its `request_id` before it stages: it takes that id's gate, looks the id up,
  and only stages if nothing is recorded under it. The second call waits on
  the gate, finds the record, and replays it — or is refused as an
  `idempotency_conflict` — without copying a deck set of its own. Both used to
  stage in full and the loser's `runs/{job_id}/` tree was deleted afterwards,
  so a crash in between left an unclaimed tree in the shared runs root.
- `edit_schematic` reports `outcome: "partial"` when a dry run's ops fail,
  when the post-commit comparison against `reference` does not match or
  cannot export, or when a requested render fails. It reported `complete`
  in all three cases while `verify_circuit` reported `partial` for the same
  conditions; every tool now decides its outcome by one rule, and a test
  scans every handler module for a hand-written outcome.
- `max_parallel_sims` is enforced per simulator runner, not per
  `run_experiments` call. Three concurrent calls in one server used to get
  three times the cap, and a request could raise its own share above it.

- An analysis failure is classified as a deadline by its exception type, not
  by whether its message contained "deadline" or "exceeded". A source path, a
  signal name, or a filesystem error carrying those words (a full disk
  reports "Disk quota exceeded") was reported as `analysis_deadline`, sending
  the caller to retry a fault that more time cannot fix.
- Opening a schematic whose symbol, sub-sheet, or model library is missing
  reports the missing dependency. The old check looked for ".asy" in the
  editor's message, so a hierarchical block whose sheet was gone reported
  "File not found" against the schematic that had opened fine.
- Cancelling a job distinguishes an unauthorized request from any other
  cancellation failure by type rather than by the words "not authorized".

- ngspice decks that drive their own analysis from a `.control` block now get
  a rawfile through `run_experiments`. The write injection was wired only into
  a path no registered tool reaches, so a scripted deck completed cleanly with
  nothing for any recipe to read. The case reports `control_write_injected`
  when the server supplied the write.
- A `.step` directive on ngspice is reported by the deck lint
  (`step-ngspice`): ngspice ignores the line in batch mode, so the deck ran
  once at its base value and nothing said the sweep had not happened. The
  lint rule set is versioned as `2`.
- `verify_circuit` and `inspect` relay the netlist lexer's own notes (unclosed
  `.SUBCKT`, unmatched `.ENDS`, stray continuation) instead of discarding
  them — into `observations` and the netlist payload's `warnings`.
- `edit_schematic` reports the warnings its ops raise. They were dropped
  entirely; identical advisories across a batch now arrive once, with how
  many ops they cover.
- `edit_schematic`'s refusal of an existing target submitted without
  `expected_sha256` carries that file's current sha256 (code
  `expected_sha256_required`), so the retry is one call.

- `plot_waveform` accepts a `run_experiments` job: pass its `job_id` with
  `run_index` (default 0), or the new `case_id`, and the chart is written next
  to the case's source circuit like any other job plot. It used to refuse the
  only job kind the consolidated tools produce, with an error that named an
  internal type. The same error, where it can still occur on the other
  job-addressed paths, now names the tools that do accept an experiment job.
- The `stability` recipe accepts `field` (for `reduce` and for `spec`) on
  `unity_gain_hz` and `dc_gain_db`, not just on the two margins. "Keep the
  unity-gain bandwidth above 2 MHz" is the most common stability spec after
  phase margin, and the recipe already reported the number per case, but asking
  it to reduce or check that number was refused. A spec on a field the loop
  never reaches is `indeterminate`.
- `verify_circuit`'s `render.delivery` now states on the wire what an inline
  image costs: about 3k tokens at the default scale, paid again on every later
  turn of the session. The bare enum gave no reason to prefer the default,
  `artifact`, which writes the file and returns its path.
- Cancelling a job, a simulation timeout, and shutdown cleanup now terminate the
  simulator process on every supported platform and simulator combination.
  Previously only WSL with LTspice worked: everywhere else the code called
  spicelib's `kill_all_spice`, which matches an empty process name in the pinned
  spicelib and therefore killed nothing, so a cancelled ngspice, Wine, or
  Windows-native run kept simulating to completion. The replacement kill
  requires both the simulator's executable name and the job id in its command
  line, so it cannot terminate a parallel session's simulators, which a
  name-global kill would have hit once it worked.
- Cancelling a batch when several batch runners were live, with decks running in
  different output folders, could signal the wrong runner instance: the
  in-flight processes were killed, but the owning runner's submission loop kept
  launching the batch's queued runs behind a `cancelled` status. Batch cancels,
  both explicit and at shutdown, now route to the runner instance that owns the
  job's cancel state.
- Simulator aborts no longer stall the event loop. The completion callback used
  to read and scan the entire log on the event-loop thread, so a stalled read —
  a huge abort log, a hung network or DrvFs mount — blocked every request in the
  server process until restart. All completion-path file I/O now runs on worker
  threads, and the log-excerpt reader caps its read to head and tail slices of
  oversized logs.
- Converting an `.asc` schematic to a runnable netlist no longer runs the
  LTspice export subprocess on the server's event loop; it is offloaded to a
  worker thread, so concurrent requests, including a cancel, stay responsive
  during the export. Exports of the same schematic are serialized by the
  per-`.asc` lock described under parallel-session coordination in Added:
  LTspice always writes the same sidecar `.net`, so concurrent exports could
  otherwise tear the output and run the wrong deck.
- WSL interop calls (`wslpath`, `cmd.exe` environment resolution) now carry a
  15-second timeout; a hung Windows-interop process previously stalled server
  startup or left a run request waiting forever.
- The `thd` per-harmonic `magnitude` is now the sinusoid amplitude in the
  signal's own units. It was the raw FFT bin, off by n_fft/2 — a 0.1 V harmonic
  read as about 819 "V" at n_fft=16384. The Hann path's ±2-bin lobe sum is
  calibrated back to amplitude as well; a bin-centered tone's lobe RSS is √1.5
  times the amplitude. THD percentages and dBc ratios were, and remain, correct.
- Setting a `.asc` behavioral source's expression — a `set_component_value` op
  with `V=...` — corrupted the schematic: the expression was routed to SpiceLine
  as a parameter while the old expression stayed in Value, netlisting two
  expressions on one B-line ("No such node") behind a success message. The whole
  value now replaces the Value slot for B references.
- Directives with embedded newlines corrupted the `.asc` TEXT record; they are
  now stored as LTspice's literal `\n` escapes, so multi-line directives
  round-trip.
- The error for setting the `Prefix` attribute no longer misdirects to
  SpiceLine. Prefix is a symbol (`.asy`) property, and the message now points at
  symbol-based placement.
- The `measurements` recipe reads each `.MEAS` directive's operator from the
  deck instead of inferring it from the results. A `FIND` probe whose value is
  constant across runs is no longer mistaken for a `WHEN` crossing, which
  previously made it aggregate over the probe axis instead of over the values. A
  deck-confirmed bare `WHEN` now aggregates crossing times even when every run
  crossed at the same instant, as in a deterministic batch. The recipe also
  warns when it aggregates a still-running or partial batch ("N of M runs —
  partial, not final").
- `transient_response(mode: "step")` no longer reports a definite
  `settling_time` when the signal entered the settle band only just before the
  window ends and the final value was derived automatically from that same short
  tail. That case cannot be told apart from a still-ringing waveform that is
  momentarily flat, such as a transmission-line staircase. When `final_value` is
  not given, the signal must now stay in the band for at least 1.5× the nominal
  trailing window, the final 10% of the analyzed time span. The requirement is
  based on elapsed time rather than sample count, so sparsely sampled settled
  tails are not falsely suppressed. Suppressed results carry the
  `settling_dwell_near_window_end` quality flag and render as "unknown"; passing
  an explicit `final_value` bypasses the check.
- Analysis window bounds a few ulps past the axis end — SI-suffix rounding, such
  as `t_end=1500u` against an axis ending at exactly 1.5 ms — are clamped
  instead of rejected with a self-contradicting "outside axis range" message.
- Repeated identical log diagnostics collapse to one entry with a repeat count.
  A PDK deck repeats "unknown model parameter" once per device instance, burying
  the fatal line. The failure-excerpt reader now also anchors on "File not
  found" and "already defined" lines, which previously fell outside the excerpt
  entirely.
- Temperature-swept runs (`.step temp`) no longer report "no temperature steps"
  on modern LTspice. The simulator log is now decoded with the same BOM, UTF-16,
  and cp1252 detection the netlist and library readers use, instead of the
  platform-default codec. A UTF-16 log (current LTspice) or a cp1252 degree byte
  previously garbled the `.step` and temperature lines so the parse found
  nothing. The same fix correctly decodes simulator diagnostics, including the
  character shown in a failed-run excerpt, and the `temp_c` and `tnom_c`
  passthrough on such logs.
- The `value` recipe labels a `.noise` spectral-density read as `V/√Hz` (or
  `A/√Hz`), not the plain `V` its trace type declares, matching `noise_integral`
  and the raw file's own "Noise Spectral Density" plotname.
- The `spice-experiments` and `ltspice` skills said that nothing adds
  `.options logopinfo` for you. In fact the server adds it to every LTspice
  `.op` run, as the packaged guide already said. Both skills now say the server
  adds it and that writing it yourself is harmless.
- The LTspice skill no longer claims device operating points — gm, gds, vth, and
  the rest — require ngspice: LTspice `.op` runs surface them via the
  auto-injected `.options logopinfo`, and the `operating_point` recipe reads both
  engines uniformly. The packaged guide was already correct; the skill's copy had
  drifted. Swept gm, the gm/ID `.dc` table, still needs ngspice, as both now say.

### Removed

- The `full` and `agentic` tool profiles, and 48 of the 49 tools they exposed.
  (`plot_waveform` is the one that carried over.) The breaking-change
  section above says where each capability went and how to pin the old surface.
- Session library mounting (`load_library`, `unload_library`, `list_libraries`)
  is removed with no replacement. Reference libraries from the deck with `.lib`
  and `.include` directives instead.
- The netlist authoring and editing tools (`create_netlist`, `read_circuit`,
  `set_component_value` on a `.cir`/`.net`, `parameter`, `edit_directive`) are
  removed with no replacement. Write `.cir`, `.net`, and `.sp` decks with your
  own file tools; `inspect` reads them and `verify_circuit` checks them.

- `analyze_results` recipes no longer accept `step` or `all_steps`; pass them on
  the call (see Changed).
- `analyze_results` recipes no longer accept `reduce_field`, and `spec` no longer
  accepts `field`; pass `field` on the recipe (see Changed).
- `verify_circuit` no longer accepts the flat `reference`, `compare_mode`,
  `anchors` or `rtol`; pass the `compare` object, which carries all four.
- `edit_schematic` no longer accepts the flat `reference`; pass `compare`.
- `edit_schematic` no longer accepts `render`, `render_format`, `render_scale`,
  or `render` in `return_views`, and its response carries neither `views.render`
  nor the `artifacts` array that rendering was the only producer for.
  Rendering is `verify_circuit`'s, whose policy adds a pixel cap, inline
  delivery and a render-only mode.
- `edit_schematic` no longer accepts `write_failed_draft`. A failed batch writes
  nothing by design and the response names the stage that failed, so there was
  no draft to quarantine that the caller's own ops did not already describe.
- `edit_schematic` no longer accepts `format`; structured-aware clients render
  only `structuredContent`, and the other six tools had already dropped it.
- The AC crossing recipe's phase level is spelled `level_deg` only; the
  earlier `phase_deg` spelling, kept as an alias for callers of an earlier
  build, is gone.

- The MCP logging capability. The 2026-07-28 revision deprecates it whole
  (SEP-2577): the `logging` server capability, the server-to-client
  `notifications/message` delivery and the per-request log-level opt-in that
  replaced `logging/setLevel`, with no replacement offered, and
  `logging/setLevel` is absent from that revision's schema. The server no
  longer advertises the capability, answers `logging/setLevel` with
  method-not-found, and sends no log notifications. Job lifecycle events and
  diagnostics go to the process's stderr logger, which `[logging] level`
  still controls.
- **The columns form of budget-limited rows.** A response cap (`budget`) used to re-render row surfaces once it got tight
  enough: rows became arrays of bare values, with a sibling `*_columns` list
  naming what each position meant. It was lossless, but it made a row's shape
  depend on how small the cap was, so reading a row meant first working out
  which form had come back.

  Rows now keep their shape at every budget: a row is always an object with
  the same keys, and a tight budget returns fewer of them rather than
  differently shaped ones. The reduction ladder is trim, then answer, then
  shrink. If you read the columns form — `items_columns`, `values_columns`,
  `reduced_columns`, or any other `*_columns` sibling — those keys are gone
  from every response and output schema; read the rows from `items`,
  `values`, or `reduced` and page on with the cursor the response carries.
- **The pre-0.6 tool handlers and job machinery.** The handlers behind the removed 0.5 tools were kept in place through 0.6
  development so a tool could be re-exposed by putting its decorator back.
  That seam is gone: a handler with no caller has been deleted, along with
  its argument model, its output schema, and the helpers only it used. The
  eight tools and the Python API are unaffected — the advertised schemas and
  `ltspice_mcp.api.__all__` are unchanged. Restoring one of the old tools
  now means restoring its module from v0.5.x.

  Gone with them: the single-simulation and batch job types, the sweep and
  Monte Carlo runners, and the batch result reader. Every job the server runs
  is an experiment, on one runner. The Monte Carlo perturbation engine is
  unchanged — it is what `run_experiments` draws its random variations from.

  Error codes that only those handlers emitted are gone with them:
  `legacy_analysis_result`, `case_selection_wrong_job_kind`,
  `run_unavailable` (now `case_not_found`),
  `run_failed`, `no_raw_output`, `parse_deadline` (now `analysis_deadline`),
  `decimated`, `window_applied`, `complex_format_used`, `unrecognized_save`,
  `max_pk_pk_bucket`, and `export_written`.
- **The job sidecars earlier releases wrote.** Releases before 0.6 wrote a job record beside each circuit, at
  `.ltspice-mcp/jobs/<job_id>.json`. Those files are no longer read: a job id
  that only one of them names is simply not found, and `jobs` says so like it
  would for any other unknown id. Nothing was ever written there by 0.6, and
  the files themselves are left alone — delete them by hand if you want the
  space back. The `legacy_job_record` observation code is gone with the
  reading of them.
- **The internal tool-profile filter.** Tool registration no longer takes a
  profile, and there is no profile filter to look a tool up through: there is
  one surface to serve. Serving zero tools is still a hard error.
- `CONTRIBUTING.md` and the code of conduct are gone, and the sdist no longer
  ships them. The project is not taking outside contributions at this stage;
  the setup notes moved into `CLAUDE.md`, and the review rules it carried
  (a regression test fails before the fix, real code paths, plain language)
  already live there and in `docs/TESTING.md`.

### Security

- Dependency upgrades for published advisories in the locked runtime set:
  `click` 8.3.2 → 8.5.0, `cryptography` 49.0.0 → 50.0.1, `pillow` 12.2.0 →
  12.3.0.

## [0.5.0] - 2026-06-30

### Added

- Configurable ngspice compatibility mode: a `[simulator] ngbehavior` option
  (and `LTSPICE_MCP_NGBEHAVIOR` env var) sets the mode spicelib passes to ngspice
  at startup. The shipped default is unchanged. spicelib's default mode reads a
  sectioned `.lib <file> <section>` — the standard PDK corner-select idiom — as
  two plain includes, dropping the section so the corner isn't found; a mode
  without the `lt`/`ps` compatibility tokens (e.g. `ngbehavior = "hsa"`) parses
  it correctly. When an ngspice run fails with a missing-include error while the
  deck uses a sectioned `.lib`, the run now surfaces an actionable hint naming
  that fix.
- `measurement_stats` now echoes the reported measurement time (the `AT` /
  crossing value) on single-run reads, matching the aggregate-mode output.
- The bundled SPICE guide (`spice://guide`) gains an RF / two-port section: a
  one-port impedance probe (with the current-source sign convention that keeps
  `V(node)` equal to `+Zin`), reflection coefficient / return loss / VSWR, a
  noise-figure formula, and insertion loss — each idiom verified against a real
  ngspice run.

### Changed

- `create_schematic` now accepts the optional `format` (`"json"`/`"text"`)
  parameter its sibling tools take, returning structured content, instead of
  rejecting it as an unknown field.
- The duplicate same-name net-label warning no longer reads as a short: it now
  states that same-name labels merge into one net (a valid way to tie distant
  nets) and that only a later `connect(net=...)` is ambiguous; the
  `create_schematic` checklist says the same.
- Error messages point at the right tool: `get_waveform` on complex AC data now
  names `export_waveform` and `resonance`, and `find_model` cross-references
  `symbol_info` when the queried name is a schematic symbol rather than a
  library model.

### Fixed

- The operating-point log classifier no longer reports a converged run as
  failed. LTspice escalates the OP solve (Direct Newton → Gmin → source stepping
  → pseudo-transient) and logs "<method> stepping failed to find operating
  point" for each abandoned rung before a later method converges; those
  intermediate rungs were classified as errors and echoed into every analysis
  tool's warnings. A rung is now an error only when its own solve block never
  converged, and detection is scoped per solve block so a converged step in a
  stepped `.op` no longer masks a genuinely failed later step. A no-data run
  (no success line) still classifies as an error.
- Every `.asc` write path now refuses an empty or whitespace-only attribute
  value up front — `add_component`'s value and attributes,
  `set_component_attribute`, and the `apply_schematic_ops` op. An empty
  `SYMATTR` value writes a two-token line the schematic parser cannot read
  back, which corrupted the file.
- `pulse_response` now interpolates the settling-band crossing between the last
  out-of-band sample and the first in-band one instead of snapping to the
  in-band sample time (which lands up to a full timestep late on a coarse run),
  and warns when the local timestep near settling is coarse.
- `set_component_value` on an `.asc` now creates the `Value` line when the
  component exists but has none (it was added without a value), symmetric with
  `add_component(value=)`, instead of failing "not found".
- Transient reads now surface a `.meas WHEN` crossing time that the solver
  dropped, and flag under-floor divergences, instead of silently omitting them
  (`run_simulation` / `simulation_summary` / `measurement_stats`). A behavioral
  (B-source) expression with unquoted internal spaces is now a warning, not a
  hard error.
- Output-schema generation stays valid for schema-checking clients:
  `NotRequired` fields are no longer marked required under stringized
  annotations, and heterogeneous tuples are refused (with an invariant guard).
- `pulse_response` no longer reports a false `settling_time` when the trailing
  window is still ringing (the auto final-value lands on a ripple sample, so a
  settle band anchored to it produced a definite-looking but wrong time). That
  state now renders as `unknown` with a distinct quality flag, kept separate from
  a genuine "never settled within the window".

## [0.4.1] - 2026-06-27

### Fixed

- Cancelling or timing out a simulation could leave the simulator process
  running while the job was reported as cancelled, when the run's output was not
  in the default folder (e.g. a deck in a subdirectory). `cancel_job` now
  resolves the runner by the job's own netlist, so it addresses the runner that
  actually launched the job.

### Changed

- The default `ltspice-mcp.toml` is written lazily on the first tool call rather
  than at server startup, so the server no longer drops a config file into every
  directory an MCP client happens to launch it from — only ones where its tools
  are actually used.
- Simulation artifacts now go to a stable `.ltspice-mcp/runs` folder under the
  working directory instead of being scattered in the project root, each named
  per job so results stay isolated. A deck with a relative `.include`/`.lib`
  still runs in its own directory so the include resolves, and on WSL with
  LTspice the output is routed to a Windows-native temp directory so `.MEAS`
  results survive (LTspice can't write its SQLite `.db` over a UNC share).

## [0.4.0] - 2026-06-27

### Added

- Config knobs and recovery actions are now named at the surface where an agent
  hits the wall, instead of only in docs. `server_status` reports the resolved
  config file path (honoring `LTSPICE_MCP_CONFIG`) and how to switch the active
  simulator, plus `persist_jobs` / `preload_recent_count`. A timeout names the
  `run_simulation(timeout=)` argument, `[simulation] timeout`, and
  `LTSPICE_MCP_TIMEOUT`; a path rejection names `[security] allowed_paths` /
  `LTSPICE_MCP_ALLOWED_PATHS` and the copy-into-sandbox fallback; an unknown tool
  distinguishes a profile-hidden tool from a bad name and points at `[tools]
  profile`; `.asc` export without LTspice names `[simulator] path` /
  `LTSPICE_MCP_SIMULATOR_EXE`; and `find_model`'s no-match hint is profile-aware
  so it never suggests a tool the active profile hides.

### Changed

- The `circuit-mcp` and `ngspice-mcp` alias packages now share the canonical
  version (derived from the release git tag via `hatch-vcs`) and publish
  automatically on each release tag, so all three packages ship at one matching
  version instead of a separately-bumped alias version.

## [0.3.0] - 2026-06-27

### Added

- Per-device small-signal operating-point parameters (`gm`, `gds`, `vth`, `id`,
  …) are now first-class on both simulators and read back by name.
  `operating_point` surfaces them in a `device_op_points` bucket; on a
  `.dc`/`.tran` sweep the by-name readers
  (`query_value`/`signal_stats`/`export_waveform`) accept a uniform `dev.param`
  shorthand (e.g. `m1.gm`) that resolves to whichever wrapped form the raw holds
  (`@m1[gm]` / `v(@m1[vth])` / `i(@m1[id])`) — the gm/ID characterization read.
  On LTspice the values live in the `.log`'s operating-point block, so
  `run_simulation` / `run_sweep` / `run_montecarlo` auto-add `.options logopinfo`
  to `.op` decks and `operating_point` folds the block in (subcircuit devices are
  matched by instance regardless of the log's colon-qualified name); on ngspice
  they are `@dev[param]` traces that must be `.save`d, and a `dev.param` absent
  from the raw hints at the missing `.save`.
- `validate_netlist` and `export_netlist` now warn when a `.meas`, output
  directive, or behavioral source references `V(name)`/`I(name)` for a node or
  device the netlist doesn't define — the common case being a schematic net that
  was wired but never labeled (so it exports as `N00x` while a `.meas V(vref)`
  still asks for `vref` and silently resolves to nothing), plus plain typos. The
  known-name set is deliberately over-approximated, so the check only fires on a
  genuinely-absent name; hierarchical refs (`V(X1:out)`) and expression fragments
  (`V(a*2)`) are left alone. `validate_netlist` runs it on `.cir`/`.net`;
  `export_netlist` runs it on the exported netlist (where an `.asc`'s final net
  names and its directives sit together).
- `bode_metrics` accepts a transfer-function ratio as its `signal` —
  `V(out)/V(mid)` divides the two complex AC traces before any mode runs, so
  inter-stage gain, loop gain, and PSRR (which the simulator never stores as a
  single trace) are analyzable directly instead of via a deck-side behavioral
  node. Restricted to a single two-signal quotient; all four modes and
  `all_steps` work on the ratio. A denominator that nulls (the ratio is
  singular — a genuine pole) is reported with the offending frequencies rather
  than silently dropped, so a hidden pole can't skew the metrics.
- `apply_schematic_ops` gained a `dry_run` flag: it validates the whole batch
  against an in-memory copy and reports per-op results without writing the file.
  Every op is attempted (errors don't stop the run), so one bad op surfaces all
  problems at once instead of rolling back a good batch — check the plan, then
  resubmit the corrected ops with `dry_run=false`.
- MCP prompts (workflow starters a host surfaces as slash-commands):
  `characterize_filter`, `run_and_plot`, and `step_response`. Each emits the
  canonical tool pipeline for that task with the circuit path filled in.
- Distribution as a Claude Code plugin (`.claude-plugin/`) and a Claude
  Desktop extension (`packaging/mcpb/`). Both wrap the published package via
  `uv` (the plugin runs `uvx`; the extension is a `type: "uv"` bundle) and so
  require `uv` and a simulator (LTspice or ngspice) on the host rather than
  bundling either.
- `thd` tool: total harmonic distortion (THD and THD+N) of a periodic transient
  signal computed by FFT — no `.four` directive needed, and works on any
  simulator. Defaults to coherent sampling (the record is trimmed to a whole
  number of fundamental cycles and a rectangular window is used) so harmonics
  land exactly on FFT bins and the result is exact; `window="hann"` is the
  approximate fallback. The fundamental is auto-detected (sub-bin accurate) or
  given. Every condition the number depends on is surfaced — the fundamental and
  whether it was given or detected, the window kind, cycles analyzed, FFT length,
  sample rate, per-harmonic levels — and the tool warns rather than lying when a
  window turns out non-coherent or the FFT-length cap forces a down-sampling that
  could alias.
- `noise_integral` tool: integrates a `.noise` spectral density to a total RMS
  over a band as `sqrt(∫ density² df)` (the amplitude-density convention shared
  by LTspice's `V(onoise)` and ngspice's `onoise_spectrum`). Reports the band
  actually integrated and the sample count; handles a high→low sweep. Noise
  figure / SNR are left to the caller (they need the source resistance and a
  reference level).
- `operating_point` gained a `device=` filter that returns just one device's
  operating-point params and terminal currents (e.g. `device="M1"` → `@m1[...]`
  plus `Id/Ig/Is(M1)`) in a single call, refusing an unknown device with the list
  of devices present. Every returned value now carries its SI unit in a `units`
  map where the simulator declared the trace type.
- `query_value` and `export_waveform` now attach SI units derived from the
  simulator's declared trace type (`query_value` returns a `unit`; the CSV's
  value columns are unchanged but the DC x-column is named — see Changed).

### Changed

- `max_parallel_sims` now defaults to the host core count capped at 8 (was a
  flat 4, so a many-core box was throttled out of the box). Still overridable via
  `[simulation] max_parallel` / `LTSPICE_MCP_MAX_PARALLEL` up to 128; the cap
  keeps parallel cold simulator processes from thrashing memory/IO.
- Sweep cross-products are now capped at 10000 runs, matching the existing Monte
  Carlo cap. A multi-axis sweep (e.g. 5×5×16×100) previously spawned tens of
  thousands of cold simulator processes silently; it's now refused at
  `configure_sweep` with the offending dimension sizes. The cap is computed from
  each dimension's *count* before any value list is built, so a single fat
  dimension (`points=1e9`, or a tiny `step` over a wide range) is rejected
  without `np.linspace`/`np.arange` allocating a multi-GB array first.
- **Breaking:** MCP resource URIs moved from the `ltspice://` scheme to
  `spice://` (`spice://results/...`, `spice://netlists/...`, `spice://config`,
  `spice://guide`, etc.). These resources are engine-agnostic, so the scheme no
  longer implies LTspice. Clients discover resources via `resources/list` and
  the URI templates, so this only affects anything that hard-coded the old
  scheme. The MCP Apps widget keeps its spec-mandated `ui://` URI.
- Generalized the `spice://guide` resource to cover both engines: it now carries
  engine-neutral SPICE fundamentals plus separate **LTspice-Specific** and
  **ngspice-Specific** sections (ngspice `.control` scripting, `.save`, XSPICE,
  control-loop Monte Carlo, expression parsers) and an LTspice-vs-ngspice
  differences table. The per-engine Claude Code skills stay engine-specific.
- The MCP handshake now reports the `ltspice-mcp` package version in
  `serverInfo.version` instead of the MCP SDK's version.
- `serverInfo.name` can be overridden with the `LTSPICE_MCP_SERVER_NAME`
  environment variable (defaults to `ltspice-mcp`), so alias launchers can
  self-identify in the handshake.
- The server instructions now open with a line naming the simulators actually
  detected at startup (e.g. "Active simulator: ngspice (LTspice not detected).")
  so a client without LTspice no longer reads the LTspice-centric name as a
  degraded state. The instructions also now point to `plot_waveform` for
  visualization.
- When no simulator is detected, the instructions, every "no simulator" tool
  error (run / sweep / Monte Carlo / batch), and `server_status` now give a
  platform-appropriate way to get one (install ngspice, or point
  `LTSPICE_MCP_SIMULATOR_EXE` at an existing binary), note that the server must
  be restarted to re-detect, and tell the agent to ask the user if it cannot
  install — instead of the prior dead-end "check server status". When ngspice
  is present but LTspice is not, the instructions now say `.asc` schematic
  editing is unavailable while simulation and analysis run unaffected.
- Reworded the "symbols unavailable" startup log so it reads as informational —
  it disables only `.asc` schematic graphics editing and leaves SPICE
  simulation and netlist editing unaffected.
- `signal_stats` and `get_waveform` descriptions say "SPICE's adaptive
  timestep" rather than "LTspice's", since the behavior is generic to ngspice
  too.
- Lowered the minimum Python from 3.13 to 3.11. The two 3.12+/3.13-only
  constructs (PEP 695 generic syntax, `tomllib`) are replaced or already
  available on 3.11, so 3.11 and 3.12 users can now install. CI (and the
  release gate) now runs the full test suite on 3.11, 3.12, and 3.13, so the
  advertised floor is proven on every push.
- **Breaking:** `measurement_stats` renamed its `best_step_index` /
  `worst_step_index` fields to `min_step_index` / `max_step_index`. "Best" and
  "worst" implied a verdict the tool can't justify — whether a low or high value
  is "best" depends on what was measured — so the fields now name the plain fact
  (the step index where the min / max value occurred). No alias is kept (this is
  a pre-1.0 clean break); the structured `outputSchema` advertises the new names.
- `query_value` and the `export_waveform` CSV now label a `.dc` sweep axis by
  its swept variable (e.g. `Vin` / a `Vin_V` CSV header) instead of a misleading
  `t=` / a bare `sweep` column.
- The "device operating-point param not found" hint is now imperative and fires
  for a bare `@dev[param]` request too: it tells you to add `.save @dev[param]`
  and re-run with ngspice, rather than passively noting the value exists
  elsewhere.

### Fixed

- The in-tool SPICE guide and the run-time pre-flight warning gave the wrong
  ngspice `.meas` recovery. ngspice suppresses `.meas` only under `-b` plus a
  command-line `-r rawfile` (this server's invocation); the documented
  workarounds were both wrong — inside a `.control` block ngspice wants the
  dot-less `meas` command (a dotted `.meas` there errors and computes nothing),
  and `measoutfile` is a no-op once the measurement is suppressed. Both are now
  corrected, verified against real ngspice.
- `validate_netlist` now flags `.backanno` when targeting ngspice — ngspice has
  no such command and aborts the run with "unimplemented dot command
  '.backanno'" (it is an LTspice-only schematic directive). Mirrors the existing
  `.tran 0` ngspice-incompatibility check; an LTspice target is unaffected.
- The batch result reader no longer silently returns the wrong sample on a
  descending sweep axis. A `.dc V1 5 0 -0.1` (or any high→low parameter sweep)
  produces a descending axis, but the `at=` slice used a binary search that
  assumes ascending order — so asking for the value at one axis point could
  return the value at another. The axis and its wave are now flipped to ascending
  before the search; ascending sweeps are unaffected, and the per-run work stays
  allocation-free.
- `batch_results` now surfaces, rather than silently drops, per-step data it
  cannot aggregate. A run whose `.raw` carries its own `.step` sweep is read at
  step 0 only; those runs are now reported in `step_collapsed_runs` with a note
  pointing at `get_waveform`/`query_value` (`job_id`, `run_index`, `step=<n>`) for
  the remaining steps. If step metadata can't be read at all, the run is reported
  in `step_unknown_runs` instead of being silently assumed single-step — step 0 is
  still returned, so the readable data is never dropped.
- `configure_sweep` now warns when a parameter axis is named `temp`/`temperature`:
  it is emitted as `.param temp=…`, which does not set the simulation temperature
  (SPICE controls that via `.temp`, `.options temp`, or `.step temp`), so a
  "temperature sweep" would otherwise run silently at a single temperature.
- `batch_results`' `raw=true` mode is now documented honestly: it returns per-run
  *reduced* rows (a single `value`, or peak/mean/min), not raw sample vectors. For
  the actual samples (e.g. a gm/ID table) use `export_waveform`/`get_waveform` with
  `job_id`+`run_index`.
- `query_value` (raw and job-run modes) and the step-by-axis-value lookup now
  resolve the nearest point correctly on a descending sweep axis. They share the
  same binary-search resolver as the batch reader, which was ascending-only — so
  a high→low `.dc`/`.step` lookup could land on the wrong point. The direction
  handling now lives in that one resolver.
- `operating_point` now carries the simulator's "unrecognized variable" warning:
  a `.save`d `@dev[param]` the device class doesn't have (a typo, or an
  unsupported parameter) is written to the raw as a real-looking `0.0`, which was
  indistinguishable from a true zero. The log warning that says it's bogus is now
  surfaced alongside the value.
- `validate_netlist`'s dangling-reference check no longer false-flags simulator
  reserved traces (`onoise`/`inoise`/`time`/`frequency`/…) or probe references
  inside `.meas` / output directives as undefined nodes.
- `periodic_metrics` warns when edge spacing is strongly bimodal — the signature
  of a frequency/duty reading that is off by ~2x because alternate edges were
  miscounted.
- Schematic guidance no longer tells you to leave every signal net unlabeled.
  `connect` wires pins but assigns no net name, so an unlabeled net exports as
  `N001`/`N002`/… — silently breaking any `.meas V(vref)`, `.param` expression,
  or behavioral `B`-source that references the net by name. The `create_schematic`
  checklist, its tool description, and the `spice://guide` "Named nets" section now
  state the rule: wire-only is fine for nets you never name, but label any net a
  directive references by name with `add_net_label`.
- A cancelled or timed-out simulation now has its partial output reclaimed
  instead of stranded on disk. A timed-out LTspice run can keep its `.raw`
  open and reach several GB; previously that file (and the run netlist/log)
  was left behind forever. The runner now deletes the run's artifacts when the
  killed process's completion callback fires (the point at which the file
  handle is released). Cleanup is gated to the killed statuses, so a completed
  run's good output is never removed.
- `get_waveform` overview buckets are capped at 2000 (was the
  `max_points_returned` ceiling, which is sized for one-value-per-point arrays).
  Each bucket carries ~8 scalar fields, so requesting the old maximum serialized
  past the MCP response budget and spilled to a file at the tool's own
  documented limit. The default (200) is unchanged.
- Clarified the `run_simulation` `timeout` help: with `wait=true` the effective
  limit is `min(timeout, 600s)` — 600s is a hard ceiling, not a floor — so the
  default 300s timeout is what bounds a `wait=true` run unless a larger timeout
  is passed.
- `add_component` (and `apply_schematic_ops` add-component) no longer fail with
  an opaque "Internal error" on two real-world cases:
  - **Vendor symbols with non-ASCII descriptions.** Hundreds of LTspice's
    bundled `.asy` symbols (op-amps, comparators, ADC/DAC, …) carry cp1252 bytes
    (`µ`/`°`/`±`/`©`) in their description fields. The symbol parser read them as
    strict UTF-8 and raised `UnicodeDecodeError`; it now uses the shared
    encoding fallback (BOM/UTF-16/cp1252), and any still-unparseable `.asy`
    degrades to "symbol unavailable" instead of crashing the tool. The same
    strict-UTF-8 hazard on the sibling reads was closed too: `export_netlist`'s
    `.net` read and the two `.log` reads that lacked the tolerant decode the rest
    of the log parser already used.
  - **spicelib 1.6.** spicelib 1.6 made `Component.attributes` a lazy property
    that raises for a from-scratch component built as a bare
    `SchematicComponent`; construction now uses `AscComponent` (the type
    spicelib's own parser builds) when present. The dependency is also capped to
    the tested range (see below), so a default install keeps working today.
- The parsed-result cache is now bounded (LRU, 32 entries). It was unbounded, so
  a long-lived session querying many circuits pinned every `.raw` it ever parsed
  (each potentially multi-MB) in memory. The editor cache stays unbounded by
  design — it can hold unsaved in-memory edits that eviction would drop.
- A netlist with a relative `.include`/`.lib` now also keeps its run in the
  working dir on WSL with a Linux-filesystem working dir — that branch relocates
  artifacts off the UNC path, but previously did so before the local-dependency
  check, orphaning the include. Self-contained decks still relocate.
- Simulation artifacts no longer flood the working directory. Runs that
  previously dropped their `.raw`/`.log`/`.db`/`.op.raw`/netlist files directly
  into the project root (the non-WSL-Linux and WSL-on-`/mnt/` cases) now write
  to a single `.ltspice-mcp/runs` sidecar alongside the job metadata — one tidy
  place to find or delete them (a 30-run Monte Carlo alone emits ~180 files).
  The sidecar stays on the working dir's own filesystem, so a Windows-native
  (`/mnt/`) working dir keeps LTspice's `.db`/`.MEAS` support. A netlist that
  pulls in a sibling file via a *relative* `.include`/`.lib` keeps its artifacts
  beside the netlist (a relocated copy couldn't resolve the dependency);
  self-contained and lib-path/absolute-include decks get the sidecar.
- Unexpected tool errors now report the actual exception type and message
  (e.g. `Internal error in foo: KeyError: 'bar'`) instead of the dead-end
  "Check server logs for details" — the server's stderr traceback isn't
  reachable from an MCP client, so the concrete cause is what makes a failure
  diagnosable. The full traceback still goes to the logs.
- `edge_metrics` no longer emits a spurious "window shows falling transition"
  warning when an explicit `edge` direction is requested over a window that
  captures a full pulse (rise *and* fall). The endpoint-derived direction is
  meaningless there; the requested direction is honored and a genuinely absent
  edge still raises a clear error.
- `export_waveform` CSVs are no longer corrupted on Windows. The atomic writer
  opened text files without disabling newline translation, so `csv.writer`'s
  `\r\n` line terminators were doubled to `\r\r\n` — a blank row between every
  data row. The writer now opens text mode with `newline=""`, which also gives
  every text artifact consistent `\n` endings cross-platform. (Latent on Linux,
  which does no newline translation.)
- `export_waveform` / `plot_waveform` no longer double-nest their sidecar when
  the source `.raw` already lives inside a `.ltspice-mcp/` tree (a job-run raw
  passed by path landed at `…/.ltspice-mcp/runs/.ltspice-mcp/waveforms/`). The
  artifact now goes into the existing tree (`…/runs/<job>/waveforms/`).

### Changed

- `apply_schematic_ops` now accepts the `format` parameter (`"json"`/`"text"`)
  that the other structured tools take, instead of rejecting it with a
  validation error.
- Capped `spicelib` to `>=1.4.9,<1.6` (the tested range). spicelib 1.6
  refactored the editor model — most visibly, `.PARAM` values now come back
  typed rather than as strings — which the parameter read/edit path doesn't yet
  handle. Without a ceiling, an unpinned install (`uvx ltspice-mcp`) resolved to
  the newest release and shipped an untested spicelib. The component-
  construction half of 1.6 support already landed (`AscComponent`); the cap
  lifts once the parameter path is migrated.

## [0.2.1] - 2026-06-16

### Added

- `validate_netlist` rejects empty / whitespace-only netlists (error) and
  warns on single-connection (dangling) nodes in `.cir`/`.net` decks. The
  dangling-node pass is scope-aware (`.subckt` bodies counted separately,
  header ports count as a connection), excludes ground and `.global` nodes,
  and counts terminals conservatively — nodes referenced only by expressions
  (`V(...)`/`I(...)`), unrecognized elements, or non-terminal tokens are never
  warned about. `.asc` schematics are exempt (their connectivity lives in
  wires and flags, which this pass cannot see).
- `validate_netlist` warns on bias-topology degeneracies in `.cir`/`.net`
  decks — a net with no DC path to ground, whose operating-point bias is
  undefined: a floating MOSFET gate, an all-capacitive (AC-coupled) island, a
  node driven only by a current source (independent, controlled, or a
  behavioral `B… I=` source without `Rpar`), or a galvanically-isolated
  domain. The check is built by conservative over-connection (only capacitor
  dielectrics, MOSFET gate oxides, and current-source branches count as DC
  opens; bias-dependent diodes, transistor channels, and switches all
  conduct), so a warning is provable rather than a guess, and each
  physically-contiguous floating domain reports once. A subckt that grounds a
  port through node 0 internally biases the net wired to it; a `.GLOBAL` rail
  is a ground reference only where it actually reaches node 0 somewhere in the
  deck (a floating global is flagged). Messages state the topology fact only —
  no convergence verdict. `.asc` is exempt, as for the dangling-node pass.

### Changed

- Heavy blocking work no longer stalls the server: `.raw` parses, batch
  result/log loops, the recent-circuits index lock and durable write, WSL
  `cmd.exe` interop resolution, and MCP resource reads now run in worker
  threads, so concurrent requests — including `cancel_job` — stay responsive
  while a multi-second result parse is in flight. (The MCP SDK dispatches
  requests concurrently; previously any large parse froze every in-flight
  request until it finished.)
- CI: the push/PR gate (`ci.yml`) and the release gate (`publish.yml`) now
  call one shared reusable workflow (`checks.yml`), so the two gates cannot
  drift. GitHub status-check contexts are renamed to `checks / check` and
  `checks / audit`.
- Monte-Carlo per-run parameters in `batch_results`/`measurement_stats` are
  now JSON numbers (the actual perturbed magnitudes), matching the sweep
  runner, which already emitted numbers — previously Monte-Carlo emitted them
  as strings, so the same field had two types across the two batch run kinds.
- The completion value-scan (NaN/Inf/extreme-magnitude surfacing) now runs on
  every result up to a total-sample budget (axis points × trace count, ~5M
  samples) instead of only on single-point operating points. A normal
  `.tran`/`.ac` is now fully scanned — so a degenerate value is surfaced and the
  `value_scan_skipped` observation no longer fires on essentially every
  multi-point run; only a result whose traces exceed the budget (a long
  transient or a wide node dump) reports the skipped scan. The budget counts
  total samples, not axis points alone, so a wide result can't force every
  trace into memory on the completion path.
- `bode_metrics` filter mode now references the `-3 dB` cutoff to the passband
  *plateau* gain (the flat DC-side edge for a lowpass, the high-frequency edge
  for a highpass, the peak for a bandpass) instead of the band median. The band
  median was dragged down by the roll-off knee inside the auto-detected band,
  which biased the reported cutoff outward (several percent on a sweep that
  starts only a decade below the cutoff); the plateau reference removes that
  bias. The explicit `passband_range` path is unchanged.

### Removed

- The `[plotting]` config section (`plot_dpi`/`plot_style`): it was never
  consumed by any code path — no plotting feature exists. Existing TOML
  files that still contain the section parse fine (it is ignored).

### Fixed

- `docs/DESIGN.md`, the example TOML, and the skills docs now match the
  shipped surface: current tool names throughout, the real diagnostics and
  observations mechanism, the real mutation-safety behavior
  (validate-before-write refusals, `reset_schematic` snapshots,
  `export_netlist` diff — no `dry_run`, no inline plot images), the exact
  agentic-profile tool set, and ngspice's first-class status. Doc-drift
  tests now pin tool names in README/DESIGN/skills against the live
  registry.
- `measurement_stats` on a batch job with no `.MEAS` results now relays the
  per-run log diagnostics that explain why (e.g. ngspice's "No .measure
  possible in batch mode"), capped at 8 distinct lines, instead of a bare
  "No .MEAS results found".
- A truncated or corrupt `.raw` file that parses as having zero variables is
  now diagnosed as corrupt when loaded, instead of surfacing downstream as
  "Signal not found" with an empty available-signals list.
- A failed or timed-out simulation now returns its `log_file` and `raw_file`
  paths (and a text footer naming them), so the caller can open the full log
  rather than working from the excerpt alone. The error excerpt is also
  tail-aware — it now anchors a window on both the first and last diagnostic
  line, so a convergence abort's failing-node tail ("Timestep too small",
  "trouble with node", "Last Node Voltages") survives even when an earlier
  benign line came first. Those phrases are now recognized as error anchors.
- An interrupted job (the server stopped mid-run and recovered it on restart)
  no longer reports a wall-clock-to-recovery "duration … (running)" in the
  `check_job` list — its true runtime is unknowable after a restart, so the
  list now shows "unknown" and omits the duration, matching the single-job view.
- `batch_results` raw-mode rows now have a uniform shape: a full-waveform run
  whose signal happens to be flat keeps `peak`/`mean`/`min` like its siblings
  instead of collapsing to a single `value` (the collapse is now scoped to
  genuine point queries — `at=` or an operating-point raw).
- `batch_results` given a single-simulation job id now points at tools that
  accept a job id (`check_job`, `query_value`) instead of `simulation_summary`,
  which takes a raw-file path and cannot be reached with a job id.
- Error hints are no longer misleading on job-state and argument-shape errors:
  cancelling an already-finished job, `query_value` argument mistakes, and a
  `simulation_summary` build failure no longer append the generic "verify
  simulator availability" / "check signal names with simulation_summary" footer
  (the latter was self-referential). Genuine result-content errors keep it.
- A bare `check_job()` whose default (queued/running) view is empty but which
  has finished jobs now says how many are hidden and that `status="all"` lists
  them, instead of a bare "No active jobs" that read as "nothing exists".
- `bode_metrics(all_steps=true)` surfaces an identical per-step warning once at
  the top level with its step coverage, instead of repeating the same string in
  every step's entry and text.
- `edit_directive`'s description now states that adding a `.param` is
  unsupported (use the `parameter` tool), matching the runtime refusal.
- Setting a new `.param` via the `parameter` tool no longer leaves spicelib's
  `; Batch instruction` boilerplate comment in the saved netlist.
- `diff_circuit` no longer reports a component or directive as changed when the
  only difference is the micro prefix's rendering (`1u` vs `1µ`); a real
  magnitude change (`1u` vs `2u`) is still reported.

### Security

- Dependency upgrades for published advisories: `cryptography` 46.0.7 →
  49.0.0 (GHSA-537c-gmf6-5ccf), `python-multipart` 0.0.27 → 0.0.32
  (CVE-2026-53538, CVE-2026-53539, CVE-2026-53540), and `starlette` 1.2.1 →
  1.3.1 (CVE-2026-54282, CVE-2026-54283). All three arrive transitively via
  the MCP SDK's HTTP transport, which this stdio server does not use.

## [0.2.0] - 2026-06-10

The largest release to date: every tool renamed, the AC-analysis surface
consolidated, ngspice promoted to a first-class simulator, an in-house
Monte Carlo engine and SPICE lexer, and a long list of correctness fixes
across simulation lifecycle, result parsing, and schematic editing.

### Breaking

- All tool names drop the `ltspice_` prefix (`ltspice_run_simulation` →
  `run_simulation`, and so on for every tool) — MCP clients already
  namespace tools by server, so the prefix was redundant.
- The four AC analysis tools `filter_metrics`, `roll_off`, `gain_at`, and
  `find_crossing` are consolidated into the new `bode_metrics` tool
  (`mode="filter" | "slope" | "point" | "crossing"`); no aliases remain.
- `measurements` is removed — `simulation_summary` now returns simulation
  metadata, structured `.MEAS` results, and Fourier data in one call.
- `model_info` is removed — `find_model(exact=true, full=true)` attaches
  the SPICE definition body to a match.
- `add_text` is removed — `edit_directive(kind="comment")` adds comments,
  and refuses comment text that looks like a mistyped directive.
- `simulation_summary` no longer reports `phase_margin` / `gain_margin`;
  loop-gain analysis lives in `stability_metrics`.
- `find_model` output: the `parameters` field is split into `ports` (the
  full subcircuit port list — previously truncated to five entries) and
  `params` (model body / parameter-clause defaults).
- Run summaries rename `trace_names` to `signals`.
- Server state (the recent-circuits index) moves from `~/.ltspice-mcp` to
  `$LTSPICE_MCP_HOME`, `$XDG_STATE_HOME`, or `~/.local/state`; old index
  files are not migrated.

### Added

- New schematic tools: `apply_schematic_ops` (a list of edits applied as
  one transaction instead of 25+ round-trips), `create_schematic` (seed an
  empty `.asc`), `schematic_from_netlist` (generate a wired `.asc` from
  netlist text for R/C/L/V/I/D circuits), `trace_net` (list every
  pin/label/wire on a net and flag shorts), and `reset_schematic` (revert
  an `.asc` to its snapshot from before the first edit).
- New circuit tools: `validate_netlist` (static pre-flight: directive and
  element-arity validation, `.MEAS`-vs-analysis mismatches, duplicate
  analyses, unparseable B-sources; on `.asc` also named-net shorts,
  floating pins, and dangling labels) and `diff_circuit` (structural diff
  of components, values, attributes, and directives between two files).
- ngspice is now a first-class simulator: raw files parse with the correct
  dialect, logs produce structured diagnostics, simulator stdout/stderr is
  captured and folded into diagnostics, and pre-flight checks reject
  `.step` (pointing to `configure_sweep`) and warn about `.MEAS`
  directives ngspice skips in batch mode.
- In-house Monte Carlo engine replacing `spicelib.Montecarlo`: per-`.MODEL`
  process variation, Pelgrom-scaled per-instance MOSFET mismatch, and
  `.PARAM` rewriting alongside the R/C/L engine — with a `seed` for
  reproducible runs, relative or absolute tolerances, and per-run realised
  values stored so statistics can be correlated with measurements.
- Result observations: every run summary carries an `observations` list of
  facts for the caller to weigh — the simulator's own error lines,
  requested `.MEAS`/`.FOUR` results that were not produced, NaN/extreme
  node values, and skipped scans — instead of a trust verdict.
- Simulator detection rework: a configured-but-missing simulator is
  reported in `server_status` instead of silently handing back another
  simulator's results; LTspice auto-detection on WSL; a `[simulator]`
  enabled-allowlist; `.asc` editing works even with no simulator detected.
- Schematic editing feedback: every mutating `.asc` operation reports
  floating pins, duplicate wires, and dangling labels; `connect` and
  `add_net_label` detect named-net shorts at edit time; `move_component` /
  `remove_component` report orphaned wires, and `remove_component` gains a
  `cleanup_wires` flag; unknown attribute names are rejected with a
  case-typo suggestion.
- Sweep/Monte-Carlo workflow: `configure_sweep` accepts an explicit
  `values` list (e.g. E-series) alongside start/stop ranges;
  `configure_montecarlo` validates per-component overrides against the
  netlist; batch jobs appear in `check_job` and job listings;
  `batch_results` surfaces per-run convergence-fallback markers
  (Gmin/source stepping) so a degenerate sweep is not reported as clean.
- `measurement_stats(job_id=...)` aggregates `.MEAS` results across every
  run of a sweep/Monte-Carlo batch, or across the steps of a stepped
  single run.
- `query_value` and `bode_metrics` accept `job_id` + `run_index` to
  analyze one run of a batch like a standalone raw file; `query_value`
  gains `step_axis`/`step_value` (query a signal at a chosen `.step`/`.dc`
  sweep value) and `magnitude_linear`; `bode_metrics` gains `all_steps`.
- `simulation_summary` auto-derives the log file from the raw file,
  accepts a `step` selector, and auto-picks an AC signal (with a warning)
  instead of silently dropping bandwidth metrics; `operating_point`
  accepts `step` for stepped `.op` results.
- Library handling: `find_model(full=true)` attaches definition bodies;
  `list_libraries` detail mode enumerates `.MODEL` names (foundry decks
  with hundreds of models were previously invisible); LTspice's bundled
  `standard.*` component decks and current LTspice install paths are
  recognized.
- The server publishes MCP usage instructions (netlist-first workflow,
  matching analysis tools to run types, checking diagnostics rather than
  trusting a "completed" status).
- Structured `.MEAS` error surfacing: unsupported expressions (`vdb()`,
  `phase()`, `group_delay()`) are blocked at directive-write time with
  concrete suggestions, and `meas_errors` extracted from logs propagate
  through `run_simulation` and `simulation_summary`.
- `SECURITY.md`, this `CHANGELOG.md`, Dependabot configuration, a
  `pip-audit` CI step, and `twine check --strict` on built artefacts
  before publish.

### Changed

- Netlist reading and editing now run on an in-house structured SPICE
  lexer instead of regex passes: format-preserving edits, hierarchical
  `.subckt` netlists, and quoted/braced/single-quoted expression tokens
  are handled correctly (one malformed expression form previously hung the
  server).
- Single-run and batch jobs live in one job store with a unified
  read-model: any job id works with `check_job`, `cancel_job`, and
  `measurement_stats`, and a wrong-tool lookup redirects to the right tool
  instead of claiming the job does not exist.
- File decoding for netlists and libraries uses a strict ladder (BOM
  sniff, UTF-16-without-BOM heuristic, UTF-8, cp1252, lossy fallback), so
  Windows-edited files with stray symbols (°, µ) read correctly.
- `pulse_response` surfaces instead of refusing: windows where
  overshoot/settling are undefined (a full pulse in the window, noisy
  baselines) return null metrics with machine-readable `quality` codes
  rather than an error or a plausible-but-wrong number; `edge_metrics`
  notes when a noisy window biased its auto-detected levels.
- `run_simulation`'s inline summary and the `simulation_summary` tool now
  share one pipeline and return the same fields.
- Errors that already carry precise guidance no longer get a generic hint
  appended; `.asc` and symbol-resolution error messages are clearer.
- `pyproject.toml` declares Trove classifiers; CI and publish workflows
  hardened (read-only token, publish action pinned by commit SHA, job
  timeouts, concurrency groups); the publish workflow now runs the same
  gate as CI (type check, real-simulator end-to-end tests, dependency
  audit) so a release cannot clear a weaker bar than an ordinary push.
- Internal API and typing tightening: `SimulationRunner.kill` is public
  API (renamed from `_kill`), `Literal`/`TypedDict` adoption across
  runners and parsers, and lint/type configuration tuned for scientific
  notation in docstrings.
- `DESIGN.md` moved to `docs/` and rewritten as a user-facing reference;
  README restructured around the user workflow with a corrected tool
  catalog.

### Fixed

Numeric correctness — these returned plausible wrong numbers:

- `bode_metrics` point mode interpolated wrapped phase, so a query between
  samples straddling the ±180° seam could be off by up to ~180°;
  interpolation now runs on unwrapped phase.
- Overshoot/undershoot detection took the first local peak, so pre-edge
  ripple in the window reported 0 % overshoot on a genuinely overshooting
  edge; the largest peak now wins.
- Monte-Carlo MOSFET mismatch silently skipped devices with lowercase
  `w=`/`l=` geometry (the ngspice convention), producing zero-mismatch
  runs that looked clean; geometry reads, tolerance specs, and
  batch-result filters are now case-insensitive.
- Parallel sweeps mislabelled run parameters when `max_parallel > 1` and
  runs completed out of order; results now pair with the submitting run.
- Replaced spicelib's broken Gaussian sampler, which used an absolute
  rather than relative sigma and an unseeded generator per call.
- AC `operating_point` no longer silently returns complex magnitudes as
  voltages, and operating-point results no longer drop their first node as
  a phantom sweep axis.
- The −3 dB half-power point uses the exact −3.0103 dB constant
  everywhere; frequency arguments accept Hz/kHz suffixes; SPICE values
  with unit annotations (`1uF`, `10MegHz`) parse instead of being skipped.
- `.step temp=-40°` log rows and `.step param NAME` sweeps whose raw files
  carry no axis now resolve correctly.

Simulation lifecycle:

- Timeouts above the synchronous threshold (including the default) were
  never enforced — a hung simulator ran forever; jobs now arm a watchdog
  that marks the job timed out and kills the process at the deadline.
- Cancelling or timing out a job on WSL did not actually kill the
  Windows-side LTspice process; single-run and batch cancel now terminate
  it, and a cancelled batch no longer launches its next queued run.
- The `max_parallel_sims` limit is now enforced for `run_simulation` jobs;
  jobs that time out while still queued terminate cleanly instead of
  launching an orphan.
- Failed or broken runs could be reported as successes: simulator failure
  placeholders leaked as `completed`, log `Error:` lines were dropped, and
  ngspice convergence failures were classified as warnings — all now
  surface as errors.
- `cancel_job` could not cancel sweep/Monte-Carlo jobs, and jobs
  interrupted by a server restart raised "unexpected status" instead of
  reporting partial results.
- Duration reporting: negative durations under clock skew, mismatches
  between the inline response and a later `check_job`, fabricated
  durations on jobs recovered after a restart, and `active_jobs` counting
  completed records.

Results and diagnostics:

- Fourier extraction returned empty harmonics for every `.four` analysis;
  zero-amplitude `.FOUR` signals no longer crash measurement parsing; an
  ignored `.fourier` directive is flagged.
- FAIL'ed `.MEAS` results are surfaced in a `failed_measurements` list
  instead of silently disappearing from the output.
- ngspice's `Circuit: <title>` log echo was misparsed as a measurement
  named "circuit".
- `batch_results`: AC aggregation no longer collapses to peak-only, value
  filters match parameter names case-insensitively, single-sample runs
  render values instead of empty columns, and an empty batch raises
  instead of returning a silent empty page.
- Structured tool output now conforms to each tool's declared schema:
  `.asc` attribute values no longer leak non-JSON-serializable objects,
  optional fields are no longer published as required, and `check_job` no
  longer emits a null `error` that schema-validating clients rejected;
  `export_netlist` is no longer mislabelled as read-only.

Netlist and schematic editing:

- `set_component_value` corrupted several element classes (PULSE/SIN
  source specs rejected, B-source `V=`/`I=` prefix dropped, controlled-
  source gain edits overwriting controlling nodes, MOSFET `W`/`L` tokens
  duplicated); values are now dispatched per element class and validated
  before writing, and batch mode validates everything up-front instead of
  half-applying.
- Schematic saves are atomic with editor-cache rollback, so a failure
  mid-write no longer leaves a corrupt `.asc` on disk or stale edits in
  memory; empty attribute values that made a schematic permanently
  unreadable are refused; multi-op transactions roll back on any error.
- `connect` no longer creates silent shorts: wires routed through another
  pin of an endpoint component are refused, and label shorts on
  mid-segment contact are detected.
- `edit_directive`: removing a directive containing parentheses no longer
  silently does nothing, `.param` edits are redirected to the `parameter`
  tool with a clear message, and `x`/`y`/`size` placement now applies to
  `.asc` directives.
- Netlists with a UTF-8 BOM, UTF-16 without BOM, an unclosed `.SUBCKT`, or
  behavioural sources with commas inside `if(...)` no longer fail
  `read_circuit`/`list_components`, and hierarchical `.subckt` netlists no
  longer crash Monte Carlo.
- `find_model` reads LTspice's UTF-16 libraries, ranks candidates with a
  length-aware similarity score (short unrelated names no longer outrank
  the real match), and `load_library` on a directory now finds LTspice
  component decks (`.bjt`, `.mos`, `.dio`, ...).
- `validate_netlist` no longer flags `.op` plus a single analysis as
  "multiple analyses", and its advertised `.asc` topology checks (shorts,
  floating pins, dangling labels) actually run.
- `diff_circuit` compares component attributes and no longer reports the
  `.END` terminator as a removed directive; `schematic_from_netlist` keeps
  the first element of a title-less netlist instead of dropping it.
- `.NOISE` signal lookup is case-insensitive with alias resolution, and
  noise and DC sweeps are no longer misclassified as transient analyses.
- Cached editors and results refresh after an in-place file rewrite that
  lands within one mtime tick.
- Packaging: numpy, scipy, pydantic, and anyio are now declared as runtime
  dependencies (installs previously worked only through transitive luck);
  an unused declared dependency was removed.

### Security

- Dependency upgrades for published advisories: `cryptography` 46.0.6 →
  46.0.7 (CVE-2026-39892), `python-multipart` 0.0.24 → 0.0.27
  (CVE-2026-40347, CVE-2026-42561), `idna` 3.11 → 3.15 (CVE-2026-45409),
  and `pyjwt` 2.12.1 → 2.13.0.

## Earlier history

Release history before the first tagged version lives in `git log`; the
`feat:` / `fix:` / `refactor:` prefixes and PR descriptions describe
each change.

