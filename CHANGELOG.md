# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [Unreleased]

### Added

- MCP prompts (workflow starters a host surfaces as slash-commands):
  `characterize_filter`, `run_and_plot`, and `step_response`. Each emits the
  canonical tool pipeline for that task with the circuit path filled in.
- Distribution as a Claude Code plugin (`.claude-plugin/`) and a Claude
  Desktop extension (`packaging/mcpb/`). Both wrap the published package via
  `uv` (the plugin runs `uvx`; the extension is a `type: "uv"` bundle) and so
  require `uv` and a simulator (LTspice or ngspice) on the host rather than
  bundling either.

### Changed

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

### Fixed

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
