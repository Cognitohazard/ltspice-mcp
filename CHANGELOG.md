# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [Unreleased]

### Added

- `validate_netlist` rejects empty / whitespace-only netlists (error) and
  warns on single-connection (dangling) nodes in `.cir`/`.net` decks. The
  dangling-node pass is scope-aware (`.subckt` bodies counted separately,
  header ports count as a connection), excludes ground and `.global` nodes,
  and counts terminals conservatively — nodes referenced only by expressions
  (`V(...)`/`I(...)`), unrecognized elements, or non-terminal tokens are never
  warned about. `.asc` schematics are exempt (their connectivity lives in
  wires and flags, which this pass cannot see).

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
