# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [Unreleased]

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
