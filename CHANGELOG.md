# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adopt [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Until then, minor versions may contain breaking
tool-surface changes.

## [Unreleased]

### Fixed

- An empty `job_id` was taken as absent where a call's sources were counted
  and as given where a case selection was checked. Beside a `raw_path` in
  `analyze_results`, `other_raw` passed unchecked, where a source with no
  `job_id` refuses it, and `runs: {case_ids: [...]}` failed with an internal
  error. `plot_waveform`, `analyze_results`' sources and `inspect` now refuse
  an empty `job_id` as they validate the call, as `jobs` does.
- The `value` recipe refused a stepped `.noise` at one frequency as a table,
  though guide section 'signals' says to read it, as a stepped `.ac` at one
  frequency is read, with no `at`. It now reads each step's point.
- Cancelling a run, or its timeout, could leave its simulator running. psutil
  keeps the process objects of one process scan for the next, and when the
  simulator had taken the process id of a process that had exited since, the
  kill found it by its command line and then skipped it as an id reused. The
  kill and the check whether a case's simulator still runs now read every
  process afresh.
- `run_experiments` exported a schematic with LTspice where it stands, which
  writes `<name>.net` beside it: a netlist the author kept under that name
  was replaced, and the reply said nothing of it. A run now exports a copy
  of the sheet, with the symbol and model files of its folder and the own
  sheet of each hierarchical block beside its symbol, in the server's store,
  and writes nothing beside the caller's files.
- `edit_schematic` and the other sheet readers opened the sheet beside a
  block symbol that gives a `Prefix` as the block's contents. LTspice reads
  such a block as a call to the subcircuit its `Value` names and reads no
  sheet for it, so neither does the server now; a block symbol with no
  `Prefix`, as LTspice draws one, is read from its own sheet as before.
- Several stepped LTspice results refused every read with "Step boundaries
  are unresolved": a stepped `.tf`, a stepped `.noise` at one frequency, a
  `.op` stepped over temperature, and the results file of a stepped `.fra`.
  The first three store one point a step with the stepped value first, as a
  stepped `.op` does; the last holds each step's sweep in turn. Each now
  reads a step at a time, sample for sample as LTspice's own reader reads
  the file. The `.op.raw`
  a stepped LTspice 26 run writes is still refused: it holds one point, and
  nothing says which step's it is.
- An AC results file from a deck with `.options numdgt` above 6 was refused
  ("unsupported double flag for this layout"). Under that option both
  LTspice builds mark an AC plot `double`, but store it as any AC plot, so
  it now reads as the same run without the option.
- An AC pane written by `set_plot_panes`, or by `plot_waveform` with
  `in_ltspice`, lacked the two lines that switch its magnitude and phase
  curves on (`PltMag: 1`, `PltPhi: 1 0`), which both builds write for an AC
  pane they make. LTspice XVII opened such a pane with both curves off, so it
  drew nothing, a trace added by hand included. Every AC pane written now has
  them. A plot settings file written by an earlier version keeps its blank
  pane until `set_plot_panes` writes it again.
- `run_experiments` said a job whose `wait_s` was held to the 120 s cap
  "keeps running" even when it had finished during the dwell. The cap is
  still said; that the job runs on is the receipt's `hint`, given only for a
  job still running.
- `edit_schematic` read each library symbol once per session. A symbol
  redrawn in a library while the server ran kept its old pins in every later
  edit, and a symbol not found stayed not found after it was added, until the
  server restarted. Every tool now reads a symbol again after its file
  changes, and looks again beside the sheet and in the configured folders for
  one it did not find. LTspice's own library, which does not change while the
  server runs, is searched once per name.
- With LTspice XVII and a later LTspice both installed, a server running XVII
  read the later build's symbols when it edited and checked sheets. On
  Windows and under Wine it searched them before XVII's own, and on WSL it
  used them instead. On WSL, with XVII the only build installed, it found no
  symbols at all. It now reads the symbols of the build it runs, as it
  already did for model libraries.
- A Monte Carlo mismatch or a subcircuit mismatch plan sized by a MOSFET
  card that gives `W` or `L` twice with different values (`w=1u W=2u`)
  read one copy without a word; which one the simulator uses is not known.
  Such a card is now refused, naming the copies; copies that agree read as
  their value.
- `run_experiments` accepted a circuit whose file is no circuit (a `.txt`,
  say) and made a job whose every case failed with `unsupported_variant`. It
  now refuses the call before a job exists, with `unsupported_file`, as it
  refuses a path outside the sandbox and as `inspect` refuses such a file.
- The server read a `;` or `$` comment to the end of the whole card, where
  both LTspice builds end it at the end of its line and read the card on
  over its next `+` line. A parameter on a line after a comment, outside a
  `.model` card's parentheses or inside them, was never seen by any check
  or edit, and an edit that wrote the card again dropped it. The server
  also read a `$` anywhere as a comment, where LTspice reads one inside or
  at the start of a word as part of it, so a netlist with PSpice or OrCAD
  net names (`N$1`, `$N_0001`) lost a node on every such card. Comments now
  end at the end of their line, and only a `$` that stands alone begins
  one. A Monte Carlo or mismatch variant reads its `.model` card's nominal
  values the same way.
- A `.model` card edited by the server (a Monte Carlo or mismatch variant,
  a parameter set by a variation) was written again on one line in the
  server's spelling, so a card laid out over continuation lines lost its
  layout and its comments. A value is now set where it stands, a new
  parameter is written after the last one, and a variant's new name
  replaces the name alone.
- The compressed-transient observation (`coarse_compressed_waveform`) reported
  a run whose deck asked for a maximum time step wider than a twentieth of the
  run, though a gap no wider than that step is what the deck asked for. A
  job's deck's maximum step now raises the bar.
- A wrapped device parameter of a device inside a subcircuit, such as
  `v(@xa:xin:mtail[gm])`, was not matched to LTspice's warning that it is
  unknown, so its zero trace read as a value. Every reader of
  `@device[param]` names now takes them apart one way.
- A stopped case's progress named the stepped parameter of a stepped `.ac` at
  one frequency as the axis it reached (`r = 2000`). It now reads the plot as a
  full read does: such a plot has no axis, and the progress gives its points.
- An edit of a sheet on which two parts share a reference deleted all but the
  last of those parts from the file, whatever the edit was. `edit_schematic`
  now refuses to edit such a sheet and names each shared reference and where
  its parts are. Neither LTspice build netlists such a sheet (recorded:
  `connectivity/shared_reference`).
- `add_component` accepted a reference that differs from another part's only
  in case, such as `r1` beside `R1`. LTspice compares references without case
  and netlists neither part (recorded: `connectivity/shared_reference_case`),
  so `add_component` now refuses it.
- `set_component_attribute` with `InstName` did not rename the part. It wrote
  the old name and the new one into the file, kept the part under the old
  name, and reported that the part had no name before. It now renames the
  part, and refuses a name that another part has in any case.

### Added

- The `measurements` recipe reads a stepped run's `.meas` results a step at a
  time, where it refused `step` and `all_steps` and gave only statistics over
  every step. With `all_steps` it returns one row per step, and with `step`
  the row of the step the selector picks, each with that step's
  `step_values`, so an `analyze` block's `group_by` can name a stepped
  parameter. Row n of a stepped `.meas` table is the step of the log's n-th
  `.step` line on both LTspice builds (recorded: `log/meas_step`,
  `log/step_two_params`, `log/step_temp`); a log whose `.step` lines do not
  number its rows is refused. Without either, the recipe still gives the
  statistics.
- The sheet rule `shared_reference`, an error for two or more parts under one
  reference, compared without case. `verify_circuit` reports it under the
  `quality` check, so it needs no LTspice, and `edit_schematic` reports it
  with its other findings.
- The `value` recipe on a stepped `.ac` at one frequency, read from its job,
  gives the frequency the deck names (`.ac list <f>`) as `actual_x`, with
  `actual_x_source: "deck"`: the raw stores no frequency, and nothing said
  at what frequency each step's value was.
- `inspect(kind="results")`'s plots view, and the Python API's
  `RawResult.descriptor` and `plots`, say which kind of stepped result an
  LTspice plot is (`stepped_kind`): each step's sweep in turn, one point a
  step for a stepped `.op`, `.tf` or one-frequency `.ac` or `.noise`, or the
  one point of a stepped run's `.op.raw` that names no step. It also lists the
  variables that hold each step's values (`step_variables`).
- `run_experiments`' deck lint warns of an instance, `.model` or `.subckt`
  card that gives one parameter more than once (`duplicate-param`,
  `linter_version` 9), as `verify_circuit` reports it (`duplicate_parameter`):
  which copy the simulator reads is not known.
- `plot_waveform`, `inspect(kind="results")` and the Python API's `load_raw`
  report a compressed transient left coarse (`coarse_compressed_waveform`) as
  `analyze_results` does, and `analyze_results` reports it for a call whose
  only recipe is `measurements`, since XVII computes its `.meas` results from
  the few points it keeps.
- `plot_waveform`'s `step` takes the `{axis, value}` selector
  `analyze_results` takes, as well as a step's 0-based index, and plots the
  step whose axis value is nearest.
- An argument several tools take has one schema in all of them:
  `plot_waveform` takes `plot_index: null` (the first plot) and refuses a
  negative `run_index`, as `inspect` does, and `analyze_results` takes
  `include: null`, as the attached `run_experiments.analyze` block does.
- The Python API's `measurements` and `device_op_points` return a `LogResult`,
  a `dict` with `observations`, so a stopped case's log says, as `load_raw`
  does, that its results end where the run was stopped (`partial_raw`).

### Changed

- **Breaking:** every argument that names a file is `path`, as
  `run_experiments`, `verify_circuit` and `inspect` already spelled it.
  `plot_waveform` takes `path` for `raw_file`. An `analyze_results` source
  takes `path` for `raw_path` and `log_path`: a `.log` (in any case) is read
  alone, as a `log_path` was, and any other file as a RAW with the `.log` of
  the same name, as a `raw_path` was. `edit_schematic` takes `path` for
  `target`, and its reply names the sheet `path`. A call that sends an old
  name is refused with the name that replaced it. The Python API's
  `load_raw`, `measurements` and `device_op_points` take `path` first; their
  older `raw_path` and `log_path` keywords are still accepted.
- A stepped LTspice result that keeps each step's values in variables of
  its own, a stepped `.op`, `.tf`, or `.ac` or `.noise` at one frequency,
  lists those variables among its steps' values and no longer among its
  signals: the Python API's `RawResult.signals`, `table()` and `compare()`,
  and `inspect`'s signals and table views. Asked for as a signal, such a
  variable is refused with where its values are.
- Refusals and help say where the answer is:
  - a `run_code` NameError hint says each call starts from a fresh
    namespace, so a name an earlier call set is gone;
  - an unknown top-level field this tool takes further down is named
    there first (`run_experiments` takes `path` as `circuits[].path`),
    before the tools that take it at their top level;
  - a sandbox refusal names the directory a relative path is taken from;
  - a `jobs` resume refused for a reused `resume_request_id`, a stale
    parent, a running parent or a case selection names the jobs and cases
    involved and what to do instead;
  - `RawResult`'s docstring, `load_raw`'s and `api.reference()` name its
    members (`signals`, `steps`, `trace(name, step=k)` and the rest);
  - the `variations` description and guide section 'variations' give the
    one-value `assign` list that holds a target fixed in every case;
  - guide section 'fundamentals' says LTspice prints every AC `.meas`
    result in its log as dB and angle, a real one included, or under
    `.options meascplxfmt` as magnitude and angle or real and imaginary
    parts, and that `measurements` decodes all three;
  - a read of one step of an LTspice 26 `.op.raw` from a stepped `.tran`
    or `.ac`, which holds one point and does not say whose step it is, is
    refused with that cause and the remedy (run `.op` under the same
    `.step`), where it said only that step boundaries were unresolved.
- `inspect(kind="symbols")` lists the LTspice build's `sym` folder as the
  `stock` library, instead of the `lib` folder that contains it.
- The equivalence compare's `duplicate_params` names a card that gives a
  parameter twice once, by its reference and the subcircuit it is in
  (`subckt`, null at the top level), where it listed one row for each
  instance of that subcircuit, by the instance's path.
- `inspect` reports a failed query by its failure's own error code, as every
  other tool does, where every one read `error`: `netlist_invalid` for a
  netlist or sheet it cannot read, `symbol_unresolved`, `library_error`,
  `other_raw_not_written`, and so on. The code `error` is no longer emitted.
- `signal_stats` refuses a plot with one point a step, such as a stepped `.ac`
  at one frequency, and a trace of one point, and names the `value` recipe,
  where it returned statistics of one point.
- The `wire_through_symbol` and `text_in_symbol_body` sheet findings test what
  a part draws, its lines, rectangles and circles, as `attribute_text_overlap`
  does, rather than its box. A wire through, or text anchored in, the empty
  corner beside an amplifier's triangle is no longer reported.
- The warning `set_component_value` gave of a subcircuit call's Value (an
  opamp's complexity label such as `Level.2`, or any Value on a part whose
  model `SpiceModel` selects) is the sheet finding `subcircuit_value_slot`:
  `edit_schematic` says it once of the finished sheet however many ops set the
  value, and `verify_circuit` reports it under `quality`. It reads the part's
  element letter from its own or its symbol's `Prefix`, so a transistor or
  diode whose Value after its `SpiceModel` is an area factor is no longer
  warned of.
- `edit_schematic` ops judge nothing themselves, so `warnings` holds the
  sheet's findings alone. `wire_pins` no longer warns that a route passing
  over a pin or touching wiring already on its net is joined there; the point
  is in its `results` entry under `junctions`, as it already was.
- `edit_schematic` reports a sheet finding as the batch's when the batch
  made or changed it, where it reported any finding that named a part or
  point an op named, unchanged or not. A part moved or given a new value
  keeps the findings it had: a text overlap is the same finding while the
  same attribute crosses the same thing, whatever its words, and a floating
  pin while the same pin floats, wherever the part went. Such findings are
  counted in `preexisting`.
- `edit_schematic`'s reply counts the ops it applied (`applied`): `results`
  lists only ops with something to report, so an op without, such as
  `add_directive`, read as one that did not run. `remove_directive` names
  each text it removed, with its kind and place (`removed_texts`), which a
  `regex:` match otherwise leaves unknown.

## [0.7.0] - 2026-10-22

### Fixed

- `plot_waveform` and the `ac_structure` recipe could fail on a frequency
  sweep whose points lie more than a quarter decade apart (`.ac dec 3` or
  sparser, an `.ac oct 1` sweep, or FRA results at one point per octave). No
  slope can be fit on such a sweep, and the search for the magnitude's
  breakpoints failed there. Such a sweep now reports no breakpoints.
- A rendered sheet placed text in different positions from LTspice, so it
  could not show whether text collides. A `Left` window on a part placed
  `R180` or `M0` was drawn starting at its point, where LTspice ends it at
  that point. A plain window on a part rotated a quarter turn stayed
  horizontal, and `Top`, `Bottom` and `Center` aligned the baseline instead of
  the line. All text was drawn at two thirds of LTspice's size in a monospace
  font, and comments and directives ignored their justification. Text is now
  laid out as both builds draw it, based on screenshots of each
  (`docs/TESTING.md`, "How LTspice draws text"). Each justification is
  rotated and mirrored with the part, text is drawn in Tahoma Bold at
  LTspice's size for each text size, a multi-line comment's lines are one line
  apart, and a net label starts or ends at its point.
- `inspect(kind="hierarchy")` told LTspice users to add `.options logopinfo`
  to read a nested MOSFET's operating point, but LTspice XVII ends the run on
  that option. It also wrote a nested resistor's `.save I(xa:xin:r1)` and a
  node's trace in LTspice 26's syntax, which XVII's `.save` refuses; XVII then
  saves everything except the current (recorded: `raw/save_subckt_names*`). On
  LTspice a MOSFET's address now has no `address.save` line, because the
  server asks LTspice 26 for the device table on every `.op` run and XVII
  prints it by default. Names inside subcircuits are now written for the
  session's LTspice, with each instance's X dropped on XVII.
  `simulator: "ltspice:<name>"` selects the spelling for a named build, and
  the reply's `profile.xvii_names` reports which spelling was used.
- A signal inside a subcircuit named by its path (`V(xa.xin.mid)`,
  `Id(xa.xin.mtail)`) resolved on LTspice 26, whose raw file writes
  `V(xa:xin:mid)`, but not on XVII, whose raw file writes each instance
  without its X (`V(a:in:mid)`; recorded: `raw/subckt_names_tran_saved`).
  Every recipe now reads the signal by its path on either build, including
  port currents (`Ix(xa.top)`). The `value` recipe on an operating point used
  a separate lookup that refused the path on both builds, and it now accepts
  the path too.
- `edit_schematic` replies gave `net: null` for a pin on an unlabeled net,
  though LTspice's export names that net (`N001`). The server cannot predict
  LTspice's numbering, so it reads the name from an export. A batch with a
  `compare` exports the committed sheet, and each such pin now also has
  `net_export`, its node in that export. Each part's card is found by the name
  the exporter writes for it, and the card's nodes are the part's pins in
  SpiceOrder. This mapping agrees with the server's own nets on every
  recorded sheet of both builds.
- A device inside a subcircuit has different names in the log's device table
  and in the raw file, and each LTspice build spells both differently
  (recorded: `log/subckt_names*`, `raw/subckt_names*`). LTspice 26 writes
  `xa:xin:Mtail`; XVII writes `m:a:in:tail` in its table and `a:in:Mtail` in
  its raw file. The `operating_point` recipe's `device=` split XVII's names
  wrongly and found no table values for a nested device, and the `value`
  recipe's `x1.m1.gm` shorthand matched only LTspice 26's spelling. Both
  recipes now map every name back to the device's path, so
  `device="xa.xin.mtail"` and `xa.xin.mtail.gm` work on either build. The
  `operating_point` result now includes `device_names`, each nested device's
  name in the table and in the raw file. A raw file holds those devices'
  currents only when LTspice's "Save subcircuit device currents" setting is
  on, and the guide documents this. A device current inside a subcircuit now
  also counts as an active device for the note that explains how to get the
  device table.
- `set_component_attribute` and `add_component` refused a part's own
  `Prefix` as a symbol property, so turning parts into subcircuit calls
  required new symbol files and removing and placing every part again. Both
  LTspice builds apply an instance's `Prefix` on export, in either letter case
  and on any symbol (recorded: `export/instance_prefix`), so it is now
  accepted. It is written in upper case, as spicelib reads it back, and sets
  the element letter used to interpret the part's value. Clearing it restores
  the symbol's letter.
- `verify_circuit`'s export check reported `value_suffix_micro_sign` on every
  export of a sheet with a `u` value. Both LTspice builds write a part's `u`
  suffix as µ in the exported netlist, whatever the sheet contains (recorded:
  `benchmark/nmos_mirror_drawn`), so no change to the sheet could clear it.
  The build that wrote the export reads its µ as micro, so an export's micro
  signs are now reported only when LTspice XVII would read them from a file
  that is not cp1252, such as a UTF-8 export, and so misread them. Each is
  then a warning per value. A netlist passed to `verify_circuit` still gets
  the observation.
- A run row's `status` is its case's status (`produced` once the simulator
  wrote its output), while the receipt's `status` is the job's
  (`completed`). This was not documented, so a caller that selected rows on
  `completed` got none. The output schemas of `run_experiments` and `jobs` now
  list a row's statuses and state that a receipt's status is the job's. The
  guide and `docs/design/mcp_surface.md` list both sets of values.
- A stepped `.ac` at one frequency could not be split into its steps; reading
  a step was refused with "Step boundaries are unresolved". Both LTspice
  builds store such a run as one point per step, with the stepped parameter
  first and no frequency, and name no steps in the log, as for a stepped
  `.op` (recorded: `raw/step_ac_one_freq`). The server now reads it the same
  way as a stepped `.op`, with the parameter values as the steps.
  `analyze_results`' `value` reads a step's one point when `at` is omitted.
  Given `at`, it now tells the caller to omit it; before, it called the run an
  operating point.
- An `edit_schematic` batch that only set values or attributes returned every
  pin of every part it named, in `views.touched` and in
  `wiring.label_only_pins`, though no pin moved or changed net; `view_limit`
  was the only way to keep the reply short. Pins are now listed for the parts
  a batch rewires (places, moves, removes, wires or labels); a value change
  lists none, and its label-only pins are counted under `preexisting`. Each
  `set_component_value` and `set_component_attribute` now reports `changed`
  in `results`, with the old and new value of every attribute it touched.
- `edit_schematic`'s compare exported a copy of the committed sheet, and of
  an `.asc` reference, in an otherwise empty folder, so a symbol saved next to
  the sheet was not found and the export failed, though `verify_circuit`
  passed on the same sheet. Each copy is now staged with the project-local
  symbol and library files under its original's folder, as `verify_circuit`
  stages them for its own export.
- `run_experiments` accepted a circuit outside the sandbox and made a job
  whose every case failed with `path_denied`. The call is now refused at
  submission with `path_denied` and the sandbox guidance, and no job is made.
  A refusal names a relative allowed directory, such as `.`, with the
  directory it resolves to.
- Reading a case that produced no raw file (`api.load_raw`, an analysis by
  job) reported only the case's status. It now reports the case's failure
  code and reason (as its receipt row does), the job, and the log the case
  left.
- A `request_id` reused for a different payload was refused without naming
  the job it had made. The refusal now names that job and tells the caller to
  read it by its `job_id` or to submit under a new `request_id`.
- `inspect(kind="net")` on a netlist kept `gnd` apart from node `0`, although
  both LTspice builds run a deck's `gnd` as ground, so a query for either
  missed the cards on the other. `gnd`, in any letter case, is now node `0`
  there, as on a sheet. The reply's `warnings` state that ngspice's kiltpsa
  mode does not treat it as ground on every version, and recommend writing
  `0` in a portable deck.
- `inspect(kind="net")` on a netlist listed the cards inside a `.subckt` body
  as if they were on the deck's nodes of the same name, and a body card that
  shared a reference with a deck card replaced it. Only the deck's own cards
  are members now.
- `inspect(kind="net")` on a netlist dropped the `warnings` about a deck the
  parser had to guess at (an unclosed `.SUBCKT`, a stray continuation),
  which `components` reports. The net reply now includes them too.
- The server treated a `GND` label, in any letter case, as a net name,
  although both LTspice builds export it as node `0`. The net trace kept it
  apart from the sheet's other grounds, and `add_net_label` and `wire_pins`
  counted it as a name, so grounding a named net with `GND` was refused or
  reported as a short. It is now ground wherever `0` is, and grounding a named
  net with it is allowed, as with `0`.
- The server treated net names that differ only in case as two nets, but
  LTspice makes them one node, so a deck with `OUT` and `out` runs with one
  node. Labels are now compared without regard to case. `out` and `OUT`
  labels on separate wires trace as one net, `net:out` finds an `OUT` label,
  a route between a `VOUT` net and a `Vout` net is drawn, `add_net_label out`
  on a pin of another named net is refused while `OUT` labels a net (as
  `add_net_label OUT` is), and a net with both spellings is not reported as
  shorted. A net whose labels use one name in two cases is reported under the
  name LTspice gives it.
- A straight wire between two pins of one part was reported as left out of
  the netlist even with a net label or a ground flag on it, on its interior or
  at a vertex in line. Both LTspice builds keep such a wire, so the
  `dropped_wire` finding, the net trace's warning and the `wire_pins` refusal
  no longer report it. A label only at one of the two pins still does not keep
  the wire.
- A `run_code` snippet blocked on a run longer than its `timeout_s` lost the
  job. On POSIX the timeout interrupted the snippet, and the reply did not
  name the job it was waiting on. On Windows the timeout killed the worker
  together with the jobs the worker owned. A blocking wait in a snippet
  (`run_experiments` with `wait=True`, or `api.wait`) now ends shortly before
  `timeout_s` and raises `ApiInterrupted`. The reply's `error` names the job,
  which keeps running in the worker, and explains how to read it from a later
  snippet.
- `plot_waveform` refused `signals` given beside `panels`, even when they
  named only signals the panels plot. Such a list is now accepted. A list
  that names a signal no panel plots is refused, and the error names that
  signal.
- `verify_circuit`'s render, without the `raster` extra, set the outcome to
  `partial` even when the caller named no format, because the default format
  is PNG. Falling back from the default is no longer a failure; `render.note`
  and the headline explain why there is no PNG and how to get one. Asking for
  `format: "png"` by name still reports the failure.
- `verify_circuit`'s `value_suffix_micro_sign` observation on an export was
  worded as a fact about the sheet, although the sheet wrote the values with
  `u`. Both LTspice builds write a part's `u` suffix as a micro sign in the
  exported netlist. Its reason now states that the export wrote them.
- `edit_schematic`'s `wire_pins` refused a coordinate endpoint at a net label
  that stands on no wire, so a wire could not end in a label placed first. A
  route also could not reach one of several labels with the same name, or one
  of several grounds, because `net:NAME` refuses those as ambiguous. A
  label's point is now an endpoint like a pin's, and the `net:NAME` refusal
  names one label by its point. Joining two named nets through a label is
  still refused.
- `add_net_label` warned on every label with a name the sheet already had
  ("already labels a net"), though the netlist merges such labels into one
  net as intended. The warning existed only because a later `net:NAME` could
  not tell the labels apart, and that refusal now names one label by its
  point, so the warning is removed. A second name
  for one net is still reported (`net_with_two_names`).
- The results file of an LTspice frequency response analysis
  (`<run>.fra_1.raw`) could not be read, because the raw reader refused its
  layout. The file uses the AC layout with every value a complex double. It
  now reads as complex values over frequency, with the analysis named `fra`,
  and the AC recipes, plots and samples accept it as an AC result. LTspice's
  `gain_N` is the negative of the loop gain the `stability` recipe expects,
  so `stability` on `-gain_1` gives the crossover and phase margin LTspice
  prints in the log (recorded on LTspice 26; XVII 17.0 refuses the
  analyzer).
- `edit_schematic`'s `remove_directive` did not find a directive or comment
  of more than one line given as the text it was added with. A sheet stores
  each line break as a `\n` escape, and the given text was compared against
  the escaped form, so only a `regex:` pattern matched. A line break may now
  be given as a newline, with either line ending, or as the escape. The
  refusal no longer points the caller to `inspect(kind="components")`, which
  does not list a sheet's directives.
- `remove_component` and `move_component` warned about a wire left on a
  former pin even when a later op of the same batch removed or reconnected
  it. A wire left unconnected is now reported only by the finished sheet's
  `dangling_wire_end` finding, which the reply lists when the batch created
  it. The ops' own warnings are removed; they also flagged a wire that still
  met another wire or a label at that point.
- A stopped case's `partial_progress` observation reported
  `raw_present: true` for a raw file the server then removed. Its evidence
  now also has `raw_kept`, which tells whether the raw file stays on disk
  (only a recoverable job keeps a stopped case's run files), and its detail
  states which applies.
- A run that edited a netlist card giving a parameter twice set only one
  copy, and any edit that rebuilt the card dropped its earlier copies without
  a warning. The affected cards were:
  - an instance card, through component assignments, random component draws
    and the mismatch rewrite (`R1 in out R=1k R=1k` varied to `2k` was
    simulated as `R=1k R=2k`);
  - a `.model` card, through random model draws and mismatch variants. Every
    edit of a parenthesised model and every rename rebuilt the card, so a
    draw of `KP` on `NMOS(VTO=0.7 KP=100u VTO=0.8)` also dropped `VTO=0.7`;
  - a `.subckt` opener's defaults, which the mismatch rewrite adds to.

  Every copy is now set, a rebuilt card keeps them all, and removing a
  parameter removes every copy.
- `set_component_value` on a schematic part wrote a `KEY=VALUE` into
  `SpiceLine` when the part already gave that key in `Value` or `Value2`, or
  in another letter case. A Schmitt trigger's thresholds, or a MOSFET's `w`,
  were then named twice in the export. The key is now set where the part
  gives it, with every copy set to the same value, and the sheet's
  `duplicate_parameter` finding names a part that gives a key more than once.
  A key the part gives nowhere is added to the end of `SpiceLine`, which is
  otherwise left as written; before, adding one rewrote the rest of the line
  (`ad=1e3` became `ad=1000.0`, and a `params:` marker was dropped).
- `verify_circuit`'s equivalence compare kept one value per instance
  parameter, so a parameter given twice was compared as one. Two copies of a
  threshold that agreed reported `equivalent: true` with no mention of the
  second copy. It now reports each such parameter in `duplicate_params` and
  treats it as equal only when it is equal whichever copy is read. The
  structural diff compares both copies, so adding a second copy counts as a
  change.
- A run that produced results was listed in a `run_experiments` or
  `jobs(status|wait)` receipt without its `raw` and `log` paths and its
  `simulator_version`, which only `jobs(runs)` and `run_fields` returned, and
  a caller's `budget` took the paths off such rows on `jobs(runs)` too. Code
  reading a run's log from its row got an empty string for the path and read
  no warnings from it, as if the run had logged none. Every run row now has
  the same keys, whatever its status, on every surface and at every budget;
  a budget shrinks the page instead, and `run_fields` still narrows a row.
- An `inspect` net trace or components listing made while an `edit_schematic`
  call was in progress could show that edit's changes before they were
  committed, next to the digest of the file as it still was. Both now read
  the file on disk, so they show only committed changes.
- On Windows, every LTspice run and every schematic export took the keyboard
  focus for its whole duration, because LTspice opens a window even in batch
  mode, and a sweep held the focus continuously. LTspice is now started on a
  separate desktop that the server creates, so it shows no window on yours.
  Starting it minimised or hidden does not help. In measurements, it was
  still the foreground window in about 80% of samples during a run, and in
  none on its own desktop. If LTspice stops on a message box there (LTspice
  XVII raises one for a sheet that starts with a byte order mark), the run or
  export now ends at once and reports the box's text. Before, it waited for
  the timeout or for someone to click the box.
  `[simulator] hidden_desktop = false` (or `LTSPICE_MCP_HIDDEN_DESKTOP=0`)
  starts LTspice on your own desktop as before. Named executables are
  launched the same way. So is the LTspice launch that resumes a recoverable
  experiment; it keeps the same command, working directory, environment and
  timeout, and does not answer any message box. WSL and Wine are unchanged.
- An `edit_schematic` commit removed every data label (`DATAFLAG` record) from
  the sheet it edited and reported a complete edit, because the underlying
  schematic editor skips them when it reads a sheet and never writes them.
  They are now written back, in their order, after the sheet's labels and
  ports.
- A sheet holding a line the schematic editor does not read, such as a bus tap
  (`BUSTAP`) or an empty line, failed `edit_schematic` as an internal error and
  `inspect`'s schematic queries as `internal_error`, naming neither the file
  nor the line. Both now refuse it with the file, the line number and the
  line, also when the line is in a sheet the target loads as a block. Such a
  sheet still cannot be opened for editing, though LTspice 26 and LTspice XVII
  both read one (`export/bus_tap`, `export/blank_line`), and a bus tap
  connects nothing in the netlist they export. `verify_circuit`, which draws
  the sheet with its own parser, left such a record out of the drawing and of
  every check built on it without reporting the omission. It now reports each
  keyword it did not read, how many lines contain it, and the first such
  line.
- `wire_pins` drew a wire that shorted two named nets in two cases, which it
  now refuses as it always refused the basic case. With an end given as
  `net:NAME`, the check for two differently named nets was skipped, so a wire
  from `net:VDD` to a pin on `VSS` was drawn without a warning. A route that
  crossed a wire at the exact point of a label was reported as a plain
  crossing that LTspice leaves unjoined, but LTspice joins two wires that
  cross at a label (the `label_at_crossing` recording), so the route joined
  that net and took its name. Such a crossing is now treated as a contact. It
  is refused when it is on another net and reported as a junction when it is
  on the route's own net.
- `add_net_label` placed a ground flag at a crossing of two named nets
  without a warning. A label at a crossing joins both wires (the
  `label_at_crossing` recording), so the flag shorted the two nets together
  and to ground. A named label or a pin at the same point was already
  refused; ground skipped that check. It is now refused, and the error names
  the point and the two nets. A label at a crossing where at most one wire is
  named is still placed, and the join is reported (`joined_at_crossing`).
  Grounding a net that has a name is unchanged.
- `verify_circuit` and `edit_schematic` reported a wire as passing through a
  part's body whenever it ended on a pin inside the part's body box. About 4%
  of LTspice's stock symbols have such a pin (a frame drawn one unit outside
  its pin column, a pin on the slanted side of a triangle). A wire that ends
  on such a pin and exits the box through the nearer edge along its own axis
  is no longer reported; one that crosses the body or ends inside it still
  is. On LTspice's 4,366 example sheets, findings drop from 1,440 to 136.
- `add_component` and `move_component` could short two named nets without a
  warning. A pin on the point where two wires cross joins them (the
  `pin_at_crossing` recording), and a part placed or moved so that a pin was
  on such a point, between wires of two differently named nets, was written.
  It is now refused, with the pin, the point and the names, and the part
  stays where it was. Where one of the two wires has no name, the part is
  placed and the join is reported (`joined_at_crossing`).
- The bounding box `inspect` and `edit_schematic` report for a part took
  each `ARC` of its symbol as the whole ellipse the arc is cut from, so a part
  drawn with arcs was reported larger than it is. A polarized capacitor was
  reported as 64 by 100 instead of 32 by 64, and an inductor as 4 units wider
  than its coil. A label or a part beside one was then reported as
  overlapping it. `verify_circuit` judged overlap by the arc as drawn, so the
  two disagreed on any sheet with an inductor. Both tools now share one
  calculation that counts only the drawn part of an arc. The boxes of 593 of
  the 6,678 symbols installed with LTspice 26.1.1 change, including `ind`,
  `ind2` and `polcap`. The direction reported for thirteen pins of the `and`,
  `or` and `xor` gates also changes, and those inputs are now reported as
  leaving to the left.

The entries in this group were found by testing the server against files
LTspice 26.1.1 and LTspice XVII 17.0.37 wrote for a fixed set of inputs
(`tests/fixtures/ltspice_recorded`, `docs/TESTING.md`); each has a test
against the recording that showed it.

- Neither LTspice 26 nor LTspice XVII reads a sheet that starts with a byte
  order mark (`export/micro_utf8_bom`, `export/micro_utf16le_bom`). With a
  UTF-8 mark, `edit_schematic` and `inspect` failed with an internal error
  that reported no `Version` line in the sheet. They now refuse the sheet and
  report that it starts with the mark. With a UTF-16 mark the sheet opened
  and an edit wrote the mark back, so the edited sheet still could not be
  exported; an edit now writes it as UTF-16 LE without the mark, which both
  builds read. `verify_circuit` passed either sheet, because its drawing code
  skips the mark; its `quality` check now reports the mark as an error
  (`byte_order_mark`).
- `edit_schematic` and `verify_circuit` each had their own rule for a
  floating pin. Both now follow LTspice's recorded behaviour. A pin on the
  interior of a diagonal wire is connected; `edit_schematic` reported it as
  floating and left it out of `wiring.pins_wired`. Two pins of one part on
  one point do not connect to each other, but `edit_schematic` treated them
  as connected. Where another part's pin is also on that point, only the
  first part's highest pin in SpiceOrder connects to it. The other pin is on
  a separate node, and neither tool reported it. A zero-length wire connects
  nothing.
- `edit_schematic` skipped a part whose symbol is not found in every check of
  the sheet's geometry, without reporting it, so the part's pins were missing
  from the floating-pin check and from the pin counts. It now reports the part
  (`unresolved_symbol`, a new kind among its findings), as `verify_circuit`
  does.
- `verify_circuit` judged two parts to overlap by what each draws, and
  `edit_schematic` by the box it reports, which includes the pins. Both now
  use the box with pins. The two are the same box for nearly every symbol;
  they differ for one with a pin drawn apart from its body.
- The symbol search used two rules, one for `edit_schematic`'s pin geometry
  and one for `verify_circuit`'s drawing and symbol check, and neither
  matched LTspice. It now uses one rule, recorded from both builds.
  `verify_circuit` no longer finds a symbol kept in a folder beside the
  sheet under its bare name, which neither build does; such a symbol is
  drawn as the placeholder and reported as not found. `edit_schematic` now
  finds a symbol whose name includes a folder when the file is beside the
  sheet, as XVII does, and a symbol whose name includes a library folder other
  than the one that contains it, as both builds do. On Linux and macOS it now also finds
  any symbol named with a library folder. The backslashes in the name were
  read as part of a file name on those systems, and the part was left with
  no pins.
- The lint and `verify_circuit` passed a `.meas` whose trigonometric
  functions LTspice computes in degrees. On the default settings of LTspice
  26 and XVII, `sin`, `cos`, `tan`, `asin`, `acos`, `atan` and `atan2` inside
  a `.meas` take and return degrees, while a B source uses radians. For
  example, `atan2(1,1)` is 45, `cos(pi)` is 0.998497, and
  `INTEG V(s)*sin(2*pi*1k*time)` over two periods of a 1 kHz sine is -3.5e-5
  instead of the intended 1e-3. `ph()` in a `.meas` also returns degrees, and
  the per-user setting "Use radian measure in waveform expressions" switches
  both to radians. `run_experiments` now refuses such a `.meas` (lint
  `meas-trig-degrees`, blocking; `linter_version` 8) and `verify_circuit`'s
  `syntax` check reports it as `meas_trig_degrees`. Both name the B-source
  form, whose unit does not depend on the setting. The `vdb()`, `phase()`
  and `group_delay()` refusals now also detect a call written after an
  operator (`2*vdb(out)`) or an equals sign (`WHEN time=cos(1)`,
  `TD={atan(1)}`), which they missed before.
- An AC `.meas` whose result is a negative real number was returned as
  positive. LTspice prints every AC result as a magnitude and an angle, and
  prints a negative `ph()`, `re()` or `im()` as its absolute value at 180°,
  so the phase at the corner of an RC low-pass, -45 degrees, was read as 45.
  A result printed at 0° or 180° is now read as the signed number; any other
  angle is still read as the magnitude. As a result, `db(mag(V(out)))` now
  reads as the gain in dB. `db(V(out))` in an AC `.meas` is LTspice's complex
  logarithm, read as its magnitude, and the guide now recommends
  `db(mag(V(out)))`.
- A `.meas` named `e`, `k`, `pi` or `q` was accepted. Those are constants of
  LTspice's expression engine, so LTspice 26 refuses the whole deck and XVII
  skips the measurement. `run_experiments` now refuses such a deck before it
  runs (lint `meas-name-ltspice`, blocking; `linter_version` 8), and the
  directive check behind `verify_circuit`'s `syntax` check and
  `edit_schematic`'s directives refuses it (`meas_reserved_name`). The error
  relayed from a run that LTspice 26 refused now includes the fix.
- Results from LTspice 26 could not be read for a deck with two or more
  subcircuit instances. LTspice 24 and later write one `Backannotation:` line
  in the raw header for each instance, and the server's check of the raw
  header refused a field that appeared twice.
- Results could not be read for a deck containing `.options plotwinsize=0`,
  which adds a `nocompression` flag to the raw header, or for a `.tf` run,
  whose `Transfer Function` plot the raw reader did not recognise. Both are
  read now.
- On LTspice XVII every `.op` deck run through the server failed. The server
  adds `.options logopinfo` so that LTspice prints the operating point of each
  semiconductor, but XVII prints it by default and stops with "unrecognized
  option" on a deck that sets the option. The option is no longer added for
  XVII or LTspice IV.
- The server could not read an LTspice XVII run stepped over temperature
  (XVII logs `.step temp=-40°C`, with the unit, and the step value did not
  parse), XVII's semiconductor operating-point block (the row check continued
  into the `Date:` line XVII writes after it), or XVII's `.four` results
  (XVII prints `Total Harmonic Distortion: 13.60%(13.61%)` on one line).
- A failed run was reported as a generic `execution_failed` even when the
  cause was known, as for a missing include or library on LTspice 26
  (`File not found.` under the directive), a missing library on XVII
  (`Could not open library file "..."`), and an undefined subcircuit on both
  (26: `This sub-circuit name is not defined.`; XVII gives the calling card
  on the following line). These are now `missing_include` and
  `missing_model`, with the name they concern.
- `edit_schematic` could write a value that LTspice read as a million times
  too large. A micro sign added to a sheet that held only ASCII was stored as
  UTF-8, and neither build reads a sheet as UTF-8. LTspice 26 exports that
  value as `2Âµ`, and XVII reads the same two bytes the same way. The first
  non-ASCII character an edit adds is now written in cp1252, which both
  builds read. A sheet that already held non-ASCII text keeps its encoding.
- A pin or label on the interior of a diagonal wire was not connected to it by
  the schematic editor's net trace and warnings; LTspice connects it, as it
  does anywhere along a horizontal or vertical wire.
- A directive on a deck's first line was checked as a directive by the lint
  and by `verify_circuit`. LTspice reads line 1 as the title whatever it
  contains, so `.param r=2k` there defines nothing.
- `inspect(kind:"hierarchy")` did not treat `gnd` as ground on LTspice. It is
  node 0 there too, except in a subcircuit that names a port `gnd`.
- The `suffix-mega-milli` lint missed a capital `M` followed by a unit,
  although LTspice reads `1MHz` as a millihertz. It is flagged now; `1Meg`
  and `1mil` are not.
- A deck or library saved in a double-byte Windows code page (Japanese cp932,
  Chinese GBK, Korean) lost bytes when the server rewrote it for a run. Five
  bytes have no character in cp1252 (0x81, 0x8D, 0x8F, 0x90, 0x9D) and are
  common in those code pages. A file containing one was decoded with every
  non-ASCII byte replaced, so the staged copy lost comments or `.include`
  paths that the original contained. Every byte of an 8-bit file now has a
  character and is written back as it was read. LTspice reads such a file a
  byte at a time, so the staged copy again matches what the simulator would
  read from the original.
- A `.four` card containing a harmonic or period count, such as
  `.four 1k 5 V(in)`, had the count read as a trace it asked for. When the
  run produced no Fourier table, as on ngspice, an observation reported a
  trace named "5" as requested and missing. Both builds read the number after
  the frequency as the harmonic count, and the counts are skipped now.
- On LTspice XVII no result of a run could be read when the deck's title line
  contained one of those bytes. XVII copies the title into its log, and the
  log was refused as undecodable; `.meas` results failed the same way inside
  the log reader the server uses. Both read the log now.
- The value parser refused values LTspice reads, such as a unit written
  directly after the number (`2Hz`, `3V`, `2ohm`) and the infix forms.
  LTspice reads digits after a scale letter or `R` as the fraction
  (`1k5` is 1500, `4R7` is 4.7, `2M2` is 2.2m, `1Meg5` is 1.5e6) and ignores
  any other letters after the number (`9V1` is 9). Every recorded spelling
  now reads as the number LTspice ran. `8%` is still refused, because
  LTspice 26 refuses it. Where a value must be distinguished from a name
  (comparing two netlists, Monte Carlo, a variation's assignment), `2N2222`
  and `1N4148` are still names.
- `inspect(kind="hierarchy")` read the same infix values wrongly. It read
  `4R7` as 4 followed by a stray 7 and gave 4.07, and read `1k5` as 1000.05.
  A bare element value is now read as LTspice reads it. Inside an expression
  (`{1k5}`, or a `.param`), LTspice's handling of a digit after the letters
  is not recorded, so the value is left unresolved with that reason.
- Three refusals that LTspice 26 prints on their own line (`No analysis
  specified.`, `More than one analysis specified.`, `R1: Resistance must not
  be zero.`) were not extracted, so the caller got a log excerpt and no
  structured error. They are errors now, as XVII's `Fatal Error:` forms were.
- `run_experiments` ran decks that LTspice refuses and reported the refusal
  only after the run. The refused forms are two exclusive analyses in one
  deck, `vdb()`, `phase()` or `group_delay()` in a `.meas` (XVII fails that
  measurement), and `.lib file section`, which LTspice reads as one file
  name. Each is a blocking lint now (`analysis-count-ltspice`,
  `meas-function-ltspice`, `lib-section-ltspice`).
- A stepped `.op` returned its first step only, though LTspice stores every
  step in its raw file. Every step is read now, with its parameter values,
  and the warning that only one step is available is given only when that is
  true.
- A run LTspice refused before it began (a missing include or library, a
  `.lib` section) returned the log reader's "Expected pattern" message for a
  request for its measurements. It now returns an empty measurement table
  with LTspice's errors.
- In a stepped measurement table LTspice XVII prints `0` for a step whose
  measurement failed, where LTspice 24 and later print `failed`, so a failure
  reached the caller as a value. The `measurements` recipe now names each step
  that reads exactly 0 on a run XVII wrote (`measurement_zero_or_failed`).
- `verify_circuit` did not recognise an LTspice XVII export, which names no
  generator, so a UTF-8 micro sign in one was only an observation. XVII copies
  a UTF-8 sheet's bytes into its export and reads the export as cp1252, so that
  value runs as 1 instead of 1e-6. An export that opens with its schematic's
  path and names no generator is now read as XVII's, and the micro sign is a
  warning.
- `inspect(kind="symbol")` gave a symbol's pins but not the attributes an
  instance of it is netlisted with, so a model name such as an A-device's had
  to be read from the `.asy`. It now lists the symbol's `SymbolType` and every
  attribute it has (`Prefix`, `SpiceModel`, `Value`, `SpiceLine`,
  `ModelFile`...).
- A sheet using a block symbol with no sheet of its own (`SymbolType BLOCK`,
  its subcircuit defined in a library, or not yet) could not be opened.
  `edit_schematic`, `inspect` and `verify_circuit` failed with "File ….asc not
  found", though LTspice exports it. Such an instance now opens with its
  subcircuit unresolved, as LTspice reads it, and the sheet can be edited.
- `verify_circuit`'s comparison read a subcircuit instance LTspice exported
  with an added `X` (`Xe` as `X§Xe` from LTspice 24 on, `XXe` from XVII) as a
  different part from the `Xe` a netlist written by hand names, so every leaf
  under it was listed as removed and added again and `equivalent` was false.
  Names that pair only across that `X`, one to one and of the same element
  type, now match; each instance paired that way is listed under `renamed`,
  which is not a difference. The `structural_diff` mode, and an export's
  `diff_vs_prior`, pair them the same way and list them under
  `components_renamed`.
- An arity error reported only `reference_arity 3, candidate_arity 2`, and
  for an instance whose node count disagrees with its own subcircuit's ports,
  those two numbers did not describe the two sides. Each arity error now has
  a `detail` that names the nodes and ports, and `side` for that case. When a
  node is the subcircuit's own name, the detail reports that the card names
  it twice, as happens with a symbol that gives the name as both its value
  and its model.
- The `compare` check of `verify_circuit` reported an arity error for a
  reference that writes a transistor with three nodes, because the export of
  a three-pin symbol writes four. Both LTspice builds run `Q1 c b e QN` as
  `Q1 c b e 0 QN`, so a three-node `Q` card now compares equal to the
  four-node card with its substrate at `0`. A three-node `M` card still does
  not compare equal. With a level-1 model, LTspice 26 refuses it and LTspice
  XVII runs it with the bulk at ground, while the export puts the bulk at the
  source. Its arity error now explains this and tells the caller to write the
  bulk as the fourth node.
- The server now warns about two cases where LTspice runs an 8-bit deck
  differently from how the server reads it. Byte 0x85 ends the line on
  LTspice 24 and later, so the text after it is read as a card
  (`byte-85-ltspice`). A node named with a byte from 0x80 to 0x9F is refused
  by LTspice 26 and saved under a control character by XVII
  (`node-control-byte-ltspice`).
- Under WSL, LTspice was not found, and its symbols not loaded, for a Windows
  user whose profile directory is not ASCII. `%LOCALAPPDATA%` was read from
  `cmd.exe` in the console's code page, which is not UTF-8 and turns a letter
  the code page lacks into a question mark. It is read as UTF-16 now. On a
  Windows system whose display language is not English, cancelling a run
  could raise an error on `taskkill`'s own message after the simulator had
  been stopped.

- `run_experiments` refused valid decks under its default `lint: "block"`, and
  an `.asc` is linted through its exported netlist, so schematics were refused
  too. `model-missing` read the model of `Q2 c b e QN 8` (the area factor of a
  bandgap's ratioed pair) as `8`, of `Q1 c b e QN off` as `off`, and of
  `X1 a b mysub params: R=2k` as `params:`. The netlist reader now takes the
  model before a BJT, JFET or MOSFET area factor or `off`, and before a subckt
  call's `params:`; a model whose name reads as a number (`Q1 c b e 555`) is
  still the model. The same parsing keeps the area factor through a
  `set_component_value` model change and stops `verify_circuit` from
  reporting the subckt name before `params:` as a dangling node.
- The arity check, which counts each card's nodes, refused valid LTspice
  cards. It refused a comma-continued value (`IC=1,2,3`, `tc=0.001,1e-6`),
  which was split at the commas, and the behavioural resistor and power forms
  `B1 a 0 R=...` and `B1 a 0 P=...`. A value given by a keyword
  (`C1 a 0 Q=...`, `L1 a 0 Flux=...`, `V1 a 0 wavefile="in.wav"`) was counted
  one node short. An expression with spaces around its operators
  (`B1 c 0 V = V(a) + V(b)`) is a warning in the arity check, but the linter
  and `verify_circuit`'s `syntax` check both reported it as an error. A
  comma-continued value is now read as one `key=value`, a keyword value
  leaves every positional field a node, and the linter and the `syntax` check
  report each arity issue at the arity check's own severity, so the spaced
  expression is a warning.
  `R=` and `P=` are still refused for ngspice, whose B-source takes only `V=`
  and `I=`.
- A deck whose title on line 1 starts with an element letter
  (`Diode clamp test`, `Bandgap reference`) was refused by `model-missing` or
  the arity check, which read the title as an element. Both simulators skip
  line 1; the lint rules that read elements or values skip it too, as
  `verify_circuit` already did.
- `verify_circuit`'s `syntax` check applied LTspice's rules to every netlist
  (`vdb()` in a `.meas`, `C=` as a capacitor's value) and never ngspice's (a
  zero `.tran` step, `.backanno`). It now checks against the session's default
  simulator, the one `run_experiments` runs on when no simulator is named.
- The netlist parser reported `.ends opamp` as not matching `.SUBCKT OPAMP`.
  SPICE names are case-insensitive, and the match now is too.
- `inspect(kind="model", mode="enumerate")` refused a `query` because it never
  filtered. It now lists only the models whose name contains the query,
  case-insensitively, and echoes the filter it applied; `search` keeps its
  fuzzy match.
- `analyze_results` refused several harmless requests. A source's `label` is
  now optional (it defaults to the job_id or the raw file's stem, with a `-2`
  suffix where two sources would share one); a
  repeated run index, `case_ids` entry, `group_by` dimension or `include.fields`
  path is read once and named in the `hint` (on an attached analysis, in the
  receipt's `warnings`); and `continue` accepts the original request's fields
  resent with it, refusing only a field that differs from the stored request,
  which it now names. A label the caller writes must still be unique.
- `jobs` refused a call with both `job_id` and `request_id`, even when both
  named the same job. Both are now accepted; when they name different jobs
  the call fails with the new code `selector_conflict`, naming the job the
  `request_id` resolves to.
- A value above one of four caps was refused instead of being capped. The
  caps are `run_experiments`' `execution.wait_s` at 120 s, `jobs(wait)`'s
  `timeout_s` at 300 s, `jobs(list)`'s `limit` at 50, and `per_run.limit` at
  100 on `analyze_results` and on an attached analysis. Each now runs at the
  cap and reports it, in a warning on `run_experiments` and `jobs` and in the
  `hint` on `analyze_results`, naming the value used and how to get the rest
  (wait again, or the next cursor). The schemas no longer advertise these as
  a `maximum`, because a strict client refuses a larger value before sending;
  the descriptions state the cap.
- `analyze_results` refused `reduce` without `field` on a multi-field recipe
  (`{metric: "stability", reduce: ["min"]}`), while a keyed recipe's bare
  `reduce` reduced every key. A bare `reduce` now reduces every field the
  recipe reports, with one row per field and statistic, each equal to the
  result of a reduction that names that field; `spec` still needs `field`.
  `edges` reports its transition time as `rise_time` or `fall_time` by the
  edge each row measured.
- `run_experiments` refused a circuit whose id came from its file stem when the
  stem was not a valid id (`amp.v2.cir`, `my amp.cir`: `invalid_circuit_id`) or
  when two files shared a stem (`a/amp.cir` and `b/amp.cir`:
  `duplicate_circuit_id`), without telling the caller that it had written no
  id. The server chose that name, so it now makes it valid (`amp_v2`,
  `my_amp`) and unique against every other id in the call (`amp`, `amp-2`),
  and each such circuit's cases have a new `circuit_id_derived` observation
  naming the id it ran under and why. An id the caller writes is still
  refused when invalid or duplicated, and an unknown `applies_to` id now lists
  the ids in the call.
- `run_experiments` refused two `random` variation entries in one call even
  when their `applies_to` lists named different circuits, so a Monte Carlo
  run of two designs could not share one job. The limit is now per circuit,
  because two `random` entries on one deck have no defined combination. A
  circuit still takes at most one (`multiple_random_variations`, now naming
  the circuit), and entries on different circuits each run with their own
  seed.
- `add_net_label` refused any second name on a net that already had one, even
  a name no other net has, which joins nothing. It now refuses only a label
  that would join two named nets. Such a label uses a name another net
  already has, or is a new name placed where two named nets cross (a label at
  a crossing joins both wires, which the old check missed). A second name for
  one net is placed with a warning that the node now has two names.
  `wire_pins`' refusal to join two named nets pointed the caller to
  `add_net_label` "to merge them deliberately", but `add_net_label` refused
  the same merge. The refusal now names the relabelling that works.
- `inspect(kind="net")` read a bare `at: "out"` as a node name on a netlist but
  refused it on a schematic. On a schematic it now names the net label, as
  `net:out` does.
- `wire_pins` refused a route that crosses an existing wire where neither
  ends, although LTspice leaves such a crossing unjoined (the LTspice 26.1.1
  export in `tests/fixtures/t_junctions/crossing_wires`), so the route joins
  nothing there. The route is now drawn and the op reports a warning naming the
  wire and the crossing point.
- `edit_schematic` refused a dry run, or an op-less read such as paging the
  `preexisting` view, on an existing sheet unless it included
  `expected_sha256`. The token guards against a lost update, which only a
  write can cause, so it is now required only to commit. A dry run returns the
  sheet's current `sha256`, and a token it is given that does not match is
  reported in `observations` rather than refused; a commit quoting that token
  is still `revision_conflict`.
- `edit_schematic`'s `set_component_value` refused any value with a space
  outside braces, quotes, a waveform's parentheses or a `MODEL KEY=VALUE` list,
  so a source's `AC 1` or `DC 5 AC 1`, a BJT's `2N3904 2` or `NPN 8`, and a
  behavioural source's `V=V(a) + V(b)` were refused, although `add_component`
  writes the same values and none of them can reach a node slot. The refusal
  now applies only where a space would split the value into an extra node.
  That is a resistor, capacitor, inductor or subcircuit value, a MOSFET value
  other than a model name and `off`, and a BJT, JFET or diode value other
  than a model name, an area factor and `off`. The element class is taken
  from the symbol's `Prefix`, as LTspice netlists it, instead of from the
  instance name. A waveform followed by a
  parameter (`PULSE(...) Rser=1`) keeps its parentheses in Value and puts the
  parameter in SpiceLine.
- Running a schematic on ngspice replaced every `µ` and `μ` in the exported
  netlist with `u` and deleted every `§`, including in comments, quoted strings
  and include paths, so an `.include` naming a folder with either character in
  it pointed at a file that does not exist. Only text that ngspice cannot read
  is changed now. The `.backanno` card is dropped, and the `§` LTspice writes
  into an instance name (`R§Load`) is removed from that name wherever the
  deck uses it, including `.meas` references. A value's micro sign is written
  as `u` when the deck is staged for the run, for every simulator, without
  touching a path.
- A deck or include that staging had to rewrite, and a case deck a variation
  wrote, was always written as UTF-8. A cp1252 deck, which is what LTspice XVII
  writes, came out with every other non-ASCII character re-encoded, and XVII
  read a `§` in an instance name as `Â§`, renaming the instance. A rewritten
  file is now written in the encoding it was read in, with UTF-8 used only for
  text that encoding cannot represent. A UTF-16 or UTF-32 deck is still
  rewritten as UTF-8, because the edits made to a deck at run time are ASCII
  bytes.
- The guide described what ngspice prints for a sectioned `.lib` under the
  default compatibility mode, but `run_experiments` refuses that deck before
  it runs (lint `lib-section-ngspice`); the guide now states this. It also
  states that a top-level `.meas` on ngspice is skipped and its value is
  absent, and its core no longer tells an ngspice session to put scalars in
  `.meas`. It no longer counts six tools where there are eight, describes the
  ngspice build it was checked on as the reader's, or states that a process
  runs one executable per simulator family.
- The `lib-section-ngspice` refusal of a sectioned `.lib` on ngspice reported
  only that the compatibility mode reads it as plain includes. Because the
  refused deck never runs, the caller had no other information on how to fix
  it. The finding now names `[simulator] ngbehavior = "hsa"` (or
  `LTSPICE_MCP_NGBEHAVIOR=hsa`) and the restart it needs.
- A case stopped for a timeout or a cancel had its partial raw read for
  progress and removed in one step, with the `partial_progress` observation
  recorded only afterwards, so a status read in between found neither the raw
  nor the progress read from it. The observation is now recorded before the
  raw is removed.
- A path containing `..` was refused before it was resolved, even when it
  resolved inside the sandbox. A deck in a subfolder with
  `.include ../models/x.lib` got an error-severity `path_denied` finding from
  `verify_circuit` and from `edit_schematic`'s compare, stating that the
  include resolved outside the allowed roots, although `run_experiments`
  staged the same include. `inspect` and `verify_circuit` also refused
  `path="sub/../x.asc"`. A path is now checked by where it resolves, with
  symlinks followed; a `..` that climbs out of the sandbox is still refused,
  as outside the allowed directories.
- `inspect` model queries refused a `libs` file inside the detected
  simulator's own model library (for example LTspice's
  `lib/cmp/standard.bjt`) under a sandbox that left that library out,
  although a run reads that library when it stages the deck and resolves its
  includes, and so does `inspect(kind="hierarchy")`. These reads now accept
  it too, and
  `verify_circuit`'s compare reads an include into any detected simulator's
  library rather than only the default simulator's, so the two agree on every
  file a model search names.
- An `inspect` model search with `libs` omitted always returned nothing,
  because it searched only libraries loaded through a call that no code makes
  any more. It now
  searches the detected simulators' own model libraries, which are the
  directories a `libs` entry may name, so every `source_path` it returns can
  be read back through `libs` and staged by a run. The first such search
  parses the whole install, in a worker thread so other requests are not
  delayed. The `missing_model` failure hint points there instead of at loaded
  libraries.
- `inspect` model rows differed by route. A search or enumerate naming `libs`
  returned name, type, source_path, ports and params, while a search of the
  simulator's libraries also returned `include_directive`, `device_type` and
  `usage`, and ranked equal scores differently. Every route now returns the
  same row from the same ranking, so a `.MODEL` found through `libs` includes
  its connection order too.
- On WSL a model row's `include_directive` was converted to a Windows path
  with one `wslpath` process per row, thousands on a search of a full install,
  and a library on the Linux side came back as a `\\wsl.localhost` path that
  staging and `verify_circuit` cannot resolve. It now names `source_path` as
  the server sees it, which is what staging reads; staging already passes
  LTspice its staged copy as a Windows path.
- `run_experiments`, `verify_circuit` and the `inspect` net, components and
  hierarchy queries refused `.spice` netlists, the extension xschem and the
  sky130 testbenches write. Every tool that reads a netlist now takes
  `.cir`, `.net`, `.sp` and `.spice`; a run hands the simulator a `.cir` copy
  of a `.spice` deck, so a simulator that needs a known extension still
  reads it.
- The `circuits[].path` description of `run_experiments` told agents to export
  an `.asc` through LTspice first. An `.asc` is accepted and exported
  automatically, and the description now states this.
- A notch whose null fell midway between two sweep samples was reported
  without the under-sampling warning. Those two samples read the same level,
  so the check, which wanted both neighbours of the lowest sample well above
  it, never fired, and `stopband_rejection_db` came back as a finite figure
  (12.96 dB for an ideal notch at 10 points per decade) with no warning that
  it was only a lower bound. The warning now fires when either neighbour is
  more than 3 dB higher, as its description always stated.
- The cap on simulators running at once (`[simulation] max_parallel`) could
  be exceeded by one. A run submitted just as a finished run handed its slot
  to a waiting run was launched together with that run. The cap now holds in
  that case, and a lowered cap takes effect as soon as no run holds or waits
  for a slot, rather than at the next submission.
- An unterminated single quote in a netlist was reported with the hint "add a
  closing \" after the opening quote". The hint now names the quote that was
  opened, and a quote left open inside a braced or parenthesized expression is
  reported at the quote, with the same hint, rather than at the enclosing
  bracket with none.
- `run_experiments` linted a QSPICE or Xyce deck with LTspice's rules in the
  two arity checks that encode them. A capacitor or inductor written with a
  keyed primary value (`C1 out 0 C=1u`) was a blocking `value-keyword-ltspice`
  finding, and a B-source written in LTspice's own `R=`/`P=` forms passed.
  Both rules are LTspice's, so they now apply to LTspice decks only; a QSPICE
  or Xyce deck is checked as an ngspice deck is.
- The `value` recipe of `analyze_results` reported input-referred noise
  (`V(inoise)`, ngspice's `inoise_spectrum`) in V/√Hz even when the deck's
  `.NOISE` input source is a current source, where the density is A/√Hz.
  LTspice declares that trace a voltage regardless of the source type, and
  `value` took the declared unit without reading the deck, so it disagreed with
  `noise_integral` on the same trace. `value` now takes the input-referred unit
  from the deck's `.NOISE` source, as `noise_integral` does, and when there is
  no deck to check (a bare raw path) it gives the same "Could not verify the
  input-referred noise unit" warning.
- `plot_waveform` drew nothing for a `.step` transient whose steps have
  different time vectors, the usual case. Each step's trace was padded onto
  the combined time axis with a gap wherever another step had a sample, so
  every sample was isolated and none was drawn. Each step's trace now keeps
  its own time vector in the chart data, and the chart draws each trace
  through its own samples, breaking it only at its own non-finite values.
  The chart file for such a run is also smaller, because no padding is
  written. Linear axes also printed
  neighbouring ticks with the same label (0.5 ms and 1 ms both as `0.001`);
  they now use one SI prefix per axis and as many decimals as the tick step
  needs.
- A job replayed by `request_id` from a process that did not own it could end
  `interrupted` although it had completed. The replay saved its own copy of the
  record, read while the job was still running, and that write could happen
  after the owner's `completed` one. The owner had exited by then, so the next
  reader found a running job with no owner and recovered it as interrupted.
  This was seen with two scripts detaching the same request. A replay now
  leaves the job's record unchanged in every process. It used to add an
  `idempotent_replay` observation to the record, which every later reader saw,
  including the original submitter. That observation is no longer written;
  the receipt's `replayed: true` reports the replay to the call that made it.
- The configuration file written on the first tool call set every key to the
  default of the release that wrote it, `default = "ltspice"` included. A host
  with only ngspice then logged a fallback warning on every start, and a later
  release's default (`default_budget`, `open_plot`, `timeout` and the rest)
  never applied to a server whose file predated it. Every key is now written
  commented out with its default shown, so the file sets nothing until a line
  is uncommented. A file written by an earlier release keeps its values.
- On Windows, a job record read while its running job rewrote it could read as
  missing, because opening a file at the instant a rename replaces it briefly
  fails with access denied. The server then created a second job for a
  repeated `request_id` instead of replaying the first, and a lookup, a
  listing or a cancel could report a live job as not found. Reading a job
  record now retries that denial on the same short schedule that writing one
  already used.
- `analyze_results` reads each source's RAW, log and console files in a
  worker process with a deadline and a memory limit, and analyzes the
  content it read. Sources that read the same files share one capture,
  whatever their labels and plot selections. A call that continues a result
  set reads each source again and compares it with the capture, including a
  log or console file that has since appeared or disappeared. A rewrite that
  kept both size and timestamp is still reported as `source_drift`, and a
  capture that runs past its deadline still reports `analysis_deadline`. A
  result set keeps the content hashes of its first capture.
- A `run_code` call that arrives while the worker is still starting is answered
  `busy`, as one arriving while a snippet runs already was. The call in
  progress reserved the worker only after it had booted, so a second call
  during a slow boot read the worker's replies alongside the first and
  failed.
- A stopped case's kill grace now ends when its last wait expires instead of
  when the loop clock reads past the deadline. asyncio fires a timer up to one
  clock resolution early (15.6 ms on Windows), so the clock could read earlier
  than a deadline the wait had reached, and the server then killed the
  process again.
- A `wire_pins` waypoint on an existing wire, or a route passing through
  another wire's end or a lone net label, joined that wiring without a
  warning. The crossing check only looked at crossings strictly inside both
  wires, but LTspice joins anything that ends on a wire. The contact is now
  refused when it would merge a net neither endpoint is on, and the refusal
  names the wire and the `{x, y}` endpoint that makes the same T junction
  deliberately. A contact with a net the route already joins is reported as a
  warning. Passing over a pin already wired to an endpoint is reported the
  same way.
- `add_net_label` warned that LTspice would ignore a label placed on a wire's
  interior. LTspice names the wire's net after it, as the net trace and the
  post-edit `dangling_label` check already assumed; the warning now fires only
  for a label that touches no wire and no pin.
- `wire_pins`' own advisories (a long run, a wire through a component's
  bounding box) appear in the `edit_schematic` response's `warnings` again.
  They had not reached the response since the op replaced the standalone
  tool.
- The refusal of a route crossing a wire where neither ends no longer states
  that the crossing would join them; LTspice leaves a plain crossing unjoined.
- Under Wine, `inspect(kind="capabilities")` reported `wine` as the LTspice
  executable, because it read the first word of the launch command. It now
  reports the simulator itself, e.g. `.../LTspice.exe`.
- `verify_circuit`'s `managed` export is documented to leave the
  caller's files untouched, but it created a `.ltspice-mcp/locks/` directory
  beside the schematic, as did every `edit_schematic` call, dry runs included.
  Circuit locks are now stored in the per-user home (see Changed).
- A relative `.include` in a compared export was resolved against the scratch
  copy the export was made from, not the schematic's folder. With the store
  outside `allowed_paths` it was refused and reported as a `path_denied` error
  on the author's own library. `verify_circuit` and `edit_schematic`'s
  reference check now resolve each side's relative includes from the folder
  of the file the caller named, as LTspice does.
- The check that refuses two variation entries writing the same field
  (`overlapping_assignment`) now compares component references without
  regard to case. It also refused an `assign` of a parameter followed by a
  `random` entry for it whenever the call also edited an unrelated nested
  device; that order is now accepted, as it is without the nested edit.
- On native Windows, an ngspice run with a `pdk_native` variation stages the
  pinned model files under short file names, because ngspice on Windows
  cannot open the long include paths of the full PDK layout. The staged
  copies keep the originals' content, and the run's records still name the
  original files.
- `execution.run_timeout_s` is now the actual per-case limit. Every simulator
  process was also capped at a fixed, undocumented 600 s inside spicelib, so a
  case given a longer timeout was killed at 600 s and reported as a generic
  failure with exit code -2. spicelib's limit is now the case's own timeout
  plus the kill grace plus 60 s. It ends the process spicelib launched when
  the server's kill of the case's processes is not confirmed, and it is
  capped at 4,000,000 s, below the longest wait Windows can express (about
  49.7 days). When spicelib does end a run, the failure names its
  `TimeoutExpired` instead of a bare -2.
- Stopping a case makes up to five attempts to kill the case's processes
  inside the kill grace rather than one. A stop that arrived between launch
  and the simulator's spawn found no process, and the simulator then kept
  running with its concurrency slot reserved. The kill also reaches a
  simulator this server launched under a name it does not know, such as a
  wrapper or launcher script and its child, matched by the run deck path in
  its arguments.
- A case stopped for `run_timeout`, `job_deadline` or `cancelled` keeps the
  killed run's diagnostics again, as the 0.6.0 notes stated;
  `run_experiments` had dropped them. Its failure evidence includes the limit
  that applied, the simulator's `exit_code`, the log's classified cause as
  `log_failure_code` when there is one, and `log_excerpt`. The message names
  the timeout or deadline value; a cancelled case no longer reads "stopped
  because cancelled elapsed". `kill_unconfirmed` rows include the stop
  reason, limit and kill grace.
- Symbols placed at `M90` and `M270` had those two orientations swapped. LTspice
  rotates a mirrored placement first and then mirrors it, so `M90` takes a
  symbol point `(x, y)` to `(y, x)`. The server applied the mirror first. On a
  sheet LTspice wired correctly, `inspect` reported the pins in the wrong place
  and `verify_circuit` reported floating pins, dangling wire ends, wires through
  symbol bodies and dropped wires. `M0`, `M180` and every `R` orientation were
  unaffected. **Sheets built by earlier versions with `add_component` or
  `wire_pins` at `M90` or `M270` put the wires at the wrong pin positions, so
  those parts are miswired when LTspice opens the sheet.** Rebuild or rewire
  those parts.
- The `value` recipe was described as evaluating an expression. It reads one
  trace at the sample nearest `at`, without interpolation, and `at` is
  required when the run's axis has more than one sample. Its argument
  descriptions and reference entry now state this.
- The guide and the bench-craft skill told readers to take open-loop gain
  from `V(out)/V(inp,inn)`, which no recipe accepted. It now works (see
  Added).
- A signal that reads as an expression (`V(a)-V(b)`, `2*V(out)`) now fails
  with a one-line Python example of the trace math. The error names
  `run_code` as the place to run it, or the `ltspice_mcp.api` library when
  `run_code` is turned off. The guide has a new section on naming signals and
  on trace math.
- An edit to `[security] allowed_paths` now takes effect on the next call
  everywhere the server checks a path against the sandbox. The capabilities
  report, the `spice://config` resource, netlist resource reads, the
  hierarchy query, raw and log source resolution and deck staging used to
  read whatever list the last path check had loaded, so a capabilities query
  could list the old roots in the same batch that read a newly allowed file.
- Every sandbox refusal gives the same guidance in its structured `hint`. It
  lists the allowed paths and the config file and key that widen them, and
  states that the file is re-read on the next call. This applies to `inspect`
  items, `verify_circuit` findings (including a refused include),
  `run_experiments` case failures, `jobs` errors, `analyze_results` sources,
  and a refusal returned as a tool error, which now includes it in
  structuredContent as well as text. A refusal raised through the Python API
  attaches it as a note on the `PathSecurityError`, so it appears in the
  traceback. With
  `LTSPICE_MCP_ALLOWED_PATHS` set, the guidance names that variable instead,
  since it overrides the file.
- A Python API session opened with an explicit `Api(allowed_paths=...)` keeps
  that sandbox. It used to be replaced by the config file's list, or by the
  default sandbox, as soon as the working directory's `ltspice-mcp.toml`
  appeared or changed, which a server session in the same directory causes by
  writing its default config on its first tool call. A refusal in such a
  session names the argument rather than the file.
- A micro-sign scale suffix (`23µ`) no longer changes value depending on the
  LTspice version that reads the deck. Case decks, and staged decks whose
  references were rewritten, are written as UTF-8, so a `µ` reached the
  simulator as bytes `C2 B5` whatever the source used; LTspice XVII decodes a
  deck as cp1252, read `Âµ`, and ran `23µ` as 23 without a diagnostic. Every
  staged deck and include, every case deck, and every per-case include copy
  now writes a `µ`/`μ` scale suffix as `u`, for every simulator. Comments,
  names, quoted strings and include paths are left alone, a file with no such
  suffix keeps its original bytes, and the schematic is never touched. Each
  file changed this way is reported as a `micro_sign_folded` observation.
- `inspect(kind="components")` on an `.asc` compares `prefix` without regard
  to case, as it already did on a netlist. The prefix went to spicelib as
  given, so `prefix: "r"` returned no components on a sheet full of resistors.
- The `wire_pins` op's `waypoints` description no longer tells callers to omit
  waypoints and "let the router pick the elbow". There is no router. With no
  waypoints, the two ends must share an x or a y, or the route is refused as
  diagonal.

- PNG renders no longer need the native Cairo library. The `raster` extra
  installs `resvg-py`, which ships prebuilt wheels and loads no other library,
  in place of `cairosvg`, which loads Cairo when imported; on Windows that
  meant installing a Cairo runtime by hand and putting it on `PATH`. A server
  with `cairosvg` and not the extra still uses it. When `cairosvg` is the only
  rasterizer and Cairo is missing, the `render` failure, `render.note` and the
  hint now report that, instead of blaming the extra, and offer the extra or
  the Cairo install for the server's platform. The server looks for a
  rasterizer once per process, so a fix takes effect after a restart, as each
  remedy states.
- A `verify_circuit` render with `delivery: "inline"` or `"both"` that came
  out as SVG, because SVG was asked for or no PNG could be made, reported only
  `returned_inline: false`, and an agent that cannot read files had no way to
  see the drawing. It now returns the SVG markup in `render.svg`, where a
  client that reads only the structured payload finds it.
- A symbol saved in the same folder as its `.asc`, where LTspice looks first and
  where a hand-drawn subcircuit symbol is usually stored, was not found for pin
  positions. `inspect(kind="symbol")` and rendering found it, but
  `inspect(kind="net")` on its pins failed with "symbol not found",
  `inspect(kind="components", detail="full")` left out its pins and bounding
  box, and `add_component`, `wire_pins` and the other ops that take a pin
  refused it. Pin positions now come from the sheet's own folder first, then
  the libraries, so a local symbol takes precedence over a library symbol of
  the same name, and two sheets in different folders each keep their own. A
  redrawn local symbol is read again. A `base="blank"` build looks beside its
  target, not beside the temporary template it starts from.
- The server's default response budget added "presentation was reduced" to
  any response over it, even when its trim removed nothing, and receipts and
  `jobs` then told the caller to "ask again with a larger 'budget'", a field
  the caller had not set. The note is now written only when the trim emptied
  a field with content, and it names that field. Under the server default it
  no longer refers the caller to a budget; `analyze_results` instead reports
  that each row still names its source and that `include.provenance` keeps
  `source_hashes`. `inspect` no longer repeats the note in its `hint`.
- Hint and observation text that restated the structured fields is removed.
  Every receipt's hint no longer repeats the `progress` counts, and a finished
  receipt for ten or more cases no longer includes a suggestion to use the
  Python API (the server's instructions introduce it). A clean terminal
  receipt's hint is one status line. `plot_waveform` drops `plot_written`,
  `step_axis_unioned`, `open_skipped` and `widget_delivered`, which repeated
  the reply's own counts, `opened` and `delivery`, and reports
  `phase_unwrapped` only when a summary's phase lies outside ±180 deg, where
  it differs from the wrapped angle. `edit_schematic` lists in `stages` only
  the stages that did not complete (the no-op `stage_assets` is removed), and
  its commit hint no longer restates the `wiring` counts.
- A `run_experiments` call repeating the `request_id` of one still being
  submitted (a client retry, or two calls racing) could find the job's record
  before the server had started tracking the job, then wait on a copy read
  from disk that was never updated, so it replied only when its whole
  `wait_s` ran out, however early the job finished. A repeat of a job another
  server process runs read its record once and waited the same way. The
  server now tracks the job before it writes the record, so a repeat waits on
  the running job and replies when it finishes, and a repeat of another
  process's job re-reads its record until it finishes, as
  `jobs(action="wait")` does.
- A `run_experiments` repeat of a job whose record names this process as its
  owner while nothing in the process was running it (a reused process id, or
  a crash that stopped the job's runner) still waited out its whole `wait_s`
  on a copy of the record that would never be finished. It now replies from
  the record at once, as `jobs(action="wait")` already did. A wait now
  applies only to a job this process runs, never to a copy of a record read
  from disk.
- On Windows, a result read that was stopped (it timed out, was cancelled or
  failed) was reported as fully ended while processes the result parser had
  started were still exiting. Windows counts the processes in a job object
  as gone as soon as it is asked to terminate them, and the server used that
  count as confirmation. In measurements, every process of a terminated job
  object was still running when the count read zero, for 2 to 50 ms
  depending on the memory the processes held. The read's scratch directory
  was removed next, and a process still exiting can hold it open. The read's
  processes are now confirmed gone only once each of them has exited.
- An unknown `[tools] listing` (or `LTSPICE_MCP_TOOL_LISTING`) value was
  logged as falling back to `full`, though the value was ignored and the
  listing already in effect, `compact` by default, was kept. The warning now
  states that the value is ignored.

### Added

- `verify_circuit`'s layout check, and `edit_schematic`'s reply, report a
  part's name or value drawn across a wire, across another part's drawing or
  across other text (`attribute_text_overlap`). The layout scan used to leave
  attribute text out. The text is placed where LTspice draws it, with its
  extent estimated from the font's character widths. A bounding box's empty
  corner, or two lines that only touch, are not counted.
- `analyze_results`, `run_experiments`' attached `analyze` block,
  `plot_waveform`, `inspect(kind="results")` and the Python API's `load_raw`
  read a case's other raw by job and case. `other_raw: "fra_1"` reads its
  frequency response analysis results and `other_raw: "op"` reads a
  transient's operating point; before, only a path to the file could. A case
  whose run wrote no such raw fails with code `other_raw_not_written`, which
  `analyze_results` reports as a missing case.
- `run_experiments` takes `execution.keep_partial_raw`. With it set, a case
  stopped by its run timeout, the job deadline or a cancel keeps its raw
  file, cut to the points that can be read in full, instead of deleting it.
  `analyze_results`, `plot_waveform`, `inspect(kind="results")` and the
  Python API read it, each with a `partial_raw` observation that reports
  where its values end. An LTspice raw file keeps the points its header
  declares, as far as LTspice's own reader reads it, because LTspice XVII's
  records past that count go back in time (recorded: `raw/tran_killed`); an
  ngspice raw file keeps every complete record. The count is patched in place
  and the file truncated, so nothing is copied. The failure row's evidence
  describes the kept raw file's contents as `partial_raw`, and the run row's
  `raw` is its path. The flag cannot be combined with `recoverable`.
- `RawResult.compare(other, signals=None, step=0, other_step=None)` in the
  Python API reports how far another run's traces are from this run's. For
  each signal it returns the largest difference, its share of the run's peak
  and where it occurs, and it lists the signals only one run has. It pairs
  signals by name ignoring case and by their path inside subcircuits on
  either LTspice build, and pairs steps by their parameters. It compares over
  the span both axes cover and reads the other run by interpolation where
  its axis differs (in log frequency, magnitude and phase on an AC run), so a
  compressed LTspice raw file is read along the straight lines between its
  points. Before, confirming that a new model library reproduces archived
  runs required writing this comparison by hand.
- `analyze_results` reports, with the observation `device_op_log_precision`,
  when device operating points it read came from LTspice's log table, which
  prints every value to three significant digits. It is reported once per
  recipe, whether the recipe is `operating_point` or a `value` of `m1.gm`.
  Recordings on both builds show that `.options numdgt=7` leaves the table
  at three digits and that neither build stores a device parameter in the
  raw file instead (LTspice 26 refuses `.save @m1[gm]`; XVII ignores it).
- A run row's `other_raws` maps each kind of other raw file the run wrote
  beside its raw file to its path. The kinds are `op`, a transient's
  operating point (`.op.raw`), and `fra_1`, a frequency response analysis's
  results (`.fra_1.raw`); a row did not name these before. A run with no raw
  file lists none, and a stopped run's other raw files are removed with its
  raw file.
- When a recipe reads its samples, `analyze_results` reports a compressed
  LTspice transient stored in few points as the `value` observation
  `coarse_compressed_waveform`. The observation gives the points stored, the
  span, and where the widest gap between two stored points lies, and it is
  reported when that gap spans more than 5% of the run. Recordings on both
  builds (`raw/compress_*`) show that LTspice XVII compresses by default. It
  stored a 1 mV RC charge over 5 ms in 13 points where the same run
  uncompressed stored 1078, and its own `.meas` results moved. LTspice 26
  stored every point.
- `RawResult.value(name, step=0)` reads a plot of one point per step, such as
  an operating point, as a number (a float, or a complex for a complex trace).
  A trace of more points is refused rather than cut to its first.
- `api.device_op_points(job_id=..., case_id=...)` (or `log_path=`) returns
  the semiconductor operating points LTspice prints in an `.op` run's log, by
  device and parameter (`ops["m1"]["gm"]`), and states that the table holds
  three significant digits. A device inside a subcircuit is keyed by its path
  (`ops["xa.xin.mtail"]`), the same on both builds, where the table names it
  `xa:xin:mtail` on LTspice 26 and `m:a:in:tail` on XVII (recorded:
  `log/subckt_names*`). The guide's Python section shows a run's
  operating point, device table and an AC trace kept in a JSON file.
- `edit_schematic` has a new op, `move`, which shifts any mix of parts,
  wires, net labels, directives and comments by one offset (`dx`, `dy`).
  Parts are chosen by reference, labels by point, wires by their two ends,
  and texts by exact text or `regex:`, and a `region` also selects everything
  inside it. With `drag`, attached items move too. These are the ends of
  other wires, and the labels and grounds, on a moved pin or wire end or
  along a wire that moves whole. A wire dragged off its axis is reported as
  diagonal. A move that would join two named nets is refused before anything
  changes, and the result's `moved` lists what moved. Before, only one part
  could be moved, with its wires left behind, and a directive had to be
  removed and added again to move it.
- A new `duplicate_parameter` finding is reported by `verify_circuit`'s
  quality check and on every `edit_schematic` reply. On a netlist it names an
  instance, `.model` or `.subckt` card that gives one parameter more than
  once, in any letter case. On a schematic it names a part that gives one
  more than once across `Value`, `Value2`, `SpiceLine` and `SpiceLine2`,
  which LTspice writes onto the one netlist line. Which copy LTspice reads is
  not recorded, so the finding lists the copies and makes no claim about
  their effect. Before, only an equivalence compare against a reference
  netlist showed a repeated parameter. `inspect`'s full component listing of
  a netlist also lists such a key's copies, under `repeated_params`.
- `edit_schematic` and `verify_circuit` report which sheet rules ran.
  `rules_run` in each reply names every rule that ran with how many findings
  it had, zero included, so a reader can tell a rule with no findings from a
  rule that did not run. In an edit's reply the counts cover the sheet as it
  now is, so a rule whose findings all predate the batch still appears. In a
  check's reply they cover the rules of the checks that ran, counted before
  the cap on how many findings of one rule a reply lists.
- `edit_schematic` has a twelfth op, `set_plot_panes`, which writes the
  waveform panes LTspice opens for a sheet into the `.plt` beside it. It
  writes the traces of each pane, top to bottom, for the `tran`, `ac`, `dc`
  (a DC sweep) or `noise` analysis, with optional log or dB scales. It
  replaces that analysis's panes, keeps the file's other analyses, and
  reports the panes it replaced so they can be restored. The file is written
  in the same transaction as the sheet, under its own file lock. The format
  LTspice writes and reads was recorded on LTspice 26.1.1 and XVII 17.0.37
  (new `plot-settings` cases in `tests/fixtures/ltspice_recorded`), and the
  op writes that format. It writes UTF-16 LE without a byte order mark and with
  LF line ends, the form XVII writes and both builds read. LTspice 26 writes
  UTF-8, and when XVII saves over a UTF-8 file it appends the old bytes after
  its own, so UTF-16 is the form that neither build's save damages. Panes are
  listed bottom first in the file, and a trace is read only up to its first
  space, so a trace with whitespace is refused. New panes keep the waveform
  grid when every pane they replace had it. A pane that LTspice saves with
  its grid setting on has a `GridStyle` line, and a pane without one has no
  grid whatever the setting. The recording script
  (`scripts/record_ltspice_fixtures.py`) has a new `plot` case kind that
  runs a sheet in LTspice's window, builds panes with the window's own menu
  commands and saves them. The script also runs every case with the
  waveform grid setting removed from its copy of the settings, as it does
  the settings that change a run's results.
- A new `analyze_results` recipe, `tone`, reads the amplitude and phase of a
  transient signal at one frequency you give. As `thd` does, it cuts the
  window to the whole periods that fit from its start and resamples it onto
  an even time grid by linear interpolation, so each sample counts by the
  time it covers. It returns `amplitude`, `phase_deg`, `dc` and
  the sine and cosine coefficients, each reducible. The phase is relative to
  `sin(2*pi*f*t)`, with `t` the simulation's own time, so it does not change
  with the window and two signals' phases can be subtracted. A warning names
  a sampling step too wide for straight lines between samples to follow a
  sinusoid. The same calculation is `analyze_tone` in the Python API. The
  guide's `signals` section now names it and LTspice's `.four`, which the
  `summary` recipe reads. LTspice 26 and XVII print a `.four` phase in
  different conventions (a `SINE(0 1 1k)` source's fundamental at 90 and 0
  degrees), so phase should not be read from `.four`.
- The warning for an ngspice run that skipped `.four` names the `tone` and
  `thd` recipes, which read harmonics from the raw file it still wrote,
  instead of stating that Fourier and THD are unavailable.
- A schematic that is open in LTspice is now updated by `edit_schematic`
  (Windows, LTspice 26.1 or later). LTspice reads a sheet only once, so an
  edit to the file did not appear in the window, and the window's next save
  wrote the old sheet back over it. The committed sheet now appears in the
  window at once, as one step of its undo history, and the reply lists the
  window under `open_in_ltspice`. An edit is refused as
  `open_window_differs`, with nothing written, while the window holds a
  different sheet from the file (unsaved changes, or a file that changed
  after it was opened). The comparison uses content, because LTspice rewrites
  a sheet's text on opening it. This uses the MCP bridge that LTspice ships,
  which starts an LTspice of its own when none is running. The server gives
  the bridge an LTspice path that does not exist, so that launch fails, and
  starts the bridge on its hidden desktop in a Windows job object that ends
  with the session, so anything it does launch is not visible and does not
  keep running. Text containing a NUL character, which makes LTspice stop
  responding, is never sent to a window.
  `[schematic] sync_open_window = false` turns it off, and
  `inspect(kind="capabilities")` reports it under `open_window_sync`.
- `inspect(kind="open_in_ltspice")` lists the sheets and netlists open in
  LTspice and marks the one in front, so the server can resolve a request
  about "this circuit". For a sheet inside the sandbox, the listing includes
  its `sha256`, whether its window differs from the file, and where.
  `only_in_window` and `only_in_file` list the entries that differ, so a part
  whose value was changed and not saved appears with both values.
- `inspect(kind="simulator_docs")` lists and reads the reference documents
  LTspice 26.1 installs (keyboard shortcuts, menus, the schematic format,
  `.MEAS`, the waveform viewer), in sections paged at their headings. They are
  read from the install and not packaged.
- `plot_waveform(in_ltspice=true)` opens the run in the LTspice window that is
  already running. For a transient, an AC run, a DC sweep or a noise run it
  also draws the plotted traces, by writing the plot settings file that
  LTspice loads beside the results, in the form `set_plot_panes` writes, and
  keeping the file's other analyses. Settings a person saved from LTspice are
  left unchanged; the reply lists the traces they hold (`saved_panes`) and,
  for a sheet's run, names `set_plot_panes` as the way to replace them.
- `verify_circuit(in_ltspice=true)` opens the checked sheet or netlist in the
  LTspice window that is already running and puts it in front. A sheet the
  window already had open is shown as the window holds it, and the reply
  reports when that is not the file that was checked.
- `in_ltspice` on `verify_circuit` and `plot_waveform` starts LTspice when no
  LTspice window is open; before, it reported that none was open. LTspice
  opens visibly and takes the keyboard focus, as a request to show something
  there requires, and the reply's `ltspice` block reports `started: true`.
  `[schematic] start_ltspice = false` (`LTSPICE_MCP_START_LTSPICE`) keeps the
  old behaviour. Nothing else starts a visible LTspice; an edit to an open
  sheet and `inspect(kind="open_in_ltspice")` still start nothing.
  `inspect(kind="capabilities")` reports the setting under
  `open_window_sync.starts_ltspice`.
- `plot_waveform(in_ltspice=true)` opens a sheet's run from the sheet, so
  LTspice links the plot to the sheet and a click on a net plots it, as after
  a run made in LTspice. The run's results and log are written beside the
  sheet under its name, replacing any there, and the sheet's own Visible
  Traces command opens them without a second simulation. The reply's
  `ltspice` block names the `sheet`. Nothing is written when no window is
  open or when the sheet is outside the sandbox. A plot of the sheet's
  results that LTspice already has open would keep showing the run it read,
  so it is closed and opened
  again with the new one (`reopened: true`); traces added to it by hand are
  not kept. A sheet that LTspice opened before it had results is closed and
  reopened too (`sheet_reopened: true`), but only when LTspice's copy matches
  the file. For a sheet the server edited while it was open, LTspice asks
  whether to save changes; the server checks the copy again and answers No
  only if it still matches. A sheet with unsaved changes stays open, and the
  run opens as a separate plot (`tied: false`). A netlist's run opens as a
  separate plot, and is also closed and reopened if it was already open.
  Nothing is closed while LTspice is showing a dialog.
- The server is also published as `osic-mcp` (open-source IC), a third alias
  beside `circuit-mcp` and `ngspice-mcp`, with the same program at the same
  version.

- `execution.recoverable=true` makes an experiment recoverable. The server
  saves copies of the circuit's validated inputs, the simulator's startup
  settings, the seeds and a record of each attempt, so a resumed case runs
  with the same inputs. `jobs(action="resume")` makes a new job that reuses
  each completed case whose saved results still match their recorded hashes.
  It runs again each case stopped by a server restart or the job deadline,
  and failed or cancelled cases when asked (`retry_failed`,
  `retry_cancelled`), all under the executable the job recorded. Recovery
  supports ngspice, run with a startup file the server provides and the
  user's startup file turned off, and LTspice 26.0.2 on native Windows, the
  one build whose startup behaviour was checked. Other builds and platforms
  are refused before launch.
- RAW, log and console files are read in worker processes with time, memory
  and output limits on Linux and native Windows. Analysis can select a RAW
  plot or read results that exist only in the log, including the tables the
  simulator prints for `.tf`, `.sens`, `.pz` and `.disto`. A raw layout the
  reader does not support, or a file whose simulator format cannot be
  determined, is reported as such instead of being guessed. Reading results
  is refused on macOS, because the workers' memory limit and the cleanup of
  their child processes have not been tested there.

- The guide is now a short core plus topic sections and task playbooks, and
  every interface can read it. `inspect` has a `guide` query kind. With no
  `section` it returns the core (how to work with the server, when to use
  Python or the tools, the rules that cause silent errors), which ends with an
  index of the sections, grouped as topics and tasks; `section` reads one
  (`"ltspice"`, `"bench-craft"`). The Python API has the same text as
  `Api.guide(section=None)`, a static method that needs no engine session, and
  `python -m ltspice_mcp.api guide [SECTION]` prints it without starting the
  engine. The resources are `spice://guide` (the core) and
  `spice://guide/{section}`. A client that could not read resources had no way
  to reach the guide before.
- The guide has a section on working in Python. It covers where `api` comes
  from, the six operations as methods, waiting on jobs, raw traces and
  measurements, the analysis primitives, errors, detached jobs, and running a
  second LTspice build. The design document that contained this material is
  not in the package, so a model could not read it.
- The first tool reply of a session that has not read the guide includes a
  one-time reminder to read its core, on the text channel and in the
  structured `hint`. Reading the guide through `inspect` or a `spice://guide`
  resource stops the reminder.
- `run_experiments` accepts `execution.simulator: "qspice"` and `"xyce"`.
  The field accepted only LTspice and ngspice (and their named executables),
  so QSPICE and Xyce could run only as the server default, after a config
  change and a restart. Its pattern now accepts all four families, each with
  named executables of its own, and a name is accepted when the family is
  detected and this host can run it. An undetected family is refused with
  the existing "not available" error, which lists what was detected. A QSPICE
  run's `.qraw` and a Xyce run's raw file are read in their own formats,
  chosen from the job's recorded simulator (Xyce's raw file does not name the
  program that wrote it).
- `inspect(kind="capabilities")` reports, for every simulator, whether a run
  can select it (`selectable`), and for a family this host cannot run, the
  `refusal` explaining why. The reference lookup finds `run_experiments`' argument
  table, where `execution.simulator` lists the families, for "qspice", "xyce"
  and "choose the simulator".
- A Xyce run records the release its log banner names
  (`Xyce Release 7.8.0-opensource`) as the case's `simulator_version`.
- `plot_waveform` replies summarize each plotted trace with `min` and `max`
  and the axis value where each occurs (`x_at_min`, `x_at_max`), `initial`
  and `final`, and the time-weighted `mean` on a transient, read from every
  sample in the window before any decimation. An AC trace is summarized in dB with
  its unwrapped phase at both ends; a stepped run gets one summary per step,
  up to 32 (when there are more, `traces_truncated` gives the total and an
  observation reports the truncation). The summaries are in `traces` and the
  text reply, on both the in-chat widget and the terminal path, and `x_unit`
  names the axis unit. Before, the reply contained only the file path and the
  plot's point counts.
- `plot_waveform(attach_plot=true)` returns a PNG of the chart as an image
  block for a vision model, and writes it beside the HTML (`image_path`,
  with its size and `estimated_tokens` in `image`). It draws the same panels
  as the interactive chart, is off by default, and takes its default from
  `[analysis] attach_plot` (`LTSPICE_MCP_ATTACH_PLOT`). Rendering needs the
  `raster` extra; without it, the reply includes an `image_unavailable`
  observation naming what is missing and how to install it, instead of the
  image.
- `plot_waveform(panels=[[...], [...]])` lays the panels out by hand, one
  list of signals per panel (up to 8), for traces of one unit but very
  different size.
- `[analysis] open_plot` (`LTSPICE_MCP_OPEN_PLOT`, on by default) sets
  whether `plot_waveform` opens its chart in a local browser window; a call's
  `open` still takes precedence. A terminal session can turn the windows off.
- `run_code` has `window_and_clean` and `compute_signal_stats` in scope, so a
  derived trace's statistics are weighted by time; `np.mean` over LTspice's
  variable timestep over-weights the samples packed around edges. The guide's
  trace-math example continues with them.
- T-junctions onto an existing wire. `wire_pins` takes `{x, y}` for
  `from_pin` or `to_pin`. The point must touch a wire or a pin, and on a
  wire's interior the new segment ends there with the wire left whole,
  because LTspice's netlister does not need the wire split to join it. The op
  reports the T under `junctions`. A leg running along the wire it joins, or
  an endpoint where wires of two nets cross, is refused.
  `inspect(kind: "net")` at a point on a wire's interior traces that wire's
  net and names the wire under `snapped_to_wire`, where it used to return
  "Nothing found"; a point where two nets' wires cross is refused as
  ambiguous. The connectivity rule was
  checked against LTspice 26.1.1 exports of a label, a pin and a wire end on
  a wire's interior, a plain crossing and a label at a crossing, recorded in
  `tests/fixtures/t_junctions/`.
- `edit_schematic` returns each op's findings on the sheet in a `results`
  list, one entry per op keyed by `index` and `op`. `wire_pins` gives the
  requested segments that were `already_present` and not redrawn and the
  `junctions` its route made; `remove_wire`, `remove_net_label` and
  `remove_component`'s wire cleanup give how much they removed. These facts
  were not reported before, so a caller saw a redundant junction
  only as warning text and did not see a skipped segment, though the design
  doc stated that `already_present` was reported. An op that found nothing
  has no entry. `Api.edit_schematic` returns the same list.
- `LTSPICE_MCP_WRITE_CONFIG=false` stops the first tool call from writing a
  default `ltspice-mcp.toml` into a directory that has none. It is read from
  the environment only, because a config file that exists is never
  rewritten, so a key inside one would have no effect.
- `LTSPICE_MCP_STORE_DIR` keeps each working directory's store under another
  directory (for example a cache directory) instead of in
  `<working_dir>/.ltspice-mcp/`. There is one store per working directory,
  named after it and keyed by its resolved path, so idempotency, job listings
  and result sets stay scoped to the working directory as before. As a
  trade-off, records are no longer found by the working directory alone. A
  server, Python script or `run_code` worker that does not have the same
  setting looks in `<working_dir>/.ltspice-mcp/` and finds none of them, and
  a `request_id` submitted from there runs again instead of replaying.
  The process that runs a detached job and the `run_code` worker inherit the
  setting from the process that starts them.
- A new `partial_progress` observation is recorded for every case the
  server stops, once the simulator's exit is confirmed. It gives the plot, its
  axis, the complete points on disk and the last axis value reached, read
  from the partial raw file before cleanup deletes it. The count comes from
  the file's length, because ngspice leaves `No. Points` at 0 until a plot
  ends and LTspice's count lags the data. Neither simulator's `.log` or
  `.exe.log` records progress for a killed run.
- A new `run_progress` observation appears for every case still running, on
  the receipts that `run_experiments`, `jobs(status)` and `jobs(wait)`
  return. It has the same progress facts plus the seconds since launch, read
  from the raw file the simulator is writing at that moment. It is read only
  when a receipt is built, is not stored in the job record, and costs a
  header and one record per running case.
- Recovery hints on `run_timeout`, `job_deadline` and `kill_unconfirmed` failure
  rows.
- Signal names accept a node-pair voltage `V(a,b)`, read as `V(a) - V(b)` from
  the same raw, step and axis. This works in every `analyze_results` recipe,
  in `plot_waveform` and in `RawResult.trace`. `V(a,0)` and `V(a,gnd)` read
  as `V(a)`, and a missing node is named in the error. On an AC run the
  difference is complex, so `V(out)/V(inp,inn)` works as a `stability` or
  `bode_*` signal. A `.noise` run refuses a pair because spectral densities
  cannot be subtracted.
- Two new lint rules check for a non-ASCII character in a scale suffix's
  position, which the simulator reads as the bare number.
  `value-suffix-mojibake` blocks a suffix that shows the file was decoded in
  an encoding it was not written in. Such a suffix is `Âµ`, a UTF-8 micro
  sign decoded as cp1252 and a factor of 1e6 off, for which the finding names
  the likely intended value, or any other `Â`, `Î` or `Ã`.
  `value-suffix-nonascii` warns about any other symbol, such as `10Ω` or
  `25°C`, where the bare number is usually what was meant. `verify_circuit`
  reports them as `value_suffix_mojibake` (an error) and
  `value_suffix_nonascii` (a warning), and reports a `µ`/`μ` suffix itself as
  `value_suffix_micro_sign`, from its `syntax` check on a netlist and from
  its `export` check on the netlist exported from an `.asc`, along with the
  file's encoding and the `Generated by LTspice` header when present, so decks
  run outside the server are also checked.
- A variation can target one exact instance inside nested subcircuits by its
  list of instance names: an `assign` entry's `instances`, or a `random`
  entry's `mismatch` rule with mismatch coefficients you give. Each case gets
  its own copies of the subcircuits on the path to that instance, so other
  instances of the same subcircuit and the original files are unchanged.
- A `pdk_native` variation runs Monte Carlo on ngspice with the statistics
  built into the Sky130 PDK models, for its `sky130_fd_pr__nfet_01v8` NMOS
  (profile `sky130-e6f9c887-ngspice-v1`, pinned to one PDK revision). Each
  sample has its own seed, so one sample can be run again by itself with
  `runs: 1` and its `sample_start`, and the job record keeps each case's
  seeds, model file hashes and paths.
- Time-weighted quantiles. The `signal_stats` recipe takes
  `quantiles: [0.01, 0.99]` (up to 16 levels in [0, 1]) on a transient run
  and adds `q01`, `q99` and `quantile_peak_to_peak`, the highest level minus
  the lowest. Each can be named as the `field` a `reduce` or a `spec` reads, so
  a robust ripple limit takes one recipe. A level `p` is the smallest value the
  signal spends a fraction `p` of the window at or below, reading the
  straight line between samples the way `mean` and `rms` already do.
  `np.percentile` over the samples gives a different answer on simulator
  output, because the simulator shortens its step around every edge and the
  samples cluster there. On the recorded RC fixture the input's sample 1st
  percentile is 0.00012 V, while the input spends 99.9% of the window at 1 V.
  `min`, `max` and `peak_to_peak` are unchanged and remain the sample
  extremes. Asking for quantiles on an AC, DC-sweep or noise run fails that
  recipe with a message. The same function is
  `ltspice_mcp.api.time_weighted_quantiles`, and `run_code` has it in scope.
  It drops and counts non-finite samples, and reads a time axis taken from a
  raw's stored data, with LTspice's negated compressed-time points, through
  `abs()` as spicelib does.

- `inspect(kind="capabilities")` reports `render: {png, missing, reason,
  remedy}`. It reports whether `verify_circuit` can make the PNG it returns
  inline and, if not, whether the `raster` extra is missing, or only
  `cairosvg` is installed without the native Cairo library, and how to
  install what is missing, before anything is drawn.
- Results record which simulator build produced them. A job records the path,
  size, modification time and SHA-256 of the executable its cases launch.
  Each case records the build the simulator named in its own output, read
  from the LTspice log banner (`LTspice 26.0.2 for Windows`), the ngspice
  console banner (`ngspice-42, Creation Date: ...`), or the raw header's
  `Command:` when neither exists (LTspice XVII). Run rows include it as
  `simulator_version`.
  `jobs(runs)` and `run_fields` return it, and the lean receipt drops it from
  produced rows along with their artifact paths. `provenance: true` adds the
  job's `simulator_executable` to the receipt. Only the first few kilobytes of
  each artifact are read, whatever its size.
- `inspect(kind="capabilities")` reports each simulator's `executable_sha256`.
  Its `version` is now the build that the latest run on that same executable
  reported, with `version_source` naming the job and case; it was always
  null. The executable is never launched to query its version.
- One server can run more than one build of a simulator family, such as
  LTspice XVII beside LTspice 24, chosen per call. `[simulator.executables]`
  names further executables (`xvii = "C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"`,
  or a `[simulator.executables.ngspice]` table, or `"family:name"` keys), and
  `LTSPICE_MCP_SIMULATOR_EXECUTABLES` takes the same as `name=path` entries
  separated by `;` on every platform, and so does
  `Api(simulator_executables={...})`. The family is read from the file name
  (`XVIIx64.exe` is LTspice), and an executable whose name does not identify
  the family needs `family:name`. A run selects one with
  `execution.simulator = "ltspice:xvii"`; a plain family name runs the
  family's own executable as before. Adding a named executable does not
  change what the family name runs, nor any run already in progress. Each
  build has its own cap on simulators running at once, so each can run
  `[simulation] max_parallel` simulations, as each family already could. The
  job record, the `request_id` replay check, the kill on stop, the raw file
  format, the lint rules, the `.asc` export and the model library folders all
  use the build the run selected. Each entry is checked at startup; one that
  cannot be used (a missing file, a family not listed in `[simulator]
  enabled`, a file that looks like another simulator) is left out with a
  startup diagnostic. `inspect(kind="capabilities")` lists them under
  `named_executables`, keyed by the selector, each with its executable,
  `executable_sha256` and last reported `version`. A `request_id` replayed
  under another name, or under a name whose executable is now another build,
  is an `idempotency_conflict`.

- `inspect(kind="capabilities")` takes an optional `fields` list naming the
  top-level keys to return, such as `["allowed_paths", "config_path"]` after a
  config edit. Without it the report is unchanged.
- `edit_schematic` has a `preexisting` view, which lists what an edit counted
  under `preexisting` (see Changed). Name it in `return_views`, or echo
  `preexisting.cursor` as `view_cursors.preexisting`.
- A `.asc` pin can be addressed by its 1-based SpiceOrder. `X1.2` addresses
  the second pin of a block whose pins are named `A`/`B`, the same terminal
  number a netlist `inspect(kind="net")` query takes. Pin names are matched
  first, because some symbols name their pins `1`/`2` in an order that need
  not be their SpiceOrder. This works everywhere a pin is accepted, which is
  `inspect(kind="net")`, the `wire_pins` op, and the `pin` of
  `add_net_label`, `remove_net_label` and `remove_wire`. An unknown pin's
  error lists each pin as `name (order)`.

### Changed

- `verify_circuit` wrote its drawings into the server's working directory
  (`.ltspice-mcp/verify/renders`), so a server started in one project
  collected drawings of sheets in others. They now go to the per-user folder's
  `renders/` (`%LOCALAPPDATA%\ltspice-mcp` on Windows,
  `~/.local/state/ltspice-mcp` elsewhere, moved by `LTSPICE_MCP_HOME`), and
  each render deletes drawings there that are a week old. The reply gives the
  path as before.
- `edit_schematic` checks the result of its ops once, on the sheet the batch
  commits, through four new sheet rules that `verify_circuit` also reports.
  They are `net_with_two_names` (a net whose labels have two or more names),
  `joined_at_crossing` (a pin or label where wires cross that joins nets
  which would otherwise be separate), `unjoined_crossing` (wires crossing
  where neither ends and nothing else is placed) and `diagonal_wire`. Ops
  used to warn of these themselves, judged on the sheet as it stood after
  each op. A net relabelled in remove-and-add pairs got a second-name warning
  though the batch ended with one name, and `move` did not report a pin it
  put on a crossing, while `move_component` warned about it. Those op
  warnings, and `wire_pins`' "crosses the wire" advice, are removed; the
  refusals are unchanged. The check in `set_component_value` that a value
  would not split into an extra node stays in the op, because it checks the
  given value.
- `Api.wait` takes its timeout as `timeout_s`, as `jobs(action="wait")` does.
  `timeout=` is still accepted; giving both is refused.
- `inspect(kind="net")` and `inspect(kind="components")` on an `.asc` read the
  file instead of the schematic editor `edit_schematic` uses. They now work
  on sheets the editor cannot open (a bus tap, a blank line, an unknown
  record, a sheet whose symbol the editor does not find), and name each line
  of the sheet they do not read in `warnings`. A components row's
  `attributes` contains only the part's `SYMATTR` lines; per-instance
  `WINDOW` lines no longer appear there as `_WINDOW n`. Two parts that share a
  reference are both listed, where only the last was; a net trace from a pin
  of such a reference is refused, naming where both parts are.
- `verify_circuit` and the findings an edit reports read a sheet through the
  same reader as `inspect`. A malformed line is now reported with its line
  number instead of being guessed at or dropped. Such lines include an
  attribute after a line that ends a part, a part with no rotation, a `TEXT`
  with no `!` or `;`, a malformed `WIRE` or `FLAG`, and a `WINDOW` under a
  part that does not parse. A symbol with a pin line that does not parse is
  drawn as a placeholder, as the editor already treated it, and the finding
  names that line. A symbol file that is found but cannot be read is
  reported as unreadable; before, it was reported as not found in any
  library.
- There is no sandbox unless `[security] allowed_paths` (or
  `LTSPICE_MCP_ALLOWED_PATHS`, or `Api(allowed_paths=...)`) is set. An unset
  list used to mean the working directory and Claude Code's scratch directory,
  so a deck an agent wrote anywhere else was refused, and the refusal's remedy
  was an edit to the server's security setting, which agents decline to make.
  Now a path may name any file the server process can open, and a relative
  path is taken from the working directory. Setting the list confines the
  server as before; set `[tools] run_code = false` as well to keep a client
  to those directories, because `run_code` is not confined by it. Related
  changes:
  - a bare string in `allowed_paths` names one directory instead of being
    ignored, so a slip in the key still confines the server;
  - with no sandbox, a run stages every file its deck reaches, wherever it is;
  - the server instructions no longer mention the sandbox, which is re-read
    from the config file on every call; a refused path's guidance explains
    what to change, with a sandbox or without one;
  - `inspect` capabilities and the `spice://config` resource report
    `allowed_paths` as null when there is no sandbox;
  - the Claude Desktop extension's folder is optional and empty by default, so
    it runs with no sandbox unless one is picked.
- `wire_pins` no longer warns about routes over 400 units ("Long wire run
  ..."). The threshold was arbitrary, because 29% of LTspice's own example
  sheets have a wire that long. The guide's wiring section now gives the measured
  distribution of wire lengths.
- `edit_schematic` no longer reports three conditions twice. A route through
  a part, a part overlapping another, and a label touching nothing were each
  reported as an op warning (`op 3 (wire_pins): Wire at y=252 crosses R4
  bounding box ...`, `Overlaps R1 bounding box`, a floating-label note) and
  again as a sheet finding. The op warnings are removed; the sheet findings in
  `warnings` remain, and match what `verify_circuit` reports. This has two
  consequences. Findings describe the sheet after the whole batch, so a label
  that a later op wires up, or an overlap a later op removes, is no longer
  reported. A route through a part is also judged by the part's body box,
  including the parts the route ends on; the op warning used the box with
  pins and skipped those parts.
- `edit_schematic` and `verify_circuit` now report one list of sheet findings.
  Each reported findings the other did not. An edit never reported, as a
  sheet finding, that a part it placed overlapped another's box, that a wire
  ran through a part, that a wire end was left unconnected, that text was
  inside a part, that a net was joined only by labels, or that LTspice would
  leave a wire out of the netlist. A check never reported that a wire was
  drawn twice, that a label was unconnected or inside a part, or that
  directives were stacked. Both tools now report all of these with the same
  wording, and an edit reports the same findings for a sheet as a check of
  the written file. In `verify_circuit` the four new findings are
  observations, with a wire drawn twice and a dangling label under `layout`,
  and a label inside a part and stacked directives under `quality`. A
  finding's `evidence.detail` is now a sentence that names its parts and its
  location, and findings are listed in one order, with findings that change
  the circuit or leave it incomplete before findings about how the sheet
  reads. In `edit_schematic` the `preexisting` view's rows include the six
  new kinds, and a row lists every part (`refs`) and point (`points`) its
  finding names.
- An edit reports a finding that was already on the sheet, instead of
  counting it under `preexisting`, when the batch named any part or point the
  finding names. A finding about two parts used to be reported only when the
  batch named the first part.
- Neither tool reports anything about the extent of a part whose symbol is
  not found. `verify_circuit` used to report overlaps with, and wires
  through, the placeholder box such a part is drawn as, which is not the
  part's real extent.
- A wire between a pin of one part and a pin of another that has the same
  reference, or none, is no longer reported as a wire LTspice leaves out,
  because the rule applies only to two pins of one part.

- The server's name in a client is `spice`, where the README and the Claude
  Code plugin used `ltspice`. LTspice 26.1 ships an MCP server of its own and
  registers it as `ltspice`, so the two would have shared a name. **For plugin
  users the tool names change once**, from `…_ltspice__<tool>` to
  `…_spice__<tool>`, so a saved permission rule or an instruction that names
  the old ones must be updated. A server registered by hand keeps whatever
  name its entry has; rename an entry called `ltspice`.
  The package, the command (`ltspice-mcp`), the repository and the Claude
  Desktop extension keep their names.

- `jobs(action="wait")` on a job another server process owns detects that
  the owner has finished within half a second; it re-read the record every
  two seconds before. The owner's watch for a cancellation requested by
  another process, and the requester's wait for that cancel to take effect,
  use the same interval.
- Reading a result again no longer starts a parser process. Each read first
  stats the source's RAW, log and console files. A source whose files still
  have the device, inode, size, modification time and change time recorded
  when it was last parsed is served from the server's in-memory cache. Any
  change, including a same-size rewrite that puts the modification time back,
  sends the read to a worker as before. A file's times are recorded for this
  check only once both are older than a filesystem timestamp tick. A parser
  process also starts faster, because the worker imports its decoders only
  when it has content to decode, and the store no longer imports psutil until
  it checks a job owner's liveness. On Linux a repeated read took about
  300 ms and now takes under a millisecond.
- A session keeps its parser process between reads of new results instead of
  starting one per read. A read's results are used only once the parser has
  replied and no other process is left in its process tree. The parser is
  replaced after 64 reads, after half an hour, and after two idle minutes,
  and at once after a failed, cancelled or timed-out read or when its tree
  holds a process the server did not start. If the server cannot confirm
  that a replaced parser and its child processes exited, the session's later
  reads of new results fail with `cleanup_failed` instead of starting another
  parser. Waits on a parser process are woken by the operating system (a
  pidfd and `SIGCHLD` on Linux, a handle to each process on Windows) rather
  than by polling every 5 or 10 ms. On Linux each read of new results took about
  400 ms; a session's first read still does, and the reads after it take
  about 15 ms.
- A run's simulator library roots, the install directories staging accepts as
  the simulator's own, are now those of the LTspice build the run launches:
  LTspice XVII's `Documents\LTspiceXVII\lib`, or LTspice 24's
  `%LOCALAPPDATA%\LTspice\lib`. Before, every LTspice library spicelib knew
  of was accepted whichever build ran. On WSL a server running XVII now finds
  XVII's library under `%USERPROFILE%`; only LTspice 24's location was probed
  before, so a transistor schematic exported by XVII could not be staged under
  a sandbox that left XVII's library out.
- On WSL, cancelling a case also matches the Windows process by the file name
  of the executable its runner launched, so a copy of LTspice under another
  name is stopped; before, only `LTspice.exe`, `XVIIx64.exe` and `scad3.exe`
  were matched.
- The `directive-arity` lint rule is split into four rules (`linter_version`
  4), so suppressing one check no longer disables the others. The new rules
  are `element-arity` (fewer nodes than the element has terminals, blocking),
  `bsource-value-prefix` (a B-source with no value key, blocking),
  `value-keyword-ltspice` (`C=`/`L=` as the primary value on LTspice,
  blocking) and `value-expression-remnant` (tokens left after the first
  `key=value`, a warning). A `suppress` entry naming `directive-arity` no
  longer matches a rule.
- `meas-ngspice-batch` is now a warning instead of blocking. ngspice runs the
  deck and skips only the top-level `.meas`, and reading the run relays
  ngspice's own notice of the skip; refusing the deck cost the caller the
  whole run.
- `verify_circuit` reports micro-sign value suffixes (`value_suffix_micro_sign`)
  as one observation per file, with their count, lines and tokens, which leaves
  the outcome `complete`. LTspice 24 and later read the UTF-8 `µ` they write, so
  an LTspice export was otherwise `partial` with one warning per value. A
  warning per value remains where the server knows of a reader that would
  misread the file. That is when the session's LTspice is XVII or earlier
  (from the executable's name, or the build a run on it reported) or the deck
  names XVII as its writer, and the file is not cp1252. The warning's
  evidence names that reader. Both value-suffix rules now count toward the
  cap of 25 findings listed per rule, and a `showing N of M` observation
  reports the total when more were found.
- `run_experiments` reports the `micro_sign_folded` observation only when the
  job runs on an LTspice that decodes decks as cp1252. Staging still writes
  every micro-sign suffix as `u`. For every other simulator the result reads
  the same as the source, so the observation appeared on nearly every LTspice
  run.
- The guide states each rule once, in the section it belongs to, and points to
  it from elsewhere. LTspice-only syntax (`.step`, PWL extras, `startup`) moved
  from the fundamentals into the LTspice section, and `run_experiments` Monte
  Carlo and mismatch from the LTspice section into variations. Test-log notes,
  arguments against positions nobody holds, and repeated examples are
  removed.
- The Claude Code plugin ships one skill, `spice-guide`, in place of
  `spice-experiments`, `ltspice`, `ngspice` and `spice-bench-craft`. Those four
  were hand-maintained copies of the guide's content and had drifted from it.
  The `ngspice` skill stated that a `GND` node floats unless declared global
  (ngspice converts `gnd` to node 0 by default) and that a `.step` line is
  rejected (lint warns, and ngspice runs the deck once). The `ltspice` skill
  listed five of the seven pin transforms, and the workflow skill's
  `verify_circuit` example passed `reference` at the top level instead of
  inside `compare`. Their content is now only in the guide, with material the
  guide lacked moved into it (a sweep-in-one-call example and per-tool notes
  in `tools`, more on ngspice's `.meas` types, statistical functions and
  `.control` variables). `spice-guide` loads on circuit and SPICE work and
  tells the session to read the guide.
- The server instructions point the model to the guide first, state when to
  use Python and when to use the tools, and keep the rules whose violation
  gives a wrong answer. Python is for anything past a single call, such as
  loops and complete results in one call; the tools are for one step per
  call, charts, and jobs the server owns. The per-tool map the instructions
  contained moved to the guide's core.
- `spice://guide` returns the guide's core and index rather than the whole
  guide; each section is `spice://guide/{section}`. The packaged file
  `assets/spice_guide.md` became `assets/guide/`, one file per section. Pointers
  in tool descriptions and hints name the section they mean
  (`guide section 'signals'`).
- `plot_waveform`'s output schema declares `hint`, which contains the guide
  reminder when that is the session's first reply.
- A run on QSPICE is refused unless the server runs natively on Windows, with
  the reason, including when QSPICE is the server default and no simulator is
  named. spicelib starts QSPICE with the host's own file paths, and only
  LTspice has a path adapter for a Windows simulator under WSL or Wine, so such
  a run used to be launched with paths QSPICE cannot open. Off Windows the
  capabilities remediation for QSPICE now states that it cannot run there
  instead of suggesting an install path.
- A `.asc` circuit on QSPICE or Xyce fails with `asc_export_unavailable`
  before LTspice is launched. A schematic runs through LTspice's netlist
  export, in LTspice's own syntax, which the server rewrites only for
  ngspice; a run on QSPICE or Xyce takes a hand-written `.cir`/`.net`/`.sp`.
- `plot_waveform` and the `analyze_results` `plot` recipe give each declared
  unit its own panel, so volts and amps no longer share a y-axis; an AC plot
  gets a magnitude and phase pair per unit. Panel titles include the unit, and
  a `.dc` sweep's x axis is labelled with its swept source. Noise densities
  are labelled V/√Hz or A/√Hz. Input-referred noise takes its unit from the
  deck's `.NOISE` source, and when there is no deck to check,
  `plot_waveform` reports a `noise_input_unit_unverified` observation. Every
  multi-panel chart shares one x cursor, where only the two Bode panels did
  before.
- `plot_waveform`'s `open` defaults to `[analysis] open_plot` instead of
  always `true`.
- The server instructions and `analyze_results`' description explain where
  trace math belongs. A recipe's signal is one trace or `V(a,b)`, and
  anything else is numpy on the traces. Before, only the error after a failed
  expression explained this. That error's example is now a product instead
  of a difference, which `V(a,b)` already reads.
- Reusing a `request_id` returns an `idempotency_conflict` when the job ran on
  a different simulator build than the request would launch now, as it does
  for an edited deck. Before, swapping the simulator executable (or the
  default simulator) and restarting replayed the earlier build's results under
  the same id. A job record that names no executable is refused the same way.
  A replay therefore needs the requested simulator to be available, as a fresh
  submission does; `jobs(action="status")` reads the recorded job without one.
- The store format is version 3. Job records now include the simulator
  executable and each case's reported build. Version 2 records still load,
  with both unknown; version 1 records (0.6.1) are not read.
- A session no longer writes bookkeeping into the folders of the circuits it
  touches. Each file it wrote there has moved:
  - the cross-process lock taken for every edit and export
    (`<circuit dir>/.ltspice-mcp/locks/`) is now in the per-user home, one lock
    per file keyed by the digest of its resolved, case-folded path. Sessions
    started in different working directories still contend on the same lock;
  - the export snapshot an experiment runs a schematic from
    (`<circuit dir>/.ltspice-mcp/exports/`, or a `<name>.run-<hash>.net`
    sibling whenever the export had a relative `.include`) is now in the
    store's `exports/`, beside the job records that name it. Staging resolves
    the snapshot's relative includes from the schematic's folder, so the
    sibling exception is removed;
  - the copy of the export rewritten for ngspice, `<name>.ngspice.net`, is
    written straight into that snapshot;
  - `plot_waveform` charts written without an `out_dir`
    (`<circuit or raw dir>/.ltspice-mcp/plots/`) are now in the store's
    `plots/`.

  LTspice's own `<name>.net` beside a schematic it exports is written by
  LTspice's exporter and stays. `edit_schematic`'s reference check keeps no export
  snapshot at all. Existing `.ltspice-mcp/` folders beside circuits are not
  read or removed.
- The per-user home is `%LOCALAPPDATA%\ltspice-mcp` on Windows (it was
  `~/.local/state/ltspice-mcp` on every platform), and the XDG state directory
  elsewhere; `LTSPICE_MCP_HOME` still overrides it. On Windows the
  recent-circuits index starts empty once and refills as circuits are touched.
- `analyze_results` reports a `raw_path` the sandbox refused as `path_denied`
  rather than `source_unavailable`, and the `inspect` hierarchy query reports
  one as `path_denied` rather than `error`.
- `jobs(action="runs")` builds the full `native_statistics` evidence only
  for the rows on the returned page, and only when asked. A `pdk_native` run
  parses each unchanged original model file once per circuit instead of once
  per case, and still checks each case's sample.
- `jobs(action="runs")` accepts `run_fields`, which selects the fields each
  row returns; naming `native_statistics` returns a `pdk_native` case's full
  coverage and dependency paths.
- A case has no time limit unless one is set. Before, every case was capped
  at spicelib's hidden 600 s. Now a case runs until it ends or is cancelled,
  unless the request sets `execution.run_timeout_s` or the operator sets the
  new `[simulation] run_timeout` (`LTSPICE_MCP_RUN_TIMEOUT`, unset by default).
  A running job reports each case's progress, which shows when a case is
  stuck.
- Capabilities `limits.default_timeout_s` is replaced by `run_timeout_s` (the
  server's per-case limit, `null` when unset) and `export_timeout_s`.
  `[simulation] timeout` bounds only LTspice netlist export, which is all it
  ever bounded.
- `edit_schematic` with `compare` no longer returns the committed sheet's
  exported netlist when the verdict is `equivalent: true`. Before, every
  compare returned the whole deck in `netlist`, even though a confirmed match
  means it is equivalent to the reference the caller supplied. A mismatch, a
  compare error or no verdict still returns it, as the sheet's side of the
  comparison to diagnose. To get the netlist of a matching sheet, run
  `verify_circuit` with the `export` check. The Python API's `edit_schematic`
  does the same.

- The Claude Code plugin and the Claude Desktop extension install
  `ltspice-mcp[raster]`. Neither lets a user add an extra to the launch it
  ships, so PNG rendering was impossible through them; it now needs no other
  step. The README documents the extra.
- CI: lint and type check run once, in a new `checks / static` job, instead of
  on every interpreter. They gave the same result on each, because ruff and
  pyright both target 3.11 and the lockfile resolves one package set for 3.11
  through 3.13. The Linux test legs run the suite on one xdist worker per
  core, which takes about 90 s on a four-core machine instead of about 200 s
  serially. The Windows leg still runs serially, so it now determines how
  long the checks take. The
  existing status-check contexts keep their names.
- The alias publish workflow no longer runs its own copy of the checks. An
  alias already waits for the canonical `ltspice-mcp` release to appear on
  PyPI, and that release publishes only after the checks pass, so the second
  run doubled every release's test matrix without gating anything more. The
  wait now allows 30 minutes, long enough to cover the checks.

- An `edit_schematic` call on an existing sheet reports only the sheet findings
  in `warnings` (floating pins, dangling labels, duplicate wires, a label
  inside a body, stacked directives) and the `wiring.label_only_pins` rows that
  the batch introduced, or that name a reference or coordinate one of its ops
  named. It used to return every one on the sheet, so a one-part edit to a
  large sheet returned every older finding and label-only pin. The rest
  are counted in a new `preexisting {count, findings, label_only_pins, cursor}`
  block rather than dropped. `pins_total`, `pins_wired` and `pins_label_only`
  stay whole-sheet, and `label_only_pins.total + preexisting.label_only_pins`
  equals `pins_label_only`. A `base: "blank"` build is unchanged, since
  everything on it is new. A caller that relied on the whole-sheet lists can
  add `preexisting` to `return_views`. An op-less read (`ops: []`) now counts
  every finding under `preexisting` too, because nothing on it is new.
- An `edit_schematic` view cursor now returns its view even when
  `return_views` does not name it. A `pin_legend` or `touched` cursor sent with
  the default `return_views` used to be accepted and then ignored.
- `prefix` on `inspect(kind="components")` and `inspect(kind="hierarchy")` is
  a case-insensitive prefix of the reference rather than a single letter.
  For example, `"LX"` keeps LX1 and LX2 but not L1, and `"MXO"` keeps MXO1. A
  one-letter prefix still selects an element type. On `hierarchy` it reads
  each instance's own reference, the last segment of its path. It is plain
  text, so a prefix with a wildcard is refused with the plain spelling to use
  (`"LX*"` names `"LX"`), and one that is empty or contains a space is also
  refused. Both kinds report these as `invalid_prefix`; `hierarchy` reported
  `invalid_query`.
- `verify_circuit`'s `export` check writes `<name>.net` beside the schematic by
  default (`export_to: "sidecar"`), the file LTspice itself writes beside a
  schematic it runs and the one `run_experiments` exports through. The default
  used to be `managed`, which copied the schematic and every `.asy`, `.lib`,
  `.sub`, `.inc` and `.mod` under its folder into the store to avoid that file,
  and is where a relative include was once refused as `path_denied`. `managed`
  is still available for a call that must write nothing beside the schematic.

### Removed

- The `spice://models/` resource. It listed libraries loaded through
  `load_library`, which was removed in 0.6.0, so it has returned an empty list
  since then. `inspect(kind="model", mode="search")` searches the simulator's
  own libraries, and its `libs` names any other file.
- The `suggestions` key of the `analyze_results` summary recipe, and the fuzzy
  matching behind it, which searched the same loaded libraries and so never
  produced one. A failed run's receipt already names the unresolved models
  (`missing_model`, with `evidence.missing_refs`) and the model search that
  finds them. The server does not run that search itself on a failure path,
  because the first search parses the whole install, which takes seconds.

### Security

- Dependency upgrade for a published advisory in the locked runtime set:
  `pyjwt` 2.13.0 → 2.15.1 (CVE-2026-102274). It is a transitive dependency of
  the MCP SDK, whose only use of it is client-side OAuth credentials, which
  this server does not import.

### Known issues

- With LTspice XVII as the simulator and LTspice 26 also installed, `.asc`
  editing and rendering read symbols from LTspice 26's library, where some
  symbols place their pins differently. On WSL with only XVII installed, no
  symbols are found. Set `[schematic] symbol_paths` to XVII's `lib/sym`
  folder to use its symbols.
- A rendered schematic draws no junction dots, so wires that cross and wires
  that join at the same point look alike.
- A symbol file with a blank line between a `PIN` line and its `PINATTR`
  lines loses that pin's name and order. One symbol LTspice XVII ships,
  `DAC/AD5679R`, has such a line.
- `edit_schematic` keeps each library symbol it has read for the life of the
  server, so a symbol file changed while the server runs is seen by
  `verify_circuit` but not by an edit until the server restarts.

## [0.6.1] - 2026-09-08

### Fixed

- The server process could hang indefinitely, without a diagnostic, after it
  released a spicelib simulation runner that held a task queued but never
  started. spicelib's runner destructor waits for its task list
  to drain and takes its deadline only from tasks that have started, so a
  task that was queued but never started is never removed and sets no
  deadline. Because the last reference usually goes away during garbage
  collection, the wait runs on whichever thread happens to be allocating. The
  runner is now built from a subclass with no destructor. The server did not
  need the destructor, because completion is reported through the run
  callback and liveness is read from the task threads.

### Changed

- A test that hangs now dumps every thread's stack and exits, naming the
  frame it is stuck in, instead of running until the caller gives up. This
  uses pytest's own faulthandler settings and adds no dependency.

## [0.6.0] - 2026-09-07

This release puts one engine behind two interfaces. The MCP surface has
eight tools in place of the 49-tool and 41-tool profiles. They are six
operations grouped as execute, understand and author, plus the waveform
widget and a Python snippet runner. The same six operations are importable
in-process as `ltspice_mcp.api.Api`, with complete results where the MCP
replies are paged. A job runs one durable experiment across declared
variations. Results come from typed recipes, and schematic edits are applied
as transactional batches. This release breaks the 0.5 tool names; the first
section maps each old tool to its replacement.

### Breaking: one tool surface replaces the `full` and `agentic` profiles

The `full` (49-tool) and `agentic` (41-tool) MCP tool profiles are removed.
The server now advertises eight tools:

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

The same six operations are importable in-process through
`from ltspice_mcp.api import Api`, which runs the same engine.

**How to keep the old tools.** Pin `ltspice-mcp==0.5.*`.

**If you don't change your config.** `[tools] profile` is not a
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
  synchronous methods with complete results where the MCP replies are paged
  or capped, plus `load_raw` and `measurements` for numpy access to a run and
  the AC and transient metric functions under their existing names. It reads
  and writes the same job records as a server in the same directory, so the
  two can run side by side. A job belongs to the process that submitted it,
  and `close()` cancels the jobs it owns, as server shutdown does. `__all__`
  is the stability boundary and is pinned. The full contract is
  `docs/design/python_api.md`.
- `run_experiments` runs one experiment across declared variations — `assign`
  entries that set exact values, plus one `random` (Monte Carlo) entry — as a
  single durable job. `request_id` is optional. If it is omitted, a fresh id
  is generated and echoed on the receipt. Pass your own to make submission
  idempotent, so a retry with the same id, the same arguments, and unchanged
  source decks replays the existing receipt instead of running anything
  again.
- A `run_experiments` call waits up to `execution.wait_s` (default 60 s,
  maximum 120 s) and returns the results inline when the job finishes in that
  time. Otherwise it returns a receipt with a `job_id` and the job keeps
  running; follow it with `jobs(action="wait")`. `wait_s: 0` returns
  immediately.
- `run_experiments` takes a per-run simulator (`execution.simulator`:
  `"ltspice"` or `"ngspice"`), so one deck can be cross-checked on a second
  engine without changing config. A run on the second engine does not
  disturb the concurrency cap or the cancels of jobs running on the first,
  and a cancel reaches the simulator the job recorded. Results are read in
  the raw format of the simulator the job ran on, instead of the session
  default's, so an ngspice run under an LTspice default parses correctly, and
  the reverse.
- `Api.run_experiments(wait=False, detach=True)` hands the job to a small
  owner process that outlives the caller. The caller validates the request,
  and the owner submits the job, supervises it to a terminal status, and
  exits. The caller's `close()` leaves a detached job running.
  `jobs(action="cancel")` from any process stops it, as it stops any job
  another process owns, and a running server sees it as another session's
  live job. The receipt names the owner's pid and log file. Requires
  `[state] persist_jobs`.

- A new `run_code` tool, on by default (`[tools] run_code`,
  `LTSPICE_MCP_RUN_CODE`), runs a Python snippet in a warm worker process
  that holds the engine as `api` (the same six ops as methods, complete
  results, plus `np`, `load_raw`, `measurements`, `reference`), for loops over
  runs and numpy on samples. Each call is a fresh namespace around the same
  live engine; `timeout_s` (default 60, max 600) interrupts the snippet and
  cancels a run it waits on; a second call during one gets a `busy` reply;
  `reset` replaces the worker. Output is capped and reports what was dropped.
  The snippet runs with the server process's own file and process authority
  instead of inside the sandbox, so grant it permission in the client as you
  would a shell. Set `run_code = false` when the server is reachable by more
  than one trusted client, for example through a proxy. The capabilities
  report names the key (`python_api.run_code`), and the reference table lists
  the tool only on a session that serves it. The served tool list, the
  reference table, the capabilities entry and the server instructions all
  read the same setting, so they agree on whether `run_code` is served.
- `[tools] listing` (env `LTSPICE_MCP_TOOL_LISTING`) selects how much of each
  tool definition the tool list includes. `compact`, the default, advertises
  the same tools and the same schemas with every per-argument description
  removed. Structure, enums, defaults, `required` and `$defs` are intact, so a
  client can still build a valid call, and the server validates and replies
  as before. It cuts roughly 45% from the text a session loads before it can
  call anything; descriptions are read on demand through
  `inspect(kind: "reference")` and included in every validation error. In a
  same-day comparison against `full`, it gave the same answers at the same
  design quality, with one mismatch caused by the listing in 22 requests.
  `full` includes every description in the listing. Both listings are
  static.
- `inspect(kind="reference")` is a searchable lookup over the tools' own
  vocabulary, covering each tool's top-level arguments and the branches
  (analysis recipes, schematic ops, variation kinds, query kinds, checks and
  job actions). A plain-words `query` ("phase margin", "connect two pins")
  returns the closest entries with their fields, types, defaults, bounds and
  units, and an argument name ("all_steps", "expected_sha256") returns its
  tool's own table; with no `query` it returns the table of contents. `limit`
  defaults to 5 and caps at 20. The index is built from the same input models
  the server validates against, so nothing callable can be missing from it.
  It is the way to look up any argument's meaning under the `compact` tool
  listing, where per-argument descriptions are not published.
- `inspect(kind: "capabilities")` reports the Python API under `python_api`,
  with the import line, the session call on this working directory, and
  where an op's arguments are read (`api.reference`, `help`,
  `inspect.signature`).
- `inspect(kind="capabilities")` reports `diagnostics`, the startup notes (a
  bad configured simulator path, a requested engine that fell back, WSL
  auto-detection) that show whether the server started in a degraded state.
  Before, they were written only to the server's own log.

- `inspect(kind: "capabilities")` reports the effective ngspice compatibility
  mode (`ngbehavior`), so a command-line ngspice run that fails differently can
  be compared with the server's setting.
- `inspect(kind: "capabilities")` also returns `config_path` and lists every
  known-but-undetected simulator with a remediation naming the exact config key
  (`simulator.path`), the environment variable, the config file, a
  platform-appropriate example executable path, and the restart requirement. The
  text is built from the same constants the config loader reads, so it cannot
  name a key that does not exist. When a non-empty `simulator.enabled` allowlist
  is the reason an engine is off, the remediation states that instead of
  pointing to an install. The WSL no-simulator error now names
  `simulator.path` alongside the environment variable.
- `inspect(kind: "capabilities")` has a new `python` block (executable,
  install kind, whether the interpreter path is ephemeral, package location),
  so a caller can check that environment before switching to the in-process
  API. The server instructions include a one-line pointer
  (`from ltspice_mcp.api import Api`), and a finished `run_experiments` receipt
  that expanded ten or more cases points to the in-process API in its `hint`,
  because per-call overhead adds up over a loop of many cases.
- `verify_circuit`'s render block includes `source_sha256`, the digest of the
  sheet it drew. Rendering reads the file on disk, so a commit by a peer
  between `edit_schematic` returning and this call running was not visible.
  The block's `sha256` is the image's, and the export block's is the
  netlist's. Compare `source_sha256` with the `sha256` `edit_schematic`
  returned to confirm that the picture shows the revision you wrote.
- `verify_circuit`'s `quality` check runs on a `.cir`/`.net`/`.sp` netlist,
  not only on a schematic. Three connectivity rules that had been written but
  never called now report findings. They are a node wired to a single element
  terminal (`dangling_node`, an observation, because a bias fragment or a
  test stub leaves nodes open on purpose), a `V(...)`/`I(...)` reference in a
  directive naming something no element declares (`undefined_reference`, a
  warning; for example, an unlabelled net exports as `N00x` while the `.meas`
  still asks for `V(vref)` and measures nothing), and a net with two or more
  terminals that reaches ground through no DC-conductive element
  (`floating_net`, a warning, because its operating point is undefined). The
  default check set for a netlist is `syntax` and `quality`; pass `checks` to
  narrow it.
- A run that fails because the simulator could not open an `.include` or
  `.lib` reports the failure code `missing_include` instead of the generic
  `execution_failed`, with the file it could not find as evidence. When that
  deck selects a `.lib` section and the run was ngspice in an LTspice or
  PSPICE compatibility `ngbehavior`, the code is `ngspice_lib_section`. The
  receipt's hint then names the fix, which is `[simulator] ngbehavior = "hsa"`
  (or `LTSPICE_MCP_NGBEHAVIOR=hsa`) and a restart, or `set ngbehavior=hsa` in
  a `.spiceinit` in the run directory. Those modes read
  `.lib <file> <section>`, the standard PDK corner syntax, as two plain
  includes and drop the section, so the corner selection was reported as a
  missing file with no indication of the cause.

- Every tool has a display title, the short label a client shows a person
  in place of the wire name (Run Simulations, Analyze Results, Edit Schematic,
  and so on).
- The tool, resource, template and prompt listings tell a client how long they
  remain valid (`ttlMs` / `cacheScope`: one hour, private). They are built
  once at startup and cannot change while the process runs, so a client no
  longer has to re-list them every turn. Clients on earlier revisions are
  unaffected.
- `server/discover` returns instructions naming the detected simulators, as
  the initialize handshake does.

- A validation error that names a branch (a recipe, an op, a check, a query
  kind, a job action) now ends with that branch's field table, so a caller
  can correct the call from the error instead of looking the branch up first.
  This applies to the tool call, the Python API, and the attached analysis of
  `run_experiments`.
- The rule "`field` is required once `reduce` or `spec` is given" is stated in
  the recipe schemas themselves (`dependentRequired`), so a client on the
  compact listing, which has no descriptions, still sees it.
- A random rule's `tolerance`, `scale` and `distribution` now document their
  meaning on the field itself. The tolerance is a fraction of the nominal (or
  in the value's units), and for a normal draw it is the 3-sigma bound. The
  convention was stated only in a design document; an agent asked for a
  Monte Carlo at a stated sigma read the package source to find it.
- The Python API drops the two MCP presentation controls, `budget` and
  `execution.wait_s`, and reports this in the result's `warnings`, instead of
  refusing the call. Neither is part of a request's identity, so an MCP call
  replayed through the API with them attached is the same request. Paging
  controls are still refused, with `raw_page=True` named as the remedy.
- `.asc` schematics now run on ngspice. The LTspice netlist export is converted
  for ngspice by removing the LTspice-only `.backanno` command (ngspice aborts
  on it) and translating `µ` suffixes and `§` name prefixes. The converted
  netlist is written to a separate `{name}.ngspice.net` file beside it
  instead of replacing the shared `.net`, so concurrent LTspice and ngspice
  runs of one schematic cannot pick up a netlist generated for the other
  simulator.
- A clean simulator exit that wrote results only to the log (for example an
  ngspice `.control` script) now completes with the log-parsed measurements
  instead of reporting "Simulation failed (no output generated)". A run with no
  raw file whose log contains errors still fails. When the deck asked for an
  analysis that must produce a `.raw` and none appeared, the run fails with a
  `missing_required_raw` observation, a log excerpt, and the known `.save`-list
  workaround.
- Timed-out runs now include diagnostics. The timeout response finds the
  run's log, which the job record does not name for a job that has already
  stopped, and shows the end of it. Cleanup of a killed run keeps
  the `.log`, `.exe.log`, and `.fail` files and removes the possibly
  multi-gigabyte `.raw` and netlist copies. A killed run exits with a nonzero
  status, so spicelib renames its log to `.fail`; the job record is updated with
  the renamed path once the process exit is observed, so the excerpt stays
  readable.
- Windowed transient time axes are rebased to deck time. LTspice stores
  `.tran 0 <tstop> <tstart>` output with the axis starting at 0 and the true
  start only in the raw header's `Offset:` field, which the server did not
  apply, so a 196–202 µs window read as 0–6 µs everywhere. All raw loads now
  add the offset back, so window arguments and reported times are in deck
  coordinates.
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
  frequencies) and, when the worst-match scan finds a nearly purely reactive
  point (|Γ| ≈ 1 for any real z0), reports this and suggests a meaningful z0
  choice for power and filter ports, where the 50 Ω RF default is rarely the
  right reference.
- The `timing` recipe aggregates over all sequential edge pairs — `pair_count`,
  `delay_min`, `delay_max`, `delay_mean`, and the times of the extremes — for
  dead-time and minimum-off audits across a pulse train. Previously only the
  first crossing pair was measured.
- The `summary` recipe reports the run temperature (`temp_c`) and nominal
  temperature (`tnom_c`) parsed from the simulator log, so tempco, noise, and
  leakage tasks don't have to assume 27 °C. Its output schema now lists
  `ac_signal_used`, and it returns `suggestions` (model-resolution candidates
  for unresolved references) on completed runs as well as on failures;
  before, they were computed for a completed run and then dropped.
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
- A new `meas_batch_abort` observation. When one `.meas` directive fails to
  parse, LTspice abandons the whole `.meas` batch, and even directives earlier
  in the deck are missing from the results. The summary now links the misses
  to the failing directive instead of listing N independent-looking gaps.
- The AC recipes accept a leading `-` on the signal expression (`-V(out)`, or
  `-V(vout)/V(vsense)`). The complex wave is negated, a 180° phase flip, so a
  loop gain probed through an inverting sense or a reversed impedance probe
  reads in its natural convention without a behavioral inverter node in the
  deck.
- Per-step AC entries include `step_values`, the `.step` name=value point from the
  log, next to the bare step index. LTspice runs a `.step ... list` sorted
  ascending rather than in declared order, so index-only labeling could
  attribute curves to the wrong list positions.
- `resonance` peaks include `magnitude_linear` alongside `magnitude_db` (|Z|
  in ohms under the 1 A impedance probe, or |H| for a transfer function),
  which distinguishes a dBΩ peak from a dB dip. The recipe already documented
  the field; it now returns it.
- Independent server processes (for example several coding agents sharing
  one directory) no longer interfere with each other.
  - Circuit-file mutations and `.asc` exports take a cross-process file lock
    (in `.ltspice-mcp/locks/` beside the file), so concurrent edits of the
    same file from two sessions run one after the other on the latest content
    instead of silently losing one session's edit. Pin and route geometry is
    read while the lock is held, so it reflects a peer's just-completed move;
    an export locks the `.net` it overwrites beside the `.asc` as well as the
    `.asc`. If the lock is still held after a 10 s
    wait, the call fails with a "locked by another ltspice-mcp process" error.
  - Each job record stores the owning server's pid. A running job whose
    owner is still alive now loads in other sessions as `running` (previously
    it was mislabeled `interrupted`), is read again from its record when its
    status or results are read or listed, and is left alone by other
    sessions' shutdown cleanup.
  - `psutil` is now a direct dependency; it was already installed as a
    transitive dependency of spicelib.
- The `edit_schematic` `set_component_attribute` op with an empty value now
  clears the attribute, removing its SYMATTR line — the only representation of
  "no value" the `.asc` format can read back. InstName remains protected.
- Placed schematic directives no longer overlap. When two directives would use
  the same anchor, which is the common case when several are added without
  explicit coordinates, the second is moved. A `stacked_directive` advisory also
  reports exact-anchor coincidences that come from other paths, such as a
  hand-authored `.asc` or a move op.
- The bundled SPICE guide documents the LTspice-first noise-figure method:
  `NF_dB = 20*log10(V(onoise)/V(Rs))`, using the per-source noise contribution
  traces LTspice writes into `.noise` raw files, with no temperature constant or
  4kT term needed (checked against a reference divider at 3.0103 dB). The
  engine-neutral `4kT·Rs` form remains the fallback and the only method on
  ngspice.

### Changed

- The sandbox now changes with the config file while the server runs.
  `[security] allowed_paths` is re-read whenever `ltspice-mcp.toml` changes,
  so a refused path names the exact line to add and states that it takes
  effect on the next call. Before, the only self-serve route the refusal
  offered was copying the file, because widening the sandbox needed a restart
  the agent cannot perform. The server instructions state the sandbox rule in
  one sentence.
- `edit_schematic`'s `compare` now matches `verify_circuit`'s. It takes
  `mode` (`equivalence` or `structural_diff`), `anchors` and `rtol`, runs on
  the same comparison engine, and `verification.comparison` contains the same
  payload (a `structural_diff` delta with the verdict derived from it). A
  compare that could not run reports `verification.compare_error` instead of
  raising.
- `include.fields` accepts a bare name for a number. A path without a dot
  that is not a row key reads under `value`, so `phase_margin_worst_deg`
  means `value.phase_margin_worst_deg`. A dotted path with an unknown root is
  still refused.

- The default sandbox is the working directory plus the Claude Code scratch
  directory (`<tempdir>/claude-<uid>`). Claude Code tells an agent to write
  throwaway files there, outside the working directory, so a deck authored
  there used to be refused and copied in first. The generated config documents
  the default in a comment and leaves `allowed_paths` unset; setting it
  replaces the default. This applies to both the server and the Python API.

- `analyze_results` takes `step` and `all_steps` as call-level arguments instead
  of per-recipe ones, and `run_experiments`' attached `analyze` block takes the
  same two, so an attached measurement and a standalone one read the same
  `.step` iterations. A run's step axis belongs to the run instead of to each
  measurement taken on it, so the choice is made once and every recipe in the
  call reads it. The selection is kept in the stored result set, so a
  continuation replays it. Because the two keys are now part of an attached
  block's request identity, the request identity format moves to version 4,
  and a `request_id` stored under version 3 raises an idempotency conflict
  instead of replaying.
- A recipe now names the number a reduction or a spec reads once, as `field`.
  It replaces `reduce_field` and `spec.field`, which specified the same thing
  and had to agree; `spec` keeps `min`, `max` and `allow_incomplete`. A
  multi-field recipe requires `field` as soon as `reduce` or `spec` is given;
  a keyed recipe requires it for `spec`, and where it is given it narrows
  that recipe's reduction to the named key too; a scalar recipe takes none.
- The published JSON Schema no longer includes `"default": null` annotations
  or `discriminator.mapping` tables, because `required` and each branch's own
  `const` already express both. The set of valid calls is unchanged, and
  together with the argument removals above, the tool listing is smaller in
  both modes.

- Calling a tool name the server does not have now returns a JSON-RPC
  invalid-params error (-32602) naming the unknown tool and listing the eight
  that exist, instead of an error-flagged tool result. A lookup failure has no
  tool to attribute a result to, and this matches how an unknown resource URI
  is already handled.

- Moved to the MCP Python SDK 2, which serves protocol revision 2026-07-28
  alongside the older initialize handshake. Clients on either revision are
  served the same tools.
- A tool that rejects its arguments now reports
  `Invalid arguments for <tool>: ...` instead of
  `Input validation error: ...`. The SDK stopped validating a call against
  the published schema, so the tool's own model reports it; both are
  generated from that model, so nothing is checked less strictly.
- Reading a resource URI the server does not serve now returns the JSON-RPC
  invalid-params code (-32602). The 2026-07-28 revision dropped the separate
  resource-not-found code earlier revisions used.

- `analyze_results`' description names every recipe with the plain words a
  caller searches for, so a host matching a request against tool descriptions
  can route "phase margin", "distortion" or "bias point" to this tool. The
  handshake instructions name the reference lookup.
- A job whose owner process has exited but has not yet been collected by its
  parent now reads as interrupted rather than running. The liveness check
  used to test only whether the pid existed, and a finished child keeps its
  pid until the process that started it collects it.

- Every tool's response is built from one envelope (`outcome`, `failures`,
  `observations`, `warnings`, `hint`), `outcome` is set by the same rule on
  every tool (`docs/design/mcp_surface.md`), and a finding's location has the
  same fields on every tool. The advertised schemas did not
  change; the internal modules behind the tools were split so that each tool
  is its own file.

- The server now uses one on-disk store. Everything the server writes under
  a working directory's `.ltspice-mcp/` is laid out by one `Store`, stamped
  with a single `store_version`. Its directories are `experiments/` (job
  records, a request index, a per-circuit index, cancellation markers),
  `runs/{job_id}/` (every artifact one job produced, with its staged decks),
  `results/`, `detached/`, `verify/` (with its `renders/` beneath it),
  `edit-exports/`, and `locks/`. The pointer files written beside each
  circuit, which let a lookup by circuit find the working directory's jobs,
  are removed; the store's own per-circuit index does that lookup.

- **The `.asc` edit engine moved into the core.** It is now
  `ltspice_mcp.lib.schematic_ops`, and every name another module imports from
  it is public. It was in the tool layer, so core modules imported back up
  into the tools package (a latent import cycle that depended on import
  order). Nothing about `edit_schematic`, `inspect`, or `verify_circuit`
  changes. The one visible difference is in the JSON Schema `edit_schematic`
  advertises, where the internal `$defs` keys for the op shapes lost their
  leading underscore (`_OpAddComponent` is now `OpAddComponent`). References
  resolve as before.
- `jobs` publishes each action's own argument shape. `status`, `wait`,
  `cancel`, `list`, and `runs` are separate branches of one schema keyed on
  `action`, instead of nine optional fields that the server checked after the
  fact. Every call that was accepted before is still accepted, unchanged. A
  call that is not accepted now gets an error naming the legal actions, or
  the exact field that action does not take, before it is dispatched.
- `run_experiments`' `analyze.recipes` is validated against the same typed
  recipe grammar `analyze_results` takes, so a malformed recipe is refused
  before any deck is staged instead of after the simulation has run. The
  schema advertises the metric names and points to `analyze_results` for the
  fields each one takes, instead of including a second copy of the grammar
  that every client would download in every session.
- One `jobs` call reads its job once. The MCP response and the in-process
  API result are built from the same evaluation, so they can no longer
  describe the job at two different times.

- Every error class declares a stable `code`. No code a client sees has
  changed; the full vocabulary (146 codes) is pinned by
  `tests/test_error_codes.py`, and renaming or removing one is a breaking
  change that will be listed here. See `docs/design/mcp_surface.md`, "Error
  codes".

- The `mcp` dependency no longer pulls in the `cli` extra, which removes six
  packages (typer, rich, markdown-it-py, mdurl, shellingham, annotated-doc)
  that the server never imported.
- New documents `THIRD_PARTY_NOTICES.md`, the tool-surface and Python API
  contracts under `docs/design/`, and the spicelib bug ledger, published as
  `docs/spicelib_bugs.md`.
- The advertised tool definitions match the source exactly. Every
  description declared on a tool or one of its arguments is served verbatim on
  the `full` listing, and the `compact` listing removes descriptions rather
  than rewording them. The models therefore show what a client is shown, and
  the same text also supplies `api.reference('...')` and the `spice://guide`
  resource. The listing stays small because the descriptions themselves are
  short. Each states the unit, the convention, the default, and how the field
  interacts with its siblings, and the fuller explanation is in
  `docs/design/mcp_surface.md` or the guide, with a pointer on the field.
  Output schemas are still not advertised, so response shapes are learned
  from responses, and leaving them out offsets the size of the argument
  descriptions. `tests/test_consolidated_contracts.py` sets an upper bound
  per tool, so the listing cannot grow unless someone raises a limit.
- Three `analyze_results` recipes — `noise_integral`, `periodic`, and
  `return_loss` — advertise only their `metric` and a one-line documentation
  pointer, about 700 characters less schema per session. They remain fully
  callable with every field they had; the full argument trees are in
  `api.reference('analyze_results')` and the `spice://guide` resource.
- `import ltspice_mcp.api` and `Api()` boot lazily. The package `__init__` is
  a PEP 562 lazy table, `SessionState` builds its tool surface on first
  access, the API method layer resolves tool modules through deferred
  imports, and scipy is imported at its call sites. A cold `Api()` drops from
  about 1.5 s to about 0.4 s and no longer imports scipy or the MCP SDK.
  Method docstrings install on the first catalogue read or operation call;
  until then `help()` on a method shows only its signature.
- The README now describes the MCP server and the Python library as two ways to
  use one engine, opens with a runnable Python-API sweep example next to the MCP
  quick start, and cites published studies for the main design choices. PyPI
  metadata and the Desktop-extension description now mention the Python library
  and one-call sweeps and Monte Carlo. The README also no longer states that
  all circuit editing is simulator-free, because `.asc` schematic editing
  needs LTspice's `.asy` symbol libraries; netlist (`.cir`/`.net`) editing
  needs no simulator.
- The `value` recipe returns an `exact_match` flag, `false` when the requested
  time, frequency, or sweep value snapped to a different sample. This matters
  on a coarse sweep, because a `.dc temp` run that silently snaps 27 °C to
  25 °C biases a tempco reading. The `at` field now documents that it
  addresses the run's primary sweep axis (time, frequency, or the `.dc` sweep
  variable), and `step_axis` documents that it is for stepped (`.step`)
  sweeps rather than a bare `.dc` or `.ac` primary axis. The no-step error now
  points to `at` instead of a bare "axis not found".
- The `set_component_value` op warns when a GUI opamp complexity label such as
  `Level.2` is written to a subcircuit's Value. LTspice emits it as a stray
  positional token, which produces a "sub-circuit name is not defined" error
  at netlist time. The warning now applies to any Value written to a symbol
  whose model is chosen through `SpiceModel` (for example `UniversalOpamp2`),
  not only to the `Level.N` label, because any value there breaks the netlist
  the same way. A library part that has its subckt name in Value, with no
  `SpiceModel`, is not affected.
- The `add_component` op rejects an unknown attribute name — a typo such as
  `Val` for `Value` — instead of silently dropping it at export time, matching
  `set_component_attribute`.
- The `thd` recipe labels its per-harmonic magnitudes with the signal's native
  `unit` (V or A); the ratios, percentages, and dB values stay dimensionless.
- A recipe that needs a sweep axis, run on a stepped `.op` whose raw file
  holds only step 0, now ends its no-axis error with the fix: convert the
  `.step` to `.dc <param> START STOP STEP`, as the `summary` recipe's hint
  already said, or use `operating_point` for a single bias point.
- The SPICE guide has several new sections and corrections. They cover the
  two-directive `.meas` argmax pattern (finding the frequency or time *of* a
  maximum), the dBΩ reading of an impedance trace under the 1 A probe, how to
  export quantized and staircase waveforms, and a cross-link to the
  `return_loss` recipe.
  `ac_structure`'s `net_order` is documented as a real computed order — 0 or
  negative is possible — never a sentinel. It adds notes on the three- versus
  four-terminal transistor symbols (`nmos4`, `pnp4`, and the rest), the diode
  symbol's built-in default-`D` model collision, and the `AC <mag>` small-signal
  source syntax; the `.meas MAX` signed-trace case, where `MAX I(...)` on an
  always-negative current returns the least-negative sample rather than the peak
  magnitude, so wrap it in `abs()`; and how to steer bistable circuits —
  bandgaps, mirrors, latches — to the intended DC root when `.nodeset` alone will
  not. Its named-nets rule now matches the runtime guidance. Repeating a net
  label with the same name correctly ties distant pins, because the netlist
  merges same-name labels into one net, and with duplicate labels a wire
  should target a component pin rather than `net:NAME`.
- The duplicate-label error names the actual net in its guidance instead of a
  canned `net='0'` / `M3.S` example.
- The schematic layout guidance in `spice://guide`, the LTspice skill, and the
  new-sheet checklist now recommends delegating placement and wiring to a
  separate agent when the harness supports one, because the build is mechanical,
  detailed work that tends to end up as net labels everywhere when done inline.
  It also asks for a verification step before that agent returns, in which
  the `verify_circuit` export must match the source netlist and
  `inspect(kind: "net")` must report no multi-label shorts.
- Auto-generated and example config files no longer write a live
  `max_parallel = 4` key; it is now a comment documenting the real default,
  which is the number of CPU cores capped at 8. The written key limited every
  auto-generated setup to 4 concurrent simulations on multi-core hosts.
- The `format` parameter description states that `'text'` is the default and
  that structured content is identical for both formats.
- The `operating_point` recipe always includes `warnings`, as `[]` on a clean
  run, instead of omitting the key.
- The `circuit-mcp` and `ngspice-mcp` alias packages now publish only after
  the test workflow passes (before, nothing gated them). They pin
  `ltspice-mcp` to exactly the co-released version at build time and wait for
  that version to be installable from PyPI first, so a published alias can no
  longer resolve an older canonical package under a newer alias version.
- The README, changelog, security policy, packaging text, developer docs,
  skills, and packaged guide were rewritten in plain language, with shorter
  sentences, no slogans or metaphors, and internal shorthand replaced by the
  terms it stands for. Facts, code, and tested instructions are unchanged.

### Fixed

- On Windows, `edit_schematic` wrote the sheet with `\r\n` line endings while
  reporting the sha256 of the `\n` text it had hashed, so the revision it
  returned never matched the file and every later `expected_sha256` was
  refused as a conflict. The staging descriptor is now opened in binary mode
  (`os.open` defaults to text mode on Windows); the bytes on disk are the
  bytes that were hashed.
- On Windows, a `.include` or `.lib` reference by drive letter
  (`C:\Users\...\standard.mos`) was resolved to `/mnt/c/...`, the WSL spelling,
  so the include could not be found, staged, linted or targeted by a
  variation. The `/mnt` mapping (and the UNC rewrite beside it) is WSL
  interop and is now taken only when the server runs on WSL; anywhere else
  the reference is the path it names.
- On Windows, closing or resetting the `run_code` worker raised on
  `signal.SIGKILL`, which does not exist there, so the tool's reset and the
  server's shutdown failed. The process-group kill names the signal only on
  POSIX.
- On Windows, no `plot` or `waveform` artifact was ever published, because
  the publish step flushed the finished file through a read-only descriptor,
  which Windows refuses. Every artifact handle was returned without a digest
  and pointed at a file that had been removed. The flush now goes through a
  writable descriptor.
- A staged deck writes an absolute reference in the form the platform's own
  netlister uses, with backslashes on Windows and POSIX form elsewhere (it
  used to write the POSIX form on Windows too).
- On Windows, two calls creating records in a fresh store at the same
  moment could fail one of them with "Invalid result_set_id" (or the job
  record's equivalent). The store resolved the new record's path while its
  peer was creating the directory, and Python's `realpath` then returns a
  path that fails every comparison. A record that does not exist yet is no
  longer resolved; the symlink check runs only on an existing record.
- On Windows, two simultaneous reads of one result set that both produced
  the same `plot` or `waveform` artifact could fail one of them with
  "access denied". The second rename onto the shared, content-addressed file
  happens while the first is still in progress, which Windows briefly
  refuses (in 11 of 15 runs of the paired read in isolation). Every atomic
  rename now retries briefly on that refusal, on Windows only.
- A path with an embedded NUL byte is refused before resolution. On Windows
  with Python 3.13 `Path.resolve` returns such a path instead of raising, so
  it used to pass the sandbox check and fail only at open time.

- The no-simulator message names the setting that points at a simulator
  executable (`LTSPICE_MCP_SIMULATOR_EXE`, `simulator.path`) on every
  platform. Only the WSL wording did, so on a native Linux, macOS or Windows
  host an agent saw "install ngspice" with no way to use the simulator it
  already had.
- A host with the `raster` extra installed but no native cairo library now
  falls back from PNG to SVG with a note, as a host without the extra does.
  cairocffi reports the missing library as an OSError from the import, which
  the optional-dependency guard did not catch, so a render call failed
  outright and the raster tests failed at collection on any machine without
  libcairo.
- A value written with the micro sign (`20µ`, `4.7µF`) parses as micro.
  LTspice's netlist exporter writes `u` that way, so an exported current
  source had no nominal that a Monte Carlo `component` rule could perturb,
  and every case failed before submission with `random_nominal_unavailable`.
  The Greek mu is accepted too.
- A compare reference written outside the sandbox is refused, and the
  refusal names the alternative, which is to pass the deck's text itself as
  `compare.reference`. A client's scratch directory is usually outside
  `allowed_paths`, and both tools now resolve the reference the same way, so
  `edit_schematic` accepts literal netlist text as its description always
  stated (it used to treat the text as a path).
- A recipe's `field` accepts the result row's own key. `stability` reports
  `phase_margin_worst_deg`, and passing that name back for a reduction or a
  spec was refused in favour of `phase_margin_deg`; either spelling now names
  the number. A key two fields read, such as `transition_time` under `edges`,
  is still refused as ambiguous.
- `edit_schematic` accepts an empty `ops` list when `return_views` is set,
  and the batch then runs as a dry run and returns the requested pin table.
  The whole-sheet table is only reachable through this tool, and reading it
  used to need a throwaway op.

- Two `run_experiments` calls sharing a `request_id` and sent at the same
  time now stage one set of decks between them instead of two. A submission
  now looks up its `request_id` before it stages and stages only if nothing
  is recorded under it, and a second call with the same id waits until the
  first has finished that step. The second call then finds the record and
  replays it (or is refused as an `idempotency_conflict`) without copying a
  deck set of its own. Both used to
  stage in full, and the losing call's `runs/{job_id}/` tree was deleted
  afterwards, so a crash in between left an unclaimed tree in the shared runs
  root.
- `edit_schematic` reports `outcome: "partial"` when a dry run's ops fail,
  when the post-commit comparison against `reference` does not match or
  cannot export, or when a requested render fails. It reported `complete`
  in all three cases while `verify_circuit` reported `partial` for the same
  conditions; every tool now sets `outcome` by the same rule
  (`docs/design/mcp_surface.md`), and a test scans every handler module for
  a hand-written outcome.
- `max_parallel_sims` now caps the simulations of each simulator across the
  server's `run_experiments` calls, instead of within each call. A deck that
  runs in its own folder, because of a relative include, has a cap of its
  own. Three concurrent calls in one server used to get
  three times the cap, and a request could raise its own share above it.

- An analysis failure is now classified as a deadline by its exception type
  instead of by whether its message contained "deadline" or "exceeded". A
  source path, a signal name, or a filesystem error containing those words (a
  full disk reports "Disk quota exceeded") was reported as
  `analysis_deadline`, which told the caller to retry a fault that more time
  cannot fix.
- Opening a schematic whose symbol, sub-sheet, or model library is missing
  reports the missing dependency. The old check looked for ".asy" in the
  editor's message, so a hierarchical block whose sheet was missing reported
  "File not found" against the schematic that had opened fine.
- Cancelling a job distinguishes an unauthorized request from any other
  cancellation failure by type rather than by the words "not authorized".

- ngspice decks that drive their own analysis from a `.control` block now get
  a rawfile through `run_experiments`. The server adds a `write` command to
  such a block, but only code that no tool called did so, so a scripted deck
  completed cleanly with nothing for any recipe to read. The case reports
  `control_write_injected` when the server supplied the write.
- A `.step` directive on ngspice is reported by the deck lint
  (`step-ngspice`). ngspice ignores the line in batch mode, so the deck ran
  once at its base value with no warning that the sweep had not happened.
  The lint rule set is versioned as `2`.
- `verify_circuit` and `inspect` relay the netlist parser's own notes (unclosed
  `.SUBCKT`, unmatched `.ENDS`, stray continuation) into `observations` and
  the netlist payload's `warnings` instead of discarding them.
- `edit_schematic` reports the warnings its ops raise. They were dropped
  before. Identical advisories across a batch now appear once, with the
  number of ops they cover.
- `edit_schematic`'s refusal of an existing target submitted without
  `expected_sha256` includes that file's current sha256 (code
  `expected_sha256_required`), so the retry is one call.

- `plot_waveform` accepts a `run_experiments` job. Pass its `job_id` with
  `run_index` (default 0), or the new `case_id`, and the chart is written next
  to the case's source circuit like any other job plot. It used to refuse the
  only job kind the consolidated tools produce, with an error that named an
  internal type. The same error, where it can still occur on the other
  job-addressed paths, now names the tools that do accept an experiment job.
- The `stability` recipe accepts `field` (for `reduce` and for `spec`) on
  `unity_gain_hz` and `dc_gain_db`, not only on the two margins. "Keep the
  unity-gain bandwidth above 2 MHz" is the most common stability spec after
  phase margin, and the recipe already reported the number per case, but asking
  it to reduce or check that number was refused. A spec on a field the loop
  never reaches is `indeterminate`.
- `verify_circuit`'s `render.delivery` description now states that an inline
  image costs about 3k tokens at the default scale, and that the cost recurs
  on every later turn of the session. The bare enum gave no reason to prefer
  the default, `artifact`, which writes the file and returns its path.
- Cancelling a job, a simulation timeout, and shutdown cleanup now terminate the
  simulator process on every supported platform and simulator combination.
  Previously only WSL with LTspice worked. Everywhere else the code called
  spicelib's `kill_all_spice`, which matches an empty process name in the pinned
  spicelib and therefore killed nothing, so a cancelled ngspice, Wine, or
  Windows-native run kept simulating to completion. The replacement kill
  requires both the simulator's executable name and the job id in its command
  line, so it cannot terminate a parallel session's simulators, which a
  name-global kill would have hit once it worked.
- Cancelling a batch while other batches with decks in different output
  folders were running could send the cancel to the wrong batch's queue. The
  running processes were killed, but the batch's queued runs kept launching
  while its status read `cancelled`. A batch cancel, explicit or at shutdown,
  now stops the queue that holds that batch's runs.
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
  per-`.asc` lock described in Added under independent server processes.
  LTspice always writes the same `.net` beside the schematic, so concurrent
  exports could otherwise corrupt the output and run the wrong deck.
- WSL interop calls (`wslpath`, `cmd.exe` environment resolution) now have a
  15-second timeout; a hung Windows-interop process previously stalled server
  startup or left a run request waiting forever.
- The `thd` per-harmonic `magnitude` is now the sinusoid amplitude in the
  signal's own units. It was the raw FFT bin, off by n_fft/2 — a 0.1 V harmonic
  read as about 819 "V" at n_fft=16384. When `thd` uses a Hann window, the
  root-sum-square over the five bins centred on each harmonic is scaled back
  to amplitude as well; for a tone centred on a bin, that sum is √1.5 times
  the amplitude. THD percentages and dBc ratios were, and remain, correct.
- Setting a `.asc` behavioral source's expression (a `set_component_value` op
  with `V=...`) corrupted the schematic. The expression was routed to
  SpiceLine as a parameter while the old expression stayed in Value, so two
  expressions were netlisted on one B-line ("No such node") while the op
  reported success. The whole value now replaces the Value slot for B
  references.
- Directives with embedded newlines corrupted the `.asc` TEXT record; they are
  now stored as LTspice's literal `\n` escapes, so multi-line directives
  round-trip.
- The error for setting the `Prefix` attribute no longer points to
  SpiceLine. Prefix is a symbol (`.asy`) property, and the message now points
  to symbol-based placement.
- The `measurements` recipe reads each `.MEAS` directive's operator from the
  deck instead of inferring it from the results. A `FIND` probe whose value is
  constant across runs is no longer mistaken for a `WHEN` crossing, which
  previously made it aggregate over the probe axis instead of over the values. A
  deck-confirmed bare `WHEN` now aggregates crossing times even when every run
  crossed at the same instant, as in a deterministic batch. The recipe also
  warns when it aggregates a still-running or partial batch ("N of M runs —
  partial, not final").
- `transient_response(mode: "step")` no longer reports a definite
  `settling_time` when the signal entered the settle band shortly before the
  window ends and the final value was derived automatically from that same short
  tail. That case cannot be distinguished from a still-ringing waveform that is
  momentarily flat, such as a transmission-line staircase. When `final_value` is
  not given, the signal must now stay in the band for at least 1.5× the nominal
  trailing window, the final 10% of the analyzed time span. The requirement is
  based on elapsed time rather than sample count, so sparsely sampled settled
  tails are not falsely suppressed. Suppressed results have the
  `settling_dwell_near_window_end` quality flag and render as "unknown"; passing
  an explicit `final_value` bypasses the check.
- Analysis window bounds a few ulps past the axis end — SI-suffix rounding, such
  as `t_end=1500u` against an axis ending at exactly 1.5 ms — are clamped
  instead of rejected with a self-contradicting "outside axis range" message.
- Repeated identical log diagnostics collapse to one entry with a repeat count.
  A PDK deck repeats "unknown model parameter" once per device instance, which
  hides the fatal line. The failure excerpt now also includes the lines around
  "File not found" and "already defined", which previously fell outside it
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
  `A/√Hz`) instead of the plain `V` its trace type declares, matching
  `noise_integral` and the raw file's own "Noise Spectral Density" plotname.
- The `spice-experiments` and `ltspice` skills stated that nothing adds
  `.options logopinfo` for you, but the server adds it to every LTspice `.op`
  run, as the packaged guide already stated. Both skills now state that the
  server adds it and that writing it yourself is harmless.
- The LTspice skill no longer states that device operating points (gm, gds,
  vth, and the rest) require ngspice. LTspice `.op` runs report them through
  the auto-injected `.options logopinfo`, and the `operating_point` recipe
  reads both engines the same way. The packaged guide was already correct;
  the skill's copy had drifted. Swept gm, the gm/ID `.dc` table, still needs
  ngspice, as both now state.

### Removed

- The `full` and `agentic` tool profiles, and 48 of the 49 tools they exposed.
  (`plot_waveform` is the one that remains.) The breaking-change section
  above lists where each capability went and how to pin the old surface.
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
  `anchors` or `rtol`; pass the `compare` object, which contains all four.
- `edit_schematic` no longer accepts the flat `reference`; pass `compare`.
- `edit_schematic` no longer accepts `render`, `render_format`, `render_scale`,
  or `render` in `return_views`, and its response contains neither
  `views.render` nor the `artifacts` array that rendering was the only
  producer for. Rendering is done by `verify_circuit`, which adds a pixel
  cap, inline delivery and a render-only mode.
- `edit_schematic` no longer accepts `write_failed_draft`. A failed batch writes
  nothing by design and the response names the stage that failed, so there was
  no draft to quarantine that the caller's own ops did not already describe.
- `edit_schematic` no longer accepts `format`; structured-aware clients render
  only `structuredContent`, and the other six tools had already dropped it.
- The AC crossing recipe's phase level is spelled `level_deg` only; the
  earlier `phase_deg` spelling, kept as an alias for callers of an earlier
  build, is removed.

- The MCP logging capability. The 2026-07-28 revision (SEP-2577) deprecates
  the `logging` server capability, the server-to-client
  `notifications/message` delivery and the per-request log-level opt-in that
  replaced `logging/setLevel`, with no replacement, and `logging/setLevel` is
  absent from that revision's schema. The server no longer advertises the
  capability, returns method-not-found for `logging/setLevel`, and sends no
  log notifications. Job lifecycle events and
  diagnostics go to the process's stderr logger, which `[logging] level`
  still controls.
- **The columns form of budget-limited rows.** A response cap (`budget`)
  used to change the row format once the cap got small enough. Rows became
  arrays of bare values, with a sibling `*_columns` list naming each
  position. The format was lossless, but a row's shape then depended on the
  cap's size, so a reader first had to work out which form had come back.

  Rows now keep their shape at every budget. A row is always an object with
  the same keys, and a tight budget returns fewer rows instead of
  differently shaped ones. A response over its budget first empties its
  presentation blocks (trim), then drops the extra detail the caller asked
  for (answer), then returns fewer rows (shrink). The columns keys
  (`items_columns`, `values_columns`, `reduced_columns`, and every other
  `*_columns` sibling) are removed from every response and output schema.
  Code that read the columns form should read the rows from `items`,
  `values`, or `reduced` and page on with the cursor the response includes.
- **The pre-0.6 tool handlers and job machinery.** The handlers behind the
  removed 0.5 tools were kept in place through 0.6 development so a tool
  could be re-exposed by putting its decorator back. That option is removed.
  Each handler with no caller has been deleted, along with its argument
  model, its output schema, and the helpers only it used. The eight tools and
  the Python API are unaffected; the advertised schemas and
  `ltspice_mcp.api.__all__` are unchanged. Restoring one of the old tools now
  means restoring its module from v0.5.x.

  The single-simulation and batch job types, the sweep and Monte Carlo
  runners, and the batch result reader are also removed. Every job the server
  runs is an experiment, on one runner. The Monte Carlo perturbation engine
  is unchanged; `run_experiments` draws its random variations from it.

  The removed error codes, which only those handlers emitted, are
  `legacy_analysis_result`, `case_selection_wrong_job_kind`,
  `run_unavailable` (now `case_not_found`),
  `run_failed`, `no_raw_output`, `parse_deadline` (now `analysis_deadline`),
  `decimated`, `window_applied`, `complex_format_used`, `unrecognized_save`,
  `max_pk_pk_bucket`, and `export_written`.
- **The job sidecars earlier releases wrote.** Releases before 0.6 wrote a
  job record beside each circuit, at `.ltspice-mcp/jobs/<job_id>.json`. Those
  files are no longer read, so a job id that only one of them names is not
  found, and `jobs` reports it as it would any other unknown id. 0.6 never
  wrote there, and the files themselves are left in place; delete them by
  hand to recover the space. The `legacy_job_record` observation code is
  removed along with the code that read them.
- `CONTRIBUTING.md` and the code of conduct are removed, and the sdist no
  longer ships them. The project is not taking outside contributions at this
  stage. The setup notes moved into `CLAUDE.md`, and the review rules
  `CONTRIBUTING.md` contained (a regression test fails before the fix, real
  code paths, plain language) were already in `CLAUDE.md` and
  `docs/TESTING.md`.

### Security

- Dependency upgrades for published advisories in the locked runtime set:
  `click` 8.3.2 → 8.5.0, `cryptography` 49.0.0 → 50.0.1, `pillow` 12.2.0 →
  12.3.0.

## [0.5.0] - 2026-06-30

### Added

- A new `[simulator] ngbehavior` option (and `LTSPICE_MCP_NGBEHAVIOR` env
  var) sets the ngspice compatibility mode that spicelib passes to ngspice at
  startup. The shipped default is unchanged. spicelib's default mode reads a
  sectioned `.lib <file> <section>`, the standard PDK corner-select syntax, as
  two plain includes and drops the section, so the corner is not found. A
  mode without the `lt`/`ps` compatibility tokens (e.g.
  `ngbehavior = "hsa"`) parses it correctly. When an ngspice run fails with a
  missing-include error while the deck uses a sectioned `.lib`, the run now
  reports a hint naming that fix.
- `measurement_stats` now echoes the reported measurement time (the `AT` /
  crossing value) on single-run reads, matching the aggregate-mode output.
- The bundled SPICE guide (`spice://guide`) has a new RF / two-port section.
  It covers a one-port impedance probe (with the current-source sign
  convention that keeps `V(node)` equal to `+Zin`), reflection coefficient /
  return loss / VSWR, a noise-figure formula, and insertion loss, each
  verified against a real ngspice run.

### Changed

- `create_schematic` now accepts the optional `format` (`"json"`/`"text"`)
  parameter its sibling tools take, returning structured content, instead of
  rejecting it as an unknown field.
- The duplicate same-name net-label warning no longer reads as a short. It
  now states that same-name labels merge into one net (a valid way to tie
  distant nets) and that only a later `connect(net=...)` is ambiguous; the
  `create_schematic` checklist states the same.
- Error messages now point to the right tool. `get_waveform` on complex AC
  data names `export_waveform` and `resonance`, and `find_model` refers to
  `symbol_info` when the queried name is a schematic symbol rather than a
  library model.

### Fixed

- The operating-point log classifier no longer reports a converged run as
  failed. LTspice escalates the OP solve (Direct Newton → Gmin → source stepping
  → pseudo-transient) and logs "<method> stepping failed to find operating
  point" for each method it abandons before a later method converges. Those
  intermediate failures were classified as errors and copied into every
  analysis tool's warnings. A failed method is now an error only when its own
  solve block never converged, and each solve block is judged on its own, so
  a converged step in a stepped `.op` no longer masks a failed later step. A
  no-data run (no success line) still classifies as an error.
- Every `.asc` write path (`add_component`'s value and attributes,
  `set_component_attribute`, and the `apply_schematic_ops` op) now refuses an
  empty or whitespace-only attribute value up front. An empty `SYMATTR` value
  writes a two-token line the schematic parser cannot read back, which
  corrupted the file.
- `pulse_response` now interpolates the settling-band crossing between the last
  out-of-band sample and the first in-band one instead of snapping to the
  in-band sample time (which can be up to a full timestep late on a coarse
  run), and warns when the local timestep near settling is coarse.
- `set_component_value` on an `.asc` now creates the `Value` line when the
  component exists but has none (it was added without a value), symmetric with
  `add_component(value=)`, instead of failing "not found".
- Transient reads now report a `.meas WHEN` crossing time that the solver
  dropped, and flag a trace that diverges without reaching the 1e8
  extreme-value threshold (a peak of at least 1e4 that is at least 1e5 times
  the trace's median), instead of silently omitting them (`run_simulation` /
  `simulation_summary` / `measurement_stats`). A behavioral
  (B-source) expression with unquoted internal spaces is now a warning instead
  of a hard error.
- The published output schemas no longer mark optional (`NotRequired`)
  fields as required, so a client that checks replies against them no longer
  rejects a valid reply. Schema generation also refuses a tuple of mixed
  types, which it cannot describe correctly, instead of publishing a wrong
  schema.
- `pulse_response` no longer reports a false `settling_time` when the trailing
  window is still ringing (the automatic final value falls on a ripple sample,
  so a settle band based on it produced a definite but wrong time). That state
  now renders as `unknown` with a distinct quality flag, kept separate from
  "never settled within the window".

## [0.4.1] - 2026-06-27

### Fixed

- Cancelling or timing out a simulation could leave the simulator process
  running while the job was reported as cancelled, when the run's output was not
  in the default folder (e.g. a deck in a subdirectory). `cancel_job` now
  finds the run through the job's own netlist, so it stops the process the
  job launched.

### Changed

- The default `ltspice-mcp.toml` is written lazily on the first tool call rather
  than at server startup, so the server no longer writes a config file into
  every directory an MCP client launches it from, only into directories where
  its tools are used.
- Simulation artifacts now go to a stable `.ltspice-mcp/runs` folder under the
  working directory instead of the project root, each named per job so
  results stay isolated. A deck with a relative `.include`/`.lib` still runs
  in its own directory so the include resolves, and on WSL with LTspice the
  output is routed to a Windows-native temp directory so `.MEAS` results are
  kept (LTspice can't write its SQLite `.db` over a UNC share).

## [0.4.0] - 2026-06-27

### Added

- Config options and recovery actions are now named in the replies where an
  agent gets stuck, instead of only in docs. `server_status` reports the
  resolved config file path (honoring `LTSPICE_MCP_CONFIG`) and how to switch
  the active simulator, plus `persist_jobs` / `preload_recent_count`. A
  timeout names the `run_simulation(timeout=)` argument,
  `[simulation] timeout`, and `LTSPICE_MCP_TIMEOUT`; a path rejection names
  `[security] allowed_paths` / `LTSPICE_MCP_ALLOWED_PATHS` and the
  copy-into-sandbox fallback; an unknown tool distinguishes a profile-hidden
  tool from a bad name and points to `[tools] profile`; `.asc` export without
  LTspice names `[simulator] path` /
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
  …) are now supported on both simulators and can be read by name.
  `operating_point` reports them under `device_op_points`. On a `.dc`/`.tran`
  sweep the by-name readers (`query_value`/`signal_stats`/`export_waveform`)
  accept a uniform `dev.param` shorthand (e.g. `m1.gm`) that resolves to
  whichever wrapped form the raw file contains (`@m1[gm]` / `v(@m1[vth])` /
  `i(@m1[id])`), which is the read a gm/ID characterization needs. On LTspice
  the values are in the `.log`'s operating-point block, so `run_simulation` /
  `run_sweep` / `run_montecarlo` auto-add `.options logopinfo` to `.op` decks
  and `operating_point` merges the block in (subcircuit devices are matched by
  instance regardless of the log's colon-qualified name). On ngspice they are
  `@dev[param]` traces that must be `.save`d, and a request for a `dev.param`
  absent from the raw file returns a hint about the missing `.save`.
- `validate_netlist` and `export_netlist` now warn when a `.meas`, output
  directive, or behavioral source references `V(name)`/`I(name)` for a node or
  device the netlist doesn't define. The common cause is a schematic net that
  was wired but never labeled, so it exports as `N00x` while a `.meas V(vref)`
  still asks for `vref` and silently resolves to nothing; plain typos are the
  other cause. The set of known names deliberately over-approximates, so the
  check fires only on a name that is absent. Hierarchical refs (`V(X1:out)`)
  and expression fragments (`V(a*2)`) are not checked. `validate_netlist`
  runs it on `.cir`/`.net`; `export_netlist` runs it on the exported netlist
  (where an `.asc`'s final net names and its directives appear together).
- `bode_metrics` accepts a transfer-function ratio as its `signal`.
  `V(out)/V(mid)` divides the two complex AC traces before any mode runs, so
  inter-stage gain, loop gain, and PSRR (which the simulator never stores as a
  single trace) can be analyzed directly instead of through a deck-side
  behavioral node. It is restricted to a single two-signal quotient; all four
  modes and `all_steps` work on the ratio. A denominator that reaches zero
  (the ratio is singular, a pole) is reported with the frequencies where this
  happens instead of being silently dropped, so a hidden pole can't skew the
  metrics.
- `apply_schematic_ops` has a new `dry_run` flag, which validates the whole
  batch against an in-memory copy and reports per-op results without writing
  the file. Every op is attempted (errors don't stop the run), so one dry run
  reports all problems at once instead of rolling back a good batch. Check the
  plan, then resubmit the corrected ops with `dry_run=false`.
- New MCP prompts `characterize_filter`, `run_and_plot`, and `step_response`
  start common workflows; a host shows them as slash commands. Each returns
  the standard sequence of tool calls for that task with the circuit path
  filled in.
- The server is distributed as a Claude Code plugin (`.claude-plugin/`) and a
  Claude Desktop extension (`packaging/mcpb/`). Both wrap the published
  package via `uv` (the plugin runs `uvx`; the extension is a `type: "uv"`
  bundle) and so require `uv` and a simulator (LTspice or ngspice) on the
  host rather than bundling either.
- A new `thd` tool computes the total harmonic distortion (THD and THD+N) of a
  periodic transient signal by FFT. It needs no `.four` directive and works on
  any simulator. It defaults to coherent sampling (the record is trimmed to a
  whole number of fundamental cycles and a rectangular window is used), so
  harmonics fall exactly on FFT bins and the result is exact;
  `window="hann"` is the approximate fallback. The fundamental is
  auto-detected (sub-bin accurate) or given. Every condition the number
  depends on is reported (the fundamental and whether it was given or
  detected, the window kind, cycles analyzed, FFT length, sample rate,
  per-harmonic levels), and the tool warns when a window turns out
  non-coherent or the FFT-length cap forces a down-sampling that could alias.
- A new `noise_integral` tool integrates a `.noise` spectral density to a
  total RMS over a band as `sqrt(∫ density² df)` (the amplitude-density
  convention shared by LTspice's `V(onoise)` and ngspice's
  `onoise_spectrum`). It reports the band integrated and the sample count,
  and handles a high→low sweep. Noise figure / SNR are left to the caller
  (they need the source resistance and a reference level).
- `operating_point` has a new `device=` filter that returns one device's
  operating-point params and terminal currents (e.g. `device="M1"` → `@m1[...]`
  plus `Id/Ig/Is(M1)`) in a single call, refusing an unknown device with the list
  of devices present. Every returned value now has its SI unit in a `units`
  map where the simulator declared the trace type.
- `query_value` and `export_waveform` now attach SI units derived from the
  simulator's declared trace type (`query_value` returns a `unit`; the CSV's
  value columns are unchanged but the DC x-column is named — see Changed).

### Changed

- `max_parallel_sims` now defaults to the host core count capped at 8 (it was
  a flat 4, which limited a many-core machine by default). Still overridable via
  `[simulation] max_parallel` / `LTSPICE_MCP_MAX_PARALLEL` up to 128; the cap
  keeps parallel cold simulator processes from thrashing memory/IO.
- Sweep cross-products are now capped at 10000 runs, matching the existing Monte
  Carlo cap. A multi-axis sweep (e.g. 5×5×16×100) previously spawned tens of
  thousands of cold simulator processes silently; it's now refused at
  `configure_sweep` with the offending dimension sizes. The cap is computed from
  each dimension's *count* before any value list is built, so a single large
  dimension (`points=1e9`, or a tiny `step` over a wide range) is rejected
  without `np.linspace`/`np.arange` allocating a multi-GB array first.
- **Breaking:** MCP resource URIs moved from the `ltspice://` scheme to
  `spice://` (`spice://results/...`, `spice://netlists/...`, `spice://config`,
  `spice://guide`, etc.). These resources are engine-agnostic, so the scheme no
  longer implies LTspice. Clients discover resources via `resources/list` and
  the URI templates, so this only affects anything that hard-coded the old
  scheme. The MCP Apps widget keeps its spec-mandated `ui://` URI.
- Generalized the `spice://guide` resource to cover both engines. It now
  contains engine-neutral SPICE fundamentals plus separate
  **LTspice-Specific** and **ngspice-Specific** sections (ngspice `.control`
  scripting, `.save`, XSPICE, control-loop Monte Carlo, expression parsers)
  and an LTspice-vs-ngspice differences table. The per-engine Claude Code
  skills stay engine-specific.
- The MCP handshake now reports the `ltspice-mcp` package version in
  `serverInfo.version` instead of the MCP SDK's version.
- `serverInfo.name` can be overridden with the `LTSPICE_MCP_SERVER_NAME`
  environment variable (defaults to `ltspice-mcp`), so alias launchers can
  self-identify in the handshake.
- The server instructions now open with a line naming the simulators
  detected at startup (e.g. "Active simulator: ngspice (LTspice not detected).")
  so a client without LTspice no longer reads the LTspice-centric name as a
  degraded state. The instructions also now point to `plot_waveform` for
  visualization.
- When no simulator is detected, the instructions, every "no simulator" tool
  error (run / sweep / Monte Carlo / batch), and `server_status` now give a
  platform-appropriate way to get one (install ngspice, or point
  `LTSPICE_MCP_SIMULATOR_EXE` at an existing binary), note that the server must
  be restarted to re-detect, and tell the agent to ask the user if it cannot
  install. Before, they gave only "check server status", with no next step.
  When ngspice is present but LTspice is not, the instructions now state that
  `.asc` schematic editing is unavailable while simulation and analysis still
  work.
- Reworded the "symbols unavailable" startup log so it reads as
  informational, because missing symbols disable only `.asc` schematic
  graphics editing and leave SPICE simulation and netlist editing unaffected.
- `signal_stats` and `get_waveform` descriptions now read "SPICE's adaptive
  timestep" rather than "LTspice's", because the behavior also applies to
  ngspice.
- Lowered the minimum Python from 3.13 to 3.11. The two 3.12+/3.13-only
  constructs (PEP 695 generic syntax, `tomllib`) are replaced or already
  available on 3.11, so 3.11 and 3.12 users can now install. CI (and the
  release gate) now runs the full test suite on 3.11, 3.12, and 3.13, so the
  advertised minimum version is tested on every push.
- **Breaking:** `measurement_stats` renamed its `best_step_index` /
  `worst_step_index` fields to `min_step_index` / `max_step_index`. "Best" and
  "worst" implied a verdict the tool can't justify, because whether a low or
  high value is "best" depends on what was measured. The fields now name the
  plain fact (the step index where the min / max value occurred). No alias is
  kept (this is a pre-1.0 breaking change); the structured `outputSchema`
  advertises the new names.
- `query_value` and the `export_waveform` CSV now label a `.dc` sweep axis by
  its swept variable (e.g. `Vin` / a `Vin_V` CSV header) instead of a misleading
  `t=` / a bare `sweep` column.
- The "device operating-point param not found" hint is now an instruction and
  also applies to a bare `@dev[param]` request. It tells you to add
  `.save @dev[param]` and re-run with ngspice, instead of noting that the
  value exists elsewhere.

### Fixed

- The in-tool SPICE guide and the run-time pre-flight warning gave the wrong
  ngspice `.meas` recovery. ngspice suppresses `.meas` only under `-b` plus a
  command-line `-r rawfile` (this server's invocation). The documented
  workarounds were both wrong. Inside a `.control` block ngspice requires the
  dot-less `meas` command (a dotted `.meas` there errors and computes
  nothing), and `measoutfile` has no effect once the measurement is
  suppressed. Both are now corrected and verified against real ngspice.
- `validate_netlist` now flags `.backanno` when targeting ngspice, because
  ngspice has no such command and aborts the run with "unimplemented dot
  command '.backanno'" (it is an LTspice-only schematic directive). This
  matches the existing `.tran 0` ngspice-incompatibility check; an LTspice
  target is unaffected.
- The batch result reader no longer silently returns the wrong sample on a
  descending sweep axis. A `.dc V1 5 0 -0.1` (or any high→low parameter sweep)
  produces a descending axis, but the `at=` slice used a binary search that
  assumes ascending order, so asking for the value at one axis point could
  return the value at another. The axis and its wave are now flipped to ascending
  before the search; ascending sweeps are unaffected.
- `batch_results` now reports per-step data it cannot aggregate instead of
  silently dropping it. A run whose `.raw` contains its own `.step` sweep is
  read at step 0 only; those runs are now reported in `step_collapsed_runs`
  with a note pointing to `get_waveform`/`query_value` (`job_id`,
  `run_index`, `step=<n>`) for the remaining steps. If step metadata can't be
  read at all, the run is reported in `step_unknown_runs` instead of being
  silently assumed single-step. Step 0 is still returned, so the readable
  data is never dropped.
- `configure_sweep` now warns when a parameter axis is named
  `temp`/`temperature`, because it is emitted as `.param temp=…`, which does
  not set the simulation temperature (SPICE controls that via `.temp`,
  `.options temp`, or `.step temp`). A "temperature sweep" would otherwise run
  silently at a single temperature.
- `batch_results`' `raw=true` mode is now documented correctly. It returns
  per-run *reduced* rows (a single `value`, or peak/mean/min) rather than raw
  sample vectors. For the samples themselves (e.g. a gm/ID table) use
  `export_waveform`/`get_waveform` with `job_id`+`run_index`.
- `query_value` (raw and job-run modes) and the step-by-axis-value lookup now
  resolve the nearest point correctly on a descending sweep axis. They used the
  same binary search as the batch reader, which assumed an ascending axis, so
  a high→low `.dc`/`.step` lookup could return the wrong point.
- `operating_point` now includes the simulator's "unrecognized variable"
  warning. A `.save`d `@dev[param]` the device class doesn't have (a typo, or
  an unsupported parameter) is written to the raw file as a plausible `0.0`,
  which was indistinguishable from a true zero. The log warning that marks
  the value as invalid is now reported alongside the value.
- `validate_netlist`'s dangling-reference check no longer false-flags simulator
  reserved traces (`onoise`/`inoise`/`time`/`frequency`/…) or probe references
  inside `.meas` / output directives as undefined nodes.
- `periodic_metrics` warns when edge spacing is strongly bimodal, which
  indicates a frequency/duty reading that is off by ~2x because alternate
  edges were miscounted.
- Schematic guidance no longer tells you to leave every signal net unlabeled.
  `connect` wires pins but assigns no net name, so an unlabeled net exports as
  `N001`/`N002`/…, which silently breaks any `.meas V(vref)`, `.param`
  expression, or behavioral `B`-source that references the net by name. The
  `create_schematic` checklist, its tool description, and the `spice://guide`
  "Named nets" section now state that wiring alone is fine for nets you never
  name, but any net a directive references by name must be labeled with
  `add_net_label`.
- A cancelled or timed-out simulation now has its partial output deleted
  instead of left on disk. A timed-out LTspice run can keep its `.raw` open
  and reach several GB; before, that file (and the run netlist/log) was never
  removed. The server now deletes the run's artifacts once the killed
  process has exited and released the file. Cleanup runs only for the killed
  statuses, so a completed run's good output is never removed.
- `get_waveform` overview buckets are capped at 2000 (it was the
  `max_points_returned` limit, which is sized for one-value-per-point arrays).
  Each bucket has ~8 scalar fields, so requesting the old maximum exceeded the
  MCP response budget and wrote the reply to a file, at the tool's own
  documented limit. The default (200) is unchanged.
- Clarified the `run_simulation` `timeout` help. With `wait=true` the
  effective limit is `min(timeout, 600s)`, where 600s is a maximum and not a
  minimum, so the default 300s timeout limits a `wait=true` run unless a
  larger timeout is passed.
- `add_component` (and `apply_schematic_ops` add-component) no longer fail with
  an unexplained "Internal error" in two cases:
  - **Vendor symbols with non-ASCII descriptions.** Hundreds of LTspice's
    bundled `.asy` symbols (op-amps, comparators, ADC/DAC, …) contain cp1252
    bytes (`µ`/`°`/`±`/`©`) in their description fields. The symbol parser read
    them as strict UTF-8 and raised `UnicodeDecodeError`; it now uses the
    shared encoding fallback (BOM/UTF-16/cp1252), and any `.asy` that still
    cannot be parsed is reported as "symbol unavailable" instead of crashing
    the tool. The same strict-UTF-8 problem was also fixed in related reads,
    which are `export_netlist`'s `.net` read and the two `.log` reads that
    lacked the tolerant decoding the rest of the log parser already used.
  - **spicelib 1.6.** spicelib 1.6 made `Component.attributes` a lazy property
    that raises for a from-scratch component built as a bare
    `SchematicComponent`; construction now uses `AscComponent` (the type
    spicelib's own parser builds) when present. The dependency is also capped to
    the tested range (see below), so a default install keeps working.
- The parsed-result cache is now bounded (LRU, 32 entries). It was unbounded, so
  a long-lived session querying many circuits kept every `.raw` it ever parsed
  (each potentially several MB) in memory. The editor cache stays unbounded by
  design, because it can hold unsaved in-memory edits that eviction would
  drop.
- A netlist with a relative `.include`/`.lib` now also keeps its run in the
  working dir on WSL with a Linux-filesystem working dir. In that case the
  server moves a run's artifacts off the UNC path, but it did so before
  checking for a relative include, so the include could not be resolved.
  Self-contained decks are still moved.
- Simulation artifacts no longer fill the working directory. Runs that
  previously wrote their `.raw`/`.log`/`.db`/`.op.raw`/netlist files directly
  into the project root (the non-WSL-Linux and WSL-on-`/mnt/` cases) now write
  to a single `.ltspice-mcp/runs` folder beside the job metadata, so they are
  in one place to find or delete (a 30-run Monte Carlo alone writes ~180
  files). The folder stays on the working dir's own filesystem, so a
  Windows-native (`/mnt/`) working dir keeps LTspice's `.db`/`.MEAS` support.
  A netlist that includes a sibling file through a *relative*
  `.include`/`.lib` keeps its artifacts beside the netlist (a relocated copy
  couldn't resolve the dependency); self-contained and
  lib-path/absolute-include decks use the `runs` folder.
- Unexpected tool errors now report the exception type and message
  (e.g. `Internal error in foo: KeyError: 'bar'`) instead of "Check server
  logs for details". An MCP client cannot reach the server's stderr
  traceback, so the reported cause is needed to diagnose a failure. The full
  traceback still goes to the logs.
- `edge_metrics` no longer emits a spurious "window shows falling transition"
  warning when an explicit `edge` direction is requested over a window that
  captures a full pulse (rise *and* fall). The endpoint-derived direction is
  meaningless there; the requested direction is used, and an absent edge
  still raises a clear error.
- `export_waveform` CSVs are no longer corrupted on Windows. The atomic writer
  opened text files without disabling newline translation, so `csv.writer`'s
  `\r\n` line terminators were doubled to `\r\r\n`, which put a blank row
  between every data row. The writer now opens text mode with `newline=""`,
  which also gives every text artifact consistent `\n` endings
  cross-platform. (Latent on Linux, which does no newline translation.)
- `export_waveform` / `plot_waveform` no longer double-nest their output
  folder when the source `.raw` is already inside a `.ltspice-mcp/` tree (a
  job-run raw passed by path was written to
  `…/.ltspice-mcp/runs/.ltspice-mcp/waveforms/`). The artifact now goes into
  the existing tree (`…/runs/<job>/waveforms/`).

### Changed

- `apply_schematic_ops` now accepts the `format` parameter (`"json"`/`"text"`)
  that the other structured tools take, instead of rejecting it with a
  validation error.
- Capped `spicelib` to `>=1.4.9,<1.6` (the tested range). spicelib 1.6
  refactored the editor model (most visibly, `.PARAM` values are now returned
  typed rather than as strings), and the parameter read/edit path doesn't
  handle this yet. Without an upper bound, an unpinned install
  (`uvx ltspice-mcp`) resolved to the newest release and shipped an untested
  spicelib. The component-construction part of 1.6 support is already done
  (`AscComponent`); the cap will be removed once the parameter path is
  migrated.

## [0.2.1] - 2026-06-16

### Added

- `validate_netlist` rejects empty / whitespace-only netlists (error) and
  warns on single-connection (dangling) nodes in `.cir`/`.net` decks. The
  dangling-node pass is scope-aware (`.subckt` bodies counted separately,
  header ports count as a connection), excludes ground and `.global` nodes,
  and counts terminals conservatively, so nodes referenced only by
  expressions (`V(...)`/`I(...)`), unrecognized elements, or non-terminal
  tokens never get a warning. `.asc` schematics are exempt (their
  connectivity is in wires and flags, which this pass cannot read).
- `validate_netlist` warns on degenerate bias topologies in `.cir`/`.net`
  decks, meaning a net with no DC path to ground, whose operating-point bias
  is undefined. Examples are a floating MOSFET gate, an all-capacitive
  (AC-coupled) island, a node driven only by a current source (independent,
  controlled, or a behavioral `B… I=` source without `Rpar`), or a
  galvanically-isolated domain. The check is built by conservative
  over-connection (only capacitor dielectrics, MOSFET gate oxides, and
  current-source branches count as DC opens; bias-dependent diodes,
  transistor channels, and switches all conduct), so a warning is provable
  rather than a guess, and each physically-contiguous floating domain is
  reported once. A subckt that grounds a port through node 0 internally
  biases the net wired to it; a `.GLOBAL` rail is a ground reference only
  where it reaches node 0 somewhere in the deck (a floating global is
  flagged). Messages state only the topology fact and give no convergence
  verdict. `.asc` is exempt, as for the dangling-node pass.

### Changed

- Heavy blocking work no longer stalls the server. `.raw` parses, batch
  result/log loops, the recent-circuits index lock and durable write, WSL
  `cmd.exe` interop resolution, and MCP resource reads now run in worker
  threads, so concurrent requests, including `cancel_job`, stay responsive
  while a multi-second result parse is in progress. (The MCP SDK dispatches
  requests concurrently; previously any large parse froze every in-flight
  request until it finished.)
- CI: the push/PR gate (`ci.yml`) and the release gate (`publish.yml`) now
  call one shared reusable workflow (`checks.yml`), so the two gates cannot
  diverge. GitHub status-check contexts are renamed to `checks / check` and
  `checks / audit`.
- Monte-Carlo per-run parameters in `batch_results`/`measurement_stats` are
  now JSON numbers (the perturbed magnitudes), matching the sweep runner,
  which already emitted numbers. Before, Monte Carlo emitted them as strings,
  so the same field had two types across the two batch run kinds.
- The completion value-scan (which reports NaN, Inf and extreme magnitudes)
  now runs on every result up to a total-sample budget (axis points × trace
  count, ~5M samples) instead of only on single-point operating points. A
  normal `.tran`/`.ac` is now fully scanned, so a degenerate value is reported
  and the `value_scan_skipped` observation no longer fires on nearly every
  multi-point run. Only a result whose traces exceed the budget (a long
  transient or a wide node dump) reports the skipped scan. The budget counts
  total samples, not axis points alone, so a wide result can't force every
  trace into memory on the completion path.
- `bode_metrics` filter mode now references the `-3 dB` cutoff to the passband
  *plateau* gain (the flat DC-side edge for a lowpass, the high-frequency edge
  for a highpass, the peak for a bandpass) instead of the band median. The band
  median was lowered by the roll-off knee inside the auto-detected band,
  which biased the reported cutoff outward (several percent on a sweep that
  starts only a decade below the cutoff); the plateau reference removes that
  bias. The explicit `passband_range` path is unchanged.

### Removed

- The `[plotting]` config section (`plot_dpi`/`plot_style`). No code path
  read it, because no plotting feature exists. Existing TOML files that still
  contain the section parse fine (it is ignored).

### Fixed

- `docs/DESIGN.md`, the example TOML, and the skills docs now match the
  shipped tools. They use current tool names throughout and describe the
  real diagnostics and observations mechanism, the real mutation-safety
  behavior (validate-before-write refusals, `reset_schematic` snapshots,
  `export_netlist` diff; no `dry_run` and no inline plot images), the exact
  agentic-profile tool set, and ngspice's full support. Doc-drift tests now
  check tool names in README/DESIGN/skills against the live registry.
- `measurement_stats` on a batch job with no `.MEAS` results now relays the
  per-run log diagnostics that explain why (e.g. ngspice's "No .measure
  possible in batch mode"), capped at 8 distinct lines, instead of a bare
  "No .MEAS results found".
- A truncated or corrupt `.raw` file that parses as having zero variables is
  now diagnosed as corrupt when loaded, instead of appearing later as
  "Signal not found" with an empty available-signals list.
- A failed or timed-out simulation now returns its `log_file` and `raw_file`
  paths (and a text footer naming them), so the caller can open the full log
  rather than working from the excerpt alone. The error excerpt now also
  covers the end of the log. It places a window at both the first and last
  diagnostic line, so the failing-node lines at the end of a convergence
  abort ("Timestep too small", "trouble with node", "Last Node Voltages") are
  kept even when an earlier harmless line came first. Those phrases now count
  as diagnostic lines for placing the excerpt.
- An interrupted job (the server stopped mid-run and recovered it on restart)
  no longer reports a wall-clock-to-recovery "duration … (running)" in the
  `check_job` list. Its runtime cannot be known after a restart, so the list
  now shows "unknown" and omits the duration, matching the single-job view.
- `batch_results` raw-mode rows now have a uniform shape. A full-waveform run
  whose signal is flat keeps `peak`/`mean`/`min` like its siblings instead of
  collapsing to a single `value` (the single `value` form now applies only to
  point queries, `at=` or an operating-point raw).
- `batch_results` given a single-simulation job id now points to tools that
  accept a job id (`check_job`, `query_value`) instead of `simulation_summary`,
  which takes a raw-file path and cannot be reached with a job id.
- Error hints are no longer misleading on job-state and argument-shape
  errors. Cancelling an already-finished job, `query_value` argument
  mistakes, and a `simulation_summary` build failure no longer append the
  generic "verify simulator availability" / "check signal names with
  simulation_summary" footer (the latter was self-referential). Errors about
  result content keep it.
- A bare `check_job()` whose default (queued/running) view is empty but which
  has finished jobs now reports how many are hidden and that `status="all"`
  lists them, instead of a bare "No active jobs" that read as "nothing
  exists".
- `bode_metrics(all_steps=true)` reports an identical per-step warning once at
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
  1.3.1 (CVE-2026-54282, CVE-2026-54283). All three are transitive
  dependencies of the MCP SDK's HTTP transport, which this stdio server does
  not use.

## [0.2.0] - 2026-06-10

This is the largest release to date. Every tool is renamed, the AC-analysis
tools are consolidated, ngspice is fully supported, the server has its own
Monte Carlo engine and SPICE lexer, and many correctness fixes cover
simulation lifecycle, result parsing, and schematic editing.

### Breaking

- All tool names drop the `ltspice_` prefix (`ltspice_run_simulation` →
  `run_simulation`, and so on for every tool). MCP clients already namespace
  tools by server, so the prefix was redundant.
- The four AC analysis tools `filter_metrics`, `roll_off`, `gain_at`, and
  `find_crossing` are consolidated into the new `bode_metrics` tool
  (`mode="filter" | "slope" | "point" | "crossing"`); no aliases remain.
- `measurements` is removed; `simulation_summary` now returns simulation
  metadata, structured `.MEAS` results, and Fourier data in one call.
- `model_info` is removed; `find_model(exact=true, full=true)` attaches
  the SPICE definition body to a match.
- `add_text` is removed; `edit_directive(kind="comment")` adds comments,
  and refuses comment text that looks like a mistyped directive.
- `simulation_summary` no longer reports `phase_margin` / `gain_margin`;
  loop-gain analysis is in `stability_metrics`.
- `find_model` output: the `parameters` field is split into `ports` (the
  full subcircuit port list, previously truncated to five entries) and
  `params` (model body / parameter-clause defaults).
- Run summaries rename `trace_names` to `signals`.
- Server state (the recent-circuits index) moves from `~/.ltspice-mcp` to
  `$LTSPICE_MCP_HOME`, `$XDG_STATE_HOME`, or `~/.local/state`; old index
  files are not migrated.

### Added

- New schematic tools `apply_schematic_ops` (a list of edits applied as
  one transaction instead of 25+ round-trips), `create_schematic` (seed an
  empty `.asc`), `schematic_from_netlist` (generate a wired `.asc` from
  netlist text for R/C/L/V/I/D circuits), `trace_net` (list every
  pin/label/wire on a net and flag shorts), and `reset_schematic` (revert
  an `.asc` to its snapshot from before the first edit).
- New circuit tools `validate_netlist` (static pre-flight checks of
  directive and element arity, `.MEAS`-vs-analysis mismatches, duplicate
  analyses, unparseable B-sources; on `.asc` also named-net shorts,
  floating pins, and dangling labels) and `diff_circuit` (structural diff
  of components, values, attributes, and directives between two files).
- ngspice is now fully supported. Raw files parse with the correct dialect,
  logs produce structured diagnostics, simulator stdout/stderr is captured
  and added to diagnostics, and pre-flight checks reject `.step` (pointing to
  `configure_sweep`) and warn about `.MEAS` directives ngspice skips in batch
  mode.
- An in-house Monte Carlo engine replaces `spicelib.Montecarlo`. It supports
  per-`.MODEL` process variation, Pelgrom-scaled per-instance MOSFET
  mismatch, and `.PARAM` rewriting alongside the R/C/L engine, with a `seed`
  for reproducible runs, relative or absolute tolerances, and per-run
  realised values stored so statistics can be correlated with measurements.
- Every run summary has an `observations` list of facts for the caller to
  evaluate, such as the simulator's own error lines, requested
  `.MEAS`/`.FOUR` results that were not produced, NaN/extreme node values,
  and skipped scans. It replaces a trust verdict.
- Reworked simulator detection. A configured but missing simulator is
  reported in `server_status` instead of silently returning another
  simulator's results. LTspice is auto-detected on WSL, `[simulator]` has an
  enabled-allowlist, and `.asc` editing works even with no simulator
  detected.
- Every mutating `.asc` operation reports floating pins, duplicate wires,
  and dangling labels; `connect` and `add_net_label` detect named-net shorts
  at edit time; `move_component` / `remove_component` report orphaned wires,
  and `remove_component` has a new `cleanup_wires` flag; unknown attribute
  names are rejected with a suggestion for a case typo.
- `configure_sweep` accepts an explicit `values` list (e.g. E-series)
  alongside start/stop ranges; `configure_montecarlo` validates
  per-component overrides against the netlist; batch jobs appear in
  `check_job` and job listings; `batch_results` reports per-run
  convergence-fallback markers (Gmin/source stepping) so a degenerate sweep
  is not reported as clean.
- `measurement_stats(job_id=...)` aggregates `.MEAS` results across every
  run of a sweep/Monte-Carlo batch, or across the steps of a stepped
  single run.
- `query_value` and `bode_metrics` accept `job_id` + `run_index` to
  analyze one run of a batch like a standalone raw file; `query_value` has
  new `step_axis`/`step_value` arguments (query a signal at a chosen
  `.step`/`.dc` sweep value) and `magnitude_linear`; `bode_metrics` has a
  new `all_steps` argument.
- `simulation_summary` auto-derives the log file from the raw file,
  accepts a `step` selector, and auto-picks an AC signal (with a warning)
  instead of silently dropping bandwidth metrics; `operating_point`
  accepts `step` for stepped `.op` results.
- `find_model(full=true)` attaches definition bodies; `list_libraries`
  detail mode lists `.MODEL` names (foundry decks with hundreds of models
  were previously not listed); LTspice's bundled `standard.*` component decks
  and current LTspice install paths are recognized.
- The server publishes MCP usage instructions (netlist-first workflow,
  matching analysis tools to run types, checking diagnostics rather than
  trusting a "completed" status).
- Unsupported `.MEAS` expressions (`vdb()`, `phase()`, `group_delay()`) are
  blocked at directive-write time with concrete suggestions, and
  `meas_errors` extracted from logs are passed through `run_simulation` and
  `simulation_summary`.
- `SECURITY.md`, this `CHANGELOG.md`, Dependabot configuration, a
  `pip-audit` CI step, and `twine check --strict` on built artefacts
  before publish.

### Changed

- Netlist reading and editing now run on an in-house structured SPICE
  lexer instead of regex passes. Format-preserving edits, hierarchical
  `.subckt` netlists, and quoted/braced/single-quoted expression tokens
  are handled correctly (one malformed expression form previously hung the
  server).
- Single-run and batch jobs are kept in one job store and read the same way.
  Any job id works with `check_job`, `cancel_job`, and
  `measurement_stats`, and a wrong-tool lookup redirects to the right tool
  instead of stating that the job does not exist.
- File decoding for netlists and libraries tries a fixed sequence (BOM
  detection, UTF-16-without-BOM heuristic, UTF-8, cp1252, lossy fallback), so
  Windows-edited files with stray symbols (°, µ) read correctly.
- `pulse_response` now returns results instead of refusing. Windows where
  overshoot/settling are undefined (a full pulse in the window, noisy
  baselines) return null metrics with machine-readable `quality` codes
  rather than an error or a plausible but wrong number; `edge_metrics`
  notes when a noisy window biased its auto-detected levels.
- `run_simulation`'s inline summary and the `simulation_summary` tool now
  share one pipeline and return the same fields.
- Errors that already include precise guidance no longer get a generic hint
  appended; `.asc` and symbol-resolution error messages are clearer.
- `pyproject.toml` declares Trove classifiers; CI and publish workflows
  hardened (read-only token, publish action pinned by commit SHA, job
  timeouts, concurrency groups); the publish workflow now runs the same
  gate as CI (type check, real-simulator end-to-end tests, dependency
  audit) so a release cannot pass weaker checks than an ordinary push.
- Internal API and typing changes. `SimulationRunner.kill` is public API
  (renamed from `_kill`), `Literal`/`TypedDict` are used across runners and
  parsers, and lint/type configuration is tuned for scientific notation in
  docstrings.
- `DESIGN.md` moved to `docs/` and rewritten as a user-facing reference;
  README restructured around the user workflow with a corrected tool
  catalog.

### Fixed

Numeric correctness (each of these returned a plausible but wrong number):

- `bode_metrics` point mode interpolated wrapped phase, so a query between
  samples on either side of the ±180° boundary could be off by up to ~180°;
  interpolation now runs on unwrapped phase.
- Overshoot/undershoot detection took the first local peak, so pre-edge
  ripple in the window reported 0 % overshoot on an edge that did overshoot;
  the largest peak is now used.
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
  a spurious sweep axis.
- The −3 dB half-power point uses the exact −3.0103 dB constant
  everywhere; frequency arguments accept Hz/kHz suffixes; SPICE values
  with unit annotations (`1uF`, `10MegHz`) parse instead of being skipped.
- `.step temp=-40°` log rows and `.step param NAME` sweeps whose raw files
  contain no axis now resolve correctly.

Simulation lifecycle:

- A `run_simulation` timeout above 30 s without `wait=true`, which returns a
  job id at once instead of waiting, was never enforced (the default timeout
  included), so a hung simulator ran forever; such jobs now start a watchdog
  that marks the job timed out and kills the process at the deadline.
- Cancelling or timing out a job on WSL did not kill the Windows-side
  LTspice process; single-run and batch cancel now terminate
  it, and a cancelled batch no longer launches its next queued run.
- The `max_parallel_sims` limit is now enforced for `run_simulation` jobs;
  jobs that time out while still queued terminate cleanly instead of
  launching an orphaned process.
- Failed or broken runs could be reported as successes. Simulator failure
  placeholders were reported as `completed`, log `Error:` lines were
  dropped, and ngspice convergence failures were classified as warnings. All
  of these are now reported as errors.
- `cancel_job` could not cancel sweep/Monte-Carlo jobs, and jobs
  interrupted by a server restart raised "unexpected status" instead of
  reporting partial results.
- Duration reporting no longer shows negative durations under clock skew,
  mismatches between the inline response and a later `check_job`, or
  fabricated durations on jobs recovered after a restart, and `active_jobs`
  no longer counts completed records.

Results and diagnostics:

- Fourier extraction returned empty harmonics for every `.four` analysis;
  zero-amplitude `.FOUR` signals no longer crash measurement parsing; an
  ignored `.fourier` directive is flagged.
- FAIL'ed `.MEAS` results are now reported in a `failed_measurements` list
  instead of silently disappearing from the output.
- ngspice's `Circuit: <title>` log echo was misparsed as a measurement
  named "circuit".
- `batch_results`: AC aggregation no longer collapses to peak-only, value
  filters match parameter names case-insensitively, single-sample runs
  render values instead of empty columns, and an empty batch raises an
  error instead of silently returning an empty page.
- Structured tool output now conforms to each tool's declared schema.
  `.asc` attribute values no longer include non-JSON-serializable objects,
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
- `connect` no longer creates shorts without a warning. Wires routed through
  another pin of an endpoint component are refused, and label shorts on
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
  floating pins, dangling labels) now run.
- `diff_circuit` compares component attributes and no longer reports the
  `.END` terminator as a removed directive; `schematic_from_netlist` keeps
  the first element of a title-less netlist instead of dropping it.
- `.NOISE` signal lookup is case-insensitive with alias resolution, and
  noise and DC sweeps are no longer misclassified as transient analyses.
- Cached editors and results refresh after an in-place file rewrite that
  happens within one mtime tick.
- numpy, scipy, pydantic, and anyio are now declared as runtime
  dependencies (installs previously worked only because other packages
  happened to pull them in); an unused declared dependency was removed.

### Security

- Dependency upgrades for published advisories: `cryptography` 46.0.6 →
  46.0.7 (CVE-2026-39892), `python-multipart` 0.0.24 → 0.0.27
  (CVE-2026-40347, CVE-2026-42561), `idna` 3.11 → 3.15 (CVE-2026-45409),
  and `pyjwt` 2.12.1 → 2.13.0.

## Earlier history

Release history before the first tagged version is in `git log`; the
`feat:` / `fix:` / `refactor:` prefixes and PR descriptions describe
each change.
