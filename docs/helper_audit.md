# Shared-helper audit

Reviewed from `4ee4db54`. The scope is duplicated rules in `src/ltspice_mcp/lib/`
and their callers, not a complete audit of every SPICE analysis or concurrency
interleaving. The pass searched all 62 library modules for competing analysis
predicates, trace buckets, path resolution, persistence status sets, file
stamps, atomic writes, cursor codecs, numeric parsing and geometry transforms.
An AST comparison also found identical function bodies; each candidate still
needed a caller and behavior review.

## Reproduced and fixed

| Rule | Observable defect | Correction and regression |
|---|---|---|
| Declared trace type precedes name fallback | A trace named `V(equivalent)` and explicitly typed `impedance` entered `voltages`. | Use name fallback only when the type is absent. `test_declared_types_reach_every_analysis_reader` exercises all four guarantees below through the analysis handler, with both voltage-shaped and bare trace names. |
| All operating-point buckets participate in filtering | `operating_point(device="M1")` retained unrelated `other` values. | Filtering uses `OP_BUCKETS`; the test verifies the selected current and removal of the unrelated trace. |
| All buckets participate in units | An impedance in `other` had no unit despite the declared type. | Unit reporting uses `OP_BUCKETS`; the test expects ohms. |
| Scalar readers preserve known units | The no-axis `value` path returned `unit: null` after computing a unit map. | Read the existing map; the test verifies both the number and its unit. |
| Header recognition agrees across consumers | Restart recovery marked a case failed when its raw was copied from the recorded LTspice transient fixture. Normal reads accept that BOM-less UTF-16 header. | Recovery and sniffing share the UTF-16 prefixes, with and without BOM. The existing restart-promotion test now includes the recorded binary fixture. This remains a header check, not full payload validation. |
| A raw's own writer beats an unrelated session default | A recorded ngspice file with `Command:` was parsed with the QSPICE dialect when QSPICE was the session default. | Preserve the sniffer's `None` so spicelib can detect the writer. `test_raw_writer_header_wins_over_session_default` checks the actual loaded reader. |
| Include resolution agrees with staging | On Linux, `models\analysis.inc` could be staged correctly but missed by the raw-requirement scanner. A clean log and missing transient raw then reported success. | The scanner calls `resolve_reference`; `test_missing_raw_detection_follows_windows_include_separators` checks the resulting failure classification. |

Each defect was observed before its correction. The bucketing cases use a
small constructed raw to separate metadata from naming; restart and dialect
cases use recorded simulator output. These are our reader and caller defects,
not demonstrated spicelib defects.

## Independent review corrections

The independent review found two regressions in the initial patch. Both were
reproduced through the analysis handler before correction:

- Copying the unit map into scalar results also exposed ngspice's generic
  voltage metadata on transfer-function gain and impedance. Derived no-axis
  results now omit generic voltage/current units; specifically typed units,
  such as impedance in ohms, remain available. The transfer-function test
  checks both metadata forms, the numeric values and the returned units.
- A sniffer result of `None` correctly delegates parsing to spicelib, but is
  not the detected writer. Operating-point processing now reads the loaded
  reader's dialect, restoring device-parameter guidance for imported ngspice
  files with a `Command:` header. The caller-supplied-raw regression covers
  files with and without that header, without producer hints.

The simplification pass builds the operating-point bucket dictionary once
and returns it directly. Its handler regression checks every guarantee for
each trace-name form instead of branching on which assertion to run.

## Candidates that do not justify a behavior change yet

- `experiment_store` repeats live and terminal status sets, and the runner
  repeats a terminal set. The live sets agree. The older shared terminal set
  also contains `timeout`, which the experiment status type does not admit.
  Consolidation should start by resolving that vocabulary difference, rather
  than silently widening what stored experiments accept.
- `runner_base` and `netlist_graph` have identical include-token parsers;
  `deck_prep` has a similar one with different malformed-quote behavior.
  The reproduced path-resolution disagreement is fixed. A single token parser
  is a follow-up requiring an explicit malformed-input contract.
- Request fingerprints and cursor checksums both serialize canonical JSON,
  but differ in Unicode escaping and non-finite-number handling. Their bytes
  are persisted contracts. Substituting one implementation for the other is
  not a behavior-preserving cleanup.
- The two lexer name extractors and two parameter-removal methods have
  identical bodies. They operate through distinct grammar/view entry points;
  this pass found no divergent output to fix.

Atomic replacement, file-cache stamps, cursor envelopes, micro-sign handling,
and schematic rotation already have shared implementations in the inspected
paths. Similar syntax alone is not evidence that two operations have the same
contract.

## Remaining limits and next work

Raw dialect inference without a writer header remains a heuristic. The
recorded files establish the covered cases; they do not prove that every
ASCII raw without `Command:` is ngspice. A bounded header parser and an
explicit caller-supplied dialect for ambiguous files deserve separate design
and binary-fixture coverage. The current sniffer also searches a fixed byte
prefix for `Command:` rather than parsing header lines.

Declared trace types relay metadata; they do not establish physical units for
every analysis. Some ngspice derived outputs carry voltage metadata for
quantities that are not voltages. The review correction suppresses those
generic units on derived no-axis scalar reads; it does not infer their true
units or prove that generic complex-value formatting is appropriate for poles.

For the cancellation-test failures, first force the suspected persistence
interleaving with events and establish the cause. The earlier successful
rerun establishes intermittency, not which component is at fault. Fix the
confirmed ordering and audit neighboring tests for the same checkpoint
assumption. A nonblocking loaded-CI job is secondary to that work.

The permanent practice is described in [TESTING.md](TESTING.md): review all
consumers of a shared rule and test a counterexample through the public path.
No recurring duplicate-code scanner or additional CI job was added.

## Validation

Each of the seven initial defect assertions and both review regressions
failed before its fix and passed after.
The full local Linux suite passed with 3,599 tests passed, 21 skipped, and
90.73% coverage. Ruff lint and format checks, Pyright, and `git diff --check`
passed. Native Windows validation was not run in this audit.
