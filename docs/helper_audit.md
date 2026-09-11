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

The first native Windows 3.12 run failed
`test_foreign_token_sets_durable_submission_barrier`: no cancellation marker
was present after the call. Its setup changed the live job's owner PID after
observing an in-memory running status, while coordinator writes could still
overwrite the foreign record. The test now drains those writes, loads a
separate record and changes that copy's owner. It also checks the cancellation
response for an error before inspecting the marker. No production cancellation
code changed. This corrects the observed test's setup; it does not establish
the cause of every earlier intermittent cancellation failure. A deterministic
interleaving audit of neighboring tests remains useful before adding a loaded
CI job.

The permanent practice is described in [TESTING.md](TESTING.md): review all
consumers of a shared rule and test a counterexample through the public path.
No recurring duplicate-code scanner or additional CI job was added.

## Validation

Each of the seven initial defect assertions and both review regressions
failed before its fix and passed after.
The full local Linux suite passed with 3,599 tests passed, 21 skipped, and
90.73% coverage. Ruff lint and format checks, Pyright, and `git diff --check`
passed.

Native Windows validation used a fresh checkout of `953780e2` with
`core.autocrlf=true`, plus the cancellation-test correction described above.
The full suite passed on Python 3.12.12 and 3.13.9: 3,540 passed and 80 skipped
on each interpreter. The first 3.12 run, before that correction, had one
failure, 3,539 passes and 80 skips.
The fixture-path correction found during skip review then passed as a
targeted test on Linux and both Windows interpreters. It removes one of those
80 skips; the full suites were not repeated for that single-test correction.

### Initial Windows skip inventory

| Count | Reason | Follow-up |
|---:|---|---|
| 27 | ngspice not on PATH, including detached-owner tests | Exercise these with native Windows ngspice installed. |
| 18 | LTspice integration not enabled (15), or real symbols not found (3) | Enable local LTspice integration and configure symbol paths; one integration case is WSL-only. The symbol-test locator currently checks the environment and WSL, so its skip does not establish that Windows lacks a symbol installation. |
| 6 | PNG rendering dependencies unavailable | Include an environment with the raster extra and native Cairo. |
| 5 | Symlink creation unavailable | Exercise path containment on a Windows test account with symlink privilege. |
| 17 | POSIX-specific assertions | Keep OS-specific assertions on their platform, but add Windows worker timeout, cancellation, parent-death and child-cleanup coverage for the six worker-related cases in this group. |
| 2 | Optional Sky130 PDK unavailable | Retain an optional PDK integration tier. |
| 3 | A parametrized module does not build an outcome | The outcome-construction rule does not apply. |
| 1 | Windows Claude scratch directory is not known | Existing Windows support gap. |
| 1 | Normal-parser deadline test looked in a nonexistent fixture subdirectory | Corrected to use the required recorded fixture directly; a missing required fixture must fail, not skip. |

### Expanded native Windows validation

The follow-up enabled installed LTspice and its real symbols, a portable
ngspice console build, and the raster extra with native Cairo. Dependencies
were confined to the validation environment; no system installation or
symlink privilege change was made.

This exposed production defects that the skipped tests had hidden:

- Windows worker timeouts returned `WorkerDied` instead of `timeout`, and
  killing a worker left its subprocesses alive. The supervisor now holds a
  Windows Job Object so worker reset, exit and supervisor death terminate
  ordinary descendants. Timeout reporting follows the deadline and measures
  actual elapsed time. A further PID regression found that Windows virtual
  environments inserted a launcher process with a restrictive child job.
  Worker and detached-owner launches now use the same base-interpreter plus
  environment-marker approach as CPython multiprocessing, preserving the
  virtual environment while supervising the executing process directly.
- A timed-out detached owner left its child processes running. Windows cleanup
  now uses a PID-scoped tree kill; each real owner also holds its own Job
  Object. An owner launched from `run_code` explicitly breaks away from the
  worker's job. The regression pauses a real ngspice process, resets the
  worker, resumes the simulator and requires the detached job to complete.
- Native Cairo crashed when text was rendered from successive short-lived
  threads. A two-thread regression reproduced the access violation and passed
  after Windows rasterization moved to one persistent renderer thread.

The live LTspice tests also had stale measurement/export response assertions.
They now check current response fields, physical RC values and exported file
contents. Three configuration tests now clear the symbol-path environment
variable before asserting TOML-only behavior.

Independent review strengthened descendant-cleanup assertions and replaced a
helper-only breakaway test with the actual detached-experiment API. Removing
the corresponding ownership fixes made those tests fail. Deterministic
injection at the taskkill boundary also exposed cleanup exceptions escaping
the API contract; owner-exit races are tolerated, and unresolved cleanup
failures are included in the structured handshake error.

The expanded 3.13 run also reported a completed concurrent detached job as
`interrupted`. Its owner logs showed normal completion. A deterministic
registry regression reproduced an ordering that explains this: read a running
record, let the owner persist completion and exit during the liveness probe,
then reconcile the stale snapshot and persist it over the final record. The
loader now rereads and validates the record after confirming owner death. If
the recorded owner changed, it does not apply the earlier owner's liveness
answer. The regression failed before this change and passed after.

The Linux run also exposed a flaw in the cancellation test's simulated
foreign owner: changing one saved record's PID did not change later writes
from the real coordinator. A subsequent read could therefore see a local
owner in a session with no coordinator and return `cancel_unavailable`. The
test now changes the owner only at the persistence boundary, consistently
across all checkpoints, and drains the terminal write before checking the
foreign receipt. It preserves the live coordinator's actual ownership; no
production cancellation code changed.

The remaining coverage limits are privilege-dependent symlink tests, the
optional Sky130 integration tier, and the unknown Windows Claude scratch
location. OS-specific assertions and non-applicable outcome-rule cases still
skip deliberately. This does not establish complete Windows feature coverage.


The expanded Windows Python 3.12 suite passed: **3,602 passed, 27 skipped**.
After the reconciliation fix and final test cleanup, its full experiment-store
and detached-owner modules passed: **68 passed, 1 skipped**. Windows Python
3.13 passed the full suite with that fix: **3,603 passed, 27 skipped**.
The final cancellation-test setup correction then passed separately on both
Windows interpreters.
The final Linux suite passed: **3,605 passed, 25 skipped**, with **90.49%**
coverage. Ruff lint/format, Pyright and `git diff --check` passed. The native
checkout's source and test files matched the final working tree byte for byte.

| Remaining Windows skips | Count |
|---|---:|
| POSIX or WSL-specific assertions | 16 |
| Symlink privilege unavailable | 5 |
| Optional Sky130 PDK unavailable | 2 |
| Outcome rule does not apply to the parametrized module | 3 |
| Windows Claude scratch directory is not known | 1 |
