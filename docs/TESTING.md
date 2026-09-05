# Testing & quality practice

This document records how the project tests and, more importantly, a class
of bug the tests kept missing and the mechanisms added to catch it. Read it
before adding a tool, an op, or a test campaign.

## The bugs this addresses

Two defects shipped despite a long run of adversarial stress passes against a
live server:

- a netlist→schematic converter that was unusable for active circuits (it
  silently dropped every transistor / controlled source, so an active circuit
  could not even start); and
- the schematic editor had no way to remove a wire or a net label — you could
  add both, but not undo either.

Neither is a wrong-output bug. Both are **absence-class** defects: a capability
that is missing, or unusable for an input class the tests never fed it. They
survived because a stress pass exercises code that exists, and a missing
capability has no code to exercise. You cannot exercise a tool that does not
exist, and a tool that quietly skips an input class looks correct on every
input outside that class.

The deeper cause was how the evaluation was run: it started from the tool
list and drove the tools through a plausible workflow ("here are the tools,
does a reasonable sequence work?"). Three biases followed from that:

- **Passive-input bias.** The circuit batteries were almost entirely R/C/V.
  An active device was never pushed through the build or convert workflow, so
  the converter's active-device blindness never triggered.
- **Additive bias.** Every workflow built something up; none took anything
  apart. No test needed removal, so nobody noticed it was missing.
- **Self-grading.** The same agent that built a tool also chose the test
  circuit and judged the result — and chose inputs that showcased what it had
  built.

## What each instrument can and cannot catch

|test method|detects|does not detect|
|-|-|-|
|stress pass / workflow tests|wrong behavior in code that runs|a missing capability, or one unusable for an input class the tests never feed it|
|unit / contract tests|wrong behavior in a unit|whether the tool set as a whole can complete a user task|
|ground-truth numeric checks|silently-wrong numbers|missing capability|

The mechanisms below exist to cover the third column.

## The four mechanisms

### 1. Inverse-operation closure (mechanical)

`tests/test_dispatch.py::TestOpInverseClosure`. The schematic-editing op surface
must be **closed under inversion**: for every op that mutates the `.asc`, an
inverse op exists (or it is self-inverse). It checks that an undo capability
exists, not that state round-trips byte-for-byte (e.g.
`remove_component(cleanup_wires=true)` drops wires `add_component` won't restore). Each op in the `SchematicOp` union is either
paired with an inverse op that exists, or declared self-inverse (re-applying it
with the prior arguments reverts it). The pairing table forces the decision: a
new `add_*` / `wire_pins` / `create` op with no entry fails the test, so a
one-way mutation ships only as a reviewed decision, not by accident. Had this
check existed earlier, it would have caught `add_net_label` shipping without
`remove_net_label` and `wire_pins` without `remove_wire`. Building the table
also turned up one missing inverse (`add_directive` had no `remove_directive`
op), which was added with the check.

The standalone mutating tools live outside the op union, so a companion guard
(`tests/test_dispatch.py::TestMutatingToolsAreReversible`) requires every
non-read-only registered tool to declare a reversal path, or to be an explicitly
accepted one-way mutation (see *Accepted one-way mutations* below). A new
standalone mutate tool added with no entry fails the test the same way.

### 2. Archetype build battery (input distribution)

`tests/test_circuit_asc.py::TestArchetypeBuildCoverage` and the active-device
end-to-end test in `tests/test_ngspice_e2e.py`. Every workflow that claims to
handle "a circuit" must be exercised against each canonical device class, not
just passives:

- passive (R / C / V)
- two-terminal active (diode)
- three-terminal active (MOSFET / BJT)
- four-terminal controlled source (VCVS / VCCS)

The build battery places and wires each class through the real build path; the
ngspice test simulates an active circuit (a saturated NPN switch) end to end and
asserts physical ground truth (the collector is pulled to saturation, proving
the device conducts). An unusable-for-a-class regression now fails on the next
run instead of after it ships. Keep the archetype set; add to it when a new
device class becomes supported, and do not remove active-device coverage.

### 3. Task-down coverage pass (discipline)

Path-walking asks "does what exists work?" Coverage asks "does what exists let
me finish the job?" Only the second can find a missing capability.
Periodically, list the **user tasks** (build a circuit, edit one, **fix a
mistake**, analyze a result, convert between forms) and for each walk the
*minimal tool sequence*, asking at every step whether the step is possible.
The removal gap was an impossible step ("undo a misplaced label") that no
happy-path walk attempted, because a happy path never needs to undo anything.
Start from what a user needs to accomplish, not from the tool list.

### 4. Blind-artifact judging (part discipline, part regression test)

The agent that builds an artifact is not the sole judge of its quality. This has
two halves, and only the second is automated — keep them distinct.

- **The discipline (not automated).** Feed the **artifact alone** — the `.asc`,
  the plot, the netlist, with no build narrative — to an independent reviewer
  (a person, or a separate model) against a rubric. This came out of the
  plot-evaluation work: leaving the title and filename on a plot leaked the
  expected answer into the vision evaluation and inflated its scores;
  stripping them corrected it. There is no automated blind
  grader in the suite; this is a review step you run by hand.
- **The code-backed piece.** `tests/test_circuit_asc.py::TestSchematicReadability`
  is a deterministic artifact-readback regression: it reads the built `.asc` back
  from disk and asserts the result is actually wired (real `WIRE` records, net
  labels only on the terminal nets) rather than connected only through net
  labels, judged on the artifact rather than on the sequence of calls that
  produced it. It is not a
  blind reviewer; it is a fixed assertion that encodes one rubric item.

## Accepted one-way mutations

Closure under inversion is the rule for the schematic op surface. A few
tool-level mutations are deliberately *not* paired, and are recorded here so they
are not mistaken for the absence-class bug above:

- File creation (`edit_schematic` with `base: "blank"`; formerly the
  `create_netlist` / `create_schematic` tools) has no delete pair; removing a
  file is a native filesystem operation, intentionally out of scope for a
  circuit editor.
- The pre-0.6.0 `configure_sweep` / `configure_montecarlo` tools created a
  persisted config with no delete-config tool. A delete tool had low value (a
  stale config is inert). The question is moot now: sweeps are
  `run_experiments` variations.

These are decisions, not oversights. If one stops being acceptable, add it to
mechanism 1 or 2.

## The practice that was already working

The following practices were already sound. They are kept, and everything
above assumes them:

- **Real-path tests.** Tests drive actual code paths — `tests/test_e2e.py`
  launches the real server over stdio and speaks the client protocol, and the
  rest go through config and startup rather than constructing state by hand.
  Substitution is confined to a few named seams, and the suite does use
  `monkeypatch` for them:
  - **The simulator subprocess boundary.** `fake_simulator` and
    `recorded_fixture_simulator` (`tests/conftest.py`) replace
    `ExperimentRunner.submit_netlist` — the one call that spawns a simulator.
    The first hands back a minimal artifact pair on a controllable delay (that
    delay is what catches a caller who printed a receipt for a job still in
    flight); the second copies a recorded real-LTspice `.raw`/`.log` pair in,
    so an analysis stage parses genuine simulator output.
  - **Platform and environment seams.** WSL detection and path conversion
    (`lib/wsl.py`), simulator detection at bootstrap, desktop browser launch
    (`lib/desktop.py`), and the optional cairosvg raster backend — so one
    machine can exercise every platform branch.
  - **Timeouts, lowered.** Parse deadlines and the shutdown cancel timeout are
    dropped to fractions of a second, so a bound can be shown to bite inside
    the suite instead of being asserted about.

  What is *not* substituted: handlers, the response path, the SPICE lexer and
  validator, the `.raw`/`.log` parsers, symbol and schematic geometry, and the
  job registry and store. A test for any of those runs the real thing.
- **Ground-truth-first numeric validation.** Numeric results are checked against
  closed-form expected values, not "it didn't crash."
- **Recorded-real fixtures.** Real simulator `.raw` / `.log` output is captured
  under `tests/fixtures/` so dialect and parse seams run against true output
  offline (see `tests/conftest.py`).
- **Tiered live tests.** `tests/test_ngspice_e2e.py` runs whenever `ngspice` is
  on PATH (so it runs in CI); `tests/test_e2e.py` runs un-gated in degraded
  mode; `tests/test_ltspice_integration.py` is opt-in via an environment flag.
- **Drift guards.** `tests/test_doc_drift.py` checks documented tool counts and
  names against the registry; `tests/test_guide_delivery.py` keeps the
  packaged guide in sync with the skill.

See `CLAUDE.md` for the canonical `pytest` / `ruff` / `pyright` commands and
`docs/DESIGN.md` for the architecture and the end-to-end verification recipe.

## Conventions

- **Behavior-named test files.** Tests are named for the behavior they cover,
  never for a test campaign, date, or version. A regression found in a test
  campaign goes in the existing behavior module it belongs to; its origin goes
  in a docstring or comment, not the filename.
- **Plain-language findings.** Shipped code, docstrings, commit messages, and
  this doc describe a bug by its actual behavior in plain technical terms — no
  internal severity codes, codenames, or pass numbers (those stay in the
  internal backlog).
- **Tables** use minimal separators (`|-|-|`); no box-drawing characters.
