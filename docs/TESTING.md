# Testing & quality practice

This is the durable record of *how* this project tests, and — more importantly
— of a class of bug its testing kept missing and the mechanisms added to stop
that. Read it before adding a tool, an op, or a stress pass.

## The failure this codifies

Two defects shipped despite a long run of adversarial stress passes against a
live server:

- a netlist→schematic converter that was unusable for active circuits (it
  silently dropped every transistor / controlled source, so an active circuit
  could not even start); and
- the schematic editor had no way to remove a wire or a net label — you could
  add both, but not undo either.

Neither is a wrong-output bug. Both are **absence-class** defects: a capability
that is missing, or unusable for an input class the tests never fed it. That is
why they survived. A stress pass *walks code paths*; absence has no path to
walk. You cannot exercise a tool that does not exist, and a tool that quietly
skips an input class looks correct on every input that isn't that class.

The deeper cause was the shape of the evaluation itself: it ran **tool-up and
happy-path-down** — "here are the tools, drive them through a plausible
workflow, did it work?" Three biases compounded inside that shape:

- **Passive-input bias.** The circuit batteries were almost entirely R/C/V.
  An active device was never pushed through the build or convert workflow, so
  the converter's active-device blindness never triggered.
- **Additive bias.** Every workflow built something up; none tore one down.
  Removal was never *needed* by a test, so its absence was never felt.
- **Self-grading.** The same agent that built a tool also chose the test
  circuit and judged the result — and chose inputs that showcased what it had
  built.

## What each instrument can and cannot catch

|instrument|catches|blind to|
|-|-|-|
|stress pass / path-walking|wrong behavior in code that runs|absence (missing or unusable-for-a-class capability)|
|unit / contract tests|wrong behavior in a unit|whether the surface as a whole covers the task|
|ground-truth numeric checks|silently-wrong numbers|missing capability|

The instruments below exist specifically to cover the blind column.

## The four mechanisms

### 1. Inverse-operation closure (mechanical)

`tests/test_dispatch.py::TestOpInverseClosure`. The schematic-editing op surface
must be **closed under inversion**: for every op that mutates the `.asc`, an
inverse op exists (or it is self-inverse). It is a *surface* guard — it asserts
an undo *capability* exists, not that state round-trips byte-for-byte (e.g.
`remove_component(cleanup_wires=true)` drops wires `add_component` won't restore;
`reset_schematic` is the recovery hatch for those). Each op in the `SchematicOp`
union is either paired with an inverse op that exists, or declared self-inverse
(re-applying it with the prior arguments reverts it). The pairing table is a
*forcing function*: a new `add_*` / `connect` / `create` op with no entry fails
the test, so shipping a one-way mutation becomes a reviewed decision instead of
an accident. This is the check that, run earlier, would have failed the day
`add_net_label` shipped without `remove_net_label` and `connect` without
`remove_wire`. Building the pairing table surfaced one missing inverse —
`add_directive` had no `remove_directive` op — which was added alongside the check.

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
run instead of after it ships. Keep the archetype set frozen and add to it when
a new device class becomes supported — do not let a battery drift back to
passive-only.

### 3. Task-down coverage pass (discipline)

Path-walking asks "does what exists work?" Coverage asks "does what exists let
me finish the job?" — and only the second can surface a missing capability.
Periodically, enumerate the **user tasks** (build a circuit, edit one, **fix a
mistake**, analyze a result, convert between forms) and for each walk the
*minimal tool sequence*, asking at every step "is this step possible?" The
removal gap was exactly an impossible step — "undo a misplaced label" — that no
happy-path walk ever attempted, because happy paths don't make mistakes. Run
this task-down, not tool-up: start from what a user needs to accomplish, not
from the tool list.

### 4. Blind-artifact judging (part discipline, part regression test)

The agent that builds an artifact is not the sole judge of its quality. This has
two halves, and only the second is automated — keep them distinct.

- **The discipline (not automated).** Feed the **artifact alone** — the `.asc`,
  the plot, the netlist, with no build narrative — to an independent reviewer
  (a person, or a separate model) against a rubric. This generalizes a lesson
  learned the hard way in the plot-evaluation work: leaving the title and
  filename on a plot leaked the expected answer into the vision evaluation and
  inflated its scores; stripping them corrected it. There is no automated blind
  grader in the suite; this is a review step you run by hand.
- **The code-backed piece.** `tests/test_circuit_asc.py::TestSchematicReadability`
  is a deterministic artifact-readback regression: it reads the built `.asc` back
  from disk and asserts the result is actually wired (real `WIRE` records, net
  labels scoped to the terminal nets), not net-label soup — judged on the
  artifact rather than on the sequence of calls that produced it. It is not a
  blind reviewer; it is a fixed assertion that encodes one rubric item.

## Accepted one-way mutations (surfaced, not gaps)

Closure under inversion is the rule for the schematic op surface. A few
tool-level mutations are deliberately *not* paired, and are recorded here so they
are not mistaken for the absence-class bug above:

- `create_netlist` / `create_schematic` create a file; deleting a file is a
  native filesystem operation, intentionally out of scope for a circuit editor.
  `reset_schematic` reverts in-session edits but does not remove a created file.
- `configure_sweep` / `configure_montecarlo` create a persisted config with no
  delete-config tool. Low value (a stale config is inert); accepted.

These are decisions, not oversights. If one stops being acceptable, it graduates
into mechanism 1 or 2.

## The practice that was already working

The half of the practice that was sound, kept, and assumed by everything above:

- **Real-path tests.** Tests drive actual code paths — `tests/test_e2e.py`
  launches the real server over stdio and speaks the client protocol; tests go
  through config and startup, not by monkey-patching internals.
- **Ground-truth-first numeric validation.** Numeric results are checked against
  closed-form expected values, not "it didn't crash."
- **Recorded-real fixtures.** Real simulator `.raw` / `.log` output is captured
  under `tests/fixtures/` so dialect and parse seams run against true output
  offline (see `tests/conftest.py`).
- **Tiered live gating.** `tests/test_ngspice_e2e.py` runs whenever `ngspice` is
  on PATH (so it runs in CI); `tests/test_e2e.py` runs un-gated in degraded
  mode; `tests/test_ltspice_integration.py` is opt-in via an environment flag.
- **Drift guards.** `tests/test_doc_drift.py` keeps documented tool counts and
  names honest against the registry; `tests/test_guide_delivery.py` keeps the
  packaged guide in sync with the skill.

See `CLAUDE.md` for the canonical `pytest` / `ruff` / `pyright` commands and
`docs/DESIGN.md` for the architecture and the end-to-end verification recipe.

## Conventions

- **Behavior-named test files.** Tests are named for the behavior they cover,
  never for a stress pass, date, or version. A regression found in a stress pass
  lands in the existing behavior module it belongs to; its origin goes in a
  docstring or comment, not the filename.
- **Plain-language findings.** Shipped code, docstrings, commit messages, and
  this doc describe a bug by its actual behavior in plain technical terms — no
  internal severity codes, codenames, or pass numbers (those stay in the
  internal backlog under `.claude/plans/`).
- **Tables** use minimal separators (`|-|-|`); no box-drawing characters.
