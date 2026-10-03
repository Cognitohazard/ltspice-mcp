---
name: hierarchy
description: >
  Finding, measuring or varying one device inside nested subcircuits.
---

# Nested instances

## Discovering nested instances

Use `inspect(queries=[{"kind":"hierarchy", "path":"amp.cir",
"simulator":"ltspice", "instance":["XA","Xleaf"], "prefix":"M"}])`
to find devices inside a repeated block. `instance` contains exact reference
segments, matched without case sensitivity; keep that list for identity.
The source file/line/section identifies the declaration, while the instance
list identifies one runtime device. `components` remains a flat source read.
Hierarchy reads netlists only; export a schematic explicitly first.

Rows carry structural scoped nodes and port mappings, model source, raw/resolved
values and parameters, MOS W/L in metres, and backend addresses. A numeric fact
with `status: "unresolved"` has no effective value: read its `reason`.
Discovery supports static arithmetic, defaults and caller overrides, with
backend-specific parameter precedence. Ngspice sibling-dependent X-call overrides and
expressions with multiple powers remain unresolved. LTspice caret expressions
are unsupported; caret is not interpreted as exponentiation. Explicit LTspice `scale`
is refused; use SI dimensions. The `mil` suffix is 25.4 micrometres.
Functions, missing parameters, cycles
and simulator steps can leave numeric facts unknown. Conditional structure,
opaque control scripts, local definitions, recursion, missing dependencies,
ambiguous declarations and resource-limit overruns are refused. Discovery is
read-only; pass its instance list to an experiment variation to edit that instance.

Declare the simulator even for offline inspection. For ngspice, `ngbehavior`
defaults to the configured effective mode; sectioned libraries need a mode
without `lt`/`ps` reinterpretation, such as `hsa`. An explicit inspection mode
does not reconfigure subsequent runs. Cursor tokens bind captured dependency
content, the declared profile and filters; after an edit, start a new query.
The initial supported ngspice profiles are `hsa`, `kiltpsa`, and the empty
string (native SPICE mode); other profiles are refused explicitly.

To measure a selected device, put its `address.save` guidance in the deck,
run it through `run_experiments`, and pass `address.device` to the existing
`analyze_results` recipe `{"key":"chosen", "metric":"operating_point",
"device":"..."}`. For example, a MOS at `["XA","Xleaf","M0"]` uses
`m.xa.xleaf.m0` and `.save @m.xa.xleaf.m0[gm]` on ngspice. LTspice uses
`xa:xleaf:m0` and `.options logopinfo`: gm is read from the log, not a guessed
raw signal. A nested resistor uses `.save @r.xa.xleaf.r1[i]` on ngspice or
`.save I(xa:xleaf:r1)` on LTspice. Full ancestral names distinguish the repeated
peer. If an address is unavailable, use the explicit reason; do not shorten or
guess a selector. Node voltage spellings do not guarantee a run saved the trace.


## Varying one nested instance

Use the exact reference-segment list from hierarchy discovery:

```json
{"kind": "assign", "instances": [
  {"instance": ["XA", "Xdev"], "attribute": "parameter",
   "parameter": "w", "values": [1, 2]},
  {"instance": ["XA", "R1"], "attribute": "value", "values": ["100k"]}
]}
```

This changes only the selected runtime instances in private case files. Peers
and original authoring files stay unchanged. Values use the deck's units:
Sky130 widths above are micrometres because its scale is `1e-6`; a typical
LTspice MOS width would instead use a value such as `"2u"`. Use `combine: "zip"`
for paired target lists of equal length. Conflicting fields and overlapping
ancestor/descendant edits are refused. For caller-defined mismatch, a random
rule may instead name the exact physical MOS with `instance`; do not also give
that rule a `prefix`.
