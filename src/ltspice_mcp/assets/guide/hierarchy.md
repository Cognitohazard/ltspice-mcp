---
name: hierarchy
description: >
  Finding, measuring or varying one device inside nested subcircuits.
---

# Nested instances

## Discovering nested instances

This lists the devices inside a repeated block:

```json
{"queries": [{"kind": "hierarchy", "path": "amp.cir", "simulator": "ltspice",
              "instance": ["XA", "Xleaf"], "prefix": "M"}]}
```

`instance` is a list of exact reference segments, matched without regard to
case; keep it as the device's identity. A row's source file, line and section
identify the declaration, and its instance list identifies one runtime device
(`components` stays a flat read of the source). Hierarchy reads netlists only,
so export a schematic first. It is read-only: to change a device, pass its
instance list to a variation (below).

Each row carries the scoped nodes and port mappings, the model source, raw and
resolved values and parameters, MOS W/L in metres, and the backend addresses.

A number discovery cannot evaluate is reported, not guessed: it has
`status: "unresolved"`, no value, and a `reason`. Discovery evaluates static
arithmetic, defaults and caller overrides, with each simulator's parameter
precedence. Functions, missing parameters, cycles, simulator steps, ngspice
sibling-dependent X-call overrides and expressions with several powers stay
unresolved. LTspice caret expressions are not supported, since `^` is not
exponentiation there. What discovery cannot expand is refused with a reason:
conditional structure, opaque control scripts, local definitions, recursion,
missing dependencies, ambiguous declarations, resource limits, and an explicit
LTspice `scale` (use SI dimensions; the `mil` suffix is 25.4 µm).

Declare the simulator even when only inspecting. For ngspice, `ngbehavior`
defaults to the configured mode, and sectioned libraries need a mode without
the `lt`/`ps` reinterpretation, such as `hsa`; a mode set here does not change
later runs. The supported ngspice modes are `hsa`, `kiltpsa` and `""` (native
SPICE); others are refused. A cursor is bound to the files, mode and filters it
was issued for, so after an edit, start a new query.

## Measuring a nested device

Put the row's `address.save` line in the deck, run it with `run_experiments`,
and pass `address.device` to the `operating_point` recipe:
`{"key": "chosen", "metric": "operating_point", "device": "..."}`. For a MOS at
`["XA", "Xleaf", "M0"]`:

| | ngspice | LTspice |
|-|-|-|
| `device` | `m.xa.xleaf.m0` | `xa:xleaf:m0` |
| deck line | `.save @m.xa.xleaf.m0[gm]` | `.options logopinfo` (gm comes from the log) |
| a nested resistor's current | `.save @r.xa.xleaf.r1[i]` | `.save I(xa:xleaf:r1)` |

Keep the full name: it is what tells repeated peers apart. When a row has no
address, it gives the reason; do not shorten or guess a selector. A node
voltage's spelling does not mean the run saved that trace.

## Varying one nested instance

Use the exact reference-segment list from discovery:

```json
{"kind": "assign", "instances": [
  {"instance": ["XA", "Xdev"], "attribute": "parameter",
   "parameter": "w", "values": [1, 2]},
  {"instance": ["XA", "R1"], "attribute": "value", "values": ["100k"]}
]}
```

This changes only the selected runtime instances, in private case files; peers
and your own files stay unchanged. Values use the deck's units: on a Sky130
deck, whose scale is `1e-6`, the widths `1` and `2` above are micrometres,
where a typical LTspice MOS width would be `"2u"`. Use `combine: "zip"` for
paired lists of equal length. Conflicting fields, and edits to both an ancestor
and its descendant, are refused. For your own mismatch coefficients, a
`random` rule can name the exact physical MOS with `instance` instead of a
`prefix` (guide section 'variations').
