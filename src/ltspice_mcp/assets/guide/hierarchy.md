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

`instance` is a list of reference names, one per level from the top of the
deck down, each matched exactly but without regard to case. In the query it
selects that instance and every device inside it. A row's `source` file, line
and section identify the line that declares the device. The row's own
`instance` list identifies one device in the expanded circuit, so each use of
a subcircuit gives its own rows; use that list wherever a later call names the
device. The `components` kind, by contrast, lists the source lines without
expanding subcircuits. The `hierarchy` kind reads only netlists, so export a
schematic first. It is read-only; to change a device, pass its instance list
to a variation (below).

Each row contains the device's `nodes`, each with the subcircuit it belongs to
(`scope`) and the simulator's name for its voltage (`voltage_trace`); for an
`X` row, the `ports` of the subcircuit and the nodes they connect to; the
`model` name and the file and line that define it; the `value`, the
`parameters` and the parameters in effect for the device (`environment`; for
an `X` row, inside its subcircuit), each as written (`expression`) and
evaluated (`value`); MOS W and L in metres under `geometry`; and the `address`
that the simulator uses for the device (below).

When discovery cannot evaluate a number, it reports `status: "unresolved"`, no
value and a `reason` instead of a guess. Discovery evaluates numbers with SI
suffixes, `+ - * /`, one power operator and parameter names. It reads a
subcircuit's defaults from its `.subckt` line and the values that the `X` line
passes, which override them. When a `.subckt` default and a `.param` inside
the subcircuit set the same name, ngspice uses the `.param` and LTspice uses
the default. Function calls, unknown parameters, parameters that depend on
themselves and expressions with more than one power operator (`**` or `^`)
stay unresolved. So does a parameter on an ngspice subcircuit (`X`) line that
refers to another parameter on the same line. An LTspice expression that
contains `^` stays unresolved, since `^` is not exponentiation there. In a deck
with a `.step` line every value stays unresolved, because `.step` gives it no
single value.

Discovery refuses the whole query with an error naming the cause for
`.if`/`.elseif`/`.else`/`.endif`, a `.control` block, `.alter`, `.altermod`,
`.del` or `.delete`, a `.subckt` defined inside another, a subcircuit that
instantiates itself, a missing include file, library section or subcircuit
definition, a name declared twice (a subcircuit, model, reference, port, or a
`.param` outside ngspice `hsa` mode), more than 16 MiB of input, 100,000 cards,
20,000 instances or 32 levels of nesting, and an explicit `.option scale` in an
LTspice deck (use SI dimensions; the `mil` suffix is 25.4 µm).

Declare the simulator even when only inspecting. For ngspice, `ngbehavior`
defaults to the configured mode. A deck with a sectioned
`.lib <file> <section>` needs a mode that contains neither `lt` nor `ps`, such
as `hsa` (guide section 'ngspice' explains why); a mode set in this query does
not change later runs. The supported ngspice modes are `hsa`, `kiltpsa` and
`""` (native SPICE); others are refused. A cursor works only with the same
files, unchanged, and the same mode and filters as the query that returned it;
otherwise the query fails with `invalid_cursor`. After an edit, start a new
query.

## Measuring a nested device

Put the row's `address.save` line in the deck, if it has one, run it with
`run_experiments`, and pass `address.device` to the `operating_point` recipe:
`{"key": "chosen", "metric": "operating_point", "device": "..."}`. For a MOS at
`["XA", "Xleaf", "M0"]`:

| | ngspice | LTspice 26 | LTspice XVII |
|-|-|-|-|
| `device` | `m.xa.xleaf.m0` | `xa:xleaf:m0` | `xa:xleaf:m0` |
| deck line | `.save @m.xa.xleaf.m0[gm]` | none (gm comes from the log) | none (gm comes from the log) |
| a nested resistor's current | `.save @r.xa.xleaf.r1[i]` | `.save I(xa:xleaf:r1)` | `.save I(a:leaf:r1)` |

LTspice needs no line for a MOS, because the server asks LTspice 26 for the
log's device table on every `.op` run. XVII prints the table for every `.op`
without being asked, and it stops with an `unrecognized option` error on a deck
that contains `.options logopinfo`. Each build's `.save` accepts a name inside a
subcircuit only in the spelling its raw file uses, which on XVII drops the X
from each instance. The other spelling stores nothing on LTspice 26 and makes
XVII save everything but the current. `simulator: "ltspice"` gives the
spelling for the session's LTspice, and `"ltspice:<name>"` the spelling for a
named build; `profile.xvii_names` reports which spelling applies.

Keep the full name, because it distinguishes other instances of the same
subcircuit. When a row has no address, `address.reason` gives the reason; do
not shorten or guess a selector. A node's `voltage_trace` is the name the
simulator would give that voltage, whether or not the run saved the trace.

## Varying one nested instance

Use the exact reference-segment list from discovery:

```json
{"kind": "assign", "instances": [
  {"instance": ["XA", "Xdev"], "attribute": "parameter",
   "parameter": "w", "values": [1, 2]},
  {"instance": ["XA", "R1"], "attribute": "value", "values": ["100k"]}
]}
```

This changes only the selected devices. For each case, the server writes
renamed copies of the subcircuits on the device's path into its own copy of
the deck, so other instances of the same subcircuit and your own files stay
unchanged. Values use the deck's units. On a Sky130
deck, whose scale is `1e-6`, the widths `1` and `2` above are micrometres,
where a typical LTspice MOS width would be `"2u"`. Use `combine: "zip"` for
paired lists of equal length. Conflicting fields, and edits to both an ancestor
and its descendant, are refused. For your own mismatch coefficients, a
`random` rule can name the exact physical MOS with `instance` instead of a
`prefix` (guide section 'variations').
