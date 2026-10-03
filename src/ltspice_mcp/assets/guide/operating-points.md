---
name: operating-points
description: >
  gm, gds, vth, vdsat or a gm/ID table from a MOSFET, BJT or diode.
---

# Reading device operating points (gm/gds/vth, gm/ID characterization)

The small-signal / model parameters of a MOSFET/BJT/diode (gm, gds, gmbs, vth,
vdsat, gm/ID, the capacitances) are returned by this server as **named
numbers**; no rawfile parsing or `.control`/`wrdata` block is needed. Both
simulators expose them, in different files; pick the method below by what you
need.

## One device at a single bias → the `operating_point` recipe (works on LTspice)

```spice
M1 d g 0 0 nch L=0.18u W=2u
Vd d 0 1.8
Vg g 0 0.9
.op
.lib /path/to/models.lib
.end
```
Run it with `run_experiments`, then read
`{"metric": "operating_point", "device": "M1"}`: gm, gds, vth, vdsat,
the caps and terminal currents at that bias. On **LTspice** these come from the
log's *Semiconductor Device Operating Points* block — the server adds
`.options logopinfo` automatically for `.op` runs (LTspice writes the block only
under that option, and only for `.op`). On **ngspice**, `.save @m1[gm] @m1[gds]
@m1[vth] @m1[id]` puts them in the raw; the recipe reads either uniformly.

## gm/ID curve vs a swept bias (the sizing table) → ngspice `.dc` + a `waveform` recipe

```spice
.dc Vg 0 1.8 0.01      ; sweep the gate
.save @m1[gm] @m1[gds] @m1[vth] @m1[id]
```
A `waveform` recipe — `{"metric": "waveform", "signals": ["m1.gm", "m1.gds",
"m1.id"], "format": "csv"}` — is the gm/ID table; one value at a chosen bias →
`{"metric": "value", "expr": "m1.gm", "at": …}`. This swept form
needs **ngspice** (LTspice's `logopinfo` is `.op`-only; for a swept gm on LTspice
you'd differentiate the drain current, `d(Id(M1))`, instead). Don't use
`run_experiments` `variations` for this: a native `.dc Vds Vgs` is one deck, not
N separate runs.

Address an operating-point param by the `m1.gm` shorthand or its literal `@m1[gm]` name; the
recipes resolve the bare / `v()` / `i()` wrapping, and a subcircuit path like
`x1.m1.gm`, for you. Values carry SI units where the simulator declares the type.
