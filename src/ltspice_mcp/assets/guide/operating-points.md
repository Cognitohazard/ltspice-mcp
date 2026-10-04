---
name: operating-points
description: >
  gm, gds, vth, vdsat or a gm/ID table from a MOSFET, BJT or diode.
---

# Device operating points (gm, gds, vth, gm/ID)

The small-signal and model parameters of a MOSFET, BJT or diode (gm, gds,
gmbs, vth, vdsat, gm/ID, the capacitances) come back as named numbers. Both
simulators expose them, in different files; pick the method by what you need.

## One device at one bias (LTspice or ngspice)

```spice
M1 d g 0 0 nch L=0.18u W=2u
Vd d 0 1.8
Vg g 0 0.9
.op
.lib /path/to/models.lib
.end
```

Run it with `run_experiments`, then read
`{"metric": "operating_point", "device": "M1"}`: gm, gds, vth, vdsat, the caps
and the terminal currents at that bias. On LTspice these come from the log's
*Semiconductor Device Operating Points* block, which LTspice writes only for
`.op` and only under `.options logopinfo`; the server adds that option to
every `.op` run. On ngspice, `.save @m1[gm] @m1[gds] @m1[vth] @m1[id]` puts
them in the raw. The recipe reads either the same way.

## gm/ID against a swept bias (ngspice)

```spice
.dc Vg 0 1.8 0.01      ; sweep the gate
.save @m1[gm] @m1[gds] @m1[vth] @m1[id]
```

This `waveform` recipe is the gm/ID table:

```json
{"metric": "waveform", "signals": ["m1.gm", "m1.gds", "m1.id"], "format": "csv"}
```

One value at a chosen bias is
`{"metric": "value", "expr": "m1.gm", "at": …}`. The swept form needs ngspice,
because LTspice's `logopinfo` is `.op`-only; on LTspice, differentiate the
drain current (`d(Id(M1))`) instead. A `.dc` sweep is one deck and one run,
where sweeping the bias with `run_experiments` variations would make one run
per point.

Address an operating-point parameter by the `m1.gm` shorthand or its literal
`@m1[gm]` name; the recipes resolve the bare, `v()` and `i()` wrappings and a
subcircuit path such as `x1.m1.gm`. Values carry SI units where the simulator
declares the type.
