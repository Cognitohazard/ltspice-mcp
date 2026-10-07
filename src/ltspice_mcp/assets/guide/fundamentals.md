---
name: fundamentals
description: >
  Writing any deck: netlist structure, component syntax, value suffixes,
  waveform sources, directives, `.meas` syntax.
---

# SPICE Fundamentals

## Netlist Structure

```spice
* Title line (first line, always a comment)
<components>
<directives>
.END
```

- `.END` must be the last line; nothing after it is read.
- `+` at the start of a line continues the previous statement.
- `*` starts a full-line comment. The inline comment is `;` in LTspice and `$`
  in ngspice (`;` is not an inline comment in ngspice).

## Component Syntax

```
<ref> <node+> <node-> <value>
R1 in out 10k
C1 out 0 100n
V1 in 0 AC 1 PULSE(0 5 0 1n 1n 0.5m 1m)
```

## Value suffixes

| Suffix | Meaning | Value |
|-|-|-|
| f | femto | 1e-15 |
| p | pico | 1e-12 |
| n | nano | 1e-9 |
| u | micro | 1e-6 |
| m | milli | 1e-3 |
| k | kilo | 1e3 |
| MEG | mega | 1e6 |
| G | giga | 1e9 |
| T | tera | 1e12 |

`M` is milli, so `1M` is 0.001; write `MEG` for 1e6. A suffix letter SPICE
does not know is ignored without an error, leaving the bare number.

## Waveform Sources

```spice
PULSE(Vinitial Vpulse Tdelay Trise Tfall Ton Tperiod Ncycles)
SINE(Voffset Vamp Freq Td Theta Phi Ncycles)
EXP(V1 V2 Td1 Tau1 Td2 Tau2)
SFFM(Voff Vamp Fcar MDI Fsig)
PWL(t1 v1 t2 v2 ...)
PWL file=<filename>
```

An `.ac` sweep needs an `AC <mag> [phase]` term on a source, such as
`V1 in 0 AC 1`. It sets only the small-signal amplitude (the transfer function
is normalized to it) and is independent of the time-domain waveform, so one
source can carry both: in `V1 in 0 AC 1 SINE(0 1 1k)`, `AC 1` drives `.ac` and
the `SINE` drives `.tran`. Without an `AC` term, an `.ac` run has no excitation
and every node reads 0.

## Directives

```spice
.tran 5m                          ; transient, 5ms stop
.tran 0 5m 0 10u                  ; tstep, tstop, tstart, tmaxstep
.ac dec 200 10 100k               ; AC sweep, 200pts/decade, 10Hz-100kHz
.dc V1 0 5 0.01                   ; DC sweep V1, 0-5V, 10mV step
.op                               ; DC operating point
.noise V(out) V1 dec 200 10 100k  ; noise analysis
.tf V(out) V1                     ; DC transfer function
.include /path/to/model.lib       ; include library
.ic V(node)=1.5                   ; initial conditions (used with UIC)
.nodeset V(node)=1.5              ; hint for DC operating point solver
```

`.ic` forces node voltages at t=0 (use it with `.tran ... UIC`). `.nodeset` is
a starting guess for the operating-point solver, which can move away from it.
Using one for the other gives wrong initial states or convergence failures.

## .MEAS Syntax

```spice
.meas TRAN vmax MAX V(out)
.meas TRAN vpp PP V(out)
.meas TRAN trise TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1
.meas AC fc WHEN mag(V(out)/V(in))=0.707
.meas AC gain_1k FIND mag(V(out)) AT=1k
.meas TRAN avg_out AVG V(out) FROM=1m TO=5m
.meas TRAN energy INTEG V(out)*I(R1)
```

Prefer `.meas` for a scalar it can express: the simulator computes it, it stays
in the deck, and the `measurements` recipe reads it back. Use the other recipes
for what `.meas` cannot express, such as FFT and THD, Bode structure or
windowed statistics. On ngspice a top-level `.meas` is skipped and its value
comes back absent (guide section 'ngspice').

A single `.meas` cannot return where a peak is:
`.meas AC fpeak MAX mag(V(out))` gives the peak's value, not its frequency.
Capture the peak, then
find where the signal equals it:

```spice
.meas AC vpeak  MAX  mag(V(out))
.meas AC fcenter FIND frequency WHEN mag(V(out))=vpeak
```

The `resonance` recipe gives the peak frequency, Q and bandwidth in one step.

**Behavior that gives a wrong number without an error:**
- RISE/FALL/CROSS numbering starts at 1, not 0.
- On LTspice, trig functions inside a `.meas` take and give degrees, where a
  B source uses radians (guide section 'ltspice').
- On LTspice, `db(V(out))` in an AC `.meas` is the complex logarithm of the
  complex voltage, and it reads back as that number's magnitude: 7.46 where
  the gain is -3.01 dB. Write `db(mag(V(out)))`. A `WHEN db(V(out))=-3`
  crossing is found either way.
- `MAX` returns the largest signed value. On a trace that stays negative, such
  as a PMOS drain current from −3 mA to −1 mA, `.meas TRAN imax MAX I(V1)`
  returns −1 mA. For the peak magnitude, measure `MAX abs(I(V1))`.
- If the TRIG event never occurs, the measurement fails silently.
- Without `TD=`, TARG matches from t=0 and can hit the wrong edge.
- AC measurements use a 65k-point ceiling; exceeding it silently reduces
  resolution.
- A WHEN or AT measurement returns the crossing time (`.tran`) or frequency
  (`.ac`) in the result's `at` field; the headline `values` scalar is the
  target level, not the crossing point.

## General notes

- Node `0` and node `00` are different nodes. Ground is `0` (or `GND`).
- Impedance ratios beyond about 1e16 cause numerical problems (64-bit doubles).
