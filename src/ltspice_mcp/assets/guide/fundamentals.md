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

- `.END` must be last line. No statements after it.
- `+` at start of line continues previous statement.
- Comments: `*` (full line). Inline comment is `;` in LTspice, `$` in ngspice
  (`;` is not an inline comment in ngspice).

## Component Syntax

```
<ref> <node+> <node-> <value>
R1 in out 10k
C1 out 0 100n
V1 in 0 AC 1 PULSE(0 5 0 1n 1n 0.5m 1m)
```

## Value Notation — CRITICAL

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

**`M` means milli, not mega. Use `MEG` for 1e6.**
`1M` = 0.001, not 1000000.
Unrecognized suffix letters are silently ignored — no error, just wrong value.

## Waveform Sources

```spice
PULSE(Vinitial Vpulse Tdelay Trise Tfall Ton Tperiod Ncycles)
SINE(Voffset Vamp Freq Td Theta Phi Ncycles)
EXP(V1 V2 Td1 Tau1 Td2 Tau2)
SFFM(Voff Vamp Fcar MDI Fsig)
PWL(t1 v1 t2 v2 ...)
PWL file=<filename>
```

**PWL extras (LTspice-specific):**
- Relative time: `PWL(0 1 +1 2 +1 3)` — times become 0, 1, 2
- Repetition: `REPEAT FOR n (...) ENDREPEAT` or `REPEAT FOREVER (...) ENDREPEAT`
- Scaling: `VALUE_SCALE_FACTOR=x`, `TIME_SCALE_FACTOR=x`
- Trigger: `TRIGGER <expression>` — output stuck at first value when expression is false

**AC small-signal stimulus:** A `.ac` sweep needs an `AC <mag> [phase]` term on a source — e.g. `V1 in 0 AC 1`. It sets the small-signal amplitude only (the transfer function is normalized to it) and is independent of any time-domain waveform, so one source can carry both: `V1 in 0 AC 1 SINE(0 1 1k)` — `AC 1` drives `.ac`, `SINE(...)` drives `.tran`. Without the `AC` term a `.ac` run has zero excitation and every node reads 0.

## Directives

```spice
.tran 5m                          ; transient, 5ms stop
.tran 0 5m 0 10u                  ; tstep, tstop, tstart, tmaxstep
.tran 0 5m 0 10u startup          ; LTspice-only: ramp sources from zero
.ac dec 200 10 100k               ; AC sweep, 200pts/decade, 10Hz-100kHz
.dc V1 0 5 0.01                   ; DC sweep V1, 0-5V, 10mV step
.op                               ; DC operating point
.noise V(out) V1 dec 200 10 100k  ; noise analysis
.tf V(out) V1                     ; DC transfer function
.include /path/to/model.lib       ; include library
.ic V(node)=1.5                   ; initial conditions (used with UIC)
.nodeset V(node)=1.5              ; hint for DC operating point solver
```

`.ic` forces node voltages at t=0 (use with `.tran ... UIC`). `.nodeset` is a suggestion to help the OP solver converge — the solver can override it. Mixing them up causes wrong initial states or convergence failures.

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

**Prefer `.meas` for any scalar it can express.** The simulator computes it,
and it stays in the deck, so it is reproducible and can be re-run in the
LTspice GUI; results come back through the `measurements` recipe. The post-hoc
recipes (`bode_filter`, `signal_stats`, `thd`, …) parse the `.raw`
in-process, which `.meas` avoids. Use them for derived metrics `.meas` cannot
express (FFT/THD, structural Bode, arbitrary windowed stats) or to skip a
re-run, not as the default for a scalar `.meas` could compute. Exception: ngspice skips `.meas` under the server's `-b -r`
batch mode — on ngspice, read the trace with an `analyze_results` recipe or use a
dot-less `meas` inside a `.control` block (the `.meas`-under-batch note in
guide section 'ngspice').

**Finding the frequency/time of a maximum (argmax):** a single `.meas` cannot
return the x-location of a peak — `.meas AC fpeak MAX mag(V(out))` gives the
peak *value*, not its frequency. Use two directives (capture the peak, then
find where the signal equals it):
```spice
.meas AC vpeak  MAX  mag(V(out))
.meas AC fcenter FIND frequency WHEN mag(V(out))=vpeak
```
Or use the `resonance` recipe (AC) for peak frequency + Q + bandwidth in one step.

**Important behavior:**
- RISE/FALL/CROSS numbering starts at **1**, not 0.
- **`MAX` on a signed (always-negative) trace does not give the peak magnitude**: for a PMOS
  drain current that swings −3 mA…−1 mA, `.meas TRAN imax MAX I(V1)` returns
  **−1 mA** (the least-negative sample), not the 3 mA peak magnitude — no
  error, just the wrong "peak". Wrap it: `.meas TRAN imax MAX abs(I(V1))`
  (expression functions work inside `.meas`; verified against LTspice).
- If TRIG event never occurs, measurement silently fails (no error, no warning).
- Without `TD=` parameter, TARG matches from t=0 — can hit wrong edge.
- AC measurements use **65k point ceiling** — exceeding this silently reduces resolution.
- WHEN/AT measurements return the crossing time (.tran) or frequency (.ac) in the result's `at` field; the headline `values` scalar is the constant target level, not the crossing point.
- **Quantized / staircase signals** (transmission-line reflections, DAC steps):
  read each plateau level directly with a `value` recipe (`{"metric": "value",
  "expr": "V(out)", "at": …}`), or take the full table with a `waveform` recipe
  at `"format": "csv"` — don't reconstruct levels from the inline `waveform`
  envelope's bucket statistics.

## General notes

- **Node "0" vs "00"**: Different nodes. Ground is `0` (or `GND`).
- **Impedance ratios**: Beyond ~1e16 cause numerical issues (64-bit doubles).
- **Parameter sweep**: `.step param <name> <start> <stop> <increment>`
- **Parameter list**: `.step param <name> list <v1> <v2> ...`
