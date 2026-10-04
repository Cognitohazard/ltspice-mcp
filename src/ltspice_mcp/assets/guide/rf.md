---
name: rf
description: >
  Input impedance, reflection coefficient, return loss, VSWR, S21 or noise
  figure from `.ac` and `.noise` runs.
---

# Impedance, return loss, and noise figure (RF / two-port idioms)

The `return_loss` recipe computes Γ, return loss and VSWR from an impedance
trace in one call. S-parameters (S21) and noise figure have no recipe; each is
a short arithmetic step from an ordinary `.ac` or `.noise` run, built as below.

## Input impedance → Z, Γ, return loss, VSWR

Drive the node with a 1 A AC current source whose `+` terminal is at ground;
then `V(node)` equals `Zin` numerically:

```spice
I1 0 in AC 1      ; + at ground, - at the probed node
* ... your one-port hangs off 'in' ...
.ac dec 50 1meg 1g
```

`V(in)` comes back complex: its magnitude is |Zin| in ohms and its phase is
∠Zin. Read it with a `value` recipe at one frequency, or with a `waveform`
recipe at `"format": "csv"` for the |Zin|(f) table; `resonance` finds the peak
or notch frequency. A `magnitude_db` reading of this trace is in dBΩ, not dBV:
-10.56 dB means 0.30 Ω. `magnitude_linear` gives the ohms directly.

Sign convention: `I1 0 in` (`+` at ground) gives `V(in) = +Zin`. The reversed
`I1 in 0` gives `V(in) = -Zin`, a phase flipped by 180° that reads as a
negative resistance (such as -50 Ω). Put `+` at ground, or use `AC -1` on the
reversed source.

Then, with a reference impedance `Z0` (usually 50 Ω):

```
Γ    = (Zin - Z0) / (Zin + Z0)      ; complex
RL_dB = -20*log10(|Γ|)              ; return loss (positive dB = better match)
VSWR  = (1 + |Γ|) / (1 - |Γ|)
```

The `return_loss` recipe applies exactly this: pass the impedance trace and
`z0` (default 50), and it returns Γ (magnitude and phase), `return_loss_db` and
`vswr` at a given `at` frequency, or at the worst-match point of the sweep when
`at` is omitted. It flags a negative-real Zin (a reversed probe) in its
warnings.

## Noise figure from `.noise`

```spice
Vin in 0 dc 0 ac 1
Rs  in n1 50            ; the source resistance whose noise sets the reference
* ... DUT from n1 to out ...
.noise V(out) Vin dec 20 1k 1g
```

Two routes, by simulator.

**LTspice: per-source contribution traces.** LTspice's `.noise` raw has one
trace per noise source (`V(Rs)`, `V(R2)`, …) beside `V(onoise)` and
`V(inoise)`. The per-source traces and `V(onoise)` are output-referred
(`V(inoise)` is the input-referred equivalent), and the contributions add in
power (`V(onoise)² = Σ V(Rk)²`). So the noise figure is a direct ratio against
the source resistor's own contribution, with no gain division and no
temperature constant:

```
NF_dB = 20*log10( V(onoise) / V(Rs) )
```

**Either simulator: the input-referred density.** `noise_integral`, or the
`inoise_spectrum` trace (the input-referred noise density in V/√Hz), gives the
total, and the noise figure is:

```
NF_dB = 10*log10( inoise_spectrum^2 / (4*k*T*Rs) )
```

with `4*k*T = 1.657e-20` V²/Hz at the 27 °C default (`.noise` prints
`TEMP = 27.000000`; scale by `T/300.15` for other temperatures). This is the
route on ngspice, whose `.noise` raw holds `inoise_spectrum` and
`onoise_spectrum` but no per-device noise traces.

A quick check of either route: `Rs` plus an equal series resistor reads
3.01 dB (noise factor 2).

## Insertion loss / S21

A matched source and load (`Rs = RL`) form a 2:1 divider, a fixed -6 dB offset
at the load. The `bode_filter` recipe measures the -3 dB corner relative to
the measured passband, so that offset does not move the corner, and bandwidth
needs no normalization. For an absolute S21 where the matched passband reads
0 dB, drive the source with `AC 2` to cancel the divider's 6 dB.
