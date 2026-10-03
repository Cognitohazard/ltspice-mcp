---
name: rf
description: >
  Input impedance, reflection coefficient, return loss, VSWR, S21 or noise
  figure from `.ac` and `.noise` runs.
---

# Impedance, return loss, and noise figure (RF / two-port idioms)

The `return_loss` recipe computes Γ / return loss / VSWR from the impedance
trace in one call (see below). S-parameters (S21) and noise figure have no recipe —
they are a short arithmetic step from an ordinary `.ac`/`.noise` run; the idioms
below are what to build.

## Input impedance → Z, Γ, return loss, VSWR

Drive the node with a **1 A AC current source whose `+` terminal is at ground**,
then `V(node)` *is* `Zin` (a 1 A probe makes V numerically equal to Z):

```spice
I1 0 in AC 1      ; + at ground, - at the probed node
* ... your one-port hangs off 'in' ...
.ac dec 50 1meg 1g
```

`V(in)` comes back complex: magnitude = |Zin| in ohms, phase = ∠Zin. Read it with
a `value` recipe at one frequency, or a `waveform` recipe at `"format": "csv"`
for the full |Zin|(f) table;
`resonance` finds the peak/notch frequency. Note that a `magnitude_db` reading of
this trace is **dBΩ**, not dBV — e.g. -10.56 dB means 0.30 Ω, not a -10.56 dB
dip; `magnitude_linear` gives the ohms directly.

**Sign convention (verified):** `I1 0 in` (+ at ground) gives `V(in) = +Zin`. The
reversed `I1 in 0` gives `V(in) = -Zin` — the phase is flipped 180° and a naive
reader sees a *negative* resistance (e.g. -50 Ω). Put `+` at ground, or use
`AC -1` on the reversed source.

Then, with a reference impedance `Z0` (usually 50 Ω):

```
Γ    = (Zin - Z0) / (Zin + Z0)      ; complex
RL_dB = -20*log10(|Γ|)              ; return loss (positive dB = better match)
VSWR  = (1 + |Γ|) / (1 - |Γ|)
```

The `return_loss` recipe applies exactly this — pass the impedance trace and `z0`
(default 50), and it returns Γ (mag/phase), `return_loss_db`, and `vswr` at a
given `at` frequency, or the worst-match point across the sweep when `at` is
omitted. It flags a negative-real Zin (a reversed probe) in its warnings.

## Noise figure from `.noise`

```spice
Vin in 0 dc 0 ac 1
Rs  in n1 50            ; the source resistance whose noise sets the reference
* ... DUT from n1 to out ...
.noise V(out) Vin dec 20 1k 1g
```

Two routes — pick by simulator.

**LTspice (primary): per-source contribution traces.** LTspice's `.noise` raw
exposes one trace per noise source (`V(Rs)`, `V(R2)`, …) alongside
`V(onoise)`/`V(inoise)`. The per-source traces and `V(onoise)` are all
output-referred (`V(inoise)` is the input-referred equivalent) and the
contributions add in power (`V(onoise)² = Σ V(Rk)²`), so the noise figure is a
direct trace ratio against the source resistor's own contribution — no gain
division and no temperature constant needed:

```
NF_dB = 20*log10( V(onoise) / V(Rs) )
```

Verified on a two-resistor reference (Rs + equal series R): the power sum holds
to numerical precision and the ratio reads 3.0103 dB (equal resistors → noise
factor F = 2).

**Engine-neutral fallback (and the only route on this ngspice build):**
`noise_integral` (or the `inoise_spectrum` trace, the input-referred noise density
in V/√Hz) gives the total. The noise figure is:

```
NF_dB = 10*log10( inoise_spectrum^2 / (4*k*T*Rs) )
```

with `4*k*T = 1.657e-20` V²/Hz at the 27 °C default (`.noise` prints
`TEMP = 27.000000`; scale by `T/300.15` for other temperatures). Verified against a
two-resistor reference (Rs + equal series R into an ideal buffer → NF = 3.01 dB).

Note: this ngspice build's `.noise` exposes only `inoise_spectrum`/`onoise_spectrum`
— **not** per-device noise vectors — so the per-source trace ratio above isn't
available there; use the `4kT·Rs` denominator (it needs `Rs` and `T`, which is
why the reference resistance is explicit).

## Insertion loss / S21

A matched source and load (`Rs = RL`) form a 2:1 divider, a fixed **-6 dB** offset
at the load. The `bode_filter` recipe measures the -3 dB corner *relative to
the measured passband*, so that -6 dB offset does not move the corner — no
normalization needed for bandwidth/corner reporting. For **absolute-dB** S21 where
the matched passband should read 0 dB, drive the source with `AC 2` to pre-cancel
the 6 dB divider loss.
