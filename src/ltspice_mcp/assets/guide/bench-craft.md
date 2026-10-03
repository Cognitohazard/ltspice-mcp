---
name: bench-craft
kind: task
description: >
  Measuring an amplifier's open-loop gain, bandwidth, phase margin, supply
  current or step response: establish the operating point, close a DC servo
  for open-loop AC, check the result, render benches from one parameter
  dictionary, and get measurements out of ngspice.
---

# Measurement bench craft

Author the bench from one parameter dictionary, establish the operating point,
then measure.

## The operating point comes first

An amplifier with 60-100 dB of DC gain multiplies any input offset by
1,000-100,000. Driving both inputs from ideal DC sources at the same potential
does not put the output at mid-rail: the amplifier's own input-referred offset
(even tens of microvolts) drives the output to a supply rail, and the gain,
bandwidth and phase you then measure are those of a saturated transistor
stack, not of the amplifier in its linear region. Symptoms: DC gain tens of dB
lower than expected, an `.op` output voltage within ~100 mV of either rail,
device operating regions showing triode/cutoff where saturation was intended.

Always check `V(out)` in the `.op` result before trusting an AC sweep. If it is
not near the intended quiescent level, the measurement is invalid.

## DC servo for open-loop AC analysis

The standard bench closes the loop at DC only, with elements too large to
matter in the measured band:

```spice
* loop closed at DC through a huge inductor; AC drive on inp
LFB  out  inn  1T        ; 1 tera-henry: short at DC, open at any AC freq
CFB  inn  0    1T        ; 1 tera-farad: blocks DC, grounds inn for AC
VIP  inp  0    DC {CM} AC 1
* above the servo corner, 1/(2*pi*sqrt(LC)), V(out)/V(inp,inn) is the
* open-loop gain
```

At DC the feedback forces the output to the level that zeroes the input
differential, so the amplifier settles at its own correct operating point.
Above a few microhertz the loop is open and `V(out)/V(inp,inn)` is the true
open-loop transfer function.

## Reading and checking the result

- Gain: magnitude at the lowest swept frequency; sweep from well below the
  dominant pole.
- Unity crossing: use the first 0 dB crossing; verify there is only one, or
  evaluate phase at every crossing, because a later crossing can carry an
  unstable mode that a first-crossing readout misses.
- Phase: unwrap before computing margin; wrapped (modulo-360) phase can show
  a margin that is not there.
- Supply current: measure the supply source current at the `.op` point, not a
  sum of device currents.
- Compare measured DC gain with `gm*ro` expectations. A roughly 25 dB shortfall
  usually means the bench is wrong, not the amplifier.
- Re-run one point at twice the sweep density. If gain or phase margin
  changes, the difference comes from the sweep density or interpolation, not
  from the circuit.

## Render benches from one parameter dictionary

Keep DUT pin order and run conditions in data, not copied through dozens of
decks. A useful dictionary has `DUT_INCLUDE`, `DUT_INSTANCE`, `VDD`, `VCM`,
`CL`, `TEMP`, and analysis limits such as `FSTART`, `FSTOP`, or `TSTOP`.
Render the `@NAME@` host placeholders below, preserve SPICE `{PARAM}` braces,
and reject output containing an unresolved `@NAME@`. Keep the title first and
`.end` last. Choose include paths (absolute or deck-relative) explicitly.

### Operating-point and supply-current archetype

Use the same servo as the AC bench so `.op` proves the exact bias that the AC
run will linearize. Read `V(out)`, device regions, and `I(VDD)`; source-current
sign follows the source orientation.

```spice
* generated operating-point bench
.include @DUT_INCLUDE@
.param VDDV=@VDD@ VCMV=@VCM@ CLV=@CL@
.temp @TEMP@
VDD  vdd 0 {VDDV}
VIP  inp 0 DC {VCMV}
@DUT_INSTANCE@
CL   out 0 {CLV}
LFB  out inn 1T
CFB  inn 0 1T
.op
* LTspice: .options logopinfo
* ngspice: .save V(out) I(VDD) @m1[gm] @m1[gds] @m1[id]
.end
```

### Open-loop AC archetype

Keep the DUT instance, load, temperature, and servo identical to the `.op`
deck. Single-ended `AC 1` makes `V(out)/V(inp,inn)` the unambiguous loop gain;
use the differential denominator even when it is numerically one.

```spice
* generated open-loop AC bench
.include @DUT_INCLUDE@
.param VDDV=@VDD@ VCMV=@VCM@ CLV=@CL@
.temp @TEMP@
VDD  vdd 0 {VDDV}
VIP  inp 0 DC {VCMV} AC 1
@DUT_INSTANCE@
CL   out 0 {CLV}
LFB  out inn 1T
CFB  inn 0 1T
.ac dec @POINTS_PER_DECADE@ @FSTART@ @FSTOP@
.save V(out) V(inp) V(inn) I(VDD)
.end
```

For balanced drive, give `VIP` `AC 0.5 0`, replace `CFB inn 0 1T` with
`CFB inn ndrive 1T`, and add `VIM ndrive 0 DC 0 AC 0.5 180`; keep the inductor
as the only DC feedback path. Compute gain and phase from `V(out)/V(inp,inn)`
(the `signal` of `stability` or a `bode_*` recipe, as written) and apply the
checks under "Reading and checking the result".

### Closed-loop transient and load-step archetype

Use ordinary resistive feedback for a closed-loop transient bench; the DC-only
servo belongs to open-loop characterization. Parameterize both the command
step and the load step so one renderer covers settling and load regulation.

```spice
* generated closed-loop transient bench
.include @DUT_INCLUDE@
.param VDDV=@VDD@ VLO=@VIN_LO@ VHI=@VIN_HI@ CLV=@CL@
.temp @TEMP@
VDD   vdd 0 {VDDV}
VIN   inp 0 PULSE({VLO} {VHI} @TDELAY@ @TRISE@ @TFALL@ @TON@ @PERIOD@)
@DUT_INSTANCE@
RFB   out inn @RFB@
RG    inn 0 @RG@
CL    out 0 {CLV}
ILOAD out 0 PULSE(@ILOAD_LO@ @ILOAD_HI@ @TDELAY@ @TRISE@ @TFALL@ @TON@ @PERIOD@)
.tran 0 @TSTOP@ 0 @MAXSTEP@
.save V(inp) V(out) I(VDD) I(ILOAD)
.end
```

Before extracting slew or settling, make sure the step does not drive the
amplifier into overload.

## ngspice batch-output practice

On ngspice, `run_experiments` refuses a top-level `.meas` (guide section
'ngspice'). Prefer saved traces and `analyze_results` recipes; where a
measurement has to run inside ngspice, use the dot-less interactive `meas`
command in a `.control` block:

```spice
.control
run
meas ac gain_10 find vdb(out) at=10
meas ac unity when vdb(out)=0 cross=1
.endc
```

If you use `wrdata`, its columns repeat the scale vector for every dumped
vector: dumping `V(out)` and `I(VDD)` yields `scale, V(out), scale, I(VDD)`,
not one shared scale followed by both values. Parse repeated scale/value
groups, and remember that `wrdata` writes only the text table; add an explicit
`write` too if later server analysis needs a rawfile.
