---
name: spice-bench-craft
description: >
  Use when authoring SPICE characterization benches or testbench netlists,
  especially for servo-loop or open-loop amplifier measurements, DC-servo
  needs, biasing struggles or railed operating points, phase-margin and
  crossing practice, supply-current measurement, or reusable parameterized
  bench-template needs. Covers LTspice and ngspice bench craft.
---

# SPICE measurement bench craft

Author the bench from one parameter dictionary, establish the operating point,
then measure. The authoritative measurement-practice source is bundled at
`references/BENCH_NOTES.md`; read it when exact source wording is useful.

## The operating point comes first

An amplifier with 60-100 dB of DC gain multiplies any input offset by
1,000-100,000. Driving both inputs from ideal DC sources at the same
potential does NOT put the output at mid-rail: the amplifier's own
input-referred offset (even tens of microvolts) slams the output into a
supply rail, and every small-signal quantity you then measure — gain,
bandwidth, phase — describes a saturated transistor stack, not the
amplifier. Symptoms: DC gain tens of dB lower than expected, an .op
output voltage within ~100 mV of either rail, device operating regions
showing triode/cutoff where saturation was intended.

ALWAYS check v(out) in the .op result before trusting an AC sweep. If it
is not near the intended quiescent level, the measurement is invalid.

## The servo trick: DC feedback that vanishes at AC

The standard bench closes the loop at DC only, with elements too large
to matter in the measured band:

    * loop closed at DC through a huge inductor; AC injected differentially
    LFB  out  inn  1T        ; 1 tera-henry: short at DC, open at any AC freq
    CFB  inn  0    1T        ; 1 tera-farad: blocks DC, grounds inn for AC
    VIP  inp  0    DC {CM} AC 0.5
    * inn receives -0.5 of the AC drive THROUGH the cap if driven; or drive
    * single-ended and read v(out)/v(inp-inn) — both are open-loop above
    * the (vanishingly low) servo corner at 1/(2*pi*sqrt(LC)).

The feedback forces the output to the level that zeroes the input
differential at DC — the amplifier chooses its own correct operating
point — while above ~microhertz the loop is open and v(out)/v(inp-inn)
is the true open-loop transfer function.

## Reading the result

- Gain: magnitude at the lowest swept frequency (sweep from well below
  the dominant pole).
- Unity crossing: the FIRST 0 dB crossing; verify there is only one, or
  evaluate phase at every crossing — a later crossing can hide an
  unstable mode a first-crossing readout misses.
- Phase: unwrap before computing margin; modulo-360 artifacts fake
  healthy margins.
- Supply current: measure the supply SOURCE current at the .op point,
  not a sum of device currents.

## Cheap sanity checks that catch most wrong benches

1. .op first, AC second: output within the linear region?
2. Does measured DC gain roughly match gm*ro expectations (within an
   order of magnitude)? A 25 dB shortfall is a broken bench, not a bad
   amplifier.
3. Re-run one point at 2x sweep density: if gain/PM move, the sweep or
   interpolation, not the circuit, is speaking.

## Render benches from one parameter dictionary

Keep DUT pin order and run conditions in data, not copied through dozens of
decks. A useful dictionary has `DUT_INCLUDE`, `DUT_INSTANCE`, `VDD`, `VCM`,
`CL`, `TEMP`, and analysis limits such as `FSTART`, `FSTOP`, or `TSTOP`.
Render the `@NAME@` host placeholders below, preserve SPICE `{PARAM}` braces,
and reject output containing an unresolved `@NAME@`. Keep the title first and
`.end` last. Use absolute or deck-relative include paths deliberately.

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
as the only DC feedback path. Always compute gain and phase from
`V(out)/V(inp,inn)`, unwrap phase, inspect every 0 dB crossing, and repeat at
twice the point density.

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

Before extracting slew or settling, confirm the initial `.op` is linear and
the requested step does not turn the measurement into an overload test.

## ngspice batch-output practice

This server invokes ngspice with batch mode and a raw output (`-b -r`), which
suppresses top-level `.meas`. Move measurements into a `.control` block as the
dot-less interactive `meas` command; a dotted `.meas` inside `.control` is not
valid.

```spice
.control
run
meas ac gain_10 find vdb(out) at=10
meas ac unity when vdb(out)=0 cross=1
.endc
```

Prefer saved traces plus the server's analysis tools when they express the
metric. If using `wrdata`, its columns repeat the scale vector for every dumped
vector: dumping `v(out)` and `i(VDD)` yields `scale, v(out), scale, i(VDD)`,
not one shared scale followed by both values. Parse repeated scale/value groups,
and remember that `wrdata` writes only the text table; add an explicit `write`
too if later server analysis needs a rawfile.

## Routing: which vehicle runs what

- A quick one-off ngspice run: shell out and run it yourself — it is the
  cheapest path. Keep the rawfile.
- Reading numbers out of any rawfile (yours or the server's): the server's
  analysis tools — `analyze_results` accepts a bare `raw_path` for runs it
  never executed, and returns parsed values, reductions, and spec verdicts.
- LTspice execution, declared sweep/corner/Monte-Carlo matrices, or a run
  long enough to outlive one call: `run_experiments` — durable idempotent
  submission, then `jobs` to wait or cancel.
- Schematic (`.asc`) work: the schematic tools, never hand-written files —
  they carry orthogonal routing and pin-collision/junction checks.
- Open-ended interrogation — loops over results, compute-decide-compute,
  arbitrary math on traces: if `ltspice_mcp` is importable, write Python
  against `ltspice_mcp.api` (`Api`, `load_raw`, the metric functions);
  intermediates stay in your interpreter.
- Result belongs on disk, not your reply, or no server connected:
  `spice-mcp <op> @plan.json --json` — same ops; exit codes are verdicts.
