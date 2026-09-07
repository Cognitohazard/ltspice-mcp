# Measurement-practice notes: AC characterization of high-gain amplifiers

Craft notes for anyone measuring open-loop gain, bandwidth, or phase
margin of an op-amp-class circuit in a SPICE simulator. They describe
practice, not any specific task.

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
