
# SPICE Circuit Simulation Guide

Covers both engines. **SPICE Fundamentals** applies to both; then read
**LTspice-Specific** or **ngspice-Specific** depending on the active simulator
(the server names it in its instructions). The engines differ in inline-comment
character, behavioral-source conditionals, MOSFET bulk handling, parameter
sweeps, and Monte Carlo — see the differences table at the end.

## SPICE Fundamentals

### Netlist Structure

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

### Component Syntax

```
<ref> <node+> <node-> <value>
R1 in out 10k
C1 out 0 100n
V1 in 0 AC 1 PULSE(0 5 0 1n 1n 0.5m 1m)
```

### Value Notation — CRITICAL

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

**`M` means MILLI, not mega. Use `MEG` for 1e6.**
This is the #1 SPICE mistake. `1M` = 0.001, not 1000000.
Unrecognized suffix letters are silently ignored — no error, just wrong value.

### Waveform Sources

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

### Directives

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

### .MEAS Syntax

```spice
.meas TRAN vmax MAX V(out)
.meas TRAN vpp PP V(out)
.meas TRAN trise TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1
.meas AC fc WHEN mag(V(out)/V(in))=0.707
.meas AC gain_1k FIND mag(V(out)) AT=1k
.meas TRAN avg_out AVG V(out) FROM=1m TO=5m
.meas TRAN energy INTEG V(out)*I(R1)
```

**Finding the frequency/time OF a maximum (argmax):** a single `.meas` cannot
return the x-location of a peak — `.meas AC fpeak MAX mag(V(out))` gives the
peak *value*, not its frequency. Use two directives (capture the peak, then
find where the signal equals it):
```spice
.meas AC vpeak  MAX  mag(V(out))
.meas AC fcenter FIND frequency WHEN mag(V(out))=vpeak
```
Or call `resonance` (AC) for peak frequency + Q + bandwidth in one step.

**Gotchas:**
- RISE/FALL/CROSS numbering starts at **1**, not 0.
- **`MAX` on a signed (always-negative) trace is a silent trap**: for a PMOS
  drain current that swings −3 mA…−1 mA, `.meas TRAN imax MAX I(V1)` returns
  **−1 mA** (the least-negative sample), not the 3 mA peak magnitude — no
  error, just the wrong "peak". Wrap it: `.meas TRAN imax MAX abs(I(V1))`
  (expression functions work inside `.meas`; verified against LTspice).
- If TRIG event never occurs, measurement silently fails (no error, no warning).
- Without `TD=` parameter, TARG matches from t=0 — can hit wrong edge.
- AC measurements use **65k point ceiling** — exceeding this silently reduces resolution.
- WHEN/AT measurements return the crossing time (.tran) or frequency (.ac) in the result's `at` field; the headline `values` scalar is the constant target LEVEL, not the crossing point.
- **Quantized / staircase signals** (transmission-line reflections, DAC steps):
  read the plateau levels directly with `query_value(at=...)` per plateau, or
  `export_waveform` for the full table — don't reconstruct levels from
  `get_waveform` bucket statistics.

### General Pitfalls

- **Node "0" vs "00"**: Different nodes. Ground is `0` (or `GND`).
- **Impedance ratios**: Beyond ~1e16 cause numerical issues (64-bit doubles).
- **Parameter sweep**: `.step param <name> <start> <stop> <increment>`
- **Parameter list**: `.step param <name> list <v1> <v2> ...`

---

## Reading device operating points (gm/gds/vth, gm/ID characterization)

The small-signal / model parameters of a MOSFET/BJT/diode (gm, gds, gmbs, vth,
vdsat, gm/ID, the capacitances) come back from this server as **named numbers** —
no rawfile parsing, no `.control`/`wrdata` block. Both simulators expose them;
they just live in different files, so pick by what you need.

### One device at a single bias → `operating_point` (works on LTspice)

```spice
M1 d g 0 0 nch L=0.18u W=2u
Vd d 0 1.8
Vg g 0 0.9
.op
.lib /path/to/models.lib
.end
```
`run_simulation` then `operating_point(device='M1')` returns gm, gds, vth, vdsat,
the caps and terminal currents at that bias. On **LTspice** these come from the
log's *Semiconductor Device Operating Points* block — `run_simulation` adds
`.options logopinfo` automatically for `.op` runs (LTspice writes the block only
under that option, and only for `.op`). On **ngspice**, `.save @m1[gm] @m1[gds]
@m1[vth] @m1[id]` puts them in the raw; `operating_point` reads either uniformly.

### gm/ID curve vs a swept bias (the sizing table) → ngspice `.dc` + `export_waveform`

```spice
.dc Vg 0 1.8 0.01      ; sweep the gate
.save @m1[gm] @m1[gds] @m1[vth] @m1[id]
```
`export_waveform(signals=['m1.gm','m1.gds','m1.id'])` is the gm/ID table; one
value at a chosen bias → `query_value(signal='m1.gm', at=...)`. This swept form
needs **ngspice** (LTspice's `logopinfo` is `.op`-only; for a swept gm on LTspice
you'd differentiate the drain current, `d(Id(M1))`, instead). Don't reach for
`configure_sweep` — a native `.dc Vds Vgs` is one deck, not N separate runs.

Address an operating-point param by the `m1.gm` shorthand or its literal `@m1[gm]` name; the
tools resolve the bare / `v()` / `i()` wrapping, and a subcircuit path like
`x1.m1.gm`, for you. Values carry SI units where the simulator declares the type.

---

## Impedance, return loss, and noise figure (RF / two-port idioms)

`return_loss` computes Γ / return loss / VSWR from the impedance trace in one
call (see below). S-parameters (S21) and noise figure have no dedicated tool —
they are a short arithmetic step from an ordinary `.ac`/`.noise` run; the idioms
below are what to build.

### Input impedance → Z, Γ, return loss, VSWR

Drive the node with a **1 A AC current source whose `+` terminal is at ground**,
then `V(node)` *is* `Zin` (a 1 A probe makes V numerically equal to Z):

```spice
I1 0 in AC 1      ; + at ground, - at the probed node
* ... your one-port hangs off 'in' ...
.ac dec 50 1meg 1g
```

`V(in)` comes back complex: magnitude = |Zin| in ohms, phase = ∠Zin. Read it with
`query_value` at one frequency or `export_waveform` for the full |Zin|(f) table;
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

The `return_loss` tool applies exactly this — pass the impedance trace and `z0`
(default 50), and it returns Γ (mag/phase), `return_loss_db`, and `vswr` at a
given `at` frequency, or the worst-match point across the sweep when `at` is
omitted. It flags a negative-real Zin (a reversed probe) in its warnings.

### Noise figure from `.noise`

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

### Insertion loss / S21

A matched source and load (`Rs = RL`) form a 2:1 divider, a fixed **-6 dB** offset
at the load. `bode_metrics(mode="filter")` measures the -3 dB corner *relative to
the measured passband*, so that -6 dB offset does not move the corner — no
normalization needed for bandwidth/corner reporting. For **absolute-dB** S21 where
the matched passband should read 0 dB, drive the source with `AC 2` to pre-cancel
the 6 dB divider loss.

---

## LTspice-Specific

### Parameters and Expressions

```spice
.param Rval=10k
.param fc={1/(2*pi*R1*C1)}
.func myfn(x) {x*2}
```

- Component values referencing params MUST use braces: `R1 in out {Rval}`
- `.param` using other params MUST use braces: `.param x={y*2}`
- `.func` body uses braces: `.func myfn(x) {x*2}`
- B source expressions: do NOT wrap the expression itself in curly braces — parameters inside B source expressions DO use braces: `B1 out 0 V=V(in)*{Rval}`

### Behavioral Sources (B sources)

Four types:

```spice
B1 out 0 V=<expression>                      ; voltage source
B2 out 0 I=<expression> [Rpar=x] [Cpar=x]    ; current source
B3 out 0 R=<expression>                       ; resistor (undocumented)
B4 out 0 P=<expression> [VprXover=x]          ; power sink (undocumented)
```

**Conditional:** `IF(cond, true, false)` — NOT ternary `?:` (that's ngspice).
B source expressions must be single-line in schematics (netlists can use `+` continuation).

**Operator precedence:**
1. `~`, `!` (boolean NOT)
2. `**` (exponentiation) — `^` is XOR except in Laplace expressions
3. `*`, `/`
4. `+`, `-`
5. `==`, `>=`, `<=`, `>`, `<` (comparisons → boolean)
6. `^` (XOR), `|` (OR), `&` (AND)

Boolean: >0.5 is True, ≤0.5 is False.

**Math functions:**
- Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y,x)`, `hypot(y,x)`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Exp/log: `exp`, `ln`, `log` (base e), `log10`
- Power: `sqrt`, `pow(x,y)`, `pwr(x,y)` (sign-preserving), `pwrs(x,y)`, `square`
- Rounding: `round`, `int`, `floor`, `ceil`
- Limits: `min`, `max`, `limit(x,lo,hi)`, `uplim(x,pos,z)`, `dnlim(x,neg,z)`
- Logic: `buf`, `inv`
- Lookup: `table(x,x1,y1,x2,y2,...)` — monotonic x required

**Time-domain functions:**
- `ddt(x)` — time derivative
- `idt(x[,ic[,assert]])` — integral; assert≠0 resets
- `sdt(x)` — alternate integral
- `delay(x,y)` — delay by y seconds
- `uramp(x)` — ramp: x if x>0, else 0
- `u(x)`, `stp(x)` — unit step (undocumented)

**Random:** `rand(x)` (sharp), `random(x)` (smooth), `white(x)` (noise ±0.5)

**Special variables:** `time`, `pi`, `boltz` (1.38e-23), `planck` (6.63e-34), `echarge` (1.60e-19), `kelvin` (-273.15), `Gmin` (1e-12)

**Laplace filter:**
```spice
B1 out 0 V=V(in) Laplace=1/(1+s/{2*pi*fc})
```
In Laplace expressions, `^` means exponentiation (not XOR). Response must roll off at high frequencies.

**Gotchas:**
- `^` is **XOR** in normal expressions, exponentiation only in Laplace. Use `**` for power.
- `R=<expr>` behavioral resistor: value must never reach zero (causes convergence failure).
- `NoJacob` flag exists but "greatly increases risk of convergence problems" — avoid.

### Monte Carlo

LTspice has no built-in `.mc` directive — use `.step` + `mc()`:

```spice
.step param run 1 100 1
R1 in out {mc(10k, 0.1)}         ; uniform dist, 10k +/-10%
```

`mc(nominal, tolerance)` — uniform between `nom*(1-tol)` and `nom*(1+tol)`.

### Convergence

```spice
.options gmin=1e-10               ; min conductance on diode/transistor junctions
.options abstol=1e-10             ; absolute current tolerance (default 1e-12)
.options reltol=0.003             ; relative tolerance (never exceed 0.003)
.options cshunt=1e-15             ; capacitance from every node to ground
.options method=gear              ; alternate integration method
```

**Circuit design tips:**
- p/n junctions should have some series resistance and parallel capacitance.
- Avoid strict ideal voltage sources — add realistic parasitics.
- Impedance ratios beyond 1e16 cause numerical issues.
- Be suspicious of circuits needing `cshunt` — may indicate unrealistic models.

**Bistable/multi-root circuits (bandgaps, current mirrors, latches):** the DC
solver converges to *a* root, not necessarily the intended one — a bandgap
happily "solves" at the degenerate 0 V state, a mirror at a spurious
high-current root, with no convergence warning. `.nodeset` alone often fails
to steer it (it's only an initial guess, released before the final solve).
What works: a startup circuit in the deck (as in real silicon); ramping the
supply with `.tran` + `V1 ... PWL(0 0 1m VDD)` and reading the settled state;
or `.dc` sweeping the supply *upward* so each solution seeds the next. Verify
which root you got (e.g. `operating_point` on a known-current branch) instead
of trusting `status: completed`.

**Hidden defaults (LTspice-specific):**
- `Gfarad` — default parallel conductance on capacitors (1e-12). Disable: `.options Gfarad=0`
- `DampInductors` — default parallel resistance on inductors (ON). Disable: `.options DampInductors=0`
- `Gfloat` — shunt conductance on floating nodes (1e-12 default)
- Inductor coupling factor K may be exactly `1.0` (the docs recommend starting at 1 to remove leakage ringing); use a value just under 1 only if `uic` on `.tran` causes trouble at K=±1

### .options Flags (LTspice-specific)

| Flag | Effect |
|-|-|
| `List` | Dump flattened netlist to error log |
| `DampInductors=0\|1` | Toggle parallel inductor damping |
| `Thev_Induc=0\|1` | Toggle 1mOhm series inductor resistance |
| `Gfarad=<value>` | Capacitor default parallel conductance |
| `Gfloat=<value>` | Floating-node shunt conductance |
| `TopologyCheck=2` | Beta circuit matrix optimizations |
| `baudrate=<rate>` | Enable eye diagram plotting |

### Subcircuits

```spice
.subckt myfilter in out params: R=10k C=100n
R1 in out {R}
C1 out 0 {C}
.ends myfilter
```

- `.include <path>` — include file contents verbatim.
- `.lib <path>` — same as .include in LTspice (no section argument needed).
- Model aliasing: `.model 3904 ako: 2N3904` — inherit and override parameters.
- Model stepping: `.step param STM list 3904 2222` with `Q1: {STM}`.

### Design workflow: .cir first, .asc last

**Design and iterate over `.cir` netlists** — plain text, no placement overhead, fast to edit and simulate. Only build `.asc` schematics after the circuit design is finalized or when the user needs a visual schematic for review. The `.asc` tools are for presentation, not design iteration.

### .asc Schematics

`.asc` files are structured text representing the schematic graphically. While technically readable, hand-editing is error-prone — use the server's schematic tools (`create_schematic`, `add_component`, `connect`, `apply_schematic_ops`, ...) or LTspice's GUI. These are available in both the full and agentic profiles — geometry-aware editing (orthogonal routing, pin-collision and junction checks) that hand-writing the file can't match. Ack-only mutations (move/remove a component, set an attribute, add or remove a net label, remove a wire) are `apply_schematic_ops` ops rather than standalone tools, so batch them in one transaction.

- Component attributes: Value, Value2, SpiceLine, SpiceLine2.
- Export to netlist for direct text editing when needed.
- Bus notation: `Data[0:7]` creates 8 nets (cosmetic — netlister flattens to individual nets).

#### Common symbol pin offsets (at R0)

| Symbol | Pins (name: x,y) | Size (WxH) |
|-|-|-|
| nmos | D:(48,0) G:(0,80) S:(48,96) | 48x96 |
| pmos | D:(48,0) G:(0,80) S:(48,96) | 48x96 |
| voltage | +:(0,16) -:(0,96) | 64x80 |
| current | +:(0,0) -:(0,80) | 64x80 |
| res | A:(16,16) B:(16,96) | 32x80 |
| cap | A:(16,0) B:(16,64) | 32x64 |

Rotations transform pin (x,y) as: R90→(-y,x), R180→(-x,-y), R270→(y,-x), M0→(-x,y), M180→(x,-y). Use `symbol_info` for exact positions.

**3- vs 4-terminal devices**: The basic `nmos`/`pmos` and `npn`/`pnp` symbols are 3-terminal — a MOSFET's bulk ties internally to its source, and a BJT has no separate substrate pin. When you need the body/substrate on its own net (e.g. a non-source bulk bias), use the 4-terminal variants (`nmos4`/`pmos4`, `npn4`/`pnp4`), which expose bulk/substrate as a 4th pin.

#### MOSFET orientation conventions

| Rotation | Gate side | D/S vertical | Typical use |
|-|-|-|-|
| R0 | Left | D top, S bottom | NMOS (drain up) |
| M0 | Right | D top, S bottom | NMOS mirrored (symmetric diff pair) |
| M180 | Left | D bottom, S top | PMOS (source to VDD at top) |
| R180 | Right | D bottom, S top | PMOS mirrored (gate faces right) |

**Choose orientation based on where the gate connects:**
- Gate wire must NOT cross through the component's own body. Pick the rotation that puts the gate on the side facing the signal source.
- Example: if M3's gate connects to M5 on the right → use M0 (gate right), not R0 (gate left).
- For diff pairs: M1 at R0 (gate left, toward Vinp), M2 at M0 (gate right, toward Vinn).
- For PMOS current mirrors: M4a at R180 (gate right, toward center), M4b at M180 (gate left, toward center) — gates face each other.
- Use `symbol_info` with the intended rotation to verify pin directions before placing.

#### Schematic layout best practices

**Delegate the build when you can.** Placement and wiring is meticulous,
mechanical work that competes with design attention — an agent doing both in
one pass tends to cut corners (net-label soup instead of routed wires). If
your environment supports subagents, hand the schematic build to one whose
entire brief is this playbook: give it the final netlist and this guide
section, require it to build with `create_schematic` / `apply_schematic_ops` /
`connect` (never by hand-writing the `.asc`), and have it verify before
returning — `export_netlist` must match the source netlist, and `trace_net`
must show no multi-label shorts. Review the result with `read_circuit`.

**Component placement:**
- **Tier alignment**: Matched/mirrored transistors (diff pairs, current mirrors, bias mirrors) MUST share the same y-coordinate. Plan horizontal tiers: VDD rail → PMOS loads → diff pair → tail/bias → VSS.
- **Drain/source alignment on each branch**: Within a vertical branch (e.g., PMOS load stacked above NMOS input), position components so the drain pin of the upper device is on the same x-column as the drain pin of the lower device. This eliminates horizontal jogs between stacked transistors.
- **Pin-to-rail alignment**: Place voltage/current sources so their pins land directly on the rail they connect to — no wire through the source body. For a VDD source, position it so the `+` pin y-coordinate equals the VDD rail y-coordinate. Use `symbol_info` to compute the exact placement origin from the desired pin position (e.g., for voltage `+` at y=128, place origin at y=128-16=112).
- **Minimum 128 units vertical spacing between pin levels** of adjacent tiers (e.g., between PMOS drain y and NMOS drain y). This leaves room for horizontal buses and net labels between tiers. With MOSFET bbox height of 96, plan tier origins ~192 units apart.
- **Bias circuit alignment**: Bias devices (e.g., M5/Ibias) should share the y-level of their functional counterpart (e.g., M3 tail current source).
- **Plan the full layout before placing**: Decide VDD rail y, tier y-coordinates, and bus y-coordinates first. Verify that buses fit between bounding boxes of adjacent tiers. Use `symbol_info` to check bbox extents at the intended rotation.

**Wiring:**
- **All wires must be orthogonal** — strictly horizontal or vertical. Never route diagonal wires. Use waypoints in `connect` for L-shaped or multi-segment routes.
- **Horizontal buses must route OUTSIDE all component bounding boxes.** Use `symbol_info` to check bbox extents. For PMOS M180 with bbox top at y=160, a gate bus at y=176 is INSIDE the bbox — route at y=144 (between VDD rail and bbox top) instead. Plan bus y-coordinates BEFORE placing components.
- **Vertical wires must not pass through component bodies to reach a bus.** When connecting a drain to a horizontal bus, jog the wire horizontally outside the bbox first, then route vertically to the bus. Example for PMOS M180 diode connection: route drain (400,256) → right to (448,256) → up to (448,144) → along bus to label, NOT straight up through the body at x=400.
- **Leave room for buses between tiers.** The minimum 128-unit tier spacing must account for bounding box height plus bus clearance. For PMOS M180 (bbox height 96), if VDD rail is at y=128 and PMOS origins at y=288: bbox occupies y=192–288, bus fits at y=144–160 (between rail and bbox top).
- **Heed `connect` warnings and errors**: the tool refuses diagonal wires, pin collisions, and wire junction overlaps. Non-blocking warnings (long runs, bbox crossings) should still be addressed.

**Ground and net labels:**
- **Local ground flags**: Place a ground (`0`) label directly at each grounded pin via an `apply_schematic_ops` `add_net_label` op. Never route wires to a distant ground flag.
- **One ground per pin**: Each component's ground connection gets its own `add_net_label` op at the pin's coordinates — do not share ground flags between components.
- **Do not use `connect` with `net:0`** when multiple ground labels exist — the tool errors on ambiguous net references. Place ground flags directly at pin coordinates with an `add_net_label` op (`net="0", pin="M3.S"`) — no wire needed when the flag is on the pin.
- **Named nets (VDD, outp, etc.)**: Repeating the same label at distant pins is the idiomatic way to tie them — the netlist merges same-name labels into one net (correct, not a short), no routing needed. Wire nearby pins with `connect`. Caveat: once a name carries duplicate labels, `connect` with `net:NAME` is ambiguous — target a component pin (`Ref.Pin`) instead.
- **Label any net you reference by name in a directive.** `connect` wires pins but assigns no name — at export an unlabeled net becomes `N001`, `N002`, …. So a `.meas V(vref)`, a `.param` expression using `V(x)`, or a behavioral `B`-source referencing `V(name)` silently breaks unless that exact net carries an `add_net_label`. Rule of thumb: wire-only is fine for nets you never name; **label any net a directive mentions by name.**

**Sources:**
- **Voltage source polarity**: `+` pin is at the top (smaller y), `-` at bottom. For VDD sources, `+` connects to the supply rail, `-` to ground.
- **Current source direction**: Current flows from `+` (top) to `-` (bottom) externally. Place with `+` on the higher-voltage rail.

**Models:**
- **Model names must not collide with type keywords**: Use `NMOS_3V3` not `NMOS` for `.model` names when the symbol Value is also a MOSFET type.
- **Diode default-model collision**: The `diode` symbol defaults its Value to `D`, which LTspice resolves to a built-in ideal diode. Adding your own `.model D D(...)` collides with that built-in — give the model a unique name (`.model MYDIODE D(...)`) and set the symbol's Value to `MYDIODE`, rather than reusing `D`.

### Other LTspice Quirks

- **Unicode mu**: LTspice replaces `u` with Unicode mu (µ) in saved files. Can corrupt netlists on copy/paste.
- **`startup` keyword**: LTspice-only in `.tran`. Ramps sources from zero. Not portable.
- **A-devices** (mixed-signal primitives like `SRflop`, `Counter`, `OTA`): LTspice-proprietary.
- **`*!LTspice: <directive>`**: Treated as a directive, not a comment — despite `*` prefix.
- **Area multipliers**: Undocumented `m=<value>` works on R, Q, J in addition to documented devices.
- Capacitor multiplier: `x<number>` instead of `m=<number>` (e.g., `x2`).

---

## ngspice-Specific

ngspice shares the **SPICE Fundamentals** above, with these deltas:

- Inline comment is `$`, not `;`.
- MOSFETs need 4 terminals (`M1 d g s b`) — the bulk is **not** auto-connected
  to the source. (LTspice auto-ties bulk→source only for 3-terminal VDMOS power
  symbols; a generic monolithic NMOS/PMOS needs the 4th node there too.)
- No `startup` keyword on `.tran`. (`.option ramptime` is only a DC source-
  stepping convergence aid in standard builds, not a transient soft-start — the
  true supply-ramp needs an `XSPICE_EXP` build. Ramp a source by hand with a
  PWL/PULSE rise instead.)
- No native `.step`. Run parametric sweeps through `configure_sweep` +
  `run_sweep` (one netlist per value); a `.step` line handed to `run_simulation`
  is rejected with a pointer to `configure_sweep`.
- `gnd` is auto-converted to ground (node `0`) by default; disable with
  `set no_auto_gnd` if you need `gnd` to be a distinct net.
- Extra `.meas` types: `MIN_AT`, `MAX_AT`, `DERIV`, `param='expr'`,
  `par('expr')`. `.meas ... FIND` takes `V(out)` (no `mag()` wrapper).
- `.meas` is suppressed only when batch mode (`-b`) AND a command-line `-r
  rawfile` are combined — ngspice prints "No .measure possible in batch mode
  (-b) with -r rawfile set!" (the invocation run_simulation uses). It is NOT a
  blanket batch limitation: move the measurement into a `.control ... run ...
  .endc` block and write it as the dot-less interactive `meas` command (e.g.
  `meas tran vmax MAX V(out)` — no leading dot; a dotted `.meas` inside
  `.control` is not a valid ngspice command and computes nothing). The result
  prints to the run's log. (`set measoutfile` / `.option measoutfile` does NOT
  help here — the `-b -r` combination suppresses the measurement before any
  output routing, so no file is written.) For named signals and device
  operating-point params you usually need none of this — `.save` them and read
  the raw back with run_simulation + export_waveform / query_value /
  operating_point. Reserve `.control` / `wrdata` for in-engine computation you
  genuinely can't express as a saved signal.

### Parameters and Expressions

```spice
.param Rval=10k
.param fc={1/(2*pi*Rval*Cval)}
.param combined='Rval + 10'
.func myfn(x) {x*2}
```

- Expressions in braces `{expr}` or single quotes `'expr'` — both work.
- Expressions without delimiters work only when spaces are absent:
  `.param c=a+123` OK, `.param c = a + 123` FAILS silently (assigns first token).
- Self-referential params fail silently: `.param x = {x+3}` does not work.
- Parameter names must start with alpha; may contain `! # $ % [ ] _`. Cannot use
  reserved words: `time`, `temper`, `hertz`, `not`, `and`, `or`, `div`, `mod`,
  `sqr`, `sqrt`, `sin`, `cos`, `exp`, `ln`, `log`, `log10`, `arctan`, `abs`,
  `pwr`, `defined`.

**Three separate expression parsers exist in ngspice** — a known source of
confusion:
1. **Front-end** (`.param`, brace expressions) — evaluated at netlist expansion.
2. **B source / behavioral** — evaluated during simulation (no braces).
3. **`.control` block** — operates on its own vectors/variables.

Braces `{...}` are "compile-time"; bare expressions in B sources are "run-time".

**Operator precedence (.param expressions):**

| Op | Prec | Description |
|-|-|-|
| `!` | 1 | unary NOT |
| `**`, `^` | 2 | power |
| `*` | 3 | multiply |
| `/`, `%`, `\` | 3 | divide, modulo, integer divide |
| `+`, `-` | 4 | add, subtract |
| `==`, `!=`/`<>` | 5 | equality |
| `<=`, `>=`, `<`, `>` | 5 | comparison |
| `&&` | 6 | boolean AND |
| `\|\|` | 7 | boolean OR |
| `c ? x : y` | 8 | ternary |

**`^` behavior depends on compatibility mode:**
- Default (`hs` compat): `x^y` = `pow(fabs(x), y)` for x>0; rounds y for x<0; 0 for x=0.
- LTspice compat (`lt`): `x^y` = `pow(x, y)` if y is close to integer; else 0 for x<0.

**Built-in functions (.param):**
- Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `arctan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Exp/log: `exp`, `ln`, `log` (base e), `log10`
- Power: `sqrt`, `pow(x,y)`, `pwr(x,y)` (= `pow(fabs(x),y)`)
- Rounding: `nint` (nearest, half to even), `int` (toward 0), `floor`, `ceil`
- Selection: `min`, `max`, `sgn`
- Conditional: `ternary_fcn(x,y,z)` (= `x ? y : z`)
- **Statistical:** `gauss(nom,rvar,sigma)`, `agauss(nom,avar,sigma)`,
  `unif(nom,rvar)`, `aunif(nom,avar)`, `limit(nom,avar)`
- Special: `var(name)` (interpreter variable), `vec(name)` (vector value)

### Behavioral Sources (B sources)

```spice
B1 out 0 V=<expression>
B2 out 0 I=<expression> [tc1=x] [tc2=x] [temp=x]
```

**Conditional:** ternary `cond ? true : false` — NOT `IF()` (that is LTspice).
Put a space before `?` so the parser does not confuse it with other tokens.
Nested ternaries need explicit parentheses.

**Functions (B source context):** `cos`, `sin`, `tan`, `acos`, `asin`, `atan`,
`cosh`, `sinh`, `acosh`, `asinh`, `atanh`, `exp`, `ln`, `log`, `log10`, `abs`,
`sqrt`, `u` (unit step), `u2` (ramp 0-1), `uramp`, `floor`, `ceil`, `min`, `max`,
`pow`, `**`, `pwr`, `^`, `i(device)`.

**Special variables:** `time` (transient), `temper` (circuit temp in C), `hertz`
(AC frequency). `time` is zero during AC; `hertz` is zero during transient.

**Piecewise linear in B source:**
```spice
Bdio 1 0 I = pwl(v(A), 0,0, 33,10m, 100,33m, 200,50m)
```
x values must be monotonically increasing — non-monotonic stops execution. Can
use `time` or expressions as the independent variable.

**Gotchas:**
- `exp()` is internally capped at argument=14 — beyond that it becomes linear.
- `log`/`ln`/`sqrt` of negatives use `fabs()` automatically — no error.
- Division by zero or `log(0)` causes an error.

### Subcircuits

```spice
.subckt myfilter in out rval=100k cval=100nF
R1 in p1 {2*rval}
C1 p1 0 {cval}
.ends myfilter

X1 input output myfilter rval=1k cval=1n
```

- Parameters on the `.subckt` line do NOT need a `params:` keyword — just
  `name=value` after the nodes.
- `.lib <filename> [section]` — the section name is **optional**. A bare `.lib
  models.lib` loads the whole file (verified: LTspice lists it under "Files
  loaded" and the models resolve). `.lib file section` pulls just that `.lib
  section … .endl` block. `.lib` differs from `.include` in scope (`.lib` skips
  global-scope circuit elements), not in whether a section is required.
  **Caveat — sectioned `.lib` under this server's default ngspice mode:** spicelib
  runs ngspice in a mixed LTspice/PSPICE-compatibility mode (`ngbehavior=kiltpsa`)
  whose `lt`/`ps` tokens split a sectioned `.lib <file> <section>` (the PDK corner
  idiom) into two plain includes and drop the section — the run fails with "could
  not find include file". Set `[simulator] ngbehavior = "hsa"` in `ltspice-mcp.toml`
  (or `LTSPICE_MCP_NGBEHAVIOR=hsa`) and restart to parse the section correctly.
- `.param` inside subcircuits is local scope (masks globals). Nesting to 10 levels.
- Subcircuit and model names are global — must be unique across the netlist.

### .save Directive

```spice
.save V(out) I(Vin)               $ save only these signals
.save @m1[id] @m1[gm]             $ save device operating-point params
.save all @m2[vdsat]              $ save defaults PLUS extras
```

- Without `.save`, all node voltages and source currents are saved (huge files).
- Adding even ONE `.save` line drops all defaults — only listed signals saved.
- Resistor current is the internal vector `@r1[i]` (via `.save @r1[i]` or
  `.options savecurrents`); under this path the `i(r1)` read-function does NOT
  resolve it — `i()`/`I()` only resolve `name#branch` vectors (voltage sources,
  and the sense source a separate `.probe I(R1)` directive inserts). This server
  reads the `@r1[i]` form.
- Saved device operating-point params (`@m1[gm]`, `v(@m1[vth])`, `i(@m1[id])`, …) are surfaced
  by `operating_point` in a `device_op_points` bucket (a bare `.op`), and on a
  `.dc`/`.tran` sweep are readable by the `dev.param` shorthand — `query_value`/
  `signal_stats`/`export_waveform` accept `m1.gm` and resolve it to the actual
  trace. This is the gm/ID idiom: `.dc Vg …` + `.save @m1[gm] @m1[id]`, then read
  `m1.gm`/`m1.id` per sweep point.

### .control / .endc Blocks

ngspice has a built-in scripting language for post-simulation analysis:

```spice
.control
run                               $ execute the simulation
let vmax = maximum(V(out))        $ create a vector
set filename = "results.csv"      $ create a string variable
write $filename V(out) I(Vin)     $ save to a rawfile
wrdata output.txt V(out)          $ save as CSV-like text
.endc
```

**Variables vs vectors — a critical distinction:**
- `set` creates string/shell variables: `set myvar = "hello"` — access `$myvar`.
- `let` creates numeric vectors: `let x = 2*pi` — access `$&x` to get a number.
- `$&param` dereferences a circuit `.param` into a control variable.

**Control structures:** `while`/`end`, `repeat`/`end`, `foreach`/`end`,
`if`/`else`/`end`, `dowhile`, `break [n]`, `continue [n]`, `label`, `goto`.
`foreach` values are space-separated (no commas).

**Key commands:** `run`, `plot`, `print`, `let`, `set`, `write`, `wrdata`,
`alter`, `altermod`, `echo`, `meas`, `linearize`, `fft`, `define`, `source`.

### Monte Carlo

ngspice has **no `.mc` directive**. Two idioms:

**(1) Per-device statistical functions (primary, simplest).** Put `agauss`/
`gauss`/`unif`/`aunif`/`limit` directly in a `.param` or a device/B-source value,
in `'…'` or `{…}`. Each device card draws a fresh value at parse time:

```spice
R1 a b 'agauss(10k, 500, 3)'      $ 10k, ±500 absolute, /3 sigma
C1 c 0 '{unif(1n, 0.1)}'          $ 1n, ±10% relative, uniform
```

These are built into the numparam frontend (no build flag) but live ONLY there,
NOT in the nutmeg/`.control` interpreter. For a distribution, re-run the deck N
times (set `.options seed=<value>`); `run_montecarlo` automates the N-run draw +
aggregation.

**(2) `.control` loop with `alter`** — vary within one ngspice invocation. Inside
`.control` only `sgauss(0)` (Gaussian, mean 0, σ 1) and `sunif(0)` (uniform
[-1,1]) are built in — scale them yourself (`agauss`/`gauss` are NOT nutmeg
functions here unless you `define` them first):

```spice
.control
let run = 1
dowhile run <= 100
  alter c1 = 1n * (1 + 0.1*sunif(0))
  alter r1 = 10k * (1 + 0.05*sgauss(0))
  tran 1u 1m
  $ ... store/process results ...
  let run = run + 1
end
.endc
```

Set the seed with `.options seed=<value>` or `seed=random`.

### .options Flags

**General:**

| Flag | Effect |
|-|-|
| `SEED=val\|random` | Random number seed |
| `TEMP=x` | Operating temperature (default 27C) |
| `TNOM=x` | Nominal temperature (default 27C) |
| `SAVECURRENTS` | Auto-save all device terminal currents |
| `KLU` | KLU matrix solver (faster for large MOS circuits) |
| `INTERP` | Interpolate output to a fixed TSTEP grid |

**Convergence:** `RELTOL` (0.001), `ABSTOL` (1e-12), `VNTOL` (1e-6), `GMIN`
(1e-12), `ITL1` (100, DC iterations), `ITL4` (10, transient iterations),
`METHOD` (`trap` or `gear`), `MAXORD` (2; Gear max 2-6), `TRTOL` (7),
`XMU` (0.5; reduce slightly to suppress ringing).

**Matrix conditioning:**
```spice
.options rshunt=1e12              $ resistor from every node to ground
.options rseries=1e-4            $ series resistor on every inductor
.options cshunt=1e-13            $ capacitor from every node to ground
```
Use `rshunt` for "no DC path to ground" errors, `rseries` when inductors across
voltage sources fail OP, `cshunt` for oscillation/noise. `AUTOSTOP` halts the
transient once all `.meas` conditions are satisfied.

### XSPICE

Mixed-signal simulation with code models. XSPICE is enabled by default in the
official ngspice Windows binaries (and this Linux build — verified: an `A`-device
`gain` code model runs); only the experimental `XSPICE_EXP` extras (e.g. the
capacitor/inductor code models and transient supply-ramping) need a custom build.

```spice
A1 [in] [out] lut1
.model lut1 d_lut(rise_delay=1n fall_delay=2n input_load=0.5p
+ table_values="0110")
```

Digital device types: `d_and`, `d_or`, `d_nand`, `d_nor`, `d_xor`,
`d_inverter`, `d_buffer`, `d_flop`, `d_latch`, `d_lut`, etc. Digital nodes use
`[name]` bracket syntax for buses.

### Key Differences: LTspice vs ngspice

| Aspect | LTspice | ngspice |
|-|-|-|
| Inline comment | `;` | `$` |
| B-source conditional | `IF(c,a,b)` | ternary `c ? a : b` |
| `^` operator | XOR (power is `**`) | power |
| MOSFET bulk | auto-tied to source only on 3-term VDMOS symbols | required 4th terminal |
| `GND` node | alias for `0` | auto-converted to `0` (disable: `set no_auto_gnd`) |
| `.tran startup` | supported | not supported (no transient soft-start) |
| Parameter sweep | `.step` | `configure_sweep`/`run_sweep` (no `.step`) |
| Monte Carlo | `.step` + `mc()` | `agauss`/`gauss`/`unif` on device values (primary); or `.control` `alter` loop |
| Post-processing | — | `.control` scripting (`let`/`plot`/`write`/`fft`) |
| Default saving | saves all | `.save` (one line drops defaults; `.save all` keeps) |
| `.raw` format | mixed precision | all doubles |
| Unicode mu | replaces `u` with µ | preserves `u` |

Other ngspice notes: A-devices ARE the XSPICE code-model primitives (the `A`
prefix — available in stock builds, above); `.func` cannot be recursive
(textual expansion, so a self-reference expands without bound). `.backanno` is
an LTspice-only directive — ngspice rejects it ("unimplemented dot command
'.backanno'") and aborts the run; ngspice current probing uses `.options
savecurrents` / `.probe` instead.
