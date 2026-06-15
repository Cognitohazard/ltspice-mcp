
# LTspice Circuit Simulation Guide

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
- Comments: `*` (full line) or `;` (inline).

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

**Gotchas:**
- RISE/FALL/CROSS numbering starts at **1**, not 0.
- If TRIG event never occurs, measurement silently fails (no error, no warning).
- Without `TD=` parameter, TARG matches from t=0 — can hit wrong edge.
- AC measurements use **65k point ceiling** — exceeding this silently reduces resolution.

### General Pitfalls

- **Node "0" vs "00"**: Different nodes. Ground is `0` (or `GND`).
- **Impedance ratios**: Beyond ~1e16 cause numerical issues (64-bit doubles).
- **Parameter sweep**: `.step param <name> <start> <stop> <increment>`
- **Parameter list**: `.step param <name> list <v1> <v2> ...`

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

**Hidden defaults (LTspice-specific):**
- `Gfarad` — default parallel conductance on capacitors (1e-12). Disable: `.options Gfarad=0`
- `DampInductors` — default parallel resistance on inductors (ON). Disable: `.options DampInductors=0`
- `Gfloat` — shunt conductance on floating nodes (1e-12 default)
- Inductor coupling factor K cannot reach exactly 1.0 — max is `1-1n`

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
| res | A:(0,0) B:(0,64) | 32x64 |
| cap | A:(16,0) B:(16,64) | 32x64 |

Rotations transform pin (x,y) as: R90→(-y,x), R180→(-x,-y), R270→(y,-x), M0→(-x,y), M180→(x,-y). Use `symbol_info` for exact positions.

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
- **Named nets (VDD, outp, etc.)**: Use a single label per unique net name. Connect components to it via `connect` with `net:NAME` or waypoints.

**Sources:**
- **Voltage source polarity**: `+` pin is at the top (smaller y), `-` at bottom. For VDD sources, `+` connects to the supply rail, `-` to ground.
- **Current source direction**: Current flows from `+` (top) to `-` (bottom) externally. Place with `+` on the higher-voltage rail.

**Models:**
- **Model names must not collide with type keywords**: Use `NMOS_3V3` not `NMOS` for `.model` names when the symbol Value is also a MOSFET type.

### Other LTspice Quirks

- **Unicode mu**: LTspice replaces `u` with Unicode mu (µ) in saved files. Can corrupt netlists on copy/paste.
- **`startup` keyword**: LTspice-only in `.tran`. Ramps sources from zero. Not portable.
- **A-devices** (mixed-signal primitives like `SRflop`, `Counter`, `OTA`): LTspice-proprietary.
- **`*!LTspice: <directive>`**: Treated as a directive, not a comment — despite `*` prefix.
- **Area multipliers**: Undocumented `m=<value>` works on R, Q, J in addition to documented devices.
- Capacitor multiplier: `x<number>` instead of `m=<number>` (e.g., `x2`).
