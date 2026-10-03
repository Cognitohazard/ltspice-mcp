---
name: ltspice
description: >
  Writing a deck for LTspice: parameters, behavioral sources, Monte Carlo,
  per-instance mismatch, convergence, options, subcircuits, and quirks such
  as the micro sign.
---

# LTspice-Specific

## Parameters and Expressions

```spice
.param Rval=10k
.param fc={1/(2*pi*R1*C1)}
.func myfn(x) {x*2}
```

- Component values referencing params must use braces: `R1 in out {Rval}`
- `.param` using other params must use braces: `.param x={y*2}`
- `.func` body uses braces: `.func myfn(x) {x*2}`
- B source expressions: do not wrap the expression itself in curly braces — parameters inside B source expressions do use braces: `B1 out 0 V=V(in)*{Rval}`

## Behavioral Sources (B sources)

Four types:

```spice
B1 out 0 V=<expression>                      ; voltage source
B2 out 0 I=<expression> [Rpar=x] [Cpar=x]    ; current source
B3 out 0 R=<expression>                       ; resistor (undocumented)
B4 out 0 P=<expression> [VprXover=x]          ; power sink (undocumented)
```

**Conditional:** `IF(cond, true, false)`, not ternary `?:` (that's ngspice).
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

**Important behavior:**
- `^` is **XOR** in normal expressions, exponentiation only in Laplace. Use `**` for power.
- `R=<expr>` behavioral resistor: value must never reach zero (causes convergence failure).
- `NoJacob` flag exists but "greatly increases risk of convergence problems" — avoid.

## Monte Carlo

LTspice has no built-in `.mc` directive — use `.step` + `mc()`:

```spice
.step param run 1 100 1
R1 in out {mc(10k, 0.1)}         ; uniform dist, 10k +/-10%
```

`mc(nominal, tolerance)` — uniform between `nom*(1-tol)` and `nom*(1+tol)`.

A `random` variation on `run_experiments` does the same without touching the
deck, and its `rules` list carries four kinds — pick the one that matches what
actually varies in the part you are modelling:

| rule | what it perturbs | typical use |
|-|-|-|
| `component` | one part's value, by tolerance | 1% resistors, 10% capacitors |
| `param` | a declared `.param` | a design variable with a spread |
| `model` | one named parameter of a `.model` card | process corners drawn statistically |
| `mismatch` | per-instance `delvto`/`mulu0` from device area | matched-transistor offset (below) |

The first three take `{target, tolerance, scale: "relative"|"absolute",
distribution: "normal"|"uniform"}`; `model` adds `param`. Full field tables:
`inspect(kind="reference", query="monte carlo")`.

## Per-instance mismatch (flat devices and subckt-wrapped devices)

Two distinct request shapes, both under `run_experiments` `variations`:

**Statistical (Pelgrom) Monte Carlo** — one `random` entry with a mismatch
rule; the engine draws per-instance `delvto`/`mulu0` from device area as
`σ(ΔVTH) = AVT/√(W·L)` with W·L in µm². `AVT` is therefore in **V·µm**
(`3.2e-3` = 3.2 mV·µm) and `AK` in fraction·µm; a coefficient written in
V·m is 10⁶ too small, so the run succeeds but the spread is effectively zero:

```json
{"kind": "random", "id": "mc", "runs": 100,
 "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 3.2e-3, "AK": 0.01}]}
```

To hit a target σ instead of a technology coefficient, invert it:
`AVT = σ · √(W_µm · L_µm)` — 5 mV **per device** on a 20 µm × 1 µm FET is
`2.24e-2`. Each instance is drawn independently, so a differential pair's
input-referred offset σ is √2 × the per-device figure.

**Flat devices on a sheet** — `.model`-based MOSFETs (what an `.asc` with
plain `nmos`/`pmos` symbols exports) take the same rule with `prefix:"M"`,
and the mismatch lands on the model card's `VTO`/`KP`:

```json
{"kind": "random", "id": "mc", "runs": 100,
 "rules": [{"rule": "mismatch", "prefix": "M", "AVT": 2.24e-2, "AK": 0.01}]}
```

To select an exact pair, use one rule per device with `instance`, such as
`instance: ["XA", "M1"]`. A wrapper path selects all its descendant MOS devices.
`prefix` matches leading characters, so `"M1"` also matches M10/M11.
`instance` and `prefix` are mutually exclusive. On BSIM model cards set `vth_param:"VTH0"` and
`k_param:"U0"`; the `VTO`/`KP` defaults are Level-1 names. Other letter
prefixes (`"Q"` for BJTs) are accepted, but the Pelgrom law and those
parameter defaults are for MOSFETs.

A `prefix` that matches subckt instances (e.g. sky130 `X`-wrapped FETs)
descends into supported ngspice-compatible BSIM3/4 wrappers. For nested
blocks, inspect the hierarchy and use the physical MOS instance list. Sampling
uses its final effective W/L after deterministic assignments. Caller-provided
coefficients remain caller assumptions; they are not PDK statistical models.

**Explicit per-instance values** — when the offsets themselves are chosen
(worst-case corners, a specific measured die), use `assign` + `combine:"zip"`
so each row is one case:

```json
{"kind": "assign", "combine": "zip",
 "assign": {"X1:delvto": [0.002, -0.002], "X2:delvto": [-0.002, 0.002]}}
```

`X1:delvto` binds when the subckt body holds a single FET; a multi-FET body
needs the qualified form `X1.M0:delvto`. These instance targets are assign
targets only — they are not valid inside `random` rules, whose mismatch path
is the Pelgrom rule above.


## Convergence

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
solver converges to one root, not necessarily the intended one: a bandgap
can settle at the degenerate 0 V state, a mirror at a spurious high-current
root, with no convergence warning. `.nodeset` alone often fails
to steer it (it's only an initial guess, released before the final solve).
What works: a startup circuit in the deck (as in real silicon); ramping the
supply with `.tran` + `V1 ... PWL(0 0 1m VDD)` and reading the settled state;
or `.dc` sweeping the supply *upward* so each solution seeds the next. Verify
which root you got (e.g. the `operating_point` recipe on a known-current branch) instead
of trusting `status: completed`.

**Hidden defaults (LTspice-specific):**
- `Gfarad` — default parallel conductance on capacitors (1e-12). Disable: `.options Gfarad=0`
- `DampInductors` — default parallel resistance on inductors (ON). Disable: `.options DampInductors=0`
- `Gfloat` — shunt conductance on floating nodes (1e-12 default)
- Inductor coupling factor K may be exactly `1.0` (the docs recommend starting at 1 to remove leakage ringing); use a value just under 1 only if `uic` on `.tran` causes trouble at K=±1

## .options Flags (LTspice-specific)

| Flag | Effect |
|-|-|
| `List` | Dump flattened netlist to error log |
| `DampInductors=0\|1` | Toggle parallel inductor damping |
| `Thev_Induc=0\|1` | Toggle 1mOhm series inductor resistance |
| `Gfarad=<value>` | Capacitor default parallel conductance |
| `Gfloat=<value>` | Floating-node shunt conductance |
| `TopologyCheck=2` | Beta circuit matrix optimizations |
| `baudrate=<rate>` | Enable eye diagram plotting |

## Subcircuits

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


## Other LTspice Quirks

- **Unicode mu**: LTspice writes the `u` suffix as the micro sign (µ) in saved files and exported netlists. LTspice 24 and later write it as UTF-8 (bytes `C2 B5`); LTspice XVII reads a deck as cp1252, sees `Âµ`, and silently drops the scale, so `23µ` runs as 23. Decks `run_experiments` stages and the case decks it writes spell it `u`; a deck you run elsewhere, or hand-write, should use `u`. `verify_circuit` flags a µ suffix (`value_suffix_micro_sign`) and any other non-ASCII character where a suffix goes (`value_suffix_nonascii`), in a netlist and in an `.asc`'s exported netlist.
- **`startup` keyword**: LTspice-only in `.tran`. Ramps sources from zero. Not portable.
- **A-devices** (mixed-signal primitives like `SRflop`, `Counter`, `OTA`): LTspice-proprietary.
- **`*!LTspice: <directive>`**: Treated as a directive, not a comment — despite `*` prefix.
- **Area multipliers**: Undocumented `m=<value>` works on R, Q, J in addition to documented devices.
- Capacitor multiplier: `x<number>` instead of `m=<number>` (e.g., `x2`).
