---
name: ltspice
description: >
  Writing a deck for LTspice: parameters, `.step` sweeps, PWL extras,
  behavioral sources, Monte Carlo with `mc()`, convergence, options,
  subcircuits, and quirks such as the micro sign.
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
- B source expressions: do not wrap the expression itself in braces, but do
  brace the parameters inside it: `B1 out 0 V=V(in)*{Rval}`

## Parameter sweeps (`.step`)

```spice
.step param RL 1k 10k 1k          ; start, stop, increment
.step param RL list 1k 2.2k 4.7k  ; explicit values
.step param STM list 3904 2222    ; model stepping, with Q1 ... {STM}
```

A stepped run holds every step in one raw; reading one step or all of them is
in guide section 'signals'. With `run_experiments`, a variation sweeps without
editing the deck (guide section 'variations').

## PWL extras

- Relative time: `PWL(0 1 +1 2 +1 3)` — the times become 0, 1, 2.
- Repetition: `REPEAT FOR n (...) ENDREPEAT` or
  `REPEAT FOREVER (...) ENDREPEAT`.
- Scaling: `VALUE_SCALE_FACTOR=x`, `TIME_SCALE_FACTOR=x`.
- Trigger: `TRIGGER <expression>` — the output holds its first value while the
  expression is false.

## Behavioral Sources (B sources)

Four types:

```spice
B1 out 0 V=<expression>                      ; voltage source
B2 out 0 I=<expression> [Rpar=x] [Cpar=x]    ; current source
B3 out 0 R=<expression>                       ; resistor (undocumented)
B4 out 0 P=<expression> [VprXover=x]          ; power sink (undocumented)
```

**Conditional:** `IF(cond, true, false)`, not the ternary `?:` (that is
ngspice). B source expressions must be one line in a schematic; a netlist can
continue them with `+`.

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

**Special variables:** `time`, `pi`, `boltz` (1.38e-23), `planck`
(6.63e-34), `echarge` (1.60e-19), `kelvin` (-273.15), `Gmin` (1e-12)

**Laplace filter:**
```spice
B1 out 0 V=V(in) Laplace=1/(1+s/{2*pi*fc})
```
In Laplace expressions, `^` means exponentiation (not XOR). The response must
roll off at high frequencies.

**Important behavior:**
- `^` is XOR in normal expressions and exponentiation only in Laplace. Use
  `**` for power.
- A behavioral resistor `R=<expr>` must never reach zero; it fails to
  converge.
- Avoid the `NoJacob` flag: LTspice's own help says it greatly increases the
  risk of convergence problems.

## Monte Carlo

LTspice has no built-in `.mc` directive — use `.step` + `mc()`:

```spice
.step param run 1 100 1
R1 in out {mc(10k, 0.1)}         ; uniform dist, 10k +/-10%
```

`mc(nominal, tolerance)` — uniform between `nom*(1-tol)` and `nom*(1+tol)`.

`run_experiments` draws the same without editing the deck: a `random`
variation, including per-instance mismatch (guide section 'variations').

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
- Be suspicious of circuits needing `cshunt` — may indicate unrealistic models.

**Bistable/multi-root circuits (bandgaps, current mirrors, latches):** the DC
solver converges to one root, not necessarily the intended one: a bandgap can
settle at the degenerate 0 V state, a mirror at a spurious high-current root,
with no convergence warning. `.nodeset` alone often fails to steer it: it is
only an initial guess, released before the final solve. What works: a startup
circuit in the deck, as in real silicon; ramping the supply in a `.tran` with
`V1 ... PWL(0 0 1m VDD)` and reading the settled state; or a `.dc` sweep of the
supply upward, so each solution seeds the next. Verify which root you got
(e.g. the `operating_point` recipe on a known-current branch) instead of
trusting `status: completed`.

**Hidden defaults (LTspice-specific):**
- `Gfarad` — parallel conductance on every capacitor (1e-12). Disable with
  `.options Gfarad=0`.
- `DampInductors` — parallel resistance on inductors (on). Disable with
  `.options DampInductors=0`.
- `Gfloat` — shunt conductance on floating nodes (1e-12).
- An inductor coupling factor K may be exactly `1.0` (the docs suggest starting
  there to remove leakage ringing); use a value just under 1 only if `uic` on
  `.tran` has trouble at K=±1.

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


## Other LTspice Quirks

- **Micro sign**: LTspice writes the `u` suffix as µ in saved files and
  exported netlists. LTspice 24 writes it in UTF-8 and reads it back; LTspice
  XVII reads the file as cp1252 and drops the scale, so `23µ` runs as 23.
  `run_experiments` stages decks with `u`; write `u` in any deck you run
  elsewhere. `verify_circuit` flags a µ suffix (`value_suffix_micro_sign`) and
  any other non-ASCII suffix (`value_suffix_nonascii`), in a netlist and in an
  `.asc`'s exported netlist.
- **`startup`** on `.tran` (`.tran 0 5m 0 10u startup`) ramps the sources up
  from zero. ngspice has no equivalent keyword.
- **A-devices** (mixed-signal primitives such as `SRflop`, `Counter`, `OTA`)
  are LTspice's own.
- **`*!LTspice: <directive>`** is read as a directive, not a comment, despite
  the `*`.
- **Area multipliers**: an undocumented `m=<value>` works on R, Q and J as well
  as the documented devices.
- Capacitor multiplier: `x<number>` instead of `m=<number>` (e.g. `x2`).
