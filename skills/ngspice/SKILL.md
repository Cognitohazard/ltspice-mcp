---
name: ngspice
description: >
  Use when writing or editing ngspice circuit netlists (.cir, .sp, .spice),
  working with ngspice scripting (.control blocks), or interpreting simulation
  results. Covers ngspice-specific SPICE syntax, behavioral sources, .control
  scripting, Monte Carlo via control loops, parameters, XSPICE, .save, .MEAS,
  convergence, and common gotchas that cause silent errors. Use this skill
  whenever the user mentions ngspice, or is writing SPICE netlists targeting
  ngspice rather than LTspice.
---

# ngspice Circuit Simulation Guide

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
- Comments: `*` (full line) or `$` (inline). Note: `;` is NOT the inline comment character in ngspice (that's LTspice).

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
```

### Directives

```spice
.tran 5m                          $ transient, 5ms stop
.tran 0 5m 0 10u                  $ tstep, tstop, tstart, tmaxstep
.ac dec 200 10 100k               $ AC sweep, 200pts/decade, 10Hz-100kHz
.dc V1 0 5 0.01                   $ DC sweep V1, 0-5V, 10mV step
.op                               $ DC operating point
.noise V(out) V1 dec 200 10 100k  $ noise analysis
.tf V(out) V1                     $ DC transfer function
.include /path/to/model.lib       $ include library
.ic V(node)=1.5                   $ initial conditions (used with UIC)
.nodeset V(node)=1.5              $ hint for DC operating point solver
```

`.ic` forces node voltages at t=0 (use with `.tran ... UIC`). `.nodeset` is a solver hint only.

### .MEAS Syntax

```spice
.meas TRAN vmax MAX V(out)
.meas TRAN vpp PP V(out)
.meas TRAN trise TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1
.meas AC fc WHEN mag(V(out)/V(in))=0.707
.meas AC gain_1k FIND V(out) AT=1k
.meas TRAN avg_out AVG V(out) FROM=1m TO=5m
.meas TRAN energy INTEG V(out)*I(R1)
.meas TRAN slope DERIV V(out) AT=2m
.meas TRAN check param='tdiff < vout_diff ? 1 : 0'
```

**Additional measurement types vs LTspice:**
- `MIN_AT`, `MAX_AT` — return the time/freq of the min/max, not the value.
- `DERIV` — derivative at a point or when a condition is met.
- `param='expression'` — evaluate an expression using `.param` values and prior `.meas` results.
- `par('expression')` — inline algebraic expression on any output variable (uses B source syntax internally).
- `SP` analysis type for spectrum (fft) measurements (via `meas` command, not `.meas` line).

**Gotchas:**
- RISE/FALL/CROSS numbering starts at **1**, not 0.
- If TRIG event never occurs, measurement silently fails.
- `.meas` does NOT work in batch mode (`-b`) when combined with `-r rawfile` — data is streamed to disk and not available for analysis. Use interactive mode or a `.control` block instead.
- `param` and `par` are not available inside `.control` blocks — use `let` instead.

### General Pitfalls

- **Node "0" is ground**. Using `GND` without `.global GND` or tying it to 0 creates a floating node — no error, wrong results.
- **MOSFET requires 4 terminals**: `M1 d g s b` — ngspice does NOT auto-connect bulk to source (LTspice does).
- **Impedance ratios**: Beyond ~1e16 cause numerical issues (64-bit doubles).
- **Parameter sweep**: ngspice has **no native `.step`** (that is LTspice syntax). The MCP runs parametric sweeps through `configure_sweep` + `run_sweep`, which generate and simulate one netlist per value. For a hand-written deck outside the MCP, use a `.control` block with an `alter`/loop. A `.step` line in a deck handed to `run_simulation` is rejected with a pointer to `configure_sweep`.

---

## ngspice-Specific

### Parameters and Expressions

```spice
.param Rval=10k
.param fc={1/(2*pi*Rval*Cval)}
.param combined='Rval + 10'
.func myfn(x) {x*2}
```

- Expressions in braces `{expr}` or single quotes `'expr'` — both work.
- Expressions without delimiters work only when spaces are absent: `.param c=a+123` OK, `.param c = a + 123` FAILS silently (assigns only first token).
- Self-referential params fail silently: `.param x = {x+3}` does not work.
- Parameter names: must start with alpha; may contain `! # $ % [ ] _`. Cannot use reserved words: `time`, `temper`, `hertz`, `not`, `and`, `or`, `div`, `mod`, `sqr`, `sqrt`, `sin`, `cos`, `exp`, `ln`, `log`, `log10`, `arctan`, `abs`, `pwr`, `defined`.
- String-valued params supported with limited concatenation.

**Three separate expression parsers exist in ngspice** — this is a known source of confusion:
1. **Front-end** (`.param`, brace expressions) — evaluated at netlist expansion time
2. **B source / behavioral** — evaluated during simulation (no braces)
3. **`.control` block** — operates on its own vectors/variables

These have slightly different function sets and precedence rules. Braces `{...}` are "compile-time"; bare expressions in B sources are "run-time".

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
- Default (`hs` compat): `x^y` = `pow(fabs(x), y)` for x>0; rounds y for x<0; 0 for x=0
- LTspice compat (`lt`): `x^y` = `pow(x, y)` if y is close to integer; else 0 for x<0

**Built-in functions (.param):**
- Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `arctan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Exp/log: `exp`, `ln`, `log` (base e), `log10`
- Power: `sqrt`, `pow(x,y)`, `pwr(x,y)` (= `pow(fabs(x),y)`)
- Rounding: `nint` (nearest, half to even), `int` (toward 0), `floor`, `ceil`
- Selection: `min`, `max`, `sgn`
- Conditional: `ternary_fcn(x,y,z)` (= `x ? y : z`)
- **Statistical:** `gauss(nom,rvar,sigma)`, `agauss(nom,avar,sigma)`, `unif(nom,rvar)`, `aunif(nom,avar)`, `limit(nom,avar)`
- Special: `var(name)` (interpreter variable), `vec(name)` (vector value)

### Behavioral Sources (B sources)

```spice
B1 out 0 V=<expression>
B2 out 0 I=<expression> [tc1=x] [tc2=x] [temp=x]
```

**Conditional:** uses ternary `cond ? true : false` — NOT `IF()` (that's LTspice). Put a space before `?` so the parser doesn't confuse it with other tokens. Nested ternaries need explicit parentheses.

**Available functions (B source context):** `cos`, `sin`, `tan`, `acos`, `asin`, `atan`, `cosh`, `sinh`, `acosh`, `asinh`, `atanh`, `exp`, `ln`, `log`, `log10`, `abs`, `sqrt`, `u` (unit step), `u2` (ramp 0-1), `uramp`, `floor`, `ceil`, `min`, `max`, `pow`, `**`, `pwr`, `^`, `i(device)`

**Special variables:** `time` (transient), `temper` (circuit temp in C), `hertz` (AC frequency). `time` is zero during AC; `hertz` is zero during transient.

**Piecewise linear in B source:**
```spice
Bdio 1 0 I = pwl(v(A), 0,0, 33,10m, 100,33m, 200,50m)
Blimit b 0 V = pwl(v(1), -4,0, -2,2, 2,4, 4,5, 6,5)
```
x values must be monotonically increasing — non-monotonic stops execution. Can use `time` or expressions as the independent variable.

**Gotchas:**
- `exp()` is internally capped at argument=14 — beyond that it becomes linear (for convergence).
- `log`/`ln`/`sqrt` of negative values use `fabs()` automatically — no error, may give unexpected results.
- Division by zero or `log(0)` causes an error.
- B source `par('expression')` can be used in `.plot`/`.print` output lines.
- Non-linear R/L/C can be synthesized from B sources using the subcircuit template pattern (see ngspice manual 5.1).

### Subcircuits

```spice
.subckt myfilter in out rval=100k cval=100nF
R1 in p1 {2*rval}
C1 p1 0 {cval}
.ends myfilter

X1 input output myfilter rval=1k cval=1n
```

**Key differences from LTspice:**
- Parameters on `.subckt` line do NOT need `params:` keyword — just `name=value` after nodes.
- `.lib <filename> <section>` — requires a section name. **Omitting the section silently loads nothing** (no error!). For unconditional inclusion, use `.include` instead.
  - **PDK corners under this MCP:** ngspice runs here in spicelib's default compatibility mode (`ngbehavior='kiltpsa'`), whose `lt`/`ps` tokens split a sectioned `.lib <file> <section>` into two plain includes — dropping the corner section, so it surfaces as a missing include (`could not find include file`). For standard-SPICE / PDK decks, set `[simulator] ngbehavior = "hsa"` in `ltspice-mcp.toml` (or `LTSPICE_MCP_NGBEHAVIOR=hsa`) and restart the server; `run_simulation` emits this hint when a failed run matches the pattern.
- `.param` inside subcircuits is local scope (masks globals). Nesting up to 10 levels.
- Subcircuit and model names are global — must be unique across the entire netlist.

### .save Directive

```spice
.save V(out) I(Vin)               $ save only these signals
.save @m1[id] @m1[gm]             $ save internal device parameters
.save all @m2[vdsat]               $ save defaults PLUS extras
```

- Without `.save`, all node voltages and source currents are saved (can create huge files).
- Adding even ONE `.save` line drops all defaults — only listed signals are saved.
- To keep defaults plus extras: `.save all @m2[vdsat]`
- `.save @r1[i]` for resistor current (not available via `I()` syntax).
- **Read internals back as named numbers** (no rawfile parsing, no `.control`):
  on a `.dc`/`.tran` sweep, `export_waveform(signals=['m1.gm','m1.id'])` gives the
  gm/ID table in one CSV; `query_value(signal='m1.gm', at=...)` reads one point;
  `operating_point(device='M1')` gives the bias snapshot of one device. Address
  an internal by the `m1.gm` shorthand or the literal `@m1[gm]` (the tools resolve
  the `v()`/`i()` wrapping and subcircuit paths). This `.dc` + `.save` + read flow
  is the gm/ID-characterization idiom — see the `spice://guide` resource.

### .control / .endc Blocks

ngspice has a built-in scripting language for post-simulation analysis:

```spice
.control
run                               $ execute the simulation
plot V(out)                       $ plot results
print V(out)                      $ print to stdout
let vmax = maximum(V(out))        $ create vector
set filename = "results.csv"      $ create string variable
write $filename V(out) I(Vin)     $ save to rawfile
wrdata output.txt V(out)          $ save as CSV-like text
.endc
```

**Variables vs vectors — a critical distinction:**
- `set` creates string/shell variables: `set myvar = "hello"` — access with `$myvar`
- `let` creates numeric vectors: `let x = 2*pi` — access with `$&x` to convert to number
- Mixing them up causes silent failures.
- `$&param` dereferences a circuit `.param` into a control variable.

**Control structures:**

```spice
$ While loop
while condition
  ...
end

$ Repeat
repeat 10
  ...
end

$ Foreach
foreach val 1 2 3 4 5
  echo value is $val
end

$ If/else
if condition
  ...
else
  ...
end

$ Also: dowhile, break [n], continue [n], label, goto
```

`foreach` values are space-separated (no commas). Can use variable expansion: `foreach var $myvariable`.

**Key commands:** `run`, `plot`, `print`, `let`, `set`, `write`, `wrdata`, `alter`, `altermod`, `echo`, `meas`, `linearize`, `fft`, `define`, `source`

### Monte Carlo

ngspice has **no `.mc` directive**. Monte Carlo is done via `.control` block loops:

```spice
.control
define unif(nom, var) (nom + nom*var * sunif(0))
define gauss(nom, var, sig) (nom + nom*var/sig * sgauss(0))

let mc_runs = 100
let run = 1
dowhile run <= mc_runs
  alter c1 = unif(1e-09, 0.1)
  alter r1 = gauss(10k, 0.05, 3)
  tran 1u 1m
  $ ... store/process results ...
  let run = run + 1
end
.endc
```

**Random functions:**
- `sunif(0)` — single uniform random number in [-1, 1]
- `sgauss(0)` — single Gaussian random number (mean=0, stddev=1)
- `.options seed=<value>` or `seed=random` — set random seed

**Statistical functions (.param context):**
- `gauss(nom, rvar, sigma)` — Gaussian, relative variation
- `agauss(nom, avar, sigma)` — Gaussian, absolute variation
- `unif(nom, rvar)` — uniform, relative
- `aunif(nom, avar)` — uniform, absolute
- `limit(nom, avar)` — binary: nom+avar or nom-avar

Using `.param` statistical functions requires multiple ngspice runs (each run re-evaluates). Using `.control` loops with `alter` varies within a single run.

### .options Flags

**General:**

| Flag | Effect |
|-|-|
| `SEED=val\|random` | Random number seed |
| `TEMP=x` | Operating temperature (default 27C) |
| `TNOM=x` | Nominal temperature (default 27C) |
| `SAVECURRENTS` | Auto-save all device terminal currents |
| `KLU` | Use KLU matrix solver (faster for large MOS circuits) |
| `INTERP` | Interpolate output to fixed TSTEP grid |

**Convergence:**

| Flag | Default | Effect |
|-|-|-|
| `RELTOL` | 0.001 | Relative tolerance |
| `ABSTOL` | 1e-12 | Absolute current tolerance |
| `VNTOL` | 1e-6 | Absolute voltage tolerance |
| `GMIN` | 1e-12 | Minimum junction conductance |
| `ITL1` | 100 | DC iteration limit |
| `ITL4` | 10 | Transient iteration limit |
| `METHOD` | trap | Integration method: `trap` or `gear` |
| `MAXORD` | 2 | Max integration order for Gear (2-6) |
| `TRTOL` | 7 | Transient error tolerance factor |
| `XMU` | 0.5 | Trapezoidal damping (reduce slightly to suppress ringing) |

**Matrix conditioning:**

```spice
.options rshunt=1e12              $ resistor from every node to ground
.options rseries=1e-4             $ series resistor on every inductor
.options cshunt=1e-13             $ capacitor from every node to ground
```

Use `rshunt` for "no DC path to ground" errors. Use `rseries` when inductors parallel voltage sources cause OP failure. Use `cshunt` for oscillation/noise issues.

`AUTOSTOP` — stops transient analysis after all `.meas` conditions are satisfied.

### XSPICE

Mixed-signal simulation with code models (requires XSPICE-enabled build):

```spice
A1 [in] [out] lut1
.model lut1 d_lut(rise_delay=1n fall_delay=2n input_load=0.5p
+ table_values="0110")
```

Digital device types: `d_and`, `d_or`, `d_nand`, `d_nor`, `d_xor`, `d_inverter`, `d_buffer`, `d_flop`, `d_latch`, `d_lut`, etc.

Digital nodes use `[name]` bracket syntax for buses.

### Key Differences from LTspice

- **Inline comments**: `$` not `;`
- **Conditional in B source**: ternary `?:` not `IF()`
- **MOSFET bulk terminal**: required (4 pins), not auto-connected
- **GND node**: must explicitly tie to 0 or use `.global`
- No `startup` keyword in `.tran`
- No A-devices (mixed-signal primitives) — use XSPICE instead
- No Unicode mu issue — ngspice preserves `u` as-is
- `.func` definitions cannot be recursive — causes hang, not error
- Different `.raw` file format (all doubles vs LTspice mixed precision)
- `.backanno` needed for current probing in post-processing
