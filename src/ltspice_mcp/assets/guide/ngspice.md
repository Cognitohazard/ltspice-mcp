---
name: ngspice
description: >
  Writing a deck for ngspice: how it differs from LTspice, parameters,
  behavioral sources, subcircuits, `.save`, `.control`, Monte Carlo,
  options, XSPICE, and the LTspice-vs-ngspice table.
---

# ngspice-Specific

ngspice shares guide section 'fundamentals', with these deltas:

- Inline comment is `$`, not `;`.
- MOSFETs need 4 terminals (`M1 d g s b`) — the bulk is **not** auto-connected
  to the source. (LTspice auto-ties bulk→source only for 3-terminal VDMOS power
  symbols; a generic monolithic NMOS/PMOS needs the 4th node there too.)
- No `startup` keyword on `.tran`. (`.option ramptime` is only a DC source-
  stepping convergence aid in standard builds, not a transient soft-start — the
  true supply-ramp needs an `XSPICE_EXP` build. Ramp a source by hand with a
  PWL/PULSE rise instead.)
- No native `.step`. Run parametric sweeps as `run_experiments` `variations`
  (one netlist per value); ngspice ignores a `.step` line in batch mode, so a
  deck that carries one runs once at the base value with no error. The lint
  warns (`step-ngspice`) rather than letting that pass silently.
- `gnd` is auto-converted to ground (node `0`) by default; disable with
  `set no_auto_gnd` if you need `gnd` to be a distinct net.
- Extra `.meas` types: `MIN_AT`/`MAX_AT` return where the minimum or maximum
  falls (the time or frequency), not its value; `DERIV` is the derivative at a
  point or where a condition is met; `param='expr'` evaluates an expression
  over `.param` values and earlier `.meas` results; `par('expr')` is an inline
  expression on output variables. `.meas ... FIND` takes `V(out)` (no `mag()`
  wrapper). Inside `.control`, `param` and `par` are not available: compute
  with `let`. The interactive `meas` command also takes an `SP` analysis, for
  measurements on a spectrum.
- `.meas` is suppressed only when batch mode (`-b`) and a command-line `-r
  rawfile` are combined — ngspice prints "No .measure possible in batch mode
  (-b) with -r rawfile set!" (the invocation this server uses). It is not a
  blanket batch limitation: move the measurement into a `.control ... run ...
  .endc` block and write it as the dot-less interactive `meas` command (e.g.
  `meas tran vmax MAX V(out)` — no leading dot; a dotted `.meas` inside
  `.control` is not a valid ngspice command and computes nothing). The result
  prints to the run's log. (`set measoutfile` / `.option measoutfile` does not
  help here — the `-b -r` combination suppresses the measurement before any
  output routing, so no file is written.) For named signals and device
  operating-point params you usually need none of this — `.save` them and read
  the raw back with `run_experiments` plus the `waveform`, `value`, or
  `operating_point` recipe. Reserve `.control` / `wrdata` for in-engine computation you
  cannot express as a saved signal.

## Parameters and Expressions

```spice
.param Rval=10k
.param fc={1/(2*pi*Rval*Cval)}
.param combined='Rval + 10'
.func myfn(x) {x*2}
```

- Expressions in braces `{expr}` or single quotes `'expr'` — both work.
- Expressions without delimiters work only when spaces are absent:
  `.param c=a+123` OK, `.param c = a + 123` fails silently (assigns first token).
- Self-referential params fail silently: `.param x = {x+3}` does not work.
- Parameter names must start with alpha; may contain `! # $ % [ ] _`. Cannot use
  reserved words: `time`, `temper`, `hertz`, `not`, `and`, `or`, `div`, `mod`,
  `sqr`, `sqrt`, `sin`, `cos`, `exp`, `ln`, `log`, `log10`, `arctan`, `abs`,
  `pwr`, `defined`.
- String-valued params are supported, with limited concatenation.

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

## Behavioral Sources (B sources)

```spice
B1 out 0 V=<expression>
B2 out 0 I=<expression> [tc1=x] [tc2=x] [temp=x]
```

**Conditional:** ternary `cond ? true : false`, not `IF()` (that is LTspice).
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

**Important behavior:**
- `exp()` is internally capped at argument=14 — beyond that it becomes linear.
- `log`/`ln`/`sqrt` of negatives use `fabs()` automatically — no error.
- Division by zero or `log(0)` causes an error.
- `par('expression')` works in `.plot`/`.print` output lines too.
- Non-linear R, L and C can be built from B sources; the ngspice manual gives
  the subcircuit templates.

## Subcircuits

```spice
.subckt myfilter in out rval=100k cval=100nF
R1 in p1 {2*rval}
C1 p1 0 {cval}
.ends myfilter

X1 input output myfilter rval=1k cval=1n
```

- Parameters on the `.subckt` line do not need a `params:` keyword — just
  `name=value` after the nodes.
- `.lib` loading depends on ngspice's compatibility mode, and no single `.lib`
  form works in every mode — so for unconditional whole-file inclusion use
  `.include <file>`, which resolves in every mode (verified on ngspice-42). If
  you use `.lib`:
  - **ngspice-native modes** (`hsa`, plain default): `.lib <file> <section>`
    pulls just that `.lib section … .endl` block; a bare `.lib <file>` with no
    section does not load the file's models.
  - **This server's default `kiltpsa`** (a mixed LTspice/PSPICE-compatibility
    mode) inverts this: a bare `.lib <file>` loads an unsectioned file, but a
    sectioned `.lib <file> <section>` (the PDK corner idiom) is mis-split by the
    `lt`/`ps` tokens into two plain includes that drop the section — the run
    fails with "could not find include file". Set `[simulator] ngbehavior =
    "hsa"` in `ltspice-mcp.toml` (or `LTSPICE_MCP_NGBEHAVIOR=hsa`) and restart to
    parse the section correctly.
  `.lib` also differs from `.include` in scope — it skips global-scope circuit
  elements.
- `.param` inside subcircuits is local scope (masks globals). Nesting to 10 levels.
- Subcircuit and model names are global — must be unique across the netlist.

## .save Directive

```spice
.save V(out) I(Vin)               $ save only these signals
.save @m1[id] @m1[gm]             $ save device operating-point params
.save all @m2[vdsat]              $ save defaults plus extras
```

- Without `.save`, all node voltages and source currents are saved (huge files).
- Adding even one `.save` line drops all defaults — only listed signals saved.
- Resistor current is the internal vector `@r1[i]` (via `.save @r1[i]` or
  `.options savecurrents`); under this path the `i(r1)` read-function does not
  resolve it — `i()`/`I()` only resolve `name#branch` vectors (voltage sources,
  and the sense source a separate `.probe I(R1)` directive inserts). This server
  reads the `@r1[i]` form.
- Saved device operating-point params (`@m1[gm]`, `v(@m1[vth])`, `i(@m1[id])`, …) are surfaced
  by the `operating_point` recipe in a `device_op_points` bucket (a bare `.op`),
  and on a `.dc`/`.tran` sweep are readable by the `dev.param` shorthand — the
  `value`, `signal_stats`, and `waveform` recipes accept `m1.gm` and resolve it
  to the actual trace. This is the gm/ID idiom: `.dc Vg …` + `.save @m1[gm] @m1[id]`, then read
  `m1.gm`/`m1.id` per sweep point.

## .control / .endc Blocks

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

**Scripts without `write`/`wrdata`.** A `.control` block replaces ngspice's
default raw output, so a script that never calls `write`/`wrdata` produces no
rawfile even though the run completes cleanly. `run_experiments` fills that
gap: when the deck has exactly one `.control` block and no `write`/`wrdata`
of its own, it adds a `write <the run's raw path>` before `.endc` and says so
in the receipt's `observations`. Your own `write` (or `wrdata`) anywhere in
the deck turns the injection off — and you need one, per result you want
kept, whenever the script runs several analyses or writes per iteration in a
Monte Carlo loop: `write` captures the current plot only.

**Variables vs vectors — a critical distinction:**
- `set` creates string/shell variables: `set myvar = "hello"` — access `$myvar`.
- `let` creates numeric vectors: `let x = 2*pi` — access `$&x` to get a number.
- Mixing up `set` and `let` fails silently.
- `$&param` dereferences a circuit `.param` into a control variable.

**Control structures:** `while`/`end`, `repeat`/`end`, `foreach`/`end`,
`if`/`else`/`end`, `dowhile`, `break [n]`, `continue [n]`, `label`, `goto`.
`foreach` values are space-separated (no commas); `foreach var $myvariable`
expands a variable into the list.

**Key commands:** `run`, `plot`, `print`, `let`, `set`, `write`, `wrdata`,
`alter`, `altermod`, `echo`, `meas`, `linearize`, `fft`, `define`, `source`.

## Monte Carlo

ngspice has **no `.mc` directive**. Two idioms:

**(1) Per-device statistical functions (primary, simplest).** Put `agauss`/
`gauss`/`unif`/`aunif`/`limit` directly in a `.param` or a device/B-source value,
in `'…'` or `{…}`. `gauss(nom, rvar, sigma)` and `unif(nom, rvar)` take a
relative variation, `agauss(nom, avar, sigma)` and `aunif(nom, avar)` an
absolute one, and `limit(nom, avar)` gives `nom+avar` or `nom-avar`. Each
device card draws a fresh value at parse time:

```spice
R1 a b 'agauss(10k, 500, 3)'      $ 10k, ±500 absolute, /3 sigma
C1 c 0 '{unif(1n, 0.1)}'          $ 1n, ±10% relative, uniform
```

These are built into the numparam frontend (no build flag) but live only there,
not in the nutmeg/`.control` interpreter. For a distribution, re-run the deck N
times (set `.options seed=<value>`); a `run_experiments` random `variations`
entry automates the N-run draw + aggregation.

**(2) `.control` loop with `alter`** — vary within one ngspice invocation. Inside
`.control` only `sgauss(0)` (Gaussian, mean 0, σ 1) and `sunif(0)` (uniform
[-1,1]) are built in — scale them yourself (`agauss`/`gauss` are not nutmeg
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

## .options Flags

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

## XSPICE

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

## Key Differences: LTspice vs ngspice

| Aspect | LTspice | ngspice |
|-|-|-|
| Inline comment | `;` | `$` |
| B-source conditional | `IF(c,a,b)` | ternary `c ? a : b` |
| `^` operator | XOR (power is `**`) | power |
| MOSFET bulk | auto-tied to source only on 3-term VDMOS symbols | required 4th terminal |
| `GND` node | alias for `0` | auto-converted to `0` (disable: `set no_auto_gnd`) |
| `.tran startup` | supported | not supported (no transient soft-start) |
| Parameter sweep | `.step` | `run_experiments` `variations` (no `.step`) |
| Monte Carlo | `.step` + `mc()` | `agauss`/`gauss`/`unif` on device values (primary); or `.control` `alter` loop |
| Post-processing | — | `.control` scripting (`let`/`plot`/`write`/`fft`) |
| Default saving | saves all | `.save` (one line drops defaults; `.save all` keeps) |
| `.raw` format | mixed precision | all doubles |
| Unicode mu | replaces `u` with µ | preserves `u` |

Other ngspice notes: A-devices are the XSPICE code-model primitives (the `A`
prefix — available in stock builds, see XSPICE); `.func` cannot be recursive
(textual expansion, so a self-reference expands without bound). `.backanno` is
an LTspice-only directive — ngspice rejects it ("unimplemented dot command
'.backanno'") and aborts the run; ngspice current probing uses `.options
savecurrents` / `.probe` instead.
