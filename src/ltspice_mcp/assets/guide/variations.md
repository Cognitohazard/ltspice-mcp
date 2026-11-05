---
name: variations
description: >
  Sweeps, corners, Monte Carlo and mismatch with run_experiments variations:
  `assign` targets, how `combine` pairs targets, `random` rules, Pelgrom
  mismatch, and explicit per-instance values.
---

# Variations: sweeps, corners, Monte Carlo and mismatch

## Targets of `assign`

Each key of `run_experiments` `variations[].assign` is a target, and its value
is the list of values that the target takes. Explicit forms are recognized
first, then bare names:

| Target | Meaning |
|-|-|
| `"M1@model"`, glob `"M*@model"` | swap the instance's model card — the corner idiom, `{"M*@model": ["NTT", "NSS", "NFF"]}` |
| `"X1:delvto"`, `"X1:mulu0"` | per-instance mismatch delta on the FET inside subckt instance X1 (ngspice BSIM3/4, through exactly one X→M level; a multi-FET body needs the qualified `"X1.M0:delvto"`) |
| a declared `.param` name | substitute that parameter |
| a component reference (`R1`, `C2`) | substitute that component's value |

A name that matches neither a declared parameter nor a component is an
`ambiguous_target` error. A schematic part named `Load` that LTspice exports
as `R§Load` is a target as `Load` or `RLoad` too; `RLoad` names it whichever
simulator runs the deck.

`combine` sets how several targets in one entry combine. `"grid"` (default)
takes the cartesian product, and `"zip"` runs the i-th value of every target
together as case i and requires equal list lengths. Separate entries always
combine as a cartesian product.

```json
{"kind": "assign", "combine": "zip",
 "assign": {"RL": ["1k", "10k"], "VDD": [1.8, 3.3]}}
```

runs two cases (1k/1.8 and 10k/3.3), where `"grid"` would run four.

To hold a value fixed in every case, give it a list of one value in an entry
of its own: `{"kind": "assign", "assign": {"SU": [0], "TLON": ["1n"]}}`
multiplies the grid by one, and every case records the value in its
assignments.

## Monte Carlo: `random` rules

A `random` variation draws values without editing the deck. Its `rules` list
takes four kinds of rule:

| rule | what it perturbs | typical use |
|-|-|-|
| `component` | one part's value, by tolerance | 1% resistors, 10% capacitors |
| `param` | a declared `.param` | a design variable with a spread |
| `model` | one named parameter of a `.model` card | process corners drawn statistically |
| `mismatch` | per-instance `delvto`/`mulu0` from device area | matched-transistor offset (below) |

The first three take `{target, tolerance, scale: "relative"|"absolute",
distribution: "normal"|"uniform"}`; `model` adds `param`.
`inspect(kind="reference", query="monte carlo")` returns the full field tables.
These coefficients are your own assumptions; for a PDK's own statistical
models, see guide section 'sky130'.

Each case records its sample index (`_random_run`) and seed (`_random_seed`);
without a `seed` the server picks one. To repeat sample 7 alone, send the same
entry with that seed, `runs: 1` and `sample_start: 7`, on the circuit under the
same id. A sample draws the same deviation at every `assign` value, so a sweep
compares the same die at each corner, and adding a corner leaves the other
draws as they were. A relative tolerance still scales with an assigned
nominal, and a mismatch draw with an assigned W and L.

The variation's `seed` controls these draws from your assumed coefficients. It
does not affect random functions written in the deck (`agauss`, `gauss`,
`aunif`, `unif`, `limit`); in a recoverable ngspice run,
`execution.simulator_seed` sets those (guide section 'ngspice'). A `pdk_native`
variation has its own `seed` (guide section 'sky130'). `simulator_seed` does
not turn on the PDK's statistical models, and a request that combines it with
a `pdk_native` variation is refused.

## Pelgrom mismatch

One `random` entry with a `mismatch` rule draws a per-instance threshold shift
and current-factor scale from device area as `σ(ΔVTH) = AVT/√(W·L)` with W·L in µm². `AVT` is therefore
in V·µm (`3.2e-3` = 3.2 mV·µm) and `AK` in fraction·µm; a coefficient written in
V·m is 10⁶ too small, so the run succeeds but the spread is effectively zero:

```json
{"kind": "random", "id": "mc", "runs": 100,
 "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 3.2e-3, "AK": 0.01}]}
```

To set a target σ instead of a technology coefficient, invert the formula to
`AVT = σ · √(W_µm · L_µm)`. For 5 mV per device on a 20 µm × 1 µm FET, `AVT` is
`2.24e-2`. Each instance is drawn independently, so a differential pair's
input-referred offset σ is √2 × the per-device figure.

A device the simulator builds as m parallel copies from one card shares that
card's one draw, so its area is `m·W·L`. On ngspice m is the product of the
device's own `m` and the `m` of every subcircuit instance above it. LTspice
ignores `m` on an X line, so there m is the device's own. On both, an inner
`m={mult}` takes the value its X line gives. A device written as 12 fingers of 1 µm
(`W=1u m=12`) gets the σ of a 12 µm device. An `m` that does not resolve to a
positive number refuses the request with `mismatch_geometry_unresolved`.

MOSFETs defined by a `.model` card, which an `.asc` with plain `nmos`/`pmos`
symbols exports, take the same rule with `prefix: "M"`, and the mismatch
applies to the model card's `VTO`/`KP`:

```json
{"kind": "random", "id": "mc", "runs": 100,
 "rules": [{"rule": "mismatch", "prefix": "M", "AVT": 2.24e-2, "AK": 0.01}]}
```

- `prefix` matches leading characters, so `"M1"` also matches M10 and M11. To
  select exact devices, use one rule per device with `instance`, such as
  `instance: ["XA", "M1"]`; a wrapper path selects all its descendant MOS
  devices. `instance` and `prefix` are mutually exclusive.
- On BSIM model cards set `vth_param: "VTH0"` and `k_param: "U0"`; the
  `VTO`/`KP` defaults are Level-1 names.
- Other prefixes (`"Q"` for BJTs) are accepted, but the Pelgrom law and those
  parameter defaults are for MOSFETs.
- When a `prefix` matches subckt instances (such as Sky130's FETs, each wrapped
  in an `X` subcircuit), the rule draws for every MOSFET inside them. On
  ngspice the draw is written as `delvto`/`mulu0` on the device, which needs a
  BSIM3 or BSIM4 model. LTspice takes neither `mulu0` nor `delvto` on BSIM3, so
  there the device gets its own copy of its model card with `vth_param` and
  `k_param` changed, which needs a model with one definition (not binned).
  List the physical MOS instances with `inspect(kind="hierarchy")` (guide
  section 'hierarchy'). Sampling uses each device's final W/L, after any
  `assign` variation has set it.

## Explicit per-instance values

When you choose the offsets yourself (worst-case corners, a measured die), use
`assign` with `combine: "zip"`, so each position across the lists is one case:

```json
{"kind": "assign", "combine": "zip",
 "assign": {"X1:delvto": [0.002, -0.002], "X2:delvto": [-0.002, 0.002]}}
```

`X1:delvto` selects the FET only when the subckt body contains exactly one; a
multi-FET body needs the qualified form `X1.M0:delvto`. These targets work
only in `assign`. In `random` rules, mismatch uses the Pelgrom rule above.
