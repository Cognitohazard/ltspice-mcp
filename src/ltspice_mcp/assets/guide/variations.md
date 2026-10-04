---
name: variations
description: >
  Sweeps, corners, Monte Carlo and mismatch with run_experiments variations:
  what an `assign` can target, how `combine` pairs targets, `random` rules,
  Pelgrom mismatch, and explicit per-instance values.
---

# Variations: sweeps, corners, Monte Carlo and mismatch

## What `assign` can target

Each key of `run_experiments` `variations[].assign` is a target, each value the
list of values it takes. Explicit forms are recognized first, then bare names:

| Target | Meaning |
|-|-|
| `"M1@model"`, glob `"M*@model"` | swap the instance's model card — the corner idiom, `{"M*@model": ["NTT", "NSS", "NFF"]}` |
| `"X1:delvto"`, `"X1:mulu0"` | per-instance mismatch delta on the FET inside subckt instance X1 (ngspice BSIM3/4, through exactly one X→M level; a multi-FET body needs the qualified `"X1.M0:delvto"`) |
| a declared `.param` name | substitute that parameter |
| a component reference (`R1`, `C2`) | substitute that component's value |

A name that matches neither a declared parameter nor a component is an
`ambiguous_target` error.

`combine` decides how several targets in one entry combine: `"grid"` (default)
takes the cartesian product, `"zip"` runs the i-th value of every target
together as case i and requires equal list lengths. Separate entries are always
cartesian with each other.

```json
{"kind": "assign", "combine": "zip",
 "assign": {"RL": ["1k", "10k"], "VDD": [1.8, 3.3]}}
```

runs two cases (1k/1.8 and 10k/3.3), not four.

## Monte Carlo: `random` rules

A `random` variation draws values without touching the deck. Its `rules` list
carries four kinds:

| rule | what it perturbs | typical use |
|-|-|-|
| `component` | one part's value, by tolerance | 1% resistors, 10% capacitors |
| `param` | a declared `.param` | a design variable with a spread |
| `model` | one named parameter of a `.model` card | process corners drawn statistically |
| `mismatch` | per-instance `delvto`/`mulu0` from device area | matched-transistor offset (below) |

The first three take `{target, tolerance, scale: "relative"|"absolute",
distribution: "normal"|"uniform"}`; `model` adds `param`. Full field tables:
`inspect(kind="reference", query="monte carlo")`. These coefficients are your
own assumptions; for a PDK's own statistical models, see guide section
'sky130'.

## Pelgrom mismatch

One `random` entry with a `mismatch` rule draws per-instance `delvto`/`mulu0`
from device area as `σ(ΔVTH) = AVT/√(W·L)` with W·L in µm². `AVT` is therefore
in V·µm (`3.2e-3` = 3.2 mV·µm) and `AK` in fraction·µm; a coefficient written in
V·m is 10⁶ too small, so the run succeeds but the spread is effectively zero:

```json
{"kind": "random", "id": "mc", "runs": 100,
 "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 3.2e-3, "AK": 0.01}]}
```

To hit a target σ instead of a technology coefficient, invert it:
`AVT = σ · √(W_µm · L_µm)` — 5 mV per device on a 20 µm × 1 µm FET is
`2.24e-2`. Each instance is drawn independently, so a differential pair's
input-referred offset σ is √2 × the per-device figure.

`.model`-based MOSFETs (what an `.asc` with plain `nmos`/`pmos` symbols
exports) take the same rule with `prefix: "M"`, and the mismatch lands on the
model card's `VTO`/`KP`:

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
- A `prefix` that matches subckt instances (such as Sky130's `X`-wrapped FETs)
  descends into ngspice-compatible BSIM3/4 wrappers. For nested blocks, list
  the physical MOS instances with `inspect(kind="hierarchy")` (guide section
  'hierarchy'). Sampling uses each device's final W/L, after deterministic
  assignments.

## Explicit per-instance values

When you choose the offsets yourself (worst-case corners, a measured die), use
`assign` with `combine: "zip"`, so each position across the lists is one case:

```json
{"kind": "assign", "combine": "zip",
 "assign": {"X1:delvto": [0.002, -0.002], "X2:delvto": [-0.002, 0.002]}}
```

`X1:delvto` binds when the subckt body holds a single FET; a multi-FET body
needs the qualified form `X1.M0:delvto`. These targets work only in `assign`,
not inside `random` rules, whose mismatch path is the Pelgrom rule above.
