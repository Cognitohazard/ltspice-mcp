---
name: variations
description: >
  Sweeps, corners and mismatch: what a run_experiments `assign` variation
  can target, and how `combine` pairs targets.
---

# Variations: what `assign` can target

Each key of `run_experiments` `variations[].assign` is a target, each value the
list of values it takes. Explicit forms are recognized first, then bare names:

| Target | Meaning |
|-|-|
| `"M1@model"`, glob `"M*@model"` | swap the instance's model card — the corner idiom, `{"M*@model": ["NTT", "NSS", "NFF"]}` |
| `"X1:delvto"`, `"X1:mulu0"` | per-instance mismatch delta on the FET inside subckt instance X1 (ngspice BSIM3/4, through exactly one X→M level; a multi-FET body needs the qualified `"X1.M0:delvto"`) |
| a declared `.param` name | substitute that parameter |
| a component reference (`R1`, `C2`) | substitute that component's value |

A name that matches neither a declared parameter nor a component is an
`ambiguous_target` error rather than a silent no-op.

`combine` decides how several targets in one entry combine: `"grid"` (default)
takes the cartesian product, `"zip"` runs the i-th value of every target
together as case i and requires equal list lengths. Separate entries are always
cartesian with each other.

```json
{"kind": "assign", "combine": "zip",
 "assign": {"RL": ["1k", "10k"], "VDD": [1.8, 3.3]}}
```

runs two cases (1k/1.8 and 10k/3.3), not four.

Three recipes appear in the tool schema by name only; their arguments are
