---
name: signals
description: >
  Naming a trace in a recipe, `V(a,b)` and ratios, trace math in Python with
  time-weighted statistics, and reading a deck that carries `.step`.
---

# Signals, trace math, and `.step`

A recipe's `signal`, `signals` or `expr` names one trace as the raw holds it
— `V(out)`, `I(R1)`, `@m1[gm]` — matched without regard to case. Two spellings
read a quantity the simulator did not write as a trace of its own:

- `V(a,b)` is `V(a) - V(b)`, both read from the same run, step and axis
  (complex on an AC run); `V(a,0)` and `V(a,gnd)` are `V(a)`. The raw must hold
  both node voltages, and ngspice keeps only what `.save` names. A `.noise` run
  holds spectral densities, which do not subtract: name the pair in the
  directive (`.noise V(a,b) ...`) and read `V(onoise)`.
- On the AC recipes (`bode_*`, `stability`, `ac_structure`, `resonance`,
  `return_loss`), `A/B` divides two complex responses and a leading `-` flips
  the sign: `V(out)/V(inp,inn)` is an open-loop gain.

`value` reads the sample nearest `at` on the run's axis, without
interpolation, and needs `at` whenever that axis has more than one sample.

## Trace math in Python

Any other trace math — a sum, a product, a function of a trace — is numpy on
the traces, in Python (guide section 'python'):

```python
r = api.load_raw(job_id=job_id, case_id=case_id)  # or api.load_raw("run.raw")
k = 0  # one step at a time
t = r.axis(step=k)
p_load = r.trace("V(out)", step=k) * r.trace("I(Rload)", step=k)
t_w, p_w, _ = window_and_clean(t, p_load, 1e-3, None)  # from 1 ms to the end
stats = compute_signal_stats(t_w, p_w)  # stats["mean"] is the average power
```

A step's traces share that step's axis. The steps of a `.step` run each have
an axis of their own, so never combine traces across steps.

Take statistics of a derived trace with `compute_signal_stats` and
`time_weighted_quantiles`, not `np.mean`, `np.std` or `np.percentile` over the
samples. LTspice varies its timestep and packs samples around every edge, so a
plain sample average or percentile over-weights the edges.
`compute_signal_stats` weights mean, RMS and standard deviation by time, the
way the `signal_stats` recipe does (its `min`, `max` and `pk_pk` are the sample
extremes). `time_weighted_quantiles(t, y, [0.01, 0.5, 0.99])["values"]` gives
one time-weighted quantile per level, in the order given. `window_and_clean`
cuts the window and drops non-finite samples first. All three are importable
from `ltspice_mcp.api` and in `run_code`'s scope.

## Quantiles in the `signal_stats` recipe

The `signal_stats` recipe keeps `min`, `max` and `peak_to_peak` as the sample
extremes, so a narrow spike or an edge's overshoot stays in them. For a spread
that leaves out the few edges of a switching train, which no single window can
skip, name quantile levels: `"quantiles": [0.01, 0.99]` adds `q01`, `q99` and
`quantile_peak_to_peak` (highest level minus lowest), each a name `field` can
reduce or spec. A key is the level as a percentage with `_` for the decimal
point, so 0.999 is `q99_9`. `q99` is the smallest value the signal spends 99%
of the window at or below.

## Reading a deck that carries `.step`

A `.step` directive puts several sweeps inside one `.raw`, and by default
`analyze_results` reads the first of them. Two call-level arguments change
that, and both apply to every recipe in the call: `step`
(`{"axis": "temp", "value": 27}`) reads the one iteration whose axis value you
name, and `all_steps: true` evaluates every recipe at every iteration. They are
mutually exclusive. `run_experiments`' attached `analyze` block takes the same
two, so an attached measurement and a standalone one read the same steps.
