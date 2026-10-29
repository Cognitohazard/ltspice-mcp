---
name: signals
description: >
  Naming a trace in a recipe, `V(a,b)` and ratios, trace math in Python with
  time-weighted statistics, the amplitude and phase of one frequency (the
  `tone` recipe, and LTspice's `.four`), and reading a deck with `.step`.
---

# Signals, trace math, and `.step`

A recipe's `signal`, `signals` or `expr` names one trace as the raw file stores
it (`V(out)`, `I(R1)`, `@m1[gm]`), matched without regard to case. Two
spellings name a quantity that the simulator did not write as a trace:

- `V(a,b)` is `V(a) - V(b)`, both read from the same run, step and axis
  (complex on an AC run); `V(a,0)` and `V(a,gnd)` are `V(a)`. The raw file must
  contain both node voltages, and ngspice saves only the signals that `.save`
  lists. A `.noise` run stores spectral densities, which cannot be subtracted,
  so name the pair in the directive (`.noise V(a,b) ...`) and read `V(onoise)`.
- On the AC recipes (`bode_*`, `stability`, `ac_structure`, `resonance`,
  `return_loss`), `A/B` divides two complex responses and a leading `-` flips
  the sign. For example, `V(out)/V(inp,inn)` is an open-loop gain.

`value` reads the sample nearest `at` on the run's axis, without
interpolation, and needs `at` whenever that axis has more than one sample. A
stepped `.ac` or `.noise` at one frequency stores one point per step and no
frequency, and a stepped `.op` or `.tf` stores one point per step and no axis,
so leave `at` out and pick the step. The stepped values are in each row's
`step_values`, not among the signals. For a stepped `.ac` read from its job,
`value` gives the frequency the deck names (`.ac list 100k`) as `actual_x`,
with `actual_x_source: "deck"`, because the raw does not hold it.

## Trace math in Python

Any other trace math, such as a sum, a product or a function of a trace, is
done with numpy on the traces in Python (guide section 'python'):

```python
r = api.load_raw(job_id=job_id, case_id=case_id)  # or api.load_raw("run.raw")
k = 0  # one step at a time
t = r.axis(step=k)
p_load = r.trace("V(out)", step=k) * r.trace("I(Rload)", step=k)
t_w, p_w, _ = window_and_clean(t, p_load, 1e-3, None)  # from 1 ms to the end
stats = compute_signal_stats(t_w, p_w)  # stats["mean"] is the average power
```

Each step of a `.step` run has its own axis, which all of that step's traces
share, so never combine traces across steps.

Take statistics of a derived trace with `compute_signal_stats` and
`time_weighted_quantiles` instead of `np.mean`, `np.std` or `np.percentile`
over the samples. LTspice varies its timestep and clusters samples around every
edge, so a plain sample average or percentile over-weights the edges.
`compute_signal_stats` weights mean, RMS and standard deviation by time, the
way the `signal_stats` recipe does (its `min`, `max` and `pk_pk` are the sample
extremes). `time_weighted_quantiles(t, y, [0.01, 0.5, 0.99])["values"]` gives
one time-weighted quantile per level, in the order given. `window_and_clean`
cuts the window and drops non-finite samples first. All three are importable
from `ltspice_mcp.api` and in `run_code`'s scope.

## Quantiles in the `signal_stats` recipe

The `signal_stats` recipe reports `min`, `max` and `peak_to_peak` as the
sample extremes, so a narrow spike or an edge's overshoot stays in them. For a
spread that leaves out the few edges of a switching train, which no single
window can skip, name quantile levels. `"quantiles": [0.01, 0.99]` adds `q01`,
`q99` and `quantile_peak_to_peak` (highest level minus lowest), and `field` can
name any of these keys for a `reduce` or a `spec` to read. A key is the level
as a percentage with `_` for the decimal point, so 0.999 is `q99_9`. `q99` is
the smallest value the signal spends 99% of the window at or below.

## One frequency: amplitude and phase

The `tone` recipe reads the amplitude and phase of a transient signal at one
frequency you give:

```json
{"key": "out_1k", "metric": "tone", "signal": "V(out)", "frequency_hz": "1k",
 "window": {"start": "5m"}}
```

It cuts the window to the whole periods that fit from its start (`t_start`,
`t_end` and `n_cycles` report which), weights the samples by time, and returns
`amplitude` (peak, in the signal's unit), `phase_deg`, `dc`, and the
coefficients `sin_component` and `cos_component`. The recipe models the signal
as `dc + amplitude*sin(2*pi*f*t + phase_deg)`, where `t` is the simulation time
from zero, so the phase does not change with the window. The gain from `V(in)`
to `V(out)` at that frequency is therefore the ratio of their amplitudes, and
the phase shift is the difference of their phases. Start the window once the
start-up transient has settled. The recipe warns when the sampling step is too
wide for straight lines between samples to follow a sinusoid; a smaller `.tran`
maximum timestep removes the warning. In Python, `analyze_tone(t, y, f)` does
the same on arrays cut by `window_and_clean`. Prefer either to a `.meas`
correlation integral or to numpy over the samples, which weights LTspice's
clustered samples by count.

LTspice's `.four <freq> [Nharmonics] [Nperiods] <trace>` prints a Fourier
table to the log, and the `summary` recipe returns it as `fourier`, with each
harmonic's `magnitude` and `phase`. The number after the frequency is the
harmonic count, and the period count follows it. `.four 1k V(out)` gives nine
harmonics over one period, and `.four 1k 5 V(in)` gives five harmonics over one
period. According to LTspice XVII's help, the window is the last period before
the stop time, the last `Nperiods` periods, or the whole run for `-1`, so you
cannot choose where it starts. Each build prints the phase in its own
convention. For a `SINE(0 1 1k)` source, LTspice XVII prints the fundamental at
0 degrees and LTspice 26 at 90, and no constant offset converts one build's
phases to the other's. Read phase with `tone`. ngspice skips `.four` when it
writes a raw file, which is how the server runs it, and the run's warnings
report the skip.

## Reading a deck with `.step`

A `.step` directive puts several sweeps inside one `.raw`, and by default
`analyze_results` reads the first of them. Two call-level arguments change
this, and both apply to every recipe in the call. `step`
(`{"axis": "temp", "value": 27}`) reads the one iteration whose axis value you
name, and `all_steps: true` evaluates every recipe at every iteration. They are
mutually exclusive. `run_experiments`' attached `analyze` block takes the same
two, so an attached measurement and a standalone one read the same steps.
