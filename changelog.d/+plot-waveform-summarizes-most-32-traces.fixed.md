`plot_waveform` summarizes at most 32 traces and counts every plotted trace
in `traces_truncated`, but named the call that reads the rest only in the
prose of a `trace_summary_truncated` observation, and that call read the
whole run whatever the plot had drawn: a `signal_stats` recipe even for an
ngspice distortion or AC sensitivity sweep, which that recipe refuses, with
no window, steps or run. The reply now carries a `hint` beside
`traces_truncated` naming the `analyze_results` call that reads the traces
left out over what was plotted: the source with `runs` set to the job's run
or case, the plot's `t_start`/`t_end` as each recipe's `window`, and its
step selector, or `all_steps=true` with the rows that are the plotted step.
The recipe is `signal_stats` where it reads the plot, else the `waveform`
recipe, which returns the samples: on a distortion or sensitivity sweep, and
on an AC or noise plot drawn over a window. The `trace_summary_truncated`
observation said nothing the count and the hint do not, and is no longer
emitted; read `hint` instead.
