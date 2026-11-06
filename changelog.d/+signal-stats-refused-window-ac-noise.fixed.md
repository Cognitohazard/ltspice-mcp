`signal_stats` refused a window on an AC or noise plot only after reading
the trace, and its refusal of a distortion or sensitivity sweep pointed at
`inspect(kind='results')`, which reads no samples of such a plot. Both are
refused before any read, and each refusal names the `waveform` recipe, which
returns the samples, over the same window where one was given.
