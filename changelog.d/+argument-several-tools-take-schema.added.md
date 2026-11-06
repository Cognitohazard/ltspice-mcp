An argument several tools take has one schema in all of them:
`plot_waveform` takes `plot_index: null` (the first plot) and refuses a
negative `run_index`, as `inspect` does, and `analyze_results` takes
`include: null`, as the attached `run_experiments.analyze` block does.
