A results file a run writes beside its main results, read by its path
(`<case>.fra_1.raw`, `<case>.op.raw`), is now read with the log of the run
that wrote it (`<case>.log`) instead of a log of its own name that no LTspice
build writes; a log of the file's own name, where there is one, is still
its run's. A stepped frequency-response analysis read this way is now read
step by step in `analyze_results`, `plot_waveform`, `inspect` and the Python
API. Where the run's log is not beside the results, the refusal now names the
log it looked for and says the steps cannot be told apart without it.
