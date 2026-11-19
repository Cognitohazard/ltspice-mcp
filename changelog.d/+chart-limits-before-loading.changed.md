A chart of several cases (`plot_waveform` with `runs`) over its cell limit is
refused without loading every results file: the cases' records, the request
and each first plot's header say beforehand whether it would be over, and it
is refused as soon as the cases loaded are over by themselves. A chart the
cases that load fit is still drawn. The files load two at a time within one
bound on the time it takes; a case not loaded by then is named in
`cases_not_drawn` as `load_time_exhausted`, and the others are drawn.
