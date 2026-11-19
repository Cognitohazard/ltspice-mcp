A chart of several cases (`plot_waveform` with `runs`) checks, before it loads
any results file, whether the cases' plots share one analysis and whether the
chart stays within its cell limit, from the cases' records, the request and
each first plot's header, and refuses as before without loading the files.
Their results then load side by side, within one bound on the time it takes;
a case not loaded by then is named in `cases_not_drawn` as
`load_time_exhausted`, and the others are drawn.
