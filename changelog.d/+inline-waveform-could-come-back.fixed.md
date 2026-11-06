An inline `waveform` could come back at several times the response budget,
and a caller's `budget` changed nothing but add `budget_not_met`: its
samples were the one part of a reply no rung reduced. Four traces of a
1,042-point transient came to about 42,000 estimated tokens. The samples
are now thinned to fit, each bucket keeping its minimum and maximum, and
the recipe's warnings say so; `points_total` still counts every sample.
Under the server's default budget only a waveform whose call sets no
`max_points` is thinned, so a count the caller set is kept; a caller's
`budget` thins any waveform from its answer rung on. A run receipt thins
its attached waveforms whatever their `max_points`, because its rows
preview what `analyze_results` over the job returns.
