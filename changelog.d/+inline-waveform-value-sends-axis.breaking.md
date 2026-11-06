An inline `waveform` value sends its axis once, as `x` beside
`series`, instead of once in every series: every trace of one plot is read
at the same points. A 32-bit trace's samples are written in their shortest
32-bit form (`0.99995446`, not `0.9999544620513916`), which reads back as
the same 32-bit value.
