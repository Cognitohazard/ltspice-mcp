Several stepped LTspice results refused every read with "Step boundaries
are unresolved": a stepped `.tf`, a stepped `.noise` at one frequency, a
`.op` stepped over temperature, and the results file of a stepped `.fra`.
The first three store one point a step with the stepped value first, as a
stepped `.op` does; the last holds each step's sweep in turn. Each now
reads a step at a time, sample for sample as LTspice's own reader reads
the file. The `.op.raw`
a stepped LTspice 26 run writes is still refused: it holds one point, and
nothing says which step's it is.
