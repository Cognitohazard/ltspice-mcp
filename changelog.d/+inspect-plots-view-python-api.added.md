`inspect(kind="results")`'s plots view, and the Python API's
`RawResult.descriptor` and `plots`, say which kind of stepped result an
LTspice plot is (`stepped_kind`): each step's sweep in turn, one point a
step for a stepped `.op`, `.tf` or one-frequency `.ac` or `.noise`, or the
one point of a stepped run's `.op.raw` that names no step. It also lists the
variables that hold each step's values (`step_variables`).
