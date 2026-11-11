`equivalence` did not compare the parameters a subcircuit instance runs with,
so two circuits that called one subcircuit with `g=2` and with `g=3`, or whose
subcircuit's `PARAMS:` default differed where an instance left it out, were
called equivalent. A subcircuit instance both circuits have is now compared
as a part is, in `param_mismatches`, by each parameter it gives (with or
without `params:`), else its definition's default, as LTspice reads an
instance: an instance that gives the default's value and one that leaves it
out compare equal.
