`equivalence` did not compare a subcircuit instance's own parameters (an `X`
card's `params:`), so two circuits that called one subcircuit with `g=2` and
with `g=3` were called equivalent. A subcircuit instance both circuits have is
now compared by its parameters as a part is, and a difference is listed in
`param_mismatches`.
