`verify_circuit`'s `equivalence` comparison, and `edit_schematic`'s comparison
with a reference, compared only the directives outside subcircuit definitions,
so two circuits whose subcircuit set a `.param`, a `PARAMS:` default or a
`.meas` otherwise were called equivalent; on a hierarchical sheet, every
sub-sheet's own directives went unchecked. Each subcircuit the circuit calls
now has its own directives compared, wherever it is defined, and a difference
is listed with `(in .subckt NAME)` after it.
