`verify_circuit`'s `equivalence` comparison, and `edit_schematic`'s comparison
with a reference, compared only the directives outside subcircuit definitions,
so two circuits whose subcircuit set a `.param`, a `PARAMS:` default or a
`.meas` otherwise were called equivalent. Each subcircuit definition that an
instance both circuits have calls now has its own directives compared,
wherever it is defined and whatever each side names it, and a difference is
listed with `(in .subckt NAME)` after it.
