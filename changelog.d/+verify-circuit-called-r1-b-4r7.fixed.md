`verify_circuit` called `R1 a b 4R7` and `R1 a b 4.7` different circuits,
and a `random` rule on a `4R7` resistor was refused as having no numeric
nominal, though both LTspice builds run it at 4.7 (recorded:
`deck/suffix_shorthand`, `deck/suffix_unknown`). The equivalence compare and
the variation rules' nominals (component and `.param` rules, `.model`
nominals, MOSFET `W`/`L` and `.option scale`) now read a value as LTspice
does.
