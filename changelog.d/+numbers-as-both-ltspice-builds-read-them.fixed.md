Values are read as both LTspice builds read them. A number is written in ASCII
digits; a letter of any script after it is a unit, so `10Ω` is 10. A value
whose digits after a suffix the builds read differently (`1.5k5`, `-1k5`,
`1e3k5`) is refused with the spelling to use instead, as is a number past a
double's range (`1e999`), `inf` and `nan`, where the builds take different
numbers. `1k5`, `+1k5`, `1.k5` and `.5k5` read as before. A braced constant
such as `{1k}` now compares equal to `1k` in `verify_circuit`, unless ngspice
is known to compute it differently (`{-2**2}`), and a Greek mu
stored as UTF-16 or as UTF-8 with a byte order mark is reported as micro on
both builds where LTspice reads it so.
