A number written directly before a closing parenthesis or a comma, such as the
last argument of `SINE(0 1 1e3)` or the `1e3` of `1e3, 2`, was read as the
digits before its exponent, so `1e3)` was 1 and `verify_circuit` found
`SINE(0 1 1e3)` and `SINE(0 1 1e6)` equivalent. Every reader of a value
(`verify_circuit`, `analyze_results` bounds, `.step` values, lint rules and
Monte Carlo values) now reads the number before the parenthesis or comma, as
both LTspice builds do, so `1e3)` is 1000 and the compare reports the
difference.
