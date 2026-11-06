A `.meas` with a parenthesis left open is refused for LTspice: LTspice 26
refused the recorded one before the run and XVII ran that deck without the
measurement. `edit_schematic` refuses to write it, `verify_circuit`'s
`syntax` check reports it as a `directive_syntax` error, `run_experiments`
refuses it (`meas-unclosed-paren-ltspice`, blocking; `linter_version` 12),
and the LTspice 26 log's failure carries the fix.
