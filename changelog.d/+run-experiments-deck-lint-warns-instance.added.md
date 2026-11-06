`run_experiments`' deck lint warns of an instance, `.model` or `.subckt`
card that gives one parameter more than once (`duplicate-param`,
`linter_version` 9), as `verify_circuit` reports it (`duplicate_parameter`):
which copy the simulator reads is not known.
