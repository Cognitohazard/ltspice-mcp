`add_component` accepted a reference that differs from another part's only
in case, such as `r1` beside `R1`. LTspice compares references without case
and netlists neither part (recorded: `connectivity/shared_reference_case`),
so `add_component` now refuses it.
