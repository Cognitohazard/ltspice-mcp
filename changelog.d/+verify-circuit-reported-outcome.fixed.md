`verify_circuit` reported `outcome: "complete"` when a check the caller
asked for could not run, so `checks=["export"]` in a session without
LTspice read as a clean result although no check ran. A check named in
`checks`, or the comparison asked for by supplying `compare`, that does not
run now makes the outcome `partial`, and `failed` when nothing the caller
asked for came back: no check ran and no drawing was returned, whether the
checks were skipped or failed. The `hint` names each such check with the
reason it did not run. A check the caller did not ask for still leaves the
outcome alone: one left out of `checks`, one that does not apply to the file
kind under the default list, every check under `render.mode: "only"`, and a
default-list check this session cannot run, such as `export` without
LTspice when nothing is compared, which `checks_skipped` still reports.
