A measurement is read as LTspice printed it, on every route. A
`DERIV ... WHEN` gives the derivative, which both LTspice 26 and XVII print
after `=`, where a job read with its deck gave the point instead. A
measurement named with a dot (`t.rise`) or, on XVII, a dash (`t-fall`) now
reads under its name; it was missing from every reply. A step of a stepped
measurement table that LTspice XVII printed as `0` is no longer given as a
result of 0: XVII prints `0` both for a step whose measurement failed and
for a crossing at 0 s, so the step's value is null and its entry or row
names it under `not_established`, with the steps and the reason. A log read
by path whose files name no build may be XVII's, and is read the same way.
