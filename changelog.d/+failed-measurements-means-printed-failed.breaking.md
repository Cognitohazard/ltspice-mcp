`failed_measurements` now means one thing in every reply: the measurements
LTspice printed as failed, `Measurement "name" FAIL'ed` or `failed` in a row of
a stepped table. A `measurements` recipe row used to list there every name
whose `measured` value is null; it now lists only those, for the step the row
reads, and names any other null by why: under `unreadable` (LTspice printed
something other than a number, as before) or under the new `not_finite`
(LTspice printed a number that is not finite, such as a stepped FRA's gain
margin of `infdB`). A caller that read the row's `failed_measurements` for
every null reads `not_finite` and `unreadable` beside it, or the nulls in
`measured`. The run summary, `inspect(kind="results", view="measurements")`
and the Python API's `measurements()` now also name a measurement whose
stepped table LTspice 24 and later printed `failed` for some step, and its
entry lists those steps, from 0, in `failed_steps`; a stepped run's recipe row
carries `failed_measurements` beside its `stats`. Each row of the `inspect`
view whose value LTspice printed as failed now carries `failed: true`, so a
failed value reads apart from one that is null for another reason. The run
summary's measurements, and so its `failed_measurements`, are the whole run's
at whichever `step` it reads, while a recipe row read at one step names only
what failed at that step.
