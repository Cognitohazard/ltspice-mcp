A `.meas ... WHEN` with no `FIND` now reads as the crossing it found on
every route. LTspice prints such a measurement as the level it searched for
followed by the crossing after `AT`, and the measurements recipe read through
a bare log path reported the level as the result, while the same log read
through a job reported the crossing; a spec on it could pass one way and fail
the other. Which field holds a measurement's result is now decided once, from
how LTspice printed it, and every reply that reports measurements names it:
the summary's `measurements`, the Python API's `measurements()` and each row
of `inspect(kind="results", view="measurements")` carry `answer` (`value` or
`at`), or `answer_unknown` with the reason where a log prints a form no
recording shows. A WHEN whose threshold is written as a parameter or an
expression now reads its crossing instead of failing.
