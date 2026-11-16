The `measurements` recipe no longer reports the `measurement_zero_or_failed`
observation. A step that LTspice XVII printed as `0` in a stepped table is
now null in the reply and named under `not_established` (`{name: {steps,
reason}}`) on the recipe's row, on the measurement's entry in the summary and
the Python API's `measurements()`, and as `not_established` on the row of
`inspect(kind="results", view="measurements")`.
