A measurement whose value LTspice printed as something other than a number
failed the run's whole measurements section, so every measurement of the
run was lost. LTspice prints a `WHEN`'s threshold as the deck writes it, a
parameter, an expression or another measurement included (recorded:
`log/meas_when_threshold_tran`, `log/meas_when_threshold_ac`). Such a value
now fails its own entry: its value is null and the entry carries
`unreadable`, the line LTspice printed and the reason. Every other
measurement reads. The entry is the same in the run summary, in `inspect`'s
`measurements` view, in the Python API's `measurements` and, as
`value.unreadable`, in the `measurements` recipe's row, which also lists the
name in `failed_measurements`.
