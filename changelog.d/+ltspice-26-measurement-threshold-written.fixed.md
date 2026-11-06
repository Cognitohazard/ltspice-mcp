On LTspice 26 a WHEN measurement whose threshold is written in braces
(`.meas tran t WHEN V(out)={vth}`) no longer disappears from a run's
measurements. It is reported as a threshold written as a parameter is: a
null value, `unreadable` with the log line and the reason, and the
crossing as `at`, in the `measurements` recipe, `inspect`'s measurements
view, the run summary and the Python API's `measurements`. A log with a
measurement that cannot be given an entry now fails the measurements read
and names it, where it was left out. An unreadable value is quoted as the
log printed it (`vth^2`), not as spicelib cut it short (`vth`).
