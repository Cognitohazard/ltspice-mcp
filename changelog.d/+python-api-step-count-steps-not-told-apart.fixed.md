In the Python API, `RawResult.step_count` was 0 and `RawResult.steps` was
empty for a plot whose steps cannot be told apart, such as the
operating-point raw LTspice 26 writes beside a stepped `.tran` or `.ac`,
which holds one point and does not say which of the run's steps it is, or a
stepped sweep read without the log that holds its step values. The tools
refuse to count those steps, and `trace` and `value` already refused to read
one. `step_count` and `steps` now count steps as the tools do, so for such a
plot they raise `ResultError` with the reason `analyze_results` gives, while
`signals`, `descriptor` and `plots` still read. The API contract in
`docs/design/python_api.md` now states this.
