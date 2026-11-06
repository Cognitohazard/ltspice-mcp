In the Python API, `RawResult` is no longer built with `step_count` or
`steps`: it reads both from its raw, so they agree with the tools. Code that
built a `RawResult` itself, rather than through `load_raw`, passes neither.
