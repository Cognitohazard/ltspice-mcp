An attached analysis on `run_experiments` no longer warns that an
`include.fields` path whose key starts with `_` is absent from every row.
It checks a projection against a key outline saved with the result, and
that outline left out every such key, so `assignments._random_run` or
`assignments._random_id` was called absent while each row it returned
carried it; `analyze_results` gave no such warning. The outline now keeps
every key, so both give the same warnings. A result saved before this
version keeps its old outline.
