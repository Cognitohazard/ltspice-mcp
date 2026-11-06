An invalid recipe in `run_experiments`' attached `analyze`
block is refused as `recipe_invalid` at stage `validate`, not retryable, as
`analyze_results` refuses it. It was `submission_failed` with
`retryable: true`, so a client that honoured the flag resent the same
request. A set `include.per_run.cursor` in the block is refused before
anything runs, where the analysis failed on it after every run.
