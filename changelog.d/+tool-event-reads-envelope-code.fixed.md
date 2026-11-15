The per-call event on stderr now logs the error code of a failed `jobs` or
`run_experiments` call. Those tools report a failure as the envelope's
`error.code`, which the event did not read, so it logged the call with no code.
