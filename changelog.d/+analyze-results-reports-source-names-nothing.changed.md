**Breaking:** `analyze_results` reports a source that names nothing as the
other tools do: an unknown `job_id` and a `path` with no file are rows of
`coverage.missing_cases`, coded `job_not_found` and `result_unreadable`,
each with its recovery hint. An unknown job was a row coded
`source_unavailable` with no hint, and a missing file failed its recipe.
`jobs` follows an error's message with its recovery hint, where it repeated
the message, and a failed `inspect` query carries the hint too.
