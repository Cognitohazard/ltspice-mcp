A call that sends a retired argument name is told the name that replaced
it only where the old name was taken. `analyze_results`' `raw_path` and
`log_path` were retired inside each source; a `raw_path` at the top level
was told it "is now 'path'", which the top level does not take, and is now
answered with the names the top level takes.
