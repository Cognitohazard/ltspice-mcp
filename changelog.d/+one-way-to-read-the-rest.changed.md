A reply that tells a caller how to read the rest of a cut list now writes
the call the same way in every tool: the MCP call with its arguments, and
the same call through the Python API beside it. This covers the paged run
rows of a job receipt, a partial `analyze_results` and its page of missing
cases, the findings `verify_circuit` lists past its cap per rule, a net
whose wire coordinates `inspect` pages, and the recovery hint of an
`invalid_cursor` error.
