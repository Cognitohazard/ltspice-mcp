`run_experiments`' refusals before a job exists give the message followed
by that kind of failure's recovery step in `hint`, where the hint repeated
the message. A failed resource read follows its message with the recovery
step and carries `{code, hint}` in the JSON-RPC error's `data`.
`inspect`'s `invalid_cursor` item carries the hint to send back the cursor
the previous page returned, as every other tool does.
