A server that serves `run_code` with `[security] allowed_paths` set now logs a
warning at startup: a snippet runs with the server process's own file access,
which the sandbox does not confine. `[tools] run_code = false` turns the tool
off.
