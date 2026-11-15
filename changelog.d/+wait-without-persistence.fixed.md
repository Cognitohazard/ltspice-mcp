With job records off (`[state] persist_jobs = false`), waiting on a job that
nothing in the session runs now returns at once, with a warning saying why.
`api.wait` used to ask again for ever, and `jobs(action="wait")` answered at
once but reported the wait as timed out.
