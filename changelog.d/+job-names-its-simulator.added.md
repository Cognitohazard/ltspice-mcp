A job names the simulator it ran with as `simulator`, the selector a caller
passes back as `execution.simulator` to run the same executable: `ltspice`, or
`ltspice:xvii` for an entry of `[simulator.executables]`, and the default's
selector when the request named none. It is on the `run_experiments` receipt,
the `jobs` status, wait, resume and runs replies, each row of
`jobs(action="list")`'s `recent_jobs`, and the same replies through the Python
API. `simulator_notes` says when the session answering has no simulator under
that name, and when a job recorded by an earlier version ran an executable
that is none of the session's, so only its family is known.
