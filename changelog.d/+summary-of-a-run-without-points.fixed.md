`analyze_results` with the `summary` recipe now answers for an LTspice run
that stopped before its first solution, such as a timestep or a voltage-source
loop error, whose results file holds no point. It used to fail with a parse
error when the file was named by path. The summary now gives the errors and
diagnostics the run's log holds and says `results_hold_no_point: true`; a
file with no log beside it says `log_not_found: true`. The answer is the same
by path, by job and through the Python API.
