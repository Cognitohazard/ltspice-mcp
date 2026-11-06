`jobs(runs)` counted an interrupted job whose cases had all written their
results as a failed call, while `jobs(status)`, `jobs(wait)` and the
`run_experiments` receipt reported it complete. It now reports the
receipt's outcome.
