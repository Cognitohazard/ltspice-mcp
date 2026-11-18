A transient whose time step collapses part way through is now reported as a
convergence failure on LTspice XVII, as it is on LTspice 26. XVII writes the
error (`Fatal Error: Analysis:  Time step too small; ...`) on the line of its
`Heightened Def Con` progress, with no line break between them, and the run was
reported as `execution_failed` with no error read from its log. The run's
failure code, its `diagnostics` and the error now say what failed in the run
rows of `run_experiments` and `jobs`, in `inspect` and in the Python API.
