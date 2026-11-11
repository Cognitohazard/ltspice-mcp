Progress notifications. A call that asks for progress and waits on a job,
`jobs` wait or resume or `run_experiments` holding the call open, is told
how many of the job's cases have finished, of how many: when the wait
begins, then at most once a second while more have finished, and as the
wait ends if more have. A client can show how far the job has got and may
restart its timeout on each report. Nothing is sent while no case finishes,
as during one long case or the analysis attached to the job. Before, such a
wait was silent however long it ran.
