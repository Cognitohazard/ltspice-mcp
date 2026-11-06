A `run_experiments` replay of a request whose job's owner had exited
reconciled the job in memory only, so the record stayed `running` on disk
and other sessions could report its cases differently. The replay now
reads the job as every other read of a job id does.
