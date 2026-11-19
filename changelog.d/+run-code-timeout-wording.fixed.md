`run_code`'s description no longer says a timeout cancels a run the snippet
waits on. A wait on a run ends just before `timeout_s`, naming the job, which
keeps running; a timeout elsewhere interrupts the snippet, and stops the
worker with the jobs it started on Windows, or when it does not stop within a
few seconds.
