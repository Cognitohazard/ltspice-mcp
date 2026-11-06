`jobs(wait, wait_for="runs")` could return while reads of the job's runs
still refused with `job_not_terminal`, until the job's next checkpoint or
its owner's next write. A job's runs are now readable once every case has
finished, by its status or by its cases, for the wait and every read alike.
