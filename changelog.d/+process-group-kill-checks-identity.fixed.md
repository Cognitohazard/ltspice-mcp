Cleaning up after a `run_code` worker no longer signals a process group by the
pid of a worker that has already exited. When the worker exited between calls,
the next call signalled the group whose id was that worker's pid, and once the
system had given the pid to another process, that could stop a process the
server never started. What a worker leaves running in its group is now stopped
as the worker exits, and a group is signalled by a stored pid only while that
pid still names the process the server started, checked by its start time. The
same check now guards the parser's cleanup of its worker and the stop of a
detached job's owner.
