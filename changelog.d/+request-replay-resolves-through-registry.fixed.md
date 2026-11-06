A `run_experiments` call that repeats a `request_id`, or a
`jobs(action="resume")` that repeats a `resume_request_id`, is no longer
logged as a new submission when the repeat reaches the request gate or comes
from the recovery journal. Every replay now finds its job as any other read
of a job id does: a job another process runs is reported as that process
last recorded it, and a job whose owner has exited is saved as
`interrupted`. A replay that a second call answered while a resume or a
recovery was starting the job no longer leaves `jobs` showing the job's
first status for the rest of the session, and the job id in a replay's
receipt is found afterwards in a session with persistence off.
