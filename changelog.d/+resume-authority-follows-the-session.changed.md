`jobs(action="resume")` without a `control_token` is authorized as a cancel
is: only by the session that runs the job. A resume took any session in the
same process as the job's owner, because the job's record names that process;
such a session now needs the job's `control_token` to resume it, as it already
did to cancel it. The `control_token` descriptions of `jobs` cancel and
resume, the Python API's `process_owned_job` and `detached_owner` notes and
its reference now say which session runs a job, where they said which process
owns it.
