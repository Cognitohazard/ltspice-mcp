`jobs(action="resume")` without a `control_token` is authorized as a cancel
is: only by the session that runs the job. A resume took any session in the
same process as the job's owner, because the job's record names that process;
such a session now needs the job's `control_token` to resume it, as it already
did to cancel it.
