Once a stopped server's process id was reused, which happens quickly on
Windows, its running job was never reconciled: `jobs(wait)` returned at
once and said to wait again, `Api.wait` spun, and `jobs(cancel)` was
allowed without a token and then failed. A job record now carries its
owner's process start marker (`owner_start`) beside the pid, and the server
decides which jobs it runs from its own registry. A record whose `status`
is not one the server writes is refused as malformed.
