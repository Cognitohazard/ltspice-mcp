A job whose process stopped before its runs finished now reads `interrupted`
whether the process crashed or shut down. A server shutdown, `Api.close()`,
the end of a `with` block and interpreter exit used to end the jobs they
stopped `cancelled`, which now means only that a caller cancelled the job; a
caller that branched on `cancelled` after a shutdown now sees `interrupted`.
The rest of the job reads as after a crash too:

- an attached analysis left pending or running is `failed`, where a shutdown
  made it `cancelled`;
- a process that stops once every run has finished leaves only the analysis
  unfinished, so the job is `completed_with_failures` (or `completed` when no
  analysis was left and every run produced), where a shutdown made it
  `cancelled`;
- unfinished cases fail with code `server_restarted` as before, and every
  route now gives them one message, which says the process running the job
  stopped rather than that the server did, since a closed `Api` is no server;
- the job and the analysis each carry a `server_restarted` lifecycle
  observation, which a shutdown left out.

Resume is unchanged: a resume takes any terminal job, and the cases a shutdown
stops were already `failed` with `server_restarted`, which a default resume
retries without `retry_cancelled`. The structured log gains an `interrupted`
job event.
