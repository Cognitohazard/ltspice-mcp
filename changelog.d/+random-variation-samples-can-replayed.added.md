A `random` variation's samples can be replayed. Each case records its sample
index (`_random_run`) and its seed (`_random_seed`) in its assignments, and
an entry without a `seed` gets one the server chooses, so a Monte Carlo run
without one is no longer impossible to repeat. A new `sample_start` sets the
first sample index: with the same seed and circuit id, `runs: 1,
sample_start: 7` draws what sample 7 drew in the full run. A retry of a
seedless request under the same `request_id` returns the job it made, with
that job's seed.
