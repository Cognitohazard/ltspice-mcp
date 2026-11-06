`plot_waveform` given a `path` and a `run_index` other than 0 plotted the
file and said nothing of the `run_index`, while `inspect(kind="results")`
and the Python API's `load_raw` refused the same pair. All three now refuse
a `run_index`, `case_id` or `other_raw` sent beside a path, in the same
words, naming the selectors sent and the fix: leave them out to read the
file, or pass the job's `job_id` instead of the path.
