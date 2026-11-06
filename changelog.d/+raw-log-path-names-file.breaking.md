A `.raw` or `.log` path that names no file is reported as
`result_file_not_found` (in the Python API, `ResultFileNotFoundError`, a
`ResultError`) by `analyze_results`, `plot_waveform`, `inspect`,
`load_raw`, `measurements` and `device_op_points`. The tools' replies
carry a hint that says to check the path, or to read a run by its
`job_id`. Before, it was `result_unreadable`, with advice to check a job's
run state.
`result_unreadable` remains for a file that exists but cannot be read.
