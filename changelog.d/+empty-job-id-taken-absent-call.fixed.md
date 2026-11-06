An empty `job_id` was taken as absent where a call's sources were counted
and as given where a case selection was checked. Beside a `raw_path` in
`analyze_results`, `other_raw` passed unchecked, where a source with no
`job_id` refuses it, and `runs: {case_ids: [...]}` failed with an internal
error. `plot_waveform`, `analyze_results`' sources and `inspect` now refuse
an empty `job_id` as they validate the call, as `jobs` does.
