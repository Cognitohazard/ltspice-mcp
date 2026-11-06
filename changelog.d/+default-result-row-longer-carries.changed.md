**Breaking:** a default result row no longer carries `snapshot_id`, as it
already left out `deck_sha256`. The reply's `source_hashes` row for the
row's `source` label has it, and `include.fields: ["snapshot_id"]` puts it
back on the row. A `waveform` or `plot` value no longer repeats the run
identity its row already carries.
