**Breaking:** every argument that names a file is `path`, as
`run_experiments`, `verify_circuit` and `inspect` already spelled it.
`plot_waveform` takes `path` for `raw_file`. An `analyze_results` source
takes `path` for `raw_path` and `log_path`: a `.log` (in any case) is read
alone, as a `log_path` was, and any other file as a RAW with the `.log` of
the same name, as a `raw_path` was. `edit_schematic` takes `path` for
`target`, and its reply names the sheet `path`. A call that sends an old
name is refused with the name that replaced it. The Python API's
`load_raw`, `measurements` and `device_op_points` take `path` first; their
older `raw_path` and `log_path` keywords are still accepted. The
observations `raw_path_without_deck_provenance` and
`log_path_without_deck_provenance`, named after the retired arguments, are
one, `path_without_deck_provenance`; the source's `source_hashes` row says
which file it read.
