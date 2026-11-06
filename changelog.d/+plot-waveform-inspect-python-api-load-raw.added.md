`plot_waveform`, `inspect(kind="results")` and the Python API's `load_raw`
report a compressed transient left coarse (`coarse_compressed_waveform`) as
`analyze_results` does, and `analyze_results` reports it for a call whose
only recipe is `measurements`, since XVII computes its `.meas` results from
the few points it keeps.
