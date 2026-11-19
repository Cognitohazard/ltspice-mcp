`plot_waveform` with `in_ltspice` now puts a job's results beside its sheet
through the same checked copy as `jobs(action="place_beside_sheet")`: a sheet
changed since the run, or gone, is no longer given that run's results. The
run opens on its own instead, and the `ltspice` block's `not_placed` gives
the refusal's code and fix. Two such calls at one sheet no longer copy over
each other.
