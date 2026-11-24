A run stepped over a model parameter, such as `.step dm(is) list 1e-14 1e-13`,
can now be read step by step. LTspice names such a step `dm(is)=1e-14` in its
log, which the server read as a step of `is` beside text it could not place, so
`analyze_results`, `plot_waveform` and the Python API refused any step
selector on the run with "Step boundaries are unresolved". The step is now
read under the name LTspice gives it, and a selector such as `DM(Is)=1e-14`
picks it.
