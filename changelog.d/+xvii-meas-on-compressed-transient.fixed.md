The guide and the `coarse_compressed_waveform` observation no longer say that
LTspice computes its `.meas` results from the waveform it stores, or that
compression changed them. In the recorded runs, LTspice XVII's `.meas` results
read points its compressed raw does not hold and came within 0.25% of an
uncompressed run's, while a trapezoid average over the 13 stored points was
0.41%, and interpolation at 1 ms 1.15%, below those `.meas` results; LTspice 26
stored every point of the same run.
