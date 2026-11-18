`plot_waveform` draws the AC results of ngspice 42, which it refused with "The
selected plot has non-real coordinates". ngspice 42 writes the frequency of an
AC plot with an imaginary part that is a subnormal number (about 4.6e-310), a
value no simulator computes as a coordinate, so the axis is read as its real
part by every reader: `plot_waveform`, `analyze_results`, `inspect` and the
Python API. An axis whose imaginary part can be a coordinate is still refused,
now also by the readers that used to read its real part without saying so (a
value at a frequency, `bode_filter` and the other AC recipes, the summary's
frequency range and a run's progress), and two results whose axes differ there
are no longer taken for the same result.
