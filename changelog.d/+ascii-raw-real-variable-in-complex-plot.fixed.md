An AC run that saves a device parameter, such as `.save @m1[vth]`, can now be
read when ngspice writes its raw file as text, as the official Windows build
does by default. ngspice writes such a parameter as one real number a point,
and every reader of the results (`analyze_results`, `inspect`,
`plot_waveform` and the Python API) refused the file after the run itself had
completed. The parameter is now read as the real values the file holds:
`analyze_results` gives its value, `inspect` lists it as real, and
`plot_waveform` draws it as its values, in a panel apart from the magnitude
and phase of the plot's complex signals. The same values in a binary raw
file, which stores them as complex numbers with a zero imaginary part, read as
before.
