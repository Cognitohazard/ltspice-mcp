`analyze_results`: the `tone` recipe reads a ratio, `V(out)/V(in)`, as the
quotient of the two phasors over the same whole periods, and takes each run's
frequency from a parameter with `frequency_param` (the step's value, else the
case's assignment), so one recipe reads every case of a frequency sweep. Every
recipe that reads a ratio also reads a reciprocal, `1/A` or `-1/A`. `stability`
on a frequency response analysis's results compares its crossover and phase
margin with the ones LTspice printed: the reply's observations say when they
differ, give both and name the form of `gain_N` whose margin agrees, or say
that the log held no margin to compare with.
