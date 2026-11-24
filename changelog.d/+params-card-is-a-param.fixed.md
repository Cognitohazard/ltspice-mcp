A `.params` card is now read as `.param`, as LTspice 26, LTspice XVII and
ngspice read it. The server read it as an unknown directive, so its
parameters were not found as `run_experiments` variation targets and the
parameter checks before a run did not see them.
