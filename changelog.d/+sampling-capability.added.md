The capability table declares statistical sampling, the `random` variation's
reproducible Monte Carlo samples with their replay and comparison, as a
baseline capability of every simulator family. LTspice and ngspice are
supported, with their recordings and live tests as evidence; QSPICE and Xyce
are unverified, since no run of either is recorded and a mismatch draw inside
a subcircuit needs the hierarchy reader, which does not model them.
`inspect(kind="capabilities")` and the `spice://capabilities` resource list it.
