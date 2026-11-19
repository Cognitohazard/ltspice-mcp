`inspect(kind="capabilities")` and the `spice://config` resource give
`default_simulator` as the selector a run names it by (`ltspice`, `ngspice`),
the same words as the keys of `simulators`, instead of the simulator's class
name (`LTspice`, `NGspiceSimulator`).
