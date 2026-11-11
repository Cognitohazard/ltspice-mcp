A `random` variation takes `simulator_draws: true` to have ngspice draw the
random functions written in the deck (`agauss`, `gauss`, `aunif`, `unif`), a
PDK's statistical models among them, anew for each sample. Each case's deck
gets its own `.options seed`, derived from the entry's seed, the circuit and the
sample index and recorded as `_random_simulator_seed`, so every corner of one
sample runs with the same seed and `sample_start` replays a sample's draws too.
`rules` may then be empty. It runs on ngspice only and not with
`execution.simulator_seed` (`simulator_draws_unsupported`), and a deck that sets
its own seed option fails its cases with `deck_sets_simulator_seed`.
