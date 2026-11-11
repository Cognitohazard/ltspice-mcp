A recoverable ngspice deck that fixes its own seed, with `.options seed=<n>` at
its top level, may now use `agauss`, `gauss`, `aunif`, `unif` and `limit`
without `execution.simulator_seed`, and with any number of analyses: ngspice
reseeds from that card when it loads the deck, so the draws repeat on every run
and retry.
