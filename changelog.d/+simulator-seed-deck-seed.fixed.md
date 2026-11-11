A recoverable ngspice job with `execution.simulator_seed` accepted a deck that
sets ngspice's seed itself with `.options seed=`. ngspice reads that card after
the server's `setseed`, so the deck's seed won: the job recorded a seed it never
drew from, and with `seed=random` a retried case drew different values. Such a
deck is now refused at submission (`recovery_seed_unsupported`).
