A `random` variation's seed draws different values than it did in earlier
versions, so a seeded Monte Carlo run from 0.7.0 does not reproduce. Its
draws were keyed on the case's position in the expanded list, so with
several `assign` values each value drew its own numbers, and adding a value
changed the draws of every case after it under the same seed. A sample's
draws now depend only on the seed, its circuit id and its sample index:
every case of one sample draws the same deviation at each `assign` value, so
a sweep compares the same die at each corner, and adding a value leaves the
other draws as they were.
