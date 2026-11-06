A measurement named as the deck spells it, such as `Vout_Max`, was not
found by the `measurements` recipe's `names` (an empty map), by a `field`
keyed by it (an indeterminate spec) or by `compute_measurement_stats`,
while `inspect` found it. A measurement name now matches in any case, and
a name the log does not hold fails the run, naming the ones it does hold.
