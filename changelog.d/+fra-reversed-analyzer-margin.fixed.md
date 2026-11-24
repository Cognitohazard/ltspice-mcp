`analyze_results`: `stability` on the results of an `.fra` analyzer placed with
its nodes the other way round (`@1 b a`) said the log printed no margin for the
analyzer and that which form of `gain_N` is the loop gain was not established.
It now says LTspice printed no finite margin, and that `gain_N` is the voltage
at the analyzer's first node over its second, so the loop gain is `-gain_N`
placed from the loop's driving output and `-1/gain_N` the other way. Results
read with no log beside them say so.
