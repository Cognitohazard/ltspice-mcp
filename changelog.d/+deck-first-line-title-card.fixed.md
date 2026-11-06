A deck's first line, its title, and any card after `.end`, the deck's or
an included file's, are no longer read as cards. `inspect(kind="components")` listed a part after `.end`, and
`run_experiments` accepted assignments and random rules naming a part,
`.param` or `.model` found only in the title or after `.end`, wrote the
change where LTspice never reads it, and reported it in the receipt.
Mismatch rules perturbed a title that reads as a transistor or a
subcircuit call. Staging copied, or refused to run without, an include
named in the title or after `.end`. The `.options logopinfo` added to an
`.op` run could land after the deck's `.end`, or be left out because of a
line LTspice ignores; it is now added before the `.end` that ends the deck,
also when the `.op` is in an included file or the deck is UTF-16, and a
comment that mentions `logopinfo` no longer keeps it out. The ngspice
`.control` write injection read the title too, so a deck titled
"Write-up ..." got no `write` and ended with no raw. Measurement kinds, the
noise input unit and the ngspice `.lib` section check read those lines
too. What LTspice does after an included file's `.end` is not recorded;
every reader now follows the reading the hierarchy already took.
