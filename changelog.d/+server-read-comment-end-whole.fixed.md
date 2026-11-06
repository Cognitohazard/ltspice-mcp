The server read a `;` or `$` comment to the end of the whole card, where
both LTspice builds end it at the end of its line and read the card on
over its next `+` line. A parameter on a line after a comment, outside a
`.model` card's parentheses or inside them, was never seen by any check
or edit, and an edit that wrote the card again dropped it. The server
also read a `$` anywhere as a comment, where LTspice reads one inside or
at the start of a word as part of it, so a netlist with PSpice or OrCAD
net names (`N$1`, `$N_0001`) lost a node on every such card. Comments now
end at the end of their line, and only a `$` that stands alone begins
one. A Monte Carlo or mismatch variant reads its `.model` card's nominal
values the same way.
