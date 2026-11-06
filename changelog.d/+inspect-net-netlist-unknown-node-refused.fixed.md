`inspect(kind="net")` on a netlist answered a name that no card of the deck
is on (`at="nosuch"` or `at="net:nosuch"`) as a net with no members, so a
mistyped node read as a real net with nothing connected, while the same
query on a schematic refused a name no label carries. Both now refuse it as
`netlist_invalid`. The netlist's refusal lists the deck's nodes, suggests
the closest one, and says when the name is a node only inside a
subcircuit's body, which the net query does not read. Ground (`0`, `gnd`)
is answered on every deck as before.
