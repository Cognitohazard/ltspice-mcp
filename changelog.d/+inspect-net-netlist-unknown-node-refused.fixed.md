`inspect(kind="net")` on a netlist answered a name that no card of the deck
is on (`at="nosuch"` or `at="net:nosuch"`) as a net with no members, so a
mistyped node read as a real net with nothing connected. It now refuses the
name as `not_found`, as the same query on a schematic now refuses a name no
label carries and a pin of a part it does not hold (it reported
`netlist_invalid`, whose hint sent the caller to lint a file that has
nothing wrong with it). The netlist's refusal lists the
deck's nodes, suggests the closest one, and says what the query does not
read: a subcircuit's body, a file the deck includes, or a card it could not
read. Ground (`0`, `gnd`) is answered on every deck as before.
