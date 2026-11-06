`inspect(kind="net")` on a netlist answered `at="M1.D"` with an empty net
named `M1.D`, which reads as nothing being on the drain. A netlist card has
no pin names, so a name after one of the deck's references is refused as
`invalid_at`, naming the `REF.<terminal>` form and its range, unless a node
of the deck is called that.
