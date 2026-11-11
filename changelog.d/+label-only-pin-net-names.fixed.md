`edit_schematic` named a label-only pin's net by the labels on the pin's own
point, so one pin could have two names in one reply. On a net labelled `OUT`
at that pin and `out` elsewhere, `wiring.label_only_pins` and the
`preexisting` view said `OUT`, while the pin legend, the `touched` view and
`inspect(kind="net")` said `OUT/out`. LTspice treats labels that differ only
in case as one net (recorded: `connectivity/node_name_case`), and an export
keeps each label's spelling. So every view of a pin's net now names it by
every label on the net, each spelling included. A label-only pin's row now
changes when its net's names do. An edit that changes them away from the
pin, by adding a name or spelling to the net or removing the last label of
one, now lists the pin in `wiring.label_only_pins`; it used to count the pin
under `preexisting`.
