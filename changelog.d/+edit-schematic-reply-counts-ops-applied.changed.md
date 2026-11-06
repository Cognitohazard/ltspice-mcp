`edit_schematic`'s reply counts the ops it applied (`applied`): `results`
lists only ops with something to report, so an op without, such as
`add_directive`, read as one that did not run. `remove_directive` names
each text it removed, with its kind and place (`removed_texts`), which a
`regex:` match otherwise leaves unknown.
