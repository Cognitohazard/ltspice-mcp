A `.lib file section` now reads only that section's cards everywhere, not
only in hierarchy discovery: a model or a `.save` that only another corner of
the library declares no longer counts as present for the `model-missing` lint
rule or for the analyses and saves a run expects from the deck. When two
subcircuit definitions share a name, the checks now read the first, which
ngspice keeps, where the dangling-node and floating-net checks took the later
header's ports and counted both bodies, and flagged ports that are connected.
