An `edit_schematic` dry run with `compare` returned no `verification`, so a
batch could be checked against its reference only by committing it. The
reference check exports a staged copy of the sheet the batch would write,
so it now runs on a dry run too, an op-less read included, and reports the
same block, the exported netlist on a mismatch and a `partial` outcome, with
nothing saved. When an op failed the batch would write nothing, so
`verification.compare_error` says the sheet was not compared.
