In `equivalence`, an `.include` or `.lib` whose file (or section) defines
subcircuits and nothing else is now compared by those definitions, not by its
line, so a subcircuit defined inline on one side and brought by such a file on
the other is no longer a directive difference. An include whose file holds
anything else (a model, a parameter, a part), or that could not be read, is
still compared by its line. `structural_diff`, which compares the two files as
written, still lists the line.
