In `equivalence`, an `.include` or `.lib` whose file defines subcircuits and
nothing else is now compared by those definitions, not by its line, so a
subcircuit defined inline on one side and brought by such a file on the other
is no longer a directive difference. An include whose file holds anything
else, a model or a parameter, is still compared by its line.
