`edit_schematic` writes the text of a directive or comment as it is to
read, with each backslash doubled, as LTspice writes one. A line break is
given as a line break: the two characters `\n` are now written as a
backslash and an `n`, where before they were written as they were and
LTspice read them as a line break. Before, a Windows path such as
`.include C:\new\models.lib` was written as given, and LTspice split the
directive at its `\n`. `remove_directive`, `replace_directive` and `move`
name a text as it reads or as the file spells it, so a `\n` escape still
names a text that holds a line break, and `replace_directive` applied again
with its reply's `now` and `was` puts back a text that holds a backslash.
