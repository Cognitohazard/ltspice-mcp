Guide section 'schematics' sent a part that has to move with its labels to
`move_component`, which moves the part alone and leaves the labels and
wires at its old pins; `move` with `references` and `drag: true` takes
them along, so a part no longer needs to be removed and added again. It
also no longer suggests a label for a supply on the same small sheet:
labels are for ground and for nets that would otherwise cross the sheet.
