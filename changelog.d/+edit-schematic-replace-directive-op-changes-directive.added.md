`edit_schematic`'s `replace_directive` op changes one directive's or
comment's text in one call. Its `instruction` names the text as
`remove_directive` does, the whole text or `regex:`, and must name one, or
`x` and `y` pick the named text anchored there; one that names several is
refused with the place of each. The text keeps its place, size,
justification and kind, which a `remove_directive` and an `add_directive`
lost unless given again, and a directive's new text is checked as
`add_directive`'s is. Its result gives the text's kind and place and what
it `was` and is `now`; applied again with that place, `now` and `was`, it
puts the text back.
