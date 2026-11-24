A ground flag in a schematic render, and in the layout check that
`verify_circuit` and `edit_schematic` share, was a triangle a third the width
and about half the depth of the one LTspice draws. It is now the triangle both
LTspice 26 and LTspice XVII draw at the default text size, so text under its
outer part is reported as `attribute_text_overlap`, as is a short text lying
wholly inside it, and a render's crop takes in the whole triangle.
