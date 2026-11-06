A ground flag drawn over a part's name or value raised no finding: the
sheet rule `attribute_text_overlap` compared a part's text with wires, other
parts and other text, and left ground flags out. It now reports text a
ground's triangle crosses (`across: "ground"`), in `verify_circuit`'s
layout check and in `edit_schematic`'s findings. The triangle is the one
the render draws; its size is not recorded from LTspice.
