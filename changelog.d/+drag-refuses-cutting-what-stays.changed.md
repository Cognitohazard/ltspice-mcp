`edit_schematic`'s `move` with `drag: true` now refuses, before changing
anything, a move that would cut something that stays off from what moves,
naming what would part and the parts that stay. A region takes a part only
when its whole box is inside it, and with `drag` it moved every wire end
inside it, so a region that held one pin of a part but not the whole part
pulled the wire off that pin and committed the disconnection, reported only
afterwards as a floating pin. A part whose pin touches a moved pin with no
wire between them is refused the same way. Without `drag` a move still takes
only what it names.
