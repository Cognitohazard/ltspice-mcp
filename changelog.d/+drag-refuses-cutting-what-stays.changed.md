`edit_schematic`'s `move` with `drag: true` now refuses, before changing
anything, a move that would cut anything that stays off from what moves, and
names what would part, what stays and how to take it along. A region takes a
part only when its whole box is inside it, and with `drag` it moved every wire
end inside it, so a region that held one pin of a part but not the whole part
pulled the wire off that pin and committed the disconnection, reported only
afterwards. The same holds for a part whose pin touches a moved pin with no
wire, a pin inside a wire that moves, and a label left on a wire that
stretches. A wire both of whose ends move now moves whole, taking the labels
and wire ends along it, so naming every part of a net moves the whole net.
Without `drag` a move still takes only what it names.
