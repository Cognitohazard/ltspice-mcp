`remove_wire` on a segment drawn more than once refused only a removal that
left two pins apart. It now reads the nets as the other edits that change
wiring do: it also refuses a removal that would cut a label off the pins it
was joined to, takes every copy when the two sides stay one net through
labels of one name, and its refusal names the point form of `remove_wire` as
the way to cut the connection on purpose.
