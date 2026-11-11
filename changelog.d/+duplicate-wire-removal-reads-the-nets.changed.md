`remove_wire` on a segment drawn more than once refused only a removal that
left two pins apart. It now reads the nets as every other edit does: it also
refuses a removal that would cut a label off the pins it was joined to, and
takes every copy when the two sides stay one net through labels of one name.
