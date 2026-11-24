`remove_component` with `cleanup_wires: true` now follows the wiring out from
the part's pins: a wire it takes frees its far end, so a chain of wires left
hanging goes as a whole, and a label or ground flag left joining nothing goes
with it. A pin another part shares is left alone. Its result replaces the
`deleted_wires` count with `removed_wires` and `removed_labels`, which say
what was taken and are enough to put it back, and `kept_wires`, which names
each wire left at a pin and what it still joins. A later `remove_wire` or
`remove_net_label` in the same call whose target the cleanup already took
reports `already_removed_by` instead of failing, and a removal that finds
nothing now names what is at the point.
