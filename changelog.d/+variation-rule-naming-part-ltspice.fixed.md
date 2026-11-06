A variation rule naming a part that LTspice exports as `R§Load` by `RLoad`,
the name `verify_circuit` reports, or by `Load`, the name in its results,
was refused on an LTspice run, while an ngspice run took `RLoad` and refused
`R§Load` (recorded: `deck/section_sign_utf8`, `deck/section_sign_cp1252`).
`RLoad` now finds the part on either simulator and `Load` on LTspice, and a
name that is one part's reference and another part's other name is refused
as ambiguous.
