Every edit of a sheet turned its comments and directives drawn on their
side (`VLeft`, `VRight` and the like) upright, whatever the edit changed:
spicelib's reader drops the turn. `edit_schematic` now saves each as the
sheet drew it, as LTspice does when it saves a sheet (recorded:
`save/shuffled`).
