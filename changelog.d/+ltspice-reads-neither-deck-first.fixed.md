LTspice reads neither a deck's first line, its title, nor any card after a
top-level `.end` (recorded: `deck/title_card`, `deck/title_directive`,
`deck/after_end`), but several readers did. The equivalence compare counted
a title that reads as a part as a second component; the structural diff
counted a title `.param` as a directive; `inspect(kind="net")` listed a part
after `.end` on its node; the linter refused a deck for an analysis after
`.end`; a title `.tran` counted as a requested analysis; and
`verify_circuit`'s directive check walked physical lines, so it checked a
title that reads as a `.meas` and missed a `.meas` whose function sits on a
`+` continuation line, which the linter refuses. Every reader of a deck now
skips both, and the directive check reads whole cards. The run observations
also find a source whose value sits on a `+` line past a comment, and no
longer take a `;` comment on a `.four` card for a requested signal.
