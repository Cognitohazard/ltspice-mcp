A directive in that state, such as `.tran 0 1m )`, `.save V(in` or
`.param r={1k`, stopped `run_experiments` with the lexer's error as a
retryable submission failure, raised by staging, the linter, the variation
reader or the hierarchy walk behind recoverable runs and nested targets.
The deck now runs with the card reported. An `.include` or `.lib` in that
state fails its circuit with `include_unstaged`, naming the line and the
fix, since which file it names is not known; a variation or random rule on
a part, `.model` or `.param` name a card in that state may declare is
refused (`ambiguous_target`), naming its line, rather than made on another
copy or left out of a glob; and `inspect(kind="hierarchy")` and a nested
target refuse a card they need in that state by its file and line. The
`unreadable-card` lint rule covers staged includes too, and a live include
holding such an include card, for which `model-missing` says nothing.
