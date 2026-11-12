A card left open (a quote, brace or parenthesis with nothing to close it) is
now refused the same way by every tool that meets it, with the new error code
`card_unreadable`: the message names the file, the line, the card as written,
what is wrong and the fix. A Monte Carlo or mismatch run on such a card ended
with the tokenizer's bare message, which named neither the card nor its file,
and a brace left open inside a `.model` card's parentheses
(`.model D D(is={1e-14)`) passed `verify_circuit` and the linter as having no
problem. A `.meas` name is now read as the deck writes it, so an indented card
and a name such as `t.rise` keep their kind in the recipes, and a `.noise`
output written `V(out, ref)` no longer hides its input source's unit.
