`remove_directive`'s `removed_texts` gives a text as it reads, as the
drawing and the sheet findings read it: a line break where the TEXT record
holds a `\n` escape, and one backslash where it holds `\\`. It gave the
record's spelling, so a client that compared it with the `.asc` line now
sees a different string.
