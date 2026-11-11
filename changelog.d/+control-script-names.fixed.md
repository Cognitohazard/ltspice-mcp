A file name with a space reaches ngspice in the control scripts the
server writes. ngspice keeps double quotes as part of a name and splits an
unquoted one at the space: a `write` then lands elsewhere with exit code 0,
and a `source` stops with a fatal error. So the `write` added to a scripted
deck was left out for a run folder whose path has a space, and a script
that ends with `quit` or issues its own analysis then left no raw to read;
and the seeded-run driver double-quoted a name that needed quoting, so a
seeded run of a deck in a folder with a space failed. One rule now writes
every such name: a plain name as it is, any other in single quotes. A name
ngspice would change (a quote, `$`, a backtick, a backslash, a brace, a
control character, a leading `~`) is not written, nor a non-ASCII one
except into a UTF-8 script off Windows: ngspice-42 stops on a deck that is
not valid UTF-8, which the added `write` in a cp1252 deck could make it.
Where a name is not written, a scripted run stays log-only, the hierarchy
check reports `path_unsupported`, and a seeded run is refused before the
job is recorded with `recovery_seed_unsupported`, naming the file and what
to rename, where it used to fail each case.
