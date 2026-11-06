**Breaking:** `remove_directive` and `move` name texts by one rule: a
literal names every text equal to it, surrounding space ignored, and
`regex:` names every text the pattern is found in, case-sensitive.
`remove_directive` removes every text it names. It removed only the first
copy of a literal and matched a `regex:` against directives only, from
their start and ignoring case, while `move` searched every text anywhere and
case-sensitively, so one pattern could move a directive and not remove it.
