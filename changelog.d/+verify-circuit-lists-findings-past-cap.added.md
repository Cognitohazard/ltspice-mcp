`verify_circuit` lists the findings past its cap of 25 a rule through the
tool itself. A rule with more has a `findings_omitted` entry, `{omitted,
total, next_cursor}`, and the new `findings_cursor` argument takes the
cursor back to list that rule's next 25 findings, so every finding can be
read without `run_code` or the Python API. The cursor is bound to the
rule's findings as they were found, and is refused with `invalid_cursor`
once they change (the file, a symbol it uses or its export), or when the
call does not run the check that reports the rule, naming what to change.
The entry replaces the `<rule>: showing 25 of <n> findings` observation,
which gave no way to read the rest, and
`dropped_wire`, which was cut at the cap with no word, now has one too.
The hint names each rule cut short and the check that reports it.
