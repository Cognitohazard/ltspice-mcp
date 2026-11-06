An argument error came back as text alone, with no code; a typed failure
carried its code only when it also had suggestions, and an unclassified
failure carried none. Each now has `structuredContent` with its `code`:
`invalid_arguments` for an argument error, with a hint naming the
`inspect(kind="reference")` lookup for the tool; the failure's own code for
a typed failure; `internal_error` otherwise. A branch name that no branch
has (an unknown `action`, or one sent as an object) is reported at the
field, as "must be one of 'status', 'wait', …", instead of pydantic's
"Input tag … does not match any of the expected tags".
