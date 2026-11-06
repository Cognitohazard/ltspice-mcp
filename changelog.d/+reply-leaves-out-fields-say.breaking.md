Every reply leaves out the fields that say nothing: an empty list or
object, a null, a false flag, and a sub-answer that the overall answer
beside it already gives (`structurally_equivalent` beside
`equivalent: true`, `selectable` beside `available: false`). A reply about
work that found nothing no longer spends tokens saying so; over the test
suite's replies this is about 9% of the characters the server sends. A missing
field means "checked, and nothing to report", never "not checked": what did
not run, was skipped or could not be read is still stated, by a
`checks_skipped` row, a failure, a `status` or a `reason`. One function
applies the rule to every tool's `structuredContent`, to the text block that
serializes it, and to the Python API's results, so the API keeps the tools'
shape. Each tool keeps a short list of fields whose empty, null or
false value is the answer itself, such as a finding's `ok`, a comparison's
`equivalent` and a recipe's measured `value`
(`docs/design/mcp_surface.md`, "Fields left out"), and output schemas mark
every other field that may be absent as optional. Code that indexes a field
without checking breaks where the field said nothing: read
`reply.get("failures", [])`, and page until `next_cursor` is absent
(`page.get("next_cursor")`) rather than null. Pages leave out empty `items`
and a false `truncated`; run rows no longer keep every key whatever their
status, so a row that wrote no raw has no `raw`; an error's `retryable` is
present only when true; an unresolved hierarchy value is left out, with
its `status` and `reason`. The guide's examples read optional fields this
way. The budget ladder's trim rung no longer lists empty blocks to remove,
since every reply leaves them out, and `verify_circuit` now says, when it
skips the comparison for want of a netlist, whether the export ran.
