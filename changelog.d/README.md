# Changelog fragments

Each file here is one entry of the next release's section of `CHANGELOG.md`.
`scripts/release.sh` renders the fragments into that section, with towncrier,
and deletes them; nobody edits `CHANGELOG.md` by hand.

Name a fragment `+<change>.<type>.md`, where `<type>` is `breaking`, `added`,
`changed` or `fixed`, and write the entry's Markdown without the leading `- `.
It may run over several lines and paragraphs and hold a nested list. A change
with nothing to tell users adds `+<change>.internal.md`, which the release
renders as nothing.

Preview the next section with `uv run towncrier build --draft --version 0.0.0`.
