A reply's text block is its `structuredContent` serialized as JSON, as the
MCP specification recommends, in place of a one-line summary. A client that
passes the model only the text now gives it the data, and a failed call's
text carries its `code` and recovery `hint`, which a client that shows the
model only an error's text (Claude Code among them) used to leave out.
`plot_waveform`'s `format` argument is removed, since every reply's text is
the same JSON; a call that still sends it is told so.
