`inspect(kind="open_in_ltspice")` pages the documents open by `cursor`.
It listed the first 100 and counted the rest in `total`, with no way to
read them. The cursor is bound to the documents open, by window and path,
so a page asked after one was opened or closed is refused as
`invalid_cursor`.
