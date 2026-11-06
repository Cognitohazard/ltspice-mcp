`edit_schematic(open_window="save"|"discard", ops=[])` settles a sheet the
user changed in LTspice and did not save, once they choose, instead of
asking them to do it by hand. `"save"` has LTspice save the sheet from the
user's window, as Ctrl+S does, and the reply gives the file's new `sha256`.
`"discard"` closes the sheet there without saving and opens it again from
the file. Use it after an edit is refused with `open_window_differs`, or
after a plot opens with `tied: false` because its sheet had unsaved changes.
A box LTspice shows instead of saving, such as for a write-protected file,
is quoted in `window_not_saved` and closed with Cancel; a window whose copy
changes during a discard is left open, with `window_not_discarded`. When
the sheet is open in more than one LTspice and each holds a different copy,
`window_pid` names the one the user chose to save, the other windows are
handed the saved sheet, and a save without it, or with a process that
holds no different copy, fails with `window_not_chosen`.
`open_window="save"` replaces `save_open_window=true`, added earlier in
this release; a call that still sends `save_open_window` is told so.
