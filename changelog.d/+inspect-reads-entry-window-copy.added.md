`inspect(kind="open_in_ltspice")` reads every entry where a window's copy
of a sheet differs from the file. The listing still names 25 entries a
side and counts the rest in `difference_omitted`, and a `hint` beside the
count now names the call that reads them: the same query with the sheet's
`path` (and `pid`, where more than one window has it open), which returns
that window's difference with both lists paged under one cursor. The
cursor is bound to the difference, so a page asked after either copy
changed is refused as `invalid_cursor`; a sheet no window has open fails as
`sheet_not_open`. Every reply that names a window's difference in part
now names the same call: `edit_schematic`'s `open_window_differs` refusal
and a dry run's note about a differing window, `verify_circuit`'s hint
when its `in_ltspice` window shows another sheet than the one checked, and
the `note` of `plot_waveform`'s `ltspice` block. One function writes that
call for all of them. The Python API's `inspect` follows the cursor and
returns the whole difference.
