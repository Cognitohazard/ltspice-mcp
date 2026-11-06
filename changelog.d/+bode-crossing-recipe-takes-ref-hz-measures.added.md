The `bode_crossing` recipe takes `ref_hz`, which measures `level_db` from
the gain at that frequency: `level_db: -3.0103, ref_hz: "1k"` finds where
the gain falls 1/sqrt(2) below its value at 1 kHz, in one call instead of a
`bode_point` read and a second crossing. The value's `level` is the
absolute level crossed, and `reference` says what it was measured from.
