The Python API's `load_raw`, `measurements` and
`device_op_points` raise `ApiValidationError` (a `ValueError`) for a source
named twice or not at all, an empty path, or a case selector beside a path,
as `analyze_results` does. They raised `TypeError`.
