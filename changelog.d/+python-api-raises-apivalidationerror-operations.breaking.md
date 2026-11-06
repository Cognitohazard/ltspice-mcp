The Python API raises `ApiValidationError`, as the operations do, where it
raised `TypeError` or a plain `ValueError` for a misused argument: `path`
together with the older `raw_path` or `log_path` in `load_raw`,
`measurements` or `device_op_points`; an empty `job_id`, both `timeout_s`
and `timeout`, or a negative or non-numeric timeout in `wait`; a `wait` or
`detach` that is not a bool; and a detached call's argument that cannot be
handed to its owner, including a dictionary key that is not a string or a
value that holds itself, where a detached call raised `TypeError` or
`ValueError` or passed a number key on as text. `ApiValidationError` is a
`ValueError`, so code that caught `TypeError` here should catch
`ApiValidationError`. A value that holds itself no longer raises
`RecursionError` in any call.
