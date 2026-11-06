`run_experiments` accepted a circuit whose file is no circuit (a `.txt`,
say) and made a job whose every case failed with `unsupported_variant`. It
now refuses the call before a job exists, with `unsupported_file`, as it
refuses a path outside the sandbox and as `inspect` refuses such a file.
