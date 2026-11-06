`run_experiments` refuses, at submission, an attached `analyze` block whose
recipe names in `sources` a label the experiment does not carry, as
`recipe_invalid` at stage `validate`, not retryable, naming the label the
block's source carries (`experiment`). Before, every case ran and the
recipe then failed.
