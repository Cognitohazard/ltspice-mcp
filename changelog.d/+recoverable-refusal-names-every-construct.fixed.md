A recoverable `run_experiments` request that its deck or an included library
cannot support is now refused once, naming the constructs at fault: how many
there are, and up to twelve of them, each with its code, the file as you wrote
it, the line and the card. Before, it stopped at the first construct and named
no file, line or function, so a deck took one submission per construct to fix.
An include read in place is now named, with the reason it was not copied.
