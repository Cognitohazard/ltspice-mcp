CI: a `checks / check-macos` job runs the suite on a native macOS runner,
with ngspice and Cairo from Homebrew. Reading results is still refused on
macOS, so a test that fails only because of that refusal is reported as
skipped, with the refusal as its reason, and so is a test of a recoverable
ngspice run, which the server also refuses there. Every other test must
pass. The release gate runs the job too, since it calls the same checks.
