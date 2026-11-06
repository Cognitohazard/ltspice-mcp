The documentation and package metadata no longer claim that the server
runs on macOS. It refuses every `.raw` and `.log` read there, because the
memory limit of the process that reads results and the cleanup of that
process and its children have not been validated on macOS. Yet the README
said it found a simulator on macOS, the PyPI classifiers listed macOS, and
the Claude Desktop extension offered itself on macOS. The extension's
manifest now lists Windows and Linux only, and the README, `docs/DESIGN.md`
and the extension's README say that macOS is not supported yet.
