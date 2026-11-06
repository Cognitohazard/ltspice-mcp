Closing and reopening an LTspice plot or sheet could fail on a busy machine
with `ltspice-mcp-bridge.exe did not answer within 5 s`, because the bridge's
time limit also counted the wait for LTspice to close the pane or answer its
question. That wait is now added to the limit.
