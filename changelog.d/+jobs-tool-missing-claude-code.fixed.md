The `jobs` tool was missing from Claude Code sessions, though the server
listed it: its input schema put one branch per action under a top-level
`allOf`, which Claude's API refuses, so the client dropped the tool. Its
input schema is now one flat object, and the tool's description lists each
action's fields and marks the ones an action requires, so the compact
listing still says which action takes which field. The tool still refuses
a field the action does not take, naming both. The tool's definition is
also about 40% smaller. A test holds every tool's input schema to an
object with no `oneOf`, `allOf` or `anyOf` at its top level.
