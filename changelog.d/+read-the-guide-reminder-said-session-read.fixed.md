The read-the-guide reminder said the session had not read the guide when
the server had only restarted, as after a reconnect, or when the guide was
read through `api.guide()` inside `run_code`, neither of which the server
sees. It now says the server has not been asked for the guide since it
started, so an agent that has read it carries on.
