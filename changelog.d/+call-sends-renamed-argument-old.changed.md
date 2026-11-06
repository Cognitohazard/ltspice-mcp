A call that sends a renamed argument's old name is refused, as before,
and the refusal now also says that a copy of the tool's arguments still
listing the old name is out of date, and to reload the tool's definition or
reconnect the server. A client that checks a call against such a copy
refuses the new name, so the tool cannot be called at all until it does.
