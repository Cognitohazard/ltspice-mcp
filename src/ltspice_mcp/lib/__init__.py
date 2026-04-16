"""Library utilities for ltspice-mcp."""

from datetime import datetime, timedelta, timezone

_EST = timezone(timedelta(hours=-5), name="EST")


def now() -> datetime:
    """Return the current time in US Eastern (EST, UTC-5)."""
    return datetime.now(tz=_EST)
