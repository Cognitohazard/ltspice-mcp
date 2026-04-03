"""Library utilities for ltspice-mcp."""

from datetime import datetime, timedelta, timezone

# US Eastern Standard Time (UTC-5).
# Using a fixed offset so the server behaves consistently without
# pulling in a third-party tz library (e.g. dateutil / zoneinfo).
EST = timezone(timedelta(hours=-5), name="EST")


def now() -> datetime:
    """Return the current time in US Eastern (EST, UTC-5)."""
    return datetime.now(tz=EST)
