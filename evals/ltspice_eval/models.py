"""The model side of the API profile: a protocol, the Anthropic client behind
it, and the prices that turn a call's usage into dollars.

The loop sends one request per turn and gets back the response as plain
Messages API dictionaries, so a scripted model (``scripted.py``) stands in for
the API in the offline tests through the same code.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens. A cache write at the default five-minute
    lifetime costs 1.25 times the input price."""

    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


#: First-party API list prices, per million tokens, as of 2026-10-06; a cache
#: read is a tenth of the input price where the list gives no other. A run
#: records the table it used. A model missing here cannot run: its dollar
#: ceiling would mean nothing.
PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(input=10.0, output=50.0, cache_read=0.25),
    "claude-opus-5-5": Price(input=4.0, output=20.0, cache_read=0.20),
    "claude-opus-5": Price(input=5.0, output=25.0, cache_read=0.50),
    "claude-sonnet-5-5": Price(input=2.0, output=10.0, cache_read=0.20),
    "claude-sonnet-5": Price(input=2.0, output=10.0, cache_read=0.20),
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Usage:
        return cls(
            input=int(raw.get("input_tokens") or 0),
            output=int(raw.get("output_tokens") or 0),
            cache_read=int(raw.get("cache_read_input_tokens") or 0),
            cache_write=int(raw.get("cache_creation_input_tokens") or 0),
        )

    @property
    def total_input(self) -> int:
        """Every input token the call read, cached or not: the size measure,
        since dollars depend on cache timing."""
        return self.input + self.cache_read + self.cache_write

    def add(self, other: Usage) -> None:
        self.input += other.input
        self.output += other.output
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write

    def usd(self, price: Price) -> float:
        return (
            self.input * price.input
            + self.output * price.output
            + self.cache_read * price.cache_read
            + self.cache_write * price.cache_write
        ) / 1e6

    def as_row(self) -> dict[str, int]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total_input": self.total_input,
        }


@dataclass
class ModelReply:
    content: list[dict[str, Any]]
    stop_reason: str | None
    usage: Usage
    model: str
    #: Seconds the call took, retries and their waits included.
    seconds: float = 0.0
    retries: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Model(Protocol):
    """One request, one reply. ``request`` holds every Messages API parameter
    the loop sets: model, system, tools, messages, and the output settings."""

    async def create(self, request: dict[str, Any]) -> ModelReply: ...


class ModelError(Exception):
    """The API refused a request, or kept failing after the retries."""


class AnthropicModel:
    """The Messages API, streamed, with the retries counted here rather than
    inside the SDK so each row can report them."""

    MAX_ATTEMPTS = 8
    BACKOFF_CAP_S = 60.0

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(max_retries=0)
        self._client = client

    async def create(self, request: dict[str, Any]) -> ModelReply:
        import anthropic

        started = time.monotonic()
        retries = 0
        while True:
            try:
                async with self._client.messages.stream(**request) as stream:
                    message = await stream.get_final_message()
                break
            except (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            ) as exc:
                retry_after = _retry_after(exc)
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500 and exc.status_code not in (408, 409):
                    raise ModelError(f"{exc.status_code}: {exc.message}") from exc
                retry_after = _retry_after(exc)
            retries += 1
            if retries >= self.MAX_ATTEMPTS:
                raise ModelError(f"gave up after {retries} attempts")
            delay = retry_after or min(self.BACKOFF_CAP_S, 2.0**retries) * (
                0.5 + random.random() / 2
            )
            await asyncio.sleep(delay)
        dumped = message.model_dump(mode="json", exclude_none=True)
        return ModelReply(
            content=dumped["content"],
            stop_reason=dumped.get("stop_reason"),
            usage=Usage.of(dumped.get("usage", {})),
            model=dumped.get("model", ""),
            seconds=time.monotonic() - started,
            retries=retries,
            extra={"id": dumped.get("id"), "stop_details": dumped.get("stop_details")},
        )


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
