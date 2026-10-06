"""Bounded resident results with content identities and per-session admission.

Byte accounting includes reachable Python metadata and distinct NumPy backing
storage, including corrected time arrays. It bounds retained entries, not total
process RSS or client references. Snapshots keep entries alive while a worker
uses their keys; no in-flight parser is shared between requests.

Entries are keyed by content, which only a parser process can establish. A
second, bounded index maps the stat stamps a read's sources carried when that
content was captured to its key, so a later read of the same unchanged files is
answered without one (``parser_service._source_stamps``). The index is
consulted only while admission is open: a retained parser failure closes it too.
"""

from __future__ import annotations

import math
import re
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_process import ParserProcessLimits, WarmParser

RESULT_CACHE_BYTES = 512 * 1024 * 1024
RESULT_CACHE_ENTRIES = 32
STAMP_ENTRIES = 256
"""Source stamps remembered at once. A stamp outliving its entry answers
nothing, so this only bounds the index's own size."""
_KEY = re.compile(r"[0-9a-f]{64}\Z")


class ParserCleanupError(ResultError):
    """Scratch remains because process exit or directory removal was unconfirmed."""

    code = "parser_cleanup_failed"

    def __init__(self, directory: Path, *, worker_pid: int | None = None) -> None:
        self.directory = directory
        self.worker_pid = worker_pid
        super().__init__(f"Parser cleanup was not confirmed; retained files at {directory}")


@dataclass(frozen=True, slots=True)
class _ParserFailure:
    directory: Path
    worker_pid: int | None


def resident_size(value: ParsedArtifacts) -> int:
    """Account immutable metadata and backing storage once within an entry."""
    seen: set[int] = set()

    def size(item: object) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        count = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                return count + size(item.base)
            # getsizeof includes owned NumPy storage already.
            return count
        if isinstance(item, memoryview):
            return count + size(item.obj)
        if is_dataclass(item) and not isinstance(item, type):
            if hasattr(item, "__dict__"):
                return count + size(vars(item))
            return count + sum(size(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, Mapping):
            return count + sum(size(key) + size(member) for key, member in item.items())
        if isinstance(item, (tuple, list)):
            return count + sum(size(member) for member in item)
        return count

    return size(value)


class ResultCache:
    """Content-keyed LRU; one contained parser at a time for this session."""

    def __init__(
        self, *, max_bytes: int = RESULT_CACHE_BYTES, max_entries: int = RESULT_CACHE_ENTRIES
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Result cache bytes must be a positive integer")
        if type(max_entries) is not int or not 1 <= max_entries <= RESULT_CACHE_ENTRIES:
            raise ValueError("Result cache entries must be between 1 and 32")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[ParsedArtifacts, int]] = OrderedDict()
        self._stamps: OrderedDict[str, str] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._parser_slot = threading.Lock()
        self._parser_failure: _ParserFailure | None = None
        self._warm: WarmParser | None = None

    @property
    def byte_count(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> ParsedArtifacts | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: str, value: ParsedArtifacts) -> ParsedArtifacts:
        if (
            _KEY.fullmatch(key) is None
            or not isinstance(value, ParsedArtifacts)
            or value.snapshot_id != key
        ):
            raise ValueError(
                "Result cache requires captured artifacts with the same content identity"
            )
        charge = resident_size(value)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if charge > self.max_bytes:
                return value
            while self._entries and (
                len(self._entries) >= self.max_entries or self._bytes + charge > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted[1]
            self._entries[key] = value, charge
            self._bytes += charge
        return value

    def snapshot(self, *, require_raw: bool = False) -> dict[str, ParsedArtifacts]:
        """Strong references, retained by the caller until the parser is reaped."""
        with self._lock:
            return {
                key: entry[0]
                for key, entry in self._entries.items()
                if not require_raw or entry[0].raw is not None
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._stamps.clear()
            self._bytes = 0

    def stamped(self, stamp: str) -> ParsedArtifacts | None:
        """The entry ``stamp`` was recorded against, while both are held.

        None while a parser failure keeps admission closed, so a read then
        reaches ``parse_slot`` and its refusal exactly as before.
        """
        with self._lock:
            if self._parser_failure is not None:
                return None
            key = self._stamps.get(stamp)
            if key is None:
                return None
            entry = self._entries.get(key)
            if entry is None:
                del self._stamps[stamp]
                return None
            self._stamps.move_to_end(stamp)
            self._entries.move_to_end(key)
            return entry[0]

    def record_stamp(self, stamp: str, key: str) -> None:
        """Remember that sources carrying ``stamp`` hold the content ``key`` names."""
        with self._lock:
            if key not in self._entries:
                return
            self._stamps[stamp] = key
            self._stamps.move_to_end(stamp)
            while len(self._stamps) > STAMP_ENTRIES:
                self._stamps.popitem(last=False)

    def warm_parser(self, limits: ParserProcessLimits, cwd: Path) -> WarmParser:
        """The parser tree this session keeps between calls, made on first use."""
        with self._lock:
            if self._warm is None:
                self._warm = WarmParser(limits, cwd)
            return self._warm

    def close_parser(self) -> bool:
        """Close the kept parser tree; True once it is confirmed gone."""
        with self._lock:
            warm, self._warm = self._warm, None
        return warm.close() if warm is not None else True

    def retain_parser_slot(self, directory: Path, *, worker_pid: int | None = None) -> None:
        """Keep admission closed when the active call cannot confirm tree exit."""
        with self._lock:
            if self._parser_failure is None:
                self._parser_failure = _ParserFailure(directory, worker_pid)

    @contextmanager
    def parse_slot(
        self, *, deadline: float, cancel: threading.Event | None = None
    ) -> Iterator[None]:
        """Queue with a finite deadline; cancellation never touches another call."""
        if not math.isfinite(deadline):
            raise ValueError("Parser admission deadline must be finite")

        def check() -> float:
            with self._lock:
                failure = self._parser_failure
            if failure is not None:
                raise ParserCleanupError(failure.directory, worker_pid=failure.worker_pid)
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Parser admission was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Parser admission exceeded its deadline")
            return remaining

        while not self._parser_slot.acquire(timeout=min(0.05, check())):
            pass
        try:
            check()
            yield
        finally:
            with self._lock:
                if self._parser_failure is None:
                    self._parser_slot.release()
