"""The detected simulators' own model libraries, and the one ranking and row
shape every model lookup uses."""

import functools
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.library_parser import (
    LibraryIndex,
    ModelEntry,
    parse_library_file,
)
from ltspice_mcp.lib.simulator import simulator_library_roots

logger = logging.getLogger(__name__)


# Suffixes treated as SPICE library files. ``.lib`` / ``.mod`` / ``.sub`` cover
# third-party packs (``.sub`` holds the subcircuit decks that make up the bulk
# of LTspice's bundled vendor models); ``standard.bjt`` / ``.mos`` / ``.dio`` /
# ``.jft`` (and the device-default ``.cap`` / ``.ind`` / ``.res`` / ``.bead``)
# are LTspice's bundled stock decks under ``lib/cmp``. What the walk over a
# simulator's library directories counts as a library file.
_SPICE_LIB_SUFFIXES = frozenset(
    {".lib", ".mod", ".sub", ".bjt", ".mos", ".dio", ".cap", ".ind", ".res", ".jft", ".bead"}
)


_WORD_TOK = re.compile(r"[A-Za-z]+|[0-9]+")


# Connection-order templates keyed by the SPICE device token of a .MODEL card.
# A .MODEL gives the model parameters but not how the part is wired; the device
# token dictates node order, and getting it wrong silently simulates the wrong
# circuit. ``<name>`` is the model name the instance references. Keys are the
# uppercased device token as the lexer reports it.
_DEVICE_USAGE: dict[str, str] = {
    "VDMOS": "Mxxx D G S <name>",
    "NMOS": "Mxxx D G S B <name>",
    "PMOS": "Mxxx D G S B <name>",
    "NPN": "Qxxx C B E <name>",
    "PNP": "Qxxx C B E <name>",
    "D": "Dxxx anode cathode <name>",
    "NJF": "Jxxx D G S <name>",
    "PJF": "Jxxx D G S <name>",
}


def _device_usage(device_type: str, name: str) -> str:
    """Return the connection-order string for a known .MODEL device token.

    Empty string when the device token is unknown or absent (so callers can
    omit the field rather than emit a misleading order). ``<name>`` in the
    template is replaced with the actual model name.
    """
    template = _DEVICE_USAGE.get(device_type.upper())
    if not template:
        return ""
    return template.replace("<name>", name)


#: Added to a candidate whose first word token matches the query's.
_FIRST_TOKEN_BONUS = 0.05


def _first_token(text: str) -> str | None:
    match = _WORD_TOK.search(text)
    return match.group() if match else None


def part_aware_score(query_lower: str, candidate_lower: str, *, cutoff: float = 0.0) -> float:
    """Similarity in [0.0, 1.0] biased for part-number-style names.

    Base is ``rapidfuzz.fuzz.ratio`` — a length-aware (Levenshtein) whole-string
    similarity. ``WRatio`` was used previously, but its partial-ratio path scores
    any short candidate that is a *substring* of the query at ~0.90, so 1-2 char
    model names ('NI', 'MP', '1') flooded the results and buried the genuine
    match. ``ratio`` keeps typo tolerance ('LTC3406'/'LTC3406A' ~0.93) while
    scoring those short substrings low (<0.3). A small bonus applies when the
    first word token of both strings matches — e.g. 'LTC3406' / 'LTC3406A' share
    'ltc', '2N3904' / '2N3906' share '2n' — to keep near-neighbour siblings
    ranked above cross-family matches with similar edit distance.

    ``cutoff`` is the score the caller filters at. A candidate whose edit
    similarity cannot reach it even with the bonus scores 0.0 without being
    tokenized, which for any one query is most of a simulator's library.
    """
    # A hair under the exact floor, so float rounding in ``cutoff - bonus``
    # never drops a candidate the bonus would have lifted to the cutoff.
    floor = max((cutoff - _FIRST_TOKEN_BONUS) * 100 - 1e-9, 0.0)
    base = fuzz.ratio(query_lower, candidate_lower, score_cutoff=floor) / 100.0
    if not base:
        return 0.0
    query_token = _first_token(query_lower)
    if query_token is not None and query_token == _first_token(candidate_lower):
        base = min(1.0, base + _FIRST_TOKEN_BONUS)
    return base


def _shared_prefix_len(a: str, b: str) -> int:
    """Length of the common leading-character run of two strings.

    Tiebreaks equal fuzzy scores toward the nearest part-number neighbour:
    '2n3905' shares '2n390' (5) with '2n3904' but only '2n3' (3) with '2n3055',
    so the electrically-adjacent sibling outranks the alphabetical accident the
    tool description advertises ('2N3905' → '2N3904').
    """
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


# Process-wide (mtime, size) cache of parsed library files. Every model lookup
# reads through it, whether a caller named the file or a search walked it out
# of a simulator's library, so a file found by one route and read back by the
# other is parsed once. Unbounded because a search of the simulator's library
# reads the whole install on every page, and an LRU smaller than the install
# would re-parse all of it each time; what it holds is bounded by the install
# plus the files callers name, and the values are immutable.
_library_file_cache: FileCache[LibraryIndex] = FileCache()


@functools.lru_cache(maxsize=1)
def _library_files_under(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Every SPICE library file under ``roots``, in a stable order.

    Library files are named by ``_SPICE_LIB_SUFFIXES``. Sorted per root,
    because ``rglob`` order is whatever the filesystem returns and a paged
    search must not reorder between pages. Kept for as long as the roots are
    the same directories: a full LTspice install holds thousands of files,
    and walking them on every page of a search would cost more than the search.
    """
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in _SPICE_LIB_SUFFIXES and path not in seen and path.is_file():
                seen.add(path)
                files.append(path)
    if files:
        logger.info(f"Found {len(files)} simulator library files")
    return tuple(files)


def parse_library_file_cached(path: Path) -> LibraryIndex:
    """Parse a library file through the process-wide (mtime, size) cache.

    A stale entry (the file's mtime or size changed) re-parses.
    """
    return _library_file_cache.get(path, parse_library_file)


def model_row(entry: ModelEntry) -> dict[str, Any]:
    """One model or subcircuit as every model lookup reports it.

    ``include_directive`` names ``source_path`` as this process sees it, which
    is the spelling the rest of the server reads: staging rewrites every
    reference the root deck carries to its staged copy, in the simulator's own
    form (Windows form for LTspice reached across WSL), and the include
    resolvers behind ``verify_circuit`` and the hierarchy reader resolve it
    the way staging does. A Windows spelling made here would cost a ``wslpath``
    process per row and, for a file on the Linux side of WSL, give a
    ``\\\\wsl.localhost`` path those resolvers cannot map back. The path is
    always quoted: a simulator's library usually sits under a directory with a
    space in it, and an unquoted ``.include`` stops at the first one.

    A ``.MODEL`` also carries its device token and the connection order that
    token dictates: the card gives parameters but not node order, and wiring
    the part in the wrong order simulates silently wrong.
    """
    row: dict[str, Any] = {
        "name": entry.name,
        "type": entry.model_type,
        "source_path": str(entry.source_path),
        "include_directive": f'.include "{entry.source_path}"',
        "ports": list(entry.ports),
        "params": dict(entry.params),
    }
    if entry.device_type:
        row["device_type"] = entry.device_type
        usage = _device_usage(entry.device_type, entry.name)
        if usage:
            row["usage"] = usage
    return row


def rank_models(
    indexes: Iterable[LibraryIndex], query: str, *, cutoff: float = 0.6
) -> list[dict[str, Any]]:
    """Every model in ``indexes`` whose name scores at least ``cutoff``
    against ``query``, best first, one row per name, each with its ``score``.

    Scored by ``part_aware_score``; equal scores go to the longer shared
    prefix (the nearest part-number sibling) before the alphabet, so an
    unrelated name that merely sorts earlier cannot bury the neighbour. A name
    defined in several files is reported once, from the first index that
    defines it (the sort is stable): vendor libraries repeat short helper
    subcircuits, and one name would otherwise fill a page. An index that
    raises while being read propagates, so a caller decides whether one
    unreadable file fails its lookup.
    """
    query_lower = query.lower()
    candidates: list[tuple[float, ModelEntry]] = []
    for index in indexes:
        for entry in index.models:
            score = part_aware_score(query_lower, entry.name_lower, cutoff=cutoff)
            if score >= cutoff:
                candidates.append((score, entry))
    candidates.sort(
        key=lambda pair: (
            -pair[0],
            -_shared_prefix_len(query_lower, pair[1].name_lower),
            pair[1].name_lower,
        )
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, entry in candidates:
        if entry.name_lower in seen:
            continue
        seen.add(entry.name_lower)
        row = model_row(entry)
        row["score"] = round(score, 3)
        rows.append(row)
    return rows


class LibraryManager:
    """The detected simulators' own model libraries: where they are, and a
    search over them that may run on a worker thread (the first one parses
    the whole install, into the shared cache of immutable indexes)."""

    def __init__(self, available_simulators: dict[str, type]) -> None:
        """Initialize library manager.

        Args:
            available_simulators: Dictionary of detected simulators from state
        """
        self._available_simulators = available_simulators

    def library_roots(self) -> list[Path]:
        """The detected simulators' own model-library directories.

        ``simulator.simulator_library_roots`` for every detected simulator, in
        detection order and without repeats. Recomputed on every call (a few
        stats; the WSL probe behind it is memoized) because LTspice extracts
        its library on first launch, and a server started before that must
        see it appear.
        """
        roots: list[Path] = []
        for simulator_class in self._available_simulators.values():
            for root in simulator_library_roots(simulator_class):
                if root not in roots:
                    roots.append(root)
        return roots

    def builtin_library_files(self) -> tuple[Path, ...]:
        """Every library file under ``library_roots()``, the set a search reads."""
        return _library_files_under(tuple(self.library_roots()))

    def search(
        self, query: str
    ) -> tuple[list[dict[str, Any]], list[tuple[str, tuple[int, int] | None]]]:
        """``rank_models`` over every library file the detected simulators
        ship, and the revision of each file it read.

        A revision is the stamp the parse cache checked for that file, so a
        paged caller binds its cursor to exactly what was searched without a
        second ``stat`` of the install. A file that cannot be read or parsed is
        skipped with a warning rather than failing the search: an install holds
        thousands of vendor files, and the caller named none of them.
        """
        indexes: list[LibraryIndex] = []
        revisions: list[tuple[str, tuple[int, int] | None]] = []
        for path in self.builtin_library_files():
            try:
                stamp, index = _library_file_cache.get_stamped(path, parse_library_file)
            except Exception as exc:
                logger.warning(f"Failed to search simulator library {path}: {exc}")
                revisions.append((str(path), None))
                continue
            indexes.append(index)
            revisions.append((str(path), stamp))
        return rank_models(indexes, query), revisions
