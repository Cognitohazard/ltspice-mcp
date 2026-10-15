"""Which file a symbol's name means.

A sheet names a symbol by a bare name (``res``) or by a folder and a name,
which LTspice writes with backslashes (``Opamps\\\\LT1001``). ``find_symbol``
turns that into a file: beside the sheet first, then in each library. It is
the one rule, for the pin geometry the schematic editor works with and for the
body the renderer draws, so the two cannot disagree about whether a part is
there.

The rule is what LTspice 26 and LTspice XVII were recorded doing (the
``symbol-beside-sheet``, ``symbol-named-with-a-folder-it-is-not-in`` and
``symbol-in-library-folder`` cases; ``docs/TESTING.md``, "Recorded LTspice
behaviour"):

- **Beside the sheet** a symbol is found only at the place its name says.
  Neither build looks into a folder under the sheet's for a bare name. For a
  name that says a folder the builds differ: LTspice 26 takes it as written
  and looks in that folder, XVII drops the folder and looks right beside the
  sheet. Both places are tried here, so a symbol either build would find is
  found. That is the safe side to err on: a part whose symbol is not found
  has no pins, and a connection to it goes unseen.
- **In a library** the folder in a name does not bind. Both builds find a bare
  name in any folder of the library, and a name that says a folder the library
  does not have.

Nothing here reads a symbol or keeps a cache; callers do both.
"""

from __future__ import annotations

import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.cache import file_stamp
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.symbol_file import SymbolFile, read_symbol


def spellings(symbol: str) -> tuple[str, str]:
    """A symbol's name as a relative path with forward slashes, and its bare name.

    The two are the same for a name that says no folder.
    """
    parts = [part for part in symbol.replace("\\", "/").split("/") if part]
    return "/".join(parts), parts[-1] if parts else ""


def find_beside(folder: Path, symbol: str) -> Path | None:
    """``symbol``'s file in ``folder``: where its name says, else under its bare name there.

    Never anywhere deeper. This is the whole rule beside a sheet, and the first
    half of the rule in a library.
    """
    for name in dict.fromkeys(spellings(symbol)):
        candidate = folder / f"{name}.asy"
        if name and candidate.is_file():
            return candidate
    return None


def find_in_library(root: Path, symbol: str) -> Path | None:
    """``symbol``'s file in the library at ``root``: as ``find_beside``, else under
    its bare name in any folder of the library."""
    found = find_beside(root, symbol)
    if found is not None:
        return found
    _relative, bare = spellings(symbol)
    if not bare:
        return None
    return next(root.rglob(f"{glob.escape(bare)}.asy"), None)


def find_symbol(symbol: str, sheet_folder: Path | None, libraries: Sequence[Path]) -> Path | None:
    """The file ``symbol`` names: beside the sheet, then in each library in turn.

    A symbol kept beside a sheet wins over a library's of the same name, as it
    does in LTspice. A folder that is not there is passed over.
    """
    if sheet_folder is not None and sheet_folder.is_dir():
        found = find_beside(sheet_folder, symbol)
        if found is not None:
            return found
    for root in libraries:
        if root == sheet_folder or not root.is_dir():
            continue
        found = find_in_library(root, symbol)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Symbol resolution (schematic-local → project → stock), content-stamp cached
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoundSymbol:
    """A symbol as a resolver found it: its name, its file, and the file read.

    ``name`` is the bare name, without any folder the sheet gave it.
    """

    name: str
    path: Path
    file: SymbolFile


class SymbolResolver:
    """Resolve a symbol name to a ``.asy`` file and read it, with caching.

    Precedence, highest first:

    1. the schematic's own directory (``local_dir``),
    2. configured project symbol paths (``project_paths``),
    3. stock LTspice library paths (``stock_paths``).

    Each resolver owns its caches; its search roots (and therefore its
    ``active_set``) are fixed at construction. The parse cache is keyed by
    ``(resolved path, content stamp, active library-path set)`` — never by bare
    symbol name — so rewriting a local ``.asy`` bumps the content stamp and
    invalidates the entry, while an unchanged file is served from cache. The
    active-set component is what keeps entries from bleeding across resolvers
    configured with different libraries: a different library set is a different
    resolver, and because it never resolves to a foreign entry, ``resolve`` for
    a symbol whose winning file changed with the search order returns the right
    path. Resolution results (including misses) are cached per instance, so a
    repeated miss does not re-walk the stock tree.
    """

    def __init__(
        self,
        local_dir: Path | None,
        project_paths: Sequence[Path] = (),
        stock_paths: Sequence[Path] = (),
    ) -> None:
        roots: list[Path] = []
        if local_dir is not None:
            roots.append(local_dir)
        roots.extend(project_paths)
        roots.extend(stock_paths)
        # De-duplicate while preserving precedence order.
        seen: set[str] = set()
        self._roots: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                self._roots.append(r)
        self._local_dir = local_dir
        self._active_set: frozenset[str] = frozenset(str(r) for r in self._roots)
        self._resolve_cache: dict[str, Path | None] = {}
        self._parse_cache: dict[tuple[str, tuple[int, int], frozenset[str]], FoundSymbol] = {}

    @property
    def active_set(self) -> frozenset[str]:
        return self._active_set

    def resolve(self, symbol: str) -> Path | None:
        """Return the ``.asy`` path for ``symbol``, or ``None`` if unresolved.

        ``symbol_library.find_symbol`` is the rule, with the sheet's own folder
        first and every other root a library: the rule the schematic editor's
        pin geometry follows too.
        """
        if symbol in self._resolve_cache:
            return self._resolve_cache[symbol]
        result = find_symbol(symbol, self._local_dir, self._roots)
        self._resolve_cache[symbol] = result
        return result

    def load(self, symbol: str) -> FoundSymbol | None:
        """Resolve and read ``symbol``; ``None`` if unresolved or unreadable."""
        path = self.resolve(symbol)
        if path is None:
            return None
        try:
            stamp = file_stamp(path)
        except OSError:
            return None
        key = (str(path), stamp, self._active_set)
        cached = self._parse_cache.get(key)
        if cached is not None:
            return cached
        try:
            found = FoundSymbol(spellings(symbol)[1], path, read_symbol(read_spice_text(path)))
        except Exception:
            return None
        self._parse_cache[key] = found
        return found
