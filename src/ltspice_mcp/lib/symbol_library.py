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

``SymbolResolver`` finds and reads the symbols of one request with that rule,
and ``SymbolLibrary`` is a session's set of libraries and its cache of files
read.
"""

from __future__ import annotations

import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.cache import FileCache
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
    ``problem`` says why the symbol cannot be used, as words that follow its
    name ("has a pin line that does not read ('PIN 32')"); it is empty for a
    usable one. A pin left out is a terminal no wire could be checked against,
    so a symbol with a pin line that does not read is unusable whole, as it is
    to the schematic editor's geometry.
    """

    name: str
    path: Path
    file: SymbolFile
    problem: str = ""


#: Symbol files as read, each with why it cannot be used ("" for a usable
#: one), by file and current for its content stamp. Shared by every resolver
#: of one ``SymbolLibrary``.
ParseCache = FileCache[tuple[SymbolFile, str]]

#: Where each name resolved in an LTspice build's own library, misses
#: included. Shared by every resolver of one ``SymbolLibrary``: that library
#: is the LTspice install, which does not change while the server runs.
StockLookups = dict[str, Path | None]


def _read(path: Path) -> tuple[SymbolFile, str]:
    """``path`` read as a symbol, and why it cannot be used ("" when it can)."""
    try:
        read = read_symbol(read_spice_text(path))
    except Exception as failed:
        return SymbolFile(), f"could not be read ({failed})"
    if read.unread_pins:
        return read, f"has a pin line that does not read ({read.unread_pins[0]!r})"
    return read, ""


class SymbolResolver:
    """Resolve a symbol name to a ``.asy`` file and read it, with caching.

    Precedence, highest first:

    1. the schematic's own directory (``local_dir``),
    2. configured project symbol paths (``project_paths``),
    3. stock LTspice library paths (``stock_paths``).

    Its search roots are fixed at construction, and what each name resolved
    to and read as, misses included, is kept for the resolver's life. A
    resolver is made for one request (``SymbolLibrary.for_sheet``), so a
    symbol added beside the sheet or to a configured folder between requests
    is found, and a rewritten ``.asy`` is read again. Where a name resolved in
    the stock library is kept by the library for the session
    (``stock_lookups``). A file read is kept by its path and content stamp, in
    the library's cache when the resolver comes from one (``parse_cache``) and
    in the resolver's own otherwise.
    """

    def __init__(
        self,
        local_dir: Path | None,
        project_paths: Sequence[Path] = (),
        stock_paths: Sequence[Path] = (),
        *,
        parse_cache: ParseCache | None = None,
        stock_lookups: StockLookups | None = None,
    ) -> None:
        tiers = [(local_dir, "schematic-local")] if local_dir is not None else []
        tiers += [(path, "configured") for path in project_paths]
        tiers += [(path, "stock") for path in stock_paths]
        # De-duplicate while preserving precedence order.
        seen: set[str] = set()
        precedence: list[tuple[Path, str]] = []
        for path, tier in tiers:
            if str(path) not in seen:
                seen.add(str(path))
                precedence.append((path, tier))
        #: The folders searched, in order, each with its tier:
        #: ``schematic-local``, ``configured`` or ``stock``.
        self.precedence: tuple[tuple[Path, str], ...] = tuple(precedence)
        self._local_dir = local_dir
        self._project = [path for path, tier in precedence if tier == "configured"]
        self._stock = [path for path, tier in precedence if tier == "stock"]
        self._resolved: dict[str, Path | None] = {}
        self._found: dict[str, FoundSymbol | None] = {}
        self._parse_cache: ParseCache = FileCache() if parse_cache is None else parse_cache
        # A stock folder that is also the sheet's own is searched as the sheet's,
        # so what this resolver finds in the stock library is not the library's.
        if stock_lookups is None or (
            local_dir is not None and str(local_dir) in {str(path) for path in stock_paths}
        ):
            stock_lookups = {}
        self._stock_lookups = stock_lookups

    @property
    def roots(self) -> tuple[Path, ...]:
        """The folders searched, in order: the sheet's own first, then the libraries."""
        return tuple(path for path, _tier in self.precedence)

    def resolve(self, symbol: str) -> Path | None:
        """Return the ``.asy`` path for ``symbol``, or ``None`` if unresolved.

        ``symbol_library.find_symbol`` is the rule, with the sheet's own folder
        first and every other root a library: the rule the schematic editor's
        pin geometry follows too.
        """
        if symbol not in self._resolved:
            found = find_symbol(symbol, self._local_dir, self._project)
            if found is None and self._stock:
                if symbol not in self._stock_lookups:
                    self._stock_lookups[symbol] = find_symbol(symbol, None, self._stock)
                found = self._stock_lookups[symbol]
            self._resolved[symbol] = found
        return self._resolved[symbol]

    def load(self, symbol: str) -> FoundSymbol | None:
        """Resolve and read ``symbol``; ``None`` when no file of that name is found.

        A file that is found and cannot be used comes back with its ``problem``.
        """
        if symbol not in self._found:
            path = self.resolve(symbol)
            found = None
            if path is not None:
                file, problem = self._parse_cache.get(path, _read)
                found = FoundSymbol(spellings(symbol)[1], path, file, problem)
            self._found[symbol] = found
        return self._found[symbol]


# ---------------------------------------------------------------------------
# The session's library
# ---------------------------------------------------------------------------


class SymbolLibrary:
    """The symbol libraries a session reads, and one cache of the files read from them.

    A session builds one (``SessionState.symbols``) from its configuration and
    the LTspice it runs: the folders ``[schematic] symbol_paths`` names, then
    the ``sym`` folder of that LTspice build's own library (``build_roots`` are
    the build's ``lib`` folders, which spicelib's editor is given and appends
    ``sym`` to itself). Every reader of a sheet's symbols asks it, so the edit,
    the check, the inspection and the drawing of one sheet find the same file
    for each symbol.

    ``for_sheet`` hands out a resolver with the sheet's own folder in front.
    All of them share the library's cache of files read, by file and content
    stamp, so a symbol redrawn while the session runs is read again by the
    next request, and where each name resolved in the build's library, which
    is looked up once per session.
    """

    def __init__(self, configured: Sequence[Path] = (), build_roots: Sequence[Path] = ()) -> None:
        self.configured: tuple[Path, ...] = tuple(configured)
        self.build_roots: tuple[Path, ...] = tuple(build_roots)
        self._parse_cache: ParseCache = FileCache()
        self._stock_lookups: StockLookups = {}

    @property
    def stock(self) -> tuple[Path, ...]:
        """The build's symbol folders: ``sym`` under each of its ``lib`` folders."""
        return tuple(root / "sym" for root in self.build_roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        """The libraries searched after a sheet's own folder, in order."""
        return (*self.configured, *self.stock)

    def for_sheet(self, path: Path | None) -> SymbolResolver:
        """The resolver for the sheet at ``path``: its folder first, then the libraries.

        ``path`` None is a lookup with no sheet in hand: the libraries only.
        """
        return SymbolResolver(
            local_dir=path.parent if path is not None else None,
            project_paths=self.configured,
            stock_paths=self.stock,
            parse_cache=self._parse_cache,
            stock_lookups=self._stock_lookups,
        )
