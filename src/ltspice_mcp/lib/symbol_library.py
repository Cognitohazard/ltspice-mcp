"""Which file a name on a sheet means: a part's symbol, and a block's own sheet.

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
- **Nowhere else**: not in the folder the server happens to run in, which is
  no part of the sheet.

A hierarchical block's own sheet is found beside the block's symbol, where
both builds read it (``export-hierarchical-block``). Where else LTspice looks
for one is not recorded, so it is then found by the rule above, as a file of
the block's name with ``.asc`` for ``.asy`` (``SymbolResolver.block_sheet``).
No reader of a sheet here needs the model library a part's symbol names (a
deck reads its libraries when it runs), so none looks for one.

``SymbolResolver`` finds and reads the files of one request with that rule,
and ``SymbolLibrary`` is a session's set of libraries and its cache of files
read.
"""

from __future__ import annotations

import functools
import glob
import os
from collections.abc import Callable, Iterator, Sequence
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


def find_beside(folder: Path, symbol: str, suffix: str = ".asy") -> Path | None:
    """``symbol``'s file in ``folder``: where its name says, else under its bare name there.

    Never anywhere deeper. This is the whole rule beside a sheet, and the first
    half of the rule in a library. ``suffix`` is the file's: a symbol's, or a
    block's sheet's (``.asc``).
    """
    for name in dict.fromkeys(spellings(symbol)):
        candidate = folder / f"{name}{suffix}"
        if name and candidate.is_file():
            return candidate
    return None


def find_in_library(root: Path, symbol: str, suffix: str = ".asy") -> Path | None:
    """``symbol``'s file in the library at ``root``: as ``find_beside``, else under
    its bare name in any folder of the library."""
    found = find_beside(root, symbol, suffix)
    if found is not None:
        return found
    _relative, bare = spellings(symbol)
    if not bare:
        return None
    return next(root.rglob(f"{glob.escape(bare)}{suffix}"), None)


def symbol_files(folder: Path, beside_sheet: bool) -> Iterator[Path]:
    """The symbol files a bare name finds in ``folder``: beside a sheet, those
    directly in it (``find_beside``); in a library, every one under it
    (``find_in_library``). What a listing of the names to place offers."""
    found = folder.glob("*.asy") if beside_sheet else folder.rglob("*.asy")
    return (path for path in found if path.is_file())


def find_file(
    name: str, sheet_folder: Path | None, libraries: Sequence[Path], suffix: str
) -> Path | None:
    """The file of ``name`` with ``suffix``: beside the sheet, then in each library in turn.

    A file kept beside a sheet wins over a library's of the same name, as a
    symbol does in LTspice. A folder that is not there is passed over, and so
    is the sheet's own, which is never searched as a library.
    """
    if sheet_folder is not None and sheet_folder.is_dir():
        found = find_beside(sheet_folder, name, suffix)
        if found is not None:
            return found
    for root in libraries:
        if root == sheet_folder or not root.is_dir():
            continue
        found = find_in_library(root, name, suffix)
        if found is not None:
            return found
    return None


def find_symbol(symbol: str, sheet_folder: Path | None, libraries: Sequence[Path]) -> Path | None:
    """The symbol file ``symbol`` names: beside the sheet, then in each library
    in turn (``find_file``)."""
    return find_file(symbol, sheet_folder, libraries, ".asy")


def _folder_key(path: Path) -> str:
    """One spelling for every spelling of a folder: its real path, in the case
    the platform compares paths in. A link to a folder, a ``..`` in its path,
    and on Windows another case or separator, name the folder it is."""
    try:
        real = path.resolve()
    except (OSError, RuntimeError):
        real = path
    return os.path.normcase(str(real))


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
#: included, by the suffix of the file looked for (``.asy``, or ``.asc`` for a
#: block's sheet) and the name. Shared by every resolver of one
#: ``SymbolLibrary``: that library is the LTspice install, which does not
#: change while the server runs.
StockLookups = dict[tuple[str, str], Path | None]

#: A rule that finds a name's file beside a sheet and in libraries, as
#: ``find_symbol`` does.
_Finder = Callable[[str, Path | None, Sequence[Path]], Path | None]


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
    """Find the files a sheet names and read its symbols, with caching.

    Precedence, highest first:

    1. the schematic's own directory (``local_dir``),
    2. configured project symbol paths (``project_paths``),
    3. stock LTspice library paths (``stock_paths``).

    A folder given twice, under any spelling (``_folder_key``), is searched
    where it first comes, so the sheet's own folder named as a library too is
    still searched only as the sheet's. The folders are worked out at the
    first lookup, off the event loop where the caller makes that one there.

    What each name resolved to and read as, misses included, is kept for the
    resolver's life. A resolver is made for one request
    (``SymbolLibrary.for_sheet``), so a symbol added beside the sheet or to a
    configured folder between requests is found, and a rewritten ``.asy`` is
    read again. Where a name resolved in the stock library is kept by the
    library for the session (``stock_lookups``). A file read is kept by its
    path and content stamp, in the library's cache when the resolver comes
    from one (``parse_cache``) and in the resolver's own otherwise.
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
        self._local_dir = local_dir
        self._given = (tuple(project_paths), tuple(stock_paths))
        self._resolved: dict[tuple[str, str], Path | None] = {}
        self._found: dict[str, FoundSymbol | None] = {}
        self._parse_cache: ParseCache = FileCache() if parse_cache is None else parse_cache
        self._session_lookups = stock_lookups

    @functools.cached_property
    def precedence(self) -> tuple[tuple[Path, str], ...]:
        """The folders searched, in order, each with its tier:
        ``schematic-local``, ``configured`` or ``stock``."""
        project, stock = self._given
        tiers = [(self._local_dir, "schematic-local")] if self._local_dir is not None else []
        tiers += [(path, "configured") for path in project]
        tiers += [(path, "stock") for path in stock]
        seen: set[str] = set()
        precedence: list[tuple[Path, str]] = []
        for path, tier in tiers:
            key = _folder_key(path)
            if key not in seen:
                seen.add(key)
                precedence.append((path, tier))
        return tuple(precedence)

    @functools.cached_property
    def _project(self) -> list[Path]:
        return [path for path, tier in self.precedence if tier == "configured"]

    @functools.cached_property
    def _stock(self) -> list[Path]:
        return [path for path, tier in self.precedence if tier == "stock"]

    @functools.cached_property
    def _stock_lookups(self) -> StockLookups:
        # A stock folder that is also the sheet's own is searched as the sheet's,
        # so what this resolver finds in the stock library is not the library's.
        stock = {_folder_key(path) for path in self._given[1]}
        if self._session_lookups is None or (
            self._local_dir is not None and _folder_key(self._local_dir) in stock
        ):
            return {}
        return self._session_lookups

    @property
    def roots(self) -> tuple[Path, ...]:
        """The folders searched, in order: the sheet's own first, then the libraries."""
        return tuple(path for path, _tier in self.precedence)

    def for_sheet(self, path: Path) -> SymbolResolver:
        """A resolver for the sheet at ``path``, with this one's libraries and
        caches: what a sheet another loads, such as a block's own, is read with."""
        return SymbolResolver(
            local_dir=path.parent,
            project_paths=self._given[0],
            stock_paths=self._given[1],
            parse_cache=self._parse_cache,
            stock_lookups=self._session_lookups,
        )

    def _lookup(self, suffix: str, name: str, find: _Finder) -> Path | None:
        """``find`` for ``name`` beside the sheet and in the configured libraries,
        then in the stock library, each answer kept as the class says."""
        key = (suffix, name)
        if key not in self._resolved:
            found = find(name, self._local_dir, self._project)
            if found is None and self._stock:
                if key not in self._stock_lookups:
                    self._stock_lookups[key] = find(name, None, self._stock)
                found = self._stock_lookups[key]
            self._resolved[key] = found
        return self._resolved[key]

    def resolve(self, symbol: str) -> Path | None:
        """Return the ``.asy`` path for ``symbol``, or ``None`` if unresolved.

        ``symbol_library.find_symbol`` is the rule, with the sheet's own folder
        first and every other root a library: the rule the schematic editor's
        pin geometry follows too.
        """
        return self._lookup(".asy", symbol, find_symbol)

    def block_sheet(self, symbol: str) -> Path | None:
        """The own sheet of the block whose symbol the sheet names ``symbol``.

        Beside the block's symbol, where both builds read it
        (``export-hierarchical-block``). Where else LTspice looks is not
        recorded, so the sheet is then found as the symbol is, as a file of the
        block's name with ``.asc`` for ``.asy`` (``find_file``). ``None`` for a
        name no symbol is found for, and for a block with no sheet.
        """
        key = ("block", symbol)
        if key not in self._resolved:
            placed = self.resolve(symbol)
            found = None
            if placed is not None:
                beside = placed.with_suffix(".asc")
                found = (
                    beside
                    if beside.is_file()
                    else self._lookup(".asc", symbol, functools.partial(find_file, suffix=".asc"))
                )
            self._resolved[key] = found
        return self._resolved[key]

    def listing(self) -> Iterator[tuple[Path, str, Iterator[Path]]]:
        """Each folder searched, in order, with its tier and the symbol files in
        it a bare name finds (``symbol_files``): those right in the sheet's own
        folder, every one under a library. What a listing of the names to place
        offers. A folder that is not there offers none."""
        for folder, tier in self.precedence:
            files = symbol_files(folder, beside_sheet=tier == "schematic-local")
            yield folder, tier, files if folder.is_dir() else iter(())

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
    the check, the inspection, the listing and the drawing of one sheet find
    the same file for each symbol, and the editor finds a block's own sheet
    with it too, wherever the server runs.

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
