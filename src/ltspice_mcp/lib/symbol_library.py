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


#: Symbol files as read, by file, content stamp and the bare name they were
#: asked for under. Shared by every resolver of one ``SymbolLibrary``.
ParseCache = dict[tuple[str, tuple[int, int], str], FoundSymbol]


class SymbolResolver:
    """Resolve a symbol name to a ``.asy`` file and read it, with caching.

    Precedence, highest first:

    1. the schematic's own directory (``local_dir``),
    2. configured project symbol paths (``project_paths``),
    3. stock LTspice library paths (``stock_paths``).

    Its search roots are fixed at construction. Where each name resolved,
    misses included, is kept for the resolver's life, so a repeated miss does
    not walk the stock tree again; a resolver is made for one request
    (``SymbolLibrary.for_sheet``), so a symbol added between requests is found.
    A file read is kept by its path, content stamp and the bare name asked
    for, so a rewritten ``.asy`` is read again. That cache is the library's
    when the resolver comes from one (``parse_cache``), and the resolver's own
    otherwise.
    """

    def __init__(
        self,
        local_dir: Path | None,
        project_paths: Sequence[Path] = (),
        stock_paths: Sequence[Path] = (),
        *,
        parse_cache: ParseCache | None = None,
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
        self._parse_cache: ParseCache = {} if parse_cache is None else parse_cache

    @property
    def active_set(self) -> frozenset[str]:
        return self._active_set

    @property
    def roots(self) -> tuple[Path, ...]:
        """The folders searched, in order: the sheet's own first, then the libraries."""
        return tuple(self._roots)

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
        """Resolve and read ``symbol``; ``None`` when no file of that name is found.

        A file that is found and cannot be used comes back with its ``problem``.
        """
        path = self.resolve(symbol)
        if path is None:
            return None
        try:
            stamp = file_stamp(path)
        except OSError:
            return None
        name = spellings(symbol)[1]
        key = (str(path), stamp, name)
        cached = self._parse_cache.get(key)
        if cached is not None:
            return cached
        try:
            read = read_symbol(read_spice_text(path))
        except Exception as failed:
            found = FoundSymbol(name, path, SymbolFile(), f"could not be read ({failed})")
        else:
            problem = (
                f"has a pin line that does not read ({read.unread_pins[0]!r})"
                if read.unread_pins
                else ""
            )
            found = FoundSymbol(name, path, read, problem)
        self._parse_cache[key] = found
        return found


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
    Each resolver keeps where names resolved only for its own life, and all of
    them share the library's cache of files read, keyed by file and content
    stamp, so a symbol redrawn or added while the session runs is seen by the
    next request.
    """

    def __init__(
        self,
        configured: Sequence[Path] = (),
        build_roots: Sequence[Path] = (),
        *,
        build: str | None = None,
    ) -> None:
        self.configured: tuple[Path, ...] = tuple(configured)
        self.build_roots: tuple[Path, ...] = tuple(build_roots)
        #: Which LTspice the library is: ``"current"`` (LTspice 24 and later),
        #: ``"xvii"``, or None when no build could be told.
        self.build = build
        self._parse_cache: ParseCache = {}

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
        )

    def precedence(self, sheet_folder: Path | None) -> list[tuple[Path, str]]:
        """The folders a lookup searches, in order, each with its tier:
        ``schematic-local``, ``configured`` or ``stock``."""
        ordered: list[tuple[Path, str]] = []
        if sheet_folder is not None:
            ordered.append((sheet_folder, "schematic-local"))
        ordered += [(path, "configured") for path in self.configured]
        ordered += [(path, "stock") for path in self.stock]
        return ordered
