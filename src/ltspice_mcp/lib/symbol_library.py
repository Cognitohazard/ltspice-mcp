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

Every part of a name, folders included, is matched without regard to letter
case, on every platform, as Windows matches it, so a sheet drawn there for
``Amp.asy`` and naming ``amp`` finds the same file on Linux. Where one folder
holds two names that differ only in case, the one spelled exactly as the sheet
spells it wins; where neither is, the name is ambiguous (``AmbiguousName``),
and the reply names every file it could mean. No recording shows which file
LTspice takes then.

A hierarchical block's own sheet is found beside the block's symbol, else
beside the sheet being exported, and nowhere else, as both builds find it
(``export-hierarchical-block``, ``block-sheet-search``; ``find_block_sheet``).
No reader of a sheet here needs the model library a part's symbol names (a
deck reads its libraries when it runs), so none looks for one.

``SymbolResolver`` finds and reads the files of one request with that rule,
and ``SymbolLibrary`` is a session's set of libraries and its cache of files
read.
"""

from __future__ import annotations

import functools
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


class AmbiguousName(FileNotFoundError):
    """A name that, without regard to letter case, matches more than one file
    or folder in one place, none of them spelled exactly as the name is.

    ``said`` names what was looked for ("Symbol 'amp'"); ``files`` are the
    matches, sorted; ``problem`` says what is wrong in words that follow the
    name, as ``FoundSymbol.problem`` does.
    """

    def __init__(self, said: str, files: Sequence[Path]) -> None:
        self.said = said
        self.files = tuple(sorted(files, key=lambda path: path.name))
        listed = ", ".join(path.name for path in self.files)
        kind = "folders" if self.files[0].is_dir() else "files"
        self.problem = (
            f"is ambiguous: {len(self.files)} {kind} differ from it only in letter case "
            f"({listed}) and none is spelled exactly so; rename all but one"
        )
        super().__init__(f"{said} {self.problem}")


def _alike(folder: Path, name: str, want_dir: bool) -> list[Path]:
    """The entries of ``folder`` named ``name`` without regard to letter case,
    folders or files as ``want_dir`` says."""
    key = name.casefold()
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    found = [folder / entry for entry in names if entry.casefold() == key]
    return [path for path in found if (path.is_dir() if want_dir else path.is_file())]


def _entry(folder: Path, name: str, want_dir: bool, said: str) -> Path | None:
    """The entry of ``folder`` that ``name`` means, as a file system that
    ignores letter case finds it, the way Windows, where LTspice runs, does.

    The entry spelled as ``name`` wins; else the one entry spelled otherwise.
    Several spelled otherwise make the name ambiguous (``AmbiguousName``,
    reported for ``said``, what was looked for): a file system that
    keeps them apart holds no answer to which one LTspice would read.
    """
    exact = folder / name
    if exact.is_dir() if want_dir else exact.is_file():
        return exact
    found = _alike(folder, name, want_dir)
    if len(found) > 1:
        raise AmbiguousName(said, found)
    return found[0] if found else None


def _walk(folder: Path, relative: str, suffix: str, said: str) -> Path | None:
    """``relative`` (forward slashes) with ``suffix`` under ``folder``, each
    part of it matched as ``_entry`` matches a name."""
    *folders, last = relative.split("/")
    for part in folders:
        found = _entry(folder, part, True, said)
        if found is None:
            return None
        folder = found
    return _entry(folder, f"{last}{suffix}", False, said)


def find_beside(folder: Path, symbol: str, suffix: str = ".asy") -> Path | None:
    """``symbol``'s file in ``folder``: where its name says, else under its bare name there.

    Never anywhere deeper. This is the whole rule beside a sheet, and the first
    half of the rule in a library. ``suffix`` is the file's. Letter case is
    matched as ``_entry`` says, and a name it finds ambiguous raises
    ``AmbiguousName``.
    """
    for name in dict.fromkeys(spellings(symbol)):
        found = _walk(folder, name, suffix, f"Symbol '{symbol}'") if name else None
        if found is not None:
            return found
    return None


def find_in_library(root: Path, symbol: str, suffix: str = ".asy") -> Path | None:
    """``symbol``'s file in the library at ``root``: as ``find_beside``, else under
    its bare name in the first folder of the library that has it (``_entry``)."""
    found = find_beside(root, symbol, suffix)
    if found is not None:
        return found
    _relative, bare = spellings(symbol)
    if not bare:
        return None
    target = f"{bare}{suffix}"
    key = target.casefold()
    for folder, _folders, files in os.walk(root):
        if target in files:
            return Path(folder, target)
        found_here = [Path(folder, name) for name in files if name.casefold() == key]
        if len(found_here) > 1:
            raise AmbiguousName(f"Symbol '{symbol}'", found_here)
        if found_here:
            return found_here[0]
    return None


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


def find_block_sheet(symbol: Path, exported_folder: Path | None) -> Path | None:
    """A hierarchical block's own sheet, given its symbol file and the folder
    of the sheet being exported: beside the symbol, else in that folder, as a
    file of the symbol's name with ``.asc`` for ``.asy``.

    Both LTspice builds look there and nowhere else (``block-sheet-search``).
    A sheet beside the exported sheet is read for a block whose symbol is in a
    library, and the one beside the symbol wins when both are there. A sheet
    elsewhere in a library, or in a folder of one, is not found, and nor is
    one beside a block's own sheet that places another block, where that
    sheet is not the one exported; the export then fails. Those cases used the
    symbol search path; a block in the build's own ``sym`` folder is not
    recorded, and is read the same way. The recorder exports with the working
    directory in the exported sheet's folder, so the two are not told apart.

    The sheet's name is matched without regard to letter case, as ``_entry``
    matches it.
    """
    folders = [symbol.parent] if exported_folder is None else [symbol.parent, exported_folder]
    name = f"{symbol.stem}.asc"
    said = f"The sheet '{name}' of block '{symbol.stem}'"
    return next(
        (found for folder in folders if (found := _entry(folder, name, False, said)) is not None),
        None,
    )


def block_sheet_named(folder: Path, name: str) -> tuple[Path, ...]:
    """The sheets in ``folder`` that a block named ``name`` may have as its own:
    the one ``find_block_sheet`` takes there, or, where the name is ambiguous,
    each one it could mean. What a copy of the folder must hold for LTspice to
    read the block's sheet from it."""
    try:
        found = _entry(folder, f"{name}.asc", False, f"The sheet '{name}.asc'")
    except AmbiguousName as ambiguous:
        return ambiguous.files
    return () if found is None else (found,)


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

#: Where each symbol name resolved in an LTspice build's own library, misses
#: included. Shared by every resolver of one
#: ``SymbolLibrary``: that library is the LTspice install, which does not
#: change while the server runs.
StockLookups = dict[str, Path | AmbiguousName | None]


def _answer(search: Callable[[], Path | None]) -> Path | AmbiguousName | None:
    """``search``'s answer, an ``AmbiguousName`` it raises kept as the answer."""
    try:
        return search()
    except AmbiguousName as ambiguous:
        return ambiguous


def _given(answer: Path | AmbiguousName | None) -> Path | None:
    """A kept answer as the search gave it: an ``AmbiguousName`` raised again."""
    if isinstance(answer, AmbiguousName):
        raise AmbiguousName(answer.said, answer.files)
    return answer


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

    1. the schematic's own directory (``local_dir``), and for a block's own
       sheet the directory of the sheet being exported (``exported_dir``),
       where both builds found a symbol that sheet places
       (``block-sheet-search``, ``blocks/nested_sheet_beside_its_symbol``),
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
        exported_dir: Path | None = None,
    ) -> None:
        self._local_dir = local_dir
        # The folder of the sheet being exported, where a block's own sheet is
        # looked for after beside its symbol (``find_block_sheet``): the
        # sheet's own folder, unless this resolver reads a block's sheet that
        # sheet loads (``for_sheet``).
        self._exported_dir = local_dir if exported_dir is None else exported_dir
        self._given = (tuple(project_paths), tuple(stock_paths))
        self._resolved: dict[tuple[str, str], Path | AmbiguousName | None] = {}
        self._found: dict[str, FoundSymbol | None] = {}
        self._parse_cache: ParseCache = FileCache() if parse_cache is None else parse_cache
        self._session_lookups = stock_lookups

    @functools.cached_property
    def precedence(self) -> tuple[tuple[Path, str], ...]:
        """The folders searched, in order, each with its tier:
        ``schematic-local``, ``configured`` or ``stock``."""
        project, stock = self._given
        tiers = [
            (folder, "schematic-local")
            # The same folder given twice is one tier, and resolved once.
            for folder in dict.fromkeys((self._local_dir, self._exported_dir))
            if folder is not None
        ]
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
    def _beside(self) -> list[Path]:
        return [path for path, tier in self.precedence if tier == "schematic-local"]

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
        if self._session_lookups is None or any(
            _folder_key(folder) in stock for folder in self._beside
        ):
            return {}
        return self._session_lookups

    @property
    def roots(self) -> tuple[Path, ...]:
        """The folders searched, in order: the sheet's own first, then the libraries."""
        return tuple(path for path, _tier in self.precedence)

    def for_sheet(self, path: Path) -> SymbolResolver:
        """A resolver for the sheet at ``path``, with this one's libraries,
        caches and exported sheet: what a sheet another loads, such as a
        block's own, is read with."""
        return SymbolResolver(
            local_dir=path.parent,
            exported_dir=self._exported_dir,
            project_paths=self._given[0],
            stock_paths=self._given[1],
            parse_cache=self._parse_cache,
            stock_lookups=self._session_lookups,
        )

    def _lookup(self, name: str) -> Path | None:
        """``find_symbol`` for ``name`` beside the sheet, then beside the sheet
        being exported, then in the configured libraries, then in the stock
        library, each answer kept as the class says. A name found ambiguous
        raises ``AmbiguousName``, every time it is asked."""
        key = (".asy", name)
        if key not in self._resolved:
            self._resolved[key] = _answer(lambda: self._search(name))
        return _given(self._resolved[key])

    def _search(self, name: str) -> Path | None:
        # A block's own sheet searches its own folder first, alone; then one
        # search of the last folder beside a sheet and the libraries, which is
        # the whole search for a sheet that is not a block's.
        *first, last = self._beside or [None]
        found = next(
            (path for folder in first if (path := find_symbol(name, folder, ())) is not None),
            None,
        )
        if found is None:
            found = find_symbol(name, last, self._project)
        if found is None and self._stock:
            if name not in self._stock_lookups:
                self._stock_lookups[name] = _answer(lambda: find_symbol(name, None, self._stock))
            found = _given(self._stock_lookups[name])
        return found

    def resolve(self, symbol: str) -> Path | None:
        """Return the ``.asy`` path for ``symbol``, or ``None`` if unresolved.

        ``symbol_library.find_symbol`` is the rule, with the sheet's own folder
        first and every other root a library: the rule the schematic editor's
        pin geometry follows too. Raises ``AmbiguousName`` for a name that
        could mean more than one file.
        """
        return self._lookup(symbol)

    def block_sheet(self, symbol: str) -> Path | None:
        """The own sheet of the block whose symbol the sheet names ``symbol``
        (``find_block_sheet``). ``None`` for a name no symbol is found for, and
        for a block with no sheet; ``AmbiguousName`` for a symbol or a sheet
        whose name could mean more than one file."""
        key = ("block", symbol)
        if key not in self._resolved:
            placed = self.resolve(symbol)
            self._resolved[key] = (
                None
                if placed is None
                else _answer(lambda: find_block_sheet(placed, self._exported_dir))
            )
        return _given(self._resolved[key])

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

        A file that is found and cannot be used comes back with its ``problem``,
        and so does a name that could mean more than one file (``AmbiguousName``),
        with the first of them as its path and nothing read.
        """
        if symbol not in self._found:
            try:
                path = self.resolve(symbol)
            except AmbiguousName as ambiguous:
                self._found[symbol] = FoundSymbol(
                    spellings(symbol)[1], ambiguous.files[0], SymbolFile(), ambiguous.problem
                )
                return self._found[symbol]
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
