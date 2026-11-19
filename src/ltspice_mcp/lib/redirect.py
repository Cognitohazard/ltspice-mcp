"""Where a path the server builds lies: its one spelling, whether it redirects
to another location, and whether two paths are one place.

None of these compares the text of a resolved path, and none gives another
answer when a folder or file is created, removed or replaced while it asks:
on Windows, ``Path.resolve`` of a path whose folder another thread creates or
removes at that moment can keep the ``\\\\?\\`` prefix (``ntpath.realpath``
strips it only when its two lookups of the path fail the same way), and a
short 8.3 name, the case of a drive letter, a UNC path or a substituted drive
spell one folder several ways.

Imports nothing beyond the standard library, so the parser worker, which names
its files through the store, starts without anything heavier.
"""

import os
import stat
from pathlib import Path, PurePath

# Windows: a file or folder carrying a reparse point, and the bit of its tag
# that marks a reparse point standing for another name (a symbolic link, a
# junction, a volume mounted on a folder), as against one that keeps data in
# place (a cloud placeholder, a deduplicated file), which leaves the path where
# it is.
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_REPARSE_TAG_NAME_SURROGATE = 0x20000000

_PREFIX = "\\\\?\\"
_UNC_PREFIX = "\\\\?\\UNC\\"


def without_added_prefix(spelled: str, resolved: str) -> str:
    """``resolved`` without a ``\\\\?\\`` prefix that ``spelled`` did not carry,
    as ``ntpath.realpath`` returns it when no other thread changes the path
    while it asks: a drive path loses the prefix, a UNC path takes ``\\\\``
    back, and a volume GUID path, which has no other spelling, keeps it."""
    if spelled.startswith(_PREFIX) or not resolved.startswith(_PREFIX):
        return resolved
    if resolved.startswith(_UNC_PREFIX):
        return "\\\\" + resolved[len(_UNC_PREFIX) :]
    rest = resolved[len(_PREFIX) :]
    if rest[1:3] == ":\\":
        return rest
    return resolved


def canonical(path: Path) -> Path:
    """The one spelling of ``path``: absolute, with links followed and ``..``
    parts folded after them as the system folds them, and on Windows short
    names expanded for the folders that exist. Made the same way whether or
    not another thread creates, removes or replaces a part of it meanwhile:
    it never carries a ``\\\\?\\`` prefix the input did not have. A path that
    cannot be resolved (a link loop) is made absolute by its spelling.

    One case differs from ``ntpath.realpath``: it keeps the prefix on purpose
    for a path the plain spelling cannot open (longer than 260 characters on
    a system without long-path support), and this removes it, so such a path
    is spelled here in a form that cannot be opened there. A store that deep
    fails without long-path support anyway, through the unresolved paths the
    store also builds."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(path))
    return Path(without_added_prefix(os.fspath(path), os.fspath(resolved)))


def _redirects(info: os.stat_result) -> bool:
    """Whether the entry ``os.lstat`` described stands for another location."""
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    tag = getattr(info, "st_reparse_tag", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT and tag & _REPARSE_TAG_NAME_SURROGATE)


def _one(first: os.stat_result, second: os.stat_result) -> bool:
    """Whether two stats are one file or folder. A file system that gives no
    file number (0) cannot tell, so it is never taken for one."""
    return first.st_ino != 0 and os.path.samestat(first, second)


def _root_among(path: Path, root: Path) -> tuple[Path, tuple[str, ...]] | None:
    """The shallowest folder of ``path`` whose one spelling (``canonical``) is
    ``root``'s, and the parts of ``path`` below it; ``None`` when none is.
    Spellings, not identities, so a root that does not exist yet is found
    too: a link, a short name or a substituted drive above it is followed the
    same way on both sides."""
    target = canonical(root)
    for folder in [*reversed(path.parents), path]:
        if canonical(folder) == target:
            return folder, path.relative_to(folder).parts
    return None


def redirect_below(path: Path, root: Path) -> Path | None:
    """The part of ``path`` below ``root`` that redirects to another location,
    or ``None`` when nothing does. The one answer to "does a path the server is
    about to use redirect", for every path it builds under a folder it owns.

    A redirect is an existing component that is a symbolic link, or on Windows
    a reparse point that stands for another name: a junction or a volume
    mounted on a folder. A reparse point that keeps its data in place (a cloud
    placeholder, a deduplicated file) is not one. The components are looked
    at from ``root`` down, each once, and a component that does not exist
    ends the walk: nothing below it exists, so another thread creating or
    removing it at that moment changes nothing here. A component that cannot
    be looked at (access denied, a cloud provider that is not running) is
    answered with that component: what it is is not known.

    ``root`` itself, and whatever lies above it, may redirect: a store kept
    behind a link is still the store. ``path`` is taken below ``root`` by its
    spelling, and when it is spelled another way (through a link above the
    root, a short name, a substituted drive), from the shallowest of its
    folders whose one spelling is ``root``'s, whether or not ``root`` exists
    yet. A
    path with a ``..`` part, or one that does not lie under ``root``, is
    answered with ``path`` itself, as a redirect is.
    """
    if not path.is_absolute() or ".." in path.parts:
        return path
    try:
        base, tail = root, path.relative_to(root).parts
    except ValueError:
        found = _root_among(path, root)
        if found is None:
            return path
        base, tail = found
    current = base
    for part in tail:
        current = current / part
        try:
            info = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError:
            return current
        if _redirects(info):
            return current
    return None


def same_location(first: Path, second: Path) -> bool:
    """Whether two absolute paths name one location, however each is spelled.

    The same spelling is one location. Otherwise each folder of ``first`` that
    exists is compared by identity (following a link, as an open would) with
    the folder of ``second`` that has the same parts below it; the paths are
    one location when any such pair is one folder. A folder created or removed
    while this asks leaves another pair to answer, up to the drive or root
    both share, so the answer does not depend on it. A file system that gives
    no file number is compared by spelling only.
    """
    if not first.is_absolute() or not second.is_absolute():
        return False
    if first == second:
        return True
    second_parts = second.parts
    for folder in [first, *first.parents]:
        tail = first.relative_to(folder).parts
        if len(tail) >= len(second_parts):
            continue
        if tail and PurePath(*second_parts[-len(tail) :]) != PurePath(*tail):
            continue
        other = Path(*second_parts[: len(second_parts) - len(tail)])
        try:
            if _one(os.stat(folder), os.stat(other)):
                return True
        except OSError:
            continue
    return False
