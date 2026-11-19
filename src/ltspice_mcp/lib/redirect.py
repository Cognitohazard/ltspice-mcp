"""Whether a path the server builds redirects to another location.

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


def _redirects(info: os.stat_result) -> bool:
    """Whether the entry ``os.lstat`` described stands for another location."""
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    tag = getattr(info, "st_reparse_tag", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT and tag & _REPARSE_TAG_NAME_SURROGATE)


def _is(info: os.stat_result, root: os.stat_result) -> bool:
    """Whether two stats are one directory. A file system that gives no file
    number (0) cannot tell, so it is never taken for one."""
    return info.st_ino != 0 and os.path.samestat(info, root)


def redirect_below(path: Path, root: Path) -> Path | None:
    """The part of ``path`` below ``root`` that redirects to another location,
    or ``None`` when nothing does. The one answer to "does a path the server is
    about to use redirect", for every path it builds under a folder it owns.

    A redirect is an existing component that is a symbolic link, or on Windows
    a reparse point that stands for another name: a junction or a volume
    mounted on a folder. A reparse point that keeps its data in place (a cloud
    placeholder, a deduplicated file) is not one. A component that does not
    exist yet redirects nowhere, so another thread creating or removing it at
    that moment changes nothing here. ``root`` itself may redirect; what lies
    above it is not looked at. A path with a ``..`` part, or one that does not
    lie under ``root``, is answered with ``path`` itself, as a redirect is. A
    component that cannot be looked at (access denied, a cloud provider that
    is not running) is answered with that component: what it is is not known.
    ``path`` is expected to be spelled from ``root``, as every caller builds
    it; a path reaching the same folder through another link is matched by
    identity once it reaches it.

    Nothing here compares the text of a resolved path: on Windows,
    ``Path.resolve`` of a path whose folder another thread creates or removes
    at that moment can keep the ``\\\\?\\`` prefix (``ntpath.realpath``), and a
    short 8.3 name, the case of a drive letter, a UNC path or a substituted
    drive spell one folder several ways. ``root`` is found among ``path``'s
    folders by identity (the volume and file number ``os.stat`` reads), or by
    its own spelling where the file system gives no file number.
    """
    if not path.is_absolute() or ".." in path.parts:
        return path
    try:
        root_info: os.stat_result | None = os.stat(root)
    except (FileNotFoundError, NotADirectoryError):
        # Nothing below a folder that does not exist exists either.
        root_info = None
    except OSError:
        return path
    current = path
    while current != root:
        try:
            info = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            info = None
        except OSError:
            return current
        if info is not None:
            if _redirects(info):
                return current
            if root_info is not None and _is(info, root_info):
                return None
        if current.parent == current:
            return path
        current = current.parent
    return None


def same_location(first: Path, second: Path) -> bool:
    """Whether two absolute paths name one location, however each is spelled:
    each is taken down to its deepest folder that exists, those two are
    compared by identity (following a link, as an open would), and the parts
    below them by their names. A file system that gives no file number is
    compared by spelling."""

    def anchor(path: Path) -> tuple[os.stat_result | None, PurePath]:
        tail: list[str] = []
        current = path
        while True:
            try:
                return os.stat(current), PurePath(*reversed(tail))
            except (FileNotFoundError, NotADirectoryError):
                if current.parent == current:
                    return None, PurePath(*reversed(tail))
                tail.append(current.name)
                current = current.parent

    if not first.is_absolute() or not second.is_absolute():
        return False
    try:
        first_info, first_tail = anchor(first)
        second_info, second_tail = anchor(second)
    except OSError:
        return first == second
    if first_info is None or second_info is None or first_info.st_ino == 0:
        return first == second
    return first_tail == second_tail and os.path.samestat(first_info, second_info)
