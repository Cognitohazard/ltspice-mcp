"""A stand-in Windows executable that carries a version resource.

What identifies an LTspice build before any run on it is the file version in the
executable's own version resource (``simulator_build.executable_file_version``),
which Windows reads with ``GetFileVersionInfo``. The recordings keep that version
for each recorded build (``manifest.json``, ``executable.file_version``); a test
writes it into a file of this shape, the smallest portable executable whose
resource section holds one ``RT_VERSION`` entry: a ``VS_VERSIONINFO`` block
opening with its ``VS_FIXEDFILEINFO``, as the Windows SDK's resource compiler
lays it out.
"""

from __future__ import annotations

import struct
from pathlib import Path

_SECTION_RVA = 0x1000
_SECTION_FILE_OFFSET = 0x200
_RT_VERSION = 16


def _version_info(file_version: str) -> bytes:
    parts = [int(part) for part in file_version.split(".")]
    major, minor, build, revision = [*parts, 0, 0, 0, 0][:4]
    ms, ls = (major << 16) | minor, (build << 16) | revision
    fixed = struct.pack(
        "<13I",
        0xFEEF04BD,  # signature
        0x00010000,  # structure version
        ms,
        ls,
        ms,  # product version, the same here
        ls,
        0x3F,  # flags mask
        0,  # flags
        0x00040004,  # VOS_NT_WINDOWS32
        1,  # VFT_APP
        0,
        0,
        0,
    )
    key = "VS_VERSION_INFO\0".encode("utf-16-le")
    header_len = 6 + len(key)
    padding = b"\0" * (-header_len % 4)
    length = header_len + len(padding) + len(fixed)
    return struct.pack("<HHH", length, len(fixed), 0) + key + padding + fixed


def _resource_section(data: bytes) -> bytes:
    def directory(entry_id: int, target: int) -> bytes:
        return struct.pack("<IIHHHH", 0, 0, 0, 0, 0, 1) + struct.pack("<II", entry_id, target)

    subdirectory = 0x80000000
    tree = (
        directory(_RT_VERSION, subdirectory | 0x18)
        + directory(1, subdirectory | 0x30)
        + directory(0x0409, 0x48)
        + struct.pack("<IIII", _SECTION_RVA + 0x60, len(data), 0, 0)
    )
    return tree + b"\0" * (0x60 - len(tree)) + data


def stand_in_windows_program(
    path: Path, file_version: str | None, *, overlay: bytes = b""
) -> Path:
    """Write at ``path`` a Windows executable whose version resource names
    ``file_version`` (``"17.0.37.0"``); with None, one with no version
    resource, as a build that dropped it. ``overlay`` is appended after the
    image, where a program keeps data the loader does not map."""
    resources = _resource_section(_version_info(file_version)) if file_version else b""
    raw_size = (len(resources) + 0x1FF) & ~0x1FF
    optional_size = 224  # PE32: 96 bytes, then 16 data directories
    pe = 0x40
    image = bytearray(_SECTION_FILE_OFFSET + raw_size)
    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, pe)
    image[pe : pe + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", image, pe + 4, 0x14C, 1, 0, 0, 0, optional_size, 0x0102)
    optional = pe + 24
    struct.pack_into("<H", image, optional, 0x10B)
    struct.pack_into("<I", image, optional + 92, 16)
    if resources:
        struct.pack_into("<II", image, optional + 96 + 2 * 8, _SECTION_RVA, len(resources))
    section = optional + optional_size
    image[section : section + 8] = b".rsrc\0\0\0"
    struct.pack_into(
        "<IIII",
        image,
        section + 8,
        max(len(resources), 1),
        _SECTION_RVA,
        raw_size,
        _SECTION_FILE_OFFSET,
    )
    image[_SECTION_FILE_OFFSET : _SECTION_FILE_OFFSET + len(resources)] = resources
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(image) + overlay)
    return path
