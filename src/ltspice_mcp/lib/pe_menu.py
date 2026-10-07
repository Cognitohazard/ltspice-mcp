"""The menus of a Windows program, read from the program file.

A menu command is sent to a window as the number the program gave it, and that
number is the build's own: nothing promises it stays the same from one release
to the next. The label a person reads is what a caller knows, so the number is
looked up by label in the menu resources of the executable itself. Reading the
file, not the running window, gives the build's own answer whatever state its
window is in, and works on any platform the file can be read on.
"""

from __future__ import annotations

import struct
from pathlib import Path

RT_MENU = 4


def resources(image: bytes, kind: int) -> list[bytes]:
    """The data of every resource of type ``kind`` in a PE image, in directory order.

    Raises ``ValueError`` for a file that is not a Windows executable.
    """

    def u16(at: int) -> int:
        return struct.unpack_from("<H", image, at)[0]

    def u32(at: int) -> int:
        return struct.unpack_from("<I", image, at)[0]

    header = u32(0x3C)
    if image[header : header + 4] != b"PE\0\0":
        raise ValueError("not a Windows executable")
    sections, optional_size = u16(header + 6), u16(header + 20)
    optional = header + 24
    directories = optional + (112 if u16(optional) == 0x20B else 96)
    resource_rva = u32(directories + 2 * 8)
    table = optional + optional_size
    spans = [
        (
            u32(table + 40 * i + 12),
            max(u32(table + 40 * i + 8), u32(table + 40 * i + 16)),
            u32(table + 40 * i + 20),
        )
        for i in range(sections)
    ]

    def offset(rva: int) -> int:
        for start, size, raw in spans:
            if start <= rva < start + size:
                return rva - start + raw
        raise ValueError("a resource lies outside every section of the executable")

    root = offset(resource_rva)

    def entries(directory: int) -> list[tuple[int, int]]:
        count = u16(directory + 12) + u16(directory + 14)
        return [(u32(directory + 16 + 8 * i), u32(directory + 20 + 8 * i)) for i in range(count)]

    found: list[bytes] = []
    for type_id, type_target in entries(root):
        if type_id != kind or not type_target & 0x80000000:
            continue
        for _name, name_target in entries(root + (type_target & 0x7FFFFFFF)):
            for _language, data in entries(root + (name_target & 0x7FFFFFFF)):
                entry = root + data
                start = offset(u32(entry))
                found.append(image[start : start + u32(entry + 4)])
    return found


def menu_items(template: bytes) -> list[tuple[int | None, str]]:
    """The (command id, text) of every item of a menu template; a submenu's id is None."""
    version, header = struct.unpack_from("<HH", template, 0)
    items: list[tuple[int | None, str]] = []

    def text_at(at: int) -> tuple[str, int]:
        end = at
        while template[end : end + 2] != b"\0\0":
            end += 2
        return template[at:end].decode("utf-16-le"), end + 2

    def extended(at: int) -> int:
        while True:
            at = (at + 3) & ~3
            _type, _state, command, flags = struct.unpack_from("<IIIH", template, at)
            text, at = text_at(at + 14)
            items.append((None if flags & 0x01 else command, text))
            if flags & 0x01:
                at = extended(((at + 3) & ~3) + 4)
            if flags & 0x80:
                return at

    def classic(at: int) -> int:
        while True:
            (flags,) = struct.unpack_from("<H", template, at)
            at += 2
            command = None
            if not flags & 0x10:
                (command,) = struct.unpack_from("<H", template, at)
                at += 2
            text, at = text_at(at)
            items.append((command, text))
            if flags & 0x10:
                at = classic(at)
            if flags & 0x80:
                return at

    if version == 1:
        extended(4 + header)
    else:
        classic(4)
    return items


def menu_label(text: str) -> str:
    """A menu item's text as a person names it: no accelerator, no ``&``, no trailing dots."""
    return text.split("\t", 1)[0].replace("&", "").rstrip(".").strip()


def menus(executable: Path) -> list[list[tuple[int | None, str]]]:
    """Every menu of the program at ``executable``, each as its ``menu_items``."""
    return [menu_items(template) for template in resources(executable.read_bytes(), RT_MENU)]
