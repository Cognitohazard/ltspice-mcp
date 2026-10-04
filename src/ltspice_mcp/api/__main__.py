"""Zero-boot reading: ``python -m ltspice_mcp.api reference [OP]`` and ``guide [SECTION]``.

Prints the same catalogue ``Api.reference()`` serves and the same guide
``Api.guide()`` serves, without booting the engine — the catalogue module is
stdlib+pydantic only, the guide module stdlib only, and the package's lazy
``__init__`` keeps scipy/mcp out of the interpreter entirely.
"""

import sys

_USAGE = "usage: python -m ltspice_mcp.api reference [OPERATION] | guide [SECTION]"


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("reference", "guide") or len(argv) > 2:
        print(_USAGE, file=sys.stderr)
        return 2
    name = argv[1] if len(argv) == 2 else None
    try:
        if argv[0] == "guide":
            from ltspice_mcp.lib import guide

            text = guide.read(name)
        else:
            from ltspice_mcp.api import _reference

            text = _reference.reference(name)
    except ValueError as exc:  # unknown name: relay the list of known ones
        print(str(exc), file=sys.stderr)
        return 2
    try:
        print(text)
    except UnicodeEncodeError:
        # A Windows pipe encodes with the ANSI code page, which has no Γ or →;
        # the text goes out as UTF-8 rather than not at all.
        sys.stdout.flush()
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
