"""Zero-boot catalogue access: ``python -m ltspice_mcp.api reference [OP]``.

Prints the same catalogue ``Api.reference()`` serves, without booting the
engine — the catalogue module is stdlib+pydantic only, and the package's
lazy ``__init__`` keeps scipy/mcp out of the interpreter entirely.
"""

import sys


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "reference" or len(argv) > 2:
        print("usage: python -m ltspice_mcp.api reference [OPERATION]", file=sys.stderr)
        return 2
    from ltspice_mcp.api import _reference

    try:
        print(_reference.reference(argv[1] if len(argv) == 2 else None))
    except ValueError as exc:  # unknown operation: relay the name list
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
