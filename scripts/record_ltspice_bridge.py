"""Maintainer entrypoint for the LTspice bridge recorder shipped with tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.ltspice_bridge_recorder import main

if __name__ == "__main__":
    raise SystemExit(main())
