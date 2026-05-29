"""SPICE library session management with built-in detection."""

import logging
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from rapidfuzz import fuzz

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.library_parser import LibraryIndex, ModelEntry, parse_library_file
from ltspice_mcp.lib.wsl import is_wsl

logger = logging.getLogger(__name__)


_WORD_TOK = re.compile(r"[A-Za-z]+|[0-9]+")


def _part_aware_score(query_lower: str, candidate_lower: str) -> float:
    """Similarity in [0.0, 1.0] biased for part-number-style names.

    Base is ``rapidfuzz.fuzz.WRatio`` (handles typos, substring containment,
    and token reorderings). A small bonus applies when the first word token
    of both strings matches — e.g. 'LTC3406' / 'LTC3406A' share 'ltc',
    '2N3904' / '2N3906' share '2n'. The bonus keeps near-neighbour siblings
    ranked above cross-family matches with similar edit distance.
    """
    base = fuzz.WRatio(query_lower, candidate_lower) / 100.0
    q_toks = _WORD_TOK.findall(query_lower)
    c_toks = _WORD_TOK.findall(candidate_lower)
    if q_toks and c_toks and q_toks[0] == c_toks[0]:
        base = min(1.0, base + 0.05)
    return base


class LibraryManager:
    """Manage loaded SPICE libraries for the session.

    Provides library loading/unloading, search across user-loaded and built-in
    libraries, and model lookup with .include directive generation.
    """

    def __init__(self, available_simulators: dict[str, type]) -> None:
        """Initialize library manager.

        Args:
            available_simulators: Dictionary of detected simulators from state
        """
        self._user_libs: FileCache[LibraryIndex] = FileCache()
        self._builtin_libs: FileCache[LibraryIndex] = FileCache()
        self._builtin_paths: list[Path] | None = None
        self._available_simulators = available_simulators

    def __len__(self) -> int:
        """Return number of loaded user libraries."""
        return len(self._user_libs)

    def _detect_builtin_paths(self) -> list[Path]:
        """Detect built-in library directories for available simulators.

        Returns:
            List of library file paths found in built-in directories
        """
        if self._builtin_paths is not None:
            return self._builtin_paths

        all_lib_files = []

        # Detect LTSpice libraries
        if "ltspice" in self._available_simulators:
            ltspice_files = self._detect_ltspice_paths()
            all_lib_files.extend(ltspice_files)
            if ltspice_files:
                logger.info(f"Found {len(ltspice_files)} LTSpice library files")

        # Detect NGspice libraries
        if "ngspice" in self._available_simulators:
            ngspice_files = self._detect_ngspice_paths()
            all_lib_files.extend(ngspice_files)
            if ngspice_files:
                logger.info(f"Found {len(ngspice_files)} NGspice library files")

        if not all_lib_files:
            logger.debug("No built-in libraries found")

        self._builtin_paths = all_lib_files
        return all_lib_files

    def _detect_ltspice_paths(self) -> list[Path]:
        """Detect LTSpice library files on current platform.

        LTspice's stock parts live in ``lib/cmp/standard.{bjt,mos,dio,cap,ind,...}``
        — those files do NOT have a ``.lib`` extension. We accept them by
        suffix list rather than only ``*.lib`` so ``find_model(include_builtin=
        True)`` actually surfaces ``2N3904`` etc.

        We also probe both the legacy ``LTspiceXVII`` install paths and the
        modern ADI LTspice 26+ paths (``%LOCALAPPDATA%/LTspice/lib`` plus the
        system-wide ``Program Files/ADI/LTspice/lib`` directory).
        """
        candidates: list[Path] = []

        if is_wsl():
            users_dir = Path("/mnt/c/Users")
            if users_dir.exists():
                for user_path in users_dir.iterdir():
                    if not user_path.is_dir():
                        continue
                    for rel in (
                        "Documents/LTspiceXVII/lib",
                        "AppData/Local/Programs/ADI/LTspice/lib",
                        "AppData/Local/LTspice/lib",  # ADI LTspice 26+ user
                    ):
                        lp = user_path / rel
                        if lp.exists():
                            candidates.append(lp)
            # System-wide install (ADI LTspice 26+)
            for sys_path in (
                Path("/mnt/c/Program Files/ADI/LTspice/lib"),
                Path("/mnt/c/Program Files (x86)/ADI/LTspice/lib"),
            ):
                if sys_path.exists():
                    candidates.append(sys_path)

        elif sys.platform == "win32":
            home = Path.home()
            candidates.extend(
                [
                    home / "Documents/LTspiceXVII/lib",
                    home / "AppData/Local/Programs/ADI/LTspice/lib",
                    home / "AppData/Local/LTspice/lib",
                    Path("C:/Program Files/ADI/LTspice/lib"),
                    Path("C:/Program Files (x86)/ADI/LTspice/lib"),
                ]
            )

        else:
            wine_prefixes = [
                Path.home() / ".wine/drive_c/Program Files/ADI/LTspice/lib",
                Path.home() / ".wine/drive_c/Program Files (x86)/ADI/LTspice/lib",
            ]
            candidates.extend(wine_prefixes)

        # Suffixes we treat as SPICE library files. ``standard.bjt`` /
        # ``standard.mos`` are bundled stock parts; the ``.lib`` / ``.mod``
        # extensions cover third-party packs that LTspice users drop into
        # the same ``sub`` / ``cmp`` directories.
        accepted_suffixes = {
            ".lib",
            ".mod",
            ".bjt",
            ".mos",
            ".dio",
            ".cap",
            ".ind",
            ".res",
            ".jft",
            ".bead",
        }
        lib_files: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if not (candidate.exists() and candidate.is_dir()):
                continue
            for f in candidate.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in accepted_suffixes:
                    continue
                if f in seen:
                    continue
                seen.add(f)
                lib_files.append(f)
                logger.debug("Found LTSpice library: %s", f)

        return lib_files

    def _detect_ngspice_paths(self) -> list[Path]:
        """Detect NGspice library files on current platform.

        Returns:
            List of library file paths
        """
        candidates = []

        if env_path := os.getenv("SPICE_LIB_DIR"):
            path = Path(env_path)
            if path.exists() and path.is_dir():
                candidates.append(path)

        if sys.platform == "win32" or is_wsl():
            if is_wsl():
                candidates.extend(
                    [
                        Path("/mnt/c/Spice/share/ngspice"),
                        Path("/mnt/c/Program Files/ngspice/share/ngspice"),
                    ]
                )
            else:
                candidates.extend(
                    [
                        Path("C:/Spice/share/ngspice"),
                        Path("C:/Program Files/ngspice/share/ngspice"),
                    ]
                )

        candidates.extend(
            [
                Path("/usr/share/ngspice"),
                Path("/usr/local/share/ngspice"),
                Path("/opt/ngspice/share/ngspice"),
                # Debian/Ubuntu ship the example model libraries here, with no
                # share/ngspice/lib subdir (the old code required one and so
                # found nothing on a stock apt install).
                Path("/usr/share/doc/ngspice/examples"),
                Path("/usr/local/share/doc/ngspice/examples"),
            ]
        )

        lib_files: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if not (candidate.exists() and candidate.is_dir()):
                continue
            # Prefer a conventional <candidate>/lib subdir, but fall back to
            # scanning the candidate tree directly — the Debian package layout
            # has no /lib subdir and keeps .lib/.mod files under examples/**.
            scan_root = candidate / "lib"
            if not scan_root.is_dir():
                scan_root = candidate
            for pattern in ["*.lib", "*.mod"]:
                for lib_file in scan_root.rglob(pattern):
                    if lib_file.is_file() and lib_file not in seen:
                        seen.add(lib_file)
                        lib_files.append(lib_file)
                        logger.debug(f"Found NGspice library: {lib_file}")

        return lib_files

    def load_library(self, path: Path) -> dict:
        """Load a library file or directory of library files.

        Args:
            path: Path to .lib file or directory containing .lib files

        Returns:
            Summary dict with path, files_loaded, models, subcircuits counts

        Raises:
            LibraryError: If path doesn't exist or no valid library files found
        """
        if not path.exists():
            raise LibraryError(f"Library path does not exist: {path}")

        files_to_load = []

        if path.is_file():
            files_to_load.append(path)
        elif path.is_dir():
            # Scan recursively for .lib and .mod files
            for pattern in ["*.lib", "*.mod"]:
                files_to_load.extend(path.rglob(pattern))
        else:
            raise LibraryError(f"Library path is not a file or directory: {path}")

        if not files_to_load:
            raise LibraryError(f"No library files found in {path}")

        total_models = 0
        total_subcircuits = 0

        for lib_file in files_to_load:
            try:
                index = parse_library_file(lib_file)
                # Store in cache
                self._user_libs.set(lib_file, index)

                # Count models vs subcircuits
                for model in index.models:
                    if model.model_type == ".MODEL":
                        total_models += 1
                    else:
                        total_subcircuits += 1

                logger.info(f"Loaded library: {lib_file} ({len(index.models)} entries)")
            except Exception as e:
                logger.warning(f"Failed to parse library file {lib_file}: {e}")

        if total_models == 0 and total_subcircuits == 0:
            raise LibraryError(f"No valid models or subcircuits found in {path}")

        return {
            "path": str(path),
            "files_loaded": len(files_to_load),
            "models": total_models,
            "subcircuits": total_subcircuits,
        }

    def unload_library(self, path: Path) -> dict:
        """Remove a library from the session.

        Args:
            path: Library path to unload

        Returns:
            Dict with path, removed status, and optional warning
        """
        # If it's a directory, remove all files under it
        if path.is_dir():
            removed_count = 0
            for cached_path in self._user_libs.keys():  # noqa: SIM118
                if cached_path.is_relative_to(path):
                    self._user_libs.invalidate(cached_path)
                    removed_count += 1

            return {"path": str(path), "removed": removed_count > 0, "warning": None}
        else:
            # Single file
            if path in self._user_libs:
                self._user_libs.invalidate(path)
                return {"path": str(path), "removed": True, "warning": None}
            else:
                return {"path": str(path), "removed": False, "warning": "Library not loaded"}

    def get_loaded_libraries(self) -> list[tuple[Path, LibraryIndex]]:
        """Return all loaded user libraries as (path, index) pairs.

        Returns:
            List of (path, LibraryIndex) tuples for all loaded libraries
        """
        return [(path, entry[1]) for path, entry in self._user_libs.items()]

    def list_libraries(self) -> list[str]:
        """List all loaded user library paths.

        Returns:
            List of library path strings
        """
        return [str(path) for path, _ in self.get_loaded_libraries()]

    def search_user_libraries(self, query: str, offset: int = 0, limit: int = 50) -> dict:
        """Search across all loaded user libraries.

        Args:
            query: Case-insensitive substring to search for
            offset: Number of results to skip
            limit: Maximum results to return

        Returns:
            Dict with results, total, offset, limit
        """
        all_matches = []

        # Search each loaded library
        for _, index in self.get_loaded_libraries():
            matches, _ = index.search(query, offset=0, limit=999999)  # Get all matches
            all_matches.extend(matches)

        # Sort all matches alphabetically
        all_matches.sort(key=lambda m: m.name_lower)

        # Apply pagination
        total = len(all_matches)
        page = all_matches[offset : offset + limit]

        # Format results
        results = [
            {
                "name": m.name,
                "type": m.model_type,
                "source_path": str(m.source_path),
                "ports": m.ports,
                "params": m.params,
            }
            for m in page
        ]

        return {"results": results, "total": total, "offset": offset, "limit": limit}

    def _iter_builtin_indexes(self) -> Iterator[LibraryIndex]:
        """Yield each built-in LibraryIndex via the mtime cache, skipping parse failures."""
        for lib_path in self._detect_builtin_paths():
            try:
                yield self._builtin_libs.get(lib_path, parse_library_file)
            except Exception as e:
                logger.warning(f"Failed to search built-in library {lib_path}: {e}")

    def find_similar_models(
        self,
        name: str,
        *,
        exact: bool = False,
        limit: int = 5,
        cutoff: float = 0.6,
        include_builtin: bool = False,
    ) -> list[dict]:
        """Return candidate matches for ``name``, each annotated with a ``score`` in [0.0, 1.0].

        With ``exact=True`` returns at most one entry (score 1.0) when the
        name matches case-insensitively. Otherwise fuzzy-ranks via
        ``_part_aware_score`` (rapidfuzz WRatio + first-word-token bonus).

        ``include_builtin=True`` lazy-parses every built-in .lib on first
        call — hundreds of ms on a full LTspice install.
        """
        if exact:
            info = self.get_model_info(name, full=False, include_builtin=include_builtin)
            if info is None:
                return []
            info["score"] = 1.0
            return [info]

        query_lower = name.lower()

        def score(entry: ModelEntry) -> float:
            return _part_aware_score(query_lower, entry.name_lower)

        candidates: list[tuple[float, ModelEntry]] = []

        def collect(index: LibraryIndex) -> None:
            for entry in index.models:
                s = score(entry)
                if s >= cutoff:
                    candidates.append((s, entry))

        for _, index in self.get_loaded_libraries():
            collect(index)
        if include_builtin:
            for index in self._iter_builtin_indexes():
                collect(index)

        candidates.sort(key=lambda pair: (-pair[0], pair[1].name_lower))

        results = []
        for s, entry in candidates[:limit]:
            info = self._format_model_info(entry, full=False)
            info["score"] = round(s, 3)
            results.append(info)
        return results

    def get_model_info(
        self, name: str, full: bool = False, include_builtin: bool = True
    ) -> dict | None:
        """Look up a model/subcircuit by exact case-insensitive name.

        Searches loaded user libraries first, then built-in libraries unless
        ``include_builtin=False``. Returns ``None`` if no exact match exists.
        """
        for _, index in self.get_loaded_libraries():
            model = index.get_model(name)
            if model:
                return self._format_model_info(model, full)

        if not include_builtin:
            return None

        for index in self._iter_builtin_indexes():
            model = index.get_model(name)
            if model:
                return self._format_model_info(model, full)

        return None

    def _format_model_info(self, model: ModelEntry, full: bool) -> dict:
        """Format ModelEntry as info dict.

        Args:
            model: ModelEntry to format
            full: Include raw_text if True

        Returns:
            Formatted model info dict
        """
        # Generate .include directive using native path
        include_directive = f".include {model.source_path}"

        info = {
            "name": model.name,
            "type": model.model_type,
            "source_path": str(model.source_path),
            "include_directive": include_directive,
            "ports": model.ports,
            "params": model.params,
        }

        if full:
            info["raw_text"] = model.raw_text

        return info
