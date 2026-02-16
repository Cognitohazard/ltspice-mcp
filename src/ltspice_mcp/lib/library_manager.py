"""SPICE library session management with built-in detection."""

import logging
import os
import sys
from pathlib import Path
from typing import Type

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.library_parser import LibraryIndex, ModelEntry, parse_library_file
from ltspice_mcp.lib.wsl import is_wsl

logger = logging.getLogger(__name__)


class LibraryManager:
    """Manage loaded SPICE libraries for the session.

    Provides library loading/unloading, search across user-loaded and built-in
    libraries, and model lookup with .include directive generation.
    """

    def __init__(self, available_simulators: dict[str, Type]) -> None:
        """Initialize library manager.

        Args:
            available_simulators: Dictionary of detected simulators from state
        """
        self._user_libs: FileCache[LibraryIndex] = FileCache()
        self._builtin_libs: FileCache[LibraryIndex] = FileCache()
        self._builtin_paths: list[Path] | None = None
        self._available_simulators = available_simulators

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

        Returns:
            List of .lib file paths
        """
        candidates = []

        if is_wsl():
            # Check Windows paths via /mnt/c
            # Try to find user directories
            users_dir = Path("/mnt/c/Users")
            if users_dir.exists():
                for user_path in users_dir.iterdir():
                    if user_path.is_dir():
                        # Check Documents/LTspiceXVII/lib/
                        lib_path = user_path / "Documents/LTspiceXVII/lib"
                        if lib_path.exists():
                            candidates.append(lib_path)
                        # Check AppData/Local/Programs/ADI/LTspice/lib/
                        lib_path = user_path / "AppData/Local/Programs/ADI/LTspice/lib"
                        if lib_path.exists():
                            candidates.append(lib_path)

        elif sys.platform == "win32":
            # Native Windows
            home = Path.home()
            candidates.extend(
                [
                    home / "Documents/LTspiceXVII/lib",
                    home / "AppData/Local/Programs/ADI/LTspice/lib",
                    Path("C:/Program Files/ADI/LTspice/lib"),
                ]
            )

        else:
            # Linux/Mac - check Wine installations
            wine_prefixes = [
                Path.home() / ".wine/drive_c/Program Files/ADI/LTspice/lib",
                Path.home() / ".wine/drive_c/Program Files (x86)/ADI/LTspice/lib",
            ]
            candidates.extend(wine_prefixes)

        # Find .lib files in candidate directories
        lib_files = []
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Recursively find .lib files
                for lib_file in candidate.rglob("*.lib"):
                    if lib_file.is_file():
                        lib_files.append(lib_file)
                        logger.debug(f"Found LTSpice library: {lib_file}")

        return lib_files

    def _detect_ngspice_paths(self) -> list[Path]:
        """Detect NGspice library files on current platform.

        Returns:
            List of library file paths
        """
        candidates = []

        # Check SPICE_LIB_DIR environment variable first
        if env_path := os.getenv("SPICE_LIB_DIR"):
            path = Path(env_path)
            if path.exists() and path.is_dir():
                candidates.append(path)

        # Platform-specific default paths
        if sys.platform == "win32" or is_wsl():
            # Windows/WSL - check common install locations
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

        # Linux paths (native or WSL)
        candidates.extend(
            [
                Path("/usr/share/ngspice"),
                Path("/usr/local/share/ngspice"),
                Path("/opt/ngspice/share/ngspice"),
            ]
        )

        # Find library files in candidate directories
        lib_files = []
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Look for models in lib subdirectory
                lib_dir = candidate / "lib"
                if lib_dir.exists():
                    for pattern in ["*.lib", "*.mod"]:
                        for lib_file in lib_dir.rglob(pattern):
                            if lib_file.is_file():
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
                self._user_libs._entries[lib_file] = (lib_file.stat().st_mtime, index)

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
            for cached_path in list(self._user_libs._entries.keys()):
                if cached_path.is_relative_to(path):
                    self._user_libs.invalidate(cached_path)
                    removed_count += 1

            return {"path": str(path), "removed": removed_count > 0, "warning": None}
        else:
            # Single file
            if path in self._user_libs._entries:
                self._user_libs.invalidate(path)
                return {"path": str(path), "removed": True, "warning": None}
            else:
                return {"path": str(path), "removed": False, "warning": "Library not loaded"}

    def list_libraries(self) -> list[str]:
        """List all loaded user library paths.

        Returns:
            List of library path strings
        """
        return [str(path) for path in self._user_libs._entries.keys()]

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
        for entry in self._user_libs._entries.values():
            index = entry[1]
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
                "parameters": m.parameters,
            }
            for m in page
        ]

        return {"results": results, "total": total, "offset": offset, "limit": limit}

    def search_builtin_libraries(self, query: str, offset: int = 0, limit: int = 50) -> dict:
        """Search across built-in simulator libraries.

        Triggers lazy detection of built-in libraries on first call.

        Args:
            query: Case-insensitive substring to search for
            offset: Number of results to skip
            limit: Maximum results to return

        Returns:
            Dict with results, total, offset, limit
        """
        # Trigger lazy detection
        builtin_paths = self._detect_builtin_paths()

        all_matches = []

        # Parse and search each built-in library
        for lib_path in builtin_paths:
            try:
                # Use cache with mtime invalidation
                index = self._builtin_libs.get(lib_path, lambda p: parse_library_file(p))
                matches, _ = index.search(query, offset=0, limit=999999)
                all_matches.extend(matches)
            except Exception as e:
                logger.warning(f"Failed to search built-in library {lib_path}: {e}")

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
                "parameters": m.parameters,
            }
            for m in page
        ]

        return {"results": results, "total": total, "offset": offset, "limit": limit}

    def get_model_info(self, name: str, full: bool = False) -> dict | None:
        """Get detailed model/subcircuit information.

        Searches both user-loaded and built-in libraries.

        Args:
            name: Model/subcircuit name (case-insensitive)
            full: If True, include full raw_text definition

        Returns:
            Dict with name, type, source_path, include_directive, parameters, and
            optionally raw_text. Returns None if not found.
        """
        # Search user libraries first
        for entry in self._user_libs._entries.values():
            index = entry[1]
            model = index.get_model(name)
            if model:
                return self._format_model_info(model, full)

        # Search built-in libraries
        builtin_paths = self._detect_builtin_paths()
        for lib_path in builtin_paths:
            try:
                index = self._builtin_libs.get(lib_path, lambda p: parse_library_file(p))
                model = index.get_model(name)
                if model:
                    return self._format_model_info(model, full)
            except Exception as e:
                logger.warning(f"Failed to search built-in library {lib_path}: {e}")

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
            "parameters": model.parameters,
        }

        if full:
            info["raw_text"] = model.raw_text

        return info
