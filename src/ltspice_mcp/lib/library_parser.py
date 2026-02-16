"""Component library parsing utilities."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """Immutable model or subcircuit definition metadata.

    Attributes:
        name: Original case model/subcircuit name
        name_lower: Pre-computed lowercase name for fast search
        model_type: ".MODEL" or ".SUBCKT"
        source_path: Library file path
        line_start: Line number where definition starts (1-indexed)
        line_count: Number of lines in definition
        raw_text: Full SPICE definition text including continuation lines
        parameters: First 5 key parameters extracted for summary view
    """

    name: str
    name_lower: str
    model_type: str
    source_path: Path
    line_start: int
    line_count: int
    raw_text: str
    parameters: dict[str, str]


@dataclass
class LibraryIndex:
    """Parsed library file with searchable model index.

    Attributes:
        path: Library file path
        models: List of all model/subcircuit entries found
        _by_name: Lookup dict mapping lowercase name to list of entries
    """

    path: Path
    models: list[ModelEntry]
    _by_name: dict[str, list[ModelEntry]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Build lookup index after initialization."""
        for model in self.models:
            if model.name_lower not in self._by_name:
                self._by_name[model.name_lower] = []
            self._by_name[model.name_lower].append(model)

    def search(self, query: str, offset: int = 0, limit: int = 50) -> tuple[list[ModelEntry], int]:
        """Search for models by case-insensitive substring match.

        Args:
            query: Search string (case-insensitive)
            offset: Skip this many results (for pagination)
            limit: Maximum results to return

        Returns:
            (results_page, total_matches) tuple
        """
        query_lower = query.lower()

        # Find all matching models
        matches = [model for model in self.models if query_lower in model.name_lower]

        # Sort alphabetically by name (case-insensitive)
        matches.sort(key=lambda m: m.name_lower)

        # Apply pagination
        total = len(matches)
        page = matches[offset : offset + limit]

        return page, total

    def get_model(self, name: str) -> ModelEntry | None:
        """Get model by exact case-insensitive name.

        Args:
            name: Model name to find

        Returns:
            First matching ModelEntry or None if not found
        """
        entries = self._by_name.get(name.lower())
        return entries[0] if entries else None


def _merge_continuation_lines(lines: list[str]) -> list[str]:
    """Merge SPICE continuation lines (lines starting with '+').

    Args:
        lines: Raw lines from library file

    Returns:
        Merged lines with continuations resolved
    """
    merged = []
    current = None

    for line in lines:
        stripped = line.strip()

        # Skip pure comment lines (starting with '*')
        if stripped.startswith("*"):
            continue

        # Remove inline comments (';' and '$')
        stripped = re.sub(r"[;$].*$", "", stripped)

        if stripped.startswith("+"):
            # Continuation line
            if current is not None:
                current += " " + stripped[1:].strip()
        else:
            if current is not None:
                merged.append(current)
            current = stripped

    if current is not None:
        merged.append(current)

    return merged


def _extract_parameters(param_text: str, limit: int = 5) -> dict[str, str]:
    """Extract key-value parameters from SPICE parameter text.

    Args:
        param_text: Text inside parentheses for .MODEL, e.g., "BF=200 IS=1e-14 VAF=100"
        limit: Maximum number of parameters to extract

    Returns:
        Dictionary of parameter names to values (up to limit entries)
    """
    params = {}
    # Match KEY=VALUE patterns (values can include scientific notation, units, etc.)
    pattern = re.compile(r"(\w+)\s*=\s*([^\s)]+)")

    for match in pattern.finditer(param_text):
        if len(params) >= limit:
            break
        key = match.group(1).upper()  # SPICE parameters are case-insensitive
        value = match.group(2)
        params[key] = value

    return params


def parse_library_file(path: Path) -> LibraryIndex:
    """Parse SPICE library file and extract .MODEL and .SUBCKT definitions.

    Args:
        path: Path to library file (.lib, .mod, etc.)

    Returns:
        LibraryIndex with all parsed models and subcircuits

    Raises:
        OSError: If file cannot be read
    """
    # Read file with encoding error handling
    try:
        content = path.read_text(errors="replace")
    except OSError as e:
        logger.error(f"Failed to read library file {path}: {e}")
        raise

    lines = content.split("\n")
    merged = _merge_continuation_lines(lines)

    models = []

    # Regex patterns
    model_pattern = re.compile(r"^\s*\.MODEL\s+(\S+)\s+(\S+)(?:\s*\((.*?)\))?", re.IGNORECASE)
    subckt_pattern = re.compile(r"^\s*\.SUBCKT\s+(\S+)", re.IGNORECASE)
    ends_pattern = re.compile(r"^\s*\.ENDS", re.IGNORECASE)

    i = 0
    while i < len(merged):
        line = merged[i]

        # Check for .MODEL
        model_match = model_pattern.match(line)
        if model_match:
            name = model_match.group(1)
            model_type = model_match.group(2)
            param_text = model_match.group(3) or ""

            # .MODEL definitions are typically single-line (after continuation merge)
            raw_text = line
            line_count = 1

            # Extract parameters for summary
            parameters = _extract_parameters(param_text)

            try:
                entry = ModelEntry(
                    name=name,
                    name_lower=name.lower(),
                    model_type=".MODEL",
                    source_path=path,
                    line_start=i + 1,  # 1-indexed
                    line_count=line_count,
                    raw_text=raw_text,
                    parameters=parameters,
                )
                models.append(entry)
                logger.debug(f"Parsed .MODEL {name} from {path.name}")
            except Exception as e:
                logger.warning(f"Malformed .MODEL at line {i+1} in {path}: {e}")

            i += 1
            continue

        # Check for .SUBCKT
        subckt_match = subckt_pattern.match(line)
        if subckt_match:
            name = subckt_match.group(1)
            start_line = i
            raw_lines = [line]

            # Find matching .ENDS
            i += 1
            found_ends = False
            while i < len(merged):
                current_line = merged[i]
                raw_lines.append(current_line)

                if ends_pattern.match(current_line):
                    found_ends = True
                    i += 1
                    break

                i += 1

            if not found_ends:
                logger.warning(f"Malformed .SUBCKT {name} at line {start_line+1} in {path}: missing .ENDS")
                continue

            raw_text = "\n".join(raw_lines)
            line_count = len(raw_lines)

            # Extract node list from first line for parameters summary
            # .SUBCKT name node1 node2 node3 ...
            parts = line.split()
            nodes = parts[2:7] if len(parts) > 2 else []  # First 5 nodes
            parameters = {f"node{i+1}": node for i, node in enumerate(nodes)}

            try:
                entry = ModelEntry(
                    name=name,
                    name_lower=name.lower(),
                    model_type=".SUBCKT",
                    source_path=path,
                    line_start=start_line + 1,  # 1-indexed
                    line_count=line_count,
                    raw_text=raw_text,
                    parameters=parameters,
                )
                models.append(entry)
                logger.debug(f"Parsed .SUBCKT {name} from {path.name}")
            except Exception as e:
                logger.warning(f"Malformed .SUBCKT at line {start_line+1} in {path}: {e}")

            continue

        i += 1

    logger.info(f"Parsed {len(models)} models/subcircuits from {path}")
    return LibraryIndex(path=path, models=models)
