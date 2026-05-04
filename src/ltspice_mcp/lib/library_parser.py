"""Component library parsing utilities."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ltspice_mcp.lib.encoding import read_spice_text as _read_library_text

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

        matches = [model for model in self.models if query_lower in model.name_lower]
        matches.sort(key=lambda m: m.name_lower)

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

        # Skip blank lines without flushing ``current`` — a blank line
        # between a definition and its continuation (``+ ...``) must not
        # break the continuation, otherwise the merged output contains a
        # garbage fragment like " BF=200" and the real definition loses
        # its parameters.
        if not stripped:
            continue

        stripped = re.sub(r"[;$].*$", "", stripped)  # remove inline comments

        if stripped.startswith("+"):
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

    Routes through ``spice_lex`` so continuation lines, balanced
    expressions, and quoted tokens are handled correctly. Compared to
    the legacy regex implementation, this version recognises ``.MODEL``
    cards inside ``.SUBCKT`` blocks (still indexed) and avoids the
    "param body lost when the closing paren spans a line" failure mode.

    Args:
        path: Path to library file (.lib, .mod, etc.)

    Returns:
        LibraryIndex with all parsed models and subcircuits

    Raises:
        OSError: If file cannot be read
    """
    from itertools import chain, islice

    from ltspice_mcp.lib.spice_lex import find_matching_ends, lex
    from ltspice_mcp.lib.spice_lex_views import ModelCard, SubcktCard

    try:
        content = _read_library_text(path)
    except OSError as e:
        logger.error(f"Failed to read library file {path}: {e}")
        raise

    cards = lex(content).cards
    models: list[ModelEntry] = []

    for idx, c in enumerate(cards):
        if c.kind == "model":
            try:
                view = ModelCard.from_card(c)
            except Exception as e:
                logger.warning(
                    f"Malformed .MODEL at line {c.line_start} in {path}: {e}"
                )
                continue
            params = dict(islice(view.params.items(), 5))
            entry = ModelEntry(
                name=view.name,
                name_lower=view.name.lower(),
                model_type=".MODEL",
                source_path=path,
                line_start=c.line_start,
                line_count=len(c.raw_lines),
                raw_text="".join(c.raw_lines).rstrip("\n"),
                parameters=params,
            )
            models.append(entry)
            logger.debug(f"Parsed .MODEL {view.name} from {path.name}")
        elif c.kind == "subckt" and c.name:
            closer_idx = find_matching_ends(cards, idx)
            if closer_idx is None:
                logger.warning(
                    f"Malformed .SUBCKT {c.name} at line {c.line_start} in {path}: "
                    "missing .ENDS"
                )
                continue
            raw_text = "".join(
                chain.from_iterable(
                    sc.raw_lines for sc in cards[idx : closer_idx + 1]
                )
            ).rstrip("\n")
            # Parse ports / param defaults via the typed view so
            # whitespace-around-equals (``gain = 10``) is correctly
            # classified as a param default, not a port.
            try:
                subckt_view = SubcktCard.from_card(c)
                ports = subckt_view.ports[:5]
            except Exception as e:
                logger.warning(
                    f"Malformed .SUBCKT {c.name} at line {c.line_start} in {path}: {e}"
                )
                continue
            parameters = {f"node{i + 1}": node for i, node in enumerate(ports)}
            line_count = (
                cards[closer_idx].line_start
                - c.line_start
                + len(cards[closer_idx].raw_lines)
            )
            entry = ModelEntry(
                name=c.name,
                name_lower=c.name.lower(),
                model_type=".SUBCKT",
                source_path=path,
                line_start=c.line_start,
                line_count=line_count,
                raw_text=raw_text,
                parameters=parameters,
            )
            models.append(entry)
            logger.debug(f"Parsed .SUBCKT {c.name} from {path.name}")

    logger.info(f"Parsed {len(models)} models/subcircuits from {path}")
    return LibraryIndex(path=path, models=models)
