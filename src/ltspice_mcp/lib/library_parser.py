"""Component library parsing utilities."""

import codecs
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# BOM probes for the encodings we care about. Order matters: UTF-32 BOMs start
# with the same two bytes as UTF-16, so check the longer ones first.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)


def _detect_utf16_endianness(probe: bytes) -> str | None:
    """Heuristic: ``"utf-16-le"`` / ``"utf-16-be"`` if every other byte is null.

    LTspice's bundled ``lib/cmp/standard.{mos,bjt,...}`` files are UTF-16 LE
    *without* a BOM in some installs. ASCII text in UTF-16 LE produces a
    null byte at every odd position; UTF-16 BE puts the null at every even
    position. Counting nulls in a small head-of-file probe is a cheap and
    reliable disambiguator that avoids false-positives on real binary blobs
    (those have mixed null distributions).
    """
    if len(probe) < 4 or len(probe) % 2:
        probe = probe[: (len(probe) // 2) * 2]
        if not probe:
            return None
    odd_nulls = sum(1 for i in range(1, len(probe), 2) if probe[i] == 0)
    even_nulls = sum(1 for i in range(0, len(probe), 2) if probe[i] == 0)
    half = len(probe) // 2
    # A high concentration of nulls on one side and few on the other is
    # the signature of UTF-16 ASCII text.
    if odd_nulls > 0.8 * half and even_nulls < 0.2 * half:
        return "utf-16-le"
    if even_nulls > 0.8 * half and odd_nulls < 0.2 * half:
        return "utf-16-be"
    return None


def _read_library_text(path: Path) -> str:
    """Read a SPICE library file with encoding auto-detection.

    LTspice's bundled ``lib/cmp/standard.{mos,bjt,...}`` files are UTF-16 LE
    — with a BOM in some installs and **without** in others. Earlier versions
    of this parser assumed UTF-8 (the platform default for ``Path.read_text``)
    and silently produced empty model lists for those files — so
    ``find_model(include_builtin=True)`` couldn't find any of LTspice's
    stock parts.

    Resolution order:

    1. BOM sniff (UTF-32 LE/BE, UTF-16 LE/BE, UTF-8 with BOM).
    2. Heuristic null-byte scan for UTF-16 LE/BE without BOM (the
       LTspice 26+ ``standard.bjt``/``standard.mos`` shape).
    3. UTF-8 with ``errors="replace"`` as the catch-all for ASCII /
       most third-party ``.lib`` files.
    """
    raw = path.read_bytes()
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(encoding, errors="replace")
    # Probe the first 256 bytes for a UTF-16 LE/BE pattern without a BOM.
    encoding = _detect_utf16_endianness(raw[:256])
    if encoding is not None:
        return raw.decode(encoding, errors="replace")
    return raw.decode("utf-8", errors="replace")


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

    Args:
        path: Path to library file (.lib, .mod, etc.)

    Returns:
        LibraryIndex with all parsed models and subcircuits

    Raises:
        OSError: If file cannot be read
    """
    try:
        content = _read_library_text(path)
    except OSError as e:
        logger.error(f"Failed to read library file {path}: {e}")
        raise

    lines = content.split("\n")
    merged = _merge_continuation_lines(lines)

    models = []

    # Regex patterns.
    # Model type is `[^\s(]+` (not `\S+`) so it does not greedily swallow the
    # opening paren when there is no space before it: `.MODEL Q NPN(BF=200)`
    # parses correctly as type=NPN, params=BF=200.
    model_pattern = re.compile(
        r"^\s*\.MODEL\s+(\S+)\s+([^\s(]+)\s*(?:\((.*?)\))?",
        re.IGNORECASE,
    )
    subckt_pattern = re.compile(r"^\s*\.SUBCKT\s+(\S+)", re.IGNORECASE)
    ends_pattern = re.compile(r"^\s*\.ENDS", re.IGNORECASE)

    i = 0
    while i < len(merged):
        line = merged[i]

        model_match = model_pattern.match(line)
        if model_match:
            name = model_match.group(1)
            param_text = model_match.group(3) or ""

            # .MODEL definitions are typically single-line (after continuation merge)
            raw_text = line
            line_count = 1

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
                logger.warning(f"Malformed .MODEL at line {i + 1} in {path}: {e}")

            i += 1
            continue

        subckt_match = subckt_pattern.match(line)
        if subckt_match:
            name = subckt_match.group(1)
            start_line = i
            raw_lines = [line]

            # Find matching .ENDS, tracking nesting depth so an inner .SUBCKT
            # / .ENDS pair doesn't accidentally terminate the outer one.
            i += 1
            depth = 1
            found_ends = False
            while i < len(merged):
                current_line = merged[i]
                raw_lines.append(current_line)

                if subckt_pattern.match(current_line):
                    depth += 1
                elif ends_pattern.match(current_line):
                    depth -= 1
                    if depth == 0:
                        found_ends = True
                        i += 1
                        break

                i += 1

            if not found_ends:
                logger.warning(
                    f"Malformed .SUBCKT {name} at line {start_line + 1} in {path}: missing .ENDS"
                )
                continue

            raw_text = "\n".join(raw_lines)
            line_count = len(raw_lines)

            # Extract node list from first line for parameters summary.
            # .SUBCKT name [node1 node2 ...] [PARAMS: key=value ...]
            # Stop at the first token that is "PARAMS:" or contains "=", since
            # those mark the start of subcircuit parameters, not nodes.
            parts = line.split()
            node_tokens: list[str] = []
            for tok in parts[2:]:
                if tok.upper() == "PARAMS:" or "=" in tok:
                    break
                node_tokens.append(tok)
                if len(node_tokens) >= 5:
                    break
            parameters = {f"node{i + 1}": node for i, node in enumerate(node_tokens)}

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
                logger.warning(f"Malformed .SUBCKT at line {start_line + 1} in {path}: {e}")

            continue

        i += 1

    logger.info(f"Parsed {len(models)} models/subcircuits from {path}")
    return LibraryIndex(path=path, models=models)
