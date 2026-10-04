"""Checks shared by the tests that read prose: docs, the guide, hints, prompts.

They test what a text states, not how it is worded: whole-word names, and
phrases compared with case and line wrapping ignored.
"""

from __future__ import annotations

import re


def names(text: str, word: str) -> bool:
    """``text`` mentions ``word`` as a whole word."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def flat(text: str) -> str:
    """Lower-cased, with line wrapping and repeated spaces collapsed."""
    return " ".join(text.split()).lower()


def says(text: str, *facts: str) -> bool:
    """``text`` states every fact, ignoring case and line wrapping."""
    flat_text = flat(text)
    return all(flat(fact) in flat_text for fact in facts)


def headings(text: str) -> list[str]:
    """Every markdown heading's title, lower-cased."""
    return [line.lstrip("#").strip().lower() for line in text.splitlines() if line.startswith("#")]


def has_heading(text: str, *words: str) -> bool:
    """Some markdown heading mentions every one of ``words``."""
    return any(all(word.lower() in heading for word in words) for heading in headings(text))


def section(text: str, topic: str) -> str:
    """The body of the first markdown section whose heading mentions ``topic``,
    up to the next heading of the same or a higher level."""
    match = re.search(rf"^(#+)[^\n]*{re.escape(topic)}[^\n]*\n", text, re.IGNORECASE | re.M)
    assert match, f"no section heading mentions {topic!r}"
    end = re.compile(rf"^#{{1,{len(match.group(1))}}}\s", re.M).search(text, match.end())
    return text[match.end() : end.start() if end else len(text)]
