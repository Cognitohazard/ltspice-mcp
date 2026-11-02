"""What the model receives for a tool result.

Two renderings, recorded on every row:

``structured``
    What Claude Code 2.1.316 does by default: a successful result with
    ``structuredContent`` reaches the model as ``JSON.stringify`` of it, plus
    any image blocks; its text blocks are dropped.
``content``
    The result's content blocks: Claude Code with its remote flag on, and any
    client that ignores ``structuredContent``.

Under both, a result with ``isError`` reaches the model as its text blocks
only, joined by newlines: Claude Code turns such a result into an error made
of that text, and ``structuredContent`` is not shown. A result over
``OVERSIZED_TOKENS`` estimated tokens is replaced by a short notice and
counted; Claude Code instead saves it to a file and hands the model the path,
which this client does not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mcp.types as mcp_types

from ltspice_eval import jsonjs

Rendering = Literal["structured", "content"]
RENDERINGS: frozenset[str] = frozenset({"structured", "content"})

#: Claude Code's limit on one tool result, in tokens.
OVERSIZED_TOKENS = 25_000


def estimated_tokens(text: str) -> int:
    """A rough size: four characters a token. Rows report it as an estimate."""
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class Rendered:
    #: The ``tool_result`` content blocks, in Messages API form.
    content: list[dict[str, Any]]
    is_error: bool
    #: The size of the text the model receives, before any oversize notice.
    tokens: int
    oversized: bool
    #: ``structuredContent`` as the server sent it, for the row's metrics.
    structured: Any


def render(result: mcp_types.CallToolResult, rendering: Rendering) -> Rendered:
    images = [
        _image(block) for block in result.content if isinstance(block, mcp_types.ImageContent)
    ]
    if result.is_error:
        text = "\n".join(_texts(result.content)) or "Unknown error"
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    elif rendering == "structured" and result.structured_content is not None:
        text = jsonjs.dumps(result.structured_content)
        blocks = [{"type": "text", "text": text}, *images]
    else:
        text = "\n".join(_texts(result.content))
        blocks = [{"type": "text", "text": t} for t in _texts(result.content)] + images
    tokens = estimated_tokens(text)
    oversized = tokens > OVERSIZED_TOKENS
    if oversized:
        blocks = [{"type": "text", "text": oversize_notice(tokens)}]
    if not blocks:
        blocks = [{"type": "text", "text": "(no content)"}]
    return Rendered(
        content=blocks,
        is_error=result.is_error,
        tokens=tokens,
        oversized=oversized,
        structured=result.structured_content,
    )


def oversize_notice(tokens: int) -> str:
    return (
        f"This result was about {tokens} tokens, over the {OVERSIZED_TOKENS}-token limit "
        "on one tool result, and was not shown."
    )


def _texts(content: list[Any]) -> list[str]:
    texts = []
    for block in content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
        elif isinstance(block, mcp_types.ResourceLink):
            line = f"[Resource link: {block.name}] {block.uri}"
            if block.description:
                line += f" ({block.description})"
            texts.append(line)
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                texts.append(resource.text)
    return texts


def _image(block: mcp_types.ImageContent) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
    }
