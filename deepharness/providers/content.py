"""Message content that is more than a string: images and files in, reasoning out.

A transcript entry's content is a `str` in the common case and a list of blocks
when it has to be - an image the model should look at, a PDF to read, or the
thinking Anthropic requires replayed alongside a tool call. Blocks are
normalized here and rendered per vendor beside each client, because the shape
of an image on the wire is exactly the kind of thing vendors disagree about.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True, slots=True)
class Text:
    """Prose, the part of a message every vendor agrees about."""

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class Image:
    """An image for the model to look at, as base64 data or a URL.

    Exactly one of the two: a block carrying both would leave each provider to
    pick one, and they would not all pick the same.
    """

    data: str | None = None
    media_type: str = "image/png"
    url: str | None = None

    def __post_init__(self) -> None:
        if bool(self.data) == bool(self.url):
            raise ConfigurationError(
                "Image needs either data= (base64) or url=, not both or neither"
            )

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """Read and encode a local image, taking its media type from the suffix."""
        file = Path(path)
        return cls(data=_encode(file), media_type=_media_type(file))

    @classmethod
    def from_url(cls, url: str) -> Image:
        return cls(url=url)

    @property
    def data_url(self) -> str:
        """The `data:` form, which is how several vendors take inline images."""
        if self.url is not None:
            return self.url
        return f"data:{self.media_type};base64,{self.data}"

    def to_dict(self) -> dict[str, Any]:
        if self.url is not None:
            return {"type": "image", "url": self.url}
        return {"type": "image", "data": self.data, "media_type": self.media_type}


@dataclass(frozen=True, slots=True)
class Document:
    """A file for the model to read - a PDF, typically - as base64 data."""

    data: str
    media_type: str = "application/pdf"
    name: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Document:
        file = Path(path)
        return cls(data=_encode(file), media_type=_media_type(file), name=file.name)

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"

    def to_dict(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": "document",
            "data": self.data,
            "media_type": self.media_type,
        }
        if self.name is not None:
            block["name"] = self.name
        return block


@dataclass(frozen=True, slots=True)
class Thinking:
    """Reasoning the model reported before answering.

    `signature` is Anthropic's attestation of that reasoning. It is carried
    through and replayed with the turn because Anthropic requires the thinking
    block back, unmodified, on the request that follows a tool call - dropping
    it breaks the chain mid-task, which is exactly where thinking matters.
    """

    text: str
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "thinking", "text": self.text}
        if self.signature is not None:
            block["signature"] = self.signature
        return block


Block = Text | Image | Document | Thinking
"""One piece of a message's content."""

Content = str | list[Block]
"""What a message carries: a string, or blocks when text is not enough."""


def parse(content: Any) -> list[Block]:
    """Whatever a transcript holds for one message, as blocks.

    Accepts a string, blocks, or the dicts a saved session gives back, so a
    provider never has to ask which of the three it was handed.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content)] if content else []
    if isinstance(content, dict):
        return [from_dict(content)]
    return [
        block if isinstance(block, Block) else from_dict(block) for block in content
    ]


def from_dict(data: dict[str, Any]) -> Block:
    """One block from its wire form, or a ConfigurationError naming the type."""
    kind = data.get("type")
    match kind:
        case "text":
            return Text(data.get("text", ""))
        case "image":
            return Image(
                data=data.get("data"),
                media_type=data.get("media_type", "image/png"),
                url=data.get("url"),
            )
        case "document":
            return Document(
                data=data.get("data", ""),
                media_type=data.get("media_type", "application/pdf"),
                name=data.get("name"),
            )
        case "thinking":
            return Thinking(data.get("text", ""), signature=data.get("signature"))
        case _:
            raise ConfigurationError(f"unknown content block type: {kind!r}")


def to_wire(content: Any) -> str | list[dict[str, Any]]:
    """Content as JSON-able data: a plain string when that is all it is.

    Keeping the string form for string content matters beyond tidiness - it is
    what makes a transcript written before blocks existed identical to one
    written now, so a saved session is not quietly reshaped by loading it.
    """
    if isinstance(content, str) or content is None:
        return content or ""
    blocks = parse(content)
    if len(blocks) == 1 and isinstance(blocks[0], Text):
        return blocks[0].text
    return [block.to_dict() for block in blocks]


def merge(blocks: Iterable[Block]) -> list[Block]:
    """Join runs of the same kind of text, so a stream reads like a whole turn.

    A streamed turn arrives as dozens of fragments; keeping them apart would
    make the assembled response a different shape from the same turn fetched in
    one call, and every caller would have to handle both.
    """
    merged: list[Block] = []
    for block in blocks:
        last = merged[-1] if merged else None
        if (
            isinstance(block, Text | Thinking)
            and type(last) is type(block)
            and getattr(last, "signature", None) == getattr(block, "signature", None)
        ):
            merged[-1] = replace(last, text=last.text + block.text)  # type: ignore[arg-type]
        else:
            merged.append(block)
    return merged


def text_of(content: Any) -> str:
    """Just the prose, for a vendor or a caller that wants one string.

    Thinking is left out: it is the model's working, not its answer, and a
    caller printing content should not be handed both run together.
    """
    return "".join(block.text for block in parse(content) if isinstance(block, Text))


def _encode(file: Path) -> str:
    return base64.b64encode(file.read_bytes()).decode("ascii")


def _media_type(file: Path) -> str:
    media_type = _MEDIA_TYPES.get(file.suffix.lower())
    if media_type is None:
        raise ConfigurationError(
            f"cannot infer a media type for {file.name}; pass media_type= explicitly"
        )
    return media_type
