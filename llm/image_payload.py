"""Image payload helpers for OpenAI multimodal requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from peps.core.exceptions import SerializationError
from peps.core.types import ImageRef


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


@dataclass(slots=True)
class ImagePayload:
    """An image reference prepared for a multimodal LLM request."""

    uri: str
    detail: str = "auto"
    view_id: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_image_ref(cls, image_ref: ImageRef, *, detail: str = "auto") -> "ImagePayload":
        return cls(
            uri=image_ref.uri,
            detail=detail,
            view_id=image_ref.view_id,
            metadata=dict(image_ref.metadata),
        )


def infer_mime_type(path_or_uri: str) -> str:
    mime_type, _ = mimetypes.guess_type(path_or_uri)
    return mime_type or "image/png"


def is_remote_uri(uri: str) -> bool:
    scheme = urlparse(uri).scheme.lower()
    return scheme in {"http", "https", "data"}


def path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(uri)


def encode_image_as_data_url(path: str | Path, *, mime_type: str | None = None) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise SerializationError(f"Image file not found: {image_path}")
    mime_type = mime_type or infer_mime_type(str(image_path))
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise SerializationError(f"Unsupported image MIME type {mime_type!r}: {image_path}")
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise SerializationError(f"Failed to read image file: {image_path}") from exc
    return f"data:{mime_type};base64,{encoded}"


def image_payload_to_url(payload: ImagePayload | ImageRef | str) -> str:
    if isinstance(payload, ImageRef):
        payload = ImagePayload.from_image_ref(payload)
    if isinstance(payload, str):
        payload = ImagePayload(uri=payload)

    if is_remote_uri(payload.uri):
        return payload.uri
    local_path = path_from_uri(payload.uri)
    return encode_image_as_data_url(local_path, mime_type=payload.mime_type)


def to_responses_image_block(payload: ImagePayload | ImageRef | str) -> dict[str, Any]:
    """Build an input_image block for the OpenAI Responses API."""
    detail = payload.detail if isinstance(payload, ImagePayload) else "auto"
    return {
        "type": "input_image",
        "image_url": image_payload_to_url(payload),
        "detail": detail,
    }


def to_chat_image_block(payload: ImagePayload | ImageRef | str) -> dict[str, Any]:
    """Build an image_url block compatible with Chat Completions-style messages."""
    detail = payload.detail if isinstance(payload, ImagePayload) else "auto"
    return {
        "type": "image_url",
        "image_url": {
            "url": image_payload_to_url(payload),
            "detail": detail,
        },
    }

