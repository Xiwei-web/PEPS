"""Message builders for PEPS multimodal LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.core.types import ImageRef
from peps.llm.image_payload import ImagePayload, to_chat_image_block, to_responses_image_block


@dataclass(slots=True)
class LLMMessage:
    """A role-based message before API-specific formatting."""

    role: str
    text: str = ""
    images: list[ImagePayload | ImageRef | str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Unsupported LLM message role: {self.role}")


def input_text_block(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def output_text_block(text: str) -> dict[str, str]:
    return {"type": "output_text", "text": text}


def chat_text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_responses_message(message: LLMMessage) -> dict[str, Any]:
    """Convert an LLMMessage into a Responses API input item."""
    content: list[dict[str, Any]] = []
    if message.text:
        if message.role == "assistant":
            content.append(output_text_block(message.text))
        else:
            content.append(input_text_block(message.text))
    for image in message.images:
        content.append(to_responses_image_block(image))
    return {"role": message.role, "content": content}


def build_chat_message(message: LLMMessage) -> dict[str, Any]:
    """Convert an LLMMessage into a Chat Completions-style message."""
    if not message.images:
        return {"role": message.role, "content": message.text}
    content: list[dict[str, Any]] = []
    if message.text:
        content.append(chat_text_block(message.text))
    for image in message.images:
        content.append(to_chat_image_block(image))
    return {"role": message.role, "content": content}


def build_user_message(
    text: str,
    *,
    images: list[ImagePayload | ImageRef | str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMMessage:
    return LLMMessage(
        role="user",
        text=text,
        images=images or [],
        metadata=metadata or {},
    )


def build_responses_input(
    user_text: str,
    *,
    images: list[ImagePayload | ImageRef | str] | None = None,
    prior_messages: list[LLMMessage] | None = None,
) -> list[dict[str, Any]]:
    """Build the `input` field for the Responses API."""
    messages = [*(prior_messages or []), build_user_message(user_text, images=images)]
    return [build_responses_message(message) for message in messages]


def build_chat_messages(
    user_text: str,
    *,
    system_prompt: str | None = None,
    images: list[ImagePayload | ImageRef | str] | None = None,
    prior_messages: list[LLMMessage] | None = None,
) -> list[dict[str, Any]]:
    """Build messages for Chat Completions-compatible clients."""
    messages: list[LLMMessage] = []
    if system_prompt:
        messages.append(LLMMessage(role="system", text=system_prompt))
    messages.extend(prior_messages or [])
    messages.append(build_user_message(user_text, images=images))
    return [build_chat_message(message) for message in messages]

