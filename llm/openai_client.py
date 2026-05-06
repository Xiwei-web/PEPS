"""OpenAI Responses API client used by PEPS agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from peps.core.exceptions import LLMOutputError
from peps.core.types import ImageRef
from peps.llm.image_payload import ImagePayload
from peps.llm.json_parser import parse_json_object
from peps.llm.message_builder import LLMMessage, build_responses_input
from peps.llm.retry import RetryConfig, retry_call


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass(slots=True)
class OpenAIClientConfig:
    """Configuration for OpenAI-compatible Responses API calls."""

    api_key: str | None = None
    model: str | None = None
    base_url: str = DEFAULT_OPENAI_BASE_URL
    organization: str | None = None
    project: str | None = None
    timeout_seconds: float = 120.0
    max_output_tokens: int | None = 2048
    temperature: float | None = 0.0
    retry: RetryConfig = field(default_factory=RetryConfig)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        max_output_tokens: int | None = 2048,
        temperature: float | None = 0.0,
    ) -> "OpenAIClientConfig":
        """Build config from environment variables.

        Model resolution order:
        explicit `model` argument -> PEPS_OPENAI_MODEL -> OPENAI_MODEL -> gpt-4o.
        """
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model
            or os.getenv("PEPS_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_OPENAI_MODEL,
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/"),
            organization=os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT_ID"),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def resolved_model(self, override: str | None = None) -> str:
        return override or self.model or DEFAULT_OPENAI_MODEL

    def require_api_key(self) -> str:
        if not self.api_key:
            raise LLMOutputError(
                "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=\"your-openai-api-key\""
            )
        return self.api_key


@dataclass(slots=True)
class OpenAIResponse:
    """Normalized OpenAI response."""

    text: str
    raw: dict[str, Any]
    model: str
    response_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def json_object(self) -> dict[str, Any]:
        return parse_json_object(self.text)


class OpenAIClient:
    """Small stdlib-based OpenAI Responses API client.

    It intentionally avoids importing the optional `openai` package so PEPS can
    run in minimal environments. The model name is always configurable per call.
    """

    def __init__(self, config: OpenAIClientConfig | None = None) -> None:
        self.config = config or OpenAIClientConfig.from_env()

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "OpenAIClient":
        return cls(OpenAIClientConfig.from_env(model=model))

    def create_response(
        self,
        *,
        user_text: str,
        system_prompt: str | None = None,
        images: list[ImagePayload | ImageRef | str] | None = None,
        prior_messages: list[LLMMessage] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        text_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> OpenAIResponse:
        payload = self._build_payload(
            user_text=user_text,
            system_prompt=system_prompt,
            images=images,
            prior_messages=prior_messages,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            text_format=text_format,
            extra_body=extra_body,
        )
        raw = retry_call(lambda: self._post_json("/responses", payload), config=self.config.retry)
        text = extract_response_text(raw)
        return OpenAIResponse(
            text=text,
            raw=raw,
            model=payload["model"],
            response_id=raw.get("id"),
            usage=raw.get("usage", {}),
        )

    def generate_text(
        self,
        *,
        user_text: str,
        system_prompt: str | None = None,
        images: list[ImagePayload | ImageRef | str] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        response = self.create_response(
            user_text=user_text,
            system_prompt=system_prompt,
            images=images,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        return response.text

    def generate_json(
        self,
        *,
        user_text: str,
        system_prompt: str | None = None,
        images: list[ImagePayload | ImageRef | str] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "peps_output",
        strict: bool = False,
    ) -> dict[str, Any]:
        text_format = (
            {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": strict,
            }
            if json_schema is not None
            else {"type": "json_object"}
        )
        response = self.create_response(
            user_text=user_text,
            system_prompt=system_prompt,
            images=images,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            text_format=text_format,
        )
        return response.json_object()

    def _build_payload(
        self,
        *,
        user_text: str,
        system_prompt: str | None,
        images: list[ImagePayload | ImageRef | str] | None,
        prior_messages: list[LLMMessage] | None,
        model: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        text_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.resolved_model(model),
            "input": build_responses_input(
                user_text,
                images=images,
                prior_messages=prior_messages,
            ),
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        resolved_max_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.config.max_output_tokens
        )
        if resolved_max_tokens is not None:
            payload["max_output_tokens"] = resolved_max_tokens
        resolved_temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        if text_format is not None:
            payload["text"] = {"format": text_format}
        if extra_body:
            payload.update(extra_body)
        return payload

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = self.config.require_api_key()
        url = f"{self.config.base_url.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(error_text)
            except json.JSONDecodeError:
                error_payload = {"error": {"message": error_text}}
            message = error_payload.get("error", {}).get("message", error_text)
            raise LLMOutputError(f"OpenAI API request failed [{exc.code}]: {message}") from exc
        except json.JSONDecodeError as exc:
            raise LLMOutputError("OpenAI API returned invalid JSON") from exc


def extract_response_text(raw: dict[str, Any]) -> str:
    """Extract assistant text from a Responses API response object."""
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]

    parts: list[str] = []
    for item in raw.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            content_type = content.get("type")
            if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])

    if parts:
        return "\n".join(parts).strip()

    if raw.get("error"):
        raise LLMOutputError(f"OpenAI response contains error: {raw['error']}")
    raise LLMOutputError("Could not extract text from OpenAI response")
