"""Retry helpers for transient LLM transport failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import random
import time
from typing import TypeVar
from urllib.error import HTTPError, URLError

T = TypeVar("T")


@dataclass(slots=True)
class RetryConfig:
    """Exponential backoff settings."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    exponential_base: float = 2.0
    jitter_seconds: float = 0.25
    retry_status_codes: set[int] = field(
        default_factory=lambda: {408, 409, 429, 500, 502, 503, 504}
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryConfig.max_attempts must be >= 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("RetryConfig.initial_delay_seconds must be >= 0")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("RetryConfig.max_delay_seconds must be >= initial delay")


def is_retryable_exception(error: Exception, config: RetryConfig) -> bool:
    """Return whether an exception is likely transient."""
    if isinstance(error, HTTPError):
        return error.code in config.retry_status_codes
    return isinstance(error, (TimeoutError, URLError, ConnectionError))


def retry_call(
    fn: Callable[[], T],
    *,
    config: RetryConfig | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run a callable with exponential backoff."""
    config = config or RetryConfig()
    last_error: Exception | None = None
    for attempt in range(config.max_attempts):
        try:
            return fn()
        except Exception as error:
            last_error = error
            retryable = (
                should_retry(error)
                if should_retry is not None
                else is_retryable_exception(error, config)
            )
            if not retryable or attempt + 1 >= config.max_attempts:
                raise
            delay = min(
                config.initial_delay_seconds * (config.exponential_base**attempt),
                config.max_delay_seconds,
            )
            if config.jitter_seconds:
                delay += random.uniform(0.0, config.jitter_seconds)
            sleep(delay)
    assert last_error is not None
    raise last_error
