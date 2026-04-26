"""Anthropic Claude client with exponential-backoff retry on transient errors."""

from __future__ import annotations

import os
from typing import Any, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import CompletionResult, LLMClient
from ..cost_tracking.pricing import compute_cost, get_model_config

_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)


class AnthropicClient(LLMClient):
    """Anthropic messages API client.

    Retries up to 5 times on rate-limit and transient server errors using
    exponential backoff (2 s → 60 s).  The *model_id* used in API calls is
    resolved from ``configs/models.yaml``; *model_name* is the registry key.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        super().__init__(model_name)
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=resolved_key)
        config = get_model_config(model_name)
        self._model_id: str = config["model_id"]

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResult:
        """Call the Anthropic messages API.

        Args:
            messages: Chat history in OpenAI-compatible format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0–1).
            **kwargs: Forwarded verbatim to ``anthropic.Anthropic.messages.create``.

        Returns:
            :class:`CompletionResult` with text, token counts, and USD cost.
        """
        response = self._client.messages.create(
            model=self._model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        text: str = response.content[0].text
        cost: float = compute_cost(self.model_name, input_tokens, output_tokens)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            raw_response=response,
        )
