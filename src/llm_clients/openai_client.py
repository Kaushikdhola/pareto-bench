"""OpenAI client stub with correct signature."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import CompletionResult, LLMClient
from ..cost_tracking.pricing import compute_cost, get_model_config


class OpenAIClient(LLMClient):
    """OpenAI chat-completions client.

    Stub: imports and signatures are wired; retry and streaming support
    should be added before running production experiments.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        super().__init__(model_name)
        import openai  # lazy import so stub is importable without openai installed

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=resolved_key)
        config = get_model_config(model_name)
        self._model_id: str = config["model_id"]

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResult:
        """Call the OpenAI chat-completions API."""
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        input_tokens: int = response.usage.prompt_tokens  # type: ignore[union-attr]
        output_tokens: int = response.usage.completion_tokens  # type: ignore[union-attr]
        text: str = response.choices[0].message.content or ""
        cost: float = compute_cost(self.model_name, input_tokens, output_tokens)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            raw_response=response,
        )
