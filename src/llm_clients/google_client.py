"""Google Gemini client stub with correct signature."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import CompletionResult, LLMClient
from ..cost_tracking.pricing import compute_cost, get_model_config


class GoogleClient(LLMClient):
    """Google Gemini client via the google-generativeai SDK.

    Stub: imports and signatures are wired; retry logic and safety-setting
    configuration should be added before running production experiments.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        super().__init__(model_name)
        import google.generativeai as genai  # lazy import

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=resolved_key)
        config = get_model_config(model_name)
        self._model = genai.GenerativeModel(config["model_id"])

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> CompletionResult:
        """Call the Gemini generateContent API.

        Converts OpenAI-style messages to Gemini's content format.
        Only the final user message is sent; prepend history as needed.
        """
        import google.generativeai as genai  # lazy import

        # Build Gemini-style history from OpenAI-style messages.
        history = []
        prompt = ""
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if msg is messages[-1] and msg["role"] == "user":
                prompt = msg["content"]
            else:
                history.append({"role": role, "parts": [msg["content"]]})

        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        text: str = response.text
        # Gemini usage metadata may not always be present.
        usage = getattr(response, "usage_metadata", None)
        input_tokens: int = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens: int = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost: float = compute_cost(self.model_name, input_tokens, output_tokens)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            raw_response=response,
        )
