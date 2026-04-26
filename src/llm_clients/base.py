"""Abstract base class and shared data models for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class CompletionResult(BaseModel):
    """Structured return value from any LLM completion call."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response: Any = None

    model_config = {"arbitrary_types_allowed": True}


class LLMClient(ABC):
    """Abstract interface for all LLM provider clients.

    Subclasses must implement :meth:`complete` and should integrate with
    :class:`~src.cost_tracking.tracker.CostTracker` at the call site.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a chat-style message list and return a completion result.

        Args:
            messages: OpenAI-style list of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Provider-specific overrides (max_tokens, temperature, …).

        Returns:
            :class:`CompletionResult` containing the response text and token counts.
        """
        ...
