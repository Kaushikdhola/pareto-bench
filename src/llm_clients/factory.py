"""Factory function for instantiating LLM clients by model name."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMClient


def get_client(model_name: str) -> "LLMClient":
    """Return an :class:`LLMClient` instance for *model_name*.

    Provider is resolved by looking up *model_name* in ``configs/models.yaml``.
    All imports are deferred so this module is importable without any provider
    SDK installed.

    Args:
        model_name: Registry key from ``configs/models.yaml``
                    (e.g. ``"gpt-4o-mini"``).

    Returns:
        A fully initialised :class:`LLMClient` subclass.

    Raises:
        ValueError: If *model_name* is not in the registry or the provider
                    is not supported.
    """
    from ..cost_tracking.pricing import get_model_config

    config = get_model_config(model_name)
    provider: str = config["provider"]

    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(model_name)
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model_name)
    elif provider == "google":
        from .google_client import GoogleClient

        return GoogleClient(model_name)
    else:
        raise ValueError(
            f"Unsupported provider {provider!r} for model {model_name!r}. "
            "Add a new client under src/llm_clients/ and register it here."
        )
