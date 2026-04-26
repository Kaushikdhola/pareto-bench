"""Load model pricing from configs/models.yaml and compute per-call cost."""

from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "models.yaml"
_MODELS: Optional[dict[str, Any]] = None


def _load_models() -> dict[str, Any]:
    global _MODELS
    if _MODELS is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        _MODELS = data.get("models", {})
    return _MODELS


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the config dict for *model_name* from models.yaml.

    Raises:
        ValueError: If *model_name* is not in the registry.
    """
    models = _load_models()
    if model_name not in models:
        available = list(models.keys())
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {available}"
        )
    return models[model_name]


def compute_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for one API call given token counts."""
    config = get_model_config(model_name)
    input_cost = (input_tokens / 1_000_000) * config["input_price_per_M"]
    output_cost = (output_tokens / 1_000_000) * config["output_price_per_M"]
    return input_cost + output_cost
