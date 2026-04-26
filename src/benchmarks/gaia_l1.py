"""GAIA Level-1 benchmark loader.

GAIA (General AI Assistants) requires Hugging Face access — run
``huggingface-cli login`` before the first use.  We restrict to Level 1
(no tool use required) for the initial pareto-bench experiments.
"""

from __future__ import annotations

from typing import Iterator

from .base import Benchmark


class GAIAL1(Benchmark):
    """GAIA Level-1 open-ended reasoning tasks.

    Args:
        split: HF dataset split (``"test"`` only for the official evaluation;
               ``"validation"`` is available for development).
    """

    name = "gaia_l1"

    def __init__(self, split: str = "validation") -> None:
        self.split = split

    def iter_tasks(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split=self.split)
        for item in ds:
            if item.get("Level", 1) != 1:
                continue
            yield {
                "id": item.get("task_id", str(hash(item["Question"]))),
                "question": item["Question"],
                "answer": item.get("Final answer", ""),
            }

    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """Exact-match after stripping whitespace and lowercasing."""
        return prediction.strip().lower() == ground_truth.strip().lower()
