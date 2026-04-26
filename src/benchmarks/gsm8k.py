"""GSM8K benchmark loader.

Uses the Hugging Face ``datasets`` library; the dataset is downloaded and
cached automatically on first use.
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import Benchmark


def _extract_number(text: str) -> str:
    """Pull the last number from *text* (handles commas and decimals)."""
    matches = re.findall(r"-?[\d,]+\.?\d*", text.replace(",", ""))
    return matches[-1].replace(",", "") if matches else text.strip()


class GSM8K(Benchmark):
    """Grade-school math word problems.

    Default split is ``"test"`` (1319 problems).

    Args:
        split: HF dataset split (``"train"`` or ``"test"``).
    """

    name = "gsm8k"

    def __init__(self, split: str = "test") -> None:
        self.split = split

    def iter_tasks(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset("gsm8k", "main", split=self.split)
        for i, item in enumerate(ds):
            yield {
                "id": str(i),
                "question": item["question"],
                "answer": _extract_number(item["answer"].split("####")[-1].strip()),
            }

    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """True if the extracted final number in *prediction* matches *ground_truth*."""
        return _extract_number(prediction) == ground_truth.strip()
