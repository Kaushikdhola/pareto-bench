"""ARC-Challenge benchmark loader.

Uses the Hugging Face ``datasets`` library.  The model is expected to respond
with the letter of the correct option (A/B/C/D).
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import Benchmark


def _extract_letter(text: str) -> str:
    """Extract the first A/B/C/D letter from *text* (case-insensitive)."""
    match = re.search(r"\b([A-Da-d])\b", text)
    return match.group(1).upper() if match else text.strip().upper()[:1]


class ARCChallenge(Benchmark):
    """ARC Challenge subset — science multiple-choice questions.

    Args:
        split: HF dataset split (``"train"``, ``"validation"``, or ``"test"``).
    """

    name = "arc_challenge"

    def __init__(self, split: str = "test") -> None:
        self.split = split

    def iter_tasks(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset("ai2_arc", "ARC-Challenge", split=self.split)
        for item in ds:
            choices = item["choices"]
            choice_text = "\n".join(
                f"{label}. {text}"
                for label, text in zip(choices["label"], choices["text"])
            )
            question = f"{item['question']}\n{choice_text}"
            yield {
                "id": item["id"],
                "question": question,
                "answer": item["answerKey"],
                "choices": choices,
            }

    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """True if the extracted letter matches the ground-truth key."""
        return _extract_letter(prediction) == ground_truth.strip().upper()
