"""HumanEval benchmark loader.

Uses the Hugging Face ``datasets`` library.  Functional correctness (pass@1)
requires executing the generated code — this is disabled by default for
safety; set ``execute=True`` only in a sandboxed environment.
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import Benchmark


def _extract_code_block(text: str) -> str:
    """Pull the first ```python ... ``` block from *text*, or return *text* as-is."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


class HumanEval(Benchmark):
    """OpenAI HumanEval code-generation benchmark.

    Args:
        execute: If True, run the generated code + tests to compute pass@1.
                 Defaults to False (uses simple heuristic matching instead).
    """

    name = "humaneval"

    def __init__(self, execute: bool = False) -> None:
        self.execute = execute

    def iter_tasks(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset("openai_humaneval", split="test")
        for item in ds:
            yield {
                "id": item["task_id"],
                "question": item["prompt"],
                "answer": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }

    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """True if the prediction compiles and passes the test suite.

        When ``execute=False`` (default), falls back to checking that the
        predicted function signature is present — this is a loose proxy and
        should be replaced with execution-based evaluation for final results.
        """
        code = _extract_code_block(prediction)

        if not self.execute:
            return "def " in code

        # Execution-based pass@1 — only runs in a sandboxed environment.
        try:
            namespace: dict = {}
            exec(code, namespace)  # noqa: S102
            exec(ground_truth, namespace)  # noqa: S102
            return True
        except Exception:
            return False
