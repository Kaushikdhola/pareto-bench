"""Single-agent strategy augmented with a simple in-memory reasoning bank.

Inspired by Ouyang et al. (ReasoningBank, 2025).  Before answering, the
agent retrieves *n_retrieve* prior (question, reasoning) pairs from the bank
and includes them in context as few-shot examples.
"""

from __future__ import annotations

from .base import Strategy, StrategyResult
from ..cost_tracking.tracker import CostTracker


class ReasoningBank(Strategy):
    """Single-agent with in-context retrieval from accumulated reasoning traces.

    The bank grows as solve() is called; earlier traces become few-shot
    examples for later tasks.  Retrieval is currently keyword-overlap (BM25
    can be swapped in for production runs).

    Args:
        model_name: Registry key from ``configs/models.yaml``.
        bank_size: Maximum number of (question, reasoning) pairs to retain.
        n_retrieve: Number of examples to prepend to each prompt.
    """

    name = "reasoning_bank"

    def __init__(
        self,
        model_name: str,
        bank_size: int = 20,
        n_retrieve: int = 3,
    ) -> None:
        self.model_name = model_name
        self.bank_size = bank_size
        self.n_retrieve = n_retrieve
        self._bank: list[dict[str, str]] = []  # [{question, reasoning, answer}]

    def _retrieve(self, query: str) -> list[dict[str, str]]:
        """Return the *n_retrieve* most relevant bank entries by token overlap."""
        if not self._bank:
            return []
        query_tokens = set(query.lower().split())

        def overlap(entry: dict[str, str]) -> int:
            return len(query_tokens & set(entry["question"].lower().split()))

        ranked = sorted(self._bank, key=overlap, reverse=True)
        return ranked[: self.n_retrieve]

    def _add_to_bank(self, question: str, reasoning: str, answer: str) -> None:
        entry = {"question": question, "reasoning": reasoning, "answer": answer}
        self._bank.append(entry)
        if len(self._bank) > self.bank_size:
            self._bank.pop(0)

    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Solve with retrieved few-shot context from the reasoning bank."""
        from ..llm_clients.factory import get_client

        client = get_client(self.model_name)
        question: str = task.get("question", "")
        task_id: str = str(task.get("id", "unknown"))

        examples = self._retrieve(question)
        few_shot_block = ""
        if examples:
            parts = []
            for ex in examples:
                parts.append(
                    f"Example question: {ex['question']}\n"
                    f"Reasoning: {ex['reasoning']}\n"
                    f"Answer: {ex['answer']}"
                )
            few_shot_block = "\n\n".join(parts) + "\n\n"

        prompt = (
            f"{few_shot_block}"
            f"Question: {question}\n"
            "Reason step by step, then state your final answer clearly."
        )
        messages = [{"role": "user", "content": prompt}]
        result = client.complete(messages)
        tracker.log_call(
            model=self.model_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            task_id=task_id,
        )

        answer = result.text.strip()
        self._add_to_bank(question, answer, answer)

        return StrategyResult(
            prediction=answer,
            reasoning_trace=[f"[retrieved={len(examples)}] {answer}"],
            total_cost_usd=result.cost_usd,
            n_api_calls=1,
        )
