"""Single-agent self-consistency: N independent samples + majority vote."""

from collections import Counter

from .base import Strategy, StrategyResult
from ..cost_tracking.tracker import CostTracker


class SingleAgentSC(Strategy):
    """Self-consistency baseline (Wang et al. 2023).

    Generates *n_samples* independent completions from one model and returns
    the plurality answer.  This is the primary iso-cost comparison point for
    all multi-agent strategies.

    Args:
        model_name: Registry key from ``configs/models.yaml``.
        n_samples: Number of independent samples to draw.
    """

    name = "single_agent_sc"

    def __init__(self, model_name: str, n_samples: int = 5) -> None:
        self.model_name = model_name
        self.n_samples = n_samples

    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Draw *n_samples* completions and return the majority-vote answer."""
        from ..llm_clients.factory import get_client

        client = get_client(self.model_name)
        question: str = task.get("question", "")
        task_id: str = str(task.get("id", "unknown"))
        messages = [{"role": "user", "content": question}]

        predictions: list[str] = []
        trace: list[str] = []
        total_cost: float = 0.0

        for i in range(self.n_samples):
            result = client.complete(messages)
            tracker.log_call(
                model=self.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                task_id=task_id,
            )
            predictions.append(result.text.strip())
            trace.append(f"[sample={i}] {result.text.strip()}")
            total_cost += result.cost_usd

        majority: str = Counter(predictions).most_common(1)[0][0]

        return StrategyResult(
            prediction=majority,
            reasoning_trace=trace,
            total_cost_usd=total_cost,
            n_api_calls=self.n_samples,
        )
