"""Vanilla multi-agent debate strategy (Du et al. 2024).

N homogeneous agents each produce an initial answer, then refine their
answer after seeing the other agents' responses.  Final answer is determined
by majority vote across all agents' final-round responses.
"""

from collections import Counter

from .base import Strategy, StrategyResult
from ..cost_tracking.tracker import CostTracker


class VanillaDebate(Strategy):
    """Homogeneous multi-agent debate.

    Args:
        model_name: Registry key from ``configs/models.yaml``.
        n_agents: Number of debating agents.
        n_rounds: Number of refinement rounds after the initial answers.
    """

    name = "vanilla_debate"

    def __init__(
        self,
        model_name: str,
        n_agents: int = 3,
        n_rounds: int = 2,
    ) -> None:
        self.model_name = model_name
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Run debate and return the majority-vote final answer."""
        from ..llm_clients.factory import get_client

        client = get_client(self.model_name)
        question: str = task.get("question", "")
        task_id: str = str(task.get("id", "unknown"))
        trace: list[str] = []
        total_cost: float = 0.0
        n_calls: int = 0

        # Round 0 — independent initial answers.
        agent_responses: list[str] = []
        for agent_idx in range(self.n_agents):
            messages = [{"role": "user", "content": question}]
            result = client.complete(messages)
            tracker.log_call(
                model=self.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                task_id=task_id,
            )
            agent_responses.append(result.text.strip())
            trace.append(f"[round=0 agent={agent_idx}] {result.text.strip()}")
            total_cost += result.cost_usd
            n_calls += 1

        # Rounds 1..n_rounds — each agent refines after seeing all peers.
        for round_idx in range(1, self.n_rounds + 1):
            context = "\n".join(
                f"Agent {i}: {r}" for i, r in enumerate(agent_responses)
            )
            new_responses: list[str] = []
            for agent_idx in range(self.n_agents):
                prompt = (
                    f"{question}\n\n"
                    f"Other agents' current answers:\n{context}\n\n"
                    "Critically evaluate these answers and provide your refined answer."
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
                new_responses.append(result.text.strip())
                trace.append(
                    f"[round={round_idx} agent={agent_idx}] {result.text.strip()}"
                )
                total_cost += result.cost_usd
                n_calls += 1
            agent_responses = new_responses

        prediction: str = Counter(agent_responses).most_common(1)[0][0]

        return StrategyResult(
            prediction=prediction,
            reasoning_trace=trace,
            total_cost_usd=total_cost,
            n_api_calls=n_calls,
        )
