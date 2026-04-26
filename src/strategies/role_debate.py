"""Role-based debate: Solver → Critic → Judge pipeline."""

from .base import Strategy, StrategyResult
from ..cost_tracking.tracker import CostTracker

_SOLVER_PROMPT = (
    "You are a careful problem solver. "
    "Solve the following problem step by step and give a clear final answer.\n\n"
    "{question}"
)

_CRITIC_PROMPT = (
    "You are a rigorous critic. "
    "Review the following solution and identify any errors or gaps in reasoning.\n\n"
    "Problem: {question}\n\nProposed solution:\n{solution}\n\n"
    "Provide specific, constructive criticism."
)

_JUDGE_PROMPT = (
    "You are an impartial judge. "
    "Given the problem, a proposed solution, and a critique, "
    "produce a final corrected answer.\n\n"
    "Problem: {question}\n\n"
    "Proposed solution:\n{solution}\n\n"
    "Critique:\n{critique}\n\n"
    "Final answer:"
)


class RoleDebate(Strategy):
    """Structured Solver / Critic / Judge pipeline.

    Each role is played by the same model by default.  The pipeline runs
    *n_rounds* times, feeding the judge's output back as the new solution.

    Args:
        model_name: Registry key from ``configs/models.yaml``.
        n_rounds: Number of full Solver→Critic→Judge cycles.
    """

    name = "role_debate"

    def __init__(self, model_name: str, n_rounds: int = 2) -> None:
        self.model_name = model_name
        self.n_rounds = n_rounds

    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Run the role-based pipeline and return the judge's final answer."""
        from ..llm_clients.factory import get_client

        client = get_client(self.model_name)
        question: str = task.get("question", "")
        task_id: str = str(task.get("id", "unknown"))
        trace: list[str] = []
        total_cost: float = 0.0
        n_calls: int = 0

        def call(prompt: str, role: str) -> str:
            nonlocal total_cost, n_calls
            result = client.complete([{"role": "user", "content": prompt}])
            tracker.log_call(
                model=self.model_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                task_id=task_id,
            )
            trace.append(f"[role={role}] {result.text.strip()}")
            total_cost += result.cost_usd
            n_calls += 1
            return result.text.strip()

        solution = call(_SOLVER_PROMPT.format(question=question), "solver")

        for round_idx in range(self.n_rounds):
            critique = call(
                _CRITIC_PROMPT.format(question=question, solution=solution),
                f"critic_r{round_idx}",
            )
            solution = call(
                _JUDGE_PROMPT.format(
                    question=question, solution=solution, critique=critique
                ),
                f"judge_r{round_idx}",
            )

        return StrategyResult(
            prediction=solution,
            reasoning_trace=trace,
            total_cost_usd=total_cost,
            n_api_calls=n_calls,
        )
