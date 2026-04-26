"""Heterogeneous debate: different model families in Solver / Critic / Judge roles."""

from .base import Strategy, StrategyResult
from ..cost_tracking.tracker import CostTracker

_SOLVER_PROMPT = (
    "Solve the following problem step by step and state your final answer clearly.\n\n"
    "{question}"
)

_CRITIC_PROMPT = (
    "Review this proposed solution for correctness and completeness.\n\n"
    "Problem: {question}\n\nSolution:\n{solution}\n\n"
    "List any errors or improvements needed."
)

_JUDGE_PROMPT = (
    "You are the final arbiter. Incorporate the critique and produce a corrected answer.\n\n"
    "Problem: {question}\n\nProposed solution:\n{solution}\n\nCritique:\n{critique}\n\n"
    "Final answer:"
)


class HeteroDebate(Strategy):
    """Mixed-model debate where each role uses a different model family.

    This strategy isolates whether accuracy gains from debate come from
    architectural diversity or simply from spending more tokens.

    Args:
        solver_model: Registry key for the solver agent.
        critic_model: Registry key for the critic agent.
        judge_model: Registry key for the judge agent.
        n_rounds: Number of critique-and-judge cycles.
    """

    name = "hetero_debate"

    def __init__(
        self,
        solver_model: str = "gpt-4o-mini",
        critic_model: str = "claude-haiku-4-5",
        judge_model: str = "gemini-2.5-flash",
        n_rounds: int = 2,
        # Accept model_name for interface compatibility; ignored here.
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.solver_model = solver_model
        self.critic_model = critic_model
        self.judge_model = judge_model
        self.n_rounds = n_rounds

    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Run the heterogeneous pipeline and return the judge's final answer."""
        from ..llm_clients.factory import get_client

        solver = get_client(self.solver_model)
        critic = get_client(self.critic_model)
        judge = get_client(self.judge_model)

        question: str = task.get("question", "")
        task_id: str = str(task.get("id", "unknown"))
        trace: list[str] = []
        total_cost: float = 0.0
        n_calls: int = 0

        def call(client_obj, prompt: str, role: str, model_key: str) -> str:
            nonlocal total_cost, n_calls
            result = client_obj.complete([{"role": "user", "content": prompt}])
            tracker.log_call(
                model=model_key,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                task_id=task_id,
            )
            trace.append(f"[role={role} model={model_key}] {result.text.strip()}")
            total_cost += result.cost_usd
            n_calls += 1
            return result.text.strip()

        solution = call(
            solver, _SOLVER_PROMPT.format(question=question), "solver", self.solver_model
        )

        for round_idx in range(self.n_rounds):
            critique = call(
                critic,
                _CRITIC_PROMPT.format(question=question, solution=solution),
                f"critic_r{round_idx}",
                self.critic_model,
            )
            solution = call(
                judge,
                _JUDGE_PROMPT.format(
                    question=question, solution=solution, critique=critique
                ),
                f"judge_r{round_idx}",
                self.judge_model,
            )

        return StrategyResult(
            prediction=solution,
            reasoning_trace=trace,
            total_cost_usd=total_cost,
            n_api_calls=n_calls,
        )
