"""Abstract base class and result model shared by all strategies."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..cost_tracking.tracker import CostTracker


class StrategyResult(BaseModel):
    """Structured output from any strategy's solve() call."""

    prediction: str
    reasoning_trace: list[str]
    total_cost_usd: float
    n_api_calls: int


class Strategy(ABC):
    """Abstract base class for debate and reasoning strategies.

    All strategies receive a *task* dict (with at minimum ``"id"`` and
    ``"question"`` keys) and a live :class:`CostTracker`.  They must call
    ``tracker.log_call(...)`` for every API call they make so that cost
    accounting is complete.
    """

    name: str = "base"

    @abstractmethod
    def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
        """Solve *task* within the budget tracked by *tracker*.

        Args:
            task: Dict with at minimum ``"id"`` and ``"question"`` keys.
            tracker: Live :class:`CostTracker`; will raise
                     :class:`~src.cost_tracking.tracker.BudgetExceededError`
                     if the budget cap is hit.

        Returns:
            :class:`StrategyResult` with prediction and full reasoning trace.
        """
        ...
