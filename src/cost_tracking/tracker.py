"""CostTracker: per-call logging with hard budget caps.

Every LLM call in pareto-bench must pass through a CostTracker so that
(a) we have a complete JSONL audit trail and (b) experiments stop cleanly
when they hit the iso-cost boundary.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class BudgetExceededError(Exception):
    """Raised before a call is logged when it would exceed a budget cap."""


@dataclass
class CallRecord:
    """One logged API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str
    strategy: str
    task_id: str
    run_id: str


class CostTracker:
    """Thread-unsafe but process-safe cost tracker for a single experiment run.

    Args:
        run_id: Unique identifier for this run. Auto-generated if omitted.
        strategy: Strategy name embedded in every log record.
        log_path: JSONL file to append records to. No file written if None.
        run_budget_usd: Hard cap on total spend for this run.
        task_budget_usd: Hard cap on spend per task_id.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        strategy: str = "unknown",
        log_path: Optional[Path] = None,
        run_budget_usd: Optional[float] = None,
        task_budget_usd: Optional[float] = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.strategy = strategy
        self.log_path = log_path
        self.run_budget_usd = run_budget_usd
        self.task_budget_usd = task_budget_usd

        self._run_total_usd: float = 0.0
        self._task_totals: dict[str, float] = {}
        self._records: list[CallRecord] = []

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        task_id: str,
    ) -> CallRecord:
        """Record an API call and enforce budget caps.

        Raises:
            BudgetExceededError: If the call would push the task or run total
                                 over the configured cap.  The call is NOT
                                 recorded when this is raised.
        """
        new_task_total = self._task_totals.get(task_id, 0.0) + cost_usd
        if self.task_budget_usd is not None and new_task_total > self.task_budget_usd:
            raise BudgetExceededError(
                f"task {task_id!r} budget exceeded: "
                f"${new_task_total:.6f} > ${self.task_budget_usd:.6f}"
            )

        new_run_total = self._run_total_usd + cost_usd
        if self.run_budget_usd is not None and new_run_total > self.run_budget_usd:
            raise BudgetExceededError(
                f"run {self.run_id!r} budget exceeded: "
                f"${new_run_total:.6f} > ${self.run_budget_usd:.6f}"
            )

        record = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=self.strategy,
            task_id=task_id,
            run_id=self.run_id,
        )

        self._task_totals[task_id] = new_task_total
        self._run_total_usd = new_run_total
        self._records.append(record)

        if self.log_path is not None:
            self._append_jsonl(record)

        return record

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def run_total_usd(self) -> float:
        """Total spend for this run in USD."""
        return self._run_total_usd

    def task_total_usd(self, task_id: str) -> float:
        """Total spend for a specific task_id in USD."""
        return self._task_totals.get(task_id, 0.0)

    def n_calls(self) -> int:
        """Number of calls logged so far."""
        return len(self._records)

    def records(self) -> list[CallRecord]:
        """Snapshot of all logged records."""
        return list(self._records)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_jsonl(self, record: CallRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(asdict(record)) + "\n")
