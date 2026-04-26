"""CLI orchestrator for pareto-bench experiments.

Usage:
    python -m src.runner --strategy single_agent_sc --dataset gsm8k \\
        --model gpt-4o-mini --budget 0.50 --seed 42 --max-tasks 50
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .cost_tracking.tracker import BudgetExceededError, CostTracker
from .utils.logging import get_logger
from .utils.seeds import set_seed

load_dotenv()
logger = get_logger(__name__)

_STRATEGY_MAP: dict[str, tuple[str, str]] = {
    "single_agent_sc": ("src.strategies.single_agent_sc", "SingleAgentSC"),
    "vanilla_debate": ("src.strategies.vanilla_debate", "VanillaDebate"),
    "role_debate": ("src.strategies.role_debate", "RoleDebate"),
    "hetero_debate": ("src.strategies.hetero_debate", "HeteroDebate"),
    "reasoning_bank": ("src.strategies.reasoning_bank", "ReasoningBank"),
}

_BENCHMARK_MAP: dict[str, tuple[str, str]] = {
    "gsm8k": ("src.benchmarks.gsm8k", "GSM8K"),
    "arc_challenge": ("src.benchmarks.arc_challenge", "ARCChallenge"),
    "gaia_l1": ("src.benchmarks.gaia_l1", "GAIAL1"),
    "humaneval": ("src.benchmarks.humaneval", "HumanEval"),
}


def run_experiment(
    strategy_name: str,
    dataset_name: str,
    model_name: str,
    budget_usd: float,
    seed: int,
    max_tasks: int,
    results_dir: Path,
) -> dict:
    """Execute one (strategy, dataset, budget, seed) cell.

    Returns a summary dict and writes two files to *results_dir*:
    - ``<run_id>.jsonl`` — one record per API call
    - ``<run_id>_summary.json`` — aggregate stats
    """
    set_seed(seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{strategy_name}__{dataset_name}__{model_name}__{seed}__{ts}"
    log_path = results_dir / f"{run_id}.jsonl"

    tracker = CostTracker(
        run_id=run_id,
        strategy=strategy_name,
        log_path=log_path,
        run_budget_usd=budget_usd,
    )

    strat_mod, strat_cls = _STRATEGY_MAP[strategy_name]
    bench_mod, bench_cls = _BENCHMARK_MAP[dataset_name]

    StratClass = getattr(importlib.import_module(strat_mod), strat_cls)
    BenchClass = getattr(importlib.import_module(bench_mod), bench_cls)

    strategy = StratClass(model_name=model_name)
    benchmark = BenchClass()

    correct = 0
    total = 0

    for task in benchmark.iter_tasks():
        if total >= max_tasks:
            break
        try:
            result = strategy.solve(task, tracker)
            is_correct = benchmark.evaluate(result.prediction, task["answer"])
            correct += int(is_correct)
            total += 1
            marker = "✓" if is_correct else "✗"
            logger.info(
                f"[{run_id}] task {total}/{max_tasks} {marker} "
                f"cost=${tracker.run_total_usd:.4f}"
            )
        except BudgetExceededError:
            logger.warning(f"Budget cap hit after {total} tasks — stopping run.")
            break
        except Exception as exc:
            logger.warning(f"Task {task.get('id', '?')} failed: {exc}")

    summary = {
        "run_id": run_id,
        "strategy": strategy_name,
        "dataset": dataset_name,
        "model": model_name,
        "budget_usd": budget_usd,
        "seed": seed,
        "n_tasks": total,
        "n_correct": correct,
        "success_rate": correct / total if total > 0 else 0.0,
        "total_cost_usd": tracker.run_total_usd,
        "n_api_calls": tracker.n_calls(),
    }

    summary_path = results_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"Run complete — SR={summary['success_rate']:.3f} "
        f"cost=${summary['total_cost_usd']:.4f} "
        f"tasks={total}"
    )
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a single pareto-bench experiment cell."
    )
    parser.add_argument(
        "--strategy", required=True, choices=list(_STRATEGY_MAP),
        help="Strategy to evaluate."
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(_BENCHMARK_MAP),
        help="Benchmark dataset."
    )
    parser.add_argument(
        "--model", required=True,
        help="Model registry key from configs/models.yaml."
    )
    parser.add_argument(
        "--budget", type=float, default=1.0,
        help="Total spend cap in USD (default: 1.00)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory to write JSONL logs and summaries."
    )

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        strategy_name=args.strategy,
        dataset_name=args.dataset,
        model_name=args.model,
        budget_usd=args.budget,
        seed=args.seed,
        max_tasks=args.max_tasks,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
