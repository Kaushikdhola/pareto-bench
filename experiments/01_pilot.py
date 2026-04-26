"""Pilot experiment: 50 tasks per strategy × dataset cell.

Run this script to get a fast first read on which strategies are promising
before committing to the full experiment budget.

Usage:
    python experiments/01_pilot.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when running as a script.
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.runner import run_experiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

STRATEGIES = [
    "single_agent_sc",
    "vanilla_debate",
    "role_debate",
    "hetero_debate",
    "reasoning_bank",
]

DATASETS = ["gsm8k", "arc_challenge", "gaia_l1", "humaneval"]

MODEL = "gpt-4o-mini"
BUDGET_USD = 0.50
MAX_TASKS = 50
SEEDS = [42]  # One seed for the pilot; expand to [42, 123, 7] for full run.


def main(dry_run: bool = False) -> None:
    results_dir = Path("results") / "pilot"
    results_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        (strategy, dataset)
        for strategy in STRATEGIES
        for dataset in DATASETS
    ]

    logger.info(f"Pilot: {len(cells)} cells × {len(SEEDS)} seed(s) = {len(cells) * len(SEEDS)} runs")
    logger.info(f"Max cost estimate: ${len(cells) * len(SEEDS) * BUDGET_USD:.2f}")

    if dry_run:
        logger.info("Dry run — no API calls made.")
        return

    summaries = []
    for seed in SEEDS:
        for strategy, dataset in cells:
            logger.info(f"--- {strategy} × {dataset} (seed={seed}) ---")
            try:
                summary = run_experiment(
                    strategy_name=strategy,
                    dataset_name=dataset,
                    model_name=MODEL,
                    budget_usd=BUDGET_USD,
                    seed=seed,
                    max_tasks=MAX_TASKS,
                    results_dir=results_dir,
                )
                summaries.append(summary)
            except Exception as exc:
                logger.error(f"Cell failed: {exc}")

    logger.info(f"Pilot complete. {len(summaries)}/{len(cells) * len(SEEDS)} cells succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
