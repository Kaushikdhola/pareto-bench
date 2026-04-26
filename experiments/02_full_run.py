"""Full experiment: all strategy × dataset × budget × seed cells.

Run this after reviewing pilot results in notebooks/01_pilot_analysis.ipynb.
Expected cost: see experiments/README.md for up-to-date estimates.

Usage:
    python experiments/02_full_run.py [--dry-run] [--budget-tier low|mid|high]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.runner import run_experiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "experiments.yaml"


def load_matrix() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def main(dry_run: bool = False, budget_tier: str = "mid") -> None:
    config = load_matrix()

    strategies: list[str] = config["strategies"]
    datasets: list[str] = config["datasets"]
    seeds: list[int] = config["seeds"]
    budget_usd: float = config["budget_tiers"][budget_tier]["budget_usd"]
    model: str = config["default_model"]

    cells = [
        (strategy, dataset, seed)
        for strategy in strategies
        for dataset in datasets
        for seed in seeds
    ]

    results_dir = Path("results") / "full" / budget_tier
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Full run: {len(cells)} cells | budget=${budget_usd:.2f}/cell | "
        f"max_total=${len(cells) * budget_usd:.2f}"
    )

    if dry_run:
        logger.info("Dry run — no API calls made.")
        for strategy, dataset, seed in cells:
            logger.info(f"  {strategy} × {dataset} (seed={seed})")
        return

    summaries = []
    for strategy, dataset, seed in cells:
        logger.info(f"--- {strategy} × {dataset} (seed={seed}, budget=${budget_usd}) ---")
        try:
            summary = run_experiment(
                strategy_name=strategy,
                dataset_name=dataset,
                model_name=model,
                budget_usd=budget_usd,
                seed=seed,
                max_tasks=500,
                results_dir=results_dir,
            )
            summaries.append(summary)
        except Exception as exc:
            logger.error(f"Cell failed: {exc}")

    logger.info(
        f"Full run complete. {len(summaries)}/{len(cells)} cells succeeded. "
        f"Total cost: ${sum(s['total_cost_usd'] for s in summaries):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--budget-tier", choices=["low", "mid", "high"], default="mid"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, budget_tier=args.budget_tier)
