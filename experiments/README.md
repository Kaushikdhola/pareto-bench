# Running Experiments

## Prerequisites

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

---

## Experiment scripts

### 01 — Pilot (recommended first step)

Runs 50 tasks per strategy × dataset cell, one seed, `$0.50` budget each.

```bash
python experiments/01_pilot.py
# Dry-run (prints cells without making API calls):
python experiments/01_pilot.py --dry-run
```

| Property | Value |
|---|---|
| Cells | 5 strategies × 4 datasets = 20 |
| Seeds | 1 |
| Budget / cell | $0.50 |
| **Max total cost** | **~$10** |
| Estimated wall time | 30–90 min (API latency varies) |

Results land in `results/pilot/`.

---

### 02 — Full run

Runs the complete matrix across all seeds and budget tiers.

```bash
# Mid budget tier (default):
python experiments/02_full_run.py

# Specific tier:
python experiments/02_full_run.py --budget-tier low
python experiments/02_full_run.py --budget-tier high

# Dry-run:
python experiments/02_full_run.py --dry-run
```

| Budget tier | $/cell | Cells | Seeds | **Max total** |
|---|---|---|---|---|
| `low` | $0.50 | 20 | 3 | **~$30** |
| `mid` | $2.00 | 20 | 3 | **~$120** |
| `high` | $5.00 | 20 | 3 | **~$300** |

Estimated wall time for `mid` tier: 4–10 hours.

---

## Output format

Each run produces two files in the results directory:

- `<run_id>.jsonl` — one JSON line per API call with fields:
  `model, input_tokens, output_tokens, cost_usd, timestamp, strategy, task_id, run_id`
- `<run_id>_summary.json` — aggregate stats:
  `run_id, strategy, dataset, model, budget_usd, seed, n_tasks, n_correct, success_rate, total_cost_usd, n_api_calls`

---

## Analysis

Open the notebooks in order:

1. `notebooks/01_pilot_analysis.ipynb` — review pilot; decide which cells to include in full run
2. `notebooks/02_pareto_plots.ipynb` — Pareto frontiers over (cost, accuracy) space
3. `notebooks/03_paper_figures.ipynb` — publication-ready figures and Table 1
