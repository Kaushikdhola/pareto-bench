# Pareto-Bench: Cost-Matched Evaluation of Multi-Agent Debate Strategies

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Pareto-Bench** is a benchmark and experimental harness that compares multi-agent debate strategies against single-agent baselines *under identical API cost budgets*. Most published comparisons pit a single model call against a multi-round, multi-agent pipeline without controlling for inference budget, making reported gains attributable to extra compute rather than the architectural design itself.

We address this gap with an **iso-cost protocol**: every strategy variant for a given task receives the same total token budget, measured in USD at current API prices. This produces Pareto frontiers over (accuracy, cost) space rather than flat accuracy tables.

---

## Motivation

The multi-agent systems literature consistently reports accuracy gains from debate-style architectures. However:

1. **No cost control.** A 3-agent, 2-round debate consumes ≈6× the tokens of a single call. Spending 6× more on a stronger single-agent strategy (e.g., self-consistency with 6 samples) is rarely the comparison point.
2. **Model homogeneity.** Debate agents typically share the same backbone; it is unclear whether gains come from architectural diversity or simple aggregation.
3. **Narrow benchmark coverage.** Most evaluations target a single benchmark, obscuring whether improvements generalise across task types.

Pareto-Bench provides a controlled scaffold to disentangle these factors.

---

## Strategies

| ID | Description |
|---|---|
| `single_agent_sc` | Single model, N samples, majority vote (self-consistency baseline) |
| `vanilla_debate` | Du et al. homogeneous debate: N agents × R rounds, majority vote |
| `role_debate` | Structured Solver / Critic / Judge pipeline |
| `hetero_debate` | Mixed model families in debate roles |
| `reasoning_bank` | Single agent augmented with a persistent memory bank of prior traces |

---

## Benchmarks

| Dataset | Task type | Metric |
|---|---|---|
| GSM8K | Grade-school math | Exact-match on final number |
| ARC-Challenge | Science MCQ | Letter accuracy |
| GAIA Level 1 | Agentic open-ended reasoning | Exact-match |
| HumanEval | Code generation | pass@1 (functional correctness) |

---

## Setup

```bash
git clone https://github.com/your-org/pareto-bench.git
cd pareto-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your API keys
```

### Environment variables

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

---

## Quickstart

Run a single experiment cell (50 tasks, $0.50 budget):

```bash
python -m src.runner \
    --strategy single_agent_sc \
    --dataset gsm8k \
    --model gpt-4o-mini \
    --budget 0.50 \
    --seed 42 \
    --max-tasks 50
```

Run the full pilot matrix (all strategies × all datasets, 50 tasks each):

```bash
python experiments/01_pilot.py
```

Results land in `results/` as JSONL cost logs plus per-run JSON summaries. Open `notebooks/01_pilot_analysis.ipynb` to analyse them.

---

## Project Structure

```
pareto-bench/
├── configs/
│   ├── models.yaml          # Model registry (provider, pricing, context window)
│   └── experiments.yaml     # Experiment matrix (strategies × datasets × budgets × seeds)
├── src/
│   ├── cost_tracking/       # CostTracker, BudgetExceededError, pricing lookup
│   ├── llm_clients/         # Abstract LLMClient + OpenAI / Anthropic / Google impls
│   ├── strategies/          # Strategy implementations
│   ├── benchmarks/          # Dataset loaders and evaluators
│   ├── utils/               # Rich logger, deterministic seeding
│   └── runner.py            # CLI orchestrator
├── experiments/             # Runnable experiment scripts
├── notebooks/               # Analysis and figure notebooks
├── paper/                   # LaTeX manuscript skeleton
├── results/                 # Output directory (contents gitignored)
└── tests/                   # pytest suite
```

---

## Reproducing Results

See [`experiments/README.md`](experiments/README.md) for per-experiment cost estimates and expected runtimes.

---

## Citation

```bibtex
@misc{pareto-bench-2025,
  title   = {Pareto-Bench: Cost-Matched Evaluation of Multi-Agent Debate Strategies for LLM Agents},
  author  = {Author, First and Author, Second},
  year    = {2025},
  url     = {https://github.com/your-org/pareto-bench},
  note    = {Preprint. arXiv:XXXX.XXXXX}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
