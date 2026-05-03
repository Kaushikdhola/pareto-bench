# pareto-bench — Project Context

> **For AI coding assistants:** This document is the single source of truth 
> for this project. Read it fully before suggesting code. The user is a 
> solo researcher working alongside a full-time job, on a tight timeline, 
> with a $50 total API budget. Optimize for: correctness, reproducibility, 
> minimal token waste, and clean code another researcher could fork.

---

## 1. Project pitch (one paragraph)

**pareto-bench** is a cost-matched benchmark for multi-agent LLM 
strategies. Existing comparisons (multi-agent debate vs. single-agent 
self-consistency vs. memory-augmented agents) are reported at iso-rounds 
or iso-agents, not iso-dollars. This project produces a clean cost-Pareto 
frontier across reasoning and coding tasks so practitioners can answer: 
*"At $X per task, which strategy wins?"* The deliverable is an arXiv 
preprint plus a public benchmark repo.

---

## 2. Research question

At matched dollar cost, when does multi-agent debate beat single-agent 
self-consistency, and does mixing model families (heterogeneous debate) 
provide gains beyond same-family debate?

**Sub-questions:**
- Does the answer depend on task type (reasoning vs. coding)?
- Does the answer depend on absolute budget ($0.02/task vs. $0.10/task)?

---

## 3. Contribution claims (what the paper will argue)

1. **A reusable iso-cost evaluation protocol** for multi-agent LLM 
   strategies, with open-source implementation.
2. **Empirical cost-Pareto frontiers** across 2 datasets (GSM8K, HumanEval) 
   and 3 strategies.
3. **Evidence on heterogeneous-model debate**: whether mixing 
   GPT-4o-mini + Claude Haiku + Gemini Flash beats 3× one model at 
   matched cost.
4. **Practical guidance** for practitioners on strategy selection by 
   task type and budget.

---

## 4. Scope and constraints

| Constraint | Value |
|---|---|
| Solo researcher | Yes — no parallel collaborators |
| Timeline | ~10 working days remaining |
| API budget | $50 total across all providers |
| Compute | API-only. No GPU training. No fine-tuning. |
| Providers | OpenAI, Anthropic, Google (Anthropic confirmed) |
| Languages | Python 3.11+ |

**Out of scope:** model training, RLHF, fine-tuning, agentic web/tool use, 
custom benchmark construction. We use existing benchmarks only.

---

## 5. Strategies under comparison (cut to 3 for v1)

Each strategy implements `Strategy.solve(task, tracker) -> StrategyResult`.

1. **`single_agent_sc`** — Single-agent self-consistency. Generate N 
   samples with temperature > 0, take majority vote on final answer.
2. **`vanilla_debate`** — Du et al. 2024 style. 3 agents, same model, 
   R rounds of debate, majority vote on final round.
3. **`hetero_debate`** — Same as vanilla_debate but each agent uses a 
   different model family (GPT-4o-mini, Claude Haiku, Gemini Flash).

**Stretch (only if time allows):**
4. `role_debate` — Solver / Critic / Judge roles
5. `reasoning_bank` — Single-agent + simple memory bank baseline

**Key design rule:** all strategies share the same `LLMClient` interface 
and the same `CostTracker`. Cost is the *only* axis of comparison — task 
correctness is measured per-benchmark.

---

## 6. Benchmarks (cut to 2 for v1)

Each benchmark implements `Benchmark.iter_tasks()` and 
`Benchmark.evaluate(prediction, ground_truth) -> bool`.

| Benchmark | Why | Sample size for paper |
|---|---|---|
| GSM8K | Math reasoning, cheap, well-known | 100-200 tasks |
| HumanEval | Code generation, programmatic eval | 164 (full set) |

All loaded from HuggingFace `datasets`. No custom data construction.

---

## 7. Iso-cost protocol (the core methodological contribution)

For each (strategy, dataset, budget) cell:

1. Pick a target budget per task: e.g. **$0.02 and $0.10**.
2. For each strategy, **calibrate** its hyperparameters (N samples, 
   R rounds, max_tokens) so that the *median* per-task cost on a 
   100-task calibration set lands within ±15% of the target budget.
3. Run the calibrated strategy on the held-out evaluation set.
4. Report: success rate (SR), median cost/task, p95 cost/task, total cost.

**Why this matters:** prior work compares strategies at iso-rounds or 
iso-N, which conflates "more compute" with "better strategy." Iso-cost 
isolates the strategy effect.

---

## 8. Tech stack

| Layer | Choice |
|---|---|
| Package manager | uv or pip + venv |
| Build backend | hatchling |
| Linter / formatter | ruff + black |
| Tests | pytest |
| Data validation | pydantic v2 |
| API retry | tenacity |
| Logging | rich |
| Plotting | matplotlib + seaborn |
| Config | YAML via pyyaml |

**No frameworks like LangChain, AutoGen, or CrewAI.** We write the 
strategies directly so the comparison is clean and the code is 
auditable. This is a deliberate choice.

---

## 9. Code style and conventions

- **Type hints everywhere.** mypy-clean is the bar.
- **Pydantic models** for all structured data crossing module boundaries 
  (`CompletionResult`, `StrategyResult`, `TaskResult`).
- **Docstrings** on every public class and function. Google style.
- **No magic numbers.** All thresholds and budgets in `configs/`.
- **Logging via rich**, not print. Logger named after the module.
- **Errors are typed.** `BudgetExceededError`, `RateLimitError`, etc.
- **Tests for load-bearing code.** Cost tracking and pricing must have 
  tests. Strategies need at least smoke tests with a mock LLM client.
- **Determinism.** Every experiment script seeds numpy + random + sets 
  temperature explicitly. Results must be reproducible from a seed.
- **Cost-first thinking.** Every API call goes through the tracker. No 
  exceptions. Bypassing the tracker is a bug.

---

## 10. Critical invariants (do not violate)

1. **Every API call is logged to `CostTracker` before the result is 
   returned to the caller.** This is non-negotiable. The whole project 
   depends on accurate cost accounting.
2. **Pricing is loaded from `configs/models.yaml`, not hardcoded.** 
   Provider prices change. Hardcoding them in client code creates silent 
   drift.
3. **No strategy may import another strategy.** They must be 
   independently runnable and comparable.
4. **No benchmark may be modified after results are run.** If a benchmark 
   loader changes, results from before the change are invalidated.
5. **The `experiments/` scripts must be runnable end-to-end from a clean 
   clone** with only `.env` populated. No notebook-only state.

---

## 11. How to help (for AI coding assistants)

When asked to implement or modify code in this repo:

1. **Read this doc and HANDOFF.md first.** Don't guess at scope.
2. **Stay within the existing abstractions.** If a base class exists, 
   subclass it. Don't introduce new abstractions without flagging.
3. **Do not bypass `CostTracker`.** Every API call goes through it.
4. **Do not hardcode model names or prices.** Use `configs/models.yaml`.
5. **Match the existing code style** (type hints, pydantic, rich logging, 
   docstrings).
6. **Suggest tests** for any non-trivial logic you add.
7. **Be cost-conscious in suggested experiments.** This is a $50 total 
   project. A suggestion that says "run it on the full 1000-task split" 
   is wrong. Default to small pilots first.
8. **When uncertain, ask.** Don't invent design decisions.

---

## 12. Key references

- **MAST (failure taxonomy):** Cemri et al., "Why Do Multi-Agent LLM 
  Systems Fail?" arXiv:2503.13657, 2025. NeurIPS 2025 D&B Spotlight.
- **ReasoningBank:** Ouyang et al., "ReasoningBank: Scaling Agent 
  Self-Evolving with Reasoning Memory." arXiv:2509.25140, 2025.
- **Multi-agent debate:** Du et al., "Improving Factuality and Reasoning 
  in Language Models through Multiagent Debate." ICML 2024.
- **Self-consistency:** Wang et al., "Self-Consistency Improves Chain of 
  Thought Reasoning in Language Models." ICLR 2023.
- **Debate or Vote:** OpenReview submission, ICLR 2026. Argues vote 
  alone explains most MAD gains.

---

## 13. Glossary

- **Iso-cost:** matching strategies at equal dollar cost per task, not 
  equal compute or equal rounds.
- **MAS:** multi-agent system.
- **MAD:** multi-agent debate.
- **SR:** success rate (fraction of tasks correctly solved).
- **SC:** self-consistency (sampling N answers, majority vote).
- **TTS:** test-time scaling (spending more compute at inference).