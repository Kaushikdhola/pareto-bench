# pareto-bench — Resumption Handoff

> **For AI coding assistants (Copilot, Claude Code, Cursor, etc.):** 
> This is a research project resuming after a 1-week pause. Read `CONTEXT.md` 
> first for the full project brief. This document tells you exactly what's 
> done, what's next, and how to help without breaking the existing structure.

---

## TL;DR for the assistant

- **Project:** `pareto-bench` — a cost-matched benchmark comparing 
  multi-agent debate strategies vs single-agent baselines for LLM agents.
- **Output:** arXiv preprint + reusable benchmark repo.
- **Constraints:** solo researcher, ~3 weeks remaining, $50 total API 
  budget, API-only (no GPU, no training).
- **Status:** scaffolding complete, 27/27 tests passing, Anthropic client 
  working. Need to wire OpenAI + Google clients, implement benchmark 
  loaders, wire the runner, and run a smoke test before any pilot.

---

## What's already done (do not re-do)

| Component | Status | Location |
|---|---|---|
| Repo scaffolding (46 files) | ✅ Complete | repo root |
| `CostTracker` (JSONL logging, budget caps) | ✅ Working + tested | `src/cost_tracking/tracker.py` |
| Pricing loader | ✅ Working | `src/cost_tracking/pricing.py` |
| Anthropic client (with tenacity retry) | ✅ Working | `src/llm_clients/anthropic_client.py` |
| Strategy abstract base + 5 strategy stubs | ✅ Signatures correct | `src/strategies/` |
| Benchmark abstract base | ✅ Done | `src/benchmarks/base.py` |
| LaTeX paper skeleton | ✅ Sections + bib | `paper/main.tex` |
| 27 unit tests | ✅ All passing | `tests/` |

**Verification command:**
```bash
pytest tests/ -v
```
All 27 tests must pass before any new work begins. If they don't, fix 
that first.

---

## Critical invariants (do not violate)

1. **Every API call goes through `CostTracker`.** No exceptions.
2. **Pricing is loaded from `configs/models.yaml`**, never hardcoded.
3. **All strategies share the same `LLMClient` interface.** Cost is the 
   only axis of comparison.
4. **No strategy imports another strategy.** Independent and comparable.
5. **Match existing code style:** type hints, pydantic v2, rich logging, 
   docstrings, tenacity for retries.
6. **Use `configs/models.yaml` model IDs verbatim** — don't invent names.

---

## The work queue (do these in order)

### TASK 1 — OpenAI client with retry
**File:** `src/llm_clients/openai_client.py`

The Anthropic client is the reference. Match its structure exactly: 
same retry decorator pattern, same `CompletionResult` return type, same 
cost-tracker integration. The OpenAI client currently has the right 
signature but no retry logic.

Retryable exceptions: `openai.RateLimitError`, `openai.InternalServerError`, 
`openai.APIConnectionError`, `openai.APITimeoutError`. Tenacity config: 
5 attempts, exponential backoff 2–60 seconds.

Use `client.chat.completions.create()`. Token counts from 
`response.usage.prompt_tokens` and `response.usage.completion_tokens`.

**Acceptance:** all 27 tests still pass; new test added covering 
`OpenAIClient.complete()` with a mocked SDK.

---

### TASK 2 — Google (Gemini) client with retry
**File:** `src/llm_clients/google_client.py`

Same pattern as Anthropic and OpenAI. Use `google-generativeai` SDK.

Retryable exceptions: `google.api_core.exceptions.ResourceExhausted`, 
`ServiceUnavailable`, `InternalServerError`, `DeadlineExceeded`. Same 
tenacity config.

Token counts from `response.usage_metadata.prompt_token_count` and 
`response.usage_metadata.candidates_token_count`.

**Acceptance:** all tests pass; new test added with mocked SDK.

---

### TASK 3 — Benchmark loaders
**Files:** 
- `src/benchmarks/gsm8k.py`
- `src/benchmarks/arc_challenge.py`
- `src/benchmarks/gaia_l1.py`
- `src/benchmarks/humaneval.py`

All subclass `Benchmark` from `src/benchmarks/base.py`. Each implements 
`iter_tasks()` and `evaluate(prediction, ground_truth) -> bool`.

Use HuggingFace `datasets`. Dataset IDs:
- GSM8K: `gsm8k`, config `main`, split `test`
- ARC-Challenge: `allenai/ai2_arc`, config `ARC-Challenge`, split `test`
- GAIA: `gaia-benchmark/GAIA`, config `2023_level1`, split `validation`
- HumanEval: `openai_humaneval`, split `test`

Each `iter_tasks()` yields `Task` pydantic models with `task_id`, 
`question`, `ground_truth`.

Evaluation logic:
- GSM8K: extract final number (regex `#### (\d+)` or last number), 
  compare to ground truth.
- ARC: extract letter (A/B/C/D), compare.
- GAIA: official GAIA exact-match scorer (string normalization + match).
- HumanEval: pass@1 — run test suite against predicted code in a 
  subprocess with timeout.

Support `--limit N` for pilot runs.

**Acceptance:** `tests/test_benchmarks.py` loads 2 tasks from each 
benchmark and asserts structure; all tests pass.

---

### TASK 4 — Runner CLI
**Files:** `src/runner.py`, `experiments/01_pilot.py`

Use `argparse` (not click). CLI signature:
```bash
python -m src.runner \
  --strategy single_agent_sc \
  --dataset gsm8k \
  --model claude-haiku-4-5 \
  --n-tasks 50 \
  --seed 42 \
  --run-budget-usd 1.00 \
  --output-dir results/pilot/
```

Behavior:
- Load benchmark, slice to `n_tasks` deterministically by seed.
- Instantiate strategy with configured model.
- Wrap whole run in `CostTracker(run_budget_usd=...)`.
- For each task: `strategy.solve(task, tracker)`, evaluate, append 
  result to JSONL.
- On `BudgetExceededError`: log clearly, save partial results, exit 
  cleanly with non-zero code.
- At end: print summary (tasks attempted, SR, total cost, median 
  cost/task, p95 cost/task).

`experiments/01_pilot.py` runs: 5 strategies × 4 datasets × 1 model 
(Claude Haiku) × 50 tasks × 1 seed. Total cap: $5. Add `--dry-run` 
flag that prints the matrix without API calls.

**Acceptance:** `python experiments/01_pilot.py --dry-run` prints a 
clean matrix summary; all 27+ tests pass.

---

### TASK 5 — $1 smoke test (the gate)
Run a real $1 smoke test before any pilot:
```bash
python -m src.runner \
  --strategy single_agent_sc \
  --dataset gsm8k \
  --model claude-haiku-4-5 \
  --n-tasks 20 \
  --seed 42 \
  --run-budget-usd 1.00 \
  --output-dir results/smoke/
```

Verify:
- JSONL output is well-formed; one row per task.
- Cost log JSONL is well-formed; one row per API call.
- Total run cost matches manual sum from cost log.
- SR on GSM8K with single-agent SC + Claude Haiku is in 80–90% range.
- All tests still pass.

**This is the gate.** Don't run the full pilot until smoke passes.

---

## Sanity checks before starting (5 minutes)

1. **Verify `configs/models.yaml` pricing** against current provider docs:
   - https://openai.com/api/pricing/
   - https://www.anthropic.com/pricing
   - https://ai.google.dev/pricing
   If anything looks off, fix before any API call.
2. **Confirm `.env` has working keys** for at least Anthropic. OpenAI 
   and Google can be added when needed.
3. **Run `pytest tests/ -v`** — all 27 must pass.

---

## How the assistant should help

- Read `CONTEXT.md` for full project context.
- Stay within existing abstractions. Subclass `LLMClient`, `Strategy`, 
  `Benchmark`. Don't introduce new layers.
- Match existing code style (type hints, pydantic, rich logging).
- Add tests for any non-trivial logic.
- Be cost-conscious. Default to `--n-tasks 20` pilots. Never suggest 
  full-set runs without explicit user approval.
- When uncertain, ask. Don't invent design decisions.

---

## Where I am in the project

**Day equivalent:** ~Day 2 of 21 (scaffolding done, no API calls made yet).

**Next milestone:** Tasks 1–5 complete → smoke test passes. After that, 
run the full $5 pilot, then design the iso-cost calibration step.

**Money spent so far:** $0.