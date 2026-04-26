"""Tests for CostTracker: logging, budget enforcement, and JSONL persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from src.cost_tracking.tracker import BudgetExceededError, CostTracker


# ---------------------------------------------------------------------------
# Logging basics
# ---------------------------------------------------------------------------


def test_single_call_logged():
    tracker = CostTracker(strategy="test", run_id="run-001")
    record = tracker.log_call(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        task_id="task-1",
    )
    assert record.model == "gpt-4o-mini"
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.cost_usd == pytest.approx(0.001)
    assert record.strategy == "test"
    assert record.run_id == "run-001"
    assert record.task_id == "task-1"


def test_run_total_accumulates():
    tracker = CostTracker(strategy="test")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.001, "task-1")
    tracker.log_call("gpt-4o-mini", 200, 100, 0.002, "task-2")
    tracker.log_call("gpt-4o-mini", 300, 150, 0.003, "task-1")
    assert tracker.run_total_usd == pytest.approx(0.006)
    assert tracker.n_calls() == 3


def test_task_totals_tracked_independently():
    tracker = CostTracker(strategy="test")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.001, "task-1")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.002, "task-1")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.004, "task-2")
    assert tracker.task_total_usd("task-1") == pytest.approx(0.003)
    assert tracker.task_total_usd("task-2") == pytest.approx(0.004)
    assert tracker.run_total_usd == pytest.approx(0.007)


def test_task_total_defaults_to_zero_for_unknown():
    tracker = CostTracker(strategy="test")
    assert tracker.task_total_usd("nonexistent") == 0.0


def test_records_snapshot_is_copy():
    tracker = CostTracker(strategy="test")
    tracker.log_call("gpt-4o-mini", 10, 5, 0.001, "t1")
    snapshot = tracker.records()
    snapshot.clear()  # mutating the snapshot must not affect the tracker
    assert tracker.n_calls() == 1


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def test_jsonl_file_created_and_populated():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "logs" / "run.jsonl"
        tracker = CostTracker(
            strategy="test",
            run_id="run-json-001",
            log_path=log_path,
        )
        tracker.log_call("gpt-4o-mini", 200, 100, 0.002, "t-1")
        tracker.log_call("gpt-4o-mini", 300, 150, 0.003, "t-2")

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

        rec0 = json.loads(lines[0])
        assert rec0["model"] == "gpt-4o-mini"
        assert rec0["cost_usd"] == pytest.approx(0.002)
        assert rec0["run_id"] == "run-json-001"
        assert rec0["strategy"] == "test"

        rec1 = json.loads(lines[1])
        assert rec1["task_id"] == "t-2"


def test_jsonl_parent_dir_autocreated():
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = Path(tmpdir) / "a" / "b" / "c" / "run.jsonl"
        tracker = CostTracker(strategy="test", log_path=deep_path)
        tracker.log_call("gpt-4o-mini", 10, 5, 0.0001, "t-1")
        assert deep_path.exists()


def test_no_log_path_no_file():
    tracker = CostTracker(strategy="test")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.001, "t-1")
    assert tracker.n_calls() == 1  # no error raised


# ---------------------------------------------------------------------------
# Budget caps
# ---------------------------------------------------------------------------


def test_task_budget_cap_exact_boundary():
    """Call that would push task total exactly to cap is allowed; next is not."""
    tracker = CostTracker(strategy="test", task_budget_usd=0.010)
    tracker.log_call("gpt-4o-mini", 100, 50, 0.005, "t-1")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.005, "t-1")  # exactly at cap — allowed

    with pytest.raises(BudgetExceededError, match="task"):
        tracker.log_call("gpt-4o-mini", 100, 50, 0.001, "t-1")


def test_task_budget_does_not_affect_other_tasks():
    """A different task_id gets its own fresh budget."""
    tracker = CostTracker(strategy="test", task_budget_usd=0.005)
    tracker.log_call("gpt-4o-mini", 100, 50, 0.005, "task-A")

    # task-B has not spent anything yet — this should succeed.
    tracker.log_call("gpt-4o-mini", 100, 50, 0.005, "task-B")
    assert tracker.task_total_usd("task-B") == pytest.approx(0.005)


def test_run_budget_cap_stops_run():
    tracker = CostTracker(strategy="test", run_budget_usd=0.010)
    tracker.log_call("gpt-4o-mini", 100, 50, 0.005, "t-1")
    tracker.log_call("gpt-4o-mini", 100, 50, 0.004, "t-2")

    with pytest.raises(BudgetExceededError, match="run"):
        tracker.log_call("gpt-4o-mini", 100, 50, 0.002, "t-3")


def test_budget_exceeded_call_not_recorded():
    """When BudgetExceededError is raised the call must not be recorded."""
    tracker = CostTracker(strategy="test", run_budget_usd=0.005)
    tracker.log_call("gpt-4o-mini", 10, 5, 0.005, "t-1")

    with pytest.raises(BudgetExceededError):
        tracker.log_call("gpt-4o-mini", 10, 5, 0.001, "t-2")

    assert tracker.n_calls() == 1
    assert tracker.run_total_usd == pytest.approx(0.005)


def test_run_id_autogenerated():
    t1 = CostTracker(strategy="test")
    t2 = CostTracker(strategy="test")
    assert t1.run_id != t2.run_id
    assert len(t1.run_id) > 0
