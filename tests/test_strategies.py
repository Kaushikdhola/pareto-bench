"""Smoke tests for strategy interfaces.

These tests verify signatures, data-model field names, and the abstract
interface — they do not make real API calls.  Tests that require pydantic
are skipped automatically if the package is not installed.
"""

import pytest

pydantic = pytest.importorskip("pydantic", reason="pydantic not installed")


# ---------------------------------------------------------------------------
# StrategyResult data model
# ---------------------------------------------------------------------------


def test_strategy_result_fields():
    from src.strategies.base import StrategyResult

    r = StrategyResult(
        prediction="42",
        reasoning_trace=["step 1: add 3+4=7", "step 2: multiply by 6"],
        total_cost_usd=0.0015,
        n_api_calls=3,
    )
    assert r.prediction == "42"
    assert len(r.reasoning_trace) == 2
    assert r.total_cost_usd == pytest.approx(0.0015)
    assert r.n_api_calls == 3


def test_strategy_result_json_roundtrip():
    from src.strategies.base import StrategyResult

    r = StrategyResult(
        prediction="Paris",
        reasoning_trace=["trace"],
        total_cost_usd=0.001,
        n_api_calls=1,
    )
    as_json = r.model_dump_json()
    r2 = StrategyResult.model_validate_json(as_json)
    assert r2.prediction == r.prediction


# ---------------------------------------------------------------------------
# Strategy abstract interface
# ---------------------------------------------------------------------------


def test_strategy_is_abstract():
    from src.strategies.base import Strategy

    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_strategy_subclass_must_implement_solve():
    from src.strategies.base import Strategy

    class Incomplete(Strategy):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_strategy_subclass_can_be_instantiated():
    from src.strategies.base import Strategy, StrategyResult
    from src.cost_tracking.tracker import CostTracker

    class Dummy(Strategy):
        name = "dummy"

        def solve(self, task: dict, tracker: CostTracker) -> StrategyResult:
            return StrategyResult(
                prediction="dummy",
                reasoning_trace=[],
                total_cost_usd=0.0,
                n_api_calls=0,
            )

    d = Dummy()
    from src.cost_tracking.tracker import CostTracker as CT

    result = d.solve({"id": "x", "question": "q", "answer": "a"}, CT())
    assert result.prediction == "dummy"


# ---------------------------------------------------------------------------
# Strategy class names and attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path,class_name,expected_name",
    [
        ("src.strategies.single_agent_sc", "SingleAgentSC", "single_agent_sc"),
        ("src.strategies.vanilla_debate", "VanillaDebate", "vanilla_debate"),
        ("src.strategies.role_debate", "RoleDebate", "role_debate"),
        ("src.strategies.hetero_debate", "HeteroDebate", "hetero_debate"),
        ("src.strategies.reasoning_bank", "ReasoningBank", "reasoning_bank"),
    ],
)
def test_strategy_name_attribute(module_path: str, class_name: str, expected_name: str):
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls.name == expected_name


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("src.strategies.single_agent_sc", "SingleAgentSC"),
        ("src.strategies.vanilla_debate", "VanillaDebate"),
        ("src.strategies.role_debate", "RoleDebate"),
        ("src.strategies.reasoning_bank", "ReasoningBank"),
    ],
)
def test_strategy_accepts_model_name(module_path: str, class_name: str):
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    instance = cls(model_name="gpt-4o-mini")
    assert hasattr(instance, "model_name")
    assert instance.model_name == "gpt-4o-mini"
