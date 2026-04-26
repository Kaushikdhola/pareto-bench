"""Abstract benchmark interface."""

from abc import ABC, abstractmethod
from typing import Iterator


class Benchmark(ABC):
    """Abstract base class for all benchmark loaders.

    Subclasses must implement :meth:`iter_tasks` (yielding task dicts) and
    :meth:`evaluate` (checking a prediction against the ground-truth string).
    """

    name: str = "base"

    @abstractmethod
    def iter_tasks(self) -> Iterator[dict]:
        """Yield task dicts in deterministic order.

        Each dict must contain at minimum:
            - ``"id"``: unique task identifier (str)
            - ``"question"``: the prompt sent to the strategy (str)
            - ``"answer"``: ground-truth string used by :meth:`evaluate` (str)

        Additional keys (e.g. ``"choices"`` for MCQ) are benchmark-specific.
        """
        ...

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """Return True if *prediction* is considered correct.

        Args:
            prediction: Raw string output from the strategy.
            ground_truth: Ground-truth string from the dataset.
        """
        ...
