"""Deterministic seeding for reproducible experiments."""

import random


def set_seed(seed: int) -> None:
    """Seed ``random`` and ``numpy`` (if available) for reproducibility.

    Call this at the start of every experiment run, before loading data
    or constructing strategies, to ensure the sampling order is identical
    across reruns with the same seed.

    Args:
        seed: Integer seed value (e.g. 42, 123, 7).
    """
    random.seed(seed)
    try:
        import numpy as np  # optional dependency at this call site

        np.random.seed(seed)
    except ImportError:
        pass
