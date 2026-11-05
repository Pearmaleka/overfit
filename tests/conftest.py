from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261105)


def ar1(n: int, phi: float, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """AR(1) series started from its stationary distribution."""
    e = rng.standard_normal(n) * scale
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x
