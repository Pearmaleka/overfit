"""Small helpers shared across modules: input validation and random number generators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
SeedLike = int | np.random.Generator | None


def as_matrix(
    x: ArrayLike | pd.DataFrame | pd.Series, name: str = "returns"
) -> tuple[FloatArray, list[str]]:
    """Convert ``x`` to a finite 2-D float array of shape (n_obs, n_series) plus column labels."""
    labels: list[str]
    if isinstance(x, pd.DataFrame):
        labels = [str(c) for c in x.columns]
        arr = x.to_numpy(dtype=float)
    elif isinstance(x, pd.Series):
        labels = [str(x.name) if x.name is not None else "0"]
        arr = x.to_numpy(dtype=float).reshape(-1, 1)
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        labels = [str(i) for i in range(arr.shape[1])] if arr.ndim == 2 else []
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got an array with {arr.ndim} dimensions")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} needs at least 2 observations and 1 series, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, labels


def as_vector(x: ArrayLike | pd.Series, name: str = "returns") -> FloatArray:
    """Convert ``x`` to a finite 1-D float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def make_rng(seed: SeedLike) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed or an existing generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def check_labels(labels: Sequence[str], n: int) -> list[str]:
    out = [str(v) for v in labels]
    if len(out) != n:
        raise ValueError(f"expected {n} labels, got {len(out)}")
    return out
