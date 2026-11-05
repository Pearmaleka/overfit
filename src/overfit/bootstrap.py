"""Block bootstraps for dependent data and automatic block-length selection.

References
----------
Politis and Romano (1994), "The Stationary Bootstrap", JASA.
Politis and White (2004), "Automatic Block-Length Selection for the Dependent
Bootstrap", Econometric Reviews; corrected by Patton, Politis and White (2009).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from overfit._lrv import autocovariances
from overfit._utils import FloatArray, SeedLike, as_matrix, make_rng

BootstrapMethod = Literal["stationary", "circular", "moving"]
IntArray = NDArray[np.intp]

# Number of resampled indices generated per chunk when building count matrices.
_INDEX_BUDGET = 2**22


def _check(n: int, block_size: float, n_boot: int) -> None:
    if n < 2:
        raise ValueError("need at least 2 observations")
    if not block_size >= 1:
        raise ValueError("block_size must be at least 1")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")


def _stationary(n: int, block_size: float, n_boot: int, rng: np.random.Generator) -> IntArray:
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < 1.0 / block_size
    new_block[:, 0] = True
    t = np.arange(n)
    last_start = np.where(new_block, t, 0)
    np.maximum.accumulate(last_start, axis=1, out=last_start)
    idx = np.take_along_axis(starts, last_start, axis=1) + (t - last_start)
    return np.asarray(idx % n, dtype=np.intp)


def _fixed_blocks(
    n: int, block_size: float, n_boot: int, rng: np.random.Generator, *, circular: bool
) -> IntArray:
    length = max(1, round(block_size))
    length = min(length, n)
    n_blocks = -(-n // length)
    high = n if circular else n - length + 1
    starts = rng.integers(0, high, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(length)).reshape(n_boot, -1)[:, :n]
    return np.asarray(idx % n if circular else idx, dtype=np.intp)


def bootstrap_indices(
    n: int,
    *,
    block_size: float,
    n_boot: int,
    method: BootstrapMethod = "stationary",
    seed: SeedLike = None,
) -> IntArray:
    """Resampled time indices, shape ``(n_boot, n)``.

    ``stationary``: blocks with geometric lengths (mean ``block_size``) that wrap
    around the end of the sample (Politis and Romano, 1994).
    ``circular``: blocks of fixed length ``round(block_size)`` with wrap-around.
    ``moving``: blocks of fixed length without wrap-around.
    """
    _check(n, block_size, n_boot)
    rng = make_rng(seed)
    if method == "stationary":
        return _stationary(n, block_size, n_boot, rng)
    if method == "circular":
        return _fixed_blocks(n, block_size, n_boot, rng, circular=True)
    if method == "moving":
        return _fixed_blocks(n, block_size, n_boot, rng, circular=False)
    raise ValueError("method must be 'stationary', 'circular' or 'moving'")


def bootstrap_counts(
    n: int,
    *,
    block_size: float,
    n_boot: int,
    method: BootstrapMethod = "stationary",
    seed: SeedLike = None,
) -> FloatArray:
    """How often each observation appears in each bootstrap sample, shape ``(n_boot, n)``.

    Bootstrap means of many series are then a single matrix product:
    ``counts @ X / n`` gives the ``(n_boot, n_series)`` matrix of resampled means
    without materialising any resampled data set.
    """
    _check(n, block_size, n_boot)
    rng = make_rng(seed)
    counts = np.empty((n_boot, n))
    chunk = max(1, _INDEX_BUDGET // n)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = bootstrap_indices(n, block_size=block_size, n_boot=size, method=method, seed=rng)
        flat = (idx + n * np.arange(size)[:, None]).ravel()
        counts[start : start + size] = np.bincount(flat, minlength=size * n).reshape(size, n)
    return counts


def _optimal_block_1d(x: FloatArray) -> tuple[float, float]:
    n = x.shape[0]
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    k_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    acov = autocovariances(x, m_max)
    if acov[0] <= 0:
        return 1.0, 1.0
    rho = acov[1:] / acov[0]
    small = np.abs(rho) < 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(1, len(small) - k_n + 2):
        if small[m - 1 : m - 1 + k_n].all():
            m_hat = m
            break
    big_m = m_max if m_hat is None else min(2 * m_hat, m_max)
    big_m = min(big_m, len(acov) - 1)
    k = np.arange(1, big_m + 1)
    lam = np.where(k / big_m <= 0.5, 1.0, 2.0 * (1.0 - k / big_m))
    g_hat = 2.0 * float(np.sum(lam * k * acov[1 : big_m + 1]))
    lrv = float(acov[0] + 2.0 * np.sum(lam * acov[1 : big_m + 1]))
    if lrv <= 0 or g_hat == 0:
        return 1.0, 1.0
    b_sb = (g_hat**2 / lrv**2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_cb = (1.5 * g_hat**2 / lrv**2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return min(b_sb, b_max), min(b_cb, b_max)


def optimal_block_length(x: ArrayLike | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Politis-White block length for the stationary and circular bootstraps.

    Uses the flat-top lag window and the corrected constants of Patton, Politis and
    White (2009): ``b = (2 G^2 / D)^(1/3) n^(1/3)`` with ``D = 2 g(0)^2`` for the
    stationary bootstrap and ``D = 4/3 g(0)^2`` for the circular bootstrap.

    Returns a DataFrame with columns ``stationary`` and ``circular``, one row per series.
    """
    arr, labels = as_matrix(x, name="x")
    rows = [_optimal_block_1d(arr[:, j]) for j in range(arr.shape[1])]
    return pd.DataFrame(rows, index=labels, columns=["stationary", "circular"])


def auto_block_size(x: FloatArray, *, max_series: int = 50) -> float:
    """Median stationary-bootstrap block length over (at most ``max_series``) columns, at least 1.

    Joint bootstraps must use one block length for every series; the median of the
    per-series Politis-White estimates is a simple, robust choice. When there are many
    columns, the ones with the largest means are used, since they drive max-type tests.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] > max_series:
        top = np.argsort(arr.mean(axis=0))[-max_series:]
        arr = arr[:, top]
    lengths = [_optimal_block_1d(arr[:, j])[0] for j in range(arr.shape[1])]
    return float(max(1.0, np.median(lengths)))
