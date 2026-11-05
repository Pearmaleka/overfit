"""Autocovariances and long-run variance estimators used for studentisation."""

from __future__ import annotations

import numpy as np
from scipy import fft

from overfit._utils import FloatArray

# Upper bound on the number of complex FFT coefficients held in memory at once.
_FFT_BUDGET = 2**24


def newey_west_bandwidth(n_obs: int) -> int:
    """Rule-of-thumb Bartlett bandwidth ``floor(4 (n/100)^(2/9))`` from Newey and West (1994)."""
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def autocovariances(x: FloatArray, max_lag: int | None = None) -> FloatArray:
    """Sample autocovariances ``gamma_0, ..., gamma_max_lag`` of each column of ``x``.

    Uses the biased estimator with divisor ``n`` (which keeps the implied covariance
    matrix positive semi-definite) and an FFT, so all lags cost ``O(n log n)`` per column.
    Returns an array of shape ``(max_lag + 1,)`` for 1-D input and
    ``(max_lag + 1, n_cols)`` for 2-D input.
    """
    arr = np.asarray(x, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    n, k = arr.shape
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    centred = arr - arr.mean(axis=0)
    nfft = fft.next_fast_len(2 * n - 1, real=True)
    chunk = max(1, _FFT_BUDGET // nfft)
    out = np.empty((max_lag + 1, k))
    for start in range(0, k, chunk):
        block = centred[:, start : start + chunk]
        spec = fft.rfft(block, n=nfft, axis=0)
        acov = fft.irfft(spec * np.conj(spec), n=nfft, axis=0)[: max_lag + 1]
        out[:, start : start + chunk] = acov / n
    return out[:, 0] if squeeze else out


def newey_west_lrv(x: FloatArray, lags: int | None = None) -> FloatArray:
    """Bartlett-kernel (Newey-West) long-run variance of each column of ``x``."""
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    lags = newey_west_bandwidth(n) if lags is None else int(lags)
    acov = autocovariances(arr, lags)
    if lags == 0:
        return np.asarray(acov[0])
    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    if arr.ndim == 1:
        return np.asarray(acov[0] + 2.0 * np.dot(weights, acov[1:]))
    return np.asarray(acov[0] + 2.0 * (weights[:, None] * acov[1:]).sum(axis=0))


def newey_west_lrcov(z: FloatArray, lags: int | None = None) -> FloatArray:
    """Bartlett-kernel long-run covariance matrix of the columns of ``z`` (shape ``(n, k)``)."""
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2:
        raise ValueError("z must be two-dimensional")
    n = arr.shape[0]
    lags = newey_west_bandwidth(n) if lags is None else int(lags)
    centred = arr - arr.mean(axis=0)
    cov = centred.T @ centred / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = centred[lag:].T @ centred[:-lag] / n
        cov += weight * (gamma + gamma.T)
    return cov


def stationary_bootstrap_lrv(x: FloatArray, block_size: float) -> FloatArray:
    """Variance of ``sqrt(n) * mean`` under the stationary bootstrap, for each column.

    This is the closed form of Politis and Romano (1994, Lemma 1), the estimator Hansen
    (2005) uses to studentise the SPA statistic::

        omega^2 = gamma_0 + 2 * sum_{i=1}^{n-1} kappa(n, i) * gamma_i
        kappa(n, i) = (1 - i/n) (1 - q)^i + (i/n) (1 - q)^(n - i),   q = 1 / block_size
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    acov = autocovariances(arr)
    q = 1.0 / block_size
    lags = np.arange(1, n)
    kappa = (1.0 - lags / n) * (1.0 - q) ** lags + (lags / n) * (1.0 - q) ** (n - lags)
    if arr.ndim == 1:
        return np.asarray(acov[0] + 2.0 * np.dot(kappa, acov[1:]))
    return np.asarray(acov[0] + 2.0 * (kappa[:, None] * acov[1:]).sum(axis=0))
