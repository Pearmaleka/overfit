"""Probability of backtest overfitting (PBO) by combinatorially symmetric cross-validation.

Reference: Bailey, Borwein, Lopez de Prado and Zhu (2017), "The Probability of Backtest
Overfitting", Journal of Computational Finance 20(4).

The sample is cut into ``S`` blocks. For every way of choosing ``S/2`` blocks as the
in-sample (IS) set, the strategy with the best IS performance is selected and its
performance rank among all strategies on the remaining out-of-sample (OOS) blocks is
recorded. PBO is the fraction of splits in which the IS winner ranks at or below the
OOS median, i.e. the logit of its relative rank is ``<= 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from overfit._utils import FloatArray, SeedLike, as_matrix, make_rng

MetricName = Literal["sharpe", "mean", "tstat"]
MetricFunc = Callable[[FloatArray], FloatArray]

_CHUNK_BUDGET = 2**22


@dataclass(frozen=True)
class PBOResult:
    """Outcome of CSCV. Arrays have one entry per train/test combination."""

    pbo: float
    logits: FloatArray
    is_performance: FloatArray
    oos_performance: FloatArray
    oos_rank: FloatArray
    selected: NDArray[np.intp]
    labels: list[str]
    n_splits: int
    metric: str

    @property
    def n_combinations(self) -> int:
        return int(self.logits.shape[0])

    @property
    def prob_oos_loss(self) -> float:
        """Share of splits in which the IS winner has negative OOS performance."""
        return float(np.mean(self.oos_performance < 0))

    @property
    def degradation(self) -> tuple[float, float]:
        """Slope and intercept of the OLS fit of OOS on IS performance of the winner."""
        x, y = self.is_performance, self.oos_performance
        var = float(np.var(x))
        if var == 0:
            return float("nan"), float(np.mean(y))
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
        return slope, float(y.mean() - slope * x.mean())

    @property
    def selection_frequency(self) -> pd.Series:
        """How often each strategy was the IS winner, most frequent first."""
        counts = np.bincount(self.selected, minlength=len(self.labels))
        freq = pd.Series(counts / counts.sum(), index=self.labels, name="frequency")
        return freq[freq > 0].sort_values(ascending=False)


def _metric_from_sums(name: str, n: FloatArray, s1: FloatArray, s2: FloatArray) -> FloatArray:
    mean = s1 / n
    if name == "mean":
        return np.asarray(mean)
    var = (s2 - s1 * mean) / (n - 1.0)
    var = np.where(var > 1e-12 * s2 / n, var, 0.0)  # constant series give rounding noise
    sd = np.sqrt(var)
    sharpe = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    if name == "sharpe":
        return np.asarray(sharpe)
    return np.asarray(sharpe * np.sqrt(n))


def _relative_rank(oos: FloatArray, selected: NDArray[np.intp]) -> FloatArray:
    """Rank (1 = worst, ties averaged) of each row's selected column, divided by N + 1."""
    rows = np.arange(oos.shape[0])
    chosen = oos[rows, selected][:, None]
    below = np.count_nonzero(oos < chosen, axis=1)
    ties = np.count_nonzero(oos == chosen, axis=1)
    return np.asarray((below + (ties + 1) / 2.0) / (oos.shape[1] + 1.0))


def probability_of_backtest_overfitting(
    returns: ArrayLike | pd.DataFrame,
    *,
    n_splits: int = 16,
    metric: MetricName | MetricFunc = "sharpe",
    max_combinations: int | None = None,
    seed: SeedLike = None,
) -> PBOResult:
    """Estimate the probability of backtest overfitting with CSCV.

    Parameters
    ----------
    returns
        ``(n_obs, n_strategies)`` returns of every configuration that was tried,
        aligned in time.
    n_splits
        Even number ``S`` of contiguous blocks; ``C(S, S/2)`` combinations are evaluated
        (12,870 for the default ``S = 16``).
    metric
        ``"sharpe"``, ``"mean"``, ``"tstat"`` (computed from block sums, fast), or a
        function mapping an ``(n, n_strategies)`` array to ``n_strategies`` scores.
    max_combinations
        Evaluate a random subset of combinations when there are more than this.
    """
    x, labels = as_matrix(returns)
    n_obs, n_strat = x.shape
    if n_strat < 2:
        raise ValueError("PBO needs at least two strategies")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even number >= 2")
    if n_splits > n_obs:
        raise ValueError("n_splits cannot exceed the number of observations")

    blocks = np.array_split(np.arange(n_obs), n_splits)
    combos = np.array(list(combinations(range(n_splits), n_splits // 2)), dtype=np.intp)
    if max_combinations is not None and combos.shape[0] > max_combinations:
        rng = make_rng(seed)
        keep = np.sort(rng.choice(combos.shape[0], size=max_combinations, replace=False))
        combos = combos[keep]
    n_comb = combos.shape[0]
    in_sample = np.zeros((n_comb, n_splits))
    in_sample[np.arange(n_comb)[:, None], combos] = 1.0

    selected = np.empty(n_comb, dtype=np.intp)
    is_perf = np.empty(n_comb)
    oos_perf = np.empty(n_comb)
    rank = np.empty(n_comb)

    if callable(metric):
        metric_name = getattr(metric, "__name__", "custom")
        for i, combo in enumerate(combos):
            is_rows = np.concatenate([blocks[j] for j in combo])
            oos_rows = np.concatenate([blocks[j] for j in range(n_splits) if j not in set(combo)])
            perf_is = np.asarray(metric(x[is_rows]), dtype=float)
            perf_oos = np.asarray(metric(x[oos_rows]), dtype=float)
            best = int(np.argmax(perf_is))
            selected[i] = best
            is_perf[i] = perf_is[best]
            oos_perf[i] = perf_oos[best]
            rank[i] = _relative_rank(perf_oos[None, :], np.array([best]))[0]
    else:
        if metric not in ("sharpe", "mean", "tstat"):
            raise ValueError("metric must be 'sharpe', 'mean', 'tstat' or a callable")
        metric_name = metric
        sizes = np.array([len(b) for b in blocks], dtype=float)
        s1 = np.stack([x[b].sum(axis=0) for b in blocks])
        s2 = np.stack([(x[b] ** 2).sum(axis=0) for b in blocks])
        chunk = max(1, _CHUNK_BUDGET // n_strat)
        for start in range(0, n_comb, chunk):
            sl = slice(start, start + chunk)
            w_is = in_sample[sl]
            w_oos = 1.0 - w_is
            perf_is = _metric_from_sums(metric, (w_is @ sizes)[:, None], w_is @ s1, w_is @ s2)
            perf_oos = _metric_from_sums(metric, (w_oos @ sizes)[:, None], w_oos @ s1, w_oos @ s2)
            winners = np.argmax(perf_is, axis=1)
            rows = np.arange(winners.shape[0])
            selected[sl] = winners
            is_perf[sl] = perf_is[rows, winners]
            oos_perf[sl] = perf_oos[rows, winners]
            rank[sl] = _relative_rank(perf_oos, winners)

    logits = np.log(rank / (1.0 - rank))
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_performance=is_perf,
        oos_performance=oos_perf,
        oos_rank=rank,
        selected=selected,
        labels=labels,
        n_splits=n_splits,
        metric=metric_name,
    )


pbo = probability_of_backtest_overfitting
