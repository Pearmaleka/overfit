"""Data-snooping tests: is the best of many strategies really better than a benchmark?

Three procedures share one block bootstrap of every strategy's performance relative to
the benchmark:

* :func:`reality_check` - White (2000). Tests the single null "no strategy beats the
  benchmark" with the (non-studentised) maximum of the performance differences.
* :func:`spa_test` - Hansen (2005). Studentises each difference and only recentres
  strategies that are not clearly worse than the benchmark, so adding poor strategies
  does not destroy power. Reports the lower, consistent and upper p-values.
* :func:`romano_wolf` - Romano and Wolf (2005, 2016). Step-down procedure that names
  *which* strategies beat the benchmark while controlling the family-wise error rate,
  with adjusted p-values for every strategy.

Performance is measured either by the mean return difference (``criterion="mean"``) or
by the Sharpe ratio difference (``criterion="sharpe"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special

from overfit._lrv import newey_west_lrv, stationary_bootstrap_lrv
from overfit._utils import FloatArray, SeedLike, as_matrix, as_vector
from overfit.bootstrap import BootstrapMethod, auto_block_size, bootstrap_counts

Criterion = Literal["mean", "sharpe"]


@dataclass(frozen=True)
class BootstrapPerformance:
    """Observed performance differences and their bootstrap replicates.

    ``theta[k]`` is strategy ``k``'s mean (or Sharpe ratio) minus the benchmark's,
    ``theta_star[b, k]`` its value in bootstrap sample ``b``, and ``omega[k]`` the
    long-run standard deviation of ``sqrt(n_obs) * theta[k]`` used to studentise.
    """

    labels: list[str]
    theta: FloatArray
    theta_star: FloatArray
    omega: FloatArray
    n_obs: int
    block_size: float
    method: str
    criterion: str

    @property
    def n_boot(self) -> int:
        return int(self.theta_star.shape[0])

    @property
    def tstats(self) -> FloatArray:
        return np.asarray(np.sqrt(self.n_obs) * self.theta / self.omega)


def _sharpe_from_moments(m1: FloatArray, m2: FloatArray) -> FloatArray:
    var = m2 - m1**2
    var = np.where(var > 1e-12 * m2, var, 0.0)  # constant series give rounding noise
    sd = np.sqrt(var)
    return np.asarray(np.divide(m1, sd, out=np.zeros_like(m1), where=sd > 0))


def bootstrap_performance(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    criterion: Criterion = "mean",
    n_boot: int = 1000,
    block_size: float | None = None,
    bootstrap: BootstrapMethod = "stationary",
    seed: SeedLike = None,
) -> BootstrapPerformance:
    """Bootstrap the performance of every strategy relative to the benchmark.

    All strategies are resampled with the same time indices, which preserves the
    cross-sectional dependence that a max-type test needs. ``block_size=None`` picks
    the median Politis-White length (see :func:`overfit.bootstrap.auto_block_size`).
    """
    x, labels = as_matrix(returns)
    n, k = x.shape
    if benchmark is None:
        bench = np.zeros(n)
    else:
        bench = as_vector(benchmark, name="benchmark")
        if bench.shape[0] != n:
            raise ValueError(f"benchmark has {bench.shape[0]} observations, returns have {n}")
    diff = x - bench[:, None]
    size = auto_block_size(diff) if block_size is None else float(block_size)
    counts = bootstrap_counts(n, block_size=size, n_boot=n_boot, method=bootstrap, seed=seed)

    if criterion == "mean":
        theta = diff.mean(axis=0)
        theta_star = counts @ diff / n
        if bootstrap == "stationary":
            omega2 = stationary_bootstrap_lrv(diff, size)
        else:
            omega2 = n * theta_star.var(axis=0)
    elif criterion == "sharpe":
        sr = _sharpe_from_moments(x.mean(axis=0), (x**2).mean(axis=0))
        sr_star = _sharpe_from_moments(counts @ x / n, counts @ (x**2) / n)
        if benchmark is None:
            sr_b, sr_b_star = np.zeros(1), np.zeros((n_boot, 1))
        else:
            sr_b = _sharpe_from_moments(np.array([bench.mean()]), np.array([np.mean(bench**2)]))
            sr_b_star = _sharpe_from_moments(counts @ bench / n, counts @ bench**2 / n)[:, None]
        theta = sr - sr_b
        theta_star = sr_star - sr_b_star
        omega2 = n * theta_star.var(axis=0)
    else:
        raise ValueError("criterion must be 'mean' or 'sharpe'")

    omega = np.sqrt(np.maximum(np.asarray(omega2, dtype=float).reshape(k), 0.0))
    omega[omega == 0] = np.inf  # degenerate series: studentised statistics become 0
    return BootstrapPerformance(
        labels=labels,
        theta=np.asarray(theta, dtype=float),
        theta_star=np.asarray(theta_star, dtype=float),
        omega=omega,
        n_obs=n,
        block_size=size,
        method=bootstrap,
        criterion=criterion,
    )


def _pvalue(null_draws: FloatArray, observed: float) -> float:
    return float((1 + np.count_nonzero(null_draws >= observed)) / (null_draws.shape[0] + 1))


# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealityCheckResult:
    statistic: float
    pvalue: float
    best: str
    best_performance: float
    n_strategies: int
    n_boot: int
    block_size: float
    criterion: str


def reality_check(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    criterion: Criterion = "mean",
    n_boot: int = 1000,
    block_size: float | None = None,
    bootstrap: BootstrapMethod = "stationary",
    seed: SeedLike = None,
) -> RealityCheckResult:
    """White's (2000) Reality Check for data snooping.

    Null hypothesis: no strategy outperforms the benchmark,
    ``max_k E[theta_k] <= 0``. The statistic is ``sqrt(n) * max_k theta_k`` and its
    null distribution is approximated by ``sqrt(n) * max_k (theta*_k - theta_k)``.
    ``benchmark=None`` compares against zero (cash).
    """
    bp = bootstrap_performance(
        returns,
        benchmark,
        criterion=criterion,
        n_boot=n_boot,
        block_size=block_size,
        bootstrap=bootstrap,
        seed=seed,
    )
    return reality_check_from(bp)


def reality_check_from(bp: BootstrapPerformance) -> RealityCheckResult:
    """Reality Check from an existing :class:`BootstrapPerformance`."""
    root_n = np.sqrt(bp.n_obs)
    best = int(np.argmax(bp.theta))
    statistic = float(root_n * bp.theta[best])
    null = root_n * (bp.theta_star - bp.theta).max(axis=1)
    return RealityCheckResult(
        statistic=statistic,
        pvalue=_pvalue(null, statistic),
        best=bp.labels[best],
        best_performance=float(bp.theta[best]),
        n_strategies=len(bp.labels),
        n_boot=bp.n_boot,
        block_size=bp.block_size,
        criterion=bp.criterion,
    )


# ---------------------------------------------------------------------------
# Hansen's test for Superior Predictive Ability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SPAResult:
    statistic: float
    pvalue: float
    pvalue_lower: float
    pvalue_upper: float
    best: str
    n_strategies: int
    n_boot: int
    block_size: float
    criterion: str
    studentized: bool


def spa_test(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    criterion: Criterion = "mean",
    n_boot: int = 1000,
    block_size: float | None = None,
    bootstrap: BootstrapMethod = "stationary",
    studentize: bool = True,
    seed: SeedLike = None,
) -> SPAResult:
    """Hansen's (2005) test for Superior Predictive Ability.

    ``pvalue`` is the consistent p-value: strategies whose performance is below
    ``-sqrt(omega^2 / n * 2 log log n)`` are treated as irrelevant for the null
    distribution and are not recentred. ``pvalue_lower`` never recentres strategies
    that underperform; ``pvalue_upper`` recentres every strategy, which with
    ``studentize=False`` reproduces White's Reality Check.
    """
    bp = bootstrap_performance(
        returns,
        benchmark,
        criterion=criterion,
        n_boot=n_boot,
        block_size=block_size,
        bootstrap=bootstrap,
        seed=seed,
    )
    return spa_test_from(bp, studentize=studentize)


def spa_test_from(bp: BootstrapPerformance, *, studentize: bool = True) -> SPAResult:
    """SPA test from an existing :class:`BootstrapPerformance`."""
    n = bp.n_obs
    root_n = np.sqrt(n)
    scale = bp.omega if studentize else np.ones_like(bp.omega)
    stats = root_n * bp.theta / scale
    statistic = float(max(0.0, stats.max()))
    finite_omega = np.where(np.isfinite(bp.omega), bp.omega, 0.0)
    threshold = np.sqrt(finite_omega**2 / n * 2.0 * np.log(np.log(n)))
    centres = {
        "lower": np.maximum(bp.theta, 0.0),
        "consistent": np.where(bp.theta >= -threshold, bp.theta, 0.0),
        "upper": bp.theta,
    }
    pvalues = {}
    for name, centre in centres.items():
        null = np.maximum(0.0, (root_n * (bp.theta_star - centre) / scale).max(axis=1))
        pvalues[name] = _pvalue(null, statistic)
    return SPAResult(
        statistic=statistic,
        pvalue=pvalues["consistent"],
        pvalue_lower=pvalues["lower"],
        pvalue_upper=pvalues["upper"],
        best=bp.labels[int(np.argmax(stats))],
        n_strategies=len(bp.labels),
        n_boot=bp.n_boot,
        block_size=bp.block_size,
        criterion=bp.criterion,
        studentized=studentize,
    )


# ---------------------------------------------------------------------------
# Romano-Wolf step-down
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepMResult:
    pvalues: pd.Series
    statistics: pd.Series
    alpha: float
    rejected: list[str] = field(default_factory=list)
    n_boot: int = 0
    block_size: float = float("nan")
    criterion: str = "mean"

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def romano_wolf(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    criterion: Criterion = "mean",
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_size: float | None = None,
    bootstrap: BootstrapMethod = "stationary",
    studentize: bool = True,
    seed: SeedLike = None,
) -> StepMResult:
    """Romano-Wolf step-down test with adjusted p-values.

    For each strategy ``k`` it tests ``H_k: theta_k <= 0`` against ``theta_k > 0``.
    The adjusted p-value of the strategy with the ``j``-th largest statistic is the
    bootstrap probability that the maximum (centred) statistic over the strategies
    ranked ``j`` and below exceeds it, made monotone in ``j`` (Romano and Wolf, 2016,
    Algorithm 4.1). Rejecting every ``k`` with ``pvalues[k] <= alpha`` controls the
    family-wise error rate at ``alpha`` asymptotically.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    bp = bootstrap_performance(
        returns,
        benchmark,
        criterion=criterion,
        n_boot=n_boot,
        block_size=block_size,
        bootstrap=bootstrap,
        seed=seed,
    )
    return romano_wolf_from(bp, alpha=alpha, studentize=studentize)


def romano_wolf_from(
    bp: BootstrapPerformance, *, alpha: float = 0.05, studentize: bool = True
) -> StepMResult:
    """Romano-Wolf step-down from an existing :class:`BootstrapPerformance`."""
    root_n = np.sqrt(bp.n_obs)
    scale = bp.omega if studentize else np.ones_like(bp.omega)
    stats = root_n * bp.theta / scale
    centred = root_n * (bp.theta_star - bp.theta) / scale
    order = np.argsort(-stats, kind="stable")
    tail_max = np.maximum.accumulate(centred[:, order][:, ::-1], axis=1)[:, ::-1]
    exceed = np.count_nonzero(tail_max >= stats[order], axis=0)
    p_sorted = np.maximum.accumulate((1.0 + exceed) / (bp.n_boot + 1.0))
    pvalues = np.empty_like(p_sorted)
    pvalues[order] = p_sorted
    rejected = [bp.labels[k] for k in order if pvalues[k] <= alpha]
    return StepMResult(
        pvalues=pd.Series(pvalues, index=bp.labels, name="pvalue"),
        statistics=pd.Series(stats, index=bp.labels, name="statistic"),
        alpha=alpha,
        rejected=rejected,
        n_boot=bp.n_boot,
        block_size=bp.block_size,
        criterion=bp.criterion,
    )


# ---------------------------------------------------------------------------
# Unadjusted per-strategy tests (the "naive" baseline)
# ---------------------------------------------------------------------------


def individual_pvalues(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    lags: int | None = None,
) -> pd.Series:
    """One-sided HAC t-test p-values of ``mean(strategy - benchmark) > 0`` per strategy.

    These ignore that many strategies were tried. Feed them to
    :func:`overfit.multitest.adjust_pvalues` for Bonferroni, Holm or FDR control.
    """
    x, labels = as_matrix(returns)
    n = x.shape[0]
    bench = np.zeros(n) if benchmark is None else as_vector(benchmark, name="benchmark")
    if bench.shape[0] != n:
        raise ValueError("benchmark and returns must have the same length")
    diff = x - bench[:, None]
    lrv = np.asarray(newey_west_lrv(diff, lags), dtype=float).reshape(-1)
    se = np.sqrt(np.maximum(lrv, 0.0) / n)
    t = np.divide(diff.mean(axis=0), se, out=np.zeros(diff.shape[1]), where=se > 0)
    return pd.Series(special.ndtr(-t), index=labels, name="pvalue")
