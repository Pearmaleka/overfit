"""Inference for the Sharpe ratio when many strategies have been tried.

Unless stated otherwise every Sharpe ratio here is *per period* (daily for daily
returns). Multiply by ``sqrt(periods_per_year)`` to annualise, or pass
``periods_per_year`` where a function accepts it.

References
----------
Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal.
Mertens (2002), "Comments on variance of the IID estimator in Lo (2002)".
Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk.
Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management.
Bailey, Borwein, Lopez de Prado and Zhu (2014), "Pseudo-Mathematics and Financial
Charlatanism", Notices of the American Mathematical Society.
Harvey and Liu (2015), "Backtesting", Journal of Portfolio Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import integrate, special

from overfit._lrv import newey_west_lrcov
from overfit._utils import FloatArray, as_matrix, as_vector

EULER_GAMMA = 0.5772156649015329

SEMethod = Literal["iid", "nonnormal", "hac"]


# ---------------------------------------------------------------------------
# Point estimates and descriptive statistics
# ---------------------------------------------------------------------------


@overload
def sharpe_ratio(
    returns: pd.DataFrame,
    *,
    risk_free: float | ArrayLike = ...,
    periods_per_year: float | None = ...,
    ddof: int = ...,
) -> pd.Series: ...


@overload
def sharpe_ratio(
    returns: ArrayLike | pd.Series,
    *,
    risk_free: float | ArrayLike = ...,
    periods_per_year: float | None = ...,
    ddof: int = ...,
) -> float | FloatArray: ...


def sharpe_ratio(
    returns: ArrayLike | pd.Series | pd.DataFrame,
    *,
    risk_free: float | ArrayLike = 0.0,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> float | FloatArray | pd.Series:
    """Sample Sharpe ratio of each series.

    Parameters
    ----------
    returns
        Simple returns, 1-D (one strategy) or 2-D with one column per strategy.
    risk_free
        Per-period risk-free rate, a scalar or a series aligned with ``returns``.
    periods_per_year
        If given, the result is annualised by ``sqrt(periods_per_year)``.
    ddof
        Degrees of freedom for the standard deviation.

    Returns
    -------
    float for 1-D input, a ``pandas.Series`` for a DataFrame, otherwise an array.
    Series with zero variance get a Sharpe ratio of 0.
    """
    arr, _ = as_matrix(returns)
    rf = np.asarray(risk_free, dtype=float)
    excess = arr - (rf.reshape(-1, 1) if rf.ndim == 1 else rf)
    mu = excess.mean(axis=0)
    sd = excess.std(axis=0, ddof=ddof)
    sr = np.divide(mu, sd, out=np.zeros_like(mu), where=np.ptp(excess, axis=0) > 0)
    if periods_per_year is not None:
        sr = sr * np.sqrt(periods_per_year)
    if isinstance(returns, pd.DataFrame):
        return pd.Series(sr, index=returns.columns, name="sharpe")
    if np.ndim(returns) == 1 or isinstance(returns, pd.Series):
        return float(sr[0])
    return sr


def annualize(sr: float, periods_per_year: float) -> float:
    """Convert a per-period Sharpe ratio to an annual one."""
    return float(sr * np.sqrt(periods_per_year))


def _moments(x: FloatArray) -> tuple[float, float, float, float]:
    """Mean, standard deviation (ddof=1), skewness and Pearson kurtosis (normal = 3)."""
    mu = float(x.mean())
    centred = x - mu
    m2 = float(np.mean(centred**2))
    if m2 <= 0 or np.ptp(x) == 0:
        return mu, 0.0, 0.0, 3.0
    skew = float(np.mean(centred**3) / m2**1.5)
    kurt = float(np.mean(centred**4) / m2**2)
    return mu, float(x.std(ddof=1)), skew, kurt


def sharpe_ratio_se(
    returns: ArrayLike | pd.Series,
    *,
    method: SEMethod = "nonnormal",
    lags: int | None = None,
    risk_free: float = 0.0,
) -> float:
    """Standard error of the per-period Sharpe ratio.

    Methods
    -------
    ``"iid"``
        Normal iid returns: ``sqrt((1 + SR^2 / 2) / (n - 1))`` (Lo, 2002).
    ``"nonnormal"``
        iid returns with skewness ``g3`` and kurtosis ``g4``:
        ``sqrt((1 - g3 SR + (g4 - 1) / 4 SR^2) / (n - 1))`` (Mertens, 2002).
    ``"hac"``
        Serially dependent returns: delta method on ``(E[r], E[r^2])`` with a
        Newey-West long-run covariance. ``lags`` defaults to ``floor(4 (n/100)^(2/9))``.
    """
    x = as_vector(returns) - risk_free
    n = x.shape[0]
    mu, sd, skew, kurt = _moments(x)
    if sd == 0:
        return float("nan")
    sr = mu / sd
    if method == "iid":
        var = (1.0 + 0.5 * sr**2) / (n - 1)
    elif method == "nonnormal":
        var = (1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2) / (n - 1)
    elif method == "hac":
        m2 = float(np.mean(x**2))
        sigma = np.sqrt(m2 - mu**2)
        grad = np.array([m2 / sigma**3, -mu / (2.0 * sigma**3)])
        omega = newey_west_lrcov(np.column_stack([x, x**2]), lags)
        var = float(grad @ omega @ grad) / n
    else:
        raise ValueError(f"unknown method {method!r}; use 'iid', 'nonnormal' or 'hac'")
    return float(np.sqrt(max(var, 0.0)))


@dataclass(frozen=True)
class SharpeStats:
    """Summary of one return series. Sharpe ratios are per period unless suffixed ``_annual``."""

    sr: float
    n_obs: int
    mean: float
    std: float
    skew: float
    kurtosis: float
    se_iid: float
    se_nonnormal: float
    se_hac: float
    psr: float
    periods_per_year: float | None

    @property
    def sr_annual(self) -> float | None:
        return None if self.periods_per_year is None else annualize(self.sr, self.periods_per_year)

    @property
    def tstat_hac(self) -> float:
        """HAC t-statistic for ``SR > 0``."""
        return self.sr / self.se_hac if self.se_hac > 0 else float("nan")


def describe(
    returns: ArrayLike | pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: float | None = None,
) -> SharpeStats:
    """Sharpe ratio, higher moments, three standard errors and the PSR against zero."""
    x = as_vector(returns) - risk_free
    mu, sd, skew, kurt = _moments(x)
    sr = mu / sd if sd > 0 else 0.0
    n = x.shape[0]
    psr = probabilistic_sharpe_ratio(sr, n, skew=skew, kurtosis=kurt) if sd > 0 else float("nan")
    return SharpeStats(
        sr=sr,
        n_obs=n,
        mean=mu,
        std=sd,
        skew=skew,
        kurtosis=kurt,
        se_iid=sharpe_ratio_se(x, method="iid"),
        se_nonnormal=sharpe_ratio_se(x, method="nonnormal"),
        se_hac=sharpe_ratio_se(x, method="hac"),
        psr=psr,
        periods_per_year=periods_per_year,
    )


# ---------------------------------------------------------------------------
# Probabilistic Sharpe ratio and minimum track record length
# ---------------------------------------------------------------------------


def _sr_variance_factor(sr: float, skew: float, kurtosis: float) -> float:
    factor = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr**2
    if factor <= 0:
        raise ValueError(
            "skewness and kurtosis imply a non-positive variance for the Sharpe ratio; "
            "check that the Sharpe ratio is per period, not annualised"
        )
    return factor


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``sr_benchmark``.

    ``PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))``
    (Bailey and Lopez de Prado, 2012). All Sharpe ratios are per period and
    ``kurtosis`` is Pearson kurtosis (3 for a normal distribution).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    factor = _sr_variance_factor(sr, skew, kurtosis)
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(factor)
    return float(special.ndtr(z))


def min_track_record_length(
    sr: float,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Number of observations needed for ``PSR(sr_benchmark) >= 1 - alpha``.

    Returns ``inf`` when ``sr <= sr_benchmark``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sr <= sr_benchmark:
        return float("inf")
    factor = _sr_variance_factor(sr, skew, kurtosis)
    z = special.ndtri(1.0 - alpha)
    return float(1.0 + factor * (z / (sr - sr_benchmark)) ** 2)


# ---------------------------------------------------------------------------
# Selection bias: expected maximum Sharpe ratio and the deflated Sharpe ratio
# ---------------------------------------------------------------------------


def _expected_max_std_normal(n: float) -> float:
    """``E[max(Z_1..Z_n)]`` for iid standard normals, by numerical integration.

    Uses ``E[M] = int_0^inf (1 - Phi(x)^n) dx - int_-inf^0 Phi(x)^n dx``, which also
    defines the expectation for non-integer ``n``.
    """

    def upper(x: float) -> float:
        return float(-np.expm1(n * special.log_ndtr(x)))

    def lower(x: float) -> float:
        return float(np.exp(n * special.log_ndtr(x)))

    mode = float(special.ndtri(1.0 - 1.0 / max(n, 2.0)))
    pos, _ = integrate.quad(upper, 0.0, 40.0, points=[max(mode, 0.5)], limit=200)
    neg, _ = integrate.quad(lower, -40.0, 0.0, limit=200)
    return float(pos - neg)


def expected_max_sharpe(
    n_trials: float,
    sr_variance: float = 1.0,
    *,
    sr_mean: float = 0.0,
    method: Literal["approx", "exact"] = "approx",
) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratio estimates when none has skill.

    The estimates are modelled as iid ``N(sr_mean, sr_variance)``. ``method="approx"``
    is the extreme-value approximation of Bailey and Lopez de Prado (2014)::

        sqrt(V) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))

    and ``method="exact"`` integrates the distribution of the maximum numerically.
    ``n_trials`` may be fractional (an effective number of trials).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    if n_trials == 1:
        return float(sr_mean)
    if method == "approx":
        z = (1.0 - EULER_GAMMA) * special.ndtri(1.0 - 1.0 / n_trials) + EULER_GAMMA * special.ndtri(
            1.0 - 1.0 / (n_trials * np.e)
        )
    elif method == "exact":
        z = _expected_max_std_normal(float(n_trials))
    else:
        raise ValueError("method must be 'approx' or 'exact'")
    return float(sr_mean + np.sqrt(sr_variance) * z)


def min_backtest_length(
    n_trials: float,
    target_sharpe: float,
    *,
    method: Literal["approx", "exact"] = "approx",
) -> float:
    """Years of data needed before ``n_trials`` strategies without skill stop producing,
    on average, a best annual Sharpe ratio of ``target_sharpe``.

    With ``y`` years of data, the annualised Sharpe ratio estimate of a strategy with
    no skill has a standard deviation of about ``1 / sqrt(y)``, so the best of ``N``
    such strategies is expected to reach ``E[max_N] / sqrt(y)``, where ``E[max_N]`` is
    the expected maximum of ``N`` standard normals (see :func:`expected_max_sharpe`).
    Bailey, Borwein, Lopez de Prado and Zhu (2014) solve for ``y``::

        MinBTL = (E[max_N] / SR_target)^2,   which is below 2 ln(N) / SR_target^2

    A backtest shorter than ``MinBTL`` years cannot tell a Sharpe ratio of
    ``target_sharpe`` from the result of trying ``N`` configurations. For example,
    45 trials need about 5 years of data for a target of 1.

    Parameters
    ----------
    n_trials
        Number of (effective, independent) configurations tried.
    target_sharpe
        Annualised Sharpe ratio of the best configuration.
    method
        ``"approx"`` for the closed form above, ``"exact"`` to integrate ``E[max_N]``.

    Returns
    -------
    float
        Years; 0 for a single trial and ``inf`` if ``target_sharpe <= 0``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if target_sharpe <= 0:
        return float("inf")
    z = expected_max_sharpe(n_trials, 1.0, method=method)
    return float((z / target_sharpe) ** 2)


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    sr_variance: float,
    n_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    method: Literal["approx", "exact"] = "approx",
) -> float:
    """Deflated Sharpe ratio: the PSR measured against the expected maximum Sharpe ratio.

    ``sr_variance`` is the variance of the per-period Sharpe ratios across all
    trials and ``n_trials`` the (effective) number of independent trials.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance, method=method)
    return probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurtosis, sr_benchmark=sr0)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of :func:`deflated_sharpe_from_trials`. Sharpe ratios are per period."""

    selected: str
    sr: float
    sr0: float
    dsr: float
    psr: float
    n_trials: float
    sr_variance: float
    n_obs: int
    skew: float
    kurtosis: float
    periods_per_year: float | None

    @property
    def sr_annual(self) -> float | None:
        return None if self.periods_per_year is None else annualize(self.sr, self.periods_per_year)

    @property
    def sr0_annual(self) -> float | None:
        return None if self.periods_per_year is None else annualize(self.sr0, self.periods_per_year)


def deflated_sharpe_from_trials(
    trial_returns: ArrayLike | pd.DataFrame,
    *,
    n_trials: float | None = None,
    selected: int | str | None = None,
    periods_per_year: float | None = None,
    method: Literal["approx", "exact"] = "approx",
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio of the best (or a chosen) strategy among all trials.

    Parameters
    ----------
    trial_returns
        Returns of every configuration that was tried, one column per trial.
    n_trials
        Effective number of independent trials. Defaults to the number of columns;
        see :func:`overfit.trials.effective_number_of_trials` for correlated trials.
    selected
        Column (position or label) of the strategy to evaluate. Defaults to the
        column with the highest Sharpe ratio.
    """
    arr, labels = as_matrix(trial_returns, name="trial_returns")
    if arr.shape[1] < 2:
        raise ValueError("need returns for at least two trials to estimate the Sharpe variance")
    srs = np.asarray(sharpe_ratio(arr))
    if selected is None:
        k = int(np.argmax(srs))
    elif isinstance(selected, str):
        k = labels.index(selected)
    else:
        k = int(selected)
    x = arr[:, k]
    _, _, skew, kurt = _moments(x)
    n_eff = float(arr.shape[1] if n_trials is None else n_trials)
    sr_var = float(np.var(srs, ddof=1))
    sr0 = expected_max_sharpe(n_eff, sr_var, method=method)
    sr = float(srs[k])
    n_obs = arr.shape[0]
    return DeflatedSharpeResult(
        selected=labels[k],
        sr=sr,
        sr0=sr0,
        dsr=probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurt, sr_benchmark=sr0),
        psr=probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurt),
        n_trials=n_eff,
        sr_variance=sr_var,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
        periods_per_year=periods_per_year,
    )


# ---------------------------------------------------------------------------
# Multiple-testing haircut (Harvey and Liu, 2015)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HaircutResult:
    sr: float
    sr_haircut: float
    haircut: float
    tstat: float
    pvalue: float
    pvalue_adjusted: float
    n_trials: float
    method: str


def haircut_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: float,
    *,
    method: Literal["bonferroni", "sidak"] = "bonferroni",
) -> HaircutResult:
    """Shrink a per-period Sharpe ratio for the number of strategies tried.

    The Sharpe ratio is turned into a t-statistic ``SR * sqrt(n)``, its one-sided
    p-value is adjusted for ``n_trials`` tests, and the adjusted p-value is mapped
    back to a Sharpe ratio. ``haircut = 1 - SR_adjusted / SR``. Step-wise
    adjustments (Holm, BHY) need the p-values of every trial; use
    :func:`overfit.multitest.adjust_pvalues` for those.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    t = sr * np.sqrt(n_obs)
    p = float(special.ndtr(-t))
    if method == "bonferroni":
        p_adj = min(1.0, n_trials * p)
    elif method == "sidak":
        p_adj = float(-np.expm1(n_trials * np.log1p(-p))) if p < 1 else 1.0
    else:
        raise ValueError("method must be 'bonferroni' or 'sidak'")
    t_adj = float(-special.ndtri(p_adj)) if 0 < p_adj < 1 else (-np.inf if p_adj >= 1 else np.inf)
    sr_adj = max(t_adj, 0.0) / np.sqrt(n_obs)
    haircut = 1.0 - sr_adj / sr if sr > 0 else float("nan")
    return HaircutResult(
        sr=float(sr),
        sr_haircut=float(sr_adj),
        haircut=float(haircut),
        tstat=float(t),
        pvalue=p,
        pvalue_adjusted=float(p_adj),
        n_trials=float(n_trials),
        method=method,
    )
