"""How many independent trials does a set of correlated backtests amount to?

Correlated strategy variants (a 50-day and a 51-day moving average) are not independent
trials. The deflated Sharpe ratio and the multiple-testing haircut need an *effective*
number of trials; the eigenvalue-based estimators below come from the genetics
literature on correlated tests:

* Nyholt (2004), Am. J. Hum. Genet. 74: ``1 + (M - 1) (1 - Var(lambda) / M)``
* Li and Ji (2005), Heredity 95: ``sum_i I(lambda_i >= 1) + (lambda_i - floor(lambda_i))``
* Galwey (2009), Genet. Epidemiol. 33: ``(sum_i sqrt(lambda_i))^2 / sum_i lambda_i``

where ``lambda_i`` are the eigenvalues of the correlation matrix of the strategies' returns.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from overfit._utils import as_matrix

EffectiveMethod = Literal["li_ji", "nyholt", "galwey"]


def correlation_matrix(returns: ArrayLike | pd.DataFrame) -> np.ndarray:
    """Pearson correlation of the columns; constant columns are treated as uncorrelated."""
    x, _ = as_matrix(returns)
    sd = x.std(axis=0)
    z = np.zeros_like(x)
    ok = sd > 0
    z[:, ok] = (x[:, ok] - x[:, ok].mean(axis=0)) / sd[ok]
    corr = z.T @ z / x.shape[0]
    np.fill_diagonal(corr, 1.0)
    return corr


def effective_number_of_trials(
    returns: ArrayLike | pd.DataFrame | None = None,
    *,
    corr: ArrayLike | None = None,
    method: EffectiveMethod = "li_ji",
) -> float:
    """Effective number of independent trials implied by the correlation of their returns.

    Pass either the trials' ``returns`` (one column per trial) or their correlation
    matrix ``corr``. The result lies between 1 (perfectly correlated) and the number of
    trials (uncorrelated).
    """
    if (returns is None) == (corr is None):
        raise ValueError("pass exactly one of returns or corr")
    c = correlation_matrix(returns) if corr is None else np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix")
    if not np.allclose(c, c.T, atol=1e-8):
        raise ValueError("corr must be symmetric")
    m = c.shape[0]
    if m == 1:
        return 1.0
    # Round so that eigenvalues such as 3.9999999999999996 are not floored to 3.
    eig = np.round(np.clip(np.linalg.eigvalsh(c), 0.0, None), 10)
    if method == "nyholt":
        value = 1.0 + (m - 1.0) * (1.0 - np.var(eig, ddof=1) / m)
    elif method == "li_ji":
        value = float(np.sum((eig >= 1.0) + (eig - np.floor(eig))))
    elif method == "galwey":
        value = float(np.sum(np.sqrt(eig)) ** 2 / np.sum(eig))
    else:
        raise ValueError("method must be 'li_ji', 'nyholt' or 'galwey'")
    return float(np.clip(value, 1.0, m))
