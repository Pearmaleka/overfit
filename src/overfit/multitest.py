"""Classical p-value adjustments for multiple testing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from overfit._utils import FloatArray

Method = Literal["bonferroni", "sidak", "holm", "hochberg", "bh", "by"]
METHODS: tuple[str, ...] = ("bonferroni", "sidak", "holm", "hochberg", "bh", "by")


def adjust_pvalues(pvalues: ArrayLike, method: Method = "holm") -> FloatArray:
    """Adjust p-values for multiple comparisons.

    ``bonferroni``, ``sidak``, ``holm`` and ``hochberg`` control the family-wise
    error rate; ``bh`` (Benjamini-Hochberg) and ``by`` (Benjamini-Yekutieli, valid
    under arbitrary dependence) control the false discovery rate. Adjusted p-values
    are returned in the original order and capped at 1.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if p.size == 0:
        return p.copy()
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    if method == "bonferroni":
        return np.minimum(1.0, m * p)
    if method == "sidak":
        with np.errstate(divide="ignore"):
            return np.asarray(-np.expm1(m * np.log1p(-p)))
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adj = np.maximum.accumulate((m - ranks + 1) * ps)
    elif method == "hochberg":
        adj = np.minimum.accumulate(((m - ranks + 1) * ps)[::-1])[::-1]
    elif method in ("bh", "by"):
        scale = 1.0 if method == "bh" else float(np.sum(1.0 / ranks))
        adj = np.minimum.accumulate((scale * m / ranks * ps)[::-1])[::-1]
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    out = np.empty(m)
    out[order] = np.minimum(1.0, adj)
    return out
