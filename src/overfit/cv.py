"""Cross-validation for labels that span several periods.

When a label is computed over an interval ``[t0, t1]`` (for example the sign of the next
20-day return), neighbouring samples share most of their label. Ordinary k-fold
cross-validation then trains on samples whose labels overlap the test fold, and a
flexible model scores well by remembering its neighbours rather than by predicting.

* :class:`PurgedKFold` removes ("purges") every training sample whose label interval
  overlaps the test fold and drops an "embargo" of samples right after it.
* :class:`CombinatorialPurgedKFold` (CPCV) tests on every combination of ``k`` of ``N``
  contiguous groups, which yields ``C(N-1, k-1)`` complete out-of-sample paths.

Both follow the scikit-learn splitter protocol (``split`` / ``get_n_splits``) and can be
passed as ``cv=`` to ``cross_val_score``, ``cross_val_predict`` or ``GridSearchCV``.

Reference: Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 and 12.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.intp]


def label_times(index: int | pd.Index | Sequence[Any], horizon: int) -> pd.Series:
    """End times of forward-looking labels over ``horizon`` steps.

    ``index`` is either the number of samples or the time index of the samples; the
    label of sample ``i`` uses data up to sample ``min(i + horizon, n - 1)``.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    idx = pd.RangeIndex(index) if isinstance(index, int) else pd.Index(index)
    n = len(idx)
    end = np.minimum(np.arange(n) + horizon, n - 1)
    return pd.Series(idx.to_numpy()[end], index=idx, name="t1")


def _as_times(t1: pd.Series | ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(t1, pd.Series):
        start = t1.index.to_numpy()
        end = t1.to_numpy()
    else:
        end = np.asarray(t1)
        if end.ndim != 1:
            raise ValueError("t1 must be one-dimensional")
        start = np.arange(end.shape[0])
    if start.shape[0] < 2:
        raise ValueError("need at least two samples")
    if np.any(start[1:] < start[:-1]):
        raise ValueError("samples must be sorted by start time")
    if np.any(end < start):
        raise ValueError("every label must end at or after it starts (t1 >= t0)")
    return start, end


def _train_mask(
    start: np.ndarray, end: np.ndarray, test_blocks: list[tuple[int, int]], embargo: int
) -> NDArray[np.bool_]:
    """Training samples that neither overlap a test block nor fall in its embargo."""
    mask = np.ones(start.shape[0], dtype=bool)
    for first, last in test_blocks:
        block_start = start[first]
        block_end = end[first : last + 1].max()
        mask &= ~((start <= block_end) & (end >= block_start))
        after = int(np.searchsorted(start, block_end, side="right"))
        mask[after : after + embargo] = False
    return mask


class PurgedKFold:
    """K-fold splitter with purging and embargo for overlapping labels.

    Parameters
    ----------
    n_splits
        Number of contiguous test folds.
    t1
        Label end times. A ``pandas.Series`` whose index holds the label start times
        (sorted) and whose values hold the end times, or an integer array of end
        positions (start positions are then ``0..n-1``). See :func:`label_times`.
    embargo
        Fraction of all samples to drop from training right after each test fold.
    """

    def __init__(self, n_splits: int = 5, *, t1: pd.Series | ArrayLike, embargo: float = 0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0.0 <= embargo < 1.0:
            raise ValueError("embargo must be in [0, 1)")
        self.n_splits = int(n_splits)
        self.embargo = float(embargo)
        self._start, self._end = _as_times(t1)
        if self.n_splits > self._start.shape[0]:
            raise ValueError("more folds than samples")

    @property
    def n_samples(self) -> int:
        return int(self._start.shape[0])

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        n = self.n_samples
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows but t1 describes {n} samples")
        idx = np.arange(n)
        embargo = int(n * self.embargo)
        for test in np.array_split(idx, self.n_splits):
            mask = _train_mask(self._start, self._end, [(int(test[0]), int(test[-1]))], embargo)
            yield idx[mask], test


class CombinatorialPurgedKFold:
    """Combinatorial purged cross-validation (CPCV).

    The samples are cut into ``n_groups`` contiguous groups; each split tests on one
    combination of ``n_test_groups`` groups and trains on the purged, embargoed rest.
    There are ``C(n_groups, n_test_groups)`` splits and every group is tested
    ``n_paths = C(n_groups - 1, n_test_groups - 1)`` times, so the out-of-sample
    predictions can be stitched into ``n_paths`` complete backtest paths
    (see :meth:`paths`).
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        t1: pd.Series | ArrayLike,
        embargo: float = 0.0,
    ):
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must be between 1 and n_groups - 1")
        if not 0.0 <= embargo < 1.0:
            raise ValueError("embargo must be in [0, 1)")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo = float(embargo)
        self._start, self._end = _as_times(t1)
        if self.n_groups > self._start.shape[0]:
            raise ValueError("more groups than samples")

    @property
    def n_samples(self) -> int:
        return int(self._start.shape[0])

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def groups(self) -> list[IntArray]:
        """Sample positions of each contiguous group."""
        return list(np.array_split(np.arange(self.n_samples), self.n_groups))

    def test_groups(self) -> list[tuple[int, ...]]:
        """The test groups of each split, in split order."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[IntArray, IntArray]]:
        n = self.n_samples
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows but t1 describes {n} samples")
        idx = np.arange(n)
        embargo = int(n * self.embargo)
        parts = self.groups()
        for combo in self.test_groups():
            test = np.concatenate([parts[g] for g in combo])
            blocks = [(int(parts[g][0]), int(parts[g][-1])) for g in combo]
            mask = _train_mask(self._start, self._end, blocks, embargo)
            yield idx[mask], test

    def paths(self) -> NDArray[np.intp]:
        """Split index used for each (path, group): array of shape ``(n_paths, n_groups)``.

        Path ``p`` takes group ``g`` from the ``p``-th split (in split order) that tests
        on ``g``. Stitching the out-of-sample predictions this way gives ``n_paths``
        backtests that each cover the whole sample exactly once.
        """
        out = np.full((self.n_paths, self.n_groups), -1, dtype=np.intp)
        seen = np.zeros(self.n_groups, dtype=np.intp)
        for split_id, combo in enumerate(self.test_groups()):
            for g in combo:
                out[seen[g], g] = split_id
                seen[g] += 1
        return out

    def backtest_paths(self, predictions: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Stitch out-of-sample predictions into complete backtest paths.

        Parameters
        ----------
        predictions
            One array per split, in the order of :meth:`split`. ``predictions[s]``
            holds the out-of-sample values (predictions, positions or strategy
            returns) for the test samples of split ``s``, in the order :meth:`split`
            yields them.

        Returns
        -------
        Array of shape ``(n_paths, n_samples)``. Row ``p`` covers every sample exactly
        once: group ``g`` comes from split ``paths()[p, g]``. Evaluating a strategy on
        each row gives ``n_paths`` out-of-sample backtests instead of one.
        """
        n_splits = self.get_n_splits()
        if len(predictions) != n_splits:
            raise ValueError(f"expected {n_splits} prediction arrays, got {len(predictions)}")
        parts = self.groups()
        sizes = [len(part) for part in parts]
        by_split: list[dict[int, NDArray[np.float64]]] = []
        for split_id, (combo, values) in enumerate(
            zip(self.test_groups(), predictions, strict=True)
        ):
            arr = np.asarray(values, dtype=float)
            expected = sum(sizes[g] for g in combo)
            if arr.ndim != 1 or arr.shape[0] != expected:
                raise ValueError(
                    f"predictions[{split_id}] must be one-dimensional with {expected} values "
                    f"(the test samples of split {split_id}), got shape {arr.shape}"
                )
            bounds = np.cumsum([0] + [sizes[g] for g in combo])
            by_split.append({g: arr[bounds[i] : bounds[i + 1]] for i, g in enumerate(combo)})
        assignment = self.paths()
        out = np.empty((self.n_paths, self.n_samples))
        for p in range(self.n_paths):
            for g, part in enumerate(parts):
                out[p, part] = by_split[assignment[p, g]][g]
        return out
