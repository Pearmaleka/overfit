from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from overfit.cv import CombinatorialPurgedKFold, PurgedKFold, label_times


def _overlaps(start, end, train, test):
    """True if any training label interval intersects the span of the test labels."""
    lo, hi = start[test].min(), end[test].max()
    return bool(np.any((start[train] <= hi) & (end[train] >= lo)))


def test_label_times_integer_and_datetime():
    t1 = label_times(6, 2)
    np.testing.assert_array_equal(t1.to_numpy(), [2, 3, 4, 5, 5, 5])
    idx = pd.bdate_range("2024-01-01", periods=5)
    t1d = label_times(idx, 1)
    assert t1d.iloc[0] == idx[1] and t1d.iloc[-1] == idx[-1]


def test_purged_kfold_partitions_and_purges():
    n, horizon = 200, 7
    t1 = label_times(n, horizon)
    start, end = t1.index.to_numpy(), t1.to_numpy()
    cv = PurgedKFold(5, t1=t1, embargo=0.03)
    tests = []
    for train, test in cv.split(np.zeros((n, 2))):
        tests.append(test)
        assert not _overlaps(start, end, train, test)
        assert np.intersect1d(train, test).size == 0
        after = test[-1] + horizon + 1
        if after < n:
            # the embargo removes int(0.03 * 200) = 6 samples after the purged zone
            assert not np.isin(np.arange(after, after + 6), train).any()
            assert after + 6 in train
    np.testing.assert_array_equal(np.concatenate(tests), np.arange(n))


def test_without_overlap_it_is_plain_kfold():
    sk = pytest.importorskip("sklearn.model_selection")
    n = 50
    ours = list(PurgedKFold(5, t1=np.arange(n)).split())
    theirs = list(sk.KFold(5).split(np.zeros(n)))
    for (a, b), (c, d) in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_works_with_scikit_learn():
    sk = pytest.importorskip("sklearn.model_selection")
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 3))
    y = (x[:, 0] + 0.1 * rng.standard_normal(120) > 0).astype(int)
    cv = PurgedKFold(4, t1=label_times(120, 3), embargo=0.01)
    scores = sk.cross_val_score(lm.LogisticRegression(), x, y, cv=cv)
    assert scores.shape == (4,) and scores.mean() > 0.8


def test_datetime_labels():
    idx = pd.bdate_range("2023-01-02", periods=60)
    t1 = label_times(idx, 5)
    for train, test in PurgedKFold(3, t1=t1).split():
        assert not _overlaps(idx.to_numpy(), t1.to_numpy(), train, test)


def test_combinatorial_splits_and_paths():
    n, groups, k = 240, 6, 2
    t1 = label_times(n, 4)
    start, end = t1.index.to_numpy(), t1.to_numpy()
    cv = CombinatorialPurgedKFold(groups, k, t1=t1, embargo=0.01)
    splits = list(cv.split())
    assert len(splits) == cv.get_n_splits() == 15
    assert cv.n_paths == 5
    parts = cv.groups()
    for (train, test), combo in zip(splits, cv.test_groups(), strict=True):
        assert np.intersect1d(train, test).size == 0
        for g in combo:
            assert not _overlaps(start, end, train, parts[g])
    paths = cv.paths()
    assert paths.shape == (5, 6)
    for g in range(groups):
        assert len(set(paths[:, g])) == cv.n_paths  # each split used once per group
        for split_id in paths[:, g]:
            assert g in cv.test_groups()[split_id]


def test_validation():
    with pytest.raises(ValueError):
        PurgedKFold(1, t1=np.arange(10))
    with pytest.raises(ValueError):
        PurgedKFold(3, t1=np.arange(10), embargo=1.5)
    with pytest.raises(ValueError):
        PurgedKFold(3, t1=np.arange(10) - 1)  # ends before it starts
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 4, t1=np.arange(10))
    with pytest.raises(ValueError):
        list(PurgedKFold(3, t1=np.arange(10)).split(np.zeros(9)))


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(20, 150),
    folds=st.integers(2, 6),
    horizons=st.lists(st.integers(0, 15), min_size=1, max_size=150),
    embargo=st.floats(0.0, 0.1),
)
def test_no_training_label_ever_overlaps_its_test_fold(n, folds, horizons, embargo):
    h = np.resize(np.asarray(horizons), n)
    end = np.minimum(np.arange(n) + h, n - 1)
    start = np.arange(n)
    for train, test in PurgedKFold(folds, t1=end, embargo=embargo).split():
        assert not _overlaps(start, end, train, test)


def test_backtest_paths_cover_every_sample_once():
    n = 241  # groups of unequal size
    cv = CombinatorialPurgedKFold(6, 2, t1=label_times(n, 4))
    preds = [test.astype(float) for _, test in cv.split()]
    paths = cv.backtest_paths(preds)
    assert paths.shape == (cv.n_paths, n)
    for row in paths:
        np.testing.assert_array_equal(row, np.arange(n))


def test_backtest_paths_take_each_group_from_the_assigned_split():
    cv = CombinatorialPurgedKFold(5, 2, t1=label_times(180, 2))
    # tag every prediction with the split that produced it
    preds = [np.full(len(test), float(s)) for s, (_, test) in enumerate(cv.split())]
    paths = cv.backtest_paths(preds)
    assignment = cv.paths()
    for p in range(cv.n_paths):
        for g, part in enumerate(cv.groups()):
            assert np.all(paths[p, part] == assignment[p, g])
    assert len({tuple(row) for row in paths}) == cv.n_paths


def test_backtest_paths_with_scikit_learn():
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3))
    y = x @ np.array([0.5, -0.2, 0.1]) + 0.1 * rng.standard_normal(300)
    cv = CombinatorialPurgedKFold(6, 2, t1=label_times(300, 5), embargo=0.01)
    preds = [lm.LinearRegression().fit(x[tr], y[tr]).predict(x[te]) for tr, te in cv.split()]
    paths = cv.backtest_paths(preds)
    assert paths.shape == (5, 300)
    for row in paths:
        assert np.corrcoef(row, y)[0, 1] > 0.9


def test_backtest_paths_validation():
    cv = CombinatorialPurgedKFold(4, 2, t1=np.arange(40))
    preds = [test.astype(float) for _, test in cv.split()]
    with pytest.raises(ValueError):
        cv.backtest_paths(preds[:-1])
    preds[2] = preds[2][:-1]
    with pytest.raises(ValueError):
        cv.backtest_paths(preds)
