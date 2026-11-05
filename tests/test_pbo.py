from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.stats import rankdata

from overfit.pbo import probability_of_backtest_overfitting as pbo


def _naive_pbo(x: np.ndarray, s: int) -> np.ndarray:
    """Literal transcription of the CSCV algorithm, used as a reference."""
    blocks = np.array_split(np.arange(x.shape[0]), s)
    logits = []
    for combo in combinations(range(s), s // 2):
        train = np.concatenate([blocks[i] for i in combo])
        test = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        sr_train = x[train].mean(0) / x[train].std(0, ddof=1)
        sr_test = x[test].mean(0) / x[test].std(0, ddof=1)
        best = int(np.argmax(sr_train))
        w = rankdata(sr_test)[best] / (x.shape[1] + 1)
        logits.append(np.log(w / (1 - w)))
    return np.array(logits)


def test_matches_a_literal_implementation(rng):
    x = rng.normal(0.0, 0.01, size=(203, 7))
    res = pbo(x, n_splits=8)
    expected = _naive_pbo(x, 8)
    np.testing.assert_allclose(res.logits, expected, atol=1e-10)
    assert res.pbo == pytest.approx(np.mean(expected <= 0))
    assert res.n_combinations == 70


def test_callable_metric_matches_builtin(rng):
    x = rng.normal(0.0002, 0.01, size=(160, 9))
    fast = pbo(x, n_splits=6, metric="mean")
    slow = pbo(x, n_splits=6, metric=lambda a: a.mean(axis=0))
    np.testing.assert_allclose(fast.logits, slow.logits)
    np.testing.assert_array_equal(fast.selected, slow.selected)


def test_noise_gives_pbo_near_one_half():
    values = []
    for seed in range(12):
        x = np.random.default_rng(seed).normal(0.0, 0.01, size=(1000, 40))
        values.append(pbo(x, n_splits=10).pbo)
    assert np.mean(values) == pytest.approx(0.5, abs=0.12)


def test_genuine_edge_gives_low_pbo(rng):
    x = rng.normal(0.0, 0.01, size=(1500, 30))
    x[:, 4] += 0.003
    res = pbo(x, n_splits=10)
    assert res.pbo < 0.05
    assert res.selection_frequency.index[0] == "4"
    assert res.prob_oos_loss < 0.05


def test_degradation_and_subsampling(rng):
    x = rng.normal(0.0, 0.01, size=(800, 50))
    res = pbo(x, n_splits=12, max_combinations=200, seed=0)
    assert res.n_combinations == 200
    slope, _ = res.degradation
    assert slope < 0.5
    assert np.all((res.oos_rank > 0) & (res.oos_rank < 1))


def test_constant_strategy_is_handled(rng):
    x = rng.normal(0.0, 0.01, size=(300, 5))
    x[:, 0] = 0.0
    res = pbo(x, n_splits=6)
    assert np.all(np.isfinite(res.logits))


def test_input_validation(rng):
    x = rng.normal(size=(100, 4))
    with pytest.raises(ValueError):
        pbo(x, n_splits=5)
    with pytest.raises(ValueError):
        pbo(x[:, :1])
    with pytest.raises(ValueError):
        pbo(x, metric="calmar")  # type: ignore[arg-type]
