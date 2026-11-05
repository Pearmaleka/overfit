from __future__ import annotations

import numpy as np
import pytest
from conftest import ar1

from overfit import bootstrap as bs
from overfit._lrv import autocovariances, newey_west_lrv, stationary_bootstrap_lrv


@pytest.mark.parametrize("method", ["stationary", "circular", "moving"])
def test_indices_are_valid(method):
    idx = bs.bootstrap_indices(97, block_size=5, n_boot=50, method=method, seed=1)
    assert idx.shape == (50, 97)
    assert idx.min() >= 0 and idx.max() < 97


def test_stationary_blocks_have_the_right_mean_length():
    n, b = 5000, 8.0
    idx = bs.bootstrap_indices(n, block_size=b, n_boot=40, seed=2)
    breaks = np.mean(idx[:, 1:] != (idx[:, :-1] + 1) % n)
    assert breaks == pytest.approx(1 / b, rel=0.05)


def test_circular_and_moving_blocks_are_contiguous():
    n, length = 100, 7
    circ = bs.bootstrap_indices(n, block_size=length, n_boot=20, method="circular", seed=3)
    for row in circ:
        for start in range(0, n - length, length):
            block = row[start : start + length]
            np.testing.assert_array_equal(block, (block[0] + np.arange(length)) % n)
    mov = bs.bootstrap_indices(n, block_size=length, n_boot=20, method="moving", seed=3)
    starts = mov[:, ::length]
    assert starts.max() <= n - length


def test_counts_agree_with_indices():
    idx = bs.bootstrap_indices(300, block_size=4, n_boot=30, seed=9)
    counts = bs.bootstrap_counts(300, block_size=4, n_boot=30, seed=9)
    expected = np.stack([np.bincount(r, minlength=300) for r in idx])
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(counts.sum(axis=1), 300)


def test_stationary_bootstrap_is_unbiased_for_the_mean(rng):
    x = rng.standard_normal(400)
    counts = bs.bootstrap_counts(400, block_size=6, n_boot=4000, seed=rng)
    assert (counts @ x / 400).mean() == pytest.approx(x.mean(), abs=0.01)
    assert counts.mean() == pytest.approx(1.0, abs=1e-12)


def test_kernel_variance_matches_bootstrap_variance(rng):
    # Politis and Romano (1994) closed form vs the actual bootstrap distribution.
    x = ar1(600, 0.6, rng)
    block = 10.0
    counts = bs.bootstrap_counts(600, block_size=block, n_boot=20000, seed=rng)
    simulated = 600 * (counts @ x / 600).var()
    assert stationary_bootstrap_lrv(x, block) == pytest.approx(simulated, rel=0.04)


def test_autocovariances_match_direct_computation(rng):
    x = rng.standard_normal((257, 3))
    acov = autocovariances(x, 10)
    c = x - x.mean(0)
    direct = np.array([(c[k:] * c[: 257 - k]).sum(0) / 257 for k in range(11)])
    np.testing.assert_allclose(acov, direct, atol=1e-12)


def test_newey_west_matches_statsmodels(rng):
    sm = pytest.importorskip("statsmodels.api")
    y = ar1(800, 0.4, rng)
    hac = {"maxlags": 7, "use_correction": False}
    fit = sm.OLS(y, np.ones(800)).fit(cov_type="HAC", cov_kwds=hac)
    assert np.sqrt(newey_west_lrv(y, 7) / 800) == pytest.approx(fit.bse[0], rel=1e-10)


def test_optimal_block_length_matches_arch(rng):
    arch_bs = pytest.importorskip("arch.bootstrap")
    for phi in (0.0, 0.3, 0.7):
        x = ar1(1500, phi, rng)
        ours = bs.optimal_block_length(x).iloc[0].to_numpy()
        theirs = arch_bs.optimal_block_length(x).iloc[0].to_numpy()
        np.testing.assert_allclose(ours, theirs, rtol=1e-8)


def test_optimal_block_length_grows_with_dependence(rng):
    weak = bs.optimal_block_length(ar1(3000, 0.1, rng)).iloc[0, 0]
    strong = bs.optimal_block_length(ar1(3000, 0.8, rng)).iloc[0, 0]
    assert strong > 3 * weak
    assert bs.auto_block_size(rng.standard_normal((500, 4))) >= 1.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        bs.bootstrap_indices(10, block_size=0.5, n_boot=5)
    with pytest.raises(ValueError):
        bs.bootstrap_indices(10, block_size=2, n_boot=5, method="block")  # type: ignore[arg-type]
