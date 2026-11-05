from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overfit import snooping as sn


def _null_plus_one(rng, n=750, k=30, edge=0.0022):
    x = rng.normal(0.0, 0.01, size=(n, k))
    x[:, 7] += edge
    return x


def test_strong_strategy_is_detected_by_all_three(rng):
    x = _null_plus_one(rng)
    rc = sn.reality_check(x, n_boot=999, seed=1)
    spa = sn.spa_test(x, n_boot=999, seed=1)
    rw = sn.romano_wolf(x, n_boot=999, seed=1)
    assert rc.best == "7" and spa.best == "7"
    assert rc.pvalue < 0.01 and spa.pvalue < 0.01
    assert rw.rejected == ["7"]
    assert rw.pvalues["7"] < 0.01


def test_spa_pvalues_are_ordered_and_upper_equals_reality_check(rng):
    x = rng.normal(0.0, 0.01, size=(500, 40))
    x[:, :20] -= 0.002  # clearly bad strategies
    bp = sn.bootstrap_performance(x, n_boot=999, seed=3)
    spa = sn.spa_test_from(bp)
    assert spa.pvalue_lower <= spa.pvalue <= spa.pvalue_upper
    raw = sn.spa_test_from(bp, studentize=False)
    assert raw.pvalue_upper == pytest.approx(sn.reality_check_from(bp).pvalue)


def test_spa_keeps_power_when_bad_strategies_are_added(rng):
    good = rng.normal(0.0012, 0.01, size=(1000, 1))
    bad = rng.normal(-0.004, 0.03, size=(1000, 300))
    alone = sn.bootstrap_performance(good, n_boot=999, seed=5)
    crowded = sn.bootstrap_performance(np.hstack([good, bad]), n_boot=999, seed=5)
    assert sn.spa_test_from(crowded).pvalue < 0.05
    # The Reality Check recentres the noisy losers and loses its power.
    assert sn.reality_check_from(crowded).pvalue > 3 * sn.reality_check_from(alone).pvalue
    assert sn.reality_check_from(crowded).pvalue > sn.spa_test_from(crowded).pvalue


def test_benchmark_and_dataframe_labels(rng):
    market = rng.normal(0.0004, 0.01, 600)
    strategies = pd.DataFrame(
        {"timing": market * 0.5, "levered": market * 1.5 + 0.0, "same": market}
    )
    res = sn.reality_check(strategies, market, n_boot=499, seed=0)
    assert res.best in strategies.columns
    rw = sn.romano_wolf(strategies, market, n_boot=499, seed=0)
    assert list(rw.pvalues.index) == list(strategies.columns)
    assert rw.statistics["same"] == 0.0  # identical to the benchmark
    with pytest.raises(ValueError):
        sn.reality_check(strategies, market[:-1])


def test_sharpe_criterion(rng):
    x = rng.normal(0.0, 0.01, size=(800, 20))
    x[:, 3] = rng.normal(0.0004, 0.002, 800)  # small mean, tiny volatility: high Sharpe
    by_mean = sn.reality_check(x, n_boot=499, seed=2)
    by_sharpe = sn.spa_test(x, criterion="sharpe", n_boot=499, seed=2)
    assert by_sharpe.best == "3"
    assert by_sharpe.pvalue < 0.01
    assert by_mean.best != "3" or by_mean.pvalue > by_sharpe.pvalue


def test_romano_wolf_pvalues_are_monotone_in_the_statistic(rng):
    x = rng.normal(0.0, 0.01, size=(600, 25))
    x[:, :5] += np.linspace(0.0005, 0.002, 5)
    rw = sn.romano_wolf(x, n_boot=999, seed=4)
    order = np.argsort(-rw.statistics.to_numpy())
    assert np.all(np.diff(rw.pvalues.to_numpy()[order]) >= 0)
    assert set(rw.rejected) <= {str(i) for i in range(5)}
    assert rw.n_rejected >= 2


def test_spa_agrees_with_arch_without_studentisation(rng):
    arch_bs = pytest.importorskip("arch.bootstrap")
    x = rng.normal(0.0, 0.01, size=(1000, 15))
    x[:, 2] += 0.0006
    x[:, 9] -= 0.002
    bench = rng.normal(0.0, 0.002, 1000)
    ours = sn.spa_test(x, bench, n_boot=5000, block_size=5, studentize=False, seed=11)
    ref = arch_bs.SPA(-bench, -x, block_size=5, reps=5000, seed=11)
    ref.compute()
    theirs = ref.pvalues
    assert ours.pvalue_lower == pytest.approx(theirs["lower"], abs=0.03)
    assert ours.pvalue == pytest.approx(theirs["consistent"], abs=0.03)
    assert ours.pvalue_upper == pytest.approx(theirs["upper"], abs=0.03)


def test_individual_pvalues_match_a_hac_t_test(rng):
    sm = pytest.importorskip("statsmodels.api")
    x = rng.normal(0.0003, 0.01, size=(700, 3))
    p = sn.individual_pvalues(x, lags=5)
    fit = sm.OLS(x[:, 0], np.ones(700)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False}
    )
    from scipy import stats

    assert p.iloc[0] == pytest.approx(stats.norm.sf(fit.tvalues[0]), rel=1e-8)


@pytest.mark.slow
def test_family_wise_error_rate_under_the_global_null():
    rng = np.random.default_rng(0)
    reps, n, k = 200, 400, 10
    reject = {"naive": 0, "rc": 0, "spa": 0, "rw": 0}
    for _ in range(reps):
        common = rng.standard_normal((n, 1))
        x = 0.01 * (0.5 * common + rng.standard_normal((n, k)))
        bp = sn.bootstrap_performance(x, n_boot=299, block_size=2, seed=rng)
        reject["naive"] += sn.individual_pvalues(x).min() <= 0.05
        reject["rc"] += sn.reality_check_from(bp).pvalue <= 0.05
        reject["spa"] += sn.spa_test_from(bp).pvalue <= 0.05
        reject["rw"] += sn.romano_wolf_from(bp).n_rejected > 0
    rates = {key: v / reps for key, v in reject.items()}
    assert rates["naive"] > 0.2
    for key in ("rc", "spa", "rw"):
        assert rates[key] <= 0.10, rates
