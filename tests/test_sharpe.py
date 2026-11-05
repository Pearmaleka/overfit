from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import ar1

from overfit import sharpe as sh


def test_sharpe_ratio_shapes_and_annualisation(rng):
    x = rng.normal(0.001, 0.01, size=(500, 3))
    per_period = sh.sharpe_ratio(x)
    assert per_period.shape == (3,)
    np.testing.assert_allclose(per_period, x.mean(0) / x.std(0, ddof=1))
    np.testing.assert_allclose(sh.sharpe_ratio(x, periods_per_year=252), per_period * np.sqrt(252))
    assert isinstance(sh.sharpe_ratio(x[:, 0]), float)
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    out = sh.sharpe_ratio(frame)
    assert isinstance(out, pd.Series) and list(out.index) == ["a", "b", "c"]


def test_sharpe_ratio_risk_free_and_constant_series():
    x = np.array([0.01, 0.02, 0.03, 0.04])
    assert sh.sharpe_ratio(x, risk_free=0.01) == pytest.approx(0.015 / x.std(ddof=1))
    assert sh.sharpe_ratio(np.full(10, 0.01)) == 0.0


def test_iid_standard_error_matches_monte_carlo(rng):
    n, reps, true_sr = 500, 4000, 0.1
    sims = rng.normal(true_sr, 1.0, size=(reps, n))
    srs = sims.mean(1) / sims.std(1, ddof=1)
    se = np.mean([sh.sharpe_ratio_se(s, method="iid") for s in sims[:200]])
    assert se == pytest.approx(srs.std(), rel=0.05)


def test_nonnormal_standard_error_matches_monte_carlo(rng):
    # Negatively skewed, fat-tailed returns: the normal formula understates the error.
    n, reps = 1000, 3000
    sims = 0.05 - rng.gamma(2.0, 0.5, size=(reps, n)) + 1.0
    srs = sims.mean(1) / sims.std(1, ddof=1)
    nonnormal = np.mean([sh.sharpe_ratio_se(s, method="nonnormal") for s in sims[:200]])
    iid = np.mean([sh.sharpe_ratio_se(s, method="iid") for s in sims[:200]])
    assert nonnormal == pytest.approx(srs.std(), rel=0.06)
    assert abs(iid - srs.std()) > abs(nonnormal - srs.std())


def test_hac_standard_error_handles_autocorrelation(rng):
    n, reps, phi = 1000, 1500, 0.5
    sims = np.array([0.1 + ar1(n, phi, rng) for _ in range(reps)])
    srs = sims.mean(1) / sims.std(1, ddof=1)
    hac = np.mean([sh.sharpe_ratio_se(s, method="hac") for s in sims[:150]])
    iid = np.mean([sh.sharpe_ratio_se(s, method="iid") for s in sims[:150]])
    # Positive autocorrelation inflates the true error well above the iid formula.
    assert srs.std() > 1.5 * iid
    assert hac == pytest.approx(srs.std(), rel=0.12)


def test_describe_fields(rng):
    x = rng.normal(0.0005, 0.01, 2000)
    s = sh.describe(x, periods_per_year=252)
    assert s.n_obs == 2000
    assert s.sr_annual == pytest.approx(s.sr * np.sqrt(252))
    assert s.kurtosis == pytest.approx(3.0, abs=0.3)
    assert 0.0 < s.psr < 1.0
    assert s.tstat_hac == pytest.approx(s.sr / s.se_hac)


def test_psr_basic_properties():
    assert sh.probabilistic_sharpe_ratio(0.1, 500, sr_benchmark=0.1) == pytest.approx(0.5)
    low = sh.probabilistic_sharpe_ratio(0.05, 250)
    high = sh.probabilistic_sharpe_ratio(0.05, 2500)
    assert low < high
    # Negative skew and fat tails reduce confidence in a positive Sharpe ratio.
    assert sh.probabilistic_sharpe_ratio(
        0.05, 1000, skew=-2, kurtosis=10
    ) < sh.probabilistic_sharpe_ratio(0.05, 1000)
    with pytest.raises(ValueError):
        sh.probabilistic_sharpe_ratio(1.0, 100, skew=3.0, kurtosis=3.0)


def test_min_track_record_length_is_consistent_with_psr():
    sr, skew, kurt, alpha = 0.08, -0.5, 6.0, 0.05
    n = sh.min_track_record_length(sr, skew=skew, kurtosis=kurt, alpha=alpha)
    psr = sh.probabilistic_sharpe_ratio(sr, n, skew=skew, kurtosis=kurt)
    assert psr == pytest.approx(1 - alpha)
    assert sh.min_track_record_length(0.0) == float("inf")


def test_deflated_sharpe_ratio_reproduces_published_example():
    # Bailey and Lopez de Prado (2014), Section 6: annualised SR 2.5 over 5 years of
    # daily data, 100 trials with annualised SR variance 0.5, skew -3, kurtosis 10.
    sr = 2.5 / np.sqrt(250)
    sr_var = 0.5 / 250
    sr0 = sh.expected_max_sharpe(100, sr_var)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    dsr = sh.deflated_sharpe_ratio(sr, 1250, sr_variance=sr_var, n_trials=100, skew=-3, kurtosis=10)
    assert dsr == pytest.approx(0.9004, abs=1e-3)


def test_expected_max_sharpe_exact_and_approx(rng):
    draws = rng.standard_normal((20000, 100)).max(axis=1).mean()
    exact = sh.expected_max_sharpe(100, method="exact")
    assert exact == pytest.approx(2.50759, abs=1e-4)
    assert exact == pytest.approx(draws, abs=0.01)
    approx = sh.expected_max_sharpe(100)
    assert approx == pytest.approx(exact, rel=0.02)
    assert sh.expected_max_sharpe(1) == 0.0
    assert sh.expected_max_sharpe(2, method="exact") == pytest.approx(1 / np.sqrt(np.pi), abs=1e-6)
    grid = [sh.expected_max_sharpe(n, 0.04, method="exact") for n in (2, 10, 100, 1000, 10000)]
    assert np.all(np.diff(grid) > 0)


def test_deflated_sharpe_from_trials(rng):
    noise = rng.normal(0.0, 0.01, size=(1260, 200))
    res = sh.deflated_sharpe_from_trials(noise, periods_per_year=252)
    assert res.selected == str(int(np.argmax(noise.mean(0) / noise.std(0, ddof=1))))
    assert res.psr > res.dsr
    assert res.dsr < 0.95
    assert res.sr0_annual == pytest.approx(res.sr0 * np.sqrt(252))
    chosen = sh.deflated_sharpe_from_trials(noise, selected=3)
    assert chosen.selected == "3"
    with pytest.raises(ValueError):
        sh.deflated_sharpe_from_trials(noise[:, :1])


def test_haircut_sharpe_ratio():
    single = sh.haircut_sharpe_ratio(0.06, 1260, 1)
    assert single.haircut == pytest.approx(0.0, abs=1e-12)
    many = sh.haircut_sharpe_ratio(0.1, 1260, 200)
    assert 0.3 < many.haircut < 0.7
    assert many.pvalue_adjusted == pytest.approx(min(1.0, 200 * many.pvalue))
    sidak = sh.haircut_sharpe_ratio(0.1, 1260, 200, method="sidak")
    assert sidak.pvalue_adjusted <= many.pvalue_adjusted
    hopeless = sh.haircut_sharpe_ratio(0.01, 252, 1000)
    assert hopeless.sr_haircut == 0.0 and hopeless.haircut == 1.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sh.sharpe_ratio([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        sh.sharpe_ratio_se(np.arange(10.0), method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sh.expected_max_sharpe(0.5)
