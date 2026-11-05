from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overfit import rules


def _random_walk(rng, n=1200):
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))


@pytest.mark.parametrize("family", rules.FAMILIES)
def test_no_look_ahead(family, rng):
    """Changing future prices must not change any position decided before them."""
    prices = _random_walk(rng)
    cut = 700
    altered = prices.copy()
    altered[cut + 1 :] *= np.exp(np.cumsum(rng.normal(0.0, 0.03, len(prices) - cut - 1)))
    builder = rules._BUILDERS[family]
    before, _ = builder(prices)
    after, _ = builder(altered)
    np.testing.assert_array_equal(before[: cut + 1], after[: cut + 1])
    assert not np.array_equal(before[cut + 1 :], after[cut + 1 :])


def test_default_universe(rng):
    prices = pd.Series(_random_walk(rng), index=pd.bdate_range("2010-01-01", periods=1200))
    rs = rules.build_rules(prices)
    assert rs.n_rules == 2902
    assert rs.specs.index.is_unique
    counts = rs.specs["family"].value_counts()
    assert counts["ma"] == 1164 and counts["filter"] == 144
    assert set(np.unique(rs.positions)) <= {0, 1}
    assert not rs.positions[: rs.warmup].any()
    rets = rs.returns(cost=0.0005)
    assert rets.shape == (1200 - rs.warmup - 1, rs.n_rules)
    assert rets.index[0] == prices.index[rs.warmup + 1]
    assert rs.asset_returns().shape[0] == rets.shape[0]


def test_long_short_mapping(rng):
    prices = _random_walk(rng, 600)
    ls = rules.build_rules(prices, families=["momentum"], long_short=True, warmup=300)
    lf = rules.build_rules(prices, families=["momentum"], warmup=300)
    assert set(np.unique(ls.positions[300:])) <= {-1, 1}
    np.testing.assert_array_equal(ls.positions[300:], 2 * lf.positions[300:] - 1)


def test_moving_average_rule_on_a_trend():
    up = np.linspace(100, 200, 400)
    pos, specs = rules.ma_rules(up, shorts=(5,), longs=(50,), bands=(0.0,), delays=())
    assert specs[0]["name"] == "ma:s=5,l=50,b=0,d=0"
    assert pos[:49].sum() == 0 and pos[60:].all()
    down = up[::-1]
    pos_down, _ = rules.ma_rules(down, shorts=(5,), longs=(50,), bands=(0.0,), delays=())
    assert not pos_down.any()


def test_filter_rule_by_hand():
    prices = np.array([100, 98, 96, 97, 101, 105, 110, 104, 103, 99, 100.0])
    pos, _ = rules.filter_rules(prices, entries=(0.05,), exits=(0.05,))
    # trough 96 -> 5% rise reached at 101 (t=4); peak 110 -> 5% fall reached at 104 (t=7)
    np.testing.assert_array_equal(pos[:, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])


def test_breakout_uses_previous_highs_only():
    prices = np.array([10, 11, 12, 11, 13, 12, 11, 10, 9, 12.0])
    pos, _ = rules.breakout_rules(prices, windows=(3,), exit_fractions=(1.0,), bands=(0.0,))
    # t=4: 13 > max(11, 12, 11) = 12 -> enter; t=6: 11 is not below min(12, 13, 12)... the low
    # of the previous three closes is 11, so t=7 (10 < 11) exits; t=9: 12 > max(11, 10, 9) enters.
    np.testing.assert_array_equal(pos[:, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1])


def test_rsi_bounds_and_flat_prices(rng):
    r = rules.rsi(_random_walk(rng, 300), 14)
    valid = r[np.isfinite(r)]
    assert valid.min() >= 0 and valid.max() <= 100
    flat = rules.rsi(np.full(50, 10.0), 5)
    assert np.allclose(flat[np.isfinite(flat)], 50.0)


def test_rule_returns_and_costs():
    pos = np.array([[0], [1], [1], [0], [0]], dtype=np.int8)
    r = np.array([0.0, 0.01, 0.02, -0.01, 0.03])
    out = rules.rule_returns(pos, r, cost=0.001)
    # held: 0, 0, 1, 1, 0 ; trades at t=1 (enter) and t=3 (exit), charged a period later
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.02 - 0.001, -0.01, -0.001])
    np.testing.assert_allclose(rules.rule_returns(pos, r)[:, 0], [0, 0, 0.02, -0.01, 0])


def test_invalid_inputs(rng):
    with pytest.raises(ValueError):
        rules.build_rules(_random_walk(rng, 100), families=["astrology"])
    with pytest.raises(ValueError):
        rules.sma(np.array([1.0, -2.0, 3.0]), 2)
    with pytest.raises(ValueError):
        rules.rule_returns(np.zeros((5, 1)), np.zeros(4))
