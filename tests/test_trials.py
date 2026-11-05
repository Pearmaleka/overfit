from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

from overfit.trials import correlation_matrix, effective_number_of_trials

METHODS = ["li_ji", "nyholt", "galwey"]


@pytest.mark.parametrize("method", METHODS)
def test_extremes(method):
    assert effective_number_of_trials(corr=np.eye(12), method=method) == pytest.approx(12)
    assert effective_number_of_trials(corr=np.ones((12, 12)), method=method) == pytest.approx(1)


def test_blocks_of_identical_strategies_count_once():
    corr = block_diag(np.ones((4, 4)), np.ones((3, 3)), np.ones((5, 5)))
    assert effective_number_of_trials(corr=corr, method="li_ji") == pytest.approx(3)
    assert effective_number_of_trials(corr=corr, method="galwey") == pytest.approx(
        (2 + np.sqrt(3) + np.sqrt(5)) ** 2 / 12
    )
    assert 1 < effective_number_of_trials(corr=corr, method="nyholt") < 12


def test_from_returns(rng):
    base = rng.standard_normal((2000, 5))
    x = np.repeat(base, 4, axis=1) + 0.05 * rng.standard_normal((2000, 20))
    n_eff = effective_number_of_trials(x)
    assert 4 < n_eff < 9
    np.testing.assert_allclose(correlation_matrix(x), np.corrcoef(x, rowvar=False), atol=1e-12)


def test_constant_columns_and_validation(rng):
    x = rng.standard_normal((100, 3))
    x[:, 1] = 0.0
    corr = correlation_matrix(x)
    assert corr[1, 1] == 1.0 and corr[0, 1] == 0.0
    with pytest.raises(ValueError):
        effective_number_of_trials()
    with pytest.raises(ValueError):
        effective_number_of_trials(corr=np.array([[1.0, 0.2], [0.3, 1.0]]))
