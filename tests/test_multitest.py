from __future__ import annotations

import numpy as np
import pytest

from overfit.multitest import METHODS, adjust_pvalues

STATSMODELS_NAME = {
    "bonferroni": "bonferroni",
    "sidak": "sidak",
    "holm": "holm",
    "hochberg": "simes-hochberg",
    "bh": "fdr_bh",
    "by": "fdr_by",
}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("method", METHODS)
def test_matches_statsmodels(method, rng):
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = np.concatenate([rng.uniform(0, 0.05, 20), rng.uniform(0, 1, 80), [0.01, 0.01, 1.0]])
    rng.shuffle(p)
    expected = multitest.multipletests(p, method=STATSMODELS_NAME[method])[1]
    np.testing.assert_allclose(adjust_pvalues(p, method), expected, atol=1e-12)


def test_orderings_between_methods(rng):
    p = rng.uniform(0, 0.2, 50)
    bonf, holm, bh, by = (adjust_pvalues(p, m) for m in ("bonferroni", "holm", "bh", "by"))
    assert np.all(holm <= bonf + 1e-15)
    assert np.all(bh <= holm + 1e-15)
    assert np.all(by >= bh - 1e-15)
    assert np.all((bonf >= p) & (bonf <= 1))


def test_edge_cases():
    assert adjust_pvalues([], "bh").size == 0
    np.testing.assert_allclose(adjust_pvalues([0.03], "holm"), [0.03])
    with pytest.raises(ValueError):
        adjust_pvalues([0.5, 1.5])
    with pytest.raises(ValueError):
        adjust_pvalues([0.5], "nope")  # type: ignore[arg-type]
