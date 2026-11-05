"""overfit: statistical tests for backtest overfitting and data snooping.

Is the best of many backtests skill or luck? The library answers with the deflated
Sharpe ratio, the probability of backtest overfitting, White's Reality Check, Hansen's
SPA test, the Romano-Wolf step-down procedure and purged cross-validation.
"""

from overfit.bootstrap import bootstrap_counts, bootstrap_indices, optimal_block_length
from overfit.multitest import adjust_pvalues
from overfit.sharpe import (
    deflated_sharpe_from_trials,
    deflated_sharpe_ratio,
    describe,
    expected_max_sharpe,
    haircut_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_ratio_se,
)
from overfit.snooping import (
    bootstrap_performance,
    individual_pvalues,
    reality_check,
    romano_wolf,
    spa_test,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "adjust_pvalues",
    "bootstrap_counts",
    "bootstrap_indices",
    "bootstrap_performance",
    "deflated_sharpe_from_trials",
    "deflated_sharpe_ratio",
    "describe",
    "expected_max_sharpe",
    "haircut_sharpe_ratio",
    "individual_pvalues",
    "min_track_record_length",
    "optimal_block_length",
    "probabilistic_sharpe_ratio",
    "reality_check",
    "romano_wolf",
    "sharpe_ratio",
    "sharpe_ratio_se",
    "spa_test",
]
