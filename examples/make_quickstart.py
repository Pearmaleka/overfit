"""Build and execute ``examples/quickstart.ipynb``.

Run from the repository root:  python examples/make_quickstart.py
(needs ``nbformat``, ``nbclient`` and ``ipykernel``).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# Quickstart: is the best backtest skill or luck?

We try 200 trading strategies that have **no skill at all** and look at the best one, first
the way a backtest report would, then with `overfit`.
""",
    ),
    (
        "code",
        """\
import numpy as np
import pandas as pd

import overfit

rng = np.random.default_rng(42)
n_days, n_strategies = 5 * 252, 200
daily_vol = 0.15 / np.sqrt(252)
market = rng.standard_normal((n_days, 1))
noise = rng.standard_normal((n_days, n_strategies))
returns = pd.DataFrame(
    daily_vol * (np.sqrt(0.3) * market + np.sqrt(0.7) * noise),
    index=pd.bdate_range("2020-01-01", periods=n_days),
    columns=[f"s{i:03d}" for i in range(n_strategies)],
)
sharpe = overfit.sharpe_ratio(returns, periods_per_year=252)
best = sharpe.idxmax()
stats = overfit.describe(returns[best], periods_per_year=252)
print(
    f"best strategy {best}: annual Sharpe {stats.sr_annual:.2f}, "
    f"probabilistic Sharpe ratio {stats.psr:.3f}"
)
""",
    ),
    (
        "markdown",
        """\
On its own the best strategy looks convincing. But it is the maximum of 200 tries. The
**deflated Sharpe ratio** measures it against the Sharpe ratio we would expect from the best
of 200 strategies with no skill:
""",
    ),
    (
        "code",
        """\
dsr = overfit.deflated_sharpe_from_trials(returns, periods_per_year=252)
print(f"expected best Sharpe with no skill: {dsr.sr0_annual:.2f}")
print(f"deflated Sharpe ratio: {dsr.dsr:.3f}   (needs >= 0.95)")
""",
    ),
    (
        "markdown",
        """\
The bootstrap tests ask the same question without distributional assumptions and keep the
correlation between strategies:
""",
    ),
    (
        "code",
        """\
spa = overfit.spa_test(returns, criterion="sharpe", seed=0)
rw = overfit.romano_wolf(returns, criterion="sharpe", seed=0)
pbo = overfit.pbo(returns)
print(f"Hansen SPA p-value: {spa.pvalue:.3f}")
print(f"strategies significant after Romano-Wolf: {rw.n_rejected}")
print(f"probability of backtest overfitting: {pbo.pbo:.2f}")
""",
    ),
    (
        "markdown",
        """\
`overfit.audit` runs every check at once and writes a report:
""",
    ),
    (
        "code",
        """\
from IPython.display import Markdown

report = overfit.audit(returns, seed=0)
Markdown(report.to_markdown())
""",
    ),
    (
        "markdown",
        """\
## A strategy that does have an edge

Give one strategy a true annual Sharpe ratio of about 2.5 and the verdict changes:
""",
    ),
    (
        "code",
        """\
skilled = returns.copy()
skilled["s007"] += 2.5 * daily_vol / np.sqrt(252)
report = overfit.audit(skilled, seed=0)
print(report.best, "|", report.verdict)
print(
    f"DSR {report.dsr:.3f}, SPA p-value {report.pvalue_spa:.3f}, "
    f"Romano-Wolf rejections {report.n_romano_wolf}"
)
""",
    ),
    (
        "markdown",
        """\
## Real data: 1,244 moving-average and momentum rules on the S&P 500

The index data ship with the `arch` package (1999-2018). Each rule is long or flat, trades
with a one-day lag and pays 5 basis points per trade; the benchmark is buy-and-hold.
""",
    ),
    (
        "code",
        """\
from arch.data import sp500

from overfit.rules import build_rules

prices = sp500.load()["Adj Close"]
rules = build_rules(prices, families=["ma", "momentum"])
rule_returns = rules.returns(cost=0.0005)
buy_and_hold = rules.asset_returns()
report = overfit.audit(rule_returns, buy_and_hold, seed=0)
Markdown(report.to_markdown())
""",
    ),
    (
        "markdown",
        """\
## Cross-validation with overlapping labels

When each label looks 20 days ahead, neighbouring samples share most of their label.
`PurgedKFold` drops those neighbours from training; it plugs into scikit-learn like any
other splitter.
""",
    ),
    (
        "code",
        """\
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

r = prices.pct_change()
X = pd.concat({f"ret_{n}": prices.pct_change(n) for n in (1, 5, 20, 60)}, axis=1)
X["vol_20"] = r.rolling(20).std()
y = (prices.shift(-20) / prices - 1 > 0).astype(int)
keep = X.notna().all(axis=1) & prices.shift(-20).notna()
X, y = X[keep], y[keep]
t1 = overfit.label_times(len(X), horizon=20)

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0)
for name, cv in {
    "shuffled k-fold": KFold(5, shuffle=True, random_state=0),
    "purged k-fold": overfit.PurgedKFold(5, t1=t1, embargo=0.01),
}.items():
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"{name:>16}: AUC {auc:.3f}")
""",
    ),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, source in CELLS:
        maker = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(maker(source.strip("\n")))
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
    )
    client.execute()
    out = HERE / "quickstart.ipynb"
    nbformat.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
