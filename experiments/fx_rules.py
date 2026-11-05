"""Technical trading rules on ten currencies since 1974: did the edge disappear?

Early studies found that simple trend rules earned significant profits in currency
markets in the 1970s and 1980s, and later work reported that those profits faded
(e.g. Neely, Weller and Ulrich, 2009). This experiment asks the question with today's
multiple-testing tools: for each currency and each period, is the best of 2,902
long/short rules significantly profitable after accounting for the search?

Data: daily USD price of AUD, CAD, CHF, DKK, EUR (from 1999), GBP, JPY, NOK, NZD and SEK
(Federal Reserve H.10). Rules trade at the close with a one-day lag and pay 2.5 basis
points per unit of turnover (5 bp to reverse a position). The benchmark is no position;
interest-rate differentials (carry) are not available and are ignored.

Run from the repository root:  python experiments/fx_rules.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RESULTS, THEMES, fx_rates, save_figure, style_axes, write_csv, write_json
from matplotlib.colors import LinearSegmentedColormap

from overfit import bootstrap_performance
from overfit.rules import build_rules, rule_returns, simple_returns
from overfit.snooping import romano_wolf_from, spa_test_from

PPY = 252
COST = 0.00025
START = "1973-03-01"  # floating exchange rates
PERIODS = {
    "1974-1983": ("1974-01-01", "1983-12-31"),
    "1984-1993": ("1984-01-01", "1993-12-31"),
    "1994-2003": ("1994-01-01", "2003-12-31"),
    "2004-2013": ("2004-01-01", "2013-12-31"),
    "2014-2026": ("2014-01-01", "2026-12-31"),
}
MIN_COVERAGE = 0.9


def analyse_currency(code: str, prices: pd.Series, n_boot: int, seed: int, quick: bool):
    prices = prices.loc[START:].dropna()
    families = ("ma", "momentum") if quick else None
    rules = (
        build_rules(prices, families=families, long_short=True)
        if families
        else build_rules(prices, long_short=True)
    )
    rets_all = pd.DataFrame(
        rule_returns(rules.positions, simple_returns(prices), cost=COST),
        index=prices.index,
        columns=rules.names,
    ).iloc[rules.warmup + 1 :]
    rows, scatter = [], {}
    for label, (lo, hi) in PERIODS.items():
        rets = rets_all.loc[lo:hi]
        end = min(pd.Timestamp(hi), prices.index[-1])
        expected_days = len(pd.bdate_range(lo, end)) * 0.96  # about 4% are holidays
        if len(rets) < MIN_COVERAGE * expected_days:
            continue
        bp = bootstrap_performance(rets, n_boot=n_boot, seed=seed)
        spa = spa_test_from(bp)
        rw = romano_wolf_from(bp)
        step = romano_wolf_from(bp, recentre="consistent")
        best = int(np.argmax(bp.theta))
        t_best = np.sqrt(bp.n_obs) * bp.theta[best] / bp.omega[best]
        null = np.sqrt(bp.n_obs) * (bp.theta_star[:, best] - bp.theta[best]) / bp.omega[best]
        sr = rets.mean() / rets.std(ddof=1) * np.sqrt(PPY)
        scatter[label] = sr
        rows.append(
            {
                "currency": code,
                "period": label,
                "days": len(rets),
                "best_rule": bp.labels[best],
                "best_annual_return": float(bp.theta[best] * PPY),
                "best_sharpe": float(sr.iloc[best]),
                "median_rule_sharpe": float(sr.median()),
                "pvalue_naive": float((1 + np.count_nonzero(null >= t_best)) / (n_boot + 1)),
                "pvalue_spa": spa.pvalue,
                "pvalue_reality_check_studentized_upper": spa.pvalue_upper,
                "n_romano_wolf": rw.n_rejected,
                "n_step_spa": step.n_rejected,
                "block_size": bp.block_size,
            }
        )
    return rows, pd.DataFrame(scatter)


def persistence(sharpes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sharpe ratio, in the next period, of the rule that was best in the previous one."""
    labels = list(PERIODS)
    rows = []
    for code, table in sharpes.items():
        for prev, nxt in itertools.pairwise(labels):
            if prev not in table or nxt not in table:
                continue
            winner = table[prev].idxmax()
            rows.append(
                {
                    "currency": code,
                    "selection_period": prev,
                    "evaluation_period": nxt,
                    "winner": winner,
                    "winner_sharpe_selection": float(table.loc[winner, prev]),
                    "winner_sharpe_next": float(table.loc[winner, nxt]),
                    "all_rules_mean_sharpe_next": float(table[nxt].mean()),
                    "top_decile_mean_sharpe_next": float(
                        table.loc[table[prev] >= table[prev].quantile(0.9), nxt].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame) -> None:
    grid = results.pivot(index="currency", columns="period", values="pvalue_spa").reindex(
        columns=list(PERIODS)
    )
    evidence = -np.log10(grid.clip(lower=1e-3))

    def draw(theme: dict[str, Any]) -> plt.Figure:
        cmap = LinearSegmentedColormap.from_list("evidence", theme["sequential"])
        cmap.set_bad(theme["surface"])
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        style_axes(ax, theme)
        ax.grid(False)
        img = ax.imshow(
            np.ma.masked_invalid(evidence.to_numpy()), cmap=cmap, vmin=0, vmax=3, aspect="auto"
        )
        ax.set_xticks(range(grid.shape[1]), grid.columns)
        ax.set_yticks(range(grid.shape[0]), grid.index)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        dark_theme = theme["surface"] == THEMES["dark"]["surface"]
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                p = grid.iloc[i, j]
                if np.isnan(p):
                    ax.text(j, i, "n/a", ha="center", va="center", color=theme["muted"], fontsize=8)
                elif p <= 0.10:
                    strong = evidence.iloc[i, j] > 1.6
                    color = ("#0b0b0b" if dark_theme else "#ffffff") if strong else theme["ink"]
                    label = "<0.001" if p < 0.001 else f"{p:.3f}"
                    ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
        bar = fig.colorbar(img, ax=ax, ticks=[0, 1, 2, 3], fraction=0.05, pad=0.02)
        bar.ax.set_yticklabels(["1", "0.1", "0.01", "\u2264 0.001"], color=theme["ink2"])
        bar.outline.set_visible(False)
        bar.set_label("SPA p-value of the best of 2,902 rules", color=theme["ink2"])
        ax.set_title("Currency trading rules: significant then, not now", loc="left")
        fig.text(
            0.01,
            -0.02,
            "Cells show p-values at or below 0.10. Long/short rules, 2.5 bp per "
            "unit turnover, no carry.",
            color=theme["muted"],
            fontsize=8,
        )
        return fig

    save_figure(draw, "fx-spa-heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two rule families, few bootstraps")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from results/")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.plot_only:
        plot_heatmap(pd.read_csv(RESULTS / "fx_rules.csv"))
        return
    n_boot = 199 if args.quick else 1999
    start = time.perf_counter()
    fx = fx_rates()
    codes = list(fx.columns[:3]) if args.quick else list(fx.columns)
    rows, sharpes = [], {}
    for code in codes:
        r, s = analyse_currency(code, fx[code], n_boot, args.seed, args.quick)
        rows += r
        sharpes[code] = s
        print(f"{code}: {len(r)} periods, {time.perf_counter() - start:.0f}s", flush=True)
    results = pd.DataFrame(rows)
    pers = persistence(sharpes)
    by_period = results.groupby("period", sort=False).agg(
        currencies=("currency", "count"),
        significant_spa=("pvalue_spa", lambda p: int((p <= 0.05).sum())),
        significant_naive=("pvalue_naive", lambda p: int((p <= 0.05).sum())),
        median_best_sharpe=("best_sharpe", "median"),
        median_rule_sharpe=("median_rule_sharpe", "median"),
        total_step_spa_rules=("n_step_spa", "sum"),
    )
    by_next = pers.groupby("evaluation_period", sort=False)[
        ["winner_sharpe_selection", "winner_sharpe_next", "top_decile_mean_sharpe_next"]
    ].median()

    suffix = "-quick" if args.quick else ""
    write_csv(results, f"fx_rules{suffix}.csv", index=False)
    write_csv(pers, f"fx_rules_persistence{suffix}.csv", index=False)
    write_json(
        {
            "by_period": by_period.reset_index().to_dict(orient="records"),
            "persistence_medians": by_next.reset_index().to_dict(orient="records"),
            "n_rules": 2902 if not args.quick else None,
            "cost_per_unit_turnover": COST,
            "n_boot": n_boot,
            "seed": args.seed,
            "seconds": time.perf_counter() - start,
        },
        f"fx_rules_summary{suffix}.json",
    )
    if not args.quick:
        plot_heatmap(results)
    pd.set_option("display.width", 160)
    print(results.pivot(index="currency", columns="period", values="pvalue_spa").round(3))
    print(by_period.round(3))
    print(by_next.round(3))


if __name__ == "__main__":
    main()
