"""Overlapping labels let ordinary k-fold cross-validation see the future.

Task: predict whether the next ``h``-day return is positive from trailing features
(past returns over 1-60 days, trailing volatility, distance from the 50-day average).
For ``h > 1`` the labels of neighbouring days overlap, and the features are smooth in
time. Shuffled k-fold puts near-identical neighbours on both sides of the split, so a
flexible model scores well by memorising them.

Two data sets:
* synthetic returns with no predictability at all (true AUC = 0.5), ten years, six draws;
* the S&P 500, 1999-2018, where the reference is a chronological hold-out (train up to
  2012, test 2013-2018).

Three cross-validation schemes, five folds each: shuffled k-fold, contiguous (unshuffled)
k-fold, and purged k-fold with a 1% embargo (``overfit.PurgedKFold``).

Run from the repository root:  python experiments/cv_leakage.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import (
    RESULTS,
    equity_index,
    save_figure,
    style_axes,
    style_legend,
    write_csv,
    write_json,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from overfit import PurgedKFold

SCHEMES = ("Purged k-fold (1% embargo)", "Shuffled k-fold", "Contiguous k-fold")
SCHEME_MARKERS = {
    "Purged k-fold (1% embargo)": "o",
    "Shuffled k-fold": "X",
    "Contiguous k-fold": "s",
}


def features(prices: pd.Series) -> pd.DataFrame:
    r = prices.pct_change()
    out = {f"ret_{n}": prices.pct_change(n) for n in (1, 5, 10, 20, 60)}
    out |= {f"vol_{n}": r.rolling(n).std() for n in (20, 60)}
    out["ma_gap_50"] = prices / prices.rolling(50).mean() - 1.0
    return pd.DataFrame(out)


def dataset(prices: pd.Series, horizon: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, labels and label end positions, without warm-up rows or unlabeled tail."""
    x = features(prices)
    fwd = prices.shift(-horizon) / prices - 1.0
    keep = x.notna().all(axis=1) & fwd.notna()
    x, fwd = x[keep], fwd[keep]
    y = (fwd > 0).astype(int).to_numpy()
    end = np.minimum(np.arange(len(x)) + horizon, len(x) - 1)
    return x, y, end


def model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, max_features=3, n_jobs=2, random_state=seed
    )


def cv_auc(x: pd.DataFrame, y: np.ndarray, splitter: Any, seed: int) -> float:
    scores = []
    for train, test in splitter.split(x):
        if len(np.unique(y[test])) < 2:
            continue
        fit = model(seed).fit(x.iloc[train], y[train])
        scores.append(roc_auc_score(y[test], fit.predict_proba(x.iloc[test])[:, 1]))
    return float(np.mean(scores))


def splitters(end: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "Purged k-fold (1% embargo)": PurgedKFold(5, t1=end, embargo=0.01),
        "Shuffled k-fold": KFold(5, shuffle=True, random_state=seed),
        "Contiguous k-fold": KFold(5, shuffle=False),
    }


def synthetic_prices(rng: np.random.Generator, n: int) -> pd.Series:
    r = rng.normal(0.0003, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2000-01-03", periods=n))


def run(horizons: list[int], seeds: list[int], quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rng = np.random.default_rng(2026)
    n_synth = 1260 if quick else 2520
    for seed in seeds:
        prices = synthetic_prices(rng, n_synth)
        for h in horizons:
            x, y, end = dataset(prices, h)
            for scheme, cv in splitters(end, seed).items():
                rows.append(
                    {
                        "data": "synthetic",
                        "seed": seed,
                        "horizon": h,
                        "scheme": scheme,
                        "auc": cv_auc(x, y, cv, seed),
                    }
                )
        print(f"synthetic seed {seed} done", flush=True)

    sp = equity_index("sp500")
    holdout_rows = []
    for h in horizons:
        x, y, end = dataset(sp, h)
        for seed in seeds[: 1 if quick else 3]:
            for scheme, cv in splitters(end, seed).items():
                rows.append(
                    {
                        "data": "S&P 500",
                        "seed": seed,
                        "horizon": h,
                        "scheme": scheme,
                        "auc": cv_auc(x, y, cv, seed),
                    }
                )
            # Chronological hold-out: train before 2013 (minus h days so labels do not
            # reach into the test period), test 2013-2018.
            cut = int(np.searchsorted(x.index, pd.Timestamp("2013-01-01")))
            train = np.arange(0, cut - h)
            test = np.arange(cut, len(x))
            fit = model(seed).fit(x.iloc[train], y[train])
            holdout_rows.append(
                {
                    "horizon": h,
                    "seed": seed,
                    "auc": roc_auc_score(y[test], fit.predict_proba(x.iloc[test])[:, 1]),
                }
            )
        print(f"S&P 500 horizon {h} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(holdout_rows)


def plot(results: pd.DataFrame, holdout: pd.DataFrame) -> None:
    summary = (
        results.groupby(["data", "scheme", "horizon"])["auc"].agg(["mean", "std"]).reset_index()
    )
    hold = holdout.groupby("horizon")["auc"].mean()

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
        titles = {
            "synthetic": "Synthetic returns, no predictability",
            "S&P 500": "S&P 500, 1999-2018",
        }
        for ax, data in zip(axes, ("synthetic", "S&P 500"), strict=True):
            style_axes(ax, theme)
            ax.axhline(0.5, color=theme["muted"], lw=1)
            for slot, scheme in enumerate(SCHEMES):
                s = summary[(summary["data"] == data) & (summary["scheme"] == scheme)]
                ax.plot(
                    s["horizon"],
                    s["mean"],
                    "-",
                    marker=SCHEME_MARKERS[scheme],
                    color=theme["series"][slot],
                    lw=2,
                    ms=6,
                    markeredgecolor=theme["surface"],
                    label=scheme,
                )
            if data == "S&P 500":
                ax.plot(
                    hold.index,
                    hold.to_numpy(),
                    linestyle="none",
                    marker="D",
                    ms=6,
                    color=theme["muted"],
                    label="Chronological hold-out (2013-2018)",
                )
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 20, 40, 60], ["1", "5", "10", "20", "40", "60"])
            ax.set_xlabel("Label horizon h (days ahead)")
            ax.set_title(titles[data], loc="left")
        axes[0].set_ylabel("Cross-validated ROC AUC")
        axes[0].text(2.2, 0.505, "no skill", color=theme["muted"], fontsize=8)
        handles, labels = [], []
        for ax in axes:
            hnd, lab = ax.get_legend_handles_labels()
            for a, b in zip(hnd, lab, strict=True):
                if b not in labels:
                    handles.append(a)
                    labels.append(b)
        leg = fig.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07)
        )
        style_legend(leg, theme)
        fig.tight_layout()
        return fig

    save_figure(draw, "cv-leakage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from results/")
    args = parser.parse_args()
    if args.plot_only:
        plot(
            pd.read_csv(RESULTS / "cv_leakage.csv"),
            pd.read_csv(RESULTS / "cv_leakage_holdout.csv"),
        )
        return
    start = time.perf_counter()
    horizons = [1, 20] if args.quick else [1, 5, 10, 20, 40, 60]
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4, 5]
    suffix = "-quick" if args.quick else ""
    results, holdout = run(horizons, seeds, args.quick)
    write_csv(results, f"cv_leakage{suffix}.csv", index=False)
    write_csv(holdout, f"cv_leakage_holdout{suffix}.csv", index=False)
    table = results.pivot_table(index=["data", "scheme"], columns="horizon", values="auc")
    write_json(
        {
            "mean_auc": {f"{d} | {s}": row.round(4).to_dict() for (d, s), row in table.iterrows()},
            "holdout_auc": holdout.groupby("horizon")["auc"].mean().round(4).to_dict(),
            "seconds": time.perf_counter() - start,
        },
        f"cv_leakage_summary{suffix}.json",
    )
    if not args.quick:
        plot(results, holdout)
    pd.set_option("display.width", 160)
    print(table.round(3))
    print(holdout.groupby("horizon")["auc"].mean().round(3))


if __name__ == "__main__":
    main()
