"""Monte Carlo calibration: do the tests hold their size, and how much power do they have?

Strategies are simulated with fat tails (Student t, 5 d.f.), a common GARCH(1,1)
volatility factor and pairwise correlation 0.3, at 15% annual volatility over five years
of daily data. Four questions:

1. How large is the best Sharpe ratio among N strategies with no skill?
2. How often does each procedure declare a winner when no strategy has skill (FWER)?
3. How often does it find the one skilled strategy among 100 (power)?
4. What happens to the Reality Check and the SPA test when poor strategies are added?

Run from the repository root:  python experiments/calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import (
    MARKERS,
    RESULTS,
    method_color,
    save_figure,
    style_axes,
    style_legend,
    write_csv,
    write_json,
)
from matplotlib.ticker import PercentFormatter

from overfit import (
    adjust_pvalues,
    bootstrap_performance,
    deflated_sharpe_from_trials,
    expected_max_sharpe,
    individual_pvalues,
    probabilistic_sharpe_ratio,
)
from overfit.pbo import probability_of_backtest_overfitting
from overfit.sharpe import describe
from overfit.snooping import reality_check_from, romano_wolf_from, spa_test_from

PPY = 252
N_OBS = 5 * PPY
SIGMA = 0.15 / np.sqrt(PPY)
ALPHA = 0.05
METHODS = [
    "Naive (best p-value)",
    "Bonferroni",
    "Benjamini-Hochberg",
    "Reality Check",
    "Hansen SPA",
    "Romano-Wolf",
    "Deflated Sharpe",
]


def simulate(
    mu_daily: np.ndarray,
    rng: np.random.Generator,
    *,
    n_obs: int = N_OBS,
    rho: float = 0.3,
    df: float = 5.0,
    vol_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Returns with a common GARCH(1,1) volatility, t(5) shocks and equicorrelation rho."""
    k = mu_daily.shape[0]
    scale = np.sqrt((df - 2.0) / df)
    z = rng.standard_t(df, n_obs) * scale
    e = rng.standard_t(df, (n_obs, k)) * scale
    a, b = 0.06, 0.92
    h = np.empty(n_obs)
    h[0] = 1.0
    for t in range(1, n_obs):
        h[t] = (1 - a - b) + a * h[t - 1] * z[t - 1] ** 2 + b * h[t - 1]
    shocks = np.sqrt(rho) * z[:, None] + np.sqrt(1.0 - rho) * e
    if vol_scale is not None:
        shocks = shocks * vol_scale
    return mu_daily + SIGMA * np.sqrt(h)[:, None] * shocks


def decisions(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    """Apply every procedure to one simulated panel; column 0 is the candidate 'skilled' one."""
    k = x.shape[1]
    p = individual_pvalues(x).to_numpy()
    bp = bootstrap_performance(x, n_boot=n_boot, seed=rng)
    rw = romano_wolf_from(bp, alpha=ALPHA)
    best = int(np.argmax(bp.theta))
    if k > 1:
        dsr = deflated_sharpe_from_trials(x)
        dsr_value, dsr_best = dsr.dsr, int(dsr.selected)
    else:
        s = describe(x[:, 0])
        dsr_value = probabilistic_sharpe_ratio(s.sr, s.n_obs, skew=s.skew, kurtosis=s.kurtosis)
        dsr_best = 0
    bonf = adjust_pvalues(p, "bonferroni")
    bh = adjust_pvalues(p, "bh")
    return {
        "reject": {
            "Naive (best p-value)": p.min() <= ALPHA,
            "Bonferroni": bonf.min() <= ALPHA,
            "Benjamini-Hochberg": bh.min() <= ALPHA,
            "Reality Check": reality_check_from(bp).pvalue <= ALPHA,
            "Hansen SPA": spa_test_from(bp).pvalue <= ALPHA,
            "Romano-Wolf": rw.n_rejected > 0,
            "Deflated Sharpe": dsr_value >= 1 - ALPHA,
        },
        # Did the procedure name column 0 (the skilled strategy, when there is one)?
        "found": {
            "Naive (best p-value)": int(np.argmin(p)) == 0 and p[0] <= ALPHA,
            "Bonferroni": bonf[0] <= ALPHA,
            "Benjamini-Hochberg": bh[0] <= ALPHA,
            "Reality Check": best == 0 and reality_check_from(bp).pvalue <= ALPHA,
            "Hansen SPA": best == 0 and spa_test_from(bp).pvalue <= ALPHA,
            "Romano-Wolf": "0" in rw.rejected,
            "Deflated Sharpe": dsr_best == 0 and dsr_value >= 1 - ALPHA,
        },
    }


# ---------------------------------------------------------------------------
# 1. Expected maximum Sharpe ratio
# ---------------------------------------------------------------------------


def expected_max_experiment(rng: np.random.Generator, quick: bool) -> pd.DataFrame:
    grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] + ([] if quick else [5000])
    reps = 30 if quick else 150
    rows = []
    for n in grid:
        best = []
        for _ in range(reps):
            x = rng.standard_normal((N_OBS, n)) * SIGMA
            best.append((x.mean(0) / x.std(0, ddof=1)).max())
        best_ann = np.asarray(best) * np.sqrt(PPY)
        rows.append(
            {
                "n_trials": n,
                "simulated_mean": best_ann.mean(),
                "simulated_q05": np.quantile(best_ann, 0.05),
                "simulated_q95": np.quantile(best_ann, 0.95),
                "exact": expected_max_sharpe(n, 1.0 / N_OBS, method="exact") * np.sqrt(PPY),
                "approx": expected_max_sharpe(n, 1.0 / N_OBS) * np.sqrt(PPY),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2-3. Size and power
# ---------------------------------------------------------------------------


def size_experiment(rng: np.random.Generator, quick: bool) -> pd.DataFrame:
    grid = [1, 10, 100, 1000]
    reps = 40 if quick else 400
    n_boot = 199 if quick else 499
    rows = []
    for k in grid:
        counts = dict.fromkeys(METHODS, 0)
        for _ in range(reps):
            out = decisions(simulate(np.zeros(k), rng), rng, n_boot)
            for m in METHODS:
                counts[m] += out["reject"][m]
        rows += [
            {"n_strategies": k, "method": m, "rate": counts[m] / reps, "reps": reps}
            for m in METHODS
        ]
    return pd.DataFrame(rows)


def power_experiment(rng: np.random.Generator, quick: bool) -> pd.DataFrame:
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    reps = 40 if quick else 400
    n_boot = 199 if quick else 499
    k = 100
    rows = []
    for sr in grid:
        mu = np.zeros(k)
        mu[0] = sr / np.sqrt(PPY) * SIGMA
        rej = dict.fromkeys(METHODS, 0)
        found = dict.fromkeys(METHODS, 0)
        for _ in range(reps):
            out = decisions(simulate(mu, rng), rng, n_boot)
            for m in METHODS:
                rej[m] += out["reject"][m]
                found[m] += out["found"][m]
        rows += [
            {
                "true_sharpe": sr,
                "method": m,
                "reject": rej[m] / reps,
                "found_skilled": found[m] / reps,
                "reps": reps,
            }
            for m in METHODS
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Adding poor strategies: Reality Check vs SPA
# ---------------------------------------------------------------------------


POOR_SCENARIOS = {"clearly poor": -1.5, "mediocre": -0.3}


def poor_strategies_experiment(rng: np.random.Generator, quick: bool) -> pd.DataFrame:
    """One skilled strategy (annual Sharpe 1.5) next to many erratic losers.

    The losers are three times as volatile as the skilled strategy and have an annual
    Sharpe ratio of -1.5 ("clearly poor") or -0.3 ("mediocre").
    """
    grid = [0, 10, 100, 1000]
    reps = 40 if quick else 300
    n_boot = 199 if quick else 499
    rows = []
    for scenario, poor_sharpe in POOR_SCENARIOS.items():
        for n_bad in grid:
            k = 1 + n_bad
            vol = np.full(k, 3.0)
            vol[0] = 1.0
            mu = poor_sharpe / np.sqrt(PPY) * SIGMA * vol
            mu[0] = 1.5 / np.sqrt(PPY) * SIGMA
            hits = dict.fromkeys(["Reality Check", "Hansen SPA", "Romano-Wolf", "Step-SPA"], 0)
            for _ in range(reps):
                bp = bootstrap_performance(
                    simulate(mu, rng, vol_scale=vol), n_boot=n_boot, seed=rng
                )
                hits["Reality Check"] += reality_check_from(bp).pvalue <= ALPHA
                hits["Hansen SPA"] += spa_test_from(bp).pvalue <= ALPHA
                hits["Romano-Wolf"] += "0" in romano_wolf_from(bp, alpha=ALPHA).rejected
                hits["Step-SPA"] += (
                    "0" in romano_wolf_from(bp, alpha=ALPHA, recentre="consistent").rejected
                )
            rows += [
                {
                    "scenario": scenario,
                    "poor_sharpe": poor_sharpe,
                    "n_poor": n_bad,
                    "method": m,
                    "power": v / reps,
                    "reps": reps,
                }
                for m, v in hits.items()
            ]
    return pd.DataFrame(rows)


def pbo_experiment(rng: np.random.Generator, quick: bool) -> pd.DataFrame:
    reps = 20 if quick else 100
    rows = []
    for label, sr in (("no skill", 0.0), ("one strategy with annual Sharpe 1.5", 1.5)):
        values, losses = [], []
        for _ in range(reps):
            mu = np.zeros(100)
            mu[0] = sr / np.sqrt(PPY) * SIGMA
            res = probability_of_backtest_overfitting(simulate(mu, rng), n_splits=10)
            values.append(res.pbo)
            losses.append(res.prob_oos_loss)
        rows.append(
            {
                "scenario": label,
                "pbo_mean": np.mean(values),
                "pbo_q10": np.quantile(values, 0.1),
                "pbo_q90": np.quantile(values, 0.9),
                "prob_oos_loss_mean": np.mean(losses),
                "reps": reps,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_expected_max(df: pd.DataFrame) -> None:
    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        style_axes(ax, theme)
        sim_c, theory_c = theme["series"][0], theme["ink2"]
        ax.fill_between(
            df["n_trials"],
            df["simulated_q05"],
            df["simulated_q95"],
            color=sim_c,
            alpha=0.15,
            linewidth=0,
            label="Simulation, 5-95% range",
        )
        ax.plot(
            df["n_trials"],
            df["simulated_mean"],
            "o",
            color=sim_c,
            ms=6,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            label="Simulation, mean",
        )
        ax.plot(df["n_trials"], df["exact"], "-", color=theory_c, lw=2, label="Theory (exact)")
        ax.set_xscale("log")
        ticks = [1, 10, 100, 1000]
        ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
        ax.set_xlabel("Number of strategies tried (none has any skill)")
        ax.set_ylabel("Annualised Sharpe ratio of the best one")
        ax.set_title("The best of many zero-skill backtests looks good", loc="left")
        last = df.iloc[-1]
        ax.annotate(
            f"{int(last['n_trials']):,} tries: {last['exact']:.2f}",
            (last["n_trials"], last["exact"]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            color=theme["ink"],
            fontsize=9,
        )
        style_legend(ax.legend(loc="upper left", frameon=True), theme)
        ax.text(
            0.99,
            0.02,
            "5 years of daily returns, 15% volatility",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=theme["muted"],
            fontsize=8,
        )
        return fig

    save_figure(draw, "expected-max-sharpe")


def _line(ax: plt.Axes, x: Any, y: Any, method: str, theme: dict[str, Any], **kw: Any) -> None:
    baseline = method not in ("Romano-Wolf", "Naive (best p-value)", "Hansen SPA")
    ax.plot(
        x,
        y,
        "-",
        marker=MARKERS[method],
        color=method_color(theme, method),
        lw=1.5 if baseline else 2.0,
        ms=6,
        markeredgecolor=theme["surface"],
        markeredgewidth=1.2,
        label=method,
        **kw,
    )


def plot_size_power(size: pd.DataFrame, power: pd.DataFrame) -> None:
    shown = ["Naive (best p-value)", "Romano-Wolf", "Hansen SPA", "Deflated Sharpe"]

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax in (left, right):
            style_axes(ax, theme)
        for m in shown:
            s = size[size["method"] == m]
            _line(left, s["n_strategies"], s["rate"], m, theme)
            p = power[power["method"] == m]
            _line(right, p["true_sharpe"], p["found_skilled"], m, theme)
        left.axhline(ALPHA, color=theme["muted"], lw=1, zorder=1)
        left.text(2.0, ALPHA + 0.02, "5% target", color=theme["muted"], fontsize=8)
        naive = size[size["method"] == "Naive (best p-value)"].iloc[-1]
        left.annotate(
            f"{naive['rate']:.0%} with {int(naive['n_strategies']):,} strategies",
            (naive["n_strategies"], naive["rate"]),
            xytext=(-10, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color=theme["ink"],
        )
        left.set_xscale("log")
        left.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        left.set_ylim(-0.02, 1.02)
        left.yaxis.set_major_formatter(PercentFormatter(1.0))
        left.set_xlabel("Number of strategies tried (none has skill)")
        left.set_ylabel("Probability of declaring a winner")
        left.set_title("False discoveries", loc="left")
        right.set_ylim(-0.02, 1.02)
        right.yaxis.set_major_formatter(PercentFormatter(1.0))
        right.set_xlabel("True annual Sharpe ratio of the one skilled strategy among 100")
        right.set_ylabel("Probability it is identified")
        right.set_title("Power", loc="left")
        handles, labels = left.get_legend_handles_labels()
        leg = fig.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07)
        )
        style_legend(leg, theme)
        fig.tight_layout()
        return fig

    save_figure(draw, "size-and-power")


def plot_poor(df: pd.DataFrame) -> None:
    methods = ("Hansen SPA", "Romano-Wolf", "Reality Check")

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
        for ax, (scenario, sr) in zip(axes, POOR_SCENARIOS.items(), strict=True):
            style_axes(ax, theme)
            for m in methods:
                s = df[(df["method"] == m) & (df["scenario"] == scenario)]
                _line(ax, s["n_poor"] + 1, s["power"], m, theme)
            ax.set_xscale("log")
            ax.set_xticks([1, 11, 101, 1001], ["0", "10", "100", "1,000"])
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlabel("Losing strategies added")
            ax.set_title(f"Losers with annual Sharpe {sr:+.1f}", loc="left")
        axes[0].set_ylabel("Power at the 5% level")
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(
            handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08)
        )
        style_legend(leg, theme)
        fig.tight_layout()
        return fig

    save_figure(draw, "rc-vs-spa")


def make_figures() -> None:
    """Draw the figures from the saved result files."""
    plot_expected_max(pd.read_csv(RESULTS / "calibration_expected_max.csv"))
    plot_size_power(
        pd.read_csv(RESULTS / "calibration_size.csv"),
        pd.read_csv(RESULTS / "calibration_power.csv"),
    )
    plot_poor(pd.read_csv(RESULTS / "calibration_poor_strategies.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="few replications, for smoke tests")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from results/")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.plot_only:
        make_figures()
        return
    rng = np.random.default_rng(args.seed)
    timings = {}
    start = time.perf_counter()

    emax = expected_max_experiment(rng, args.quick)
    timings["expected_max"] = time.perf_counter() - start
    size = size_experiment(rng, args.quick)
    timings["size"] = time.perf_counter() - start
    power = power_experiment(rng, args.quick)
    timings["power"] = time.perf_counter() - start
    poor = poor_strategies_experiment(rng, args.quick)
    timings["poor"] = time.perf_counter() - start
    pbo = pbo_experiment(rng, args.quick)
    timings["pbo"] = time.perf_counter() - start

    suffix = "-quick" if args.quick else ""
    write_csv(emax, f"calibration_expected_max{suffix}.csv", index=False)
    write_csv(size, f"calibration_size{suffix}.csv", index=False)
    write_csv(power, f"calibration_power{suffix}.csv", index=False)
    write_csv(poor, f"calibration_poor_strategies{suffix}.csv", index=False)
    write_csv(pbo, f"calibration_pbo{suffix}.csv", index=False)
    write_json(
        {
            "seed": args.seed,
            "quick": args.quick,
            "n_obs": N_OBS,
            "annual_volatility": 0.15,
            "dgp": "t(5) shocks, common GARCH(1,1) (a=0.06, b=0.92), pairwise correlation 0.3",
            "alpha": ALPHA,
            "seconds": timings,
        },
        f"calibration_meta{suffix}.json",
    )
    if not args.quick:
        make_figures()

    pd.set_option("display.width", 140)
    print(emax.round(3).to_string(index=False))
    print(size.pivot(index="method", columns="n_strategies", values="rate").loc[METHODS].round(3))
    print(
        power.pivot(index="method", columns="true_sharpe", values="found_skilled")
        .loc[METHODS]
        .round(3)
    )
    print(poor.pivot_table(index=["scenario", "method"], columns="n_poor", values="power").round(3))
    print(pbo.round(3).to_string(index=False))
    print({k: round(v, 1) for k, v in timings.items()})


if __name__ == "__main__":
    main()
