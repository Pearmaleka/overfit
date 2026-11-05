"""Technical trading rules on the S&P 500 and the NASDAQ Composite: skill or luck?

For each index, 2,902 long/flat rules from ``overfit.rules`` trade at the close with a
one-day lag and pay 5 basis points per unit of turnover. The benchmark is buy-and-hold
(price index, no dividends; cash earns nothing, which roughly offsets the missing
dividend yield over 1999-2018).

Questions:
1. Full sample (2000-2018): does the best rule beat buy-and-hold once we account for
   having tried 2,902 rules? (naive bootstrap p-value, Reality Check, SPA, Romano-Wolf,
   step-wise SPA, deflated Sharpe ratio of returns in excess of buy-and-hold)
2. How likely is the in-sample winner to be overfit? (PBO by CSCV)
3. Split sample: choose the best rule on 2000-2009 and trade it on 2010-2018.

Run from the repository root:  python experiments/equity_rules.py [--quick]
"""

from __future__ import annotations

import argparse
import json
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
from scipy import stats

from overfit import bootstrap_performance, effective_number_of_trials
from overfit.pbo import probability_of_backtest_overfitting
from overfit.rules import build_rules
from overfit.sharpe import deflated_sharpe_ratio, describe, expected_max_sharpe, sharpe_ratio
from overfit.snooping import (
    BootstrapPerformance,
    reality_check_from,
    romano_wolf_from,
    spa_test_from,
)

PPY = 252
COST = 0.0005
INDICES = {"sp500": "S&P 500", "nasdaq": "NASDAQ Composite"}
SPLIT = "2010-01-01"


def annual_stats(r: pd.Series | np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1.0 + r)
    drawdown = 1.0 - wealth / np.maximum.accumulate(wealth)
    return {
        "annual_return": float((wealth[-1]) ** (PPY / len(r)) - 1.0),
        "annual_volatility": float(r.std(ddof=1) * np.sqrt(PPY)),
        "sharpe": float(sharpe_ratio(r) * np.sqrt(PPY)),
        "max_drawdown": float(drawdown.max()),
    }


def naive_pvalue(bp: BootstrapPerformance, k: int) -> float:
    """Single-test bootstrap p-value of strategy k, ignoring that others were tried."""
    t = np.sqrt(bp.n_obs) * bp.theta[k] / bp.omega[k]
    null = np.sqrt(bp.n_obs) * (bp.theta_star[:, k] - bp.theta[k]) / bp.omega[k]
    return float((1 + np.count_nonzero(null >= t)) / (bp.n_boot + 1))


def snooping_tests(
    rets: pd.DataFrame,
    bench: pd.Series,
    *,
    criterion: str,
    n_boot: int,
    seed: int,
    block_size: float | None = None,
) -> dict[str, Any]:
    bp = bootstrap_performance(
        rets,
        bench,
        criterion=criterion,
        n_boot=n_boot,
        block_size=block_size,
        seed=seed,  # type: ignore[arg-type]
    )
    best = int(np.argmax(bp.theta))
    spa = spa_test_from(bp)
    rw = romano_wolf_from(bp)
    step_spa = romano_wolf_from(bp, recentre="consistent")
    single = np.sqrt(bp.n_obs) * bp.theta / bp.omega
    return {
        "criterion": criterion,
        "block_size": bp.block_size,
        "best_rule": bp.labels[best],
        "best_difference": float(bp.theta[best]),
        "pvalue_naive": naive_pvalue(bp, best),
        "share_rules_naive_p05": float(np.mean(stats.norm.sf(single) <= 0.05)),
        "pvalue_reality_check": reality_check_from(bp).pvalue,
        "pvalue_spa_lower": spa.pvalue_lower,
        "pvalue_spa": spa.pvalue,
        "pvalue_spa_upper": spa.pvalue_upper,
        "n_romano_wolf": rw.n_rejected,
        "n_step_spa": step_spa.n_rejected,
        "romano_wolf_rejected": rw.rejected[:20],
    }


def analyse_index(name: str, n_boot: int, seed: int, quick: bool) -> dict[str, Any]:
    prices = equity_index(name)
    families = ("ma", "momentum") if quick else None
    rules = build_rules(prices, families=families) if families else build_rules(prices)
    rets = rules.returns(cost=COST)
    bench = rules.asset_returns()
    active = rets.sub(bench, axis=0)
    out: dict[str, Any] = {
        "index": INDICES[name],
        "n_rules": rules.n_rules,
        "start": rets.index[0],
        "end": rets.index[-1],
        "n_days": len(rets),
        "cost_per_unit_turnover": COST,
        "buy_and_hold": annual_stats(bench),
    }

    # 1. Full-sample data-snooping tests
    out["tests_sharpe"] = snooping_tests(rets, bench, criterion="sharpe", n_boot=n_boot, seed=seed)
    out["tests_mean"] = snooping_tests(rets, bench, criterion="mean", n_boot=n_boot, seed=seed)
    out["tests_sharpe_block20"] = snooping_tests(
        rets, bench, criterion="sharpe", n_boot=n_boot, seed=seed, block_size=20
    )
    out["tests_sharpe_no_costs"] = snooping_tests(
        rules.returns(cost=0.0), bench, criterion="sharpe", n_boot=n_boot, seed=seed
    )
    best = out["tests_sharpe"]["best_rule"]
    pos = rules.positions[rules.warmup :, rules.names.index(best)]
    out["best_rule"] = {
        "name": best,
        **annual_stats(rets[best]),
        "time_in_market": float(pos.mean()),
        "trades_per_year": float(np.abs(np.diff(pos)).sum() / (len(pos) / PPY)),
    }

    # Deflated Sharpe ratio of the best rule's returns in excess of buy-and-hold
    srs = np.asarray(sharpe_ratio(active.to_numpy()))
    k_best = int(np.argmax(srs))
    s = describe(active.iloc[:, k_best])
    sr_var = float(np.var(srs, ddof=1))
    n_eff = effective_number_of_trials(active.to_numpy())
    out["deflated_sharpe"] = {
        "rule": active.columns[k_best],
        "excess_sharpe": s.sr * np.sqrt(PPY),
        "psr": s.psr,
        "n_trials": rules.n_rules,
        "n_trials_effective": n_eff,
        "expected_max_excess_sharpe": expected_max_sharpe(rules.n_rules, sr_var) * np.sqrt(PPY),
        "dsr": deflated_sharpe_ratio(
            s.sr,
            s.n_obs,
            sr_variance=sr_var,
            n_trials=rules.n_rules,
            skew=s.skew,
            kurtosis=s.kurtosis,
        ),
        "dsr_effective": deflated_sharpe_ratio(
            s.sr, s.n_obs, sr_variance=sr_var, n_trials=n_eff, skew=s.skew, kurtosis=s.kurtosis
        ),
    }

    # 2. Probability of backtest overfitting
    res = probability_of_backtest_overfitting(rets, n_splits=16, metric="sharpe")
    slope, intercept = res.degradation
    out["pbo"] = {
        "pbo": res.pbo,
        "n_combinations": res.n_combinations,
        "prob_oos_loss": res.prob_oos_loss,
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "median_logit": float(np.median(res.logits)),
    }
    pbo_logits = res.logits

    # 3. Split sample: select on 2000-2009, evaluate on 2010-2018
    is_rets, oos_rets = rets.loc[: pd.Timestamp(SPLIT) - pd.Timedelta(days=1)], rets.loc[SPLIT:]
    is_sr = np.asarray(sharpe_ratio(is_rets.to_numpy())) * np.sqrt(PPY)
    oos_sr = np.asarray(sharpe_ratio(oos_rets.to_numpy())) * np.sqrt(PPY)
    winner = int(np.argmax(is_sr))
    top_decile = is_sr >= np.quantile(is_sr, 0.9)
    out["split_sample"] = {
        "in_sample": [is_rets.index[0], is_rets.index[-1]],
        "out_of_sample": [oos_rets.index[0], oos_rets.index[-1]],
        "winner": rets.columns[winner],
        "winner_is_sharpe": float(is_sr[winner]),
        "winner_oos_sharpe": float(oos_sr[winner]),
        "winner_oos_percentile": float(stats.percentileofscore(oos_sr, oos_sr[winner]) / 100),
        "buy_and_hold_is_sharpe": annual_stats(bench.loc[is_rets.index])["sharpe"],
        "buy_and_hold_oos_sharpe": annual_stats(bench.loc[oos_rets.index])["sharpe"],
        "share_rules_beating_bh_is": float(
            np.mean(is_sr > annual_stats(bench.loc[is_rets.index])["sharpe"])
        ),
        "share_rules_beating_bh_oos": float(
            np.mean(oos_sr > annual_stats(bench.loc[oos_rets.index])["sharpe"])
        ),
        "top_decile_mean_oos_sharpe": float(oos_sr[top_decile].mean()),
        "all_rules_mean_oos_sharpe": float(oos_sr.mean()),
        "spearman_is_oos": float(stats.spearmanr(is_sr, oos_sr).statistic),
    }
    scatter = pd.DataFrame({"rule": rets.columns, "is_sharpe": is_sr, "oos_sharpe": oos_sr})
    return {"summary": out, "scatter": scatter, "logits": pbo_logits}


def plot_split(results: dict[str, dict[str, Any]]) -> None:
    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.4))
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes, results.items(), strict=True):
            style_axes(ax, theme)
            sc = res["scatter"]
            split = res["summary"]["split_sample"]
            ax.scatter(
                sc["is_sharpe"],
                sc["oos_sharpe"],
                s=6,
                color=theme["background_marks"],
                linewidths=0,
                label=f"Each of the {len(sc):,} rules",
                rasterized=True,
            )
            ax.axhline(split["buy_and_hold_oos_sharpe"], color=theme["muted"], lw=1)
            ax.axvline(split["buy_and_hold_is_sharpe"], color=theme["muted"], lw=1)
            ax.text(
                ax.get_xlim()[0],
                split["buy_and_hold_oos_sharpe"],
                "  buy-and-hold",
                va="bottom",
                color=theme["muted"],
                fontsize=8,
            )
            ax.scatter(
                [split["winner_is_sharpe"]],
                [split["winner_oos_sharpe"]],
                s=70,
                color=theme["series"][0],
                edgecolors=theme["surface"],
                linewidths=2,
                zorder=5,
                label="In-sample winner",
            )
            ax.annotate(
                f"winner: {split['winner_is_sharpe']:.2f} in sample,\n"
                f"{split['winner_oos_sharpe']:.2f} out of sample",
                (split["winner_is_sharpe"], split["winner_oos_sharpe"]),
                xytext=(-10, -28),
                textcoords="offset points",
                ha="right",
                color=theme["ink"],
                fontsize=8,
            )
            ax.set_title(INDICES[name], loc="left")
            ax.set_xlabel("Sharpe ratio 2000-2009 (selection)")
            ax.set_ylabel("Sharpe ratio 2010-2018 (out of sample)")
        style_legend(axes[0].legend(loc="upper left", frameon=True, markerscale=1.5), theme)
        fig.text(
            0.01,
            -0.02,
            "Grey lines: Sharpe ratio of buy-and-hold in each period. Rules are long or flat, "
            "trade with a one-day lag and pay 5 bp per trade.",
            color=theme["muted"],
            fontsize=8,
        )
        fig.tight_layout()
        return fig

    save_figure(draw, "equity-split-sample")


def plot_logits(results: dict[str, dict[str, Any]]) -> None:
    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes, results.items(), strict=True):
            style_axes(ax, theme)
            logits = res["logits"]
            lo, hi = np.floor(logits.min()), np.ceil(logits.max())
            bins = np.linspace(lo, hi, 41)
            ax.hist(
                logits,
                bins=bins,
                color=theme["series"][0],
                edgecolor=theme["surface"],
                linewidth=1.0,
            )
            ax.axvline(0, color=theme["muted"], lw=1)
            pbo = res["summary"]["pbo"]["pbo"]
            ax.set_title(f"{INDICES[name]}: PBO = {pbo:.2f}", loc="left")
            ax.set_xlabel("Logit of the winner's out-of-sample rank (left of 0 = below median)")
        axes[0].set_ylabel("Train/test splits")
        fig.tight_layout()
        return fig

    save_figure(draw, "equity-pbo-logits")


def summary_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results.values():
        s = res["summary"]
        t = s["tests_sharpe"]
        rows.append(
            {
                "index": s["index"],
                "rules": s["n_rules"],
                "buy_and_hold_sharpe": s["buy_and_hold"]["sharpe"],
                "best_rule": t["best_rule"],
                "best_rule_sharpe": s["best_rule"]["sharpe"],
                "pvalue_naive": t["pvalue_naive"],
                "pvalue_reality_check": t["pvalue_reality_check"],
                "pvalue_spa": t["pvalue_spa"],
                "romano_wolf_rejections": t["n_romano_wolf"],
                "step_spa_rejections": t["n_step_spa"],
                "dsr_effective": s["deflated_sharpe"]["dsr_effective"],
                "pbo": s["pbo"]["pbo"],
                "winner_is_sharpe": s["split_sample"]["winner_is_sharpe"],
                "winner_oos_sharpe": s["split_sample"]["winner_oos_sharpe"],
                "buy_and_hold_oos_sharpe": s["split_sample"]["buy_and_hold_oos_sharpe"],
            }
        )
    return pd.DataFrame(rows)


def make_figures() -> None:
    """Draw the figures from the saved result files."""
    summary = json.loads((RESULTS / "equity_rules.json").read_text())
    results = {
        name: {
            "summary": summary[name],
            "scatter": pd.read_csv(RESULTS / f"equity_rules_split_{name}.csv"),
            "logits": pd.read_csv(RESULTS / f"equity_rules_pbo_logits_{name}.csv")[
                "logit"
            ].to_numpy(),
        }
        for name in INDICES
    }
    plot_split(results)
    plot_logits(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two rule families, few bootstraps")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from results/")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.plot_only:
        make_figures()
        return
    n_boot = 199 if args.quick else 1999
    start = time.perf_counter()
    results = {name: analyse_index(name, n_boot, args.seed, args.quick) for name in INDICES}
    suffix = "-quick" if args.quick else ""
    write_json(
        {name: r["summary"] for name, r in results.items()}
        | {"seconds": time.perf_counter() - start, "n_boot": n_boot, "seed": args.seed},
        f"equity_rules{suffix}.json",
    )
    table = summary_table(results)
    write_csv(table, f"equity_rules_summary{suffix}.csv", index=False)
    for name, r in results.items():
        write_csv(r["scatter"], f"equity_rules_split_{name}{suffix}.csv", index=False)
        write_csv(
            pd.DataFrame({"logit": r["logits"]}),
            f"equity_rules_pbo_logits_{name}{suffix}.csv",
            index=False,
        )
    if not args.quick:
        make_figures()
    pd.set_option("display.width", 160)
    print(table.T.to_string())
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
