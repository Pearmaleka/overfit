"""Fill the result tables in docs/results.md from the files in experiments/results/.

Each table sits between ``<!-- table:NAME -->`` and ``<!-- /table -->`` markers and is
regenerated in place, so the numbers in the write-up always match the result files.

Run from the repository root:  python experiments/tables.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable

import pandas as pd
from _common import RESULTS, ROOT

DOC = ROOT / "docs" / "results.md"
METHOD_ORDER = [
    "Naive (best p-value)",
    "Bonferroni",
    "Benjamini-Hochberg",
    "Reality Check",
    "Hansen SPA",
    "Romano-Wolf",
    "Deflated Sharpe",
]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def markdown(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def expected_max() -> str:
    df = pd.read_csv(RESULTS / "calibration_expected_max.csv")
    out = pd.DataFrame(
        {
            "Strategies tried": [f"{n:,}" for n in df["n_trials"]],
            "Simulated mean": df["simulated_mean"].map("{:.2f}".format),
            "Simulated 5-95%": [
                f"{a:.2f} to {b:.2f}"
                for a, b in zip(df["simulated_q05"], df["simulated_q95"], strict=True)
            ],
            "Exact": df["exact"].map("{:.2f}".format),
            "Approximation": df["approx"].map("{:.2f}".format),
        }
    )
    return markdown(out)


def size() -> str:
    df = pd.read_csv(RESULTS / "calibration_size.csv")
    wide = df.pivot(index="method", columns="n_strategies", values="rate").loc[METHOD_ORDER]
    wide.columns = [f"{c:,} strateg{'y' if c == 1 else 'ies'}" for c in wide.columns]
    wide = wide.map(pct).reset_index().rename(columns={"method": "Procedure"})
    return markdown(wide)


def power() -> str:
    df = pd.read_csv(RESULTS / "calibration_power.csv")
    wide = df.pivot(index="method", columns="true_sharpe", values="found_skilled").loc[METHOD_ORDER]
    wide.columns = [f"SR {c:g}" for c in wide.columns]
    wide = wide.map(pct).reset_index().rename(columns={"method": "Procedure"})
    return markdown(wide)


def poor() -> str:
    df = pd.read_csv(RESULTS / "calibration_poor_strategies.csv")
    order = ["Reality Check", "Romano-Wolf", "Hansen SPA", "Step-SPA"]
    rows = []
    for (_, sr), part in df.groupby(["scenario", "poor_sharpe"], sort=False):
        wide = part.pivot(index="method", columns="n_poor", values="power").loc[order]
        for method, values in wide.iterrows():
            rows.append(
                {"Losers' Sharpe": f"{sr:+.1f}", "Procedure": method}
                | {f"{n:,} losers": pct(v) for n, v in values.items()}
            )
    return markdown(pd.DataFrame(rows))


def pbo_sim() -> str:
    df = pd.read_csv(RESULTS / "calibration_pbo.csv")
    out = pd.DataFrame(
        {
            "Scenario (100 strategies)": df["scenario"],
            "Mean PBO": df["pbo_mean"].map("{:.2f}".format),
            "10-90% range": [
                f"{a:.2f} to {b:.2f}" for a, b in zip(df["pbo_q10"], df["pbo_q90"], strict=True)
            ],
            "P(winner loses out of sample)": df["prob_oos_loss_mean"].map("{:.2f}".format),
        }
    )
    return markdown(out)


def _equity() -> dict:
    return json.loads((RESULTS / "equity_rules.json").read_text())


def equity_tests() -> str:
    eq = _equity()
    rows = []
    labels = {
        "tests_sharpe": "Sharpe ratio, 5 bp costs (main)",
        "tests_mean": "Mean return, 5 bp costs",
        "tests_sharpe_block20": "Sharpe ratio, block length 20",
        "tests_sharpe_no_costs": "Sharpe ratio, no costs",
    }
    for key in ("sp500", "nasdaq"):
        s = eq[key]
        for test, label in labels.items():
            t = s[test]
            rows.append(
                {
                    "Index": s["index"],
                    "Specification": label,
                    "Best rule": f"`{t['best_rule']}`",
                    "Naive p": pval(t["pvalue_naive"]),
                    "Reality Check p": pval(t["pvalue_reality_check"]),
                    "SPA p (lower / consistent / upper)": f"{pval(t['pvalue_spa_lower'])} / "
                    f"{pval(t['pvalue_spa'])} / {pval(t['pvalue_spa_upper'])}",
                    "Rules significant (Romano-Wolf / step-SPA)": f"{t['n_romano_wolf']} / "
                    f"{t['n_step_spa']}",
                }
            )
    return markdown(pd.DataFrame(rows))


def equity_summary() -> str:
    eq = _equity()
    rows = []
    for key in ("sp500", "nasdaq"):
        s = eq[key]
        b, w, d, p, sp = (
            s["buy_and_hold"],
            s["best_rule"],
            s["deflated_sharpe"],
            s["pbo"],
            s["split_sample"],
        )
        rows += [
            {
                "": "Buy-and-hold Sharpe ratio (annual return, max drawdown)",
                key: f"{b['sharpe']:.2f} "
                f"({100 * b['annual_return']:.1f}%, {100 * b['max_drawdown']:.0f}%)",
            },
            {"": "Best rule by Sharpe ratio", key: f"`{w['name']}`"},
            {
                "": "Its Sharpe ratio (annual return, max drawdown)",
                key: f"{w['sharpe']:.2f} "
                f"({100 * w['annual_return']:.1f}%, {100 * w['max_drawdown']:.0f}%)",
            },
            {
                "": "Best excess Sharpe ratio over buy-and-hold (PSR)",
                key: f"{d['excess_sharpe']:.2f} ({d['psr']:.2f})",
            },
            {
                "": "Expected best excess Sharpe ratio with no skill",
                key: f"{d['expected_max_excess_sharpe']:.2f}",
            },
            {
                "": "Deflated Sharpe ratio (effective trials)",
                key: f"{d['dsr']:.2f} "
                f"({d['dsr_effective']:.2f} with {d['n_trials_effective']:.0f})",
            },
            {"": "Probability of backtest overfitting", key: f"{p['pbo']:.2f}"},
            {
                "": "2000-2009 winner: Sharpe in / out of sample",
                key: f"{sp['winner_is_sharpe']:.2f} / {sp['winner_oos_sharpe']:.2f}",
            },
            {
                "": "Buy-and-hold Sharpe 2000-2009 / 2010-2018",
                key: f"{sp['buy_and_hold_is_sharpe']:.2f} / {sp['buy_and_hold_oos_sharpe']:.2f}",
            },
            {
                "": "Rules beating buy-and-hold 2000-2009 / 2010-2018",
                key: f"{pct(sp['share_rules_beating_bh_is'])} / "
                f"{pct(sp['share_rules_beating_bh_oos'])}",
            },
            {
                "": "Rank correlation of rule Sharpe ratios across halves",
                key: f"{sp['spearman_is_oos']:.2f}",
            },
        ]
    frame = pd.DataFrame(rows).groupby("", sort=False).first().reset_index()
    frame.columns = ["", "S&P 500", "NASDAQ Composite"]
    return markdown(frame)


def fx_grid() -> str:
    df = pd.read_csv(RESULTS / "fx_rules.csv")
    periods = list(dict.fromkeys(df["period"]))
    wide = df.pivot(index="currency", columns="period", values="pvalue_spa")[sorted(periods)]
    wide = wide.map(lambda p: "" if pd.isna(p) else (f"**{pval(p)}**" if p <= 0.05 else pval(p)))
    return markdown(wide.reset_index().rename(columns={"currency": "Currency"}))


def fx_periods() -> str:
    summary = json.loads((RESULTS / "fx_rules_summary.json").read_text())
    by = pd.DataFrame(summary["by_period"])
    pers = pd.DataFrame(summary["persistence_medians"]).set_index("evaluation_period")
    out = pd.DataFrame(
        {
            "Period": by["period"],
            "Currencies": by["currencies"],
            "Best rule significant, naive": [
                f"{a} of {n}"
                for a, n in zip(by["significant_naive"], by["currencies"], strict=True)
            ],
            "Best rule significant, SPA": [
                f"{a} of {n}" for a, n in zip(by["significant_spa"], by["currencies"], strict=True)
            ],
            "Rules significant, step-SPA (all currencies)": by["total_step_spa_rules"],
            "Median best Sharpe": by["median_best_sharpe"].map("{:.2f}".format),
            "Median rule Sharpe": by["median_rule_sharpe"].map("{:.2f}".format),
            "Previous winner's Sharpe (median)": [
                f"{pers.loc[p, 'winner_sharpe_next']:.2f}" if p in pers.index else ""
                for p in by["period"]
            ],
        }
    )
    return markdown(out)


def cv_table() -> str:
    results = pd.read_csv(RESULTS / "cv_leakage.csv")
    holdout = pd.read_csv(RESULTS / "cv_leakage_holdout.csv")
    mean = results.groupby(["data", "scheme", "horizon"])["auc"].mean()
    schemes = ["Purged k-fold (1% embargo)", "Contiguous k-fold", "Shuffled k-fold"]
    names = {"synthetic": "Synthetic, no signal", "S&P 500": "S&P 500"}
    rows = []
    for data in ("synthetic", "S&P 500"):
        for scheme in schemes:
            values = mean.loc[(data, scheme)]
            rows.append(
                {"Data": names[data], "Scheme": scheme}
                | {f"h = {h}": f"{v:.3f}" for h, v in values.items()}
            )
    hold = holdout.groupby("horizon")["auc"].mean()
    rows.append(
        {"Data": "S&P 500", "Scheme": "Chronological hold-out, 2013-2018"}
        | {f"h = {h}": f"{v:.3f}" for h, v in hold.items()}
    )
    return markdown(pd.DataFrame(rows))


TABLES: dict[str, Callable[[], str]] = {
    "expected_max": expected_max,
    "size": size,
    "power": power,
    "poor": poor,
    "pbo_sim": pbo_sim,
    "equity_tests": equity_tests,
    "equity_summary": equity_summary,
    "fx_grid": fx_grid,
    "fx_periods": fx_periods,
    "cv": cv_table,
}


def render(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return f"<!-- table:{name} -->\n{TABLES[name]()}\n<!-- /table -->"

    return re.sub(r"<!-- table:(\w+) -->.*?<!-- /table -->", replace, text, flags=re.S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the doc is out of date")
    args = parser.parse_args()
    text = DOC.read_text(encoding="utf-8")
    new = render(text)
    if args.check:
        if new != text:
            sys.exit("docs/results.md is out of date: run python experiments/tables.py")
        print("docs/results.md tables are up to date")
        return
    DOC.write_text(new, encoding="utf-8")
    print(f"updated {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
