"""Command-line interface: ``overfit audit returns.csv`` and ``overfit demo``."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from overfit import __version__
from overfit.report import AuditReport, audit


def _read_table(path: str, *, prices: bool) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    with contextlib.suppress(ValueError, TypeError):
        table.index = pd.to_datetime(table.index)
    table = table.apply(pd.to_numeric, errors="coerce")
    if prices:
        table = table.pct_change(fill_method=None).iloc[1:]
    table = table.dropna(axis=0, how="any")
    if table.empty:
        raise SystemExit(f"{path}: no complete rows of numeric data")
    return table


def _emit(report: AuditReport, fmt: str, output: str | None) -> None:
    text = (
        json.dumps(report.to_dict(), indent=2, default=float)
        if fmt == "json"
        else report.to_markdown()
    )
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"wrote {output}")
    else:
        sys.stdout.write(text)


def _demo_returns(n_strategies: int, years: float, seed: int) -> pd.DataFrame:
    """Daily returns of strategies with no skill: zero mean, 15% volatility, correlated."""
    rng = np.random.default_rng(seed)
    n = round(252 * years)
    vol = 0.15 / np.sqrt(252)
    common = rng.standard_t(5, size=(n, 1)) / np.sqrt(5 / 3)
    own = rng.standard_t(5, size=(n, n_strategies)) / np.sqrt(5 / 3)
    rets = vol * (np.sqrt(0.3) * common + np.sqrt(0.7) * own)
    index = pd.bdate_range("2020-01-01", periods=n)
    names = [f"strategy_{i:03d}" for i in range(n_strategies)]
    return pd.DataFrame(rets, index=index, columns=names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overfit",
        description="Check whether the best of many backtests reflects skill or luck.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit the strategies stored in a CSV file")
    a.add_argument("path", help="CSV with dates in the first column and one column per strategy")
    a.add_argument("--benchmark", help="name of the column holding benchmark returns")
    a.add_argument("--prices", action="store_true", help="columns hold prices, not returns")
    a.add_argument("--periods-per-year", type=float, default=252.0)
    a.add_argument("--n-trials", type=float, help="number of trials run, if more than the columns")
    a.add_argument("--criterion", choices=["sharpe", "mean"], default="sharpe")
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--format", choices=["markdown", "json"], default="markdown")
    a.add_argument("-o", "--output", help="write the report to this file")

    d = sub.add_parser("demo", help="audit strategies that have no skill by construction")
    d.add_argument("--strategies", type=int, default=200)
    d.add_argument("--years", type=float, default=5.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--format", choices=["markdown", "json"], default="markdown")
    d.add_argument("-o", "--output", help="write the report to this file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        table = _read_table(args.path, prices=args.prices)
        bench = None
        if args.benchmark:
            if args.benchmark not in table.columns:
                raise SystemExit(f"column {args.benchmark!r} not found in {args.path}")
            bench = table.pop(args.benchmark)
        report = audit(
            table,
            bench,
            periods_per_year=args.periods_per_year,
            n_trials=args.n_trials,
            criterion=args.criterion,
            alpha=args.alpha,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    else:
        table = _demo_returns(args.strategies, args.years, args.seed)
        report = audit(table, seed=args.seed)
    _emit(report, args.format, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
