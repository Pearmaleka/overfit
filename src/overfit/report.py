"""One-call audit of a set of backtests: is the best one skill or luck?"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special

from overfit._utils import SeedLike, as_matrix, as_vector, make_rng
from overfit.pbo import probability_of_backtest_overfitting
from overfit.sharpe import (
    deflated_sharpe_ratio,
    describe,
    expected_max_sharpe,
    haircut_sharpe_ratio,
    sharpe_ratio,
)
from overfit.snooping import (
    bootstrap_performance,
    reality_check_from,
    romano_wolf_from,
    spa_test_from,
)
from overfit.trials import effective_number_of_trials


@dataclass(frozen=True)
class AuditReport:
    """Results of :func:`audit`. Sharpe ratios are annualised."""

    n_obs: int
    n_strategies: int
    periods_per_year: float
    criterion: str
    benchmark: str
    best: str
    sharpe: float
    sharpe_se: float
    pvalue_naive: float
    n_trials: float
    n_trials_effective: float
    expected_max_sharpe: float
    dsr: float
    dsr_effective: float
    sharpe_haircut: float
    haircut: float
    pvalue_reality_check: float
    pvalue_spa: float
    n_romano_wolf: int
    romano_wolf_rejected: tuple[str, ...]
    pbo: float | None
    pbo_prob_oos_loss: float | None
    alpha: float
    block_size: float
    n_boot: int

    @property
    def verdict(self) -> str:
        survives = [
            self.dsr_effective >= 1 - self.alpha,
            self.pvalue_spa <= self.alpha,
            self.n_romano_wolf > 0,
        ]
        if self.pvalue_naive > self.alpha:
            return "No evidence of skill, even before correcting for the number of trials."
        if all(survives) and (self.pbo is None or self.pbo < 0.5):
            return "Evidence of skill survives the multiple-testing corrections."
        if not any(survives):
            return (
                "Significant on its own, but not after accounting for the trials: "
                "consistent with selection luck."
            )
        return "Mixed evidence: some corrections reject the null and others do not."

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["romano_wolf_rejected"] = list(self.romano_wolf_rejected)
        out["verdict"] = self.verdict
        return out

    def to_markdown(self) -> str:
        def fmt(v: float | None, digits: int = 3) -> str:
            return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"

        level = f"{100 * self.alpha:g}%"
        sharpe_label = (
            "Annualised Sharpe ratio"
            if self.benchmark == "zero"
            else "Annualised Sharpe ratio of returns in excess of the benchmark"
        )
        rows = [
            ("Best strategy (in-sample)", f"`{self.best}`"),
            (sharpe_label, f"{fmt(self.sharpe, 2)} (HAC s.e. {fmt(self.sharpe_se, 2)})"),
            ("Naive one-sided p-value", fmt(self.pvalue_naive, 4)),
            ("Trials (effective)", f"{self.n_trials:.0f} ({self.n_trials_effective:.1f})"),
            ("Expected best Sharpe under no skill", fmt(self.expected_max_sharpe, 2)),
            (
                f"Deflated Sharpe ratio (pass if >= {1 - self.alpha:.2f})",
                f"{fmt(self.dsr)} ({fmt(self.dsr_effective)} with effective trials)",
            ),
            (
                "Haircut Sharpe ratio (Bonferroni)",
                f"{fmt(self.sharpe_haircut, 2)} ({100 * self.haircut:.0f}% haircut)"
                if np.isfinite(self.haircut)
                else "n/a",
            ),
            ("White Reality Check p-value", fmt(self.pvalue_reality_check)),
            ("Hansen SPA p-value (consistent)", fmt(self.pvalue_spa)),
            (f"Strategies significant after Romano-Wolf ({level} FWER)", str(self.n_romano_wolf)),
            ("Probability of backtest overfitting (CSCV)", fmt(self.pbo)),
            ("P(in-sample winner loses out of sample)", fmt(self.pbo_prob_oos_loss)),
        ]
        lines = [
            "# Backtest audit",
            "",
            f"{self.n_strategies} strategies, {self.n_obs} observations "
            f"({self.n_obs / self.periods_per_year:.1f} years), criterion: {self.criterion}, "
            f"benchmark: {self.benchmark}.",
            "",
            "| Check | Result |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in rows],
            "",
            f"**Verdict:** {self.verdict}",
        ]
        if self.romano_wolf_rejected:
            shown = ", ".join(f"`{s}`" for s in self.romano_wolf_rejected[:10])
            more = len(self.romano_wolf_rejected) - 10
            tail = f" and {more} more" if more > 0 else ""
            lines += ["", f"Significant after Romano-Wolf: {shown}{tail}"]
        lines += [
            "",
            f"Bootstrap: stationary, mean block length {self.block_size:.1f}, "
            f"{self.n_boot} replications.",
        ]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def audit(
    returns: ArrayLike | pd.DataFrame,
    benchmark: ArrayLike | pd.Series | None = None,
    *,
    periods_per_year: float = 252.0,
    n_trials: float | None = None,
    criterion: Literal["sharpe", "mean"] = "sharpe",
    alpha: float = 0.05,
    n_boot: int = 2000,
    block_size: float | None = None,
    n_splits: int = 16,
    seed: SeedLike = 0,
) -> AuditReport:
    """Run every check in the library on a set of backtests.

    Parameters
    ----------
    returns
        Per-period returns of every strategy variant that was tried (one column each).
    benchmark
        Benchmark returns (e.g. buy-and-hold). ``None`` compares against zero.
    n_trials
        Number of trials actually run, if larger than the number of columns supplied.
    criterion
        Performance measure for selecting the best strategy and for the bootstrap tests.
    """
    x, labels = as_matrix(returns)
    n, k = x.shape
    rng = make_rng(seed)
    bench = None if benchmark is None else as_vector(benchmark, name="benchmark")
    excess = x if bench is None else x - bench[:, None]

    bp = bootstrap_performance(
        pd.DataFrame(x, columns=labels),
        bench,
        criterion=criterion,
        n_boot=n_boot,
        block_size=block_size,
        seed=rng,
    )
    best = int(np.argmax(bp.theta))
    stats = describe(excess[:, best], periods_per_year=periods_per_year)
    p_naive = float(special.ndtr(-stats.tstat_hac)) if np.isfinite(stats.tstat_hac) else 1.0

    n_tried = float(k if n_trials is None else max(n_trials, k))
    if k > 1:
        srs = np.asarray(sharpe_ratio(excess))
        sr_var = float(np.var(srs, ddof=1))
        n_eff = effective_number_of_trials(excess) * n_tried / k
    else:
        sr_var, n_eff = stats.se_nonnormal**2, n_tried
    sr0 = expected_max_sharpe(n_tried, sr_var)
    dsr, dsr_eff = (
        deflated_sharpe_ratio(
            stats.sr,
            n,
            sr_variance=sr_var,
            n_trials=trials,
            skew=stats.skew,
            kurtosis=stats.kurtosis,
        )
        for trials in (n_tried, max(n_eff, 1.0))
    )
    hc = haircut_sharpe_ratio(stats.sr, n, n_tried)

    rc = reality_check_from(bp)
    spa = spa_test_from(bp)
    rw = romano_wolf_from(bp, alpha=alpha)

    pbo_value = loss = None
    if k >= 2 and n >= 10 * n_splits:
        res = probability_of_backtest_overfitting(excess, n_splits=n_splits, metric="sharpe")
        pbo_value, loss = res.pbo, res.prob_oos_loss

    root = np.sqrt(periods_per_year)
    return AuditReport(
        n_obs=n,
        n_strategies=k,
        periods_per_year=periods_per_year,
        criterion=criterion,
        benchmark="zero" if bench is None else "supplied series",
        best=labels[best],
        sharpe=stats.sr * root,
        sharpe_se=stats.se_hac * root,
        pvalue_naive=p_naive,
        n_trials=n_tried,
        n_trials_effective=float(n_eff),
        expected_max_sharpe=sr0 * root,
        dsr=dsr,
        dsr_effective=dsr_eff,
        sharpe_haircut=hc.sr_haircut * root,
        haircut=hc.haircut,
        pvalue_reality_check=rc.pvalue,
        pvalue_spa=spa.pvalue,
        n_romano_wolf=rw.n_rejected,
        romano_wolf_rejected=tuple(rw.rejected),
        pbo=pbo_value,
        pbo_prob_oos_loss=loss,
        alpha=alpha,
        block_size=bp.block_size,
        n_boot=n_boot,
    )
