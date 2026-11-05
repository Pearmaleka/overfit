"""A universe of classic technical trading rules, computed without look-ahead.

``positions[t]`` is decided with prices up to and including ``t`` and earns the return
from ``t`` to ``t + 1``. Long/flat by default; ``long_short=True`` maps flat to short.
Families (parameter grids follow Sullivan, Timmermann and White, 1999, and later work):

``ma``        moving-average crossover, with a percentage band or a delay filter
``momentum``  time-series momentum (return over the last ``n`` periods) with a band
``breakout``  support/resistance breakout: enter above the ``n``-period high, exit below
              the ``m``-period low
``filter``    Alexander filter: enter after an ``x`` rise from the trough, exit after a
              ``y`` fall from the peak
``rsi_rev``   RSI mean reversion: enter when oversold, exit when overbought
``rsi_trend`` RSI trend following: enter when RSI is high, exit when it drops
``bb_rev``    Bollinger-band mean reversion
``bb_break``  Bollinger-band breakout

The default universe has 2,902 rules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from overfit._utils import FloatArray

BoolArray = NDArray[np.bool_]
PosArray = NDArray[np.int8]
Specs = list[dict[str, Any]]

FAMILIES: tuple[str, ...] = (
    "ma",
    "momentum",
    "breakout",
    "filter",
    "rsi_rev",
    "rsi_trend",
    "bb_rev",
    "bb_break",
)

# ---------------------------------------------------------------------------
# Indicators (all causal: the value at t uses prices up to t only)
# ---------------------------------------------------------------------------


def _series(prices: ArrayLike | pd.Series) -> pd.Series:
    p = pd.Series(np.asarray(prices, dtype=float))
    if p.isna().any() or (p <= 0).any():
        raise ValueError("prices must be positive and free of NaNs")
    return p


def sma(prices: ArrayLike | pd.Series, window: int) -> FloatArray:
    """Simple moving average including the current price (NaN during warm-up)."""
    return _series(prices).rolling(window).mean().to_numpy()


def rsi(prices: ArrayLike | pd.Series, window: int) -> FloatArray:
    """Wilder's relative strength index (exponential smoothing with ``alpha = 1/window``)."""
    p = _series(prices)
    delta = p.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    g, lo = gain.to_numpy(), loss.to_numpy()
    out = np.full(g.shape, np.nan)
    ok = np.isfinite(g) & np.isfinite(lo)
    total = g + lo
    flat = ok & (total == 0)
    live = ok & (total > 0)
    out[live] = 100.0 * g[live] / total[live]
    out[flat] = 50.0
    return out


def bollinger_z(prices: ArrayLike | pd.Series, window: int) -> FloatArray:
    """Distance of the price from its moving average in rolling standard deviations."""
    p = _series(prices)
    mean = p.rolling(window).mean()
    sd = p.rolling(window).std(ddof=0)
    z = (p - mean) / sd.where(sd > 0)
    return z.to_numpy()


def _rolling_extreme(prices: pd.Series, window: int, kind: str) -> FloatArray:
    """Max or min over the ``window`` periods *before* t (excludes the current price)."""
    roll = prices.rolling(window)
    ext = roll.max() if kind == "max" else roll.min()
    return ext.shift(1).to_numpy()


# ---------------------------------------------------------------------------
# Signal plumbing
# ---------------------------------------------------------------------------


def _hysteresis(enter: BoolArray, exit_: BoolArray) -> PosArray:
    """Go long on ``enter``, go flat on ``exit_``, otherwise keep the previous position."""
    code = np.full(enter.shape, -1, dtype=np.int8)
    code[exit_] = 0
    code[enter] = 1
    t = np.arange(enter.shape[0], dtype=np.int32)[:, None]
    last = np.where(code >= 0, t, np.int32(-1)).astype(np.int32)
    np.maximum.accumulate(last, axis=0, out=last)
    pos = np.take_along_axis(code, np.maximum(last, 0), axis=0)
    return np.where(last >= 0, pos, 0).astype(np.int8)


def _persist(cond: BoolArray, days: int) -> BoolArray:
    """True where ``cond`` has held for ``days`` consecutive periods up to and including t."""
    if days <= 1:
        return cond
    csum = np.cumsum(cond, axis=0, dtype=np.int32)
    lagged = np.zeros_like(csum)
    lagged[days:] = csum[:-days]
    return np.asarray((csum - lagged) >= days)


def _compare(a: FloatArray, op: str, b: FloatArray | float) -> BoolArray:
    with np.errstate(invalid="ignore"):
        return np.asarray(a > b if op == ">" else a < b)


# ---------------------------------------------------------------------------
# Rule families. Each returns positions (n_obs, n_rules) and one spec per rule.
# ---------------------------------------------------------------------------


def ma_rules(
    prices: ArrayLike | pd.Series,
    *,
    shorts: Sequence[int] = (1, 2, 5, 10, 15, 20, 25, 30, 50),
    longs: Sequence[int] = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200, 250),
    bands: Sequence[float] = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05),
    delays: Sequence[int] = (2, 5),
) -> tuple[PosArray, Specs]:
    """Moving-average crossover: long while the short MA is above the long MA.

    With band ``b`` the rule enters when ``MA_s > (1 + b) MA_l`` and exits when
    ``MA_s < (1 - b) MA_l``. Delay variants (no band) act only after the crossover has
    held for ``d`` consecutive periods. A short window of 1 is the price itself.
    """
    pairs = [(s, lg) for s in shorts for lg in longs if s < lg]
    ratio = np.column_stack([sma(prices, s) / sma(prices, lg) for s, lg in pairs])
    blocks: list[PosArray] = []
    specs: Specs = []
    for b in bands:
        blocks.append(_hysteresis(_compare(ratio, ">", 1.0 + b), _compare(ratio, "<", 1.0 - b)))
        specs += [
            {
                "name": f"ma:s={s},l={lg},b={b:g},d=0",
                "family": "ma",
                "short": s,
                "long": lg,
                "band": b,
                "delay": 0,
            }
            for s, lg in pairs
        ]
    up, down = _compare(ratio, ">", 1.0), _compare(ratio, "<", 1.0)
    for d in delays:
        blocks.append(_hysteresis(_persist(up, d), _persist(down, d)))
        specs += [
            {
                "name": f"ma:s={s},l={lg},b=0,d={d}",
                "family": "ma",
                "short": s,
                "long": lg,
                "band": 0.0,
                "delay": d,
            }
            for s, lg in pairs
        ]
    return np.hstack(blocks), specs


def momentum_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (5, 10, 20, 40, 60, 90, 120, 150, 200, 250),
    bands: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1),
) -> tuple[PosArray, Specs]:
    """Time-series momentum: enter when the ``n``-period return exceeds ``b``, exit below ``-b``."""
    p = _series(prices)
    mom = np.column_stack([(p / p.shift(n) - 1.0).to_numpy() for n in windows])
    blocks, specs = [], []
    for b in bands:
        blocks.append(_hysteresis(_compare(mom, ">", b), _compare(mom, "<", -b)))
        specs += [
            {"name": f"mom:n={n},b={b:g}", "family": "momentum", "window": n, "band": b}
            for n in windows
        ]
    return np.hstack(blocks), specs


def breakout_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (5, 10, 15, 20, 25, 50, 100, 150, 200, 250),
    exit_fractions: Sequence[float] = (1.0, 0.5, 0.25),
    bands: Sequence[float] = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05),
) -> tuple[PosArray, Specs]:
    """Support/resistance breakout.

    Enter when the price exceeds the highest close of the previous ``n`` periods by a
    fraction ``b``; exit when it falls ``b`` below the lowest close of the previous
    ``m = max(2, round(f * n))`` periods.
    """
    p = _series(prices)
    pairs: list[tuple[int, int]] = []
    for n in windows:
        for f in exit_fractions:
            pair = (n, max(2, round(f * n)))
            if pair not in pairs:
                pairs.append(pair)
    hi = np.column_stack([_rolling_extreme(p, n, "max") for n, _ in pairs])
    lo = np.column_stack([_rolling_extreme(p, m, "min") for _, m in pairs])
    price = p.to_numpy()[:, None]
    blocks, specs = [], []
    for b in bands:
        blocks.append(
            _hysteresis(_compare(price, ">", hi * (1.0 + b)), _compare(price, "<", lo * (1.0 - b)))
        )
        specs += [
            {
                "name": f"brk:n={n},m={m},b={b:g}",
                "family": "breakout",
                "window": n,
                "exit_window": m,
                "band": b,
            }
            for n, m in pairs
        ]
    return np.hstack(blocks), specs


def filter_rules(
    prices: ArrayLike | pd.Series,
    *,
    entries: Sequence[float] = (
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
    ),
    exits: Sequence[float] = (
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
    ),
) -> tuple[PosArray, Specs]:
    """Alexander filter rules (path dependent).

    While flat, track the lowest price since going flat and enter once the price is
    ``x`` above it; while long, track the highest price since entry and exit once the
    price is ``y`` below it.
    """
    price = _series(prices).to_numpy()
    grid = [(x, y) for x in entries for y in exits]
    x = np.array([g[0] for g in grid])
    y = np.array([g[1] for g in grid])
    long_ = np.zeros(len(grid), dtype=bool)
    trough = np.full(len(grid), price[0])
    peak = np.full(len(grid), price[0])
    pos = np.zeros((price.shape[0], len(grid)), dtype=np.int8)
    for t in range(1, price.shape[0]):
        pt = price[t]
        trough = np.where(long_, trough, np.minimum(trough, pt))
        peak = np.where(long_, np.maximum(peak, pt), peak)
        enter = ~long_ & (pt >= trough * (1.0 + x))
        leave = long_ & (pt <= peak * (1.0 - y))
        long_ = (long_ | enter) & ~leave
        peak = np.where(enter, pt, peak)
        trough = np.where(leave, pt, trough)
        pos[t] = long_
    specs = [
        {"name": f"flt:x={xx:g},y={yy:g}", "family": "filter", "entry": xx, "exit": yy}
        for xx, yy in grid
    ]
    return pos, specs


def rsi_reversal_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (2, 3, 5, 7, 10, 14, 21, 28),
    lowers: Sequence[float] = (5, 10, 15, 20, 25, 30, 35, 40, 45),
    uppers: Sequence[float] = (50, 55, 60, 65, 70, 75, 80, 85, 90),
) -> tuple[PosArray, Specs]:
    """Enter when RSI falls below ``lower`` (oversold), exit when it rises above ``upper``."""
    blocks, specs = [], []
    for n in windows:
        r = rsi(prices, n)[:, None]
        grid = [(lo, hi) for lo in lowers for hi in uppers if lo < hi]
        lo_arr = np.array([g[0] for g in grid], dtype=float)
        hi_arr = np.array([g[1] for g in grid], dtype=float)
        blocks.append(_hysteresis(_compare(r, "<", lo_arr), _compare(r, ">", hi_arr)))
        specs += [
            {
                "name": f"rsirev:n={n},lo={lo:g},hi={hi:g}",
                "family": "rsi_rev",
                "window": n,
                "lower": lo,
                "upper": hi,
            }
            for lo, hi in grid
        ]
    return np.hstack(blocks), specs


def rsi_trend_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (5, 7, 10, 14, 21, 28),
    uppers: Sequence[float] = (50, 55, 60, 65, 70, 75, 80),
    lowers: Sequence[float] = (20, 25, 30, 35, 40, 45, 50),
) -> tuple[PosArray, Specs]:
    """Enter when RSI rises above ``upper``, exit when it falls below ``lower``."""
    blocks, specs = [], []
    for n in windows:
        r = rsi(prices, n)[:, None]
        grid = [(hi, lo) for hi in uppers for lo in lowers if lo < hi]
        hi_arr = np.array([g[0] for g in grid], dtype=float)
        lo_arr = np.array([g[1] for g in grid], dtype=float)
        blocks.append(_hysteresis(_compare(r, ">", hi_arr), _compare(r, "<", lo_arr)))
        specs += [
            {
                "name": f"rsitr:n={n},hi={hi:g},lo={lo:g}",
                "family": "rsi_trend",
                "window": n,
                "upper": hi,
                "lower": lo,
            }
            for hi, lo in grid
        ]
    return np.hstack(blocks), specs


def bollinger_reversion_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (10, 20, 30, 50, 75, 100),
    ks: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    exits: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> tuple[PosArray, Specs]:
    """Enter when the price is ``k`` standard deviations below its mean, exit above ``e``."""
    blocks, specs = [], []
    for n in windows:
        z = bollinger_z(prices, n)[:, None]
        grid = [(k, e) for k in ks for e in exits if e > -k]
        k_arr = np.array([g[0] for g in grid])
        e_arr = np.array([g[1] for g in grid])
        blocks.append(_hysteresis(_compare(z, "<", -k_arr), _compare(z, ">", e_arr)))
        specs += [
            {
                "name": f"bbrev:n={n},k={k:g},e={e:g}",
                "family": "bb_rev",
                "window": n,
                "k": k,
                "k_exit": e,
            }
            for k, e in grid
        ]
    return np.hstack(blocks), specs


def bollinger_breakout_rules(
    prices: ArrayLike | pd.Series,
    *,
    windows: Sequence[int] = (10, 20, 30, 50, 75, 100),
    ks: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    exits: Sequence[float] = (-0.5, 0.0, 0.5, 1.0),
) -> tuple[PosArray, Specs]:
    """Enter when the price is ``k`` standard deviations above its mean, exit below ``e``."""
    blocks, specs = [], []
    for n in windows:
        z = bollinger_z(prices, n)[:, None]
        grid = [(k, e) for k in ks for e in exits if e < k]
        k_arr = np.array([g[0] for g in grid])
        e_arr = np.array([g[1] for g in grid])
        blocks.append(_hysteresis(_compare(z, ">", k_arr), _compare(z, "<", e_arr)))
        specs += [
            {
                "name": f"bbbrk:n={n},k={k:g},e={e:g}",
                "family": "bb_break",
                "window": n,
                "k": k,
                "k_exit": e,
            }
            for k, e in grid
        ]
    return np.hstack(blocks), specs


_BUILDERS: dict[str, Callable[..., tuple[PosArray, Specs]]] = {
    "ma": ma_rules,
    "momentum": momentum_rules,
    "breakout": breakout_rules,
    "filter": filter_rules,
    "rsi_rev": rsi_reversal_rules,
    "rsi_trend": rsi_trend_rules,
    "bb_rev": bollinger_reversion_rules,
    "bb_break": bollinger_breakout_rules,
}

# ---------------------------------------------------------------------------
# Rule universe
# ---------------------------------------------------------------------------


def simple_returns(prices: ArrayLike | pd.Series) -> FloatArray:
    """``P_t / P_{t-1} - 1`` with a 0 in the first position."""
    p = _series(prices).to_numpy()
    out = np.zeros_like(p)
    out[1:] = p[1:] / p[:-1] - 1.0
    return out


def rule_returns(
    positions: NDArray[Any], asset_returns: ArrayLike, *, cost: float = 0.0
) -> FloatArray:
    """Returns of each rule: ``pos[t-1] * r[t] - cost * |pos[t-1] - pos[t-2]|``.

    ``cost`` is charged per unit of position change (0.0005 = 5 basis points one way),
    in the first period the new position is held.
    """
    pos = np.asarray(positions)
    r = np.asarray(asset_returns, dtype=float)
    if pos.ndim == 1:
        pos = pos[:, None]
    if r.shape[0] != pos.shape[0]:
        raise ValueError("positions and asset_returns must have the same length")
    out = np.zeros(pos.shape, dtype=float)
    out[1:] = pos[:-1] * r[1:, None]
    if cost:
        turnover = np.abs(np.diff(pos.astype(np.int16), axis=0, prepend=0))
        out[1:] -= cost * turnover[:-1]
    return out


@dataclass(frozen=True)
class RuleSet:
    """Positions of every rule on one price series."""

    positions: PosArray
    specs: pd.DataFrame
    prices: pd.Series
    warmup: int
    long_short: bool

    @property
    def names(self) -> list[str]:
        return list(self.specs.index)

    @property
    def n_rules(self) -> int:
        return int(self.positions.shape[1])

    def asset_returns(self) -> pd.Series:
        """Buy-and-hold returns over the evaluation window (after the warm-up)."""
        r = simple_returns(self.prices)
        index = self.prices.index[self.warmup + 1 :]
        return pd.Series(r[self.warmup + 1 :], index=index, name="asset")

    def returns(self, *, cost: float = 0.0) -> pd.DataFrame:
        """Rule returns over the evaluation window, one column per rule."""
        r = rule_returns(self.positions, simple_returns(self.prices), cost=cost)
        return pd.DataFrame(
            r[self.warmup + 1 :], index=self.prices.index[self.warmup + 1 :], columns=self.names
        )


def build_rules(
    prices: pd.Series | ArrayLike,
    *,
    families: Sequence[str] = FAMILIES,
    long_short: bool = False,
    warmup: int = 260,
) -> RuleSet:
    """Positions of the default rule universe on ``prices``.

    Positions are forced to zero during the first ``warmup`` periods so every rule is
    evaluated over the same window. With ``long_short=True`` flat becomes short.
    """
    series = prices if isinstance(prices, pd.Series) else pd.Series(np.asarray(prices, dtype=float))
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; choose from {FAMILIES}")
    if not 0 <= warmup < len(series) - 1:
        raise ValueError("warmup must leave at least two observations")
    blocks, specs = [], []
    for fam in families:
        pos, spec = _BUILDERS[fam](series.to_numpy())
        blocks.append(pos)
        specs += spec
    positions = np.hstack(blocks)
    if long_short:
        positions = (2 * positions - 1).astype(np.int8)
    positions[:warmup] = 0
    table = pd.DataFrame(specs).set_index("name")
    if not table.index.is_unique:
        raise RuntimeError("rule names are not unique")
    return RuleSet(
        positions=positions,
        specs=table,
        prices=series.astype(float),
        warmup=warmup,
        long_short=long_short,
    )
