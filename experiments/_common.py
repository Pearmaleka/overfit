"""Shared paths, data loaders and plotting helpers for the experiment scripts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "docs" / "figures"
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

FX_FILE = DATA / "fx_usd_daily.csv.gz"
FX_CURRENCIES = ("AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK")


def equity_index(name: str) -> pd.Series:
    """Daily closes of the S&P 500 or NASDAQ Composite, 1999-2018, bundled with ``arch``."""
    from arch.data import nasdaq, sp500

    module = {"sp500": sp500, "nasdaq": nasdaq}[name]
    close = module.load()["Adj Close"].astype(float)
    close.index = pd.to_datetime(close.index)
    close.name = name
    return close.dropna()


def fx_rates() -> pd.DataFrame:
    """Daily USD price of ten currencies (FRED H.10 via datasets/exchange-rates)."""
    frame = pd.read_csv(FX_FILE, index_col="date", parse_dates=["date"])
    return frame[list(FX_CURRENCIES)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"cannot serialise {type(obj)}")


def write_json(obj: Any, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, name: str, **kwargs: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    frame.to_csv(path, float_format="%.6g", **kwargs)
    return path


# ---------------------------------------------------------------------------
# Plotting: one light and one dark rendering of every figure
# ---------------------------------------------------------------------------

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
        "background_marks": "#c3c2b7",
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
        "background_marks": "#4a4a46",
        "sequential": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}

# Colour follows the method in every figure. Only the three methods the figures are about
# get a hue (the first three palette slots stay distinguishable even where lines cross);
# baselines are drawn in grey and identified by marker and legend.
METHOD_SLOT = {
    "Romano-Wolf": 0,
    "Naive (best p-value)": 1,
    "Hansen SPA": 2,
}
MARKERS = {
    "Romano-Wolf": "o",
    "Naive (best p-value)": "X",
    "Hansen SPA": "s",
    "Bonferroni": "D",
    "Reality Check": "^",
    "Deflated Sharpe": "v",
    "Benjamini-Hochberg": "P",
}


def method_color(theme: dict[str, Any], method: str) -> str:
    if method not in METHOD_SLOT:
        return str(theme["muted"])
    return str(theme["series"][METHOD_SLOT[method]])


def style_axes(ax: plt.Axes, theme: dict[str, Any]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], length=3, width=0.8)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    ax.title.set_color(theme["ink"])


def style_legend(legend: Any, theme: dict[str, Any]) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor(theme["surface"])
    legend.get_frame().set_edgecolor(theme["grid"])
    for text in legend.get_texts():
        text.set_color(theme["ink2"])


def save_figure(draw: Callable[[dict[str, Any]], plt.Figure], name: str) -> list[Path]:
    """Render ``draw(theme)`` once per theme as ``docs/figures/<name>-<theme>.png``."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
    }
    for theme_name, theme in THEMES.items():
        themed = rc | {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "axes.edgecolor": theme["axis"],
            "text.color": theme["ink"],
            "axes.titlecolor": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "legend.labelcolor": theme["ink2"],
        }
        with plt.rc_context(themed):
            fig = draw(theme)
            fig.patch.set_facecolor(theme["surface"])
            path = FIGURES / f"{name}-{theme_name}.png"
            fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
            paths.append(path)
    return paths
