from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from overfit import audit
from overfit.cli import _demo_returns, main


@pytest.fixture
def noise() -> pd.DataFrame:
    return _demo_returns(n_strategies=60, years=4, seed=3)


def test_audit_flags_selection_luck(noise):
    report = audit(noise, n_boot=499)
    assert report.n_strategies == 60
    assert report.sharpe > report.sharpe_haircut
    assert report.dsr < 0.95
    assert report.pvalue_spa > 0.05
    assert report.n_romano_wolf == 0
    assert "Evidence of skill survives" not in report.verdict
    assert 0.0 <= report.pbo <= 1.0


def test_audit_recognises_a_real_edge(noise):
    skilled = noise.copy()
    skilled.iloc[:, 5] += 0.0025
    report = audit(skilled, n_boot=499)
    assert report.best == skilled.columns[5]
    assert report.dsr_effective > 0.95
    assert report.pvalue_spa < 0.05
    assert skilled.columns[5] in report.romano_wolf_rejected
    assert report.verdict.startswith("Evidence of skill")


def test_report_rendering(noise):
    report = audit(noise.iloc[:, :10], noise.iloc[:, 10], n_boot=199)
    text = report.to_markdown()
    assert "| Deflated Sharpe ratio" in text and "**Verdict:**" in text
    assert "excess of the benchmark" in text
    payload = json.loads(json.dumps(report.to_dict(), default=float))
    assert payload["n_strategies"] == 10


def test_cli_audit_roundtrip(tmp_path, capsys, noise):
    path = tmp_path / "returns.csv"
    frame = noise.iloc[:, :8].copy()
    frame["bench"] = noise.iloc[:, 9]
    frame.to_csv(path)
    assert main(["audit", str(path), "--benchmark", "bench", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    assert "# Backtest audit" in out and "8 strategies" in out

    out_file = tmp_path / "report.json"
    main(["audit", str(path), "--n-boot", "199", "--format", "json", "-o", str(out_file)])
    assert json.loads(out_file.read_text())["n_strategies"] == 9


def test_cli_prices_and_demo(tmp_path, capsys):
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, (400, 3)), axis=0)),
        index=pd.bdate_range("2021-01-01", periods=400),
        columns=["a", "b", "c"],
    )
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    assert main(["audit", str(path), "--prices", "--n-boot", "99"]) == 0
    assert "399 observations" in capsys.readouterr().out
    assert main(["demo", "--strategies", "30", "--years", "2"]) == 0
    assert "Backtest audit" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["audit", str(path), "--benchmark", "missing"])
