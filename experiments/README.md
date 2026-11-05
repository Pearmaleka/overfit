# Experiments

Each script writes its numbers to `experiments/results/` and its figures (a light and a
dark rendering) to `docs/figures/`. Run them from the repository root after
`pip install -e ".[experiments]"`. `--quick` runs a reduced version in about a minute
and writes `*-quick.*` files instead.

| Script | Question | Data | Runtime* |
|---|---|---|---|
| `calibration.py` | Do the tests keep their size and how much power do they have? | simulated | ~13 min |
| `equity_rules.py` | Does the best of 2,902 technical rules beat buy-and-hold? | S&P 500, NASDAQ, 1999-2018 | ~1 min |
| `fx_rules.py` | Did currency trading rules lose their edge? | 10 currencies, 1973-2026 | ~3 min |
| `cv_leakage.py` | How much does k-fold CV overstate skill with overlapping labels? | simulated, S&P 500 | ~9 min |
| `build_fx_data.py` | Rebuild `data/fx_usd_daily.csv.gz` from the pinned source | FRED H.10 | seconds |
| `tables.py` | Fill the tables in `docs/results.md` from `experiments/results/` | | seconds |

\* On two CPU cores. `--plot-only` redraws a script's figures from its saved results;
`tables.py` refreshes the tables in `docs/results.md`.

All randomness is seeded, so rerunning a script with the same library versions reproduces
its files. The results
are discussed in [`docs/results.md`](../docs/results.md).
