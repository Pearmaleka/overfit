# Contributing

Bug reports, questions and pull requests are welcome. For anything larger than a small
fix, please open an issue first so the approach can be discussed before you write code.

## Setup

```bash
git clone https://github.com/Pearmaleka/overfit && cd overfit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # runs ruff and basic file checks on every commit
```

## Checks

Everything below runs in CI on every pull request.

```bash
pytest -m "not slow"        # unit and property tests, a few seconds
pytest -m slow              # Monte Carlo calibration tests
ruff check . && ruff format --check .
mypy src
python experiments/tables.py --check   # tables in docs/results.md match the result files
```

## Conventions

- **New methods need a reference and a test that pins them to it.** Reproduce a worked
  example from the paper, or compare against an established implementation such as
  `statsmodels` or `arch`. A Monte Carlo test of size or coverage is a good addition.
- **No look-ahead.** New trading rules must pass the test in `tests/test_rules.py` that
  changes future prices and checks that past positions do not move.
- **Numbers in the docs come from files.** If a change affects results, re-run the
  experiment script, commit the files it writes to `experiments/results/` and
  `docs/figures/`, and refresh the tables with `python experiments/tables.py`. Do not
  edit numbers in `docs/results.md` by hand.
- **Sharpe ratios are per period** inside the library unless a name or argument says
  otherwise (`periods_per_year`, `sr_annual`).
- Keep the core dependencies to NumPy, SciPy and pandas. Anything else belongs in the
  `experiments` or `dev` extras.

## Pull requests

Keep each pull request to one change, link the issue it resolves, and describe how you
checked it. Squash merges are used, so the pull request title becomes the commit message
on `main`.
