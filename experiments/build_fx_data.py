"""Rebuild ``data/fx_usd_daily.csv.gz`` from the pinned public source.

Source: Federal Reserve H.10 daily exchange rates as republished (ODC-PDDL licence) by
https://github.com/datasets/exchange-rates. The file there quotes units of foreign
currency per US dollar; we store the US dollar price of one unit of each currency, so
that a long position is long the foreign currency.

Run from the repository root:  python experiments/build_fx_data.py
"""

from __future__ import annotations

import hashlib
import io
import urllib.request

import pandas as pd
from _common import DATA, FX_CURRENCIES, FX_FILE

COMMIT = "2c93f64390781ffa082441909b6ea86a760bc214"
URL = f"https://raw.githubusercontent.com/datasets/exchange-rates/{COMMIT}/data/daily.csv"
COUNTRY_TO_CODE = {
    "Australia": "AUD",
    "Canada": "CAD",
    "Switzerland": "CHF",
    "Denmark": "DKK",
    "Euro": "EUR",
    "United Kingdom": "GBP",
    "Japan": "JPY",
    "Norway": "NOK",
    "New Zealand": "NZD",
    "Sweden": "SEK",
}


def main() -> None:
    with urllib.request.urlopen(URL, timeout=120) as response:
        raw = response.read()
    print(f"downloaded {len(raw):,} bytes, sha256 {hashlib.sha256(raw).hexdigest()}")
    long = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
    long = long[long["Country"].isin(COUNTRY_TO_CODE)]
    wide = long.pivot_table(index="Date", columns="Country", values="Exchange rate", aggfunc="last")
    wide = wide.rename(columns=COUNTRY_TO_CODE)[list(FX_CURRENCIES)]
    usd_price = 1.0 / wide.where(wide > 0)
    usd_price.index.name = "date"
    usd_price = usd_price.loc["1971-01-01":].dropna(how="all")
    DATA.mkdir(exist_ok=True)
    usd_price.to_csv(FX_FILE, float_format="%.8g", compression={"method": "gzip", "mtime": 0})
    print(
        f"wrote {FX_FILE.relative_to(DATA.parent)}: {usd_price.shape[0]} days, "
        f"{usd_price.index[0]:%Y-%m-%d} to {usd_price.index[-1]:%Y-%m-%d}"
    )
    print(usd_price.notna().sum().to_string())


if __name__ == "__main__":
    main()
