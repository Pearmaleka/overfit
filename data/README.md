# Data

## `fx_usd_daily.csv.gz`

Daily US dollar price of one unit of ten currencies: AUD, CAD, CHF, DKK, EUR, GBP, JPY,
NOK, NZD and SEK. 14,006 business days from 1971-01-04 to 2026-10-30; EUR starts on
1999-01-04. Missing days (holidays) are left empty.

- **Source:** Board of Governors of the Federal Reserve System, H.10 release (noon buying
  rates in New York), as republished by
  [datasets/exchange-rates](https://github.com/datasets/exchange-rates) under the Open
  Data Commons Public Domain Dedication and Licence, commit `2c93f64`.
- **Transformation:** the source quotes units of foreign currency per dollar; the file
  stores the reciprocal, so a long position in a column is long that currency.
- **Rebuild:** `python experiments/build_fx_data.py`. The script pins the source commit
  and prints the SHA-256 of the download
  (`013e19fbdcae7d2317bd01838974bdaa71d1db1020751940c6ec673e8e085641`).

## Equity indices

Daily closes of the S&P 500 and the NASDAQ Composite from 1999-01-04 to 2018-12-31
(5,031 days each) ship with the [`arch`](https://github.com/bashtage/arch) package
(`arch.data.sp500`, `arch.data.nasdaq`) and are loaded from there rather than copied.
