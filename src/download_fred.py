"""Pull macroeconomic series from FRED via pandas-datareader.

Series:
    VIXCLS   - CBOE VIX
    DGS10    - 10-Year Treasury constant-maturity yield
    FEDFUNDS - Effective federal funds rate
    CPIAUCSL - CPI, all urban consumers
    UNRATE   - Unemployment rate

Merged into a single daily DataFrame (forward-filled for monthly series),
written to data/parquet/fred_macro/data.parquet.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr

REPO_ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = REPO_ROOT / "data" / "parquet" / "fred_macro"
OUT_PATH = FRED_DIR / "data.parquet"

SERIES = ["VIXCLS", "DGS10", "FEDFUNDS", "CPIAUCSL", "UNRATE"]
START_DATE = "2010-01-01"
MAX_RETRIES = 3
BACKOFF_SEC = 10


def fetch_series(code: str, start: str, end: str) -> pd.Series:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            s = pdr.DataReader(code, "fred", start, end)[code]
            s.name = code
            return s
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"  {code} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC)
    raise RuntimeError(f"FRED fetch failed for {code}") from last_exc


def main() -> None:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    end_date = date.today().isoformat()
    print(f"Fetching FRED series {SERIES} from {START_DATE} to {end_date}")

    frames = []
    for code in SERIES:
        print(f"- {code}")
        frames.append(fetch_series(code, START_DATE, end_date))

    daily_idx = pd.date_range(start=START_DATE, end=end_date, freq="D")
    df = pd.concat(frames, axis=1).reindex(daily_idx).ffill()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.date
    # Drop leading rows where every macro column is still NaN (pre-first-obs gap).
    df = df.dropna(subset=SERIES, how="all").reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Rows: {len(df):,} | columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
