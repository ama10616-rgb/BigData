"""Scrape the current S&P 500 constituents from Wikipedia.

Writes:
    data/raw/sp500_tickers.txt   one ticker per line (yfinance convention)
    data/raw/sp500_sectors.csv   ticker,sector
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (bigdata-trading academic project; ama10616@nyu.edu)"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def fetch_sp500() -> pd.DataFrame:
    """Return DataFrame with columns: ticker, sector."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0][["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    )
    # yfinance uses '-' instead of '.' (BRK.B -> BRK-B, BF.B -> BF-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    df = df.drop_duplicates(subset="ticker").sort_values("ticker").reset_index(drop=True)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_sp500()

    tickers_path = RAW_DIR / "sp500_tickers.txt"
    sectors_path = RAW_DIR / "sp500_sectors.csv"

    tickers_path.write_text("\n".join(df["ticker"].tolist()) + "\n")
    df.to_csv(sectors_path, index=False)

    print(f"Wrote {len(df)} tickers to {tickers_path}")
    print(f"Wrote sector map to {sectors_path}")
    print(f"Sectors: {df['sector'].nunique()} unique")


if __name__ == "__main__":
    main()
