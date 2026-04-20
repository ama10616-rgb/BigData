"""Bulk-download daily OHLCV for the S&P 500 universe via yfinance.

Writes Parquet per ticker at data/parquet/ohlcv/ticker=<TKR>/data.parquet
Failures are appended to data/raw/failed_tickers.txt.
Idempotent: skips tickers whose Parquet already exists unless --force.
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OHLCV_DIR = REPO_ROOT / "data" / "parquet" / "ohlcv"
TICKERS_FILE = RAW_DIR / "sp500_tickers.txt"
FAILED_FILE = RAW_DIR / "failed_tickers.txt"

START_DATE = "2010-01-01"
BATCH_SIZE = 50
RETRY_WAIT_SEC = 60

SCHEMA_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def load_tickers() -> list[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(
            f"{TICKERS_FILE} missing. Run `python -m src.universe` first."
        )
    return [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]


def ticker_parquet_path(ticker: str) -> Path:
    return OHLCV_DIR / f"ticker={ticker}" / "data.parquet"


def tidy_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Convert a yfinance single-ticker frame into our schema. None if empty/bad."""
    if df is None or df.empty:
        return None
    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        return None
    df = df[SCHEMA_COLUMNS].dropna(subset=["close"])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df


def write_ticker(df: pd.DataFrame, ticker: str) -> int:
    path = ticker_parquet_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)


def download_batch(
    batch: list[str], start: str, end: str, attempt: int = 1
) -> pd.DataFrame | None:
    """Single attempt at a yf.download. Returns None on total failure."""
    try:
        data = yf.download(
            tickers=batch,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=True,
        )
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"[batch attempt {attempt}] download error: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download existing tickers")
    args = parser.parse_args()

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    end_date = date.today().isoformat()

    if args.force:
        todo = tickers
    else:
        todo = [t for t in tickers if not ticker_parquet_path(t).exists()]
    skipped = len(tickers) - len(todo)

    print(f"Total tickers: {len(tickers)} | already present: {skipped} | to download: {len(todo)}")
    print(f"Date range: {START_DATE} -> {end_date}")

    succeeded: list[str] = []
    failed: list[str] = []
    total_rows = 0

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}: {batch[0]}..{batch[-1]} ({len(batch)} tickers)")

        data = download_batch(batch, START_DATE, end_date, attempt=1)
        if data is None or data.empty:
            print(f"  batch empty on attempt 1; waiting {RETRY_WAIT_SEC}s and retrying once")
            time.sleep(RETRY_WAIT_SEC)
            data = download_batch(batch, START_DATE, end_date, attempt=2)

        if data is None or data.empty:
            print(f"  batch failed after retry; marking {len(batch)} tickers failed")
            failed.extend(batch)
            continue

        for t in batch:
            try:
                if len(batch) == 1:
                    tdf = data.copy()
                else:
                    if t not in data.columns.get_level_values(0):
                        failed.append(t)
                        continue
                    tdf = data[t].copy()
                tidy = tidy_single(tdf, t)
                if tidy is None or tidy.empty:
                    failed.append(t)
                    continue
                rows = write_ticker(tidy, t)
                total_rows += rows
                succeeded.append(t)
            except Exception as exc:  # noqa: BLE001
                print(f"  {t}: {exc}")
                failed.append(t)

    if failed:
        FAILED_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if FAILED_FILE.exists():
            existing = {l.strip() for l in FAILED_FILE.read_text().splitlines() if l.strip()}
        existing.update(failed)
        FAILED_FILE.write_text("\n".join(sorted(existing)) + "\n")

    print("\n==== SUMMARY ====")
    print(f"Succeeded : {len(succeeded)}")
    print(f"Failed    : {len(failed)}")
    print(f"Skipped   : {skipped}")
    print(f"Total new rows written: {total_rows:,}")
    print(f"Date range: {START_DATE} -> {end_date}")
    if failed:
        print(f"Failed tickers appended to {FAILED_FILE}")


if __name__ == "__main__":
    main()
