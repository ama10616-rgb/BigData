# Distributed Backtesting of US Equity Trading Strategies

A PySpark-based pipeline for ingesting 15 years of S&P 500 OHLCV data, engineering financial features, and running a distributed parameter sweep over two trading strategies (cross-sectional momentum, mean reversion) with Deflated Sharpe Ratio correction.

Built for **CS-GY 6513 Big Data, NYU Tandon, Spring 2026**.

## Team
- Alamri
- Daniela
- Carol

## Tech stack
- **Python** 3.10.14 (venv)
- **PySpark** 3.5.3 (local mode, 12-core laptop)
- **Parquet** via pyarrow 17.0.0 (per-ticker OHLCV, per-year features)
- **yfinance** 1.3.0, **pandas-datareader** 0.10.0 for ingestion
- **Plotly** 5.24.1 + **Kaleido** 0.2.1 for interactive + static figures
- **Java** 17 (PySpark requirement)

## Quickstart

```bash
git clone <repo>
cd bigdata-trading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Pull data (15 minutes total)
python -m src.universe           # ~10s, scrapes S&P 500 constituents
python -m src.download_ohlcv     # ~15min, 503 tickers × 15 years via yfinance
python -m src.download_fred      # ~30s, 5 macro series from FRED

# 2. Execute notebooks in order
jupyter nbconvert --to notebook --execute --inplace notebooks/00_smoke_test.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_strategies_and_backtest.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_parameter_sweep.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_visualization.ipynb

# 3. Or launch Jupyter and run interactively
jupyter notebook
```

## Notebook execution order

| # | Notebook | Depends on | Produces |
|---|---|---|---|
| 00 | smoke test | OHLCV + FRED parquet | prints schema + counts |
| 01 | EDA | OHLCV + FRED parquet | 6 Plotly sections |
| 02 | features | OHLCV parquet | `data/parquet/features/` |
| 03 | single backtest | features parquet | metrics table for 1 mom + 1 MR config |
| 04 | parameter sweep (Spark) | features parquet | `reports/parameter_sweep_results.{csv,parquet}` + `dsr_top5.csv` |
| 05 | visualization | sweep results | 7 PNG + 7 HTML in `reports/figures/` |

## Data provenance

- **Equities:** yfinance (Yahoo Finance), batch download in chunks of 50. Known caveats: survivorship bias (current SP500 constituents only; delisted or dropped names absent), and point-in-time split/dividend adjustments reflect current snapshot, not historical.
- **Macroeconomic series:** FRED via `pandas-datareader` — VIXCLS, DGS10, FEDFUNDS, CPIAUCSL, UNRATE.
- **Index constituents:** Wikipedia page *List of S&P 500 companies*, scraped with a browser User-Agent (pandas read_html via raw HTTP request).

## Repo structure

```
bigdata-trading/
├── src/
│   ├── spark_session.py          SparkSession builder (local[*], 4g driver)
│   ├── universe.py               S&P 500 constituents scrape
│   ├── download_ohlcv.py         yfinance bulk pull, per-ticker Parquet
│   ├── download_fred.py          FRED macro series
│   ├── features.py               Spark window-function feature engineering
│   ├── strategies.py             Signal generators (pandas, in-task)
│   ├── backtest.py               Vectorized backtest engine (5bps cost)
│   └── metrics.py                Sharpe, Sortino, MaxDD, Calmar, CAGR, DSR
├── notebooks/                    00–05 as above
├── tests/
│   └── test_metrics.py           9 pytest sanity tests
├── reports/
│   ├── final_report.md           8-section writeup
│   ├── parameter_sweep_results.{csv,parquet}
│   ├── dsr_top5.csv
│   ├── teammate_handoff.md
│   └── figures/                  7 Plotly charts, HTML + PNG
├── data/
│   ├── raw/
│   │   ├── sp500_tickers.txt     tracked — reproducibility anchor
│   │   └── sp500_sectors.csv     tracked
│   └── parquet/                  gitignored, regenerated from scripts
├── requirements.txt              pinned dependency list
└── README.md                     this file
```

## License

MIT
