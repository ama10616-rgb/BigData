# Teammate Handoff — bigdata-trading

**For:** Daniela, Carol
**From:** Alamri
**Date:** 2026-04-20
**Status:** Pipeline complete end-to-end; all 9 phases executed successfully.

## What's built

- **Ingestion** (`src/universe.py`, `download_ohlcv.py`, `download_fred.py`): 503 S&P 500 tickers, ~1.94M rows of daily OHLCV (2010–present), plus 5 FRED macro series. All written to Parquet.
- **Feature engineering** (`src/features.py`): 27 features including log/simple returns, rolling vol (20/60/252d), SMAs (20/50/200d), momentum (63/126/252d), Wilder RSI, Bollinger bands, cross-sectional ranks. Implemented as Spark window functions.
- **Strategies** (`src/strategies.py`): Cross-sectional momentum + mean reversion.
- **Backtest engine** (`src/backtest.py`): Vectorized pandas, 5 bps turnover cost, no look-ahead bias.
- **Metrics** (`src/metrics.py`): Sharpe, Sortino, Max DD, Calmar, CAGR, and the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014).
- **Distributed parameter sweep** (`notebooks/04_parameter_sweep.ipynb`): 45 configs × 12 cores via Spark RDD, **3.36× speedup** vs. sequential.
- **Visualizations** (`notebooks/05_visualization.ipynb`): 7 Plotly charts exported to both interactive HTML and static PNG.
- **Final report** (`reports/final_report.md`): 2000 words across 8 sections, ready for submission.

## How to run it from scratch (≈ 10 min)

```bash
git clone <repo> && cd bigdata-trading
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.universe         # <10s  — S&P 500 scrape from Wikipedia
python -m src.download_ohlcv   # ~7min — 503 tickers via yfinance 1.3.0
python -m src.download_fred    # <30s  — 5 macro series from FRED

# Execute notebooks in order:
for nb in 00_smoke_test 01_eda 02_features 03_strategies_and_backtest \
          04_parameter_sweep 05_visualization; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
```

Everything after ingestion is deterministic — rerunning produces the same numbers.

## How we can divide the final polish

| Task | Owner | Notes |
|---|---|---|
| Proofread `reports/final_report.md` | Daniela | Flag anything that over-claims alpha; our DSR finding is the honest headline |
| Revise `notebooks/01_eda.ipynb` narrative | Carol | Add markdown commentary between charts — currently sparse |
| Deck slides (10 min, 8–10 slides) | Alamri | Use PNGs from `reports/figures/`; headline slide is the DSR finding |
| Test on teammate's machine | each | `pip install -r requirements.txt` → full pipeline run; report any diffs |
| Presentation Q&A prep | all | Expect questions on: why Spark here? why momentum? why DSR? survivorship? |

## Open questions / known issues

1. **yfinance version pinned to 1.3.0** — the original 0.2.50 pin in the proposal was broken against the current Yahoo endpoint (`YFTzMissingError` on every ticker). Upgrading fixed it. `requirements.txt` reflects this.
2. **Wikipedia scrape requires a User-Agent header** — `pandas.read_html` alone gets HTTP 403. `src/universe.py` uses a browser-identifying UA. Not fragile, but worth knowing if Wikipedia changes their bot policy.
3. **All DSR probabilities are 0.0** — this is correct, not a bug. The point is that our top 5 configurations are not statistically distinguishable from noise after multiple-testing correction. Lead with this finding rather than apologizing for it.
4. **Current best raw Sharpe is 0.27** — well below institutional thresholds (~1.0+). That's consistent with simple cross-sectional momentum on a current-SP500 universe net of 5 bps costs. Not a bug; not expected to be beating renaissance.
5. **Data covers 2010-01-04 → 2026-04-17** — an unusual stop date reflects the yfinance pull on 2026-04-20. Re-running updates everything through the call date.

## Key files to review before the presentation

1. `reports/final_report.md` — the writeup
2. `reports/figures/fig1_top5_equity.png` — headline chart
3. `reports/figures/fig5_sharpe_vs_turnover.png` — good "where strategies live" chart
4. `notebooks/04_parameter_sweep.ipynb` — the Spark showpiece; walk through the RDD call
