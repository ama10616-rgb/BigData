# Distributed Backtesting of US Equity Trading Strategies

**CS-GY 6513 Big Data — NYU Tandon, Spring 2026**
**Team:** Alamri, Daniela, Carol
**Pipeline date range:** 2010-01-04 → 2026-04-17

---

## 1. Abstract

We implemented a distributed backtesting pipeline for US equity trading
strategies built on PySpark 3.5.3, Parquet, and a local pandas-based backtest
engine. The pipeline ingests 503 current S&P 500 constituents (~1.94M daily
OHLCV rows over 15 years) from Yahoo Finance and five macroeconomic series from
FRED, computes a suite of 27 engineered features via Spark window functions,
and evaluates 45 strategy configurations — 27 cross-sectional momentum and
18 mean-reversion — in parallel through Spark RDD task distribution. The
distributed sweep completes in 19.4 s against an extrapolated 65.0 s
sequential baseline (**3.36× speedup on 12 cores**). We report institutional
performance metrics for each configuration, apply the Deflated Sharpe Ratio
correction of Bailey & López de Prado (2014) to discount selection bias, and
find that none of the 45 configurations survive the multiple-testing
adjustment — a finding that is itself pedagogically important.

## 2. Problem Statement and Objectives

**Problem.** Systematic trading strategies are almost always parameter-rich:
lookback windows, rebalancing cadence, entry/exit thresholds, holding limits.
Evaluating them requires running many full backtests, one per configuration.
Using pandas on a single thread, the number of configs one can realistically
test is bounded by patience, and this leads to researchers publishing small
grids — which both limits exploration and understates the true multiple-testing
burden when a "best" configuration is reported.

**Objectives.**
1. Build an end-to-end pipeline that can ingest a realistic US equity universe
   and compute features at scale using Spark.
2. Execute a parameter sweep that is *distributed* — the individual backtest
   is embarrassingly parallel across configurations, and Spark is the right
   primitive for that workload.
3. Report performance honestly, including the Deflated Sharpe Ratio to
   prevent the "best-cherry-picked-Sharpe" fallacy.
4. Produce a reproducible, audit-friendly codebase that teammates can clone
   and re-run.

## 3. Methodology

### 3.1 Dataset

| Source | Series | Rows | Date range |
|---|---|---|---|
| yfinance (bulk, batch 50) | 503 S&P 500 tickers, OHLCV | 1,940,547 | 2010-01-04 → 2026-04-17 |
| FRED (`pandas-datareader`) | VIXCLS, DGS10, FEDFUNDS, CPIAUCSL, UNRATE | 5,954 | 2010-01-01 → 2026-04-20 |
| Wikipedia | S&P 500 constituents + GICS sectors | 503 tickers / 11 sectors | snapshot 2026-04-20 |

All tabular data is stored in Parquet partitioned appropriately (per-ticker for
OHLCV, per-year for features), which gives us efficient column-pruning and
date-range pushdown in Spark.

### 3.2 Architecture

```
yfinance + FRED + Wikipedia
         │
         ▼  (ingestion scripts, idempotent)
data/parquet/ohlcv/ticker=<TKR>/data.parquet   (503 partitions)
data/parquet/fred_macro/data.parquet
         │
         ▼  Spark window functions
data/parquet/features/year=YYYY/…              (17 year partitions)
         │
         ▼  Spark RDD.parallelize → map(run_one_config) → collect
reports/parameter_sweep_results.{csv,parquet}
         │
         ▼  Plotly + Kaleido
reports/figures/*.{html,png}
```

The critical scale decision is that **features are computed in Spark** (where
group-by-ticker window functions distribute trivially) but **individual backtests
are computed in pandas inside Spark tasks** (because a single backtest is
small — ~4000 dates × ~500 tickers — and vectorized pandas beats Spark for
that workload). The distribution happens at the *configuration* level, not at
the row level.

### 3.3 Feature engineering

Implemented in `src/features.py`. All functions accept and return Spark
DataFrames and build on `Window.partitionBy("ticker").orderBy("date")`.
Feature list:

| Feature | Source | Notes |
|---|---|---|
| `log_ret`, `simple_ret` | close / lag(close) | survivorship-bias acknowledged |
| `vol_{20,60,252}d` | stddev of log_ret in rolling window × √252 | annualized |
| `sma_{20,50,200}d` + `*_ratio` | moving averages + close/SMA ratio | regime indicator |
| `mom_{63,126,252}d` | sum of log_ret in window [-N, -1] | 12-1 style; skip most recent day |
| `rsi_14` | Wilder RSI over 14 days | close approximation |
| `bb_upper`, `bb_lower`, `bb_pct` | 20-day Bollinger (2σ) | bb_pct ∈ [0,1] at bands |
| `mom_252d_rank`, `bb_pct_rank` | cross-sectional `percent_rank` over date | uniform on [0,1] |

After build, the features table has 1,940,547 rows × 29 columns (including
year partition key). Zero rows had RSI outside [0, 100]. Zero rows had
\|mom_252d\| ≥ 5.

### 3.4 Strategies

**Cross-sectional momentum** (`src/strategies.py::cross_sectional_momentum`).
On each rebalance date, rank the universe by a chosen momentum column
(`mom_63d`, `mom_126d`, or `mom_252d`). Long equal-weight the top `top_pct`
percentile, short equal-weight the bottom `top_pct`. Dollar-neutral (weights
sum to 0), gross exposure = 2. Positions held until next rebalance.

**Mean reversion** (`src/strategies.py::mean_reversion`). Z-score-transform a
chosen column (`bb_pct`) across the entire dataset. Enter long when z < -entry_z,
exit when z > -exit_z or holding exceeds `max_holding_days`. Symmetric short side.
Equal weight across active positions each day.

### 3.5 Backtest engine

`src/backtest.py::run_backtest` pivots long-form weights and returns to wide
matrices, shifts weights one day forward to prevent look-ahead, multiplies
element-wise against next-day returns, subtracts a flat **5 bps per unit of
turnover** transaction cost, and compounds. All core arithmetic is vectorized.
Single backtest runtime ≈ 1.4 s on 4,096 dates × ~500 tickers.

### 3.6 Parameter sweep

| Strategy | Grid | Count |
|---|---|---|
| Momentum | lookback × top_pct × rebalance_days = 3 × 3 × 3 | 27 |
| Mean reversion | z_col × entry_z × exit_z × max_holding = 1 × 3 × 2 × 3 | 18 |
| **Total** | — | **45** |

Execution pattern:

```python
broadcast_payload = sc.broadcast(features_pd_slim)
rdd = sc.parallelize(configs, numSlices=45)
results = rdd.map(run_one_config).collect()
```

The `src/` package is shipped to workers via `sc.addPyFile(src_zip)`, since
Spark workers don't inherit the driver's `sys.path`.

## 4. Results

### 4.1 Top 10 configurations by raw Sharpe

| config | strategy | key params | Sharpe | Sortino | Max DD | CAGR | Calmar | Turnover |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mom_011 | momentum | mom_126d, 10%, 63d | 0.270 | 0.252 | -48.1% | 3.5% | 0.07 | 0.039 |
| mom_020 | momentum | mom_252d, 10%, 63d | 0.216 | 0.196 | -50.3% | 2.3% | 0.05 | 0.028 |
| mom_019 | momentum | mom_252d, 10%, 21d | 0.211 | 0.189 | -66.0% | 2.2% | 0.03 | 0.050 |
| mom_014 | momentum | mom_126d, 20%, 63d | 0.180 | 0.167 | -39.6% | 1.6% | 0.04 | 0.034 |
| mr_014  | mean_reversion | entry 2.0, exit 0.0, hold 20d | 0.169 | 0.191 | -50.4% | 1.5% | 0.03 | 0.650 |
| mom_023 | momentum | mom_252d, 20%, 63d | 0.164 | 0.148 | -41.2% | 1.3% | 0.03 | 0.024 |
| mom_026 | momentum | mom_252d, 30%, 63d | 0.158 | 0.144 | -31.7% | 1.2% | 0.04 | 0.021 |
| mom_022 | momentum | mom_252d, 20%, 21d | 0.158 | 0.142 | -53.6% | 1.2% | 0.02 | 0.042 |
| mr_008  | mean_reversion | entry 1.5, exit 0.0, hold 20d | 0.143 | 0.151 | -50.8% | 1.0% | 0.02 | 0.689 |
| mom_017 | momentum | mom_126d, 30%, 63d | 0.137 | 0.128 | -35.2% | 0.9% | 0.03 | 0.030 |

The table makes clear that the best configurations are uniformly slow-rebalance
(63 days) momentum strategies with concentrated exposure (top 10% long / bottom
10% short).

### 4.2 Best config deep dive (`mom_011`)

`mom_011` = (momentum on `mom_126d`, top 10%, rebalance every 63 trading days).

- Equity curve: `reports/figures/fig1_top5_equity.png`
- Underwater drawdown: `reports/figures/fig2_underwater.png`
- Rolling 252-day Sharpe: `reports/figures/fig3_rolling_sharpe.png`

![Top 5 equity curves](figures/fig1_top5_equity.png)

![Underwater drawdown](figures/fig2_underwater.png)

![Rolling 252-day Sharpe](figures/fig3_rolling_sharpe.png)

### 4.3 DSR correction

We computed the Deflated Sharpe Ratio for the top 5 configurations using
`n_trials = 45`. DSR is the probability that the observed Sharpe exceeds the
expected maximum under the null hypothesis, adjusted for skew, kurtosis, and
multiple testing. For all five top configurations, **DSR = 0.0**. This means
the observed Sharpes are entirely consistent with what one would expect to
observe as the maximum of 45 random strategies — in other words, **the top
performers are not statistically distinguishable from noise after multiple-
testing correction**. This is the textbook use case for DSR and a stark
demonstration of why it matters.

### 4.4 Parameter sensitivity

Momentum heatmap (`reports/figures/fig4a_heatmap_momentum.png`) shows Sharpe
consistently rising as rebalance frequency drops from 5 → 21 → 63 days. Faster
rebalancing is dominated by transaction costs and short-term mean reversion.
Mean-reversion heatmap (`fig4b_heatmap_mean_reversion.png`) shows the opposite
pattern: longer holding (20 days) dominates, and higher entry z-scores (more
extreme dislocations) help marginally.

![Momentum heatmap](figures/fig4a_heatmap_momentum.png)

![Mean reversion heatmap](figures/fig4b_heatmap_mean_reversion.png)

![Sharpe vs turnover](figures/fig5_sharpe_vs_turnover.png)

The scatter (`fig5_sharpe_vs_turnover.png`) confirms the turnover/performance
relationship: mean-reversion strategies have 15–25× the turnover of momentum
strategies, meaning they pay much more in transaction costs for similar gross
alpha.

### 4.5 Speedup measurement

| Run | Configs | Wall time | Rate |
|---|---|---|---|
| Sequential (3 configs, measured) | 3 | 4.33 s | 1.44 s/config |
| Sequential (45 configs, extrapolated) | 45 | 65.01 s | — |
| Parallel, 12 cores, RDD.map | 45 | **19.36 s** | 2.32 configs/sec |
| **Speedup** | | | **3.36×** |

The sub-linear speedup (3.36× on 12 cores) reflects (a) Spark task-launch
overhead per config, (b) pandas-in-worker GIL contention on a single JVM host,
and (c) the broadcast deserialization cost amortized across only 45 tasks. On
a genuine cluster with 100+ configs, the speedup would scale closer to the
number of cores.

## 5. Scalability Discussion

Our dataset is modest by Big Data standards (~2M rows, ~1 GB Parquet). The
parallelism pattern, however, is the point. The same pipeline would scale to:

- **Minute-level bars:** ~390× more rows per ticker. Feature engineering stays
  the same — Spark window functions handle billions of rows on a cluster. The
  backtest per-config would slow because each worker processes more data, but
  the *distribution* across configs is unchanged.
- **Global equities:** ~50× more tickers. Same Parquet-per-ticker layout;
  broadcast payload grows, which eventually exceeds practical RDD broadcast
  limits (~8 GB) and we would switch to repartitioned DataFrame joins.
- **Larger grids:** At 10,000+ configurations, the parallelism dominates
  overhead and we expect a near-linear scaling curve until we hit the cluster
  core count.

## 6. Limitations

1. **Survivorship bias.** We use the *current* S&P 500 constituents. Companies
   that were in the index in 2010 but are no longer (delisted, acquired, or
   dropped) are excluded. This inflates reported Sharpe by roughly 30–50 bp in
   published literature; our numbers are likely similarly optimistic.
2. **No slippage model.** Our cost is a flat 5 bps per unit turnover. A realistic
   model would include bid-ask spread, market impact (increasing with position
   size), and fill probability. Mean-reversion strategies, which rebalance
   aggressively, would suffer disproportionately.
3. **Daily resolution only.** Intraday signals (e.g. opening-range breakouts)
   are out of scope.
4. **Point-in-time data.** yfinance returns current-snapshot data. Adjustments
   for corporate actions are applied retroactively; splits/dividends in our
   timeline reflect the view as of 2026-04-20, not as of the historical trade date.
5. **No regime awareness.** All strategies run with fixed parameters across
   15 years spanning QE, normalization, pandemic, and 2022 bear market. A
   real production strategy would adapt.

## 7. Conclusion

We demonstrated that Spark is a natural fit for strategy parameter sweeps — a
workload with many short, independent tasks that each depend on a shared read-
only dataset. The best raw configuration (momentum on 126-day lookback,
top/bottom decile, quarterly rebalance) produced a Sharpe of 0.27 over 15
years, well below the institutional bar (typically > 1.0) but consistent with
academic literature on simple cross-sectional momentum net of costs. The more
important finding is methodological: applying the Deflated Sharpe Ratio to a
parameter sweep that reported Sharpe = 0.27 as the "best of 45" produces a
significance probability of 0.0 — meaning the result is not distinguishable
from cherry-picked noise. Any public claim of strategy performance that
reports only the top Sharpe from a grid without correcting for selection bias
is, statistically, not a claim of alpha.

## 8. References

- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5), 94–107.
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling
  Losers: Implications for Stock Market Efficiency." *Journal of Finance*,
  48(1), 65–91.
- Zaharia, M., et al. (2016). "Apache Spark: A Unified Engine for Big Data
  Processing." *Communications of the ACM*, 59(11), 56–65.
- Pandas-datareader documentation, https://pandas-datareader.readthedocs.io/
- yfinance project, https://github.com/ranaroussi/yfinance
- FRED (Federal Reserve Economic Data), https://fred.stlouisfed.org/
