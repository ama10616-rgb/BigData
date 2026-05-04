Distributed Backtesting of US Equity Trading Strategies
CS-GY 6513 Big Data | NYU Tandon School of Engineering | Spring 2026
Authors: Daniela Cruz, Abdullah Alamri, Carol Zipphora David

1. Abstract
We implemented a distributed backtesting pipeline for US equity trading strategies on Apache PySpark 3.5.3, ingesting 503 current S&P 500 constituents (1.94M daily OHLCV rows, 2010–2026) stored as Parquet. We tested two classic quantitative strategies — cross-sectional momentum and mean reversion — across parameter grids of up to 2,160 configurations using Spark RDDs for parallel execution. Results were validated using the Deflated Sharpe Ratio (DSR) and walk-forward out-of-sample testing. Our honest conclusion is that naive in-sample Sharpe ratios are statistically indistinguishable from noise, and that real-world deployment frictions reduce apparent alpha by over 62%. The project's primary contribution is a transparent, scalable, and methodologically rigorous backtesting infrastructure rather than the discovery of a profitable strategy.

2. Introduction
2.1 Motivation
Momentum and mean reversion are two of the most extensively studied phenomena in quantitative finance. Momentum posits that stocks with strong recent performance tend to continue outperforming. Mean reversion posits that stocks that have deviated too far from their historical average tend to snap back. Despite decades of academic study, the practical profitability of these strategies remains contested — largely because reported results are routinely inflated by overfitting, survivorship bias, and inadequate correction for multiple testing.
The core problem is that trading strategies are highly parameter-rich. A momentum strategy alone involves choices about lookback window, rebalance frequency, portfolio concentration, and long/short sizing. Testing many combinations and reporting only the best result is a form of data snooping that produces an optimistic but misleading picture of true strategy quality.
2.2 Objectives
This project addresses that challenge with four concrete goals:

Build an end-to-end Spark pipeline with explicit data-quality auditing before any analysis begins.
Run distributed parameter sweeps across a large configuration grid using Spark RDDs to enable honest multiple-testing correction.
Report results using statistically rigorous metrics — specifically the Deflated Sharpe Ratio and walk-forward out-of-sample validation.
Quantify real-world deployment frictions including market impact, borrow costs, taxes, and the adj_close data bug.

2.3 Why Distributed Computing
A single-threaded pandas workflow bounds the parameter grid by the analyst's patience. Small grids understate the multiple-testing burden when a "winner" is reported — because you simply couldn't afford to test enough configurations to know how lucky you got. Spark allows us to test hundreds or thousands of configurations in parallel, which both accelerates the research cycle and forces honest statistical accounting for every trial run.

3. Dataset
3.1 Data Sources
OHLCV Price Data: Daily Open, High, Low, Close, Volume data was sourced via yfinance for all current S&P 500 constituents. The dataset spans January 4, 2010 through April 17, 2026, covering 503 tickers and totaling 1,940,547 daily rows.
FRED Macroeconomic Data: A supplementary macro panel was constructed from the Federal Reserve Economic Data (FRED) API, including five daily series: VIX, DGS10 (10-Year Treasury yield), FEDFUNDS (Federal Funds rate), CPIAUCSL (Consumer Price Index), and UNRATE (Unemployment rate). Monthly series were forward-filled to daily frequency.
3.2 Known Limitations
Two structural biases are present in the data source and are explicitly acknowledged:
Survivorship Bias: yfinance only includes companies currently in the S&P 500. Companies that were delisted, went bankrupt, or were removed from the index are absent. This systematically inflates backtest performance. Importantly, this bias works against our null result — meaning that even with this favorable inflation, our strategies failed to demonstrate genuine alpha.
Snapshot vs. Point-in-Time: The constituent list reflects today's membership, not historical membership at each point in time. True point-in-time data requires professional databases such as CRSP or Compustat.
3.3 Data Quality Audit (Smoke Test)
Before any feature engineering or strategy testing, a dedicated smoke-test notebook audited the raw data across four dimensions:
CheckResultMeaningPer-ticker coverage (≥ 4,000 days)426 / 503 full77 partial — IPOs, spinoffs, recent listingsPer-column null census0 nulls (all OHLCV columns)Schema fully enforced at ingestionVolume = 0 days (halts)3,865 / 1.94M = 0.20%Trading halts and pre-merger suspensions|return| > 50% (outlier scan)16 / 1.94M = 0.0008%Stock splits — surfaces the adj_close bug
Zero-volume days were retained in the panel with a flag rather than dropped, since price data remains meaningful on halt days. The 16 extreme return outliers were traced to stock splits not absorbed by the raw close price, directly motivating the adj_close correction described in Section 8.3.
3.4 Exploratory Data Analysis
Six analytical questions were answered in the EDA notebook:
Coverage by Year: Most tickers show complete annual coverage throughout the 16-year window. Gaps are concentrated in recent years for newly added index constituents.
Return Distribution: Daily log returns exhibit the well-known fat-tail property of equity markets. Excess kurtosis across the full panel is 58.5 — far above the Gaussian assumption of 0. Skewness is −1.01, confirming a left tail from occasional large losses.
Sector Dollar Volume: Information Technology dominates average daily dollar volume, consistent with the market-cap weight of mega-cap technology names in the S&P 500.
Volatility Regimes: Cross-sectional median 20-day realized volatility clearly identifies three major stress periods: Q4 2018 (Fed tightening panic), March 2020 (COVID-19 shock), and 2022 (Fed rate hike cycle).
VIX vs. Realized Vol: Pearson correlation between cross-sectional median realized volatility and the VIX is 0.806, confirming that bottom-up individual-stock volatility and the index-implied VIX capture the same underlying market risk environment.

4. Feature Engineering
All features were computed using a single Spark pipeline (src/features.py) over the full 1.94M-row OHLCV panel using Window functions partitioned by ticker and ordered by date. Output was persisted as Parquet partitioned by year for efficient partition-pruned reads in downstream notebooks.
4.1 Feature Catalog
A total of 27 engineered features were produced across six categories:
Return Features

log_ret — Daily log return: ln(close_t / close_{t-1})
simple_ret — Daily simple return: (close_t / close_{t-1}) − 1

Volatility Features

vol_20d, vol_60d, vol_252d — Rolling annualized standard deviation of log returns at 20, 60, and 252-day windows

Moving Average Features

sma_20d, sma_50d, sma_200d — Simple moving averages at 20, 50, and 200 days
sma_20d_ratio, sma_50d_ratio, sma_200d_ratio — Ratio of current close to corresponding SMA

Momentum Features

mom_63d, mom_126d, mom_252d — Cumulative log return over 63, 126, and 252 trading days (~3, 6, and 12 months)

Technical Indicator Features

rsi_14 — 14-day Relative Strength Index, bounded [0, 100]
bb_upper, bb_lower — Bollinger Band bounds (20-day SMA ± 2 standard deviations)
bb_pct — Bollinger percent-B: position of current price within the bands (0 = lower band, 1 = upper band)

Cross-Sectional Rank Features

mom_252d_rank — Per-date cross-sectional percentile rank of mom_252d across all tickers
bb_pct_rank — Per-date cross-sectional percentile rank of bb_pct across all tickers

4.2 Sanity Checks
Three automated checks were run on the computed feature layer:

RSI bounds: Zero rows outside [0, 100] ✓
Momentum plausibility: Zero rows with |mom_252d| ≥ 5 (annual log return exceeding 500%) ✓
Cross-sectional rank uniformity: Distribution of mom_252d_rank across a 5% sample is approximately uniform on [0, 1], confirming even spread across tickers ✓


5. Trading Strategies
5.1 Cross-Sectional Momentum
Premise: Stocks with strong recent returns relative to their peers will continue to outperform over the near term.
Mechanics:

On each rebalance date, rank all tickers cross-sectionally by a chosen momentum signal (mom_252d, mom_126d, or mom_63d).
Go long the top top_pct of tickers and short the bottom top_pct, equal-weighted within each side.
The portfolio is dollar-neutral with gross exposure of 2×.
Rebalance at a configurable cadence (21, 42, or 63 trading days).

Key properties: Relatively low turnover. Positions held for weeks to months. 5 bps transaction cost per unit of turnover. Trade execution assumed at T+1.
5.2 Mean Reversion
Premise: Stocks that have moved unusually far from their cross-sectional average will revert back toward the mean.
Mechanics:

On each day, compute the cross-sectional z-score of bb_pct across all tickers.
Enter a long position when a ticker's z-score falls below −entry_z.
Exit when the z-score rises above −exit_z, or when the holding period reaches max_holding_days.

Key properties: Very high turnover — 15 to 25× that of momentum. Transaction costs dramatically erode gross returns. More sensitive to borrow costs due to frequent short-side activity.
5.3 Backtest Engine
All backtests share three common mechanics:

T+1 execution: Weights computed on day T are applied to returns on day T+1, preventing look-ahead bias.
Transaction costs: 5 basis points per unit of one-way portfolio turnover at each rebalance.
Runtime: ~1.4 seconds per backtest on a single core.


6. Distributed Parameter Sweeps
6.1 Motivation
Any single backtest result is meaningless without knowing how many configurations were tested to find it. The more configurations tested, the more likely it is to find a strong result by chance — even if no underlying signal exists. Spark allows exhaustive grid testing, which is a prerequisite for honest statistical correction.
6.2 Sweep Architecture
Parameter grids were distributed using RDD.parallelize().map().collect(). Each configuration — a tuple of (lookback, rebalance cadence, top_pct, strategy variant) — was mapped to a self-contained backtest function. Results were collected back to the driver as a list of performance metric dictionaries.
Three sweep tiers were executed:
SweepGridConfigurationsWall TimeBaselineMomentum 3×3×3 + Mean-rev 1×3×2×34519.4 sBonus ExtendedExpanded momentum + long-only variants18039.1 sWalk-Forward3 lookbacks × 5 top% × 6 rebal × 2 sides2,160214.8 s
6.3 Speedup Analysis
The baseline 45-configuration sweep achieved a 3.36× speedup over sequential execution (65 seconds reduced to 19.4 seconds on 12 local cores). The speedup is sub-linear due to three factors:

Spark task-launch overhead — fixed startup cost is significant when tasks complete in ~1.4 seconds each.
GIL contention — pandas-based workers within the same JVM share Python's Global Interpreter Lock.
Broadcast overhead — amortizes more favorably at 100+ tasks than at 45.

On a real multi-machine cluster with 100+ configurations per batch, speedup approaches core count as overhead becomes negligible.

7. Results
7.1 Baseline 45-Configuration Sweep
The baseline sweep produced a maximum raw Sharpe Ratio of 0.27, achieved by mom_011 (63-day rebalance, top 10%). After applying the Deflated Sharpe Ratio correction for the number of trials, the DSR collapses to ≈ 0. Discovery verdict: FALSE.
Key observations:

Best configs consistently used slow rebalancing (63-day) with concentrated portfolios (top 10%).
Mean reversion performed worst — trading costs consumed all gross alpha due to 15–25× higher turnover.
Return distribution: skew = −0.47, excess kurtosis = +6.96, both of which further depress the DSR.

7.2 Bonus 180-Configuration Sweep
Expanding to 180 configurations, the best in-sample result — lo_124 (long-only, 126-day momentum, top 5%, 63-day rebalance) — appeared compelling:
MetricValueStarting Capital$1,000,000Ending Capital$116,700,000Sharpe Ratio1.27CAGR34%Maximum Drawdown−36%
The DSR is again ≈ 0. Out of 180 trials, finding one configuration with Sharpe 1.27 is exactly what you would expect by random chance. Additionally, 66 of 180 configurations beat the equal-weight benchmark on both Sharpe and max drawdown simultaneously — confirming results are dominated by selection bias, not signal.
7.3 Walk-Forward Out-of-Sample Validation
Walk-forward testing is the only methodology that guards against selection bias by evaluating each selected strategy exclusively on data it has never seen.
Method: For each OOS year 2014–2025, all 180 configurations were trained on data from 2010 through year Y−1. The highest-Sharpe configuration from that training window was selected and evaluated on year Y only. Total: 2,160 backtests in 214.8 seconds on 12 local cores.
ScenarioEnding Capital ($1M start)In-sample bonus winner$116,700,000Walk-forward (honest)$14,500,000Equal-weight benchmark$5,000,000
OOS Sharpe: 1.03 | DSR: 1.5% (n = 12 trials) | Verdict: Borderline — best-defended
The strategy beat the benchmark in 7 of 12 years. Worst year was 2022, with an OOS Sharpe of −0.90.

8. Production Frictions
Even the honest walk-forward result of $14.5M deteriorates substantially under real-world deployment constraints.
8.1 Capacity
At $1 billion AUM, the median trade would represent approximately 19% of a stock's average daily volume — practically infeasible to execute. Effective capacity ceiling: $100M – $1B.
8.2 Borrow Cost
Short selling requires borrowing shares at an annual fee. At a conservative 50 bps/year, borrow costs reduce Sharpe from 0.37 to 0.35 for the best long/short configuration — which already loses to the benchmark before borrow charges. Long-only is the only deployable configuration.
8.3 The adj_close Bug
Returns in the feature pipeline were initially computed from raw close prices rather than split-adjusted adj_close. Stock splits cause an apparent price drop on the split date that appears as a large negative return in the raw series. Fixing this required changing one line in src/features.py:

Sharpe Ratio: 1.27 → 1.31
Portfolio terminal value: +$19M

This is the highest-leverage finding in the entire project — a single data engineering error worth more than any strategy parameter optimization.
8.4 Stacked Friction
Combining permanent market impact (10 bps) and a 35% short-term capital gains tax applies a −62.3% cumulative haircut to the walk-forward result:
$14.5M → $5.5M deployable
The $5.5M outcome still beats the $5.0M equal-weight benchmark, but only marginally and only under generous assumptions.

9. Spark's Role
OutcomeWorkloadWhy Spark WinsSelection bias correction2,160 train backtestsRDD.parallelize().map().collect() — 3.5 min on 12 coresCapacity / impact analysis1.94M rows × weights × ADVSame join scales to 200M+ rows without redesignFriction sensitivitiesMulti-scenario cost stacksTrivially fans out to (config × cost-regime) gridsadj_close fixWindow recompute over 1.94M rowsAlready the established pattern in src/features.pyData-quality auditNull census, coverage, outlier scanOne groupBy().agg() per check over the full panel
9.1 Scalability

Minute-bar data (~390× more rows): Feature engineering via Window functions is unchanged; distribution pattern is identical.
Global equities (~50× more tickers): Switch to repartitioned DataFrame joins; no architectural change required.
10,000+ configurations: Near-linear scaling until cluster core count is saturated.


10. Lessons Learned
DSR Is Brutal, But Right
A Sharpe of 1.27 across 180 trials is statistically indistinguishable from a random walk. Walk-forward is the only honest answer. Reporting raw in-sample Sharpe without DSR correction is not just incomplete — it is misleading.
Audit Before Analyzing
The smoke-test notebook is not optional housekeeping. It is the first substantive analysis step. It directly surfaced the stock-split issue later quantified at $19M.
One Line of Code = $19M
Switching close to adj_close in one line of feature engineering code was the single highest-leverage improvement in the entire project. Build features from properly adjusted prices.
Sub-Linear Speedup on a Laptop Is Normal
3.36× on 12 cores is expected behavior for short-running tasks on a single JVM. On a real multi-machine cluster with 100+ tasks, overhead amortizes and speedup approaches core count.
Distribute at the Right Granularity
Spark excels at large DataFrame operations — group-by-ticker Window functions over 1.94M rows, joins between price data and portfolio weights. Individual 1.4-second backtests are better handled by pandas inside Spark workers. Using Spark at the wrong layer loses to overhead.
Production Caveats Are Tractable
Capacity, impact, borrow, and taxes each required one additional notebook because Spark made them joins over existing data structures — not new architectural projects.

11. Limitations and Future Work
11.1 Addressed and Quantified
LimitationMitigationImpactSelection biasWalk-forward validationOOS DSR = 1.5% (n = 12)Flat 5 bps slippageLinear-impact + AUM sweepCapacity ceiling $100M–$1BBorrow cost on shorts50 bps/yr appliedSharpe 0.37 → 0.35Splits via raw closeadj_close fix+0.04 Sharpe, +$19MStacked frictionsImpact + tax combined−62.3% haircut on walk-forward
11.2 Deferred to Future Work

Survivorship bias — Need point-in-time constituent data from CRSP/Compustat
Daily resolution only — Intraday microstructure signals are out of scope
No regime-aware switching — Same parameters applied across QE, COVID, and 2022 bear market
No risk controls — Inverse-vol sizing, sector neutrality, and concentration caps not implemented
Linear impact model — Almgren-Chriss square-root model would be more accurate at high AUM


12. Conclusion
This project set out to answer whether momentum and mean reversion strategies actually make money — and arrived at a nuanced but honest answer captured in three numbers:
ResultSharpeDSRMeaningNaive 45-config grid0.27≈ 0NoiseIn-sample best (180 configs)1.27≈ 0Selection bias fantasyWalk-forward OOS1.031.5%Only defensible result — $14.5M → $5.5M post-friction
The progression from 0.27 → 1.27 → 1.03 tells the complete story of quantitative backtesting: naive results are meaningless, optimized in-sample results are misleading, and only honest out-of-sample validation with statistical correction for multiple testing produces a result worth discussing.
The primary contribution of this project is not the discovery of a profitable trading strategy. It is the demonstration — through a reproducible, scalable, and transparent PySpark pipeline — that rigorous methodology dramatically changes what results mean.

References

López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management.
Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers. Journal of Finance, 48(1), 65–91.
Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. Journal of Risk, 3(2), 5–39.
Apache Spark Documentation. (2024). PySpark RDD Programming Guide. https://spark.apache.org/docs/latest/rdd-programming-guide.html
Federal Reserve Bank of St. Louis. FRED Economic Data. https://fred.stlouisfed.org
