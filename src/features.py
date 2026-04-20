"""Feature engineering for OHLCV data using Spark window functions.

All functions take and return a Spark DataFrame. The canonical ordering is
Window.partitionBy("ticker").orderBy("date").
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

TRADING_DAYS = 252


def _ticker_window() -> Window:
    return Window.partitionBy("ticker").orderBy("date")


def add_returns(df: DataFrame) -> DataFrame:
    """Add log_ret and simple_ret columns from close."""
    w = _ticker_window()
    return (
        df.withColumn("prev_close", F.lag("close").over(w))
        .withColumn("log_ret", F.log(F.col("close") / F.col("prev_close")))
        .withColumn("simple_ret", F.col("close") / F.col("prev_close") - F.lit(1.0))
        .drop("prev_close")
    )


def add_rolling_vol(df: DataFrame, windows=(20, 60, 252)) -> DataFrame:
    """Annualized rolling stdev of log_ret, one column per window size."""
    out = df
    for n in windows:
        w = _ticker_window().rowsBetween(-(n - 1), 0)
        out = out.withColumn(
            f"vol_{n}d",
            F.stddev("log_ret").over(w) * F.sqrt(F.lit(float(TRADING_DAYS))),
        )
    return out


def add_moving_averages(df: DataFrame, windows=(20, 50, 200)) -> DataFrame:
    """Simple moving averages and close/SMA ratio."""
    out = df
    for n in windows:
        w = _ticker_window().rowsBetween(-(n - 1), 0)
        out = (
            out.withColumn(f"sma_{n}d", F.avg("close").over(w))
            .withColumn(f"sma_{n}d_ratio", F.col("close") / F.col(f"sma_{n}d"))
        )
    return out


def add_momentum(df: DataFrame, lookbacks=(63, 126, 252)) -> DataFrame:
    """12-1 style momentum: total log return over (t-N, t-1), skipping most recent day.

    Computed as sum(log_ret) over rows [-N, -1] (inclusive, excluding current row).
    """
    out = df
    for n in lookbacks:
        w = _ticker_window().rowsBetween(-n, -1)
        out = out.withColumn(f"mom_{n}d", F.sum("log_ret").over(w))
    return out


def add_rsi(df: DataFrame, window: int = 14) -> DataFrame:
    """Wilder RSI using Spark windows.

    Implemented as simple average gain / loss over the lookback window
    (a close approximation to Wilder smoothing that avoids stateful recursion,
    standard in vectorized backtests).
    """
    diff_col = "_close_diff"
    gain_col = "_gain"
    loss_col = "_loss"
    w = _ticker_window()
    wr = _ticker_window().rowsBetween(-(window - 1), 0)
    out = (
        df.withColumn(diff_col, F.col("close") - F.lag("close").over(w))
        .withColumn(gain_col, F.when(F.col(diff_col) > 0, F.col(diff_col)).otherwise(F.lit(0.0)))
        .withColumn(loss_col, F.when(F.col(diff_col) < 0, -F.col(diff_col)).otherwise(F.lit(0.0)))
        .withColumn("_avg_gain", F.avg(gain_col).over(wr))
        .withColumn("_avg_loss", F.avg(loss_col).over(wr))
        .withColumn(
            "rsi_14",
            F.when(
                F.col("_avg_loss") == 0, F.lit(100.0)
            ).otherwise(
                F.lit(100.0) - (F.lit(100.0) / (F.lit(1.0) + F.col("_avg_gain") / F.col("_avg_loss")))
            ),
        )
        .drop(diff_col, gain_col, loss_col, "_avg_gain", "_avg_loss")
    )
    return out


def add_bollinger(df: DataFrame, window: int = 20, n_std: float = 2.0) -> DataFrame:
    """Bollinger bands and band position (bb_pct in [0,1] at band edges)."""
    w = _ticker_window().rowsBetween(-(window - 1), 0)
    out = (
        df.withColumn("_bb_mean", F.avg("close").over(w))
        .withColumn("_bb_std", F.stddev("close").over(w))
        .withColumn("bb_upper", F.col("_bb_mean") + F.lit(n_std) * F.col("_bb_std"))
        .withColumn("bb_lower", F.col("_bb_mean") - F.lit(n_std) * F.col("_bb_std"))
        .withColumn(
            "bb_pct",
            (F.col("close") - F.col("bb_lower"))
            / (F.col("bb_upper") - F.col("bb_lower")),
        )
        .drop("_bb_mean", "_bb_std")
    )
    return out


def add_cross_sectional_rank(df: DataFrame, col: str, rank_col_name: str) -> DataFrame:
    """Rank-transform `col` across tickers on each date to [0, 1].

    Uses percent_rank over a date-partitioned window. NaN inputs become NaN ranks.
    """
    w = Window.partitionBy("date").orderBy(col)
    return df.withColumn(
        rank_col_name,
        F.when(F.col(col).isNull(), F.lit(None).cast("double")).otherwise(F.percent_rank().over(w)),
    )


def build_all_features(df: DataFrame) -> DataFrame:
    """Apply the full feature pipeline in dependency order."""
    out = add_returns(df)
    out = add_rolling_vol(out)
    out = add_moving_averages(out)
    out = add_momentum(out)
    out = add_rsi(out)
    out = add_bollinger(out)
    out = add_cross_sectional_rank(out, "mom_252d", "mom_252d_rank")
    out = add_cross_sectional_rank(out, "bb_pct", "bb_pct_rank")
    return out
