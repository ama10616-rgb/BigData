"""Vectorized backtest engine.

Design:
- weights_df : long-form (date, ticker, weight). The weight on date D is the
  target exposure held FROM D through D+1, i.e. it earns the return from D to D+1.
- returns_df : long-form (date, ticker, simple_ret). simple_ret on date D is
  close_D / close_{D-1} - 1, so it is the return EARNED ON date D from the
  exposure entered end-of-D-1.

To avoid look-ahead, we shift weights forward one trading day before multiplying
by returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_BPS = 5.0  # 5 basis points per unit of turnover


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df[["date", "ticker", value]]
        .drop_duplicates(subset=["date", "ticker"])
        .pivot(index="date", columns="ticker", values=value)
        .sort_index()
    )


def compute_turnover(weights_df: pd.DataFrame) -> float:
    """Average daily turnover = mean over time of sum |Δweight|."""
    W = _pivot(weights_df, "weight").fillna(0.0)
    dW = W.diff().abs().sum(axis=1)
    return float(dW.mean())


def run_backtest(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Return DataFrame indexed by date with columns:
        gross_ret, turnover, cost, net_ret, equity
    """
    W = _pivot(weights_df, "weight").fillna(0.0)
    R = _pivot(returns_df, "simple_ret").fillna(0.0)

    common_tickers = W.columns.intersection(R.columns)
    W = W[common_tickers]
    R = R[common_tickers]

    common_dates = W.index.union(R.index).sort_values()
    W = W.reindex(common_dates).fillna(0.0)
    R = R.reindex(common_dates).fillna(0.0)

    # Shift weights forward by one day so weight_{t-1} earns return_t
    W_eff = W.shift(1).fillna(0.0)
    gross_ret = (W_eff * R).sum(axis=1)

    # Turnover on date t = sum |W_t - W_{t-1}|; cost charged on that date
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 10_000.0)
    net_ret = gross_ret - cost

    equity = (1.0 + net_ret).cumprod()

    out = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
        }
    )
    return out
