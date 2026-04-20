"""Strategy signal generators.

Each function takes a pandas features DataFrame (long form: one row per
ticker-date) and a params dict, and returns a weights DataFrame with columns
(date, ticker, weight). Weights are interpreted as target exposure on the
given date, applied to the NEXT day's return in the backtest engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rebalance_dates(all_dates: pd.DatetimeIndex, every_n: int) -> set:
    """Every Nth unique sorted date is a rebalance."""
    sorted_dates = pd.DatetimeIndex(sorted(all_dates.unique()))
    return set(sorted_dates[::every_n])


def cross_sectional_momentum(
    features_df: pd.DataFrame,
    lookback_col: str,
    top_pct: float,
    rebalance_days: int,
) -> pd.DataFrame:
    """Long top-decile / short bottom-decile by a momentum column.

    Dollar-neutral (weights sum to 0), gross exposure = 2 (equal long + short book).
    Holds until next rebalance.
    """
    df = features_df[["date", "ticker", lookback_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[lookback_col])

    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    rebal = _rebalance_dates(all_dates, rebalance_days)

    weights_rows = []
    current_weights: dict[str, float] = {}

    for d in all_dates:
        if d in rebal:
            snap = df[df["date"] == d]
            if len(snap) < 10:
                continue
            q_hi = snap[lookback_col].quantile(1 - top_pct)
            q_lo = snap[lookback_col].quantile(top_pct)
            longs = snap[snap[lookback_col] >= q_hi]["ticker"].tolist()
            shorts = snap[snap[lookback_col] <= q_lo]["ticker"].tolist()
            current_weights = {}
            if longs:
                w = 1.0 / len(longs)
                for t in longs:
                    current_weights[t] = w
            if shorts:
                w = -1.0 / len(shorts)
                for t in shorts:
                    current_weights[t] = current_weights.get(t, 0.0) + w
        for t, w in current_weights.items():
            weights_rows.append((d, t, w))

    if not weights_rows:
        return pd.DataFrame(columns=["date", "ticker", "weight"])
    return pd.DataFrame(weights_rows, columns=["date", "ticker", "weight"])


def mean_reversion(
    features_df: pd.DataFrame,
    z_col: str,
    entry_z: float,
    exit_z: float,
    max_holding_days: int,
) -> pd.DataFrame:
    """Z-score mean reversion on z_col.

    z_col is assumed to be roughly zero-mean, unit-variance cross-sectionally or
    within ticker (e.g. bb_pct scaled to z). Enters long when z < -entry_z,
    exits when z > -exit_z or holding >= max_holding_days. Symmetric short side.
    Equal weight across active positions; positions refreshed daily.
    """
    df = features_df[["date", "ticker", z_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[z_col])

    # Convert bb_pct (in [0,1]) or rank-style columns into a symmetric z-like score.
    mu = df[z_col].mean()
    sigma = df[z_col].std()
    if sigma == 0 or np.isnan(sigma):
        return pd.DataFrame(columns=["date", "ticker", "weight"])
    df["_z"] = (df[z_col] - mu) / sigma

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    position: dict[str, int] = {}  # ticker -> +1/-1/0
    entered: dict[str, pd.Timestamp] = {}

    rows: list[tuple] = []
    for ticker, g in df.groupby("ticker", sort=False):
        position_t = 0
        entered_t: pd.Timestamp | None = None
        for d, z in zip(g["date"].values, g["_z"].values):
            d_ts = pd.Timestamp(d)
            if position_t == 0:
                if z < -entry_z:
                    position_t = 1
                    entered_t = d_ts
                elif z > entry_z:
                    position_t = -1
                    entered_t = d_ts
            else:
                held = (d_ts - entered_t).days if entered_t is not None else 0
                if position_t == 1 and (z > -exit_z or held >= max_holding_days):
                    position_t = 0
                    entered_t = None
                elif position_t == -1 and (z < exit_z or held >= max_holding_days):
                    position_t = 0
                    entered_t = None
            if position_t != 0:
                rows.append((d_ts, ticker, float(position_t)))

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "weight"])

    raw = pd.DataFrame(rows, columns=["date", "ticker", "sign"])
    # Equal-weight across active positions on each date; dollar-neutral if both sides present.
    grouped = raw.groupby("date")
    out_frames = []
    for d, g in grouped:
        longs = g[g["sign"] == 1]
        shorts = g[g["sign"] == -1]
        parts = []
        if len(longs):
            parts.append(longs.assign(weight=1.0 / len(longs)))
        if len(shorts):
            parts.append(shorts.assign(weight=-1.0 / len(shorts)))
        if parts:
            out_frames.append(pd.concat(parts))
    out = pd.concat(out_frames)[["date", "ticker", "weight"]].reset_index(drop=True)
    return out
