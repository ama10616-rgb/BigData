"""Performance metrics: Sharpe, Sortino, Max Drawdown, Calmar, CAGR, Deflated Sharpe."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# Minimal normal CDF (avoids a scipy dep for one function).
class _Norm:
    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_norm = _Norm()


def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        x = x.dropna().to_numpy()
    else:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
    return x


_ZERO_VOL_EPS = 1e-12


def sharpe(returns, rf: float = 0.0, ann: int = 252) -> float:
    r = _as_array(returns) - rf / ann
    if r.size == 0:
        return float("nan")
    sd = r.std(ddof=1)
    scale = max(abs(r.mean()), 1.0)
    if sd < _ZERO_VOL_EPS * scale or np.isnan(sd):
        return float("inf") if r.mean() != 0 else float("nan")
    return float(r.mean() / sd * math.sqrt(ann))


def sortino(returns, rf: float = 0.0, ann: int = 252) -> float:
    r = _as_array(returns) - rf / ann
    if r.size == 0:
        return float("nan")
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf") if r.mean() > 0 else float("nan")
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("inf") if r.mean() > 0 else float("nan")
    return float(r.mean() / dd * math.sqrt(ann))


def max_drawdown(equity_curve) -> float:
    e = _as_array(equity_curve)
    if e.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(e)
    dd = e / running_max - 1.0
    return float(dd.min())


def cagr(equity_curve, ann: int = 252) -> float:
    e = _as_array(equity_curve)
    if e.size < 2 or e[0] <= 0:
        return float("nan")
    n_periods = len(e) - 1
    years = n_periods / ann
    if years <= 0:
        return float("nan")
    return float((e[-1] / e[0]) ** (1 / years) - 1)


def calmar(returns, equity_curve, ann: int = 252) -> float:
    c = cagr(equity_curve, ann=ann)
    dd = max_drawdown(equity_curve)
    if dd == 0 or np.isnan(dd) or np.isnan(c):
        return float("nan")
    return float(c / abs(dd))


def deflated_sharpe(sharpe_obs: float, n_trials: int, returns) -> float:
    """Deflated Sharpe Ratio, Bailey & Lopez de Prado (2014).

    Computes a probability that the observed Sharpe exceeds the expected maximum
    under the null, adjusted for skew and kurtosis of the return stream and for
    multiple-testing across `n_trials` configurations.

    Reference:
        Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
        Journal of Portfolio Management 40 (5): 94-107.
    """
    r = _as_array(returns)
    n = r.size
    if n < 10 or n_trials < 1 or np.isnan(sharpe_obs):
        return float("nan")

    # Sample skew / kurtosis (Fisher definition: excess kurtosis = kurt - 3)
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return float("nan")
    g1 = float(((r - mean) ** 3).mean() / std ** 3)
    g2 = float(((r - mean) ** 4).mean() / std ** 4 - 3.0)

    # Expected maximum Sharpe across n_trials independent trials ~ N(0,1):
    # E[max_N] ~= (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))
    # (Bailey & Lopez de Prado 2014 eq. 7; gamma is Euler-Mascheroni constant)
    gamma = 0.5772156649
    eN = 1.0 / n_trials
    z1 = _inv_norm_cdf(1.0 - eN)
    z2 = _inv_norm_cdf(1.0 - eN / math.e)
    exp_max_sharpe = (1.0 - gamma) * z1 + gamma * z2  # annualized SR (per sqrt(252))

    # Convert observed sharpe to non-annualized (per-period) for the DSR formula
    sr_period = sharpe_obs / math.sqrt(252.0)
    numerator = (sr_period - exp_max_sharpe / math.sqrt(252.0)) * math.sqrt(n - 1)
    denominator = math.sqrt(1.0 - g1 * sr_period + (g2 / 4.0) * sr_period ** 2)
    if denominator <= 0 or math.isnan(denominator):
        return float("nan")
    p = _norm.cdf(numerator / denominator)
    return float(p)


def _inv_norm_cdf(p: float) -> float:
    """Rational approximation to the inverse normal CDF (Beasley-Springer-Moro)."""
    if p <= 0.0 or p >= 1.0:
        if p <= 0.0:
            return -float("inf")
        return float("inf")
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
