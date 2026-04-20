"""Sanity tests for src/metrics.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    cagr,
    calmar,
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
)


def test_sharpe_constant_returns_infinite_or_nan():
    r = np.array([0.001] * 252)
    s = sharpe(r)
    assert math.isinf(s) or math.isnan(s)


def test_sharpe_known_series():
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.001, scale=0.01, size=1000)
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert abs(sharpe(r) - expected) < 1e-6


def test_sortino_all_positive_returns():
    r = np.array([0.01, 0.005, 0.003])
    assert math.isinf(sortino(r))


def test_sortino_mixed():
    rng = np.random.default_rng(1)
    r = rng.normal(loc=0.0005, scale=0.01, size=500)
    s = sortino(r)
    assert np.isfinite(s)


def test_max_drawdown_simple_case():
    eq = np.array([1.0, 1.1, 1.2, 0.9, 1.3])
    # peak 1.2, trough 0.9 -> dd = 0.9/1.2 - 1 = -0.25
    assert abs(max_drawdown(eq) - (-0.25)) < 1e-12


def test_cagr_double_in_one_year():
    eq = np.linspace(1.0, 2.0, 253)  # 252 periods = 1 year
    c = cagr(eq)
    assert abs(c - 1.0) < 1e-6


def test_calmar_sign():
    # Declining then recovering series: ensure calmar computable
    eq = np.array([1.0, 1.1, 0.9, 1.15])
    rets = np.diff(eq) / eq[:-1]
    assert np.isfinite(calmar(rets, eq))


def test_dsr_basic_shape():
    rng = np.random.default_rng(42)
    r = rng.normal(loc=0.0005, scale=0.01, size=2520)  # ~10 years
    s = sharpe(r)
    p = deflated_sharpe(s, n_trials=1, returns=r)
    assert 0.0 <= p <= 1.0


def test_dsr_decreases_with_more_trials():
    rng = np.random.default_rng(7)
    r = rng.normal(loc=0.0005, scale=0.01, size=2520)
    s = sharpe(r)
    p1 = deflated_sharpe(s, n_trials=1, returns=r)
    p100 = deflated_sharpe(s, n_trials=100, returns=r)
    # More trials -> expected max-Sharpe is higher -> observed is less surprising -> DSR prob lower
    assert p100 <= p1 + 1e-9
