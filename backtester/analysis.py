"""Performance analysis helpers for the backtester.

Provides simple metric computations and a benchmark-comparison framework.

Usage:
    from backtester.analysis import compute_nav_metrics, compare_to_benchmark

    metrics = compute_nav_metrics(nav_series)
    comparison = compare_to_benchmark(nav_series, benchmark_prices)

All functions expect `pd.Series` or `pd.DataFrame` indexed by `pd.DatetimeIndex`.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_nav_metrics(nav: pd.Series, trading_days: int = 252, risk_free: float = 0.0) -> Dict[str, float]:
    """Compute basic performance metrics from a NAV series.

    Returns a dict with: total_return, years, cagr, annual_vol, sharpe, max_drawdown, peak_date, trough_date
    """
    if not isinstance(nav.index, pd.DatetimeIndex):
        raise ValueError("nav must be indexed by a DatetimeIndex")

    nav = nav.sort_index().dropna()
    if len(nav) < 2:
        return {}

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float('nan')

    returns = nav.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(trading_days)
    ann_return = returns.mean() * trading_days
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else float('nan')

    # drawdown
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_dd = drawdown.min()
    if pd.isna(max_dd):
        peak_date = trough_date = None
    else:
        trough_date = drawdown.idxmin()
        peak_date = nav.loc[:trough_date].idxmax()

    return {
        'total_return': float(total_return),
        'years': float(years),
        'cagr': float(cagr),
        'annual_vol': float(ann_vol),
        'annual_return': float(ann_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'peak_date': pd.Timestamp(peak_date) if peak_date is not None else None,
        'trough_date': pd.Timestamp(trough_date) if trough_date is not None else None,
    }


def compare_to_benchmark(nav: pd.Series, benchmark_prices: pd.Series, trading_days: int = 252, risk_free: float = 0.0) -> Dict[str, float]:
    """Compare NAV to a benchmark price series.

    Returns: dict with tracking_error, information_ratio, active_return (annualized), benchmark_cagr, and fund metrics.
    """
    if not isinstance(nav.index, pd.DatetimeIndex) or not isinstance(benchmark_prices.index, pd.DatetimeIndex):
        raise ValueError('Both inputs must have a DatetimeIndex')

    nav = nav.sort_index().dropna()
    bench = benchmark_prices.sort_index().dropna()

    # Align on dates (inner join)
    df = pd.DataFrame({'nav': nav, 'bench': bench}).dropna()
    if df.empty:
        raise ValueError('No overlapping dates between nav and benchmark')

    # Daily returns
    r_nav = df['nav'].pct_change().dropna()
    r_bench = df['bench'].pct_change().dropna()
    # align again after pct_change
    r = pd.concat([r_nav, r_bench], axis=1).dropna()
    r_nav = r.iloc[:, 0]
    r_bench = r.iloc[:, 1]

    active = r_nav - r_bench
    tracking_error = active.std() * np.sqrt(trading_days)
    active_annual = active.mean() * trading_days
    information_ratio = active_annual / tracking_error if tracking_error > 0 else float('nan')

    bench_total = df['bench'].iloc[-1] / df['bench'].iloc[0] - 1.0
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    bench_cagr = (df['bench'].iloc[-1] / df['bench'].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float('nan')

    fund_metrics = compute_nav_metrics(df['nav'], trading_days=trading_days, risk_free=risk_free)

    return {
        'tracking_error': float(tracking_error),
        'information_ratio': float(information_ratio),
        'active_annual_return': float(active_annual),
        'benchmark_cagr': float(bench_cagr),
        **{f'fund_{k}': v for k, v in fund_metrics.items()}
    }


__all__ = ['compute_nav_metrics', 'compare_to_benchmark']
