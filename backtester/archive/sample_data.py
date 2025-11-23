"""Deprecated: sample_data generator replaced by consolidated tooling.

This file is intentionally left as a harmless placeholder.
"""

print('backtester/sample_data.py is deprecated. Use backtester_single.py at repo root if you need a runner.')
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List


def generate_sample_prices(tickers: List[str], start: str = '2021-01-01', end: str = None, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily close prices for `tickers` between `start` and `end` (business days).

    Prices are simulated with a simple geometric Brownian motion. This is for backtest/demo only.
    """
    if end is None:
        end = datetime.utcnow().date().isoformat()

    rng = pd.bdate_range(start=start, end=end)
    np.random.seed(seed)

    prices = pd.DataFrame(index=rng)
    for t in tickers:
        S0 = 100.0 + np.random.rand() * 50
        mu = 0.06  # annual drift
        sigma = 0.25  # annual vol
        dt = 1/252
        increments = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=len(rng))
        logp = np.log(S0) + np.cumsum(increments)
        prices[t] = np.exp(logp)

    prices.index.name = 'date'
    return prices


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample price CSV for demo backtests')
    parser.add_argument('--tickers', nargs='+', required=False, default=['AAPL','MSFT','GOOG','AMZN'])
    parser.add_argument('--start', default='2021-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--out', default='backtester/sample_prices.csv')
    args = parser.parse_args()

    df = generate_sample_prices(args.tickers, start=args.start, end=args.end)
    df.to_csv(args.out)
    print(f'Wrote sample prices to {args.out} with {len(df)} rows and {len(args.tickers)} tickers')
