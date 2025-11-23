"""Single-file monthly backtester.

Usage:
  - Edit `backtester/instructions.py` to set `instructions_df` (columns: month,buy,sell).
  - Ensure `backtester/sample_prices.csv` exists with `date` column and ticker columns.
  - Run:
      python -c "import backtester_single; backtester_single.main()"

Outputs:
  - `backtester/trades.csv`
  - `backtester/nav.csv`

This file consolidates the backtester into a single script and reads instructions
from `backtester/instructions.py`.
"""

from __future__ import annotations
import os
import math
import pandas as pd
import numpy as np
import importlib.util
from typing import Dict, List, Tuple


class MonthlyBacktester:
    """Minimal monthly liquidation + rebalance backtester.

    - prices: DataFrame indexed by business dates with ticker columns.
    - start_date: string date where month=1 maps to that month (default Feb 1, 2022).
    - allows fractional shares.
    """

    def __init__(self, prices: pd.DataFrame, start_date: str = "2022-02-01", initial_cash: float = 1_000_000.0):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must be indexed by a DatetimeIndex")
        self.prices = prices.sort_index()
        self.start_date = pd.to_datetime(start_date)
        self.initial_cash = float(initial_cash)

    def _parse_buy_cell(self, cell: str) -> Dict[str, float]:
        if not isinstance(cell, str) or cell.strip() == "":
            return {}
        cell = cell.strip()
        # explicit weights
        if ":" in cell:
            parts = [p.strip() for p in cell.replace(',', ';').split(';') if p.strip()]
            out = {}
            for p in parts:
                if ':' in p:
                    t, w = p.split(':', 1)
                    try:
                        out[t.strip()] = float(w)
                    except Exception:
                        out[t.strip()] = 0.0
            s = sum(out.values())
            if s > 0:
                for k in out:
                    out[k] = out[k] / s
            return out
        # simple list -> equal weight
        tickers = [t.strip() for t in cell.replace(';', ',').split(',') if t.strip()]
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    def _first_price_on_or_after(self, ticker: str, date: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
        idx = self.prices.index[self.prices.index >= date]
        if len(idx) == 0:
            raise KeyError(f"No price for {ticker} on or after {date.date()}")
        for d in idx:
            if ticker not in self.prices.columns:
                raise KeyError(f"Ticker {ticker} not in prices")
            p = self.prices.at[d, ticker]
            if pd.notna(p):
                return d, float(p)
        raise KeyError(f"No valid price for {ticker} on or after {date.date()}")

    def _build_month_starts(self) -> List[pd.Timestamp]:
        all_dates = self.prices.index[self.prices.index >= self.start_date]
        if len(all_dates) == 0:
            raise ValueError("No price data on/after start_date")
        months = sorted({(d.year, d.month) for d in all_dates})
        month_dates = pd.to_datetime([min(d for d in all_dates if (d.year, d.month) == (y, m)) for y, m in months])
        return list(month_dates)

    def run_with_df_instructions(self, instr_df: pd.DataFrame, month_is_relative: bool = True) -> Dict[str, pd.DataFrame]:
        if 'month' not in instr_df.columns or 'buy' not in instr_df.columns or 'sell' not in instr_df.columns:
            raise ValueError("instructions must include columns ['month','buy','sell']")

        month_dates = self._build_month_starts()

        trades = []
        nav_rows = []

        cash = self.initial_cash
        positions: Dict[str, float] = {}

        for _, row in instr_df.sort_values('month').iterrows():
            m = int(row['month'])
            idx = m - 1 if month_is_relative else int(m)
            if idx < 0 or idx >= len(month_dates):
                continue
            rebalance_day = month_dates[idx]

            # SELL
            sell_cell = str(row['sell']).strip() if pd.notna(row['sell']) else ''
            if sell_cell.upper() == 'ALL':
                for t, s in list(positions.items()):
                    if s <= 0:
                        continue
                    try:
                        d, p = self._first_price_on_or_after(t, rebalance_day)
                    except KeyError:
                        continue
                    amt = s * p
                    cash += amt
                    trades.append({'date': d, 'ticker': t, 'side': 'SELL', 'shares': s, 'price': p, 'amount': amt})
                    positions.pop(t, None)
            elif sell_cell != '':
                sell_list = [s.strip() for s in sell_cell.replace(';', ',').split(',') if s.strip()]
                for t in sell_list:
                    if t in positions and positions[t] > 0:
                        s = positions[t]
                        try:
                            d, p = self._first_price_on_or_after(t, rebalance_day)
                        except KeyError:
                            continue
                        amt = s * p
                        cash += amt
                        trades.append({'date': d, 'ticker': t, 'side': 'SELL', 'shares': s, 'price': p, 'amount': amt})
                        positions.pop(t, None)

            # BUY
            buy_cell = row['buy'] if pd.notna(row['buy']) else ''
            buy_alloc = self._parse_buy_cell(buy_cell)
            if buy_alloc:
                for t, w in buy_alloc.items():
                    if w <= 0:
                        continue
                    try:
                        d, p = self._first_price_on_or_after(t, rebalance_day)
                    except KeyError:
                        continue
                    invest = cash * w
                    shares = invest / p if p > 0 else 0.0
                    # allow fractional shares
                    shares = float(math.floor(shares * 10**6) / 10**6)
                    amt = shares * p
                    if shares <= 0:
                        continue
                    cash -= amt
                    positions[t] = positions.get(t, 0.0) + shares
                    trades.append({'date': d, 'ticker': t, 'side': 'BUY', 'shares': shares, 'price': p, 'amount': amt})

            # NAV sampling
            next_idx = idx + 1
            end_date = month_dates[next_idx] if next_idx < len(month_dates) else self.prices.index[-1]
            mask = (self.prices.index >= rebalance_day) & (self.prices.index < end_date)
            dates_span = self.prices.index[mask]
            for d in dates_span:
                total = cash
                for t, s in positions.items():
                    if t in self.prices.columns:
                        p = self.prices.at[d, t]
                        if pd.notna(p):
                            total += s * float(p)
                nav_rows.append({'date': d, 'nav': total})

        if not nav_rows:
            mask = self.prices.index >= self.start_date
            for d in self.prices.index[mask]:
                nav_rows.append({'date': d, 'nav': self.initial_cash})

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values('date').reset_index(drop=True)
        nav_df = pd.DataFrame(nav_rows).drop_duplicates(subset=['date']).set_index('date').sort_index()

        return {'trades': trades_df, 'nav': nav_df}


def load_instructions(path: str):
    spec = importlib.util.spec_from_file_location('bt_instructions', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'instructions_df'):
        raise RuntimeError(f"{path} must define `instructions_df` DataFrame with columns ['month','buy','sell']")
    return mod.instructions_df


def main():
    here = os.path.dirname(__file__)
    instr_path = os.path.join(here, 'backtester', 'instructions.py')
    if not os.path.exists(instr_path):
        raise FileNotFoundError(f"Instructions file not found: {instr_path}")
    instructions_df = load_instructions(instr_path)

    prices_csv = os.path.join(here, 'backtester', 'sample_prices.csv')
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(f"Prices CSV not found: {prices_csv}")
    prices = pd.read_csv(prices_csv, parse_dates=['date'], index_col='date')

    bt = MonthlyBacktester(prices, start_date='2022-02-01', initial_cash=1_000_000)
    res = bt.run_with_df_instructions(instructions_df)

    out_dir = os.path.join(here, 'backtester')
    res['trades'].to_csv(os.path.join(out_dir, 'trades.csv'), index=False)
    res['nav'].to_csv(os.path.join(out_dir, 'nav.csv'))
    print('Wrote trades.csv and nav.csv to backtester/')


if __name__ == '__main__':
    main()
