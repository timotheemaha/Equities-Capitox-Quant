"""Deprecated: replaced by `backtester_single.py`.

This file is intentionally left as a harmless placeholder.
"""

print('backtester/monthly_rebalance.py is deprecated. Use backtester_single.py at repo root.')
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional


class MonthlyBacktester:
    """Simple monthly rebalancing backtester.

    Behavior:
    - Accepts a price DataFrame indexed by trading dates (DatetimeIndex) with tickers as columns.
    - Accepts an instructions DataFrame with columns: `date`, `ticker`, `weight` (weights sum to 1 per date),
      or a `strategy_fn` that returns the same for each rebalance date.
    - On each rebalance date we: liquidate all positions at the next available prices, then allocate cash
      according to the provided weights (integer shares, floor allocation).
    - Produces a trade log and daily NAV series.
    """

    def __init__(self, prices: pd.DataFrame, initial_cash: float = 1_000_000.0, commission: float = 0.0):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must be a DataFrame indexed by pd.DatetimeIndex")
        self.prices = prices.sort_index()
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)

        # internal state
        self._snapshots: List[Tuple[pd.Timestamp, Dict[str, int], float]] = []
        # each snapshot is (effective_date, positions dict, cash)
        # initial snapshot before any trades
        self._snapshots.append((self.prices.index[0] - pd.Timedelta(days=1), {}, self.initial_cash))
        self.trades: List[Dict] = []

    @staticmethod
    def _ensure_instructions_df(instructions: pd.DataFrame) -> pd.DataFrame:
        if not {'date', 'ticker', 'weight'}.issubset(instructions.columns):
            raise ValueError("instructions must contain 'date','ticker','weight' columns")
        df = instructions.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _first_price_on_or_after(self, ticker: str, date: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
        if ticker not in self.prices.columns:
            return None
        series = self.prices[ticker].dropna()
        if series.empty:
            return None
        idx = series.index.searchsorted(date)
        if idx >= len(series):
            return None
        ts = series.index[idx]
        return ts, float(series.iloc[idx])

    def _last_price_on_or_before(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if ticker not in self.prices.columns:
            return None
        series = self.prices[ticker].loc[:date].dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    def _portfolio_value_with_snapshot(self, snapshot: Tuple[pd.Timestamp, Dict[str, int], float], as_of: pd.Timestamp) -> float:
        _, positions, cash = snapshot
        pv = float(cash)
        for t, shares in positions.items():
            price = self._last_price_on_or_before(t, as_of)
            if price is not None:
                pv += shares * price
        return pv

    def run_with_instructions(self, instructions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run backtest given an instructions DataFrame.

        Returns (trades_df, nav_df)
        """
        instructions = self._ensure_instructions_df(instructions)

        # Group by date and apply rebalances in chronological order
        for date, group in instructions.groupby('date'):
            rebalance_date = pd.Timestamp(date)

            # compute current portfolio value as of rebalance_date using last snapshot
            last_snapshot = self._snapshots[-1]
            pv = self._portfolio_value_with_snapshot(last_snapshot, rebalance_date)

            # Liquidate at next available prices: we'll use, for simplicity, the first available price on or after rebalance_date for each ticker
            # Cash after liquidation equals pv (we ignore per-trade commissions on liquidation to keep example simple)
            cash = pv
            self.trades.append({'date': rebalance_date, 'action': 'LIQUIDATE', 'value': pv})

            # Start new positions empty
            new_positions: Dict[str, int] = {}

            # Allocate according to weights
            for _, row in group.iterrows():
                ticker = row['ticker']
                weight = float(row['weight'])
                if weight <= 0:
                    continue
                price_pair = self._first_price_on_or_after(ticker, rebalance_date)
                if price_pair is None:
                    # no available price — skip
                    self.trades.append({'date': rebalance_date, 'action': 'SKIP', 'ticker': ticker, 'reason': 'no_price'})
                    continue
                exec_date, price = price_pair
                allocation = weight * cash
                shares = int(allocation // price)
                cost = shares * price
                if shares <= 0:
                    self.trades.append({'date': exec_date, 'action': 'NOOP', 'ticker': ticker, 'shares': 0, 'price': price})
                    continue
                new_positions[ticker] = new_positions.get(ticker, 0) + shares
                cash -= cost + self.commission
                self.trades.append({'date': exec_date, 'action': 'BUY', 'ticker': ticker, 'shares': shares, 'price': price, 'cost': cost})

            # Save snapshot effective at rebalance_date (we consider trades done at rebalance_date / exec_date)
            self._snapshots.append((rebalance_date, new_positions, cash))

        # After processing all rebalances, build daily NAV using snapshots
        nav = self._build_nav_series()
        trades_df = pd.DataFrame(self.trades)
        return trades_df, nav

    def run_with_strategy(self, strategy_fn: Callable[[pd.Timestamp, pd.DataFrame], pd.DataFrame], start: pd.Timestamp = None, end: pd.Timestamp = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run backtest by calling `strategy_fn` for each monthly rebalance date.

        `strategy_fn(rebalance_date, prices)` must return a DataFrame with columns `ticker` and `weight` (weights summing to 1).
        """
        # determine date range
        idx = self.prices.index
        if start is None:
            start = idx[0]
        if end is None:
            end = idx[-1]

        # choose rebalance dates: first trading day of each month in range
        months = pd.date_range(start=start, end=end, freq='MS')
        instructions_rows = []
        for m in months:
            # find first trading day on or after month start
            pos = idx.searchsorted(m)
            if pos >= len(idx):
                continue
            reb_date = idx[pos]
            alloc = strategy_fn(reb_date, self.prices)
            # normalize/validate alloc
            if alloc is None or alloc.empty:
                continue
            alloc = alloc.copy()
            if 'ticker' not in alloc.columns or 'weight' not in alloc.columns:
                raise ValueError('strategy_fn must return DataFrame with columns [ticker,weight]')
            alloc['date'] = reb_date
            instructions_rows.append(alloc[['date', 'ticker', 'weight']])

        if not instructions_rows:
            return pd.DataFrame(), pd.DataFrame()
        instructions = pd.concat(instructions_rows, ignore_index=True)
        return self.run_with_instructions(instructions)

    def _build_nav_series(self) -> pd.DataFrame:
        # For each trading date, find latest snapshot with effective_date <= date
        dates = self.prices.index
        nav_list = []
        # precompute snapshots sorted by effective_date
        snapshots_sorted = sorted(self._snapshots, key=lambda x: x[0])
        snap_idx = 0
        for dt in dates:
            # move snap_idx to last snapshot <= dt
            while snap_idx + 1 < len(snapshots_sorted) and snapshots_sorted[snap_idx + 1][0] <= dt:
                snap_idx += 1
            _, positions, cash = snapshots_sorted[snap_idx]
            pv = float(cash)
            for t, shares in positions.items():
                # try exact price at dt, otherwise last price before dt
                price = None
                if t in self.prices.columns:
                    price = self.prices.at[dt, t] if pd.notna(self.prices.at[dt, t]) else None
                    if price is None:
                        price = self._last_price_on_or_before(t, dt)
                if price is not None:
                    pv += shares * price
            nav_list.append(pv)
        return pd.DataFrame({'portfolio_value': nav_list}, index=dates)

    @staticmethod
    def _parse_cell(cell):
        """Parse a buy/sell cell value from an instructions DF.

        Supported formats:
        - list/tuple of strings: returned as-is
        - string of comma/semicolon-separated tickers: 'AAPL,MSFT' -> ['AAPL','MSFT']
        - string with weights: 'AAPL:0.6;MSFT:0.4' -> [('AAPL',0.6),('MSFT',0.4)]
        - the string 'ALL' (case-insensitive) is returned as 'ALL'
        - NaN/None returns []
        """
        if cell is None:
            return []
        if isinstance(cell, (list, tuple)):
            return list(cell)
        if isinstance(cell, float) and np.isnan(cell):
            return []
        s = str(cell).strip()
        if not s:
            return []
        if s.upper() == 'ALL':
            return 'ALL'
        # if contains ':' treat as ticker:weight pairs separated by ; or ,
        sep_items = [it for it in re.split('[;,]', s) if it.strip()]
        items = []
        has_weights = any(':' in it for it in sep_items)
        if has_weights:
            for it in sep_items:
                if ':' not in it:
                    continue
                t, w = it.split(':', 1)
                try:
                    items.append((t.strip(), float(w)))
                except Exception:
                    items.append((t.strip(), None))
            return items
        # otherwise return list of tickers
        return [it.strip() for it in sep_items]

    def run_with_df_instructions(self, instr_df: pd.DataFrame, month_is_relative: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run backtest using a DataFrame with columns [month, buy, sell].

        - `month` is an integer index (1-based) representing the rebalance number relative to the start
          (first trading month = 1). If `month_is_relative` is False, the column is interpreted as calendar
          month numbers (1-12) and the rebalance will be executed on each matching month in the date range.
        - `buy` can be a list, comma-separated tickers, or 'T:W;T2:W2' weight pairs. If no weights provided,
          equal-weight allocation is used among tickers.
        - `sell` can be a list, comma-separated tickers, or 'ALL'.

        Returns (trades_df, nav_df).
        """
        import re

        df = instr_df.copy()
        if df.shape[1] < 3:
            raise ValueError('instr_df must have at least 3 columns: month,buy,sell')
        df = df.rename(columns={df.columns[0]: 'month', df.columns[1]: 'buy', df.columns[2]: 'sell'})

        # build rebalance dates (first trading day of each month)
        idx = self.prices.index
        months = pd.date_range(start=idx[0], end=idx[-1], freq='MS')
        rebalance_dates = []
        for m in months:
            pos = idx.searchsorted(m)
            if pos < len(idx):
                rebalance_dates.append(idx[pos])

        # map month index to date
        def month_to_date(m:int) -> Optional[pd.Timestamp]:
            if month_is_relative:
                if m <= 0 or m > len(rebalance_dates):
                    return None
                return rebalance_dates[m-1]
            else:
                # calendar months: choose first rebalance date matching month number
                for d in rebalance_dates:
                    if d.month == int(m):
                        return d
                return None

        # iterate rows ordered by month index
        rows = df.sort_values('month').itertuples(index=False)
        for row in rows:
            m = int(row.month)
            reb_date = month_to_date(m)
            if reb_date is None:
                self.trades.append({'date': pd.NaT, 'action': 'SKIP_MONTH', 'month': m, 'reason': 'no_rebalance_date'})
                continue

            # start from last snapshot
            last_snap_date, last_positions, last_cash = self._snapshots[-1]
            positions = dict(last_positions)  # copy

            # compute current portfolio value to determine available cash if full liquidation requested
            pv = self._portfolio_value_with_snapshot((last_snap_date, positions, last_cash), reb_date)

            # handle sells
            sell_cell = row.sell
            parsed_sell = self._parse_cell(sell_cell)
            cash = last_cash
            # if 'ALL' -> liquidate entire portfolio
            if parsed_sell == 'ALL':
                # add proceeds of liquidation using first available prices on/after reb_date where available
                for t, shares in list(positions.items()):
                    if shares <= 0:
                        continue
                    price_pair = self._first_price_on_or_after(t, reb_date)
                    if price_pair is None:
                        continue
                    exec_date, price = price_pair
                    proceeds = shares * price
                    cash += proceeds - self.commission
                    self.trades.append({'date': exec_date, 'action': 'SELL', 'ticker': t, 'shares': shares, 'price': price, 'proceeds': proceeds})
                    positions[t] = 0
                # remove zero positions
                positions = {k:v for k,v in positions.items() if v>0}
            else:
                # sell specified tickers only
                sell_list = parsed_sell if isinstance(parsed_sell, list) else []
                for t in sell_list:
                    if t not in positions or positions.get(t,0) <= 0:
                        continue
                    shares = positions.get(t,0)
                    price_pair = self._first_price_on_or_after(t, reb_date)
                    if price_pair is None:
                        continue
                    exec_date, price = price_pair
                    proceeds = shares * price
                    cash += proceeds - self.commission
                    self.trades.append({'date': exec_date, 'action': 'SELL', 'ticker': t, 'shares': shares, 'price': price, 'proceeds': proceeds})
                    positions[t] = 0
                positions = {k:v for k,v in positions.items() if v>0}

            # if sells were empty, keep last_cash (no change)

            # handle buys
            buy_cell = row.buy
            parsed_buy = self._parse_cell(buy_cell)
            buy_items: List[Tuple[str, Optional[float]]] = []
            if parsed_buy == []:
                # nothing to buy
                pass
            elif isinstance(parsed_buy, list) and parsed_buy and isinstance(parsed_buy[0], tuple):
                # [('AAPL',0.5), ...]
                buy_items = parsed_buy
            else:
                # list of tickers -> equal weight
                tickers = parsed_buy if isinstance(parsed_buy, list) else [parsed_buy]
                tickers = [t for t in tickers if t]
                if tickers:
                    w = 1.0 / len(tickers)
                    buy_items = [(t, w) for t in tickers]

            # allocate cash for buys
            for t, w in buy_items:
                if w is None:
                    # treat as equal-weight if weight missing
                    # compute remaining equal weight among unspecified? For simplicity, skip
                    continue
                price_pair = self._first_price_on_or_after(t, reb_date)
                if price_pair is None:
                    self.trades.append({'date': reb_date, 'action': 'SKIP_BUY', 'ticker': t, 'reason': 'no_price'})
                    continue
                exec_date, price = price_pair
                allocation = w * cash
                shares = int(allocation // price)
                if shares <= 0:
                    self.trades.append({'date': exec_date, 'action': 'NOOP_BUY', 'ticker': t, 'shares': 0, 'price': price})
                    continue
                cost = shares * price
                positions[t] = positions.get(t, 0) + shares
                cash -= cost + self.commission
                self.trades.append({'date': exec_date, 'action': 'BUY', 'ticker': t, 'shares': shares, 'price': price, 'cost': cost})

            # save snapshot effective at rebalance date
            self._snapshots.append((reb_date, positions, cash))

        nav = self._build_nav_series()
        trades_df = pd.DataFrame(self.trades)
        return trades_df, nav


def load_prices_from_csv(path: str, date_col: str = 'date') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def example_strategy_from_csv(path: str) -> Callable[[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Return a strategy function that reads allocations from a CSV with columns date,ticker,weight.

    The returned function ignores its `prices` param and simply returns allocations for the requested rebalance date.
    """
    allocs = pd.read_csv(path)
    allocs['date'] = pd.to_datetime(allocs['date'])

    def strat(rebalance_date: pd.Timestamp, prices: pd.DataFrame) -> pd.DataFrame:
        df = allocs.loc[allocs['date'] == pd.Timestamp(rebalance_date)]
        return df[['ticker', 'weight']].reset_index(drop=True)

    return strat


if __name__ == '__main__':
    # quick demo (requires real price CSV and optional instructions CSV)
    import argparse

    parser = argparse.ArgumentParser(description='Monthly liquidation + rebalance backtester demo')
    parser.add_argument('--prices', help='CSV file with date column named "date" and tickers as other columns', required=True)
    parser.add_argument('--instructions', help='CSV file with columns date,ticker,weight', required=False)
    args = parser.parse_args()

    prices = load_prices_from_csv(args.prices)
    bt = MonthlyBacktester(prices)
    if args.instructions:
        instr = pd.read_csv(args.instructions)
        trades, nav = bt.run_with_instructions(instr)
    else:
        print('No instructions provided. Exiting.')
        raise SystemExit(0)

    print('Trades:')
    print(trades.head(50).to_string(index=False))
    print('\nNAV tail:')
    print(nav.tail())
