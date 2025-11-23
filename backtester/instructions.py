import pandas as pd

# 12-month sample instruction set (month=1 -> Feb 2022)
# Uses tickers: AAPL, MSFT, AMZN, GOOG (matches `backtester/sample_prices.csv`)
instructions_df = pd.DataFrame([
    {'month': 1, 'buy': 'AAPL,MSFT', 'sell': 'ALL'},
    {'month': 2, 'buy': 'AMZN,GOOG', 'sell': 'ALL'},
    {'month': 3, 'buy': 'AAPL:0.7;MSFT:0.3', 'sell': 'ALL'},
    {'month': 4, 'buy': 'AMZN:0.6;GOOG:0.4', 'sell': 'ALL'},
    {'month': 5, 'buy': 'AAPL,AMZN', 'sell': 'ALL'},
    {'month': 6, 'buy': 'MSFT,GOOG', 'sell': 'ALL'},
    {'month': 7, 'buy': 'AAPL:0.5;GOOG:0.5', 'sell': 'ALL'},
    {'month': 8, 'buy': 'MSFT:0.5;AMZN:0.5', 'sell': 'ALL'},
    {'month': 9, 'buy': 'GOOG,AAPL', 'sell': 'ALL'},
    {'month': 10, 'buy': 'MSFT:0.6;AMZN:0.4', 'sell': 'ALL'},
    {'month': 11, 'buy': 'AAPL,MSFT', 'sell': 'ALL'},
    {'month': 12, 'buy': 'AMZN,GOOG', 'sell': 'ALL'},
])



if __name__ == '__main__':
    print('instructions_df sample:')
    print(instructions_df)
