import pandas as pd
import yfinance as yf
import numpy as np
import requests_cache
from matplotlib import pyplot as plt

# https://www.youtube.com/watch?v=bUejGzheCac

session = requests_cache.CachedSession('cache/yfinance.cache')
session.headers['User-agent'] = 'my-program/1.0'

ticker_df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
tickers = ticker_df["Ticker"].to_list()
print(tickers)

df = yf.download(tickers, start='2010-01-01')['Adj Close']

df = df.dropna(axis=1)
print(df)

dailyReturns = (df.pct_change() + 1)[1:]
print("daily:")
print(dailyReturns.index.dtype)
print(dailyReturns.index[0])
print(dailyReturns)

mtl = dailyReturns.resample('M').prod()
print("monthly:")
print(mtl)


def get_rolling_ret(dataFrame, n):
    return dataFrame.rolling(n).apply(np.prod)


ret_12 = get_rolling_ret(mtl, 12)
ret_6 = get_rolling_ret(mtl, 6)
ret_3 = get_rolling_ret(mtl, 3)


def get_top(date):
    top_50 = ret_12.loc[date].nlargest(50).index
    top_30 = ret_6.loc[date, top_50].nlargest(30).index
    return ret_3.loc[date, top_30].nlargest(10).index

def pf_performance(date):
    top_10 = get_top(date)
    portfolio = mtl.loc[date:, top_10][1:2]
    return portfolio.mean(axis=1).values[0]


print(pf_performance('2010-12-31'))

returns = []
for date in mtl.index[:-1]:
    returns.append(pf_performance(date))

print(returns)

returns_series = pd.Series(returns, index=mtl.index[1:])

frame = returns_series.cumprod()
plt.plot(frame)
plt.show()