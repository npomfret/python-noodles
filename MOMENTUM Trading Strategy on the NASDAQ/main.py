# see https://www.youtube.com/watch?v=bUejGzheCac

import io
import pandas as pd
import yfinance as yf
import numpy as np
import requests_cache
from matplotlib import pyplot as plt
from _secrets_ import EODHISTORICALDATA_API_KEY

session = requests_cache.CachedSession('cache/http-cache', backend='filesystem')
session.headers['User-agent'] = 'my-program/1.0'
print(session.cache.cache_dir)

def get_all_symbols(exchange_code):
    # exchange can be: 'US', NYSE', 'NASDAQ', 'BATS', 'OTCQB', 'PINK', 'OTCQX', 'OTCMKTS', 'NMFQS', 'NYSE MKT','OTCBB', 'OTCGREY', 'BATS', 'OTC'
    response = session.get(f'https://eodhistoricaldata.com/api/exchange-symbol-list/{exchange_code}?api_token={EODHISTORICALDATA_API_KEY}&delisted=1')
    print(response.status_code)

    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    return df['Code'].to_list()


tickers = get_all_symbols('NASDAQ')
print(tickers)

# ticker_df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
# tickers = ticker_df["Ticker"].to_list()
# print(tickers)

# print(si.tickers_nasdaq())
start_date = '2000-01-01'
cache_file = f'cache/dataframe.{start_date}.pkl'

try:
    df = pd.read_pickle(cache_file)
except:
    df = yf.download(tickers[0], start=start_date)['Adj Close']
    df.to_pickle(cache_file)
print(df)

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


returns = {}
for date in mtl.index[:-1]:
    returns[date] = pf_performance(date)

returns_series = pd.Series(returns)
print(returns_series)

# https://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
def max_drawdown(series):
    mdd = 0
    mdd_index = ''
    peak = series[0]
    for i in series.index:
        x = series[i]
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
            mdd_index = i
    return [mdd_index, mdd]


frame = returns_series.cumprod()

print("max draw down:")
print(max_drawdown(frame))

plt.plot(frame)
plt.ylabel('cumulative returns')
plt.xlabel('time')
plt.show()