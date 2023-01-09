import numpy as np
import pandas as pd
import requests
import math
from scipy import stats
import xlsxwriter
import requests_cache
import tabulate
import json
from statistics import mean

from io import StringIO
from tabulate import tabulate

session = requests_cache.CachedSession('http_cache', backend='filesystem')
print(session.cache.cache_dir)

base_url = 'https://cloud.iexapis.com'

stocks = pd.read_csv('starter_files/sp_500_stocks.csv')
all_symbols = stocks['Ticker']
step = 100

from _secrets import IEX_CLOUD_API_TOKEN

my_columns = ['Ticker', 'Price', 'Price-to-Earnings-Ratio', 'Number of Shares to Buy']
df = pd.DataFrame(columns=my_columns)

for i in range(0, len(all_symbols), step):
    symbols = all_symbols[i:i + step]
    symbols_text = ','.join(symbols)

    url = f'{base_url}/stable/stock/market/batch/?symbols={symbols_text}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    response = session.get(url)
    data = response.json()
    for symbol in data:
        quote = data[symbol]['quote']
        price = quote['latestPrice']
        pe_ratio = quote['peRatio']
        row = pd.DataFrame([[symbol, price, pe_ratio, 'N/A']], columns=my_columns)
        df = pd.concat([df, row], ignore_index=True)

df = df[df['Price-to-Earnings-Ratio'] > 0]
df.sort_values('Price-to-Earnings-Ratio', ascending=True, inplace=True)
df = df[:50]
df.reset_index(inplace=True, drop=True)

print(tabulate(df, headers='keys', tablefmt='psql'))

portfolio_size = 1000000
position_size = portfolio_size / len(df.index)

for row in df.index:
    price = df.loc[row, 'Price']
    df.loc[row, 'Number of Shares to Buy'] = math.floor(position_size / price)

print(tabulate(df, headers='keys', tablefmt='psql'))

# more 'realistic' approach...

for i in range(0, len(all_symbols), step):
    symbols = all_symbols[i:i + step]
    symbols_text = ','.join(symbols)

    url = f'{base_url}/stable/stock/market/batch/?symbols={symbols_text}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
    response = session.get(url)
    print(response.status_code)
    print(response.text)
    data = response.json()
    for symbol in data:
        quote = data[symbol]['quote']
        stats = data[symbol]['advanced-stats']
        pe_ratio = quote['peRatio']
        pb_ratio = np.Nan
        ps_ratio = np.Nan
        ev_to_ebitda = np.Nan
        ev_to_gross_profit = np.Nan
        print(f'${symbol}')
        print(json.dumps(quote, indent=2))
        print(json.dumps(stats, indent=2))
