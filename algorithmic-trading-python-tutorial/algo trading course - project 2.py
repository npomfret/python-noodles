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

from _secrets import IEX_CLOUD_API_TOKEN

step = 100
all_symbols = stocks['Ticker']

my_columns = ['Ticker', 'Stock Price', 'One Year Price Return', 'Number of Shares to Buy'];
df = pd.DataFrame(columns=my_columns)

for i in range(0, len(all_symbols), step):
    symbols = all_symbols[i:i + step]
    symbols_text = ','.join(symbols)
    url = f'{base_url}/stable/stock/market/batch/?symbols={symbols_text}&types=stats,quote&token={IEX_CLOUD_API_TOKEN}'
    response = session.get(url)
    # check for response.status == 200?

    data = response.json()
    for symbol in data:
        change = data[symbol]['stats']['year1ChangePercent']
        price = data[symbol]['quote']['latestPrice']
        row = pd.DataFrame([[symbol, price, change, 'N/A']], columns=my_columns)
        df = pd.concat([df, row], ignore_index=True)

df.sort_values('One Year Price Return', ascending=False, inplace=True)
df = df[:50]
df.reset_index(inplace=True)

portfolio_size = 1000000
position_size = portfolio_size / len(df.index)

for i in range(0, len(df)):
    price = df.loc[i, 'Stock Price']
    totalReturn = df.loc[i, 'One Year Price Return']
    df.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / price)

print(tabulate(df, headers='keys', tablefmt='psql'))

hqm_columns = [
    'Ticker',
    'Price',
    'Number of Shares to Buy',
    'One-Year Price Return',
    'One-Year Return Percentile',
    'Six-Month Price Return',
    'Six-Month Return Percentile',
    'Three-Month Price Return',
    'Three-Month Return Percentile',
    'One-Month Price Return',
    'One-Month Return Percentile',
    'HQM Score'
]

hqm_df = pd.DataFrame(columns=hqm_columns)

for i in range(0, len(all_symbols), step):
    symbols = all_symbols[i:i + step]
    symbols_text = ','.join(symbols)
    url = f'{base_url}/stable/stock/market/batch/?symbols={symbols_text}&types=stats,quote&token={IEX_CLOUD_API_TOKEN}'
    response = session.get(url)
    # check for response.status == 200?

    data = response.json()
    for symbol in data:
        quoteItem = data[symbol]['quote']
        statsItem = data[symbol]['stats']
        # print(json.dumps(stats, indent=2))
        row = [
            symbol,
            quoteItem['latestPrice'],
            'N/A',
            statsItem['year1ChangePercent'],
            'N/A',
            statsItem['month6ChangePercent'],
            'N/A',
            statsItem['month3ChangePercent'],
            'N/A',
            statsItem['month1ChangePercent'],
            'N/A',
            'N/A',
        ]
        hqm_df = pd.concat([hqm_df, (pd.DataFrame([row], columns=hqm_columns))], ignore_index=True)

time_periods = [
    'One-Year',
    'Six-Month',
    'Three-Month',
    'One-Month',
]

# print(tabulate(hqm_df, headers='keys', tablefmt='psql'))

hqm_df.dropna(inplace=True)

for row in hqm_df.index:
    for time_period in time_periods:
        input_col_name = f'{time_period} Price Return'
        input_col = hqm_df[input_col_name]
        input_val = hqm_df.loc[row, input_col_name]
        return_percentile = stats.percentileofscore(input_col, input_val)

        output_col_name = f'{time_period} Return Percentile'
        hqm_df.loc[row, output_col_name] = return_percentile

print(tabulate(hqm_df, headers='keys', tablefmt='psql'))

for row in hqm_df.index:
    momentum_percentiles = []
    for time_period in time_periods:
        momentum_percentiles.append(hqm_df.loc[row, f'{time_period} Return Percentile'])
    hqm_df.loc[row, 'HQM Score'] = mean(momentum_percentiles)

hqm_df.sort_values('HQM Score', ascending=False, inplace=True)
hqm_df = hqm_df[:50]
hqm_df.reset_index(inplace=True, drop=True)

print(tabulate(hqm_df, headers='keys', tablefmt='psql'))

portfolio_size = 1000000
position_size = portfolio_size / len(hqm_df.index)
print(position_size)

for i in hqm_df.index:
    hqm_df.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_df.loc[i, 'Price'])

print(tabulate(hqm_df, headers='keys', tablefmt='psql'))
