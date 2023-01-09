import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import requests_cache
import tabulate
import json

from io import StringIO
from tabulate import tabulate

from _secrets import IEX_CLOUD_API_TOKEN

print(f'token: {IEX_CLOUD_API_TOKEN}')

session = requests_cache.CachedSession('http_cache', backend='filesystem')
print(session.cache.cache_dir)

base_url = 'https://cloud.iexapis.com'

my_columns = ['Ticker', 'Stock Price', 'Market Capitalization', 'Number of Shares to Buy']
df = pd.DataFrame(columns=my_columns)

stocks = pd.read_csv('starter_files/sp_500_stocks.csv')

step = 100
all_symbols = stocks['Ticker']

for i in range(0, len(all_symbols), step):
    symbols = all_symbols[i:i + step]
    symbols_text = ','.join(symbols)
    url = f'{base_url}/stable/stock/market/batch/?symbols={symbols_text}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    response = session.get(url)

    if response.status_code == 404:
        continue

    data = response.json()

    for symbol in data:
        item = data[symbol]['quote']
        # print(json.dumps(item, indent=2))

        price = item['latestPrice']
        market_cap = item['marketCap']
        new_row = pd.DataFrame([[symbol, price, market_cap, 'N/A']], columns=my_columns)
        df = pd.concat([df, new_row], ignore_index=True)

# portfolio_size = input('enter the value of your portfolio')
# print(portfolio_size)
#
# try:
#     val = float(portfolio_size)
#     print(val)
# except ValueError:
#     print('please enter a number')

print(tabulate(df, headers='keys', tablefmt='psql'))
portfolio_size = 1000000
position_size = portfolio_size / len(df.index)

for i in range(0, len(df.index)):
    price = df.loc[i, 'Stock Price']
    df.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / price)

print(tabulate(df, headers='keys', tablefmt='psql'))

writer = pd.ExcelWriter('foo.xlsx', engine='xlsxwriter')
df.to_excel(writer, "foo", index=False)
bg_color = '#0a0a23'
font_color = '#ffffff'

string_format = writer.book.add_format({
    'font_color': font_color,
    'bg_color': bg_color,
    'border': 1
})
dollar_format = writer.book.add_format({
    'num_format': '$0.00',
    'font_color': font_color,
    'bg_color': bg_color,
    'border': 1
})
integer_format = writer.book.add_format({
    'num_format': 0,
    'font_color': font_color,
    'bg_color': bg_color,
    'border': 1
})

col_formats = {
    'A': ['Ticker', string_format],
    'B': ['Stock Price', dollar_format],
    'C': ['Market Capitalization', dollar_format],
    'D': ['Number of Shares to Buy', integer_format],
}

for col in col_formats.keys():
    f = col_formats[col][1]
    title = col_formats[col][0]
    writer.sheets['foo'].set_column(f'{col}:{col}', 18, f)
    writer.sheets['foo'].write(f'{col}1', title, string_format)

writer.save()

