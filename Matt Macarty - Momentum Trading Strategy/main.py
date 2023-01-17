import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yfin

yfin.pdr_override()

gld = pdr.get_data_yahoo('GLD')
print(gld.head())

day = np.arange(1, len(gld) + 1)
print(day)

gld['day'] = day
gld.drop(columns=["Adj Close", "Volume"], inplace=True)

best_perf = -100000

for short in [10, 20, 30, 40, 60, 80]:
    for step in [20, 40, 60, 80, 100, 120, 140]:

        copy = gld[['day', 'Open', 'High', 'Low', 'Close']].copy()
        # print(copy.head())

        long = short + step

        short_label = f'{short}-day'
        long_label = f'{long}-day'

        # add the moving averages (shifted by 1 day so we don't peek into the future)
        copy[short_label] = copy['Close'].rolling(short).mean().shift()
        copy[long_label] = copy['Close'].rolling(long).mean().shift()

        # add trade signal column (1 or -1 to indicate long or short)
        copy['signal'] = np.where(copy[short_label] > copy[long_label], 1, 0)
        copy['signal'] = np.where(copy[short_label] < copy[long_label], -1, copy['signal'])
        copy.dropna(inplace=True)

        # add daily returns
        copy['return'] = np.log(copy['Close']).diff()

        # add our algo returns (same as daily returns, or the opposite if we are short)
        copy['system_return'] = copy['signal'] * copy['return']

        # add column to highlight our trading days
        copy['entry'] = copy['signal'].diff()

        performance = np.exp(copy['system_return']).cumprod()[-1] - 1

        if performance > best_perf:
            print(f'short: {short_label}, long: {long_label}, perf: {performance}')
            best_perf = performance

            # plt.rcParams['figure.figsize'] = 12, 6
            # plt.plot(copy.iloc[-252:]['Close'], label='GLD')
            # plt.plot(copy.iloc[-252:][short_label], label=short_label)
            # plt.plot(copy.iloc[-252:][long_label], label=long_label)
            # plt.plot(copy[-252:].loc[copy.entry == 2].index, copy[-252:][short_label][copy.entry == 2], '^', color='g', markersize=12)
            # plt.plot(copy[-252:].loc[copy.entry == -2].index, copy[-252:][long_label][copy.entry == -2], 'v', color='r', markersize=12)
            # plt.legend(loc=2)
            # plt.grid(True, alpha=.3)
            # plt.show()

            plt.plot(np.exp(copy['return']).cumprod(), label='Buy/Hold')
            plt.plot(np.exp(copy['system_return']).cumprod(), label=f'{short_label} / {long_label}')
            plt.legend(loc=2)
            plt.grid(True, alpha=.3)
            plt.show()
