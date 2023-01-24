import numpy as np
from alpha_vantage.timeseries import TimeSeries
from plots import plot_raw_prices

CONFIG = {
    "key": "YOUR_API_KEY",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
    "outputsize": "full",
    "key_adjusted_close": "5. adjusted close",
}


def download_price_history(symbol):
    api_key = CONFIG["key"]
    outputsize = CONFIG["outputsize"]

    ts = TimeSeries(key=api_key)

    json_data, *_ = ts.get_daily_adjusted(symbol, outputsize=outputsize)

    dates = list(json_data.keys())
    dates.reverse()

    close_col_name = CONFIG["key_adjusted_close"]
    adjusted_close_prices = [float(json_data[date][close_col_name]) for date in json_data.keys()]
    adjusted_close_prices.reverse()
    adjusted_close_prices = np.array(adjusted_close_prices)

    print(f'Number data points: {len(dates)} (from {dates[0]} to {dates[len(dates) - 1]})')
    plot_raw_prices(dates, adjusted_close_prices, symbol)

    return dates, adjusted_close_prices
