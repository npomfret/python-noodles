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

    print(f'Loaded {len(dates)} data points for {symbol}, from {dates[0]} to {dates[-1]}')
    plot_raw_prices(dates, adjusted_close_prices, symbol)

    return dates, adjusted_close_prices


def create_windowed_data(x, window_size):
    number_of_observations = x.shape[0]
    number_of_output_rows = number_of_observations - window_size + 1

    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
    windowed_data = np.lib.stride_tricks.as_strided(
        x,
        shape=(number_of_output_rows, window_size),
        strides=(x.strides[0], x.strides[0]),
        writeable=False
    )

    # 'output' is a 2D view of the 1D input array 'x',
    # each entry is of width 'window_size' and is effectively a shifted copy of the previous entry
    # [
    #     [1,2,3,4,5],
    #     [2,3,4,5,6],
    #     [3,4,5,6,7],
    #     ...

    data = windowed_data[:-1]
    row_with_no_tomorrow = windowed_data[-1]

    return data, row_with_no_tomorrow


def create_output_array(x, window_size):
    # use the next day as label
    output = x[window_size:]
    return output
