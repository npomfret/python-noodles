import numpy as np
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset

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

    return PriceHistory(dates, adjusted_close_prices)


class PriceHistory:
    def __init__(self, dates, prices):
        self.dates = dates
        self.prices = prices

    def first_date(self):
        return self.dates[0]

    def last_date(self):
        return self.dates[-1]


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


class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        number_of_features = 1
        number_of_samples = len(x)
        if number_of_samples != len(y):
            raise ValueError('x and y are not same length')
        number_of_time_steps = len(x[0])

        # in our case, we have only 1 feature, so we need to convert `x` into [n_samples, n_steps, n_features] for LSTM
        x = np.expand_dims(x, 2)
        if x.shape != (number_of_samples, number_of_time_steps, number_of_features):
            raise ValueError('x is wrong shape for LSTM')

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

