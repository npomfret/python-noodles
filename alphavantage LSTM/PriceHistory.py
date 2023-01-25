import numpy as np

from LSTMData import LSTMData
from plots import plot_raw_prices


def create_windowed_data(x, window_size):
    number_of_observations = x.shape[0]
    number_of_output_rows = number_of_observations - window_size + 1

    # make a 2D *view* of the 1D input array 'x',
    # each entry is of width 'window_size' and is effectively a shifted copy of the previous entry
    # eg, x = [1,2,3,4,5,6,7...] with a window size of 5 becomes....
    # windowed_data = [
    #     [1,2,3,4,5],
    #     [2,3,4,5,6],
    #     [3,4,5,6,7],
    #     ...
    # see: https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
    windowed_data = np.lib.stride_tricks.as_strided(
        x,
        shape=(number_of_output_rows, window_size),
        strides=(x.strides[0], x.strides[0]),
        writeable=False
    )

    return windowed_data


class Normalizer:
    def __init__(self):
        self.mean = None
        self.sd = None

    def fit_transform(self, x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        return (x - self.mean) / self.sd

    def inverse_transform(self, x):
        return (x * self.sd) + self.mean


class PriceHistory:
    def __init__(self, symbol, dates, prices):
        self.symbol = symbol
        self.dates = dates
        self.prices = prices

    def size(self):
        return len(self.dates)

    def first_date(self):
        return self.dates[0]

    def last_date(self):
        return self.dates[-1]

    def plot(self):
        plot_raw_prices(self)

    def to_lstm_data(self, split_ratio, window_size):
        scaler = Normalizer()
        normalized_prices = scaler.fit_transform(self.prices)

        data_x = create_windowed_data(normalized_prices, window_size)

        # discard the last row as we don't know what the 'y' is (because the 'y' for this row is in the future)
        data_x_unseen = data_x[-1]
        data_x = data_x[:-1]

        # we just use the next day as label, starting at index 'window_size'
        data_y = normalized_prices[window_size:]

        # sanity check that x and y are of equal length (after we dropped the last row from x above)
        if len(data_x) != len(data_y):
            raise ValueError('x and y are not same length')

        # split dataset: early stuff for training, later stuff for testing
        split_index = int(data_y.shape[0] * split_ratio)

        data_x_train = data_x[:split_index]
        data_x_test = data_x[split_index:]

        data_y_train = data_y[:split_index]
        data_y_test = data_y[split_index:]

        return LSTMData(
            split_index,
            data_x_train,
            data_x_test,
            data_y_train,
            data_y_test,
            scaler,
            data_x_unseen,
            window_size,
        )


