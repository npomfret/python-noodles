import numpy as np

from LSTMData import LSTMData
from TimeSeriesDataset import TimeSeriesDataset
from plots import plot_raw_prices


def create_windowed_data(x, window_size):
    number_of_observations = x.shape[0]
    number_of_output_rows = number_of_observations - window_size + 1

    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
    # make a 2D *view* of the 1D input array 'x',
    # each entry is of width 'window_size' and is effectively a shifted copy of the previous entry
    # eg, x = [1,2,3,4,5,6,7...] with a window size of 5 becomes....
    # windowed_data = [
    #     [1,2,3,4,5],
    #     [2,3,4,5,6],
    #     [3,4,5,6,7],
    #     ...
    windowed_data = np.lib.stride_tricks.as_strided(
        x,
        shape=(number_of_output_rows, window_size),
        strides=(x.strides[0], x.strides[0]),
        writeable=False
    )

    row_with_no_tomorrow = windowed_data[-1]
    windowed_data = windowed_data[:-1]

    return windowed_data, row_with_no_tomorrow


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

        data_x, data_x_unseen = create_windowed_data(normalized_prices, window_size)
        data_y = create_output_array(normalized_prices, window_size=window_size)

        if len(data_x) != len(data_y):
            raise ValueError('x and y are not same length')

        # split dataset, early stuff for training, later stuff for testing

        split_index = int(data_y.shape[0] * split_ratio)

        data_x_train = data_x[:split_index]
        data_x_test = data_x[split_index:]
        data_y_train = data_y[:split_index]
        data_y_test = data_y[split_index:]

        dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
        dataset_test = TimeSeriesDataset(data_x_test, data_y_test)

        print(f'Train data shape, x: {dataset_train.x.shape}, y: {dataset_train.y.shape}')
        print(f'Testing data shape, x: {dataset_test.x.shape}, y: {dataset_test.y.shape}')

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


