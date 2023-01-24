import numpy as np
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from plots import plot_raw_prices, plot_train_vs_test

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

    return PriceHistory(symbol, dates, adjusted_close_prices)


class LSTMData:
    def __init__(self, split_index, data_x_train, data_x_test, data_y_train, data_y_test, scaler, data_x_unseen, window_size):
        self.split_index = split_index
        self.data_x_train = data_x_train
        self.data_x_test = data_x_test
        self.data_y_train = data_y_train
        self.data_y_test = data_y_test
        self.scaler = scaler
        self.data_x_unseen = data_x_unseen
        self.window_size = window_size

    def training_dataloader(self, batch_size, shuffle=True):
        dataset = TimeSeriesDataset(self.data_x_train, self.data_y_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def testing_dataloader(self, batch_size, shuffle=True):
        dataset = TimeSeriesDataset(self.data_x_test, self.data_y_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def plot(self, price_history):
        plot_train_vs_test(
            self.window_size,
            self.split_index,
            self.scaler,
            self.data_y_train,
            self.data_y_test,
            price_history
        )


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
        plot_raw_prices(self.dates, self.prices, self.symbol)

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
