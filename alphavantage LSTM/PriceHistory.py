import numpy as np
from nptyping import assert_isinstance
from LSTMData import LSTMData
from Normalizer import Normalizer
from plots import plot_raw_prices
from typing import List, Any
from nptyping import NDArray, Shape, Float


def create_windowed_data(x: NDArray, window_size: int) -> NDArray:
    assert len(x.shape) == 1

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

    # sanity checks
    assert len(windowed_data.shape) == 2
    assert windowed_data.shape[0] == number_of_output_rows
    assert windowed_data.shape[1] == window_size
    assert_isinstance(windowed_data, NDArray[Shape[f"{number_of_output_rows}, {window_size}"], Float])
    assert_isinstance(windowed_data, NDArray)

    return windowed_data


class PriceHistory:
    def __init__(self, symbol: str, dates: List[str], prices: NDArray[Shape['*'], Float]):
        self.symbol = symbol
        self.dates = dates
        self.prices = prices
        assert len(self.dates) == len(self.prices)

    def size(self) -> int:
        return len(self.dates)

    def first_date(self) -> str:
        return self.dates[0]

    def last_date(self) -> str:
        return self.dates[-1]

    def plot(self) -> None:
        plot_raw_prices(self)

    def slice(self, from_x: int, to_x: int):
        new_dates = self.dates[from_x:to_x]
        new_prices = self.prices[from_x:to_x]
        return PriceHistory(self.symbol, new_dates, new_prices)

    def to_lstm_data(self, split_ratio: float, window_size: int) -> LSTMData:
        scaler = Normalizer()
        normalized_prices: NDArray = scaler.fit_transform(self.prices)

        data_x: NDArray = create_windowed_data(normalized_prices, window_size)

        # discard the last row as we don't know what the y value is (because it's in the future)
        data_x_unseen: NDArray = data_x[-1]
        data_x: NDArray = data_x[:-1]

        # sanity check
        assert_isinstance(data_x_unseen, NDArray)
        assert_isinstance(data_x_unseen, NDArray[Shape[f"{window_size}"], Float])

        # we just use the next day as label, starting at index 'window_size'
        data_y: NDArray = normalized_prices[window_size:]

        # sanity check that x and y are of equal length (after we dropped the last row from x above)
        if len(data_x) != len(data_y):
            raise ValueError('x and y are not same length')

        # split dataset: early stuff for training, later stuff for testing
        split_index: int = int(data_y.shape[0] * split_ratio)

        data_x_train: NDArray = data_x[:split_index]
        data_y_train: NDArray = data_y[:split_index]

        data_x_test: NDArray = data_x[split_index:]
        data_y_test: NDArray = data_y[split_index:]

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
