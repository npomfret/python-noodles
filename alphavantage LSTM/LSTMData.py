from numpy import ndarray
from torch.utils.data import DataLoader
from Normalizer import Normalizer
from TimeSeriesDataset import TimeSeriesDataset
from plots import plot_train_vs_test


class LSTMData:
    def __init__(self, split_index: int, data_x_train: ndarray[float], data_x_test: ndarray[float], data_y_train: ndarray[float], data_y_test: ndarray[float], scaler: Normalizer, data_x_unseen: ndarray[float], window_size: int):
        self.split_index = split_index
        self.data_x_train = data_x_train
        self.data_x_test = data_x_test
        self.data_y_train = data_y_train
        self.data_y_test = data_y_test
        self.scaler = scaler
        self.data_x_unseen = data_x_unseen
        self.window_size = window_size

    def training_dataloader(self, batch_size: int, shuffle=True) -> DataLoader:
        dataset = TimeSeriesDataset(self.data_x_train, self.data_y_train)
        print(f'Train data shape, x: {dataset.x.shape}, y: {dataset.y.shape}')

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def testing_dataloader(self, batch_size: int, shuffle=True) -> DataLoader:
        dataset = TimeSeriesDataset(self.data_x_test, self.data_y_test)
        print(f'Testing data shape, x: {dataset.x.shape}, y: {dataset.y.shape}')

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def plot(self, price_history) -> None:
        plot_train_vs_test(self, price_history)

    def unscale(self, data: ndarray[float]) -> ndarray[float]:
        return self.scaler.inverse_transform(data)