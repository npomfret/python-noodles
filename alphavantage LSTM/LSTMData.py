from nptyping import NDArray, Shape, Float
from torch.utils.data import DataLoader
from Normalizer import Normalizer
from TimeSeriesDataset import TimeSeriesDataset
from plots import plot_train_vs_test


class LSTMData:
    def __init__(self,
                 split_index: int,
                 data_x_train: NDArray,
                 data_x_test: NDArray,
                 data_y_train: NDArray,
                 data_y_test: NDArray,
                 scaler: Normalizer,
                 data_x_unseen: NDArray,
                 window_size: int):
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
        print(f'Training data shape, x: {dataset.x.shape}, y: {dataset.y.shape}')

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def testing_dataloader(self, batch_size: int, shuffle=True) -> DataLoader:
        dataset = TimeSeriesDataset(self.data_x_test, self.data_y_test)
        print(f'Testing data shape, x: {dataset.x.shape}, y: {dataset.y.shape}')

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def plot(self, price_history) -> None:
        plot_train_vs_test(price_history, self)

    def unscale(self, data: NDArray) -> NDArray:
        return self.scaler.inverse_transform(data)

    def print_summary(self):
        print(self.data_x_train[:3])
