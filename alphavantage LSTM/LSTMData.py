from torch.utils.data import DataLoader

from TimeSeriesDataset import TimeSeriesDataset
from plots import plot_train_vs_test


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
        plot_train_vs_test(self, price_history)

    def unscale(self, data):
        return self.scaler.inverse_transform(data)