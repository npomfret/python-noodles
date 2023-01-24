import numpy as np
from torch import nn as nn
from torch.utils.data import Dataset


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


LSTM_CONFIG = {
    "input_size": 1,  # since we are only using 1 feature, close price
    "num_lstm_layers": 2,
    "lstm_size": 32,
    "dropout": 0.2,
}


class LSTMModel(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()

        input_size = LSTM_CONFIG["input_size"]
        hidden_layer_size = LSTM_CONFIG["lstm_size"]
        num_layers = LSTM_CONFIG["num_lstm_layers"]
        dropout = LSTM_CONFIG["dropout"]

        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # layer 1
        tensor = self.linear_1(tensor)
        tensor = self.relu(tensor)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(tensor)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        tensor = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # layer 2
        tensor = self.dropout(tensor)
        predictions = self.linear_2(tensor)
        return predictions[:, -1]
