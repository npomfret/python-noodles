from torch import nn as nn

CONFIG = {
    "input_size": 1,  # since we are only using 1 feature, close price
    "num_lstm_layers": 2,
    "lstm_size": 32,
    "dropout": 0.2,
}


class LSTMModel(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()

        input_size = CONFIG["input_size"]
        hidden_layer_size = CONFIG["lstm_size"]
        num_layers = CONFIG["num_lstm_layers"]
        dropout = CONFIG["dropout"]

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
