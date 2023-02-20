from torch import nn as nn, Tensor

LSTM_CONFIG = {
    "input_size": 1,  # since we are only using 1 feature, close price
    "num_lstm_layers": 2,
    "lstm_size": 32,
    "dropout": 0.2,
}


class LSTMModelDefinition(nn.Module):
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

    def forward(self, tensor: Tensor):
        batch_size = tensor.shape[0]

        # layer 1
        tensor_1 = self.linear_1(tensor)

        # apply activation function
        tensor_2 = self.relu(tensor_1)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(tensor_2)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        tensor_3 = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # layer 2
        tensor_4 = self.dropout(tensor_3)
        predictions = self.linear_2(tensor_4)
        return predictions[:, -1]
