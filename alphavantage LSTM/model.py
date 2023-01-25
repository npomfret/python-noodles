import numpy as np
from numpy import ndarray
from torch import nn as nn, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Any

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


class LSTMModel:
    def __init__(self, model_definition: LSTMModelDefinition, hw_device: str, learning_rate: float, scheduler_step_size: int):
        super().__init__()

        self.model_def = model_definition.to(hw_device)
        self.hw_device = hw_device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model_definition.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step_size,
            gamma=0.1
        )

    def learn(self, number_of_epochs: int, training_dataloader: DataLoader, testing_dataloader: DataLoader) -> None:
        for epoch in range(number_of_epochs):
            loss_train, learning_rate_train = self.run_epoch(training_dataloader, is_training=True)
            loss_test, learning_rate_test = self.run_epoch(testing_dataloader)
            self.scheduler.step()

            print(f'Epoch[{epoch + 1}/{number_of_epochs}], training loss:{loss_train:.6f}, testing loss:{loss_test:.6f}, lr:{learning_rate_train:.6f}]')

        # Set the module in evaluation mode from here
        self.model_def.eval()

    def run_epoch(self, dataloader: DataLoader, is_training=False) -> Tuple[float, float]:
        epoch_loss = 0

        if is_training:
            self.model_def.train()
        else:
            self.model_def.eval()

        for idx, (x_tensor, y_tensor) in enumerate(dataloader):
            if is_training:
                self.optimizer.zero_grad()

            x_tensor = x_tensor.to(self.hw_device)
            y_tensor = y_tensor.to(self.hw_device)

            out_tensor: Tensor = self.model_def(x_tensor)
            loss = self.criterion(out_tensor.contiguous(), y_tensor.contiguous())

            if is_training:
                loss.backward()
                self.optimizer.step()

            batch_size = x_tensor.shape[0]
            epoch_loss += (loss.detach().item() / batch_size)

        learning_rate = self.scheduler.get_last_lr()[0]

        return epoch_loss, learning_rate

    def make_prediction(self, x_tensor: Tensor) -> ndarray[(Any, 1)]:
        self.model_def.eval()

        x_tensor = x_tensor.to(self.hw_device)
        out: Tensor = self.model_def(x_tensor)

        return out.cpu().detach().numpy()

    def make_predictions(self, data_loader) -> ndarray[(Any, 1)]:
        results: ndarray[(Any, 1)] = np.array([])

        for idx, (x_tensor, *_) in enumerate(data_loader):
            result: ndarray[(Any, 1)] = self.make_prediction(x_tensor)
            results = np.concatenate((results, result))

        return results
