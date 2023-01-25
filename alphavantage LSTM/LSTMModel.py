from typing import Tuple

import numpy as np
import torch
from nptyping import NDArray, Shape, Float
from torch import nn as nn, optim as optim, Tensor
from torch.utils.data import DataLoader

from LSTMModelDefinition import LSTMModelDefinition


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

    def make_prediction(self, x_tensor: Tensor) -> NDArray[Shape["*"], Float]:
        self.model_def.eval()

        x_tensor = x_tensor.to(self.hw_device)
        out: Tensor = self.model_def(x_tensor)

        return out.cpu().detach().numpy()

    def make_predictions(self, data_loader: DataLoader) -> NDArray[Shape["*"], Float]:
        results: NDArray[Shape["*"], Float] = np.array([])

        for idx, (x_tensor, *_) in enumerate(data_loader):
            result: NDArray[Shape["*"], Float] = self.make_prediction(x_tensor)
            results = np.concatenate((results, result))

        return results

    def convert_1d_row_to_tensor(self, x: NDArray[Shape["*"], Float]):
        return torch.tensor(x).float().to(self.hw_device).unsqueeze(0).unsqueeze(2)
