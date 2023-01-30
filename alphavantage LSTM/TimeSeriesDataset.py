import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Tuple, Any
from nptyping import NDArray, Shape, Float


class TimeSeriesDataset(Dataset):
    def __init__(self, x: NDArray[Shape["*"], Float], y: NDArray[Shape["*"], Float]):
        number_of_features = 1
        number_of_samples = len(x)
        if number_of_samples != len(y):
            raise ValueError('x and y are not same length')
        number_of_time_steps = len(x[0])

        # we need to convert `x` into [n_samples, n_steps, n_features] for LSTM
        # in our case, we have only 1 feature (the price)...
        x = np.expand_dims(x, 2)
        if x.shape != (number_of_samples, number_of_time_steps, number_of_features):
            raise ValueError('x is wrong shape for LSTM')

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[float, float]:
        return self.x[idx], self.y[idx]
