import numpy as np
from nptyping import NDArray, Shape, Float


class Normalizer:
    def __init__(self):
        self._mean = None
        self._sd = None

    def fit_transform(self, x: NDArray) -> NDArray:
        assert len(x.shape) == 1

        self._mean = np.mean(x, axis=0, keepdims=True)
        self._sd = np.std(x, axis=0, keepdims=True)

        return (x - self._mean) / self._sd

    def inverse_transform(self, x: NDArray) -> NDArray:
        return (x * self._sd) + self._mean
