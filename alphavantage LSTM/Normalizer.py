import numpy as np
from numpy import ndarray


class Normalizer:
    def __init__(self):
        self.mean = None
        self.sd = None

    def fit_transform(self, x: ndarray[float]) -> ndarray[float]:
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        return (x - self.mean) / self.sd

    def inverse_transform(self, x: ndarray[float]) -> ndarray[float]:
        return (x * self.sd) + self.mean
