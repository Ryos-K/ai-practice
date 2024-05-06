import numpy as np
from ..module import Module

class ReLU(Module):
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(0, X)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * (self.X > 0)