import numpy as np
from ..module import Module

class Sigmoid(Module):
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * self.Y * (1 - self.Y)