import numpy as np
from ..module import Module


class PReLU(Module):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(self.alpha * X, X)
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * np.where(self.X > 0, 1, self.alpha)