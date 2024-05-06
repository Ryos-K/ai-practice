import numpy as np
from ..module import Module

class Softmax(Module):
    def forward(self, X: np.ndarray) -> np.ndarray:
        maxs = np.max(X, axis=1, keepdims=True)
        exps = np.exp(X - maxs)
        self.Y = exps / np.sum(exps, axis=1, keepdims=True)
        return self.Y
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * self.Y * (1 - self.Y)