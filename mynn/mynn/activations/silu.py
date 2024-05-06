import numpy as np
from ..module import Module

class SiLU(Module):
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.sigmoidX = 1 / (1 + np.exp(-X))
        self.Y = X * self.sigmoidX
        return self.Y
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        return dY * (self.sigmoidX + self.Y - self.Y * self.sigmoidX)
