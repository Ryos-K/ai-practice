class Module:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dY):
        raise NotImplementedError