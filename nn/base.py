import numpy as np


class Tensor:
    def __init__(self, shape, x: np.array = None, dx: np.array = None):
        if x:
            self.x = np.array(x).reshape(shape)
        else:
            self.x = np.empty(shape=shape)
        if dx:
            self.dx = np.array(dx).reshape(shape)
        else:
            self.dx = np.empty(shape=shape)
