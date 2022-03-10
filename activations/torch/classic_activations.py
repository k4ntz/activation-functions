from .functions import ActivationModule
import torch.nn.functional as F
from torch import sin


class ReLU(ActivationModule):
    def __init__(self, *args, **kwargs):
        function = F.relu
        self.function = function
        super().__init__(function, *args, **kwargs)

class LReLU(ActivationModule):
    def __init__(self, *args, **kwargs):
        function = F.leaky_relu
        super().__init__(function, *args, **kwargs)

class Tanh(ActivationModule):
    def __init__(self, device):
        function = F.tanh
        super().__init__(function, device)

class Sigmoid(ActivationModule):
    def __init__(self, device):
        function = F.sigmoid
        super().__init__(function, device)

class GLU(ActivationModule):
    def __init__(self, device, dim=-1):
        function = F.glu
        super().__init__(function, device)

class OneSin(ActivationModule):
    def __init__(self, *args, **kwargs):
        function = lambda x: (x+1>0).float() * (x-1<0).float() * sin(x*3.141592653589793)
        super().__init__(function, *args, **kwargs)
