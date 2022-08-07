from activations.utils.activation_logger import ActivationLogger
from .functions import ActivationModule
import torch.nn.functional as F
from torch import sin
import logging


class ReLU(ActivationModule):
    logger = logging.getLogger("ReLU Logger")
    def __init__(self, *args, **kwargs):
        self.function = F.relu
        super().__init__(self.function, *args, **kwargs)

class LReLU(ActivationModule):
    logger = logging.getLogger("LReLU Logger")
    def __init__(self, *args, **kwargs):
        self.function = F.leaky_relu
        super().__init__(self.function, *args, **kwargs)

class Tanh(ActivationModule):
    logger = logging.getLogger("Tanh Logger")
    def __init__(self, device):
        self.function = F.tanh
        super().__init__(self.function, device)

class Sigmoid(ActivationModule):
    logger = logging.getLogger("Sigmoid Logger")
    def __init__(self, device):
        self.function = F.sigmoid
        super().__init__(self.function, device)

class GLU(ActivationModule):
    logger = logging.getLogger("GLU Logger")
    def __init__(self, device, dim=-1):
        self.function = F.glu
        super().__init__(self.function, device)

class OneSin(ActivationModule):
    logger = logging.getLogger("OneSin Logger")
    def __init__(self, *args, **kwargs):
        self.function = lambda x: (x+1>0).float() * (x-1<0).float() * sin(x*3.141592653589793)
        super().__init__(self.function, *args, **kwargs)