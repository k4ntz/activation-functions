from .functions_new import ActivationModule
import torch.nn.functional as F
from torch import sin
from activations.utils.activation_logger import ActivationLogger


class ReLU(ActivationModule):
    logger = ActivationLogger("ReLU Logger")
    def __init__(self, *args, **kwargs):
        self.function = F.relu
        super().__init__(self.function, *args, **kwargs)

class LReLU(ActivationModule):
    logger = ActivationLogger("LReLU Logger")
    def __init__(self, *args, **kwargs):
        self.function = F.leaky_relu
        super().__init__(self.function, *args, **kwargs)

class Tanh(ActivationModule):
    logger = ActivationLogger("Tanh Logger")
    def __init__(self, device):
        self.function = F.tanh
        super().__init__(self.function, device)

class Sigmoid(ActivationModule):
    logger = ActivationLogger("Sigmoid Logger")
    def __init__(self, device):
        self.function = F.sigmoid
        super().__init__(self.function, device)

class GLU(ActivationModule):
    logger = ActivationLogger("GLU Logger")
    def __init__(self, device, dim=-1):
        self.function = F.glu
        super().__init__(self.function, device)

class OneSin(ActivationModule):
    logger = ActivationLogger("OneSin Logger")
    def __init__(self, *args, **kwargs):
        self.function = lambda x: (x+1>0).float() * (x-1<0).float() * sin(x*3.141592653589793)
        super().__init__(self.function, *args, **kwargs)
