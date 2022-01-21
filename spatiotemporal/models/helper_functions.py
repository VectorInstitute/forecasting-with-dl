import torch
import torch.nn as nn
from torch import Tensor

def down_shift(x: Tensor, d: int) -> Tensor:
    """Downshifts tensor x by displacement d
    """
    pad = nn.ZeroPad2d((0, 0, d, 0))
    x_pad = pad(x)[:, :, :-d, :]
    return x_pad
