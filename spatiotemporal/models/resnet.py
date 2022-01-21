from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .multi_input_sequential import MultiInputSequential


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_bias_index: int,
        norm: bool = True,
        expand_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_bias_index = scale_bias_index
        self.expand_residual = expand_residual

        self.cn1 = nn.Conv2d(self.in_channels, self.out_channels, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.cn2 = nn.Conv2d(self.out_channels, self.out_channels, (3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

        if expand_residual is True:
            self.expandres_cn = MultiInputSequential(
                nn.Conv2d(self.in_channels, self.out_channels, (1, 1), stride=1, padding=0),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        assert len(scale_bias) == 16
        identity = x
        scale_cn1 = scale_bias[self.scale_bias_index * 4]
        bias_cn1 = scale_bias[self.scale_bias_index * 4 + 1]
        scale_cn2 = scale_bias[self.scale_bias_index * 4 + 2]
        bias_cn2 = scale_bias[self.scale_bias_index * 4 + 3]
        
        x = self.relu(self.bn1(self.cn1(x) * scale_cn1 + bias_cn1))
        x = self.bn2(self.cn2(x) * scale_cn2 + bias_cn2)

        if self.expand_residual is True:
            identity = self.expandres_cn(identity)

        out = x + identity
        out = self.relu(out)
        return out, scale_bias



class ResNet10(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.initial_cn = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.in_channels,
                                    kernel_size=7,
                                    padding=3,)
        self.block1 = MultiInputSequential(
            Block(self.in_channels, self.in_channels * 2, 0, expand_residual=True),
            Block(self.in_channels * 2, self.in_channels * 2, 1)
        )
        self.block2 = MultiInputSequential(
            Block(self.in_channels * 2, self.in_channels * 2, 2, expand_residual=False),
            Block(self.in_channels * 2, self.in_channels * 2, 3)
        )
        self.final_cn = nn.Conv2d(in_channels=self.in_channels * 2,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  padding=0,)

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        assert len(scale_bias) == 16
        x = self.initial_cn(x)
        x, _ = self.block1(x, scale_bias)
        x, _ = self.block2(x, scale_bias)
        out = self.final_cn(x)
        return out
