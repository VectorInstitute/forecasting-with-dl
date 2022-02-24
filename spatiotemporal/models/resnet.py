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
        lt_features: int,
        expand_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lt_features = lt_features
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

        self.scale_bias_cn = nn.Sequential(
            nn.Linear(self.lt_features, self.lt_features, bias=False),
            nn.Linear(self.lt_features, self.out_channels * 4, bias=False)
        )

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        s1, b1, s2, b2 = self.scale_bias_cn(scale_bias).view(x.shape[0], -1, 1, 1).chunk(4, dim=-3)
        identity = x
        
        x = self.relu(self.bn1(self.cn1(x) * s1 + b1))
        x = self.bn2(self.cn2(x) * s2 + b2)

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
        n_layers: int,
        lt_features: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lt_features = lt_features

        #TODO: Variable number of resnet layers
        self.initial_cn = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.in_channels,
                                    kernel_size=7,
                                    padding=3,)
        self.block1 = MultiInputSequential(
            Block(self.in_channels, self.in_channels * 2, self.lt_features, expand_residual=True),
            Block(self.in_channels * 2, self.in_channels * 2, self.lt_features)
        )
        self.block2 = MultiInputSequential(
            Block(self.in_channels * 2, self.in_channels * 4, self.lt_features, expand_residual=True),
            Block(self.in_channels * 4, self.in_channels * 4, self.lt_features)
        )
        self.block3 = MultiInputSequential(
            Block(self.in_channels * 4, self.in_channels * 8, self.lt_features, expand_residual=True),
            Block(self.in_channels * 8, self.in_channels * 8, self.lt_features)
        )
        self.block4 = MultiInputSequential(
            Block(self.in_channels * 8, self.in_channels * 8, self.lt_features, expand_residual=False),
            Block(self.in_channels * 8, self.in_channels * 8, self.lt_features)
        )
        self.final_cn = nn.Conv2d(in_channels=self.in_channels * 8,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  padding=0,)

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        x = self.initial_cn(x)
        x, _ = self.block1(x, scale_bias)
        x, _ = self.block2(x, scale_bias)
        x, _ = self.block3(x, scale_bias)
        x, _ = self.block4(x, scale_bias)
        out = self.final_cn(x)
        return out
