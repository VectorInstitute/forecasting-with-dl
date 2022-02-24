from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .multi_input_sequential import MultiInputSequential


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        h_dim: int,
        w_dim: int,
        dilation: int,
        scale_bias_index: int,
        lt_features: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels if scale_bias_index == 0 else hidden_channels
        self.hidden_channels = hidden_channels
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.dilation = dilation
        self.scale_bias_index = scale_bias_index
        self.lt_features = lt_features
        self.cn1 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=self.dilation,
                             dilation=(self.dilation, self.dilation),
                             bias=True)
        self.cn2 = nn.Conv2d(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=self.dilation,
                             dilation=(self.dilation, self.dilation),
                             bias=True)
        if scale_bias_index == 0:
            self.upsample_cn = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=self.hidden_channels,
                                         kernel_size=(1, 1),
                                         stride=1,
                                         padding=0,
                                         bias=True)
        self.scale_bias_cn = nn.Sequential(
            nn.Linear(self.lt_features, self.lt_features, bias=False),
            nn.Linear(self.lt_features, self.hidden_channels * 2, bias=False)
        )
        self.layer_norm = nn.LayerNorm(
            [self.hidden_channels, self.h_dim, self.w_dim], elementwise_affine=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tuple[Tensor, Tensor]:
        s, b = self.scale_bias_cn(scale_bias).view(x.shape[0], -1, 1, 1).chunk(2, dim=-3)
        identity = x
        x = self.cn1(x)
        x = self.layer_norm(x)
        x = x * s + b
        x = self.relu(x)
        x = self.cn2(x)
        x = self.layer_norm(x)
        x = x * s + b
        x = self.relu(x)
        if self.scale_bias_index == 0:
            identity = self.upsample_cn(identity)
        out = x + identity
        return out, scale_bias


class DilatedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dilations: list,
        h_dim: int,
        w_dim: int,
        lt_features: int,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.lt_features = lt_features
        self.upsample = upsample

        self.layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for i, dl in enumerate(self.dilations * 2):
            layers.append(
                DilatedBlock(self.in_channels, self.hidden_channels, self.h_dim, self.w_dim, dl, i, self.lt_features))
        return MultiInputSequential(*layers)

    def forward(self, x: Tensor, scale_bias: list) -> Tensor:
        out, _ = self.layers(x, scale_bias)
        return out
