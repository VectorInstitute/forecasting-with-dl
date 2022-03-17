from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .multi_input_sequential import MultiInputSequential


class DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        h_dim: int,
        w_dim: int,
        dilation: int,
        lt_features: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.dilation = dilation
        self.lt_features = lt_features

        if self.in_channels != self.hidden_channels:
            self.expandres_cn = nn.Conv2d(
                self.in_channels, self.hidden_channels, (1, 1), stride=1, padding=0
            )

        self.cn1 = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            (3, 3),
            stride=1,
            padding=self.dilation,
            dilation=(self.dilation, self.dilation),
        )

        self.cn2 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            (3, 3),
            stride=1,
            padding=self.dilation,
            dilation=(self.dilation, self.dilation),
        )

        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(
            [self.hidden_channels, self.h_dim, self.w_dim], elementwise_affine=False
        )

        self.scale_bias_fc = nn.Sequential(
            nn.Linear(self.lt_features, self.lt_features, bias=False),
            nn.Linear(self.lt_features, self.hidden_channels * 4, bias=False),
        )

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass for a dilated convolution layer
        Args:
            x (Tensor): Input to be convolved
            scale_bias (Tensor): (1, 2D) tensor containing D-dimensional scale and bias vectors
        """
        s1, b1, s2, b2 = (
            self.scale_bias_fc(scale_bias).view(x.shape[0], -1, 1, 1).chunk(4, dim=-3)
        )

        identity = x

        out = self.relu(self.ln(self.cn1(x)) * s1 + b1)
        out = self.relu(self.ln(self.cn2(out)) * s2 + b2)

        if self.in_channels != self.hidden_channels:
            identity = self.expandres_cn(identity)

        result = out + identity

        return result, scale_bias


class DilatedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dilations: list,
        h_dim: int,
        w_dim: int,
        lt_features: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.lt_features = lt_features

        self.layers = self._make_layers()

    def _make_layers(self):
        """Create all dilated convolution layers
        Args:
        """
        layers = []
        for dilation in self.dilations:
            in_channels = self.in_channels if dilation == 1 else self.hidden_channels
            layers.append(
                DilatedBlock(
                    in_channels,
                    self.hidden_channels,
                    self.h_dim,
                    self.w_dim,
                    dilation,
                    self.lt_features,
                )
            )
        return MultiInputSequential(*layers)

    def forward(self, x: Tensor, scale_bias: list) -> Tensor:
        """Forward pass for dilated convolution stack
        Args:
            x (Tensor): Input to be convolved
            scale_bias (Tensor): Concatenated scale and bias vectors from the master MLP
        """
        out, scale_bias_out = self.layers(x, scale_bias)

        return out, scale_bias_out  # Return the final scale and bias for testing
