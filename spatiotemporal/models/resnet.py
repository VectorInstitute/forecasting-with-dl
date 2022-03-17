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
        h_dim: int,
        w_dim: int,
        lt_features: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.lt_features = lt_features

        if self.in_channels != self.out_channels:  # Convolve residual to match channels
            self.expandres_cn = MultiInputSequential(
                nn.Conv2d(
                    self.in_channels, self.out_channels, (1, 1), stride=1, padding=0
                ),
                nn.LayerNorm([self.out_channels, self.h_dim, self.w_dim]),
            )

        self.cn1 = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), stride=1, padding=1
        )
        self.cn2 = nn.Conv2d(
            self.out_channels, self.out_channels, (3, 3), stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(
            [self.out_channels, self.h_dim, self.w_dim], elementwise_affine=False
        )

        self.scale_bias_fc = nn.Sequential(
            nn.Linear(self.lt_features, self.lt_features, bias=False),
            nn.Linear(self.lt_features, self.out_channels * 4, bias=False),
        )

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through one ResNet block
        Args:
            x (Tensor): Input tensor
            scale_bias (Tensor): Scale and bias vectors from the master MLP
        """
        s1, b1, s2, b2 = (
            self.scale_bias_fc(scale_bias).view(x.shape[0], -1, 1, 1).chunk(4, dim=-3)
        )

        identity = x

        out = self.relu(self.ln(self.cn1(x)) * s1 + b1)
        out = self.relu(self.ln(self.cn2(out)) * s2 + b2)

        if self.in_channels != self.out_channels:
            identity = self.expandres_cn(identity)

        result = out + identity

        return result, scale_bias


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        h_dim: int,
        w_dim: int,
        n_layers: int,
        lt_features: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.h_dim = h_dim
        self.w_dim = w_dim
        if n_layers % 4 != 0:
            raise ValueError("Minimum resblock size is 4")
        self.n_layers = n_layers
        self.lt_features = lt_features

        self.initial_cn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            padding=3,
        )

        self.res_layers = self._make_layers()

        self.final_cn = nn.Conv2d(
            in_channels=self.hidden_channels * 2 ** ((self.n_layers // 4) - 1),
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
        )

    def _make_layers(self) -> MultiInputSequential:
        """Make the res blocks for the core resnet, excluding the inital and final conv layers.
        Args:
        """
        layers = []
        for i in range(self.n_layers // 4):
            in_channels = (
                self.in_channels if i == 0 else 2 ** (i - 1) * self.hidden_channels
            )
            hidden_channels = 2**i * self.hidden_channels
            res_block1 = MultiInputSequential(  # 4 covn2d layers in each res_block
                Block(
                    in_channels,
                    hidden_channels,
                    self.h_dim,
                    self.w_dim,
                    self.lt_features,
                ),
                Block(
                    hidden_channels,
                    hidden_channels,
                    self.h_dim,
                    self.w_dim,
                    self.lt_features,
                ),
            )
            layers.append(res_block1)
        return MultiInputSequential(*layers)

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        """Forward pass through ResNet stack
        Args:
            x (Tensor): Input tensor
            scale_bias: Scale and bias vectors from the master MLP
        """
        out = self.initial_cn(x)
        out, scale_bias_out = self.res_layers(out, scale_bias)
        out = self.final_cn(out)

        return out, scale_bias_out  # Return the final scale and bias for testing
