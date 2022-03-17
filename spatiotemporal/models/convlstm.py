from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        lt_features: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lt_features = lt_features

        self.cns = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=self.hidden_channels * 4,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )

        self.scale_bias_fc = nn.Sequential(
            nn.Linear(self.lt_features, self.lt_features, bias=False),
            nn.Linear(self.lt_features, self.hidden_channels * 2, bias=False)
        )

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        """Forward pass for single convLSTM layer. Derived from: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py.
        Args:
            x (Tensor): Combination of input, hidden, and cell inputs
            scale_bias (Tensor): Scale and bias vectors from the master MLP
        """
        s, b = self.scale_bias_fc(scale_bias).view(x.shape[0], -1, 1, 1).chunk(2, dim=-3)

        x_and_hidden, cell = torch.split(
            x, self.in_channels + self.hidden_channels, dim=1
        )

        x_and_hidden = self.cns(x_and_hidden)

        xh_i, xh_f, xh_o, xh_c = torch.split(x_and_hidden, self.hidden_channels, dim=1)

        c_t = torch.sigmoid(xh_f * s + b) * cell + torch.sigmoid(xh_i * s + b) * torch.tanh(xh_c * s + b)
        h_t = torch.sigmoid(xh_o * s + b) * torch.tanh(c_t)

        return torch.cat((h_t, c_t), dim=-3)


class ConvLSTMEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        network_hidden_channels: list,
        lt_features: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.network_hidden_channels = network_hidden_channels
        self.lt_features = lt_features

        self.lstm_net = self._make_lstms()

    def _make_lstms(self) -> nn.ModuleList:
        """Create lstm layers
        Args:
        """
        all_in_channels = [self.in_channels, *self.network_hidden_channels[:-1]]
        lstms = nn.ModuleList(
            [
                ConvLSTMLayer(c, h, self.lt_features)
                for c, h in zip(all_in_channels, self.network_hidden_channels)
            ]
        )
        return lstms

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        """Forward pass through all lstm layers for all timesteps
        Args:
            x (Tensor): Context tensor containing all timesteps
            scale_bias (Tensor): Scale and bias vectors from the master MLP
        """
        B, T, _, H, W = x.shape
        new_states = [0] * len(self.network_hidden_channels)    # Record all the hidden states

        for t in range(T):
            for i, lstm in enumerate(self.lstm_net):
                x_in = (
                    x[:, t, :, :, :] if i == 0 else new_states[i - 1].chunk(2, dim=-3)[0]
                )
                hc = (
                    torch.zeros((B, self.network_hidden_channels[i] * 2, H, W)).type_as(x)
                    if t == 0
                    else new_states[i]
                )
                new_states[i] = lstm(torch.cat((x_in, hc), dim=-3), scale_bias)

        return new_states[-1].chunk(2, dim=-3)[0]
