from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.cns = nn.Conv2d(in_channels=self.in_channels + self.hidden_channels,
                             out_channels=self.hidden_channels * 4,
                             kernel_size=5,
                             stride=1,
                             padding=2,
                             bias=False)    # No bias, added by lead time stack

    def forward(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tuple[Tensor, Tensor]:
        assert x.shape[1] == self.in_channels + self.hidden_channels * 2

        x_and_hidden, cell = torch.split(x, self.in_channels + self.hidden_channels, dim=1)

        x_and_hidden = self.cns(x_and_hidden)

        xh_i, xh_f, xh_o, xh_c = torch.split(x_and_hidden, self.hidden_channels, dim=1)

        c_t = torch.sigmoid(xh_f * scale + bias ) * cell + torch.sigmoid(xh_i * scale + bias) * torch.tanh(xh_c * scale + bias)
        h_t = torch.sigmoid(xh_o * scale * bias) * torch.tanh(c_t)

        return torch.cat((h_t, c_t), dim=1)


class ConvLSTMEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        network_hidden_channels: list,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.network_hidden_channels = network_hidden_channels
        
        self.lstm_net = self._make_lstms()

    def _make_lstms(self) -> nn.ModuleList:
        all_in_channels = [self.in_channels, *self.network_hidden_channels[:-1]]
        lstms = nn.ModuleList([ConvLSTMLayer(c, h) for c, h in zip(all_in_channels, self.network_hidden_channels)])
        return lstms

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        B, T, _, H, W = x.shape
        new_states = [0] * len(self.network_hidden_channels)
        assert len(scale_bias) == len(self.lstm_net) * 2
        for t in range(T):
            for i, lstm in enumerate(self.lstm_net):
                scale = scale_bias[i * 2]
                bias = scale_bias[i * 2 + 1]
                initial_states = torch.zeros((B, self.network_hidden_channels[i] * 2, H, W))    #.to(x.device)
                initial_states = initial_states.type_as(x)
                x_in = x[:, t, :, :, :] if i == 0 else new_states[i - 1].chunk(2, dim=1)[0]
                hidden, cell = initial_states.chunk(2, dim=1) if t == 0 else new_states[i].chunk(2, dim=1)

                new_states[i] = lstm(torch.cat((x_in, hidden, cell), dim=1), scale, bias)
        output = []
        for layer_state in new_states:
            output.append(layer_state.chunk(2, dim=1)[0])
        return torch.cat(output, dim=1)
