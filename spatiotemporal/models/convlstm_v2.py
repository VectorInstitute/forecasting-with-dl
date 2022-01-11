import os
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cell_channels: int,
    ) -> None:
        super().__init__()
        self.cell_channels = cell_channels
        self.in_channels = in_channels

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.cns = nn.Conv2d(in_channels=self.in_channels + self.cell_channels,
                             out_channels=self.cell_channels * 4,
                             kernel_size=5,
                             stride=1,
                             padding=2,
                             bias=True)

    def forward(self, x: Tensor, hidden: Tensor,
            cell: Tensor) -> Tuple[Tensor, Tensor]:
        B, _, H, W = hidden.shape
        if x is None:
            x = torch.zeros((B, self.in_channels, H, W)).to(hidden.device)

        all_inputs = torch.cat([x, hidden], dim=1)
        all_inputs_conv = self.cns(all_inputs)

        xh_i, xh_f, xh_o, xh_c = torch.split(all_inputs_conv, self.cell_channels, dim=1)
        i_t = self.sigmoid(xh_i)
        f_t = self.sigmoid(xh_f)
        o_t = self.sigmoid(xh_o)
        g_t = self.tanh(xh_c)
        c_t = f_t * cell + i_t * g_t
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t


class ConvLSTMEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cell_channels: tuple,
        seq_len: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cell_channels = cell_channels
        self.seq_len = seq_len
        self.lstms = self._make_lstms()

    def _make_lstms(self) -> nn.ModuleDict:
        lstms = nn.ModuleDict({})
        for i in range(len(self.cell_channels)):
            in_channels = self.in_channels if i == 0 else self.cell_channels[i - 1]
            lstms[f"lstm_{i}"] = ConvLSTMLayer(in_channels, self.cell_channels[i])
        return lstms

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        new_states = {}
        B, _, _, H, W = x.shape
        for t in range(self.seq_len):
            for i, (k, lstm) in enumerate(self.lstms.items()):
                initial_states = torch.zeros((B, self.cell_channels[i], H, W)).to(x.device) # LSTM init state is 0
                x_in = x[:, t] if i == 0 else new_states[f"h_e_lstm_{i - 1}"] # Only lstm_0 gets input frame
                hidden = initial_states if t == 0 else new_states[f"h_e_{k}"] # 
                cell = initial_states if t == 0 else new_states[f"c_e_{k}"]   # Initial state = 0 only for first timestep

                new_hidden, new_cell = lstm(x_in, hidden, cell)
                new_states[f"h_e_{k}"] = new_hidden
                new_states[f"c_e_{k}"] = new_cell 
        return new_states


class ConvLSTMDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cell_channels: tuple,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cell_channels = cell_channels
        self.seq_len = seq_len
        self.kernel_size = 5
        self.stride = 1
        self.padding = 1

        self.lstms = self._make_lstms()
        self.final_cn = nn.Conv2d(sum(self.cell_channels),
                                  1, (1, 1),
                                  stride=1,
                                  padding=0)

    def _make_lstms(self) -> nn.ModuleDict:
        lstms = nn.ModuleDict({})
        for i in range(len(self.cell_channels)):
            in_channels = self.in_channels if i == 0 else self.cell_channels[i - 1]
            lstms[f"lstm_{i}"] = ConvLSTMLayer(in_channels, self.cell_channels[i])
        return lstms

    def forward(
        self, states: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # TODO: states dict is mutated and returned, not good
        preds = []
        for t in range(self.seq_len):
            for i, (k, lstm) in enumerate(self.lstms.items()):
                state_id = "e" if t == 0 else "d"
                x_in = None if i == 0 else states[f"h_d_lstm_{i - 1}"]
                hidden, cell = lstm(x_in, states[f"h_{state_id}_{k}"], states[f"c_{state_id}_{k}"])
                states[f"h_d_{k}"] = hidden
                states[f"c_d_{k}"] = cell

            final_hidden = [v for k, v  in states.items() if "h_d" in k]
            predicted_frame = self.final_cn(torch.cat(tuple(final_hidden), dim=1))
            preds.append(predicted_frame.unsqueeze(1))
        preds = torch.cat(preds, dim=1)
        return preds, states


class ConvLSTMForecastingNet(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 seq_len: int,
                 cell_channels: tuple,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.cell_channels = cell_channels
        self.encoder_net = ConvLSTMEncoder(self.in_channels, self.cell_channels, self.seq_len)
        self.decoder_net = ConvLSTMDecoder(self.in_channels, self.cell_channels, self.seq_len)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        encoder_states = self.encoder_net(x)
        pred, states = self.decoder_net(encoder_states)
        return pred, states
