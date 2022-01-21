from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


from convlstm_simple import ConvLSTMEncoder
from metnet2_dilated import DilatedEncoder
from lead_time_system import LeadTimeMLPSystem


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MetNet2(nn.Module):
    def __init__(
        self,
        context_timesteps: int,
        lstm_in_channels: int,
        lstm_network_hidden_channels: list,
        dilated_in_channels: int,
        dilated_h_dim: int,
        dilated_w_dim: int,
        dilated_dilations: list,
        lt_slave_out_features: list,
        lt_lead_timesteps: int = 10,
        lt_master_layers: int = 2,
        lt_master_out_features: int = 512,
        lt_master_bias: bool = False,
        lt_slave_layers: int = 2,
        lt_slave_bias: bool = False,
        resnet_in_channels: int = 384,
        resnet_out_channels: int = 8,
        resnet_layers: int = 8,
        final1x1_out_channels: int = 1,
    ) -> None:
        super().__init__()
        # MetNet-2 params
        self.context_timesteps = context_timesteps

        # ConvLSTM params
        self.lstm_in_channels = lstm_in_channels
        self.lstm_network_hidden_channels = lstm_network_hidden_channels

        # Dilated conv params
        self.dilated_in_channels = dilated_in_channels
        self.dilated_dilations = dilated_dilations
        self.dilated_h_dim = dilated_h_dim
        self.dilated_w_dim = dilated_w_dim

        # Lead time system params
        self.lt_context_timesteps = context_timesteps
        self.lt_lead_timesteps = lt_lead_timesteps
        self.lt_master_layers = lt_master_layers
        self.lt_master_out_features = lt_master_out_features
        self.lt_master_bias = lt_master_bias
        self.lt_slave_layers = lt_slave_layers
        self.lt_slave_out_features = lt_slave_out_features
        self.lt_slave_bias = lt_slave_bias

        # Resnet params
        self.resnet_layers = resnet_layers
        self.resnet_in_channels = resnet_in_channels
        self.resnet_out_channels = resnet_out_channels

        # Final 1x1 params
        self.final1x1_out_channels = final1x1_out_channels

        self.convlstm_network = ConvLSTMEncoder(self.lstm_in_channels, self.lstm_network_hidden_channels)
        self.dilated_encoder = DilatedEncoder(self.dilated_in_channels, self.dilated_dilations, self.dilated_h_dim, self.dilated_w_dim)
        self.dilated_decoder = DilatedEncoder(self.dilated_in_channels, self.dilated_dilations, self.dilated_h_dim, self.dilated_w_dim)
        self.leadtime_system = LeadTimeMLPSystem(self.lt_slave_out_features, self.lt_context_timesteps, self.lt_lead_timesteps, self.lt_master_layers, self.lt_master_out_features, self.lt_master_bias, self.lt_slave_layers, self.lt_slave_bias)

    def forward(self, x: Tensor) -> Tensor:
        # Lead time system forward to produce all scale_bias 2d tensors

        # ConvLSTM network taking x and producing hidden

        # Dilated convolution set 1

        # Dilated convolution set 2

        # ResNet + 1x1 to produce categorical (or sigmoid to get (0, 1) interval)
        pass


model = MetNet2(10,
                1,
                [128, 128, 64, 64],
                384,
                64,
                64,
                [1, 2, 4, 8, 16],
                [128, 128, 64, 64, 384, 384, 384, 384, 384, 384, 384, 384,384, 384, 384, 384,384, 384, 384, 384, 384, 384, 384, 384],
                10,
                2,
                512,
                False,
                2,
                False,
                384,
                8,
                8,
                22)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))
import pdb; pdb.set_trace()
