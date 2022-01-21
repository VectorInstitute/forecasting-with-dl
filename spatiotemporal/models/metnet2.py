from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .convlstm_simple import ConvLSTMEncoder
from .dilated_net import DilatedEncoder
from .lead_time_system import LeadTimeMLPSystem
from .resnet import ResNet10


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
        context_timesteps: int = 10,
        lstm_in_channels: int = 1,
        lstm_network_hidden_channels: list = [128, 128, 64, 64],
        dilated_in_channels: int = 384,
        dilated_h_dim: int = 64,
        dilated_w_dim: int = 64,
        dilated_dilations: list = [1, 2, 4, 8, 16],
        lt_slave_out_features: list = [128, 128, 64, 64,        # ConvLSTM 
                                       384, 384, 384, 384, 384, # Dilated ENC
                                       384, 384, 384, 384, 384, # Dilated DEC
                                       768, 768, 768, 768,      # ResNet10 
                                       768, 768, 768, 768],      #
        lt_lead_timesteps: int = 10,
        lt_master_layers: int = 2,
        lt_master_out_features: int = 512,
        lt_master_bias: bool = False,
        lt_slave_layers: int = 2,
        lt_slave_bias: bool = False,
        resnet_in_channels: int = 384,
        resnet_out_channels: int = 1,
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
        self.resnet_in_channels = resnet_in_channels
        self.resnet_out_channels = resnet_out_channels

        self.convlstm_network = ConvLSTMEncoder(
            self.lstm_in_channels, self.lstm_network_hidden_channels)
        self.dilated_encoder = DilatedEncoder(self.dilated_in_channels,
                                              self.dilated_dilations,
                                              self.dilated_h_dim,
                                              self.dilated_w_dim)
        self.dilated_decoder = DilatedEncoder(self.dilated_in_channels,
                                              self.dilated_dilations,
                                              self.dilated_h_dim,
                                              self.dilated_w_dim)
        self.leadtime_system = LeadTimeMLPSystem(
            self.lt_slave_out_features, self.lt_context_timesteps,
            self.lt_lead_timesteps, self.lt_master_layers,
            self.lt_master_out_features, self.lt_master_bias,
            self.lt_slave_layers, self.lt_slave_bias)
        self.resnet = ResNet10(self.resnet_in_channels, self.resnet_out_channels)

    def forward(self, x: Tensor, leadtime: Tensor) -> Tensor:
        # Lead time system forward to produce all scale_bias 2d tensors
        slave_outputs = self.leadtime_system(leadtime, x.shape[0])
        assert len(slave_outputs) == 44

        # ConvLSTM network taking x and producing hidden
        convlstm_net_output = self.convlstm_network(x, slave_outputs[:8])
        assert convlstm_net_output.shape[1:] == (384, 64, 64)

        # Dilated convolution set 1
        dilated_encoder_output = self.dilated_encoder(convlstm_net_output, slave_outputs[8:18])
        assert dilated_encoder_output.shape[1:] == (384, 64, 64)

        # Dilated convolution set 2
        dilated_encoder_output = self.dilated_encoder(convlstm_net_output, slave_outputs[18:28])
        assert dilated_encoder_output.shape[1:] == (384, 64, 64)

        # ResNet + 1x1 to produce categorical (or sigmoid to get (0, 1) interval)
        frame_prediction = self.resnet(dilated_encoder_output, slave_outputs[28:])
        assert frame_prediction.shape[1:] == (1, 64, 64)

        return frame_prediction
