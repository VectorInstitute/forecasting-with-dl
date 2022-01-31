from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

from .convlstm_simple import ConvLSTMEncoder
from .dilated_net import DilatedEncoder
from .lead_time_system import LeadTimeMLPSystem
from .resnet import ResNet10
from .multi_input_sequential import MultiInputSequential, boolean_string


class MetNet2(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-5,
        wd: float = 0,
        context_timesteps: int = 10,
        lstm_in_channels: int = 1,
        lstm_network_hidden_channels: list = [128, 128, 64, 64],    # None default arg handled
        dilated_in_channels: int = 384,
        dilated_h_dim: int = 64,
        dilated_w_dim: int = 64,
        dilated_num_dilations: int = 5, # NEW
        
        lt_lead_timesteps: int = 10,
        lt_master_layers: int = 2,
        lt_master_out_features: int = 512,
        lt_master_bias: bool = False,
        lt_slave_layers: int = 2,
        lt_slave_bias: bool = False,
        resnet_num_layers: int = 10,    # NEW
        resnet_in_channels: int = 384,
        resnet_out_channels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # MetNet-2 params
        self.context_timesteps = context_timesteps

        # ConvLSTM params
        self.lstm_in_channels = lstm_in_channels
        self.lstm_network_hidden_channels = [128, 128, 64, 64] if lstm_network_hidden_channels is None else list(map(int, lstm_network_hidden_channels))

        # Dilated conv params
        self.dilated_in_channels = dilated_in_channels
        self.dilated_h_dim = dilated_h_dim
        self.dilated_w_dim = dilated_w_dim
        self.dilated_num_dilations = dilated_num_dilations
        self.dilated_dilations = [2 ** i for i in range(dilated_num_dilations)]

        # Lead time system params
        self.lt_context_timesteps = context_timesteps
        self.lt_lead_timesteps = lt_lead_timesteps
        self.lt_master_layers = lt_master_layers
        self.lt_master_out_features = lt_master_out_features
        self.lt_master_bias = lt_master_bias
        self.lt_slave_layers = lt_slave_layers
        self.lt_slave_out_features = self.lstm_network_hidden_channels + ([dilated_in_channels] * dilated_num_dilations * 2) + ([resnet_in_channels] * (resnet_num_layers - 2))  
        self.lt_slave_bias = lt_slave_bias

        # Resnet params
        self.resnet_num_layers = resnet_num_layers
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

        # PL params
        self.lr = lr
        self.wd = wd
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x: Tensor, leadtime: Tensor) -> Tensor:
        """Forward pass assertions are for MovingMnist"""
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

    def configure_optimizers(self): 
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        x, target, leadtime_vec = batch
        target = torch.as_tensor(target)  #.to(x.device)
        pred = self(x, leadtime_vec)
        loss = self.loss(pred, target)
        return loss, pred, target

    def training_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetNet2")
        parser.add_argument('--lr', type=float, default=2e-3 / 8)   # Scale down for distributed
        parser.add_argument("--wd", type=float, default=0)
        parser.add_argument("--context_timesteps", type=int, default=10)
        parser.add_argument("--lstm_in_channels", type=int, default=1)
        parser.add_argument("--lstm_network_hidden_channels", nargs="+")
        parser.add_argument("--dilated_in_channels", type=int, default=384)
        parser.add_argument("--dilated_h_dim", type=int, default=64)
        parser.add_argument("--dilated_w_dim", type=int, default=64)
        parser.add_argument("--lt_lead_timesteps", type=int, default=10)
        parser.add_argument("--lt_master_layers", type=int, default=2)
        parser.add_argument("--lt_master_out_features", type=int, default=512)
        parser.add_argument("--lt_master_bias", type=boolean_string, default=False)
        parser.add_argument("--lt_slave_layers", type=int, default=2)
        parser.add_argument("--lt_slave_bias", type=boolean_string, default=False)
        parser.add_argument("--resnet_in_channels", type=int, default=384)
        parser.add_argument("--resnet_out_channels", type=int, default=1)
        return parent_parser
