from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

from .convlstm_simple import ConvLSTMEncoder
from .dilated_net import DilatedEncoder
from .lead_time_system import LeadTimeMLPSystem
from .resnet import ResNet10
from .custom_loss_fns import CRPSLoss
from .multi_input_sequential import MultiInputSequential, boolean_string


class MetNet2(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        wd: float,
        context_timesteps: int,
        lstm_in_channels: int,
        lstm_network_hidden_channels: list,    # Mutable, be careful
        dilated_h_dim: int,
        dilated_w_dim: int,
        dilated_num_dilations: int,
        dilated_hidden_channels: int,
        lt_lead_timesteps: int,
        lt_master_layers: int,
        lt_master_out_features: int,
        lt_master_bias: bool,
        resnet_out_channels: int,
        resnet_n_layers: int,
        **kwargs,
    ) -> None:
        super().__init__()
        # MetNet-2 params
        self.context_timesteps = context_timesteps

        # ConvLSTM params
        self.lstm_in_channels = lstm_in_channels
        self.lstm_network_hidden_channels = [128] * 12 if lstm_network_hidden_channels is None else list(map(int, lstm_network_hidden_channels))

        # Dilated conv params
        self.dilated_in_channels = self.lstm_network_hidden_channels[-1]
        self.dilated_h_dim = dilated_h_dim
        self.dilated_w_dim = dilated_w_dim
        self.dilated_dilations = [2 ** i for i in range(dilated_num_dilations)]
        self.dilated_hidden_channels = dilated_hidden_channels

        # Lead time system params
        self.lt_lead_timesteps = lt_lead_timesteps
        self.lt_master_layers = lt_master_layers
        self.lt_master_out_features = lt_master_out_features
        self.lt_master_bias = lt_master_bias

        # Resnet params
        self.resnet_in_channels = dilated_hidden_channels
        self.resnet_out_channels = resnet_out_channels
        self.resnet_n_layers = resnet_n_layers

        self.convlstm_network = ConvLSTMEncoder(self.lstm_in_channels, self.lstm_network_hidden_channels)
        self.dilated_encoder1 = DilatedEncoder(self.dilated_in_channels,
                                              self.dilated_hidden_channels,
                                              self.dilated_dilations,
                                              self.dilated_h_dim,
                                              self.dilated_w_dim,
                                              self.lt_master_out_features,
        )
        self.leadtime_system = LeadTimeMLPSystem(
            self.lt_lead_timesteps, 
            self.lt_master_layers,
            self.lt_master_out_features, 
            self.lt_master_bias
        )
        self.resnet = ResNet10(self.resnet_in_channels, 
                               self.resnet_out_channels,
                               self.resnet_n_layers,
                               self.lt_master_out_features
        )

        # PL params
        self.lr = lr
        self.wd = wd
        self.loss = nn.MSELoss()
        #self.loss = CRPSLoss()
        #self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x: Tensor, leadtime: Tensor) -> Tensor:
        """Forward pass assertions are for MovingMnist"""
        # Lead time system forward to produce all scale_bias 2d tensors
        master_leadtime_vec = self.leadtime_system(leadtime)

        # ConvLSTM network taking x and producing hidden
        output = self.convlstm_network(x)

        # Dilated convolution set 1
        output = self.dilated_encoder1(output, master_leadtime_vec)

        # ResNet + 1x1 to produce categorical (or sigmoid to get (0, 1) interval)
        output = self.resnet(output, master_leadtime_vec)

        return output

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
        parser.add_argument("--context_timesteps", type=int, default=12)
        parser.add_argument("--lstm_in_channels", type=int, default=1)
        parser.add_argument("--lstm_network_hidden_channels", nargs="+")
        parser.add_argument("--dilated_h_dim", type=int, default=32)
        parser.add_argument("--dilated_w_dim", type=int, default=32)
        parser.add_argument("--dilated_num_dilations", type=int, default=6)
        parser.add_argument("--dilated_hidden_channels", type=int, default=256)
        parser.add_argument("--lt_lead_timesteps", type=int, default=12)
        parser.add_argument("--lt_master_layers", type=int, default=2)
        parser.add_argument("--lt_master_out_features", type=int, default=1024)
        parser.add_argument("--lt_master_bias", type=boolean_string, default=True)
        parser.add_argument("--resnet_out_channels", type=int, default=1)
        parser.add_argument("--resnet_n_layers", type=int, default=10)
        return parent_parser
