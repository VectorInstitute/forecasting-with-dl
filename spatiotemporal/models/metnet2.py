from typing import Tuple, List
import argparse

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

from .downsample import Downsample2d, ParameterizedDownsample2d, CropAndTile
from .convlstm import ConvLSTMEncoder
from .dilated_net import DilatedEncoder
from .lead_time_system import LeadTimeMLPSystem
from .resnet import ResNet

# from .custom_loss_fns import CRPSLoss
from .multi_input_sequential import boolean_string


class MetNet2(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        wd: float,
        context_timesteps: int,
        lstm_in_channels: int,
        lstm_network_hidden_channels: list,
        h_dim: int,
        w_dim: int,
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
        self.context_timesteps = context_timesteps

        # Downsample, crop, and tile operations
        # TODO: Make these modular
        self.downsample = Downsample2d("maxpool", (2, 2), 0)
        self.cropandtile = CropAndTile(16, 16, (2, 2))

        # ConvLSTM params
        self.lstm_in_channels = lstm_in_channels
        self.lstm_network_hidden_channels = (
            [128] * 12
            if lstm_network_hidden_channels is None
            else list(map(int, lstm_network_hidden_channels))
        )

        # Dilated conv params
        self.dilated_in_channels = self.lstm_network_hidden_channels[-1]
        self.dilated_h_dim = h_dim
        self.dilated_w_dim = w_dim
        self.dilated_dilations = [2**i for i in range(dilated_num_dilations)]
        self.dilated_hidden_channels = dilated_hidden_channels

        # Lead time system params
        self.lt_lead_timesteps = lt_lead_timesteps
        self.lt_master_layers = lt_master_layers
        self.lt_master_out_features = lt_master_out_features
        self.lt_master_bias = lt_master_bias

        # Resnet params
        self.resnet_in_channels = dilated_hidden_channels
        self.resnet_hidden_channels = dilated_hidden_channels
        self.resnet_out_channels = resnet_out_channels
        self.resnet_h_dim = h_dim
        self.resnet_w_dim = w_dim
        self.resnet_n_layers = resnet_n_layers

        self.convlstm_network = ConvLSTMEncoder(
            self.lstm_in_channels, 
            self.lstm_network_hidden_channels,
            self.lt_master_out_features
        )
        self.dilated_encoder1 = DilatedEncoder(
            self.dilated_in_channels,
            self.dilated_hidden_channels,
            self.dilated_dilations,
            self.dilated_h_dim,
            self.dilated_w_dim,
            self.lt_master_out_features,
        )
        self.dilated_encoder2 = DilatedEncoder(
            self.dilated_hidden_channels,
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
            self.lt_master_bias,
        )
        self.resnet = ResNet(
            self.resnet_in_channels,
            self.resnet_hidden_channels,
            self.resnet_out_channels,
            self.resnet_h_dim,
            self.resnet_w_dim,
            self.resnet_n_layers,
            self.lt_master_out_features,
        )

        # PL params
        self.lr = lr
        self.wd = wd
        self.loss = nn.MSELoss()
        # self.loss = CRPSLoss()
        # self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x: Tensor, leadtime: Tensor) -> Tensor:
        """Forward pass through all MetNet-2 layers
        Args:
            x (Tensor): Input context for processing
            leadtime (Tensor): One-hot encoding vector for some timestep 0 < t < lead_timesteps
        """
        # Lead time system forward to produce all scale_bias 2d tensors
        master_leadtime_vec = self.leadtime_system(leadtime)

        # Downsample 64x64 tensor
        output = self.downsample(x)

        # ConvLSTM network taking x and producing hidden
        output = self.convlstm_network(output, master_leadtime_vec)

        # Dilated convolution sets
        output, _ = self.dilated_encoder1(output, master_leadtime_vec)
        output, _ = self.dilated_encoder2(output, master_leadtime_vec)

        # Crop and tile
        output = self.cropandtile(output)

        # ResNet
        output, _ = self.resnet(output, master_leadtime_vec)

        return output

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configures the optimizer and (or) scheduler for implementing a lr schedule
        Args:
        """
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
        return [optimizer]  # [optimizer], [scheduler]

    def shared_step(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Generic step for a single batch of examples
        Args:
            batch (Tensor): Batch of inputs to net
        """
        x, target, leadtime_vec = batch
        target = torch.as_tensor(target)
        pred = self(x, leadtime_vec)
        loss = self.loss(pred, target)
        return loss, pred, target

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step for batch of inputs
        Args:
            batch (Tensor): Batch of inputs to net
            batch_idx (int): Batch index
        """
        loss, pred, target = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Validation step for batch of inputs
        Args:
            batch (Tensor): Batch of inputs to net
            batch_idx (int): Batch index
        """
        loss, pred, target = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.Argument_parser:
        """Prepares arguments specific to the model architecture and optimizers
        Args:
            parent_parser (ArgumentParser): Parent parser which receives extra model arguments
        """
        parser = parent_parser.add_argument_group("MetNet2")
        parser.add_argument(
            "--lr", type=float, default=2e-5
        )  # Scale down for distributed
        parser.add_argument("--wd", type=float, default=0)
        parser.add_argument("--context_timesteps", type=int, default=12)
        parser.add_argument("--lstm_in_channels", type=int, default=2)
        parser.add_argument("--lstm_network_hidden_channels", nargs="+")
        parser.add_argument("--h_dim", type=int, default=32)
        parser.add_argument("--w_dim", type=int, default=32)
        parser.add_argument("--dilated_num_dilations", type=int, default=6)
        parser.add_argument("--dilated_hidden_channels", type=int, default=256)
        parser.add_argument("--lt_lead_timesteps", type=int, default=12)
        parser.add_argument("--lt_master_layers", type=int, default=2)
        parser.add_argument("--lt_master_out_features", type=int, default=1024)
        parser.add_argument("--lt_master_bias", type=boolean_string, default=True)
        parser.add_argument("--resnet_hidden_channels", type=int, default=64)
        parser.add_argument("--resnet_out_channels", type=int, default=1)
        parser.add_argument("--resnet_n_layers", type=int, default=8)
        return parent_parser
