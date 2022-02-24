import os
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


class LeadTimeMLPSystem(nn.Module):
    def __init__(self,
                 lead_timesteps: int,
                 master_layers: int = 2,
                 master_out_features: int = 512,
                 master_bias: bool = False,
        ) -> None:
        super().__init__()

        # Training args
        self.lead_timesteps = lead_timesteps
        
        # Master MLP args
        self.master_layers = master_layers
        self.master_in_features = lead_timesteps    # Input to master MLP is one-hot for lead timesteps
        self.master_out_features = master_out_features
        self.master_bias = master_bias

        # Master inits
        self.master = self._make_master()

    def _make_master(self) -> nn.Sequential:    # Expands feature dims on last layer
        master = []
        for i in range(self.master_layers - 1):
            master.append(nn.Linear(in_features=self.master_in_features,
                                    out_features=self.master_in_features,
                                    bias=self.master_bias,))
        master.append(nn.Linear(in_features=self.master_in_features,
                                out_features=self.master_out_features,
                                bias=self.master_bias,))
        return nn.Sequential(*master)

    def forward(self, leadtime: Tensor) -> Tensor:
        # Master
        return self.master(leadtime)
