import os
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


class LeadTimeMLPSystem(nn.Module):
    def __init__(self,
                 slave_out_features: list, 
                 context_timesteps: int = 10,
                 lead_timesteps: int = 10,
                 master_layers: int = 2,
                 master_out_features: int = 512,
                 master_bias: bool = False,
                 slave_layers: int = 2,
                 slave_bias: bool = False,
        ) -> None:
        super().__init__()

        # Training args
        self.context_timesteps = context_timesteps
        self.lead_timesteps = lead_timesteps
        
        # Master MLP args
        self.master_layers = master_layers
        self.master_in_features = lead_timesteps    # Input to master MLP is one-hot for lead timesteps
        self.master_out_features = master_out_features
        self.master_bias = master_bias

        # Slave MLP args
        self.slave_layers = slave_layers
        self.slave_in_features = master_out_features   # Slaves takes input from master output
        self.slave_out_features = slave_out_features
        self.slave_bias = slave_bias

        # Master inits
        self.master = self._make_master()

        # Slave inits
        self.slaves = self._make_slaves()

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

    def _make_slaves(self) -> nn.ModuleDict:    # Expands feature dims on last layer
        slaves = nn.ModuleDict({})
        for i, out_features in enumerate(self.slave_out_features):
            slave = []
            for j in range(self.slave_layers - 1):
                slave.append(nn.Linear(in_features=self.slave_in_features,
                                       out_features=self.slave_in_features,
                                       bias=self.slave_bias,))
            slave.append(nn.Linear(in_features=self.slave_in_features,
                                   out_features=out_features * 2,   # Scale and bias
                                   bias=self.slave_bias,))
            slaves[f"slave_{i}"] = nn.Sequential(*slave)
        return slaves

    def forward(self, leadtime: Tensor, batch_size: int) -> Dict[str, Tensor]:
        # Master
        master_encoded_vector = self.master(leadtime)

        # Slaves
        slave_vectors = []
        for i, slave in enumerate(self.slaves.values()):
            slave_vector = slave(master_encoded_vector)
            scale, bias = slave_vector.chunk(2, dim=1)
            slave_vectors.append(scale.unsqueeze(2).unsqueeze(3))
            slave_vectors.append(bias.unsqueeze(2).unsqueeze(3))
        return slave_vectors
