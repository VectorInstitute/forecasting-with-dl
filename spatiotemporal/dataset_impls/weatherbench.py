import os
from dataclasses import dataclass
import time
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision import transforms
import xarray as xr
import dask


@dataclass
class WeatherBench():
    dset_attr: str = "tcc" 
    src: str = "/ssd003/projects/aieng/datasets/forecasting"
    split: str = "train"
    every_n: int = 2
    n_obs: int = 12
    n_target: int = 12
    n_blackout: int = 8

    def __post_init__(self) -> None:
        """Init dataset
           Note: Each call to __getitem__ extracts from an xarray object, which is extremely slow
                - Need to convert dataset to a better backend
        """
        # Constants
        self.land_constants = self._extract_from_xarray("constants_1.40625deg.nc", "lsm", time=None, change_axes=True)
        self.land_constants = Tensor(self.land_constants.values)

        # Variable
        self.dataset = self._extract_from_xarray("*.nc", self.dset_attr, time=slice("2000-01-01", "2018-12-30"), change_axes=True, combine="by_coords")

        # Need to manually partition dataset
        total_frame_len = self.n_obs + self.n_target + self.n_blackout
        self._slices = [i for i in range(0, self.dataset.shape[0], self.every_n)]                                           # Create list of all frame indices skipping every_n
        self._samples = [list(self._slices[i:i + total_frame_len]) for i in range(0, len(self._slices), total_frame_len)]   # frames, then slice into examples
        del self._samples[-1]   # Drop last example (ie. drop_last=True for avoiding example truncation) 

        split_idx = int(len(self._samples) * 0.8)
        self.data_idxs = self._samples[:split_idx] if self.split == "train" else self._samples[split_idx:]

        self.target_shape  = [[8, 40, 160, 192], # y_low, y_high, x_low, x_high
                              [8, 40, 192, 224]] # Hardcoded example frames
        self.context_shape = [[0, 56, 144, 208], # These also overlap, so be careful
                              [0, 56, 176, 240]] #

        self.zeropad = nn.ZeroPad2d((0, 0, 8, 0))   # Padding for context area (0 is minimum upper bound on y)

        self._len = len(self.data_idxs) * 2 * 12    # 2 regions, 12 target lead timeframes per example

    def _latlon_to_cartesian(self, dset):
        """Switch latitude from [-90, 90] to [0, _], (ie. swap origin from bottom left to top left)
        Args:
            dset (xarray.DataArray): xarray dataset object to swap coordinate systems
        """
        with dask.config.set(**{'array.slicing.split_large_chunks': False}): # Silence dataset chunking warning
            return dset.reindex(lat=list(reversed(dset.lat)))

    def _extract_from_xarray(self, filename: str, dset_attr: str, time: slice = None, change_axes: bool = False, **open_kwargs: Any) -> Tensor:
        """Get the corresponding dataset slice from the dataset file(s)
        Args:
            filename (str): Dataset filename (can use wildcard * for multiple files)
            dset_attr (str): xarray variable attribute (ie. tcc = total_cloud_cover)
            time (slice): Time frame for data extraction from dataset file
            change_axes (bool): If True, the xarray will be converted from lat/lon to Cartesian, with origin at top left
            open_kwargs: kwargs for the call to xr.open_mfdataset()
        """
        xr_dset = xr.open_mfdataset(f"{self.src}/{filename}", **open_kwargs)
        dset_attr_data = getattr(xr_dset, dset_attr)
        dset_data = dset_attr_data.sel(time=time) if time is not None else dset_attr_data   # Get a certain time range 
        dset = self._latlon_to_cartesian(dset_data) if change_axes is True else dset_data   # Convert to cartesian coords if necessary
        return dset

    def _get_idxs(self, idx: int) -> Tuple[int, int]:
        """Retrieve the location and leadtime indexes for the given dataset scheme
        Args:
            idx (int): Index of dataset
        """
        loc_idx = idx % 2   # Grid 0 or 1
        lt_idx = idx // 2 % self.n_target   # Timestep from 0-11 inclusive
        return loc_idx, lt_idx

    def _get_slice(self, idx: int, loc_idx: int, lt_idx: int):
        """Slice into the netCDF/xarray obj
        Args:
            loc_idx (int): The location index
            lt_idx (int): The leadtime index
        """
        context_y_low, context_y_high, context_x_low, context_x_high = self.context_shape[loc_idx]
        target_y_low, target_y_high, target_x_low, target_x_high = self.target_shape[loc_idx]

        # Slice example into context (n_obs) and target (n_target)
        data_range = self.data_idxs[idx // 24][self.n_blackout:]
        context = self.dataset[data_range[:self.n_obs]]
        target = self.dataset[data_range[self.n_obs + lt_idx]]

        # Slice into xarray obj
        context = context[:, context_y_low:context_y_high, context_x_low:context_x_high]
        land_constant = self.land_constants[context_y_low:context_y_high, context_x_low:context_x_high]
        target = target[target_y_low:target_y_high, target_x_low:target_x_high]

        return context, land_constant, target

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve example from dataset, where each example=[n_blackout, n_obs, n_target]
        Args:
            idx (int): Index of dataset for retrieval
        Returns:
            context (Tensor): Context tensor of shape (T, C, H, W); aggregate of multiple variables (ie. tcc + land_constant) across channels
            target (Tensor): Target tensor of shape (C, H, W)
            leadtime_vec (Tensor): Leadtime vector of shape (1, T)
        """
        loc_idx, lt_idx = self._get_idxs(idx)

        context, land_constant, target = self._get_slice(idx, loc_idx, lt_idx)
        context = Tensor(context.values).unsqueeze(-3)
        target = Tensor(target.values).unsqueeze(-3)
        leadtime_vec = F.one_hot(Tensor([lt_idx]).long(), num_classes=self.n_target).float() 

        context = torch.cat((context, land_constant.view(1, *context.shape[1:]).tile(12, 1, 1, 1)), dim=      -3)
        context = self.zeropad(context)

        return context, target, leadtime_vec

    def __len__(self):
        return self._len
