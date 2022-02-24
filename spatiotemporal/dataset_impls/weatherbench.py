import os
from dataclasses import dataclass
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import BatchSampler, SequentialSampler
import torchvision
from torchvision import transforms
import xarray as xr
import dask


@dataclass
class WeatherBench():
    dset_attr: str = "tcc"
    src: str = "/ssd003/projects/aieng/datasets/forecasting"
    split: str = "train"
    grid_size: int = 32
    every_n: int = 2
    n_obs: int = 12
    n_target: int = 12
    n_blackout: int = 8
    div: float = 1.0
    transform: transforms = None

    def __post_init__(self) -> None:
        # Constants
        self._raw_constants = xr.open_mfdataset(f"{self.src}/constants_1.40625deg.nc")
        self._land_constant = getattr(self._raw_constants, "lsm")
        self._land_constant = self._latlon_to_cartesian(self._land_constant)
        self.land_constant = torch.tensor(self._land_constant.values)

        # TCC examples
        self._raw_dset = xr.open_mfdataset(f"{self.src}/*.nc",
                                           combine="by_coords")
        self.dataset = getattr(
            self._raw_dset, self.dset_attr).sel(time=slice("2000-01-01", "2018-12-30"))
        self.dataset = self._latlon_to_cartesian(self.dataset)

        total_frame_len = self.n_obs + self.n_target + self.n_blackout
        #TODO: Fix this for every_n
        self._slices = [i for i in range(0, self.dataset.shape[0], self.every_n)]
        self._samples = [list(self._slices[i:i + total_frame_len]) for i in range(0, len(self._slices), total_frame_len)]

        self._split_idx = int(len(self._samples) * 0.8)
        if self.split == "train":
            self.data_idxs = self._samples[:self._split_idx]
        else:
            self.data_idxs = self._samples[self._split_idx:]
        self._len = 12 * 32 * 1000 #len(self.data_idxs * 12 * 32)

    def _latlon_to_cartesian(self, dset):
        # Silence dataset chunking warning
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            return dset.reindex(lat=list(reversed(dset.lat)))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """example=[n_blackout, n_obs, n_target]"""
        # 1 example = 32 locations * 12 leadtime frames per location = 384 effective examples per 1 raw example

        # Isolate location and leadtime idxs
        loc_idx = idx // 12 % 32     # This isolates idx to tell us location 
        lt_idx = idx % 12       # This is the leadtime idx for the example

        # Get grid location
        y_low = (loc_idx // 8) * self.grid_size
        x_low = (loc_idx % 8) * self.grid_size
        y_hi = y_low + self.grid_size
        x_hi = x_low + self.grid_size

        #print(f"idx: {idx} | loc_idx: {loc_idx}, lt_idx: {lt_idx} | y: {y_low}-{y_hi}, x: {x_low}-{x_hi}")
        data_range = self.data_idxs[idx // 384][self.n_blackout:]
        #print(data_range)

        example = self.dataset[data_range[:self.n_obs]]
        target = self.dataset[data_range[self.n_obs + (lt_idx)]]
        leadtime_vec = nn.functional.one_hot(Tensor([lt_idx]).long(), num_classes=12)

        example = torch.tensor(example.values).unsqueeze(-3)[:, :, y_low:y_hi, x_low:x_hi]    # (T, C, H, W)
        target = torch.tensor(target.values).unsqueeze(-3)[:, y_low:y_hi, x_low:x_hi]         # (C, H, W)
        leadtime_vec = leadtime_vec.float()                                                   # (1, T)
        land_constant = self.land_constant[y_low:y_hi, x_low:x_hi]                            # (H, W)

        #example = torch.cat((example, land_constant.unsqueeze(0).unsqueeze(0)), dim=-3)
        return example, target, leadtime_vec    #, land_constant

    def __len__(self):
        return self._len


def sample_dataset(grid_size, every_n, idx):
    train_dset = WeatherBench(split="train", grid_size=grid_size, every_n=every_n)
    val_dset = WeatherBench(split="val", grid_size=grid_size, every_n=every_n)

    train_example, train_target, train_leadtime_vec, train_land_constant = train_dset[idx]
    val_example, val_target, val_leadtime_vec, val_land_constant = val_dset[idx]

    train_target = train_target.unsqueeze(0)    # Make them all (T, C, H, W) for cat
    train_land_constant = train_land_constant.view(1, 1, train_land_constant.shape[-2], train_land_constant.shape[-1])
    val_target = val_target.unsqueeze(0)
    val_land_constant = val_land_constant.view(1, 1, val_land_constant.shape[-2], val_land_constant.shape[1])

    train_ex_grid = torchvision.utils.make_grid(torch.cat((train_example, train_target, train_land_constant), dim=0))
    val_ex_grid = torchvision.utils.make_grid(torch.cat((val_example, val_target, val_land_constant), dim=0))
    return train_ex_grid, val_ex_grid

"""
train_dset = WeatherBench(split="train", grid_size=32, every_n=2)
for i in range(500):
    train_dset[i]
"""

"""
GRID_SIZES = [32, 32, 32, 32]
EVERY_N = [1, 1, 1, 1]
for idx in range(0, 42, 3):
    train_sample, val_sample = sample_dataset(32, 2, idx)
    torchvision.utils.save_image(train_sample, f"wb_slicing_samples/train_sample_{idx}.png")
    torchvision.utils.save_image(val_sample, f"wb_slicing_samples/val_sample_{idx}.png")
"""
