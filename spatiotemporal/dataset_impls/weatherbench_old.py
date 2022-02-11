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
    #TODO: Rotate 180deg because data is upside down due to lat/long coord system
    dset_attr: str = "tcc"
    src: str = "/ssd003/projects/aieng/datasets/forecasting"
    split: str = "train"
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

        self._sampler = list(
            BatchSampler(SequentialSampler(range(self.dataset.shape[0])),
                         batch_size=self.n_obs + self.n_target +
                         self.n_blackout,
                         drop_last=True))
        self._split_idx = int(len(self._sampler) * 0.8)
        if self.split == "train":
            self.data_idxs = self._sampler[:self._split_idx]
        else:
            self.data_idxs = self._sampler[self._split_idx:]
        self._len = 12 * 1000 #len(self.data_idxs * 12 * 8)

    def _latlon_to_cartesian(self, dset):
        # Silence dataset chunking warning
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            return dset.reindex(lat=list(reversed(dset.lat)))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """example=[n_blackout, n_obs, n_target]"""
        data_range = self.data_idxs[idx // 12][self.n_blackout:]
        example = self.dataset[data_range[:self.n_obs]]
        target = self.dataset[data_range[self.n_obs + (idx % 12)]]
        leadtime_vec = nn.functional.one_hot(Tensor([idx % 12]).long(), num_classes=12)
        return torch.tensor(example.values).unsqueeze(-3)[:, :, :32, :32], torch.tensor(target.values).unsqueeze(-3)[:, :32, :32], leadtime_vec.float() #, self.land_constant

    def __len__(self):
        return self._len

"""
train_dset = WeatherBench(split="train")
val_dset = WeatherBench(split="val")

for i in range(0, 120, 12):
    example, target, _ = train_dset[i]
    sample = torchvision.utils.make_grid(torch.cat((example, target.unsqueeze(0)), dim=0), nrow=13)
    torchvision.utils.save_image(sample, f"./test_samples/WB_SAMPLE_train{i}.png")

for i in range(0, 120, 12):
    example, target, _ = val_dset[i]
    sample = torchvision.utils.make_grid(torch.cat((example, target.unsqueeze(0)), dim=0), nrow=13)
    torchvision.utils.save_image(sample, f"./test_samples/WB_SAMPLE_val{i}.png")
"""
