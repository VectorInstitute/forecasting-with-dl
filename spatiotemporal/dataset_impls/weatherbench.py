import os
from dataclasses import dataclass
import time
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, SequentialSampler
import xarray as xr


@dataclass
class WeatherBench():
    #TODO: Implement train/val/test splitting
    date_range: list
    dset_attr: str = "tcc"
    src_dir: str = "/ssd003/projects/aieng/datasets/forecasting"
    split: str = "train"
    n_obs: int = 12
    n_target: int = 12
    n_blackout: int = 8

    def __post_init__(self) -> None:
        torch.manual_seed(69)
        assert self.split in ["train", "val"]
        self._raw_dset = xr.open_mfdataset(f"{self.src_dir}/*.nc",
                                           combine="by_coords")
        self.dataset = getattr(
            self._raw_dset, self.dset_attr).sel(time=slice(*self.date_range))
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
        self._len = len(self.data_idxs * 12)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """example=[n_blackout, n_obs, n_target]"""
        data_range = self.data_idxs[idx // 12][self.n_blackout:]
        example = self.dataset[data_range[:self.n_obs]]
        target = self.dataset[data_range[self.n_obs % 12]]
        return torch.tensor(example.values).unsqueeze(-3), torch.tensor(target.values).view(1, 1, 128, 256)

    def __len__(self):
        return self._len

"""
train_dset = WeatherBench(["2000-01-01", "2018-12-30"], split="train")
val_dset = WeatherBench(["2000-01-01", "2018-12-30"], split="val")
"""
