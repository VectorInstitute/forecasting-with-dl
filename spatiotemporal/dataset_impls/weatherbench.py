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
    src_dir: str = "./datasets/total_cloud_cover_1.40625deg"
    n_obs: int = 4
    n_target: int = 12
    n_blackout: int = 8
    rdm_seed: int = 420

    def __post_init__(self) -> None:
        torch.manual_seed(self.rdm_seed)
        self._raw_dset = xr.open_mfdataset(f"{self.src_dir}/*.nc",
                                           combine="by_coords")
        self.dataset = getattr(
            self._raw_dset, self.dset_attr).sel(time=slice(*self.date_range))
        self._sampler = list(
            BatchSampler(SequentialSampler(range(self.dataset.shape[0])),
                         batch_size=self.n_obs + self.n_target +
                         self.n_blackout,
                         drop_last=True))
        self._len = len(self.sampler)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        data_range = self._sampler[idx][self.n_blackout:]
        example = self.dataset[data_range[:self.n_obs]]
        target = self.dataset[data_range[self.n_obs:]]
        assert example.shape[0] == self.n_obs and target.shape[
            0] == self.n_target
        return torch.tensor(example.values), torch.tensor(target.values)

    def __len__(self):
        return self._len


dset = WeatherBench(["2000-01-01", "2018-12-30"])
print(dset)
print(len(dset))
start_time = time.time()
from tqdm import tqdm
for i in tqdm(range(len(dset))):
    x, y = dset[i]
    breakpoint()
end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")
