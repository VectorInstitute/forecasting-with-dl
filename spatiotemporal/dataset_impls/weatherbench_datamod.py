from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .weatherbench import WeatherBench


class WeatherBenchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        every_n: int,
        n_obs: int = 12,
        n_target: int = 12,
        n_blackout: int = 8,
        batch_size: int = 4,
        num_workers: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.every_n = every_n
        self.n_obs = n_obs
        self.n_target = n_target
        self.n_blackout = n_blackout
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the training and validation datasets
        Args:
            stage (str): 
        """
        self.train_dset = WeatherBench(
            src=self.data_dir,
            split="train",
            every_n=self.every_n,
            n_obs=self.n_obs,
            n_target=self.n_target,
            n_blackout=self.n_blackout
        )
        self.val_dset = WeatherBench(
            src=self.data_dir,
            split="val",
            every_n=self.every_n,
            n_obs=self.n_obs,
            n_target=self.n_target,
            n_blackout=self.n_blackout
        )

    def train_dataloader(self) -> DataLoader:
        """Yield the training dataloader
        Args:
        """
        return DataLoader(
            dataset=self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Yield the validation dataloader
        Args:
        """
        return DataLoader(
            dataset=self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Yield the test dataloader
        Args:
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Teardown logic
        Args:
        """
        raise NotImplementedError
