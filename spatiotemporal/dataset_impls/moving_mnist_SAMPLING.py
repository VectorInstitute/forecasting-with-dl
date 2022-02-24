import os
from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision import transforms


@dataclass
class MovingMnist():
    src_file: str = "./datasets/moving_mnist.npy"
    split: str = "train"
    div: float = 1.0    # Ratio of dataset length
    transform: transforms = None

    def __post_init__(self) -> None:
        if self.split not in ["train", "test", "val"]:
            raise KeyError(f"Incorrect dataset split: {self.split}")
        self._raw_dataset = np.load(self.src_file)
        self._raw_dataset = self._raw_dataset.transpose(1, 0, 2, 3)

        if self.split == "train":
            self.dataset = self._raw_dataset[:7000]
            self._len = int(len(self.dataset) * 10 * self.div) 

        elif self.split == "test":
            self.dataset = self._raw_dataset[7000:9000]
            self._len = len(self.dataset) * 10   

        else:
            self.dataset = self._raw_dataset[9000:]
            self._len = len(self.dataset) * 10   

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        example = Tensor(self.dataset[idx // 10][:10]).float()
        target = Tensor(self.dataset[idx // 10][idx % 10]).float()
        leadtime_vec = nn.functional.one_hot(Tensor([idx % 10]).long(), num_classes=10)
        if self.transform is not None:
            example = self.transform(example)
            target = self.transform(target)
        # Unsqueeze channel dim
        return example.unsqueeze(-3), target.unsqueeze(-3), leadtime_vec.float()

    def __len__(self) -> None:
        return self._len

dset = MovingMnist()
samples = torch.zeros(10 * 11, 1, 64, 64)
for i in range(0, 110, 11):
    example, target, _ = dset[i]
    sample = torch.cat((example, target.unsqueeze(0)), dim=0)
    samples[i:i + 11, :, :, :] = sample
grid_samples = torchvision.utils.make_grid(samples, nrow=11)
import pdb; pdb.set_trace()
