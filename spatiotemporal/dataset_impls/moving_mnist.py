import os
from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
import torchvision
from torchvision import transforms


@dataclass
class MovingMnist():
    src_file: str = "./datasets/moving_mnist.npy"
    split: str = "train"
    transform: transforms = None

    def __post_init__(self) -> None:
        if self.split not in ["train", "test", "val"]:
            raise KeyError(f"Incorrect dataset split: {self.split}")
        self._raw_dataset = np.load(self.src_file)
        self._raw_dataset = self._raw_dataset.transpose(1, 0, 2, 3)
        if self.split == "train":
            self.dataset = self._raw_dataset[:7000]
        elif self.split == "test":
            self.dataset = self._raw_dataset[7000:9000]
        else:
            self.dataset = self._raw_dataset[9000:]

        self._len = len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        example = Tensor(self.dataset[idx][:10]).float()
        target = Tensor(self.dataset[idx][10:]).float()
        if self.transform is not None:
            example = self.transform(example)
            target = self.transform(target)
            return example.unsqueeze(1), target.unsqueeze(1)
        return example, target

    def __len__(self) -> None:
        return self._len
