from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision import transforms


@dataclass
class CustomCIFAR10():
    src_file: str = "/scratch/ssd002/datasets/cifar10/"
    train: bool = True
    transform: transforms = None

    def __post_init__(self) -> None:
        self._raw_dataset = torchvision.datasets.CIFAR10(self.src_file,
                                                         train=self.train,
                                                         transform=None,
                                                         download=False)
        self._len = len(self._raw_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        example, target = self._raw_dataset[idx]
        if self.transform is not None:
            example = self.transform(example)
        return example, int(target)

    def __len__(self):
        return self._len
