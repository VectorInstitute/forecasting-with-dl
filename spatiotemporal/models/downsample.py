from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Downsample2d(nn.Module):
    def __init__(
        self, pool_type: str, kernel_size: Union[Tuple[int, int], int], padding: int
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        if pool_type in "maxpool":
            self.pool = nn.MaxPool2d(self.kernel_size, padding=padding)
        elif pool_type in "avgpool":
            self.pool = nn.AvgPool2d(self.kernel_size, padding=padding)
        else:
            raise ValueError("Invalid pooling type")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for downsampling layer
        Args:
            x (Tensor): Input to be downsampled via selected downsampling function
        """
        x = x.transpose(0, 1)  # BTCHW -> TBCHW
        result = []
        for time_frame in x:    # Torch doesn't support 5-dim input
            result.append(self.pool(time_frame))
        return torch.stack(result).transpose(0, 1)  # TBCHW -> BTCHW


class ParameterizedDownsample2d(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int],
        padding: Union[Tuple[int, int], int],
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.downsample = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for a parameterized (convolutional) 2d downsampling layer
        Args:
            x (Tensor): Input to be downsampled
        """
        x = x.transpose(0, 1)  # BTCHW -> TBCHW
        result = []
        for time_frame in x:
            result.append(self.downsample(time_frame))
        return torch.stack(result).transpose(0, 1)  # TBCHW -> BTCHW


class CropAndTile(nn.Module):
    def __init__(
        self, crop_height: int, crop_width: int, tile: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.tile = tile

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for crop and tile. Center crops the input, then tiles it in the H and W dimensions
        Args:
            x (Tensor): Input to be cropped and tiled
        """
        assert len(x.shape) == 4  # B, C, H, W input tensor
        _, _, H, W = x.shape
        y_low = (H - self.crop_height) // 2
        y_high = y_low + self.crop_height
        x_low = (W - self.crop_width) // 2
        x_high = x_low + self.crop_width

        cropped = x[:, :, y_low:y_high, x_low:x_high]

        return cropped.tile(self.tile)
