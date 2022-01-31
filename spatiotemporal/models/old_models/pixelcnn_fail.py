import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .helper_functions import down_shift


class IndependentLayerNorm(nn.Module):
    def __init__(self, out_channels: int, h_dim: int, w_dim: int) -> None:
        super().__init__()
        self.v_ln = nn.LayerNorm((out_channels, h_dim, w_dim), elementwise_affine=False)
        self.h_ln = nn.LayerNorm((out_channels, h_dim, w_dim), elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        vx, hx = x.chunk(2, dim=1)
        vx = self.v_ln(vx)
        hx = self.h_ln(hx)
        return torch.cat((vx, hx), dim=1)


class MaskedConv2d(nn.Module):
    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 7
        self.padding = 3
        self.dilation = 1

        self.cn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            bias=False
        )

        self.register_buffer("mask", torch.ones(self.cn.weight.data.shape))
        self.mask[:, :, self.kernel_size // 2 + 1:, :] = 0
        self.mask[:, :, self.kernel_size // 2, self.kernel_size // 2:] = 0
        if self.mask_type == "B":
            self.mask[:, :, self.kernel_size // 2, self.kernel_size // 2] = 1

    def forward(self, x: Tensor) -> Tensor:
        self.cn.weight.data *= self.mask
        out = self.cn(x)
        return out

class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout

        self.v_cn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * 2,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            dilation=1,
            bias=False
        )
        self.h_cn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels * 2,
            kernel_size=(1, self.kernel_size),
            padding=(0, self.kernel_size // 2),
            dilation=1,
            bias=False
        )
        self.v1x1_cn = nn.Conv2d(
            in_channels=self.out_channels * 2,
            out_channels=self.out_channels * 2,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=False
        )
        self.h1x1_cn = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias=False
        )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(self.dropout)

        self.register_buffer("v_cn_mask",
                             torch.ones(self.v_cn.weight.data.shape))
        self.register_buffer("h_cn_mask",
                             torch.ones(self.h_cn.weight.data.shape))
        self.v_cn_mask[:, :, self.kernel_size // 2:, :] = 0
        #self.v_cn_mask[:, :, self.kernel_size // 2, self.kernel_size //2] = 1
        self.h_cn_mask[:, :, :, self.kernel_size // 2 + 1:] = 0

    def forward(self, x: Tensor) -> Tensor:
        # Apply masks to convs
        self.v_cn.weight.data *= self.v_cn_mask
        self.h_cn.weight.data *= self.h_cn_mask

        v_x, h_x = x.chunk(2, dim=1)
        v_x_stream = self.v_cn(v_x)
        h_x_stream = self.h_cn(h_x)
        h_x_stream = h_x_stream + self.v1x1_cn(down_shift(v_x_stream, 1))

        v_x_1, v_x_2 = v_x_stream.chunk(2, dim=1)
        v_x_stream = torch.tanh(v_x_1) * torch.sigmoid(v_x_2)

        h_x_1, h_x_2 = h_x_stream.chunk(2, dim=1)
        h_x_stream = torch.tanh(h_x_1) * torch.sigmoid(h_x_2)
        h_x_stream = self.h1x1_cn(h_x_stream)
        h_x_stream = h_x_stream + h_x

        new_out = torch.cat((v_x_stream, h_x_stream), dim=1)
        return new_out


class GatedPixelCNN(nn.Module):
    def __init__(
        self,
        h_dim: int,
        w_dim: int,
        in_channels: int = 3,
        out_channels: int = 768,
        kernel_size: int = 7,
        dilation: int = 1,
        n_layers: int = 8,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2 
        self.dilation = dilation
        self.n_layers = n_layers
        self.dropout = dropout
        

        self.mask_cn_in = MaskedConv2d("A", self.in_channels, self.out_channels)
        self.layers = self._make_layers()
        self.mask_cn_out = MaskedConv2d("B", self.out_channels, self.in_channels * 8)
        self.relu = nn.ReLU()
        #self.final_cn1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, padding=0)
        #self.final_cn2 = nn.Conv2d(self.out_channels, self.in_channels * 8, kernel_size=1, padding=0)
        #self.final_cn = GatedConv2d(self.out_channels, self.out_channels, self.kernel_size,
        #                            self.padding, self.dilation, self.dropout)

    def _make_layers(self):
        layers = []
        for i in range(self.n_layers - 2):
            layers.append(nn.ReLU())
            layers.append(  # Note that GatedConv2d actually outputs out_channels*2
                GatedConv2d(self.out_channels, self.out_channels,
                            self.kernel_size, self.padding, self.dilation, self.dropout))
            layers.append(IndependentLayerNorm(self.out_channels, self.h_dim, self.w_dim))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        x_mask_in = self.mask_cn_in(x)
        x_net_in = torch.cat((x_mask_in, x_mask_in), dim=1)
        net_out = self.layers(x_net_in)
        h_net_out = net_out.chunk(2, dim=1)[1]    # Take output from horizontal stream
        # NEW #
        h_net_out = self.relu(h_net_out)
        out = self.mask_cn_out(h_net_out)
        #out = self.relu(out)
        #out = self.final_cn1(out)
        #out = self.relu(out)
        #out = self.final_cn2(out)

        # Want a distribution over all pixel values for each colour channel
        return out.view(N, 8, C, H, W)      

    def sample(self, n_samples: int, shape: tuple) -> Tensor:
        C, H, W = shape
        samples = torch.zeros(n_samples, C, H, W).cuda()
        with torch.no_grad():
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        output = self(samples)
                        logits = output[:, :, c, h, w]
                        prob = F.softmax(logits, dim=1)
                        samples[:, c, h, w] = torch.multinomial(prob,
                                                                1).squeeze(-1)
        return samples
