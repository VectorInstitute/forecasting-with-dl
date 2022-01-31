from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PSResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.causal_cn1 = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.in_channels,
                                    kernel_size=self.kernel_size,)
        self.causal_cn2 = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.in_channels,
                                    kernel_size=self.kernel_size,)
        self.causal_cn3 = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.in_channels,
                                    kernel_size=self.kernel_size,)
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))   # Need custom padding for 2x2 conv
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        trunk = self.elu(x)
        trunk = self.causal_cn1(self.pad(trunk))
        trunk = self.elu(trunk)

        branch_l = self.causal_cn2(self.pad(trunk))
        branch_r = self.causal_cn3(self.pad(trunk))
        branch_r = self.sigmoid(branch_r)

        out = branch_l * branch_r
        out += identity

        return out


class PSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 k_filters: int,
                 v_filters: int,
                 h_dim: int, 
                 w_dim: int,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.k_filters = k_filters
        self.v_filters = v_filters
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.qk_cn = nn.Conv2d(in_channels=self.in_channels * 2,
                                 out_channels=self.k_filters * 2,
                                 kernel_size=1,
                                 padding=0,)
        self.v_cn = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.v_filters,
                              kernel_size=1,
                              padding=0,)

        self.causal_attn_mask = self._create_causal_attn_mask()

    def _create_causal_attn_mask(self):
        mask = torch.zeros(self.k_filters, self.h_dim, self.w_dim)
        for i in range(self.h_dim):
            mask[:, i:, i] = 1 
        return mask

    def _causal_softmax(self, x: Tensor, dim: int = -1) -> Tensor:
        masked_exps = torch.exp(x) * self.causal_attn_mask.to(x.device)
        masked_sums = masked_exps.sum(dim=dim, keepdim=True)
        masked_softmax = masked_exps / masked_sums
        return masked_softmax

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_qk = torch.cat((x, x), dim=1)
        q, k = self.qk_cn(x_qk).chunk(2, dim=1)
        masked_qk = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        causal_softmax = self._causal_softmax(masked_qk, dim=-1)

        v = self.v_cn(x)
        out = torch.einsum('jilm,jklm->jklm', causal_softmax, v)
        import pdb; pdb.set_trace()

        return out


class PSBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 k_filters: int,
                 v_filters: int,
                 h_dim: int,
                 w_dim: int,
                 n_residuals: int,
        ) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.hidden_channels=hidden_channels
        self.k_filters=k_filters
        self.v_filters=v_filters
        self.h_dim=h_dim
        self.w_dim=w_dim
        self.n_residuals = n_residuals

        self.residual_blocks = self._make_residuals()
        self.attn_block = PSAttentionBlock(self.in_channels + self.hidden_channels,
                                           self.k_filters,
                                           self.v_filters,
                                           self.h_dim,
                                           self.w_dim,)
        self.res_cn = nn.Conv2d(in_channels=self.hidden_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=1,
                                padding=0,)
        self.attn_cn = nn.Conv2d(in_channels=self.v_filters,
                                 out_channels=self.hidden_channels,
                                 kernel_size=1,
                                 padding=0,)
        self.final_cn = nn.Conv2d(in_channels=self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=1,
                                  padding=0,)

        self.elu = nn.ELU()

    def _make_residuals(self):
        residuals = []
        for i in range(self.n_residuals):
            residuals.append(PSResidualBlock(self.hidden_channels))
        return nn.Sequential(*residuals)

    def forward(self, x: Tensor, x_residual: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.residual_blocks(x)

        branch_l = self.elu(self.res_cn(self.elu(x)))

        branch_r = torch.cat((x, x_residual), dim=1)
        branch_r = self.attn_block(branch_r)
        branch_r = self.elu(self.attn_cn(self.elu(branch_r)))

        final_trunk = branch_l + branch_r
        final_trunk = self.elu(self.final_cn(self.elu(final_trunk)))

        return final_trunk, x_residual


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PixelSNAIL(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 k_filters: int,
                 v_filters: int,
                 h_dim: int,
                 w_dim: int,
                 n_res_per_layer: int,
                 m_layers:int,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k_filters = k_filters
        self.v_filters = v_filters
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.n_res_per_layer = n_res_per_layer
        self.m_layers = m_layers

        self.causal_cn = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.hidden_channels,
                                   kernel_size=2,)
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))   # Need custom padding for 2x2 conv
        self.layers = self._make_layers()
        self.elu = nn.ELU()
        self.final_cn = nn.Conv2d(in_channels=self.hidden_channels,
                                  out_channels=self.in_channels * 256,
                                  kernel_size=1,)

    def _make_layers(self):
        layers = []
        for i in range(self.m_layers):
            layers.append(PSBlock(self.in_channels,
                                  self.hidden_channels,
                                  self.k_filters,
                                  self.v_filters,
                                  self.h_dim,
                                  self.w_dim,
                                  self.n_res_per_layer))
        return MultiInputSequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        cn_in = self.causal_cn(self.pad(x))
        out = self.layers(cn_in, x)
        out = self.final_cn(self.elu(out[0]))
        return out.view(N, C, 256, H, W)

    def loss_fn(self, x: Tensor) -> Tensor:
        out = self(x)
        x = (x - x.min()) * (255 / (x.max() - x.min()))
        return F.cross_entropy(out.permute(0, 2, 1, 3, 4), x.long())
