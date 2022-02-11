import torch
import torch.nn as nn

    
class CRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def accumulated_indicator(self, target, output_shape):
        cdf_bins = torch.arange(output_shape).view(-1, 1, 1).type_as(target)
        acc_ind = torch.where(torch.repeat_interleave(target, output_shape, dim=-3) >= cdf_bins, 1, 0)
        return acc_ind

    def forward(self, output, target):
        output_shape = output.shape[-3]
        if output_shape > 1:
            target = target * (output_shape - 1)
        output = output.softmax(-3)
        #zero = output - output.detach()
        brier_const = self.accumulated_indicator(target, output_shape).detach()
        brier_sqrd = torch.square(output - brier_const) / (brier_const.shape[-1] * brier_const.shape[-2])
        crps = torch.sum(brier_sqrd)
        return crps
