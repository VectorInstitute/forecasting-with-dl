import torch
import torch.nn as nn


class Custom_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        preds = preds.softmax(-1)
        target = target.softmax(-1)
        loss = -torch.mean(target * torch.log(preds) +
                           (1 - target) * torch.log(1 - preds))
        return loss
