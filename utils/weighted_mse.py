"""
Weighted MSE Loss for regression tasks with aspect ratio compensation.
Ported from 3ET+ utils/metrics.py.
"""
import torch
import torch.nn as nn


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('weights', weights)
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss


class weighted_RMSELoss(nn.Module):
    """Weighted RMSE loss: sqrt of weighted MSE (as used in JaneEye-Net)."""

    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('weights', weights)
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        return torch.sqrt(torch.mean(batch_loss))
