import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization over the time axis of each window."""

    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x):
        # x: (B, C, L)
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        self.std = torch.sqrt(
            x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x

    def denormalize(self, x):
        # x: (B, C, H)
        if self.affine:
            x = (x - self.bias.view(1, -1, 1)) / (self.weight.view(1, -1, 1) + 1e-8)
        x = x * self.std + self.mean
        return x
