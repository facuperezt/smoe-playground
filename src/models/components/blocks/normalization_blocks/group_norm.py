import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GroupNorm",
]

class GroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return self.gn(x)