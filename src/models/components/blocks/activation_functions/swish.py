import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Swish",
]

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)