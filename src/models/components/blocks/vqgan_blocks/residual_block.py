import torch

__all__ = [
    "ResidualBlock"
]

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups: int = 32, eps: float = 1e-6, affine: bool = True):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=in_channels, num_groups=num_groups, eps=eps, affine=affine),
            Swish(),
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            torch.nn.GroupNorm(num_channels=out_channels, num_groups=num_groups, eps=eps, affine=affine),
            Swish(),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)