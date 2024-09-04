import torch

from src.models.components.blocks import ResNetBlock


__all__ = [
    "ResNetEncoder"
]


class ResNetEncoder(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 28):
        super().__init__()
        blocks = [
            torch.nn.Conv2d(in_channels, out_channels=256, kernel_size=1),
            ResNetBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1)
        ]
        self.convs = torch.nn.Sequential(*blocks)
        blocks = [
            torch.nn.Linear(in_features=256, out_features=4096),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=4096, out_features=out_features)
        ]
        self.lins = torch.nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        # x = self.convs(x)
        x = x.view((*x.shape[:-2], -1)).mean(dim=-1)  # Avg Pool all remaining "pixels"
        x = self.lins(x)
        return x
