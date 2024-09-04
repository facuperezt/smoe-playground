import torch


__all__ = [
    "DownSampleBlock"
]


class DownSampleBlock(torch.nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)