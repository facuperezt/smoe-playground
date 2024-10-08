"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn

__all__ = [
    "PatchGanDiscriminator",
]

class PatchGanDiscriminator(nn.Module):
    def __init__(self, input_channels: int, num_filters_last: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [nn.Conv2d(input_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(
                        in_channels=num_filters_last * num_filters_mult,
                        out_channels=1,
                        kernel_size=4,
                        stride=1,
                        padding=1
                    ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
