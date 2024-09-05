from typing import List, Optional
import torch.nn as nn
from src.models.components.blocks import (
    VqGanResidualBlock,
    NonLocalAttentionBlock,
    GroupNorm,
    Swish,
    VqGanDownsampleBlock,
)

__all__ = [
    "Encoder"
]

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 28,
        attn_resolutions: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        img_size: int = 384,
    ):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 256, 256, 512]
        if attn_resolutions is None:
            attn_resolutions = [16]
        current_resolution = img_size
        layers = [nn.Conv2d(in_channels, hidden_dims[0], 3, 1, 1)]
        for i in range(len(hidden_dims) - 1):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1]
            for j in range(num_res_blocks):
                layers.append(VqGanResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if current_resolution in attn_resolutions:
                    layers.append(NonLocalAttentionBlock(in_channels))
            if i != len(hidden_dims):
                layers.append(VqGanDownsampleBlock(hidden_dims[i + 1]))
                current_resolution //= 2
        layers.append(VqGanResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        layers.append(NonLocalAttentionBlock(hidden_dims[-1]))
        layers.append(VqGanResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        layers.append(GroupNorm(hidden_dims[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(hidden_dims[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # for layer in self.model:
        #     x = layer(x)
        # return x.squeeze()
        return self.model(x)
