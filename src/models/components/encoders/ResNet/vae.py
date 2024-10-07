from typing import Tuple
import torch

from src.models.components.blocks import ResNetResidualBlock


__all__ = [
    "ResNetVaeEncoder"
]

class ResNetVaeEncoder(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 28):
        super().__init__()
        blocks = [
            torch.nn.Conv2d(in_channels, out_channels=256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
            ResNetResidualBlock(in_channels=256, filter_sizes=[1, 3, 1], hidden_dims=[64, 64], out_channels=256),
            torch.nn.Conv2d(256, 256, kernel_size=1),
        ]
        self.convs = torch.nn.Sequential(*blocks)
        blocks = [
            torch.nn.Linear(in_features=256, out_features=4096),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=4096, out_features=256)
        ]
        self.lins = torch.nn.Sequential(*blocks)
        self.fc_mean = torch.nn.Linear(256, out_features)
        self.fc_log_var = torch.nn.Linear(256, out_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        # x = self.convs(x)
        x = x.view((*x.shape[:-2], -1)).mean(dim=-1)  # Avg Pool all remaining "pixels"
        x = self.lins(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = eps * std + mu
        # out = self.output_nonlinearities(out)
        return out

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var