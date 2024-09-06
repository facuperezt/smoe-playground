from typing import Any, Dict, Union
import torch

from src.models.components.decoders import SmoeDecoder
from src.models.components.encoders import ResNetEncoder


__all__ = [
    "ResNetWeirdness"
]


class ResNetWeirdness(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 16,):
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.cfg = "default"
        self.encoder = ResNetEncoder(in_channels=1, out_features=7*n_kernels)
        self.decoder = SmoeDecoder(n_kernels, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def loss(self, input: torch.Tensor, output: torch.Tensor, extra_information: torch.Tensor) -> Dict[str, Union[torch.Tensor, Any]]:
        batch_reshape = lambda x: x.reshape((*x.shape[:2], -1)) 
        min_loss = 0.25*torch.nn.functional.mse_loss(batch_reshape(input).min(dim=-1).values, batch_reshape(output).min(dim=-1).values)
        max_loss = 0.25*torch.nn.functional.mse_loss(batch_reshape(input).max(dim=-1).values, batch_reshape(output).max(dim=-1).values)
        rec_loss = torch.nn.functional.mse_loss(output, input)
        loss = rec_loss + min_loss + max_loss
        return loss, {"Reconstruction Loss": rec_loss, "Min/Max Value Loss": {"min": min_loss, "max": max_loss}}