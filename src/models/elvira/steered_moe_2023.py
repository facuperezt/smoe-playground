from typing import Literal
import torch

from src.models.components.encoders import CaeEncoder
from src.models.components.decoders import SmoeDecoder

__all__ = [
    "Elvira2023Full",
    "Elvira2023Small",
]

class Elvira2023Full(torch.nn.Module):
    def __init__(self, block_size: Literal[8, 16] = 16):
        super().__init__()
        self.n_kernels = 4  # K = 4
        self.block_size = block_size  # 16 or 8 in the 2023 paper
        self.encoder = CaeEncoder(
            in_channels=1,  # Grayscale images
            n_kernels=self.n_kernels,  # K = 4
            block_size=block_size,  # 16 or 8 in the 2023 paper
            hidden_dims_conv=[16, 32, 64, 128, 256, 512, 1024],  # given in paper
            # important: last conv dim and first lin dim must match
            hidden_dims_lin=[1024, 512, 256, 128, 64, 24],  # given in paper
            order="la",  # First the linear or conv layer and then activation, nothing else.
            activation="relu",  # paper simply uses relu
        )
        self.decoder = SmoeDecoder(
            n_kernels=self.n_kernels,  # K = 4
            block_size=block_size,  # 16 or 8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def loss(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Loss functions are made on a model by model basis. The trainers will just feed the whole output of the 
        model's forward function into the model's loss function, so you have control over what you need.

        Args:
            input (torch.Tensor): the input to the model
            output (torch.Tensor): the output of the model

        Returns:
            torch.Tensor: the loss
        """
        return {"loss": torch.nn.functional.mse_loss(output, input), "logging": None}
    
class Elvira2023Small(torch.nn.Module):
    def __init__(self, block_size: Literal[8, 16] = 16):
        super().__init__()
        self.n_kernels = 4  # K = 4
        self.block_size = block_size  # 16 or 8 in the 2023 paper
        self.encoder = CaeEncoder(
            in_channels=1,  # Grayscale images
            n_kernels=4,  # K = 4
            block_size=block_size,  # 16 or 8 in the 2023 paper
            hidden_dims_conv=[16, 32, 64, 128],  # given in paper
            # important: last conv dim and first lin dim must match
            hidden_dims_lin=[128, 64, 24],  # given in paper
            order="la",  # First the linear or conv layer and then activation, nothing else.
            activation="relu",  # paper simply uses relu
        )
        self.decoder = SmoeDecoder(
            n_kernels=4,  # K = 4
            block_size=block_size,  # 16 or 8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def loss(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Loss functions are made on a model by model basis. The trainers will just feed the whole output of the 
        model's forward function into the model's loss function, so you have control over what you need.

        Args:
            input (torch.Tensor): the input to the model
            output (torch.Tensor): the output of the model

        Returns:
            torch.Tensor: the loss
        """
        return torch.nn.functional.mse_loss(output, input)