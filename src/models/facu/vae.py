from typing import Any, Dict, Union
import torch

from src.models.components.encoders import VaeEncoder
from src.models.components.decoders import SmoeDecoder

__all__ = [
    "VariationalAutoencoder",
]

class AutoVariationalAutoencoder(torch.nn.Module):
    def __init__(self, encoder_configs: Dict[str, Any], smoe_configs: Dict[str, Any], loss_configs: Dict[str, Any]):
        super().__init__()
        self.beta = loss_configs["beta"]
        self.n_kernels = smoe_configs["n_kernels"]
        self.block_size = smoe_configs["block_size"]
        self.encoder = VaeEncoder(
            in_channels=1,
            n_kernels=self.n_kernels, 
            block_size=self.block_size,
            hidden_dims_conv=encoder_configs["hidden_dims"]["conv"], 
            hidden_dims_lin=encoder_configs["hidden_dims"]["lin"],
            kernels_outside=smoe_configs["kernels_outside"],
            negative_experts=smoe_configs["negative_experts"],
            downsample=encoder_configs["downsample"],
            batch_norm=encoder_configs["batch_norm"],
            bias=encoder_configs["bias"],
            residual=encoder_configs["residual"],
            order=encoder_configs["order"],
            activation=encoder_configs["activation"],
        )
        self.decoder = SmoeDecoder(
            n_kernels=self.n_kernels,
            block_size=self.block_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_var
    
    def loss(self, input: torch.Tensor, output: torch.Tensor) -> Dict[str, Union[torch.Tensor, Any]]:
        """Loss functions are made on a model by model basis. The trainers will just feed the whole output of the 
        model's forward function into the model's loss function, so you have control over what you need.

        Args:
            input (torch.Tensor): the input to the model
            output (torch.Tensor): the full output of the model

        Returns:
            torch.Tensor: the loss
        """
        y, z, mu, log_var = output
        recons_loss = torch.nn.functional.mse_loss(y, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        total_loss = recons_loss + self.beta*kld_loss
        return {"loss": total_loss, "logging": {"KLDiv Loss": kld_loss.item(), "Reconstruction Loss": recons_loss.item()}}