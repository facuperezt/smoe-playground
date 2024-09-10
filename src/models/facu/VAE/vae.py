import copy
import json
import os
from typing import Any, Dict, Tuple, Union
import torch

from src.models.components.encoders import AutoVaeEncoder, ManualVaeEncoder
from src.models.components.decoders import SmoeDecoder
from src.models.base_model import SmoeModel

__all__ = [
    "VariationalAutoencoder",
]

class VariationalAutoencoder(SmoeModel):
    _saves_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saves")
    def __init__(self, config_path: str):
        try:
            # First assume that the config_path is a relative or absolute path that's findable
            file = open(config_path, "r")
        except FileNotFoundError:
            # Then try to find it in the folder where the model is saved
            file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", config_path), "r")
        model_configs: Dict[str] = json.load(file)
        self._cfg = copy.deepcopy(model_configs)
        file.close()
        model_type = model_configs.pop("model_type", "").lower() 
        super().__init__()
        if model_type == "auto":
            self.model = AutoVariationalAutoencoder(**model_configs)
        elif model_type == "manual":
            self.model = ManualVariationalAutoencoder(**model_configs)
        elif model_type == "":
            raise ValueError("JSON config needs to specify a model_type.")
        else:
            raise ValueError("model_type is not recognized.")
    
    @property
    def n_kernels(self) -> int:
        """Number of kernels that the model uses for SMoE

        Returns:
            int: # of kernels
        """
        return self.model.n_kernels
    
    @property
    def block_size(self) -> int:
        """Size of the blocks that the image has been divided into, assumes quadratic blocks

        Returns:
            int: # of pixels for H and W
        """
        return self.model.block_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def loss(self, input: torch.Tensor, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], extra_information: torch.Tensor) -> Dict[str, Union[torch.Tensor, Any]]:
        return self.model.loss(input, output, extra_information)

    def reconstruct_input(self, input: torch.Tensor) -> torch.Tensor:
        return self.model.forward(input)[0]
    
class AutoVariationalAutoencoder(torch.nn.Module):
    def __init__(self, encoder_configs: Dict[str, Any], smoe_configs: Dict[str, Any], loss_configs: Dict[str, Any]):
        super().__init__()
        self.beta = loss_configs["beta"]
        self.n_kernels = smoe_configs["n_kernels"]
        self.block_size = smoe_configs["block_size"]
        self.encoder = AutoVaeEncoder(
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
            dropout=encoder_configs["dropout"],
            order=encoder_configs["order"],
            activation=encoder_configs["activation"],
        )
        self.decoder = SmoeDecoder(
            n_kernels=self.n_kernels,
            block_size=self.block_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_var
    
    def loss(self, inputs: torch.Tensor, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], extra_information: torch.Tensor) -> Dict[str, Union[torch.Tensor, Any]]:
        """Loss functions are made on a model by model basis. The trainers will just feed the whole output of the 
        model's forward function into the model's loss function, so you have control over what you need.

        Args:
            input (torch.Tensor): the input to the model
            output (torch.Tensor): the full output of the model

        Returns:
            torch.Tensor: the loss
        """
        y, z, mu, log_var = output
        true_z = extra_information
        recons_loss = torch.nn.functional.mse_loss(y, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        total_loss = recons_loss + self.beta*kld_loss
        return total_loss, {"KLDiv Loss": kld_loss.item(), "Reconstruction Loss": recons_loss.item()}
    
class ManualVariationalAutoencoder(AutoVariationalAutoencoder):
    def __init__(self, encoder_configs: Dict[str, Any], smoe_configs: Dict[str, Any], loss_configs: Dict[str, Any]):
        torch.nn.Module.__init__(self)
        self.beta = loss_configs["beta"]
        self.n_kernels = smoe_configs["n_kernels"]
        self.block_size = smoe_configs["block_size"]
        self.encoder = ManualVaeEncoder(
            in_channels=1,
            n_kernels=self.n_kernels, 
            block_size=self.block_size,
            hidden_dims_conv=encoder_configs["hidden_dims"]["conv"], 
            hidden_dims_lin=encoder_configs["hidden_dims"]["lin"],
            kernels_outside=smoe_configs["kernels_outside"],
            negative_experts=smoe_configs["negative_experts"],
            downsample_factor=encoder_configs["downsample_factor"],
            batch_norm=encoder_configs["batch_norm"],
            bias=encoder_configs["bias"],
            residual=encoder_configs["residual"],
            dropout=encoder_configs["dropout"],
            order=encoder_configs["order"],
            activation=encoder_configs["activation"],
        )
        self.decoder = SmoeDecoder(
            n_kernels=self.n_kernels,
            block_size=self.block_size,
        )
