import copy
import json
import os
from typing import Any, Dict, Tuple, Union
import torch

from src.models.components.decoders import SmoeDecoder
from src.models.components.encoders import ResNetVaeEncoder
from src.models.base_model import SmoeModel

__all__ = [
    "ResNetVae"
]


class ResNetVae(SmoeModel):
    _saves_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saves")
    def __init__(self, config_path: str):
        try:
            # First assume that the config_path is a relative or absolute path that's findable
            file = open(config_path, "r")
        except FileNotFoundError:
            # Then try to find it in the folder where the model is saved
            file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", config_path), "r")
        super().__init__()
        model_configs: Dict[str] = json.load(file)
        n_kernels = model_configs["smoe_configs"]["n_kernels"]
        block_size = model_configs["smoe_configs"]["block_size"]
        self.n_kernels = n_kernels
        self.block_size = block_size
        self._cfg = copy.deepcopy(model_configs)
        self.encoder = ResNetVaeEncoder(in_channels=1, out_features=7*n_kernels)
        self.decoder = SmoeDecoder(n_kernels, block_size)
        self.beta = model_configs["loss_configs"]["beta"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_var

    def reconstruct_input(self, input: torch.Tensor) -> torch.Tensor:
        # Need to go through the __call__ method for hooks to work properly
        return self(input)[0]
    
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