import copy
import json
import os
from typing import Any, Dict, Union
import torch

from src.models.components.decoders import SmoeDecoder
from src.models.components.encoders import ResNetEncoder
from src.models.base_model import SmoeModel

__all__ = [
    "ResNetWeirdness"
]


class ResNetWeirdness(SmoeModel):
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
        self.encoder = ResNetEncoder(in_channels=1, out_features=7*n_kernels)
        self.decoder = SmoeDecoder(n_kernels, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def reconstruct_input(self, input: torch.Tensor) -> torch.Tensor:
        # Need to go through the __call__ method for hooks to work properly
        return self(input)
    
    def loss(self, input: torch.Tensor, output: torch.Tensor, extra_information: torch.Tensor) -> Dict[str, Union[torch.Tensor, Any]]:
        batch_reshape = lambda x: x.reshape((*x.shape[:2], -1)) 
        min_loss = 0.25*torch.nn.functional.mse_loss(batch_reshape(input).min(dim=-1).values, batch_reshape(output).min(dim=-1).values)
        max_loss = 0.25*torch.nn.functional.mse_loss(batch_reshape(input).max(dim=-1).values, batch_reshape(output).max(dim=-1).values)
        rec_loss = torch.nn.functional.mse_loss(output, input)
        loss = rec_loss + min_loss + max_loss
        return loss, {"Reconstruction Loss": rec_loss, "Min/Max Value Loss": {"min": min_loss, "max": max_loss}}