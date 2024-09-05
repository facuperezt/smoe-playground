import copy
import json
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.models.components.encoders import VqGanEncoder
from src.models.components.blocks import SmoeActivations, VqCodebook
from src.models.components.decoders import SmoeDecoder

__all__ = [
    "VQVAE"
]

class VQVAE_Simple(nn.Module):
    def __init__(self, smoe_args: Dict[str, Any], encoder_args: Dict[str, Any], codebook_args: Optional[Dict[str, Any]] = None, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.n_kernels = smoe_args["n_kernels"]
        self.block_size = smoe_args["block_size"]
        self.encoder = VqGanEncoder(**encoder_args).to(device=device)
        self.decoder = SmoeDecoder(**smoe_args, device=device)
        self.smoe_activations = SmoeActivations(
            (2*smoe_args["n_kernels"], 1*smoe_args["n_kernels"], 4*smoe_args["n_kernels"]),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
        )
        latent_dim = encoder_args["latent_dim"]
        if codebook_args is None:
            codebook_args = {}
        self.codebook = VqCodebook(**codebook_args).to(device=device)
        self.quant_conv = nn.Conv2d(encoder_args["latent_dim"], codebook_args["latent_dim"], 1).to(device=device)
        self.post_quant_conv = nn.Conv2d(codebook_args["latent_dim"], encoder_args["latent_dim"], 1).to(device=device)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        post_quant_conv_mapping = post_quant_conv_mapping.reshape(*post_quant_conv_mapping.shape[:-2], -1).mean(dim=-1)
        decoded_images = self.decoder(self.smoe_activations(post_quant_conv_mapping))

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def loss(self, input: torch.Tensor, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], extra_information: Dict[str, Any]):
        decoded_input, _, q_loss = output
        rec_loss = torch.nn.functional.mse_loss(decoded_input, input)
        loss = rec_loss + q_loss
        return {"loss": loss, "logging": {"Reconstruction Loss": rec_loss, "Quantization Loss": q_loss}}

class VQVAE(VQVAE_Simple):
    def __init__(self, config_path: str):
        try:
            # First assume that the config_path is a relative or absolute path that's findable
            file = open(config_path, "r")
        except FileNotFoundError:
            # Then try to find it in the folder where the model is saved
            file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", config_path), "r")
        model_configs: Dict[str] = json.load(file)
        self.cfg = copy.deepcopy(model_configs)
        file.close()
        super().__init__(**model_configs)

class VQGAN(VQVAE):
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))